"""HuggingFace plumbing shared by extraction, sampling and the A/B evals."""

import inspect
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

# Compatibility shim: microsoft/Phi-tiny-MoE-instruct's custom modeling code
# (trust_remote_code) calls DynamicCache.get_usable_length(seq_length), a method
# newer transformers dropped in favor of get_seq_length(). DynamicCache is
# unbounded, so the old method's semantics reduced to get_seq_length() anyway
# (the max-length clamp it applied only mattered for bounded caches). No-op for
# every other model here, which never calls this method.
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)

# Compatibility shim: GSAI-ML/LLaDA-8B-Instruct's custom LLaDAModelLM predates the
# transformers convention of calling self.post_init() at the end of __init__, so
# `all_tied_weights_keys` never gets set, and its tie_weights() override uses the
# pre-5.x no-kwargs signature. transformers 5.1.0's _finalize_model_loading
# unconditionally reads/calls both with the new contract and crashes without this.
# No-op for every model whose post_init() already ran and already accepts the new
# tie_weights() kwargs.
_orig_finalize_model_loading = PreTrainedModel._finalize_model_loading


def _patch_tie_weights_signature(cls):
    """Make cls.tie_weights tolerate the transformers-5.x kwargs by ignoring them.

    Patched on the CLASS, never on the instance. An earlier version of this shim
    stored `lambda **kwargs: bound_tie_weights()` on the model, which built a
    reference cycle (model.__dict__ -> lambda -> closure cell -> bound method ->
    model). Refcounting cannot break a cycle, so `del model` in the extraction
    loop freed nothing and torch.cuda.empty_cache() had nothing to return — every
    per-layer reload stranded a full copy of the weights on the GPU until the
    generational GC happened to run, which OOM'd the sweep partway through.
    A class-level patch captures only the original *function*, so no model is
    referenced and instances die by refcount alone."""
    if getattr(cls, "_acttrans_tie_weights_compat", False):
        return
    if {"missing_keys", "recompute_mapping"} <= set(inspect.signature(cls.tie_weights).parameters):
        return  # already the new signature — nothing to adapt
    orig_tie_weights = cls.tie_weights

    def tie_weights(self, *args, **kwargs):
        return orig_tie_weights(self)

    cls.tie_weights = tie_weights
    cls._acttrans_tie_weights_compat = True


def _finalize_model_loading_compat(model, load_config, loading_info):
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = {}
    _patch_tie_weights_signature(type(model))
    return _orig_finalize_model_loading(model, load_config, loading_info)


PreTrainedModel._finalize_model_loading = staticmethod(_finalize_model_loading_compat)

# Compatibility shim: inclusionAI/LLaDA2.0-mini's vendored rotary-embedding code
# does an unconditional ROPE_INIT_FUNCTIONS["default"] lookup (the pre-refactor
# convention); transformers 5.1.0 dropped that entry (default RoPE is now
# special-cased inline in each model's own RotaryEmbedding class instead of going
# through this dict). Restore a generic "default" entry computing the original
# RoPE formula directly, reading rope_theta from either the old flat
# `config.rope_theta` or the newer nested `config.rope_parameters["rope_theta"]`.
# setdefault is a no-op for any model that already special-cases "default" itself.
def _default_rope_parameters_compat(config, device=None, seq_len=None):
    # theta: prefer the FLAT config.rope_theta. It is the field the checkpoint's own
    # config.json carries and the one the vendored attention reads
    # (self.rope_theta = config.rope_theta), whereas transformers 5.1's migration
    # synthesizes config.rope_parameters with a 10000.0 default that can silently
    # disagree — LLaDA2.0-mini ships rope_theta 600000 but reports 10000.0 nested.
    base = getattr(config, "rope_theta", None)
    if base is None:
        base = config.rope_parameters["rope_theta"]
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    # partial_rotary_factor: only the first fraction of each head is rotated
    # (LLaDA2.0-mini rotates half of its 128-wide heads, so dim 64 -> 32 freqs).
    # Ignoring it yields a table of the wrong length AND the wrong frequencies,
    # because dim appears in the exponent's denominator too.
    partial = getattr(config, "partial_rotary_factor", None) or 1.0
    dim = int(head_dim * partial)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


def _repair_rope_buffers(model):
    """Recompute rotary inv_freq tables that were never materialized.

    LLaDA2.0-mini's LLaDA2MoeRotaryEmbedding registers inv_freq with
    ``persistent=False``, so it is absent from the checkpoint and the weight
    loader never restores it. It is computed once in __init__ — which, under
    transformers 5.x fast init, runs on the META device. A meta tensor carries no
    storage, so the subsequent ``.to(cuda)`` produces UNINITIALIZED memory instead
    of the frequency table: observed values were a denormal (3.5e-41) and zeros.

    That is worse than a crash. Uninitialized memory varies run to run, so the
    same extraction silently produced NaN in some layers and finite-but-wrong
    activations in others — which is exactly how 10 corrupt caches passed as OK.

    Gated on the buffer actually being unhealthy, so this is a strict no-op for
    every model whose rotary table is already correct (all the causal models here),
    leaving their existing activation caches reproducible."""
    for module in model.modules():
        init_fn = getattr(module, "rope_init_fn", None)
        buf = getattr(module, "inv_freq", None)
        if init_fn is None or not torch.is_tensor(buf):
            continue
        # A valid inv_freq is 1/base**x: every entry strictly positive and finite.
        healthy = (
            not buf.is_meta
            and bool(torch.isfinite(buf).all())
            and bool((buf > 0).all())
        )
        if healthy:
            continue
        device = buf.device if not buf.is_meta else next(model.parameters()).device
        inv_freq, attention_scaling = init_fn(module.config, device)
        module.register_buffer("inv_freq", inv_freq.to(device), persistent=False)
        module.original_inv_freq = module.inv_freq
        module.attention_scaling = attention_scaling
        print(
            f"  [hf] repaired uninitialized rotary inv_freq on "
            f"{type(module).__name__} ({tuple(inv_freq.shape)})"
        )


ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope_parameters_compat)


def _patch_forward_for_block_attention_mask(cls):
    """Compatibility shim: inclusionAI/LLaDA2.0-mini's forward() rejects the
    standard (B, T) padding mask every other model here accepts, requiring instead
    a pre-built (B, 1, T, T) block-diffusion attention bias (see its own
    generate()'s ``torch.tril(...).log()`` construction). Extraction and the A/B
    eval only ever run a single clean (fully unmasked) forward pass — never the
    iterative block-denoising loop — which collapses that convention to full
    bidirectional attention restricted to real (non-pad) key positions. Build that
    here from the ordinary 2D mask so callers (dataset.py, ab_eval.py) stay
    unchanged.

    Patched on the CLASS, and reading dtype off `self`, so nothing that references
    a model instance is stored on the instance — see
    _patch_tie_weights_signature's docstring for the reference-cycle leak that an
    instance-level `model.forward = ...` caused here (~33 GB stranded per
    per-layer reload for this model, which is 16B params resident)."""
    if getattr(cls, "_acttrans_block_mask_compat", False):
        return
    orig_forward = cls.forward

    def forward(self, *args, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None and attention_mask.dim() == 2:
            b, t = attention_mask.shape
            allow = attention_mask[:, None, None, :].to(torch.float32).expand(b, 1, t, t).clone()
            kwargs["attention_mask"] = allow.log().to(self.dtype)
        return orig_forward(self, *args, **kwargs)

    cls.forward = forward
    cls._acttrans_block_mask_compat = True


def load_model_and_tokenizer(model_name: str, device: str, dtype=torch.bfloat16):
    """Load a causal LM + tokenizer onto `device`, using HF_CACHE_DIR as the
    weights cache. The pad token defaults to EOS when the tokenizer has none
    (the Llama/Gemma tokenizers used here ship without one).

    trust_remote_code=True is required for models that ship custom modeling
    code (e.g. microsoft/Phi-tiny-MoE-instruct's PhiMoEForCausalLM); it's a
    no-op for models with no auto_map (Llama/Qwen/Gemma), so it's always on."""
    cache = os.getenv("HF_CACHE_DIR")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, cache_dir=cache, trust_remote_code=True
    ).to(device)

    # Compatibility shim: LLaDA-8B-Instruct's forward() unconditionally reads
    # self.config.use_cache, but LLaDAConfig never defines it (no KV cache for a
    # diffusion LM). No-op for every model whose config already has this field.
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # See _patch_forward_for_block_attention_mask's docstring.
    if getattr(model.config, "model_type", None) == "llada2_moe":
        _patch_forward_for_block_attention_mask(type(model))

    # Must run after .to(device): the repair writes the table on the real device,
    # and the corruption it fixes is caused by that very move. See its docstring.
    _repair_rope_buffers(model)

    return model, tokenizer


def resolve_mask_token_id(model, tokenizer) -> int:
    """The id of the [MASK] token a masked-diffusion LM was trained to fill in.

    The two LLaDA families expose it in *different* places, so both must be
    consulted — neither accessor alone covers them:

        GSAI-ML/LLaDA-8B-Instruct  config.mask_token_id = 126336, tokenizer has none
        inclusionAI/LLaDA2.0-mini  tokenizer '<|mask|>' = 156895, config has no field

    Raises for a model with neither (i.e. any ordinary causal LM), because
    token_position = "mask" is meaningless there and a wrong id would poison a
    whole 30k-row extraction with a plausible-looking but arbitrary token."""
    for source in (
        getattr(tokenizer, "mask_token_id", None),
        getattr(model.config, "mask_token_id", None),
        getattr(model.config, "mask_id", None),
    ):
        if source is not None:
            return int(source)
    raise ValueError(
        f"{type(model).__name__} exposes no mask token (checked tokenizer.mask_token_id, "
        "config.mask_token_id, config.mask_id), so token_position = \"mask\" cannot be used "
        "with it. Mask-position extraction only applies to masked-diffusion LMs "
        "(LLaDA-family); use -1 or \"mean\" for causal models."
    )


def append_mask_token(encoded, mask_token_id: int, pad_token_id: int, padding_side: str = "right"):
    """Append one [MASK] token after the real tokens of every row in a padded batch.

    Returns ``(input_ids, attention_mask, mask_pos)`` — all three widened by one
    column — where ``mask_pos[i]`` is the column holding row i's mask token, i.e.
    the position whose hidden state a "mask" readout gathers.

    Widening rather than overwriting a pad slot is what makes this safe for the
    longest row in the batch, which has no pad slot to spare. Both padding sides
    are handled because the placement differs: under right-padding the mask goes
    at the first pad column (``length``), under left-padding the real tokens
    already end at the final column, so the mask goes just past it.

    The appended position is attended (mask = 1), so the model sees a genuine
    "predict a token here" slot rather than a padded one."""
    ids, attn = encoded["input_ids"], encoded["attention_mask"]
    b, t = ids.shape
    lengths = attn.sum(dim=1)

    new_ids = torch.full((b, t + 1), pad_token_id, dtype=ids.dtype, device=ids.device)
    new_attn = torch.zeros((b, t + 1), dtype=attn.dtype, device=attn.device)
    new_ids[:, :t] = ids
    new_attn[:, :t] = attn

    if padding_side == "left":
        mask_pos = torch.full((b,), t, dtype=torch.long, device=ids.device)
    else:
        mask_pos = lengths.to(torch.long)

    rows = torch.arange(b, device=ids.device)
    new_ids[rows, mask_pos] = mask_token_id
    new_attn[rows, mask_pos] = 1
    return new_ids, new_attn, mask_pos


def get_layer(model, layer_idx: int, module_name: str = "residual"):
    """Architecture-agnostic decoder-layer lookup: Llama/Gemma/Qwen-style
    ``model.model.layers``, GPT-style ``model.transformer.h``, or LLaDA-8B-style
    ``model.model.transformer.blocks``. With module_name="residual" the layer
    itself is returned (its output is the residual stream); otherwise the named
    submodule of that layer."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer = model.transformer.h[layer_idx]
    elif hasattr(model, "model") and hasattr(getattr(model.model, "transformer", None), "blocks"):
        layer = model.model.transformer.blocks[layer_idx]
    else:
        raise ValueError(
            f"Unsupported model architecture: {type(model)}. Expected Llama/Gemma-style "
            "(model.model.layers), GPT-style (model.transformer.h), or LLaDA-8B-style "
            "(model.model.transformer.blocks)."
        )
    return layer if module_name == "residual" else getattr(layer, module_name)
