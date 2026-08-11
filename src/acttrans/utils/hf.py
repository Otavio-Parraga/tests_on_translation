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


def _finalize_model_loading_compat(model, load_config, loading_info):
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = {}
    accepted = set(inspect.signature(model.tie_weights).parameters)
    if not {"missing_keys", "recompute_mapping"} <= accepted:
        bound_tie_weights = model.tie_weights
        model.tie_weights = lambda **kwargs: bound_tie_weights()
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
    if getattr(config, "rope_parameters", None):
        base = config.rope_parameters["rope_theta"]
    else:
        base = config.rope_theta
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


ROPE_INIT_FUNCTIONS.setdefault("default", _default_rope_parameters_compat)


def _wrap_forward_for_block_attention_mask(model):
    """Compatibility shim: inclusionAI/LLaDA2.0-mini's forward() rejects the
    standard (B, T) padding mask every other model here accepts, requiring instead
    a pre-built (B, 1, T, T) block-diffusion attention bias (see its own
    generate()'s ``torch.tril(...).log()`` construction). Extraction and the A/B
    eval only ever run a single clean (fully unmasked) forward pass — never the
    iterative block-denoising loop — which collapses that convention to full
    bidirectional attention restricted to real (non-pad) key positions. Build that
    here from the ordinary 2D mask so callers (dataset.py, ab_eval.py) stay
    unchanged."""
    orig_forward = model.forward

    def forward(*args, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None and attention_mask.dim() == 2:
            b, t = attention_mask.shape
            allow = attention_mask[:, None, None, :].to(torch.float32).expand(b, 1, t, t).clone()
            kwargs["attention_mask"] = allow.log().to(model.dtype)
        return orig_forward(*args, **kwargs)

    model.forward = forward


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

    # See _wrap_forward_for_block_attention_mask's docstring.
    if getattr(model.config, "model_type", None) == "llada2_moe":
        _wrap_forward_for_block_attention_mask(model)

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
