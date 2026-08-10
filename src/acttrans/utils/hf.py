"""HuggingFace plumbing shared by extraction, sampling and the A/B evals."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# Compatibility shim: microsoft/Phi-tiny-MoE-instruct's custom modeling code
# (trust_remote_code) calls DynamicCache.get_usable_length(seq_length), a method
# newer transformers dropped in favor of get_seq_length(). DynamicCache is
# unbounded, so the old method's semantics reduced to get_seq_length() anyway
# (the max-length clamp it applied only mattered for bounded caches). No-op for
# every other model here, which never calls this method.
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)


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
    return model, tokenizer


def get_layer(model, layer_idx: int, module_name: str = "residual"):
    """Architecture-agnostic decoder-layer lookup (Llama/Gemma-style
    ``model.model.layers`` or GPT-style ``model.transformer.h``). With
    module_name="residual" the layer itself is returned (its output is the
    residual stream); otherwise the named submodule of that layer."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer = model.transformer.h[layer_idx]
    else:
        raise ValueError(
            f"Unsupported model architecture: {type(model)}. "
            "Expected Llama/Gemma-style (model.model.layers) or GPT-style (model.transformer.h)."
        )
    return layer if module_name == "residual" else getattr(layer, module_name)
