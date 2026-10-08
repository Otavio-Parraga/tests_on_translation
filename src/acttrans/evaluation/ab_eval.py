"""Closed-ended (A/B) steering evaluation, following nrimsky/CAA.

The heavy machinery shared by ab_comparison.py and ab_sweep.py: the additive
steering hook, the A/B probability metric, and the by-coefficient aggregation.
Replicates the evaluation from activation_engineering (evaluation/guidance.py).

TWO READOUT POSITIONS. For a causal LM the answer letter is the next token after
"Answer: (", so P(A)/P(B) is read at the final position. A masked-diffusion LM
(LLaDA family) is trained to predict tokens ONLY at [MASK] positions; at an
ordinary final position its logits are off-distribution, and measurably so —
on LLaDA-8B-Instruct the final position puts a mean P(A)+P(B) of 0.0004 on the
two answer letters, versus 0.998 at an appended [MASK] over the same items. The
metric therefore has essentially no mass to move there, and a steering sweep
reads as a flat line of noise no matter how good the translated vector is. Which
position is used is chosen per model by ``needs_mask_readout`` below.
"""

from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import torch
from tqdm import tqdm

from ..utils.hf import append_mask_token, get_layer, iter_layers, resolve_mask_token_id


@dataclass
class ABResult:
    a_prob: float
    b_prob: float
    behavior_prob: float
    is_match: bool
    coefficient: float
    question: str


@contextmanager
def apply_steering(model, layer_idx: int, module_name: str,
                   sv: torch.Tensor, coeff: float, prompt_len: int):
    """Additive steering hook (mirrors base.py::_make_additive_hook)."""
    target = get_layer(model, layer_idx, module_name)

    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        act = output[0] if is_tuple else output
        seq = act.shape[1]
        if seq > prompt_len:
            act[:, prompt_len:, :] = act[:, prompt_len:, :] + coeff * sv.to(act)
        elif seq == 1:
            act = act + coeff * sv.to(act)
        return (act,) + output[1:] if is_tuple else act

    handle = target.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def apply_projection_ablation(model, v: torch.Tensor, alpha: float = 1.0):
    """Orthogonal-projection steering hook (arXiv 2512.24143v2, "Activation
    Steering for Masked Diffusion Language Models"): subtracts the SAME
    direction from every layer's residual stream, at every token position,
    rather than adding a scaled vector at one chosen layer over the response
    region only (apply_steering above).

        h~ = h - alpha * (h . v_hat) v_hat

    `v` need not be pre-normalized — it is normalized here, since a
    projection operator requires a unit vector; `alpha=1.0` is the paper's
    exact ablation, `alpha=0` is a no-op (mirrors apply_steering's coeff=0
    baseline)."""
    unit = torch.nn.functional.normalize(v.float(), dim=-1)

    def hook(module, input, output):
        is_tuple = isinstance(output, tuple)
        act = output[0] if is_tuple else output
        u = unit.to(act)
        proj = (act * u).sum(dim=-1, keepdim=True) * u
        act = act - alpha * proj
        return (act,) + output[1:] if is_tuple else act

    handles = [layer.register_forward_hook(hook) for layer in iter_layers(model)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def prompt_len(tok, messages) -> int:
    """Number of tokens up to (not including) the assistant turn."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]


# Masked-diffusion LMs, whose A/B readout must happen at an appended [MASK]
# rather than at the final position — see the module docstring. Gating on
# config.model_type rather than on "does the tokenizer expose a mask token"
# keeps this decision explicit: it is a statement about what the model was
# TRAINED to predict where, not about which special tokens happen to exist.
MASKED_DIFFUSION_MODEL_TYPES = frozenset({"llada", "llada2_moe"})


def needs_mask_readout(model) -> bool:
    """True if this model scores A/B at an appended [MASK] instead of the last position.

    ``llada`` is GSAI-ML/LLaDA-8B-Instruct, ``llada2_moe`` is
    inclusionAI/LLaDA2.0-mini. Every causal LM in this repo returns False and so
    keeps the original last-position readout, byte-for-byte."""
    return getattr(getattr(model, "config", None), "model_type", None) in MASKED_DIFFUSION_MODEL_TYPES


def run_ab_eval(
    model,
    tok,
    items: List[dict],
    sv: torch.Tensor,
    layer_idx: int,
    module_name: str,
    coefficients: List[float],
    limit: int = None,
    mask_readout: bool = None,
    steering_mode: str = "additive",
) -> List[ABResult]:
    """
    Closed-ended A/B evaluation following nrimsky/CAA:
      - prompt = chat_template(question) + "Answer: ("
      - logits at the readout position → P(A) and P(B)
      - metric: avg P(answer matching behavior)

    The readout position is the final token for a causal LM, or an appended
    [MASK] for a masked-diffusion LM — see the module docstring for why this is
    not cosmetic. ``mask_readout`` defaults to auto-detection via
    ``needs_mask_readout(model)``; pass True/False only to force one path (e.g.
    to reproduce the pre-fix numbers on a diffusion target).

    ``steering_mode`` selects how `sv` is applied at each nonzero item in
    `coefficients`: "additive" (default) is apply_steering — one layer
    (`layer_idx`/`module_name`), response region only, `act + coeff*sv`.
    "projection" is apply_projection_ablation — every layer, every position,
    `act - coeff*(act.v_hat)v_hat` (`layer_idx`/`module_name` are unused in
    this mode). The two modes are not swept on the same physical axis —
    "additive" coefficients are a signed magnitude, "projection" coefficients
    are the fraction of the direction removed (alpha) — but both reuse the
    `coefficients` list and the `coeff == 0` baseline branch below unchanged.
    """
    if steering_mode not in ("additive", "projection"):
        raise ValueError(f"Unknown steering_mode {steering_mode!r}, expected 'additive' or 'projection'")
    if limit:
        items = items[:limit]

    a_id = tok.encode("A", add_special_tokens=False)[-1]
    b_id = tok.encode("B", add_special_tokens=False)[-1]

    use_mask = needs_mask_readout(model) if mask_readout is None else mask_readout
    # Resolved once, outside the loop: it raises for a model with no mask token,
    # and that should fail immediately rather than midway through a sweep.
    mask_id = resolve_mask_token_id(model, tok) if use_mask else None

    results: List[ABResult] = []

    for coeff in tqdm(coefficients, desc="  coeff", leave=False):
        for item in tqdm(items, desc=f"    coeff={coeff:+.0f}", leave=False):
            question = item["question"]
            matching = item["answer_matching_behavior"]

            messages = [{"role": "user", "content": question}]
            plen = prompt_len(tok, messages)

            full = (
                tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                + "Answer: ("
            )
            tokens = tok(full, return_tensors="pt", add_special_tokens=False).to(model.device)

            if use_mask:
                # One un-padded row, so the appended [MASK] lands at the final
                # column — but read it by its returned index rather than -1, so
                # this stays correct if the caller ever batches padded rows.
                # It sits past `plen`, so apply_steering's hook steers it along
                # with the rest of the answer region, exactly as the final token
                # is steered in the causal path.
                ids, attn, mask_pos = append_mask_token(
                    tokens, mask_id, tok.pad_token_id,
                    padding_side=getattr(tok, "padding_side", "right"),
                )
                tokens = {"input_ids": ids, "attention_mask": attn}
                read_at = int(mask_pos[0])
            else:
                read_at = -1

            with torch.no_grad():
                if coeff == 0:
                    out = model(**tokens)
                elif steering_mode == "projection":
                    with apply_projection_ablation(model, sv, alpha=coeff):
                        out = model(**tokens)
                else:
                    with apply_steering(model, layer_idx, module_name, sv, coeff, plen):
                        out = model(**tokens)

            probs = torch.softmax(out.logits[0, read_at, :], dim=-1)
            a_p = probs[a_id].item()
            b_p = probs[b_id].item()

            matching_is_a = matching.strip().startswith("(A)")
            behavior_prob = a_p if matching_is_a else b_p
            is_match = (a_p > b_p) if matching_is_a else (b_p > a_p)

            results.append(ABResult(
                a_prob=a_p, b_prob=b_p,
                behavior_prob=behavior_prob,
                is_match=is_match,
                coefficient=coeff,
                question=question,
            ))

    return results


def aggregate_by_coefficient(results: List[ABResult]) -> dict:
    """coeff -> {avg_p_match, accuracy, n} for one eval run."""
    by = defaultdict(list)
    for r in results:
        by[r.coefficient].append(r)
    out = {}
    for c, items in by.items():
        out[c] = {
            "avg_p_match": sum(r.behavior_prob for r in items) / len(items),
            "accuracy": sum(1 for r in items if r.is_match) / len(items),
            "n": len(items),
        }
    return out
