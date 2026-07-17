"""Closed-ended (A/B) steering evaluation, following nrimsky/CAA.

The heavy machinery shared by ab_comparison.py and ab_sweep.py: the additive
steering hook, the A/B probability metric, and the by-coefficient aggregation.
Replicates the evaluation from activation_engineering (evaluation/guidance.py).
"""

from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import torch
from tqdm import tqdm

from ..utils.hf import get_layer


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


def prompt_len(tok, messages) -> int:
    """Number of tokens up to (not including) the assistant turn."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]


def run_ab_eval(
    model,
    tok,
    items: List[dict],
    sv: torch.Tensor,
    layer_idx: int,
    module_name: str,
    coefficients: List[float],
    limit: int = None,
) -> List[ABResult]:
    """
    Closed-ended A/B evaluation following nrimsky/CAA:
      - prompt = chat_template(question) + "Answer: ("
      - logits at last token → P(A) and P(B)
      - metric: avg P(answer matching behavior)
    """
    if limit:
        items = items[:limit]

    a_id = tok.encode("A", add_special_tokens=False)[-1]
    b_id = tok.encode("B", add_special_tokens=False)[-1]

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

            with torch.no_grad():
                if coeff == 0:
                    out = model(**tokens)
                else:
                    with apply_steering(model, layer_idx, module_name, sv, coeff, plen):
                        out = model(**tokens)

            probs = torch.softmax(out.logits[0, -1, :], dim=-1)
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
