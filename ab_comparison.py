"""
A/B comparison: original steering vector vs translated steering vector.

Replicates the closed-ended (A/B) evaluation from activation_engineering
(evaluation/guidance.py), running:
  - the ORIGINAL steering vector on its source model (Llama-3.2-1B-Instruct)
  - the TRANSLATED steering vector on the target model  (Gemma-3-1B-it)

Both evaluated on the same CAA A/B test set for the chosen behavior.
The goal is to verify that the translated vector produces a similar steering
effect on the target model as the original vector does on the source model.

Usage (from tests_on_translation/):
    conda run -n acteng python ab_comparison.py
    conda run -n acteng python ab_comparison.py --behavior sycophancy --limit 30
    conda run -n acteng python ab_comparison.py --coefficients -20 -10 0 10 20

The translated vector must already exist (run translate_steering_vector.py first):
    steering_vectors/google_gemma-3-1B-it/{method}/{behavior}/{module}/layer_{layer}/sv.pt
"""

import argparse
import json
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Repo roots ─────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_ACTENG = _HERE.parent / "activation_engineering"


# ── Load .env from activation_engineering (sets HF_CACHE_DIR etc.) ────────────

def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — sets vars not already present in the environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv(_ACTENG / ".env")


# ── Model loading ──────────────────────────────────────────────────────────────

def _load_model(model_name: str):
    cache = os.getenv("HF_CACHE_DIR")
    print(f"  Loading {model_name} …")
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, cache_dir=cache
    ).to("cuda")
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    return model, tok


# ── Steering hook ──────────────────────────────────────────────────────────────

def _get_layer(model, layer_idx: int, module_name: str):
    """Architecture-agnostic layer lookup (mirrors base.py::_get_target_module)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer = model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer = model.transformer.h[layer_idx]
    else:
        raise ValueError(f"Unsupported model architecture: {type(model)}")
    return layer if module_name == "residual" else getattr(layer, module_name)


@contextmanager
def _apply_steering(model, layer_idx: int, module_name: str,
                    sv: torch.Tensor, coeff: float, prompt_len: int):
    """Additive steering hook (mirrors base.py::_make_additive_hook)."""
    target = _get_layer(model, layer_idx, module_name)

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


# ── A/B evaluation ─────────────────────────────────────────────────────────────

@dataclass
class ABResult:
    a_prob: float
    b_prob: float
    behavior_prob: float
    is_match: bool
    coefficient: float
    question: str


def _prompt_len(tok, messages) -> int:
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
            plen = _prompt_len(tok, messages)

            full = (
                tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                + "Answer: ("
            )
            tokens = tok(full, return_tensors="pt", add_special_tokens=False).to(model.device)

            with torch.no_grad():
                if coeff == 0:
                    out = model(**tokens)
                else:
                    with _apply_steering(model, layer_idx, module_name, sv, coeff, plen):
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


# ── Reporting ──────────────────────────────────────────────────────────────────

def _summarize(results: List[ABResult], label: str, baseline_acc: float = None):
    by_coeff = defaultdict(list)
    for r in results:
        by_coeff[r.coefficient].append(r)

    print(f"\n  {label}")
    header = f"  {'coeff':>8}  {'avg P(match)':>13}  {'accuracy':>9}  {'Δ acc':>7}  {'n':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for coeff in sorted(by_coeff):
        items = by_coeff[coeff]
        avg_p = sum(r.behavior_prob for r in items) / len(items)
        acc = sum(1 for r in items if r.is_match) / len(items)
        n = len(items)
        delta = f"{acc - baseline_acc:+.4f}" if baseline_acc is not None and coeff != 0 else "      —"
        print(f"  {coeff:>+8.1f}  {avg_p:>13.4f}  {acc:>9.4f}  {delta:>7}  {n:>5}")
        if coeff == 0:
            baseline_acc = acc  # use coeff=0 row as the delta baseline


def _comparison_table(src_results, tgt_results, src_label, tgt_label):
    src_by = defaultdict(list)
    tgt_by = defaultdict(list)
    for r in src_results:
        src_by[r.coefficient].append(r)
    for r in tgt_results:
        tgt_by[r.coefficient].append(r)

    all_coeffs = sorted(set(src_by) | set(tgt_by))
    col = 14
    print(f"\n  {'coeff':>8}  "
          f"{'P(match) src':>{col}}  {'acc src':>8}  "
          f"{'P(match) tgt':>{col}}  {'acc tgt':>8}  "
          f"{'Δ P(match)':>11}")
    print("  " + "-" * (8 + 2 + col + 2 + 8 + 2 + col + 2 + 8 + 2 + 11))

    for coeff in all_coeffs:
        si = src_by.get(coeff, [])
        ti = tgt_by.get(coeff, [])
        sp = sum(r.behavior_prob for r in si) / len(si) if si else float("nan")
        tp = sum(r.behavior_prob for r in ti) / len(ti) if ti else float("nan")
        sa = sum(1 for r in si if r.is_match) / len(si) if si else float("nan")
        ta = sum(1 for r in ti if r.is_match) / len(ti) if ti else float("nan")
        delta = f"{tp - sp:+.4f}"
        print(f"  {coeff:>+8.1f}  "
              f"{sp:>{col}.4f}  {sa:>8.4f}  "
              f"{tp:>{col}.4f}  {ta:>8.4f}  "
              f"{delta:>11}")


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="A/B comparison of original vs translated steering vectors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--behavior",  default="sycophancy",
                   help="CAA behavior name (default: sycophancy)")
    p.add_argument("--method",    default="CAA",
                   help="Steering method folder name (default: CAA)")
    p.add_argument("--source-layer", type=int, default=8,
                   help="Layer of the source steering vector (default: 8)")
    p.add_argument("--target-layer", type=int, default=None,
                   help="Layer of the translated steering vector (default: same as --source-layer)")
    p.add_argument("--module",    default="residual",
                   help="Module name (default: residual)")
    p.add_argument("--coefficients", type=float, nargs="+",
                   default=[-20, -10, 0, 10, 20],
                   help="Steering coefficients to evaluate (default: -20 -10 0 10 20)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of test items per coefficient (useful for quick runs)")
    p.add_argument("--source-model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--target-model", default="google/gemma-3-1B-it")
    p.add_argument("--acteng-root",  default=str(_ACTENG),
                   help="Path to activation_engineering repo")
    p.add_argument("--translation-root", default=str(_HERE),
                   help="Path to tests_on_translation repo (where translated SVs live)")
    return p.parse_args()


def main():
    args = parse_args()
    acteng = Path(args.acteng_root)
    translation = Path(args.translation_root)
    target_layer = args.target_layer if args.target_layer is not None else args.source_layer

    src_sv = (
        acteng / "steering_vectors"
        / args.source_model.replace("/", "_")
        / args.method / args.behavior / args.module
        / f"layer_{args.source_layer}" / "sv.pt"
    )
    tgt_sv = (
        translation / "steering_vectors"
        / args.target_model.replace("/", "_")
        / args.method / args.behavior / args.module
        / f"layer_{target_layer}" / "sv.pt"
    )
    data_dir = acteng / "data" / "CAA_datasets"

    print("=" * 64)
    print(f"  A/B Comparison: {args.method} / {args.behavior}")
    print("=" * 64)
    print(f"  Source SV    : {src_sv}")
    print(f"  Translated SV: {tgt_sv}")
    print(f"  Data dir     : {data_dir}")
    print(f"  Coefficients : {args.coefficients}")
    if args.limit:
        print(f"  Limit        : {args.limit} items per coeff")
    print()

    for path, label in [(src_sv, "source SV"), (tgt_sv, "translated SV")]:
        if not path.exists():
            hint = "\nRun translate_steering_vector.py first." if "translation" in str(path) else ""
            raise FileNotFoundError(f"{label} not found: {path}{hint}")

    test_items = json.loads((data_dir / "test" / args.behavior / "test_dataset_ab.json").read_text())
    print(f"  Loaded {len(test_items)} test items for '{args.behavior}'\n")

    # ── Source model ──────────────────────────────────────────────────────────
    print(f"[1/2] {args.source_model}")
    src_model, src_tok = _load_model(args.source_model)
    src_sv_tensor = torch.load(src_sv, map_location=src_model.device, weights_only=True)

    src_results = run_ab_eval(
        src_model, src_tok, test_items, src_sv_tensor,
        layer_idx=args.source_layer,
        module_name=args.module,
        coefficients=args.coefficients,
        limit=args.limit,
    )

    del src_model, src_tok, src_sv_tensor
    torch.cuda.empty_cache()

    # ── Target model ──────────────────────────────────────────────────────────
    print(f"\n[2/2] {args.target_model}")
    tgt_model, tgt_tok = _load_model(args.target_model)
    tgt_sv_tensor = torch.load(tgt_sv, map_location=tgt_model.device, weights_only=True)

    tgt_results = run_ab_eval(
        tgt_model, tgt_tok, test_items, tgt_sv_tensor,
        layer_idx=target_layer,
        module_name=args.module,
        coefficients=args.coefficients,
        limit=args.limit,
    )

    del tgt_model, tgt_tok, tgt_sv_tensor
    torch.cuda.empty_cache()

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  RESULTS — per-model breakdown")
    print("=" * 64)
    _summarize(src_results, f"ORIGINAL   {args.source_model}")
    _summarize(tgt_results, f"TRANSLATED {args.target_model}")

    print("\n" + "=" * 64)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 64)
    _comparison_table(
        src_results, tgt_results,
        args.source_model, args.target_model,
    )
    print()


if __name__ == "__main__":
    main()
