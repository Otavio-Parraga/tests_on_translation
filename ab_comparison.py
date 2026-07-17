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
from pathlib import Path

import torch
from dotenv import load_dotenv

from acttrans.evaluation.ab_eval import aggregate_by_coefficient, run_ab_eval
from acttrans.utils.hf import load_model_and_tokenizer
from acttrans.utils.paths import sv_path


# ── Repo roots ─────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve().parent
_ACTENG = _HERE.parent / "activation_engineering"

# Load .env from activation_engineering (sets HF_CACHE_DIR etc.); never
# overrides variables already present in the environment.
load_dotenv(_ACTENG / ".env")


# ── Reporting ──────────────────────────────────────────────────────────────────

def _summarize(results, label: str, baseline_acc: float = None):
    agg = aggregate_by_coefficient(results)

    print(f"\n  {label}")
    header = f"  {'coeff':>8}  {'avg P(match)':>13}  {'accuracy':>9}  {'Δ acc':>7}  {'n':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for coeff in sorted(agg):
        row = agg[coeff]
        acc = row["accuracy"]
        delta = f"{acc - baseline_acc:+.4f}" if baseline_acc is not None and coeff != 0 else "      —"
        print(f"  {coeff:>+8.1f}  {row['avg_p_match']:>13.4f}  {acc:>9.4f}  {delta:>7}  {row['n']:>5}")
        if coeff == 0:
            baseline_acc = acc  # use coeff=0 row as the delta baseline


def _comparison_table(src_results, tgt_results, src_label, tgt_label):
    src_agg = aggregate_by_coefficient(src_results)
    tgt_agg = aggregate_by_coefficient(tgt_results)

    all_coeffs = sorted(set(src_agg) | set(tgt_agg))
    col = 14
    print(f"\n  {'coeff':>8}  "
          f"{'P(match) src':>{col}}  {'acc src':>8}  "
          f"{'P(match) tgt':>{col}}  {'acc tgt':>8}  "
          f"{'Δ P(match)':>11}")
    print("  " + "-" * (8 + 2 + col + 2 + 8 + 2 + col + 2 + 8 + 2 + 11))

    for coeff in all_coeffs:
        si = src_agg.get(coeff)
        ti = tgt_agg.get(coeff)
        sp = si["avg_p_match"] if si else float("nan")
        tp = ti["avg_p_match"] if ti else float("nan")
        sa = si["accuracy"] if si else float("nan")
        ta = ti["accuracy"] if ti else float("nan")
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

    src_sv = sv_path(acteng, args.source_model, args.method, args.behavior,
                     args.module, args.source_layer)
    tgt_sv = sv_path(translation, args.target_model, args.method, args.behavior,
                     args.module, target_layer)
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
    print(f"  Loading {args.source_model} …")
    src_model, src_tok = load_model_and_tokenizer(args.source_model, "cuda")
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
    print(f"  Loading {args.target_model} …")
    tgt_model, tgt_tok = load_model_and_tokenizer(args.target_model, "cuda")
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
