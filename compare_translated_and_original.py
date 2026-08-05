"""
Compare steering vectors extracted natively on Llama-3.2-3B against vectors
translated from Llama-3.2-1B through the FineWeb translators.

Covers every single-direction discovery method in acttrans.constants.METHODS
(CAA, RepE, GCAV) — all three are one flat residual-stream vector per
(model, behavior, layer), so they share this tooling. Output tables carry a
`method` column; cosines are scale-free, so the numbers are directly comparable
across methods despite CAA carrying its own magnitude and RepE/GCAV being unit
vectors.

Two approaches, implemented in comparison/:

  geometric      (comparison/geometric.py)  Direction agreement in 3B space:
                 cosine / centered cosine vs the native SV, norm diagnostics,
                 cross-behavior confusion matrix, RSA of the pairwise-cosine
                 geometry, and a sweep of the translated vector against native
                 SVs at every 3B layer.

  decomposition  (comparison/decomposition.py)  Splits each translated vector
                 into its component parallel to the native SV plus a residual,
                 then asks what the residual is: mean-activation leakage,
                 another behavior's direction, or something outside the span
                 of all native SVs.

Usage:
    conda run -n acteng python compare_translated_and_original.py geometric
    conda run -n acteng python compare_translated_and_original.py decomposition
    conda run -n acteng python compare_translated_and_original.py all

    # restrict translators / behaviors / methods, pick a device:
    conda run -n acteng python compare_translated_and_original.py geometric \
        --translators 'outputs/fineweb/best_translator__*mlp__cosine.pt' \
        --behaviors refusal sycophancy --methods RepE GCAV --device cuda:0

Results land in outputs/comparison/{geometric,decomposition}/ as CSVs
(long-format tables ready for pandas/plotting), with a per-run console summary.
"""

import argparse
from pathlib import Path

from acttrans.comparison.common import (
    BEHAVIORS,
    DEFAULT_OUT_DIR,
    DEFAULT_TRANSLATOR_GLOB,
    METHOD,
    METHODS,
    discover_translators,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "approach",
        choices=["geometric", "decomposition", "all"],
        help="Which comparison to run ('all' runs both).",
    )
    parser.add_argument(
        "--translators",
        nargs="+",
        default=[DEFAULT_TRANSLATOR_GLOB],
        metavar="GLOB",
        help=f"Checkpoint glob(s) (default: {DEFAULT_TRANSLATOR_GLOB})",
    )
    parser.add_argument(
        "--behaviors",
        nargs="+",
        default=None,
        choices=BEHAVIORS,
        metavar="BEHAVIOR",
        help=f"Subset of behaviors (default: all 7). Choices: {', '.join(BEHAVIORS)}",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[METHOD],
        choices=list(METHODS),
        metavar="METHOD",
        help=f"Discovery methods to compare (default: {METHOD} only, so existing "
             f"callers keep their previous behavior). Choices: {', '.join(METHODS)}. "
             "Pass several to get one comparable table across methods; every output "
             "table carries a `method` column either way.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if _cuda_available() else "cpu",
        help="Device for translator forward passes (default: cuda if available).",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output root (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--no-layer-sweep",
        action="store_true",
        help="geometric only: skip the sweep against native SVs at every 3B layer.",
    )
    args = parser.parse_args()

    translators = discover_translators(args.translators)
    if not translators:
        parser.error(f"No translator checkpoints matched: {args.translators}")
    print(f"{len(translators)} translator(s), "
          f"{len(args.behaviors or BEHAVIORS)} behavior(s), "
          f"methods={args.methods}, device={args.device}")

    out_dir = Path(args.out)

    if args.approach in ("geometric", "all"):
        from acttrans.comparison import geometric
        geometric.run(
            translators,
            behaviors=args.behaviors,
            device=args.device,
            out_dir=out_dir,
            layer_sweep=not args.no_layer_sweep,
            methods=args.methods,
        )

    if args.approach in ("decomposition", "all"):
        from acttrans.comparison import decomposition
        decomposition.run(
            translators,
            behaviors=args.behaviors,
            device=args.device,
            out_dir=out_dir,
            methods=args.methods,
        )


def _cuda_available() -> bool:
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
