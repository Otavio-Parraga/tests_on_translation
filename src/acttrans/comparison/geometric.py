"""
Approach 1 — Geometric comparison of translated vs native CAA steering vectors,
entirely in the target model's (Llama-3.2-3B) activation space.

For every translator x behavior it computes:
  - cosine(translated, native)          the headline direction-agreement number
  - centered cosine                      same, after removing each vector's
                                         component along the mean-activation
                                         direction (mean-collapse control)
  - cos with the mean-activation dir     of both translated and native vectors
  - norm diagnostics                     raw translator-output norm, transported
                                         norm, native norm, and their ratios
  - random-vector null                   |cos| scale expected by chance in D=3072

Per translator (across behaviors):
  - cross-behavior confusion matrix      cos(translated_i, native_j): the
                                         diagonal should dominate, otherwise the
                                         translation preserves "a populated
                                         direction" but not behavior identity
  - rank of the correct behavior + margin (diag - best off-diag)
  - RSA / second-order agreement         correlation between the native 7x7
                                         pairwise-cosine matrix and the
                                         translated one (is relational geometry
                                         preserved even if absolute cosines are
                                         mediocre?)
  - layer sweep                          cos(translated, native at EVERY 3B
                                         layer with an SV): is the mismatch
                                         "wrong direction" or "right direction,
                                         wrong depth"?

Outputs (under {out}/geometric/):
  per_behavior.csv    one row per translator x behavior
  cross_behavior.csv  long format: translator, behavior_translated, behavior_native, cos
  layer_sweep.csv     long format: translator, behavior, native_layer, cos
  per_translator.csv  aggregates: mean cos, RSA, identity-rank stats
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .common import (
    BEHAVIORS,
    DEFAULT_OUT_DIR,
    SOURCE_MODEL,
    TARGET_MODEL,
    TranslatorInfo,
    TranslatorRunner,
    available_native_layers,
    centered_cosine,
    cosine,
    load_sv,
    mean_activation_direction,
    pearson,
    random_cosine_null,
    spearman,
)


def _pairwise_cos_upper(vecs: List[torch.Tensor]) -> List[float]:
    """Upper triangle (i<j) of the pairwise cosine matrix, row-major order."""
    out = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            out.append(cosine(vecs[i], vecs[j]))
    return out


def run(
    translators: List[TranslatorInfo],
    behaviors: Optional[List[str]] = None,
    device: str = "cpu",
    out_dir: Path = DEFAULT_OUT_DIR,
    layer_sweep: bool = True,
) -> Dict[str, dict]:
    behaviors = behaviors or BEHAVIORS
    out = Path(out_dir) / "geometric"
    out.mkdir(parents=True, exist_ok=True)

    per_behavior_rows: List[dict] = []
    cross_rows: List[dict] = []
    sweep_rows: List[dict] = []
    per_translator_rows: List[dict] = []
    summary: Dict[str, dict] = {}

    # Native SVs and layer lists are shared across translators; cache per layer.
    native_cache: Dict[tuple, torch.Tensor] = {}

    def native(behavior: str, layer: int) -> torch.Tensor:
        key = (behavior, layer)
        if key not in native_cache:
            native_cache[key] = load_sv(TARGET_MODEL, behavior, layer)
        return native_cache[key]

    null = None  # random-cosine null, computed once from the first native SV

    for info in translators:
        print(f"\n=== {info.name} ===")
        runner = TranslatorRunner(info.path, device=device)
        norm_mode = runner.default_norm_mode()
        mean_dir = mean_activation_direction(
            TARGET_MODEL, layer=info.tgt_layer, pooling=info.pooling
        )

        translated: Dict[str, torch.Tensor] = {}
        natives: Dict[str, torch.Tensor] = {}
        skipped: List[str] = []

        for behavior in behaviors:
            try:
                src = load_sv(SOURCE_MODEL, behavior, info.src_layer)
            except FileNotFoundError as e:
                print(f"  !! {e}")
                skipped.append(behavior)
                continue
            try:
                v_n = native(behavior, info.tgt_layer)
            except FileNotFoundError as e:
                print(f"  !! {e}")
                skipped.append(behavior)
                continue

            v_t = runner.transport(src, norm_mode=norm_mode)
            raw = runner.transport(src, norm_mode="none")
            translated[behavior] = v_t
            natives[behavior] = v_n

            if null is None:
                null = random_cosine_null(v_n.numel(), v_n)

            row = {
                "translator": info.name,
                "ttype": info.ttype,
                "loss": info.loss,
                "pooling": info.pooling,
                "src_layer": info.src_layer,
                "tgt_layer": info.tgt_layer,
                "norm_mode": norm_mode,
                "behavior": behavior,
                "cos_native": cosine(v_t, v_n),
                "cos_centered": (
                    centered_cosine(v_t, v_n, mean_dir) if mean_dir is not None else None
                ),
                "cos_meandir_translated": (
                    cosine(v_t, mean_dir) if mean_dir is not None else None
                ),
                "cos_meandir_native": (
                    cosine(v_n, mean_dir) if mean_dir is not None else None
                ),
                "norm_native": v_n.norm().item(),
                "norm_source": src.norm().item(),
                "norm_raw_output": raw.norm().item(),
                "norm_transported": v_t.norm().item(),
                "norm_ratio_transported_vs_native": (v_t.norm() / v_n.norm()).item(),
                "null_cos_mean_abs": null["mean_abs"],
                "null_cos_std": null["std"],
            }
            per_behavior_rows.append(row)
            print(
                f"  {behavior:24s} cos={row['cos_native']:+.3f}"
                + (f"  centered={row['cos_centered']:+.3f}" if row["cos_centered"] is not None else "")
                + f"  |T(v)|/|v_native|={row['norm_ratio_transported_vs_native']:.2f}"
            )

        done = [b for b in behaviors if b in translated]
        if len(done) < 2:
            print("  !! fewer than 2 behaviors translated; skipping matrix/RSA")
            continue

        # Cross-behavior confusion matrix + identity rank/margin.
        ranks, margins = [], []
        for bi in done:
            cos_row = {bj: cosine(translated[bi], natives[bj]) for bj in done}
            for bj, c in cos_row.items():
                cross_rows.append(
                    {"translator": info.name, "behavior_translated": bi,
                     "behavior_native": bj, "cos": c}
                )
            ordered = sorted(cos_row, key=cos_row.get, reverse=True)
            rank = ordered.index(bi) + 1
            off = [c for bj, c in cos_row.items() if bj != bi]
            ranks.append(rank)
            margins.append(cos_row[bi] - max(off))

        # RSA: second-order similarity between native and translated geometry.
        upper_native = _pairwise_cos_upper([natives[b] for b in done])
        upper_trans = _pairwise_cos_upper([translated[b] for b in done])
        rsa_p = pearson(upper_native, upper_trans)
        rsa_s = spearman(upper_native, upper_trans)

        cos_vals = [r["cos_native"] for r in per_behavior_rows if r["translator"] == info.name]
        cen_vals = [r["cos_centered"] for r in per_behavior_rows
                    if r["translator"] == info.name and r["cos_centered"] is not None]
        agg = {
            "translator": info.name,
            "ttype": info.ttype,
            "loss": info.loss,
            "pooling": info.pooling,
            "src_layer": info.src_layer,
            "tgt_layer": info.tgt_layer,
            "n_behaviors": len(done),
            "mean_cos": sum(cos_vals) / len(cos_vals),
            "mean_cos_centered": (sum(cen_vals) / len(cen_vals)) if cen_vals else None,
            "identity_rank1_frac": sum(1 for r in ranks if r == 1) / len(ranks),
            "mean_identity_rank": sum(ranks) / len(ranks),
            "mean_margin": sum(margins) / len(margins),
            "rsa_pearson": rsa_p,
            "rsa_spearman": rsa_s,
        }
        per_translator_rows.append(agg)
        summary[info.name] = agg
        print(
            f"  -> mean cos={agg['mean_cos']:+.3f}"
            + (f"  centered={agg['mean_cos_centered']:+.3f}" if agg["mean_cos_centered"] is not None else "")
            + f"  rank1={agg['identity_rank1_frac']:.0%}  margin={agg['mean_margin']:+.3f}"
            + f"  RSA(p/s)={rsa_p:+.2f}/{rsa_s:+.2f}"
        )

        # Layer sweep: is the translated direction better matched at another depth?
        if layer_sweep:
            for behavior in done:
                best_layer, best_cos = None, -2.0
                for layer in available_native_layers(TARGET_MODEL, behavior):
                    c = cosine(translated[behavior], native(behavior, layer))
                    sweep_rows.append(
                        {"translator": info.name, "behavior": behavior,
                         "native_layer": layer, "cos": c}
                    )
                    if c > best_cos:
                        best_layer, best_cos = layer, c
                if best_layer is not None and best_layer != info.tgt_layer:
                    print(
                        f"  layer sweep: {behavior} best matches native layer "
                        f"{best_layer} (cos={best_cos:+.3f}) not target layer {info.tgt_layer}"
                    )

    _write_csv(out / "per_behavior.csv", per_behavior_rows)
    _write_csv(out / "cross_behavior.csv", cross_rows)
    _write_csv(out / "per_translator.csv", per_translator_rows)
    if layer_sweep:
        _write_csv(out / "layer_sweep.csv", sweep_rows)

    print(f"\nGeometric comparison written to {out}/")
    return summary


def _write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")
