"""
Approach 2 — Decomposition of the translated vector: where does the
disagreement with the native vector live?

Each translated vector v_T is split against the native vector v_N:

    v_T = (v_T . v_N_hat) v_N_hat  +  r        (parallel + residual)

and the residual r is then characterized:
  - fraction of v_T's energy that is parallel to v_N vs residual
  - cos(r, mean-activation direction)    mean leakage: the translator injected
                                         the generic "populated cone" direction
  - cos(r, other behaviors' native SVs)  behavior confusion: the residual is
                                         another behavior's direction
  - least-squares projection of v_T onto span{native SVs of all behaviors}:
    in-span energy fraction + per-behavior coefficients. A translated vector
    that is mostly in-span but with off-behavior coefficients is confused; one
    that is mostly out-of-span carries directions the method never found (or noise).

Runs over one or more discovery methods (CAA / RepE / GCAV). Each method is
decomposed against ITS OWN native basis — a translated RepE vector is judged
against the span of native RepE vectors — so the per-method numbers are
independent readings of the same question, not one basis reused.

Outputs (under {out}/decomposition/):
  per_behavior.csv  one row per method x translator x behavior
  span_coefs.csv    long: method, translator, behavior_translated, basis_behavior, coef

The returned summary dict is keyed "<method>/<translator>".
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .common import (
    BEHAVIORS,
    DEFAULT_OUT_DIR,
    METHOD,
    TranslatorInfo,
    TranslatorRunner,
    cosine,
    load_sv,
    mean_activation_direction,
)


def run(
    translators: List[TranslatorInfo],
    behaviors: Optional[List[str]] = None,
    device: str = "cpu",
    out_dir: Path = DEFAULT_OUT_DIR,
    methods: Optional[List[str]] = None,
) -> Dict[str, dict]:
    behaviors = behaviors or BEHAVIORS
    methods = list(methods or [METHOD])
    out = Path(out_dir) / "decomposition"
    out.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    coef_rows: List[dict] = []
    summary: Dict[str, dict] = {}

    # Cache native SVs per (method, model, behavior, layer) so a mixed run (both
    # 1B->3B and 3B->1B checkpoints, several methods) stays correct.
    native_cache: Dict[tuple, torch.Tensor] = {}

    def native(method: str, model: str, behavior: str, layer: int) -> torch.Tensor:
        key = (method, model, behavior, layer)
        if key not in native_cache:
            native_cache[key] = load_sv(model, behavior, layer, method=method)
        return native_cache[key]

    for info in translators:
        print(f"\n=== {info.name} ===")
        runner = TranslatorRunner(info.path, device=device)
        # Read the model pair from the checkpoint so decomposition is
        # direction-agnostic (1B->3B and 3B->1B both work).
        src_model = runner.config["source_model"]["name"]
        tgt_model = runner.config["target_model"]["name"]
        norm_mode = runner.default_norm_mode()
        # Property of the target model/layer, so shared across methods.
        mean_dir = mean_activation_direction(
            tgt_model, layer=info.tgt_layer, pooling=info.pooling
        )

        for method in methods:
            print(f"  -- method {method} --")
            # Native basis at the target layer, in THIS method's own vectors:
            # the span a translated RepE vector is judged against is the span of
            # native RepE vectors, not of CAA's.
            basis: Dict[str, torch.Tensor] = {}
            for b in behaviors:
                try:
                    basis[b] = native(method, tgt_model, b, info.tgt_layer)
                except FileNotFoundError as e:
                    print(f"  !! {e}")
            if not basis:
                continue
            basis_names = list(basis)
            B = torch.stack([basis[b] for b in basis_names])  # [K, D]

            par_fracs = []
            for behavior in basis_names:
                try:
                    src = load_sv(src_model, behavior, info.src_layer, method=method)
                except FileNotFoundError as e:
                    print(f"  !! {e}")
                    continue

                v_t = runner.transport(src, norm_mode=norm_mode)
                v_n = basis[behavior]
                n_hat = F.normalize(v_n, dim=-1)

                # Parallel/residual split against the native vector.
                coef_par = (v_t @ n_hat).item()          # signed length along v_N
                r = v_t - coef_par * n_hat
                energy_t = (v_t @ v_t).item()
                par_energy_frac = coef_par**2 / energy_t if energy_t > 0 else float("nan")

                # Residual characterization.
                cos_r_mean = cosine(r, mean_dir) if mean_dir is not None else None
                off = {
                    b: cosine(r, basis[b]) for b in basis_names if b != behavior
                }
                worst_b = max(off, key=lambda b: abs(off[b])) if off else None

                # Least-squares fit of v_T in the span of ALL native SVs.
                # coefs: [K], minimizing ||B^T coefs - v_t||.
                coefs = torch.linalg.lstsq(B.T, v_t.unsqueeze(1)).solution.squeeze(1)
                in_span = B.T @ coefs
                in_span_frac = ((in_span @ in_span) / energy_t).item() if energy_t > 0 else float("nan")
                for b, c in zip(basis_names, coefs.tolist()):
                    coef_rows.append(
                        {"method": method, "translator": info.name,
                         "behavior_translated": behavior, "basis_behavior": b, "coef": c}
                    )

                row = {
                    "method": method,
                    "translator": info.name,
                    "ttype": info.ttype,
                    "loss": info.loss,
                    "pooling": info.pooling,
                    "src_model": src_model,
                    "tgt_model": tgt_model,
                    "src_layer": info.src_layer,
                    "tgt_layer": info.tgt_layer,
                    "norm_mode": norm_mode,
                    "behavior": behavior,
                    "norm_translated": v_t.norm().item(),
                    "norm_native": v_n.norm().item(),
                    "coef_parallel": coef_par,
                    "parallel_energy_frac": par_energy_frac,
                    "residual_norm": r.norm().item(),
                    "residual_energy_frac": 1.0 - par_energy_frac,
                    "cos_residual_meandir": cos_r_mean,
                    "max_abs_cos_residual_offbehavior": abs(off[worst_b]) if worst_b else None,
                    "argmax_offbehavior": worst_b,
                    "in_span_energy_frac": in_span_frac,
                }
                rows.append(row)
                par_fracs.append(par_energy_frac)
                print(
                    f"    {behavior:24s} parallel={par_energy_frac:.1%}"
                    + (f"  cos(r,mean)={cos_r_mean:+.3f}" if cos_r_mean is not None else "")
                    + (f"  worst off-behavior |cos(r,.)|={abs(off[worst_b]):.3f} ({worst_b})" if worst_b else "")
                    + f"  in-span={in_span_frac:.1%}"
                )

            if par_fracs:
                agg = {
                    "method": method,
                    "translator": info.name,
                    "mean_parallel_energy_frac": sum(par_fracs) / len(par_fracs),
                }
                summary[f"{method}/{info.name}"] = agg
                print(f"    -> mean parallel energy fraction: "
                      f"{agg['mean_parallel_energy_frac']:.1%}")

    _write_csv(out / "per_behavior.csv", rows)
    _write_csv(out / "span_coefs.csv", coef_rows)
    print(f"\nDecomposition written to {out}/")
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
