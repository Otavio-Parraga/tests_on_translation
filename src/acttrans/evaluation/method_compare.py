"""Cross-method reading of ab_sweep results (CAA vs RepE vs GCAV).

WHY THIS IS NOT ab_report.py
----------------------------
ab_report.py compares translators *within* one discovery method, so it can index
everything by raw `coefficient`. Across methods that axis is meaningless: a CAA
vector carries its own behavior-dependent magnitude (|sv| ~0.2 to ~5) while RepE
and GCAV are unit-norm by construction, so "coefficient 5" is a ~5x larger
physical perturbation for CAA-refusal than for RepE. Comparing methods at equal
coefficient would reward CAA for nothing but its scale convention.

The fix is to read every run on two axes that survive a change of scale:

  dose = coefficient * sv_norm
        The actual size of the vector added to the residual stream. This is the
        common x-axis; it is the post-hoc version of the "relative" hook in
        activation_engineering/methods/base.py, computed from `sv_norm`, which
        ab_sweep already logs on every row.

  dP_peak = max over the symmetric grid of [ P(match | +c) - P(match | -c) ]
        The best behavior swing the vector achieves at ANY dose, with the dose
        that achieves it reported alongside. Searching for the peak instead of
        fixing a coefficient is what makes the number method-comparable.

        This needs no coherence cutoff. Past the coherent window the model
        collapses and P(match) -> 0 on BOTH sides, so those doses contribute
        ~0 swing and can never be the argmax — the peak search excludes the
        collapsed tail on its own, rather than by an arbitrary |c| <= 5 rule
        that would itself mean different doses per method.

Reported per (method, scope, translator, behavior), then aggregated per
(method, scope) so the headline question — does translation carry RepE/GCAV
directions as well as it carries CAA ones — is one table lookup.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# scope meanings, as written by ab_sweep.py
SCOPES = ("source", "target", "native")


def normalize_method(row: dict) -> str:
    """Method of a result row; rows predating the method dimension are CAA."""
    return row.get("method") or "CAA"


def block_key(row: dict) -> Tuple:
    """One evaluation block = one steering curve over the coefficient grid."""
    return (
        normalize_method(row),
        row.get("scope", ""),
        row.get("translator", ""),
        row.get("norm_mode", ""),
        row.get("behavior", ""),
    )


def _spearman(xs: List[float], ys: List[float]) -> float:
    """Rank correlation, no scipy. Returns nan for degenerate input."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(vs):
        order = sorted(range(n), key=lambda i: vs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx ** 0.5 * vy ** 0.5)


def analyze_block(rows: List[dict]) -> dict:
    """Peak-swing metrics for one steering curve.

    `rows` are the per-coefficient rows of a single block (same method, scope,
    translator, norm_mode, behavior). Returns the dose-axis metrics described in
    the module docstring.
    """
    # sv_norm is constant within a block; guard against it being absent/zero so a
    # malformed row degrades to a dose axis equal to the coefficient axis.
    sv_norm = next((r.get("sv_norm") for r in rows if r.get("sv_norm")), None)
    if not sv_norm or sv_norm != sv_norm:  # None / 0 / nan
        sv_norm = float("nan")

    p = {}     # coefficient -> avg_p_match
    acc = {}   # coefficient -> accuracy
    for r in rows:
        c = r["coefficient"]
        p[c] = r["avg_p_match"]
        acc[c] = r["accuracy"]

    # Peak swing over symmetric coefficient pairs present in the grid.
    best = None
    for c in sorted(x for x in p if x > 0):
        if -c not in p:
            continue
        swing = p[c] - p[-c]
        if best is None or swing > best[1]:
            best = (c, swing)

    coeff_peak = best[0] if best else float("nan")
    dP_peak = best[1] if best else float("nan")

    # Monotonicity of the response up to the peak: does dialing the coefficient
    # up steadily raise P(match)? Restricted to [-c*, +c*] because past the peak
    # the curve turns over as the model collapses, which is not a defect.
    if best:
        window = sorted(c for c in p if abs(c) <= coeff_peak)
        mono = _spearman(window, [p[c] for c in window])
    else:
        mono = float("nan")

    ex = rows[0]
    return {
        "method": normalize_method(ex),
        "scope": ex.get("scope", ""),
        "translator": ex.get("translator", ""),
        "translator_type": ex.get("translator_type", ""),
        "loss": ex.get("loss", ""),
        "pooling": ex.get("pooling", ""),
        "norm_mode": ex.get("norm_mode", ""),
        "behavior": ex.get("behavior", ""),
        "source_layer": ex.get("source_layer"),
        "target_layer": ex.get("target_layer"),
        "sv_norm": sv_norm,
        "coeff_at_peak": coeff_peak,
        "dose_at_peak": coeff_peak * sv_norm,
        "dP_peak": dP_peak,
        "acc_swing_at_peak": (
            acc.get(coeff_peak, float("nan")) - acc.get(-coeff_peak, float("nan"))
            if best else float("nan")
        ),
        "p_at_0": p.get(0.0, float("nan")),
        "monotonic_to_peak": mono,
        "n_coefficients": len(p),
    }


def analyze(rows: List[dict], methods: Optional[List[str]] = None) -> List[dict]:
    """Per-block dose metrics for every block in `rows`."""
    blocks: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in rows:
        if methods and normalize_method(r) not in methods:
            continue
        blocks[block_key(r)].append(r)
    out = [analyze_block(rs) for rs in blocks.values()]
    out.sort(key=lambda r: (r["method"], r["scope"], r["behavior"], r["translator"]))
    return out


def _mean(vals: List[float]) -> float:
    vs = [v for v in vals if v == v]
    return sum(vs) / len(vs) if vs else float("nan")


def _median(vals: List[float]) -> float:
    vs = sorted(v for v in vals if v == v)
    if not vs:
        return float("nan")
    mid = len(vs) // 2
    return vs[mid] if len(vs) % 2 else (vs[mid - 1] + vs[mid]) / 2.0


# Minimum native swing that may serve as a retention denominator. A native vector
# that barely steers (dP_peak ~ 0) is not a meaningful reference: dividing by it
# turns noise into retention figures of several hundred percent. Below this floor
# retention is reported as nan instead.
MIN_NATIVE_DP = 0.02


def summarize(block_rows: List[dict],
              min_native_dP: float = MIN_NATIVE_DP) -> List[dict]:
    """Aggregate per (method, scope): the headline cross-method table.

    `retention` answers the question the whole repo exists for: of the steering
    swing the natively-extracted target vector achieves, how much does the
    translated vector recover? It is computed per behavior against the native
    block for the same method and target layer, then averaged — a native-scope
    row therefore has retention 1.0 by construction and a source-scope row has
    none (it is a different model).

    Behaviors whose native reference swings less than `min_native_dP` are left
    out of the retention average and counted in `n_no_native_ref`, so a weak
    native baseline shows up as missing coverage rather than as a wild ratio.
    """
    # native reference: (method, target_layer, behavior) -> dP_peak
    native_dP: Dict[Tuple, float] = {}
    for r in block_rows:
        if r["scope"] == "native":
            native_dP[(r["method"], r["target_layer"], r["behavior"])] = r["dP_peak"]

    groups: Dict[Tuple, List[dict]] = defaultdict(list)
    for r in block_rows:
        groups[(r["method"], r["scope"])].append(r)

    out = []
    for (method, scope), rs in sorted(groups.items()):
        retentions = []
        n_no_ref = 0
        for r in rs:
            ref = native_dP.get((method, r["target_layer"], r["behavior"]))
            if ref is not None and ref == ref and ref >= min_native_dP:
                retentions.append(r["dP_peak"] / ref)
            else:
                n_no_ref += 1
        out.append({
            "method": method,
            "scope": scope,
            "n_blocks": len(rs),
            "n_behaviors": len({r["behavior"] for r in rs}),
            "n_translators": len({r["translator"] for r in rs}),
            "mean_dP_peak": _mean([r["dP_peak"] for r in rs]),
            "median_dP_peak": _median([r["dP_peak"] for r in rs]),
            "best_dP_peak": max([r["dP_peak"] for r in rs if r["dP_peak"] == r["dP_peak"]],
                                default=float("nan")),
            "mean_acc_swing": _mean([r["acc_swing_at_peak"] for r in rs]),
            "mean_monotonic_to_peak": _mean([r["monotonic_to_peak"] for r in rs]),
            "median_dose_at_peak": _median([r["dose_at_peak"] for r in rs]),
            "median_coeff_at_peak": _median([r["coeff_at_peak"] for r in rs]),
            "mean_sv_norm": _mean([r["sv_norm"] for r in rs]),
            "mean_retention_vs_native": _mean(retentions) if retentions else float("nan"),
            "n_retention_refs": len(retentions),
            "n_no_native_ref": n_no_ref,
        })
    return out


def best_per_method(block_rows: List[dict], scope: str = "target") -> List[dict]:
    """Best translator per (method, behavior) within a scope — the "how well can
    this method be translated at all" view, ignoring weak translators."""
    best: Dict[Tuple, dict] = {}
    for r in block_rows:
        if r["scope"] != scope:
            continue
        k = (r["method"], r["behavior"])
        if k not in best or r["dP_peak"] > best[k]["dP_peak"]:
            best[k] = r
    return [best[k] for k in sorted(best)]
