"""Response-metric math shared by the A/B report and dashboards.

These are the small, dependency-free statistics the HTML builders reuse to turn a
per-coefficient P(behavior) curve into headline numbers (Pearson/Spearman
correlation, coherent-window effect size, monotonicity). They live here — as a
public package API — so ``ab_report.py``, ``ab_dashboard.py`` and
``ab_pivot_dashboard.py`` all import the SAME implementation instead of one CLI
reaching into another's private ``_``-prefixed functions.

Pure Python (no numpy): the inputs are short coefficient grids, so hand-rolled
sums keep these importable everywhere without pulling in heavy deps.
"""


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx ** 0.5 * vy ** 0.5)


def spearman(xs, ys):
    """Rank correlation — robust to the uneven coefficient spacing; measures the
    monotonic trend (does P rise as the coefficient rises?) rather than linear fit."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        i = 0
        while i < n:  # average ranks for ties
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    return pearson(ranks(xs), ranks(ys))


def curve(rows_for_block):
    """coeff -> (avg_p_match, accuracy) for one eval block."""
    return {r["coefficient"]: (r["avg_p_match"], r["accuracy"])
            for r in rows_for_block}


def nan_key(x, miss=-9.0):
    """Sort helper: push NaN to the bottom for descending sorts."""
    return x if x == x else miss


def mean(vals):
    vs = [v for v in vals if v == v]
    return sum(vs) / len(vs) if vs else float("nan")


def response_metrics(cur, coherent_max):
    """Given coeff -> (p_match, acc), measure how P(behavior) responds to the
    coefficient WITHIN the coherent regime (|coeff| <= coherent_max), where the
    model has not yet collapsed.

    Returns dict with:
      response_corr : Pearson corr(coeff, P_behavior) over the coherent window.
                      +1 => turning the coefficient up reliably raises P(behavior)
                      (the desired CAA direction);  ~0 => no monotonic response;
                      <0 => coefficient up LOWERS the behavior (wrong direction).
      dP_coherent   : P(beh)@(+max_coh) - P(beh)@(-max_coh)  (signed effect size)
      p_pos / p_neg : P(behavior) at the largest coherent +/- coefficient
      p_at_0        : baseline P(behavior) with no steering
      monotonic     : fraction of adjacent coherent steps where P increases
    """
    coh = sorted(c for c in cur if abs(c) <= coherent_max)
    ps = [cur[c][0] for c in coh]
    # Spearman: monotonic "coeff up -> P(behavior) up?" robust to coeff spacing.
    corr = spearman([float(c) for c in coh], ps) if len(coh) >= 2 else float("nan")

    pos = [c for c in coh if c > 0]
    neg = [c for c in coh if c < 0]
    p_pos = cur[max(pos)][0] if pos else float("nan")
    p_neg = cur[min(neg)][0] if neg else float("nan")
    dP = (p_pos - p_neg) if (pos and neg) else float("nan")

    ups = sum(1 for a, b in zip(ps, ps[1:]) if b > a)
    monotonic = ups / (len(ps) - 1) if len(ps) >= 2 else float("nan")

    return {
        "response_corr": corr,
        "dP_coherent": dP,
        "p_pos": p_pos,
        "p_neg": p_neg,
        "p_at_0": cur.get(0.0, (float("nan"),))[0],
        "monotonic": monotonic,
    }
