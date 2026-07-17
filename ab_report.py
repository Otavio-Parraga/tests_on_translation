"""
Build comparison tables + an HTML report from ab_sweep.py results.

Reads one or more results JSONL files (shards are merged), then writes:

  outputs/ab_eval/results.csv          tidy long table (one row per
                                       scope/translator/norm/behavior/coeff)
  outputs/ab_eval/summary.csv          one row per (translator,norm,behavior):
                                       steering effect size + fidelity to source
  outputs/ab_eval/report.html          per-behavior heatmap tables, source row
                                       vs every translator/norm row, all coeffs

"Effect size" of a steering run = accuracy(max +coeff) - accuracy(max -coeff),
i.e. how strongly the behavior probability swings across the coefficient grid.
"Fidelity" = how closely the target P(match)-vs-coeff curve tracks the source
curve (Pearson correlation over shared coefficients).

Usage:
    conda run -n acteng python ab_report.py
    conda run -n acteng python ab_report.py --results outputs/ab_eval/*.jsonl
"""

import argparse
import csv
import glob
import html
import json
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_OUT = _HERE / "outputs" / "ab_eval"


def load_rows(patterns):
    rows = []
    seen = set()
    for pat in patterns:
        for fp in glob.glob(pat):
            for line in Path(fp).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                # de-dup identical rows across shards / re-runs
                k = (r.get("scope"), r.get("translator"), r.get("norm_mode"),
                     r.get("behavior"), r.get("coefficient"))
                if k in seen:
                    continue
                seen.add(k)
                rows.append(r)
    return rows


def write_results_csv(rows, path):
    cols = ["scope", "translator", "translator_type", "loss", "pooling",
            "norm_mode", "behavior", "coefficient", "avg_p_match", "accuracy",
            "n", "sv_norm", "source_layer", "target_layer"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["scope"], r.get("translator", ""),
                                             r.get("norm_mode", ""),
                                             r["behavior"], r["coefficient"])):
            w.writerow(r)


def _pearson(xs, ys):
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


def _spearman(xs, ys):
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

    return _pearson(ranks(xs), ranks(ys))


def _curve(rows_for_block):
    """coeff -> (avg_p_match, accuracy) for one eval block."""
    return {r["coefficient"]: (r["avg_p_match"], r["accuracy"])
            for r in rows_for_block}


def _nan_key(x, miss=-9.0):
    """Sort helper: push NaN to the bottom for descending sorts."""
    return x if x == x else miss


def _response_metrics(cur, coherent_max):
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
    corr = _spearman([float(c) for c in coh], ps) if len(coh) >= 2 else float("nan")

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


def build_summary(rows, coherent_max=5.0):
    # index by block
    blocks = defaultdict(list)
    for r in rows:
        blocks[(r["scope"], r.get("translator", ""), r.get("norm_mode", ""),
                r["behavior"])].append(r)

    # Source curves + their own response metrics, keyed by (source layer,
    # behavior): with layer-pair sweeps there is one baseline per source layer,
    # and each translator is judged against the baseline of ITS source layer.
    # Rows from runs predating layer support carry no source_layer -> default 8.
    source_curve = {}
    source_stats = {}
    for (scope, tr, nm, beh), rs in blocks.items():
        if scope == "source":
            lay = rs[0].get("source_layer", 8)
            source_curve[(lay, beh)] = _curve(rs)
            source_stats[(lay, beh)] = _response_metrics(source_curve[(lay, beh)],
                                                         coherent_max)

    summary = []
    for (scope, tr, nm, beh), rs in blocks.items():
        if scope != "target":
            continue
        cur = _curve(rs)
        m = _response_metrics(cur, coherent_max)

        # fidelity vs source: corr of the two P(behavior) curves, but ONLY over
        # the coherent window — beyond it both collapse to ~0 and spuriously
        # "agree", which inflates the score.
        fid = float("nan")
        src_lay = rs[0].get("source_layer", 8)
        sc = source_curve.get((src_lay, beh))
        if sc:
            shared = sorted(c for c in (set(cur) & set(sc)) if abs(c) <= coherent_max)
            if len(shared) >= 2:
                fid = _pearson([cur[c][0] for c in shared],
                               [sc[c][0] for c in shared])

        ex = rs[0]
        summary.append({
            "translator": tr,
            "translator_type": ex.get("translator_type", ""),
            "loss": ex.get("loss", ""),
            "pooling": ex.get("pooling", ""),
            "norm_mode": nm,
            "behavior": beh,
            "response_corr": m["response_corr"],
            "dP_coherent": m["dP_coherent"],
            "p_neg": m["p_neg"],
            "p_at_0": m["p_at_0"],
            "p_pos": m["p_pos"],
            "monotonic": m["monotonic"],
            "fidelity_vs_source": fid,
            "sv_norm": ex.get("sv_norm", float("nan")),
            "source_layer": src_lay,
            "target_layer": ex.get("target_layer", 8),
        })
    return summary, source_curve, source_stats


def write_summary_csv(summary, path):
    cols = ["translator_type", "loss", "pooling", "norm_mode", "behavior",
            "response_corr", "dP_coherent", "p_neg", "p_at_0", "p_pos",
            "monotonic", "fidelity_vs_source", "sv_norm",
            "source_layer", "target_layer", "translator"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        # sort: behavior, then strongest "coeff up -> more behavior" first
        for r in sorted(summary, key=lambda r: (r["behavior"],
                                                 -_nan_key(r["response_corr"]))):
            w.writerow(r)


# ── HTML ──────────────────────────────────────────────────────────────────────

def _cell_color(val, lo, hi):
    """Blue (low) -> white -> red (high) for a value in [lo,hi]."""
    if val != val:  # nan
        return "#eee"
    t = 0.0 if hi == lo else (val - lo) / (hi - lo)
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        f = t / 0.5
        r, g, b = int(60 + 195 * f), int(120 + 135 * f), 255
    else:
        f = (t - 0.5) / 0.5
        r, g, b = 255, int(255 - 135 * f), int(255 - 195 * f)
    return f"rgb({r},{g},{b})"


def _corr_color(c):
    """Green for + (coeff up -> more behavior), red for - (wrong direction)."""
    if c != c:
        return "#eee"
    c = max(-1.0, min(1.0, c))
    if c >= 0:
        return f"rgb({int(255-155*c)},{int(255-40*c)},{int(255-155*c)})"  # -> green
    return f"rgb(255,{int(255+155*c)},{int(255+155*c)})"                  # -> red


def build_html(rows, summary, source_curve, source_stats, path, coherent_max=5.0):
    behaviors = sorted({r["behavior"] for r in rows})
    coeffs = sorted({r["coefficient"] for r in rows})

    # index: (scope,tr,nm,beh,coeff) -> (p,acc)
    idx = {}
    for r in rows:
        idx[(r["scope"], r.get("translator", ""), r.get("norm_mode", ""),
             r["behavior"], r["coefficient"])] = (r["avg_p_match"], r["accuracy"])

    def fmt(x):
        return f"{x:+.3f}" if x == x else "—"

    parts = ["""<title>A/B Steering Sweep — does coefficient ↑ raise P(behavior)?</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;color:#1a1a1a;background:#fafafa}
 h1{font-size:22px} h2{font-size:18px;margin-top:36px;border-bottom:2px solid #ddd;padding-bottom:4px}
 table{border-collapse:collapse;margin:10px 0;font-size:11px}
 th,td{border:1px solid #ccc;padding:3px 5px;text-align:center;white-space:nowrap}
 th{background:#f0f0f0;position:sticky;top:0}
 th.broken{background:#d9d9d9;color:#888}
 td.broken{opacity:.45}
 td.lbl{text-align:left;font-weight:600;background:#f7f7f7}
 .src td{font-weight:700;outline:2px solid #333}
 .wrap{overflow-x:auto;max-width:100%}
 .note{color:#444;font-size:13px;max-width:1000px;line-height:1.5}
 .best{box-shadow:inset 0 0 0 3px #c8a200}
 code{background:#eee;padding:1px 4px;border-radius:3px}
</style>
<h1>A/B Steering Sweep — does turning the coefficient up raise P(behavior)?</h1>
<p class="note">
<b>What "better" means here:</b> a good translated vector should behave like the original CAA
vector — pushing the coefficient <b>more positive raises P(behavior)</b> and more negative lowers it.
The headline metric is <b>response_corr</b> = Pearson correlation between the coefficient and
<b>P(behavior)</b> (avg&nbsp;P(match)). <b>+1</b> = coefficient↑ reliably raises the behavior (what we want);
<b>~0</b> = no monotonic response; <b>&lt;0</b> = coefficient↑ <i>lowers</i> the behavior (wrong direction).
<br><b>Coherent regime:</b> at large |coeff| the model collapses and P(behavior)→0 on
<i>both</i> sides, which destroys monotonicity and falsely inflates similarity (the dead tails
agree at ~0). For these Llama-3.2 layer-8 residual vectors the source still steers cleanly through
<code>|c|≤5</code> and is fully collapsed by <code>|c|=10</code>, so response_corr, dP and fidelity are computed
only over <code>|coeff| ≤ __CM__</code> (the collapsed columns are greyed in the heatmaps). Collapse onset
is behavior-dependent (roughly ±2 to ±5), so this cutoff is a heuristic — widen/tighten it with
<code>--coherent-max</code>.
<br><b>dP_coherent</b> = P(beh)@+max − P(beh)@−max within that window (signed effect size).
The bold-outlined <b>SOURCE</b> row is the original Llama-3.2-1B vector on the 1B model (the reference);
every other row is a translated vector on Llama-3.2-3B. Heatmap cells show <b>P(behavior)</b> (colored blue→red)
with accuracy below. Per behavior, the row with the highest response_corr is boxed in gold.
</p>
""".replace("__CM__", f"{coherent_max:g}")]

    # ── Summary table: ranked by response_corr, with a SOURCE reference row ──
    parts.append("<h2>Summary — coefficient→behavior response (per behavior)</h2>")
    parts.append('<div class="wrap"><table>')
    parts.append("<tr><th>behavior</th><th>type</th><th>loss</th><th>pooling</th>"
                 "<th>norm</th><th>layers</th><th>response_corr<br>(coeff→Pbeh)</th>"
                 "<th>dP coherent</th><th>P@−max</th><th>P@0</th><th>P@+max</th>"
                 "<th>monotonic</th><th>fidelity<br>vs src</th><th>|sv|</th></tr>")

    best_by_beh = {}
    for s in summary:
        v = s["response_corr"]
        if v == v and (s["behavior"] not in best_by_beh or v > best_by_beh[s["behavior"]]):
            best_by_beh[s["behavior"]] = v

    for beh in behaviors:
        # source reference rows first — one per source layer present
        src_layers = sorted({lay for (lay, b) in source_stats if b == beh})
        for lay in src_layers:
            ss = source_stats[(lay, beh)]
            parts.append(
                "<tr class='src'><td class='lbl'>%s</td><td colspan='4'>SOURCE "
                "(Llama-1B original)</td><td>l%s</td>"
                "<td style='background:%s'>%s</td><td>%s</td><td>%s</td><td>%s</td>"
                "<td>%s</td><td>%s</td><td>—</td><td>—</td></tr>" % (
                    html.escape(beh), lay,
                    _corr_color(ss.get("response_corr", float('nan'))),
                    fmt(ss.get("response_corr", float('nan'))),
                    fmt(ss.get("dP_coherent", float('nan'))),
                    fmt(ss.get("p_neg", float('nan'))), fmt(ss.get("p_at_0", float('nan'))),
                    fmt(ss.get("p_pos", float('nan'))),
                    fmt(ss.get("monotonic", float('nan')))))
        # target rows, best response first
        for s in sorted([x for x in summary if x["behavior"] == beh],
                        key=lambda x: -_nan_key(x["response_corr"])):
            cls = ' class="best"' if s["response_corr"] == best_by_beh.get(beh, 99) else ""
            parts.append(
                f"<tr{cls}><td class='lbl'></td>"
                f"<td>{s['translator_type']}</td><td>{s['loss']}</td>"
                f"<td>{s['pooling']}</td><td>{s['norm_mode']}</td>"
                f"<td>l{s['source_layer']}→l{s['target_layer']}</td>"
                f"<td style='background:{_corr_color(s['response_corr'])}'>"
                f"{fmt(s['response_corr'])}</td>"
                f"<td>{fmt(s['dP_coherent'])}</td><td>{fmt(s['p_neg'])}</td>"
                f"<td>{fmt(s['p_at_0'])}</td><td>{fmt(s['p_pos'])}</td>"
                f"<td>{fmt(s['monotonic'])}</td>"
                f"<td>{fmt(s['fidelity_vs_source'])}</td>"
                f"<td>{s['sv_norm']:.3f}</td></tr>")
    parts.append("</table></div>")

    # ── per-behavior coefficient heatmaps (cells = P(behavior)) ──
    for beh in behaviors:
        parts.append(f"<h2>{html.escape(beh)}</h2>")
        parts.append('<div class="wrap"><table>')
        parts.append("<tr><th>config</th>" +
                     "".join(f"<th class='{'broken' if abs(c) > coherent_max else ''}'>"
                             f"{c:g}</th>" for c in coeffs) + "</tr>")

        def row_html(label, getter, is_src=False):
            tds = []
            for c in coeffs:
                v = getter(c)
                bk = " broken" if abs(c) > coherent_max else ""
                if v is None:
                    tds.append(f"<td class='{bk.strip()}' style='background:#eee'>—</td>")
                else:
                    p, acc = v
                    tds.append(f"<td class='{bk.strip()}' style='background:{_cell_color(p,0,1)}'>"
                               f"{p:.2f}<br><span style='color:#555'>{acc:.2f}</span></td>")
            cls = ' class="src"' if is_src else ""
            return f"<tr{cls}><td class='lbl'>{html.escape(label)}</td>" + "".join(tds) + "</tr>"

        # one SOURCE row per source layer present (non-default layers carry a
        # "source_l{n}" label in the translator field; the default layer is "")
        src_labels = sorted({(r.get("source_layer", 8), r.get("translator", ""))
                             for r in rows
                             if r["scope"] == "source" and r["behavior"] == beh})
        for lay, slbl in src_labels:
            parts.append(row_html(
                f"SOURCE (Llama-1B l{lay}, original)",
                lambda c, slbl=slbl: idx.get(("source", slbl, "", beh, c)),
                is_src=True))

        tgt_keys = sorted({(r.get("translator_type", ""), r.get("loss", ""),
                            r.get("pooling", ""), r.get("norm_mode", ""),
                            r.get("source_layer", 8), r.get("target_layer", 8),
                            r.get("translator", ""))
                           for r in rows
                           if r["scope"] == "target" and r["behavior"] == beh})
        for ttype, loss, pooling, nm, slay, tlay, tr in tgt_keys:
            label = f"{ttype}/{loss}/{pooling}/{nm} l{slay}→l{tlay}"
            parts.append(row_html(
                label,
                lambda c, tr=tr, nm=nm: idx.get(("target", tr, nm, beh, c))))
        parts.append("</table></div>")

    Path(path).write_text("\n".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+",
                    default=[str(DEFAULT_OUT / "*.jsonl")],
                    help="JSONL result file glob(s) to merge")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--coherent-max", type=float, default=5.0,
                    help="Max |coeff| treated as coherent (beyond it the model "
                         "collapses); response/fidelity computed within this window. "
                         "Default 5: for these Llama-3.2 layer-8 residual vectors the "
                         "source steers cleanly through |c|<=5 and is fully collapsed "
                         "(P->0 both sides) by |c|=10. Collapse onset is behavior-"
                         "dependent (~+-2 to +-5), so treat this as a heuristic.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results)
    if not rows:
        raise SystemExit(f"No result rows found in {args.results}")
    print(f"  Loaded {len(rows)} rows from {args.results}")

    write_results_csv(rows, out / "results.csv")
    summary, source_curve, source_stats = build_summary(rows, args.coherent_max)
    write_summary_csv(summary, out / "summary.csv")
    build_html(rows, summary, source_curve, source_stats, out / "report.html",
               args.coherent_max)

    print(f"  Wrote:\n    {out/'results.csv'}\n    {out/'summary.csv'}\n    {out/'report.html'}")


if __name__ == "__main__":
    main()
