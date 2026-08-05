"""
Cross-method report from ab_sweep.py results: CAA vs RepE vs GCAV.

Where ab_report.py compares translators *within* one discovery method on the raw
coefficient axis, this compares the METHODS to each other. It cannot use raw
coefficients: CAA vectors carry their own behavior-dependent magnitude while RepE
and GCAV are unit-norm, so equal coefficient is not equal perturbation. Each run
is instead read at its own best dose (see acttrans.evaluation.method_compare):

  dose    = coefficient * sv_norm          the real perturbation size
  dP_peak = max over the symmetric grid of P(match|+c) - P(match|-c)

Outputs (under --out-dir):
  cross_method.csv     one row per (method, scope, translator, behavior)
  method_summary.csv   one row per (method, scope) — the headline table
  best_per_method.csv  best translator per (method, behavior) in the target scope
  method_report.html   the same, with grouped-bar charts per scope

Usage:
    conda run -n acteng python method_report.py
    conda run -n acteng python method_report.py \
        --results 'outputs/ab_eval/methods/*.jsonl' \
        --out-dir outputs/ab_eval/methods
"""

import argparse
import csv
import glob
import html
import json
from pathlib import Path

from acttrans.constants import METHODS
from acttrans.evaluation.method_compare import (
    analyze,
    best_per_method,
    normalize_method,
    summarize,
)

_HERE = Path(__file__).resolve().parent
DEFAULT_OUT = _HERE / "outputs" / "ab_eval" / "methods"

# Categorical slots 1-3 of the validated default palette (blue / orange / aqua).
# These three validate on the all-pairs list in both light and dark modes; the
# dark column is the same hues re-stepped for the dark surface, not a flip.
# Aqua is below 3:1 on the light surface, so the relief rule applies — bars carry
# visible direct labels and the page ships the full table view.
SERIES = {
    "CAA":  {"light": "#2a78d6", "dark": "#3987e5"},
    "RepE": {"light": "#eb6834", "dark": "#d95926"},
    "GCAV": {"light": "#1baf7a", "dark": "#199e70"},
}
_FALLBACK = {"light": "#4a3aa7", "dark": "#9085e9"}

SCOPE_TITLE = {
    "source": "Source model (Llama-1B, original vector)",
    "target": "Target model (Llama-3B, translated vector)",
    "native": "Target model (Llama-3B, natively extracted vector)",
}


def load_rows(patterns):
    rows, seen = [], set()
    for pat in patterns:
        for fp in glob.glob(pat):
            for line in Path(fp).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["method"] = normalize_method(r)
                k = (r["method"], r.get("scope"), r.get("translator"),
                     r.get("norm_mode"), r.get("behavior"), r.get("coefficient"))
                if k in seen:
                    continue
                seen.add(k)
                rows.append(r)
    return rows


def write_csv(path, rows, cols=None):
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"    {path} ({len(rows)} rows)")


# ── SVG grouped bar chart ────────────────────────────────────────────────────

def _bar_chart(blocks, methods, behaviors, scope):
    """Grouped bars: dP_peak per behavior, one bar per method.

    Magnitude comparison across a small set of categories, so bars — not lines.
    Each bar is the best block for that (method, behavior) in this scope, i.e.
    the most steering the method achieved at any dose through any translator.
    """
    best = {}
    for r in blocks:
        if r["scope"] != scope:
            continue
        k = (r["behavior"], r["method"])
        if r["dP_peak"] != r["dP_peak"]:
            continue
        if k not in best or r["dP_peak"] > best[k]["dP_peak"]:
            best[k] = r
    if not best:
        return ""

    vals = [v["dP_peak"] for v in best.values()]
    hi = max(0.0, max(vals))
    lo = min(0.0, min(vals))
    span = (hi - lo) or 1.0
    # pad the top so direct labels never clip
    hi_pad = hi + span * 0.14
    span = (hi_pad - lo) or 1.0

    W, H = 900, 300
    ml, mr, mt, mb = 52, 12, 14, 62
    pw, ph = W - ml - mr, H - mt - mb

    def y_of(v):
        return mt + ph * (hi_pad - v) / span

    y0 = y_of(0.0)
    group_w = pw / max(1, len(behaviors))
    # 2px surface gap between adjacent bars within a group
    bar_w = max(6.0, (group_w * 0.68) / max(1, len(methods)) - 2)

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="Peak behavior swing by behavior and method, {scope} scope">']

    # recessive gridlines + y ticks
    n_ticks = 5
    for i in range(n_ticks + 1):
        v = lo + (hi_pad - lo) * i / n_ticks
        y = y_of(v)
        parts.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{ml - 8}" y="{y + 3.5:.1f}" '
                     f'text-anchor="end">{v:+.2f}</text>')
    # zero baseline sits above the grid
    parts.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{y0:.1f}" y2="{y0:.1f}"/>')

    for bi, beh in enumerate(behaviors):
        gx = ml + group_w * bi
        present = [m for m in methods if (beh, m) in best]
        total_w = len(present) * bar_w + max(0, len(present) - 1) * 2
        x = gx + (group_w - total_w) / 2
        for m in present:
            r = best[(beh, m)]
            v = r["dP_peak"]
            top = y_of(max(v, 0.0))
            h = abs(y_of(v) - y0)
            # 4px rounded data-end, square against the baseline
            rx = min(4.0, bar_w / 2)
            up = v >= 0
            yy = top if up else y0
            parts.append(
                f'<rect class="bar s-{m}" x="{x:.1f}" y="{yy:.1f}" '
                f'width="{bar_w:.1f}" height="{max(h, 1.0):.1f}" rx="{rx:.1f}" '
                f'data-m="{html.escape(m)}" data-b="{html.escape(beh)}" '
                f'data-v="{v:.4f}" data-dose="{r["dose_at_peak"]:.4g}" '
                f'data-coeff="{r["coeff_at_peak"]:g}" '
                f'data-tr="{html.escape(str(r["translator"]))}"><title>'
                f'{html.escape(m)} · {html.escape(beh)}\ndP_peak {v:+.3f}\n'
                f'dose {r["dose_at_peak"]:.3g} (coeff {r["coeff_at_peak"]:g})\n'
                f'{html.escape(str(r["translator"]) or "—")}</title></rect>'
            )
            # direct label on every bar (required relief for the light-mode aqua slot)
            ly = (top - 5) if up else (y0 + h + 12)
            parts.append(f'<text class="blab" x="{x + bar_w / 2:.1f}" y="{ly:.1f}" '
                         f'text-anchor="middle">{v:+.2f}</text>')
            x += bar_w + 2
        short = beh.replace("-", "‑")
        parts.append(f'<text class="xlab" x="{gx + group_w / 2:.1f}" y="{H - mb + 22}" '
                     f'text-anchor="end" transform="rotate(-32 '
                     f'{gx + group_w / 2:.1f} {H - mb + 22})">{html.escape(short)}</text>')

    parts.append(f'<text class="ylab" transform="rotate(-90 14 {mt + ph / 2:.1f})" '
                 f'x="14" y="{mt + ph / 2:.1f}" text-anchor="middle">dP_peak</text>')
    parts.append("</svg>")
    return "".join(parts)


def _legend(methods):
    items = "".join(
        f'<span class="lg"><i class="sw s-{m}"></i>{html.escape(m)}</span>'
        for m in methods
    )
    return f'<div class="legend">{items}</div>'


def _table(rows, cols, headers=None, cls="tbl"):
    if not rows:
        return "<p class='muted'>no rows</p>"
    headers = headers or cols
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = []
    for r in rows:
        tds = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                txt = "—" if v != v else (f"{v:+.3f}" if abs(v) < 1000 else f"{v:.3g}")
            else:
                txt = "—" if v is None or v == "" else str(v)
            meth_cls = f' class="s-{v}-ink"' if c == "method" and v in SERIES else ""
            tds.append(f"<td{meth_cls}>{html.escape(txt)}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<div class="scroll"><table class="{cls}"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


_CSS = """
:root { color-scheme: light; }
.viz-root {
  --surface-1:#fcfcfb; --surface-2:#f3f3f1; --border:#e2e1dc;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#7a7873;
  --s-CAA:#2a78d6; --s-RepE:#eb6834; --s-GCAV:#1baf7a; --s-other:#4a3aa7;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3a37;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#95948b;
    --s-CAA:#3987e5; --s-RepE:#d95926; --s-GCAV:#199e70; --s-other:#9085e9;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#242423; --border:#3a3a37;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#95948b;
  --s-CAA:#3987e5; --s-RepE:#d95926; --s-GCAV:#199e70; --s-other:#9085e9;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--surface-1); color:var(--text-primary);
  font:14px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
.wrap { max-width:1120px; margin:0 auto; padding:34px 22px 70px; }
h1,h2,h3 { font-family:Charter,Georgia,serif; font-weight:600; letter-spacing:-.01em; }
h1 { font-size:27px; margin:0 0 6px; }
h2 { font-size:20px; margin:38px 0 4px; padding-top:16px; border-top:1px solid var(--border); }
h3 { font-size:15px; margin:22px 0 6px; }
p { color:var(--text-secondary); margin:6px 0 12px; }
.sub { color:var(--text-muted); font-size:13px; }
code { background:var(--surface-2); padding:1px 5px; border-radius:4px; font-size:12.5px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:20px 0 4px; }
.tile { background:var(--surface-2); border:1px solid var(--border); border-radius:10px; padding:13px 15px; }
.tile .k { font-size:12px; color:var(--text-muted); display:flex; align-items:center; gap:6px; }
.tile .v { font-size:24px; font-family:Charter,Georgia,serif; margin-top:3px; }
.tile .d { font-size:12px; color:var(--text-muted); }
.note { background:var(--surface-2); border:1px solid var(--border);
  border-left:3px solid var(--s-RepE); border-radius:8px; padding:12px 15px; margin:16px 0; }
.note p { margin:0; color:var(--text-secondary); font-size:13px; }
.chart { width:100%; height:auto; display:block; margin:6px 0 2px; }
.grid { stroke:var(--border); stroke-width:1; }
.axis { stroke:var(--text-muted); stroke-width:1.5; }
.tick,.xlab,.ylab,.blab { fill:var(--text-muted); font-size:11px;
  font-family:system-ui,sans-serif; }
.blab { fill:var(--text-secondary); font-size:10.5px; }
.xlab { font-size:11.5px; }
.bar { stroke:var(--surface-1); stroke-width:2; paint-order:stroke; }
.bar:hover { opacity:.82; }
.s-CAA { fill:var(--s-CAA); } .s-RepE { fill:var(--s-RepE); } .s-GCAV { fill:var(--s-GCAV); }
.s-CAA-ink { color:var(--s-CAA); } .s-RepE-ink { color:var(--s-RepE); }
.s-GCAV-ink { color:var(--s-GCAV); }
.legend { display:flex; flex-wrap:wrap; gap:14px; margin:2px 0 10px; font-size:12.5px;
  color:var(--text-secondary); }
.lg { display:inline-flex; align-items:center; gap:6px; }
.sw { width:11px; height:11px; border-radius:3px; display:inline-block; }
.sw.s-CAA { background:var(--s-CAA); } .sw.s-RepE { background:var(--s-RepE); }
.sw.s-GCAV { background:var(--s-GCAV); }
.scroll { overflow-x:auto; border:1px solid var(--border); border-radius:9px; margin:10px 0 4px; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th,td { padding:6px 10px; text-align:right; white-space:nowrap;
  border-bottom:1px solid var(--border); }
th { background:var(--surface-2); color:var(--text-secondary); font-weight:600;
  position:sticky; top:0; text-align:right; }
td:first-child,th:first-child,td:nth-child(2),th:nth-child(2) { text-align:left; }
tbody tr:last-child td { border-bottom:none; }
tbody tr:hover { background:var(--surface-2); }
.muted { color:var(--text-muted); }
"""


def build_html(summary, blocks, best, methods, behaviors, path, sources):
    tiles = []
    tgt = {r["method"]: r for r in summary if r["scope"] == "target"}
    nat = {r["method"]: r for r in summary if r["scope"] == "native"}
    for m in methods:
        t = tgt.get(m)
        if not t:
            continue
        ret = t["mean_retention_vs_native"]
        n = nat.get(m)
        tiles.append(
            f'<div class="tile"><div class="k"><i class="sw s-{m}"></i>{html.escape(m)}'
            f' · translated</div>'
            f'<div class="v">{t["mean_dP_peak"]:+.3f}</div>'
            f'<div class="d">mean dP_peak over {t["n_blocks"]} blocks'
            + (f' · native {n["mean_dP_peak"]:+.3f}' if n else "")
            + (f' · retention {ret:.0%}' if ret == ret else "")
            + '</div></div>'
        )

    scope_sections = []
    for scope in ("target", "native", "source"):
        rows = [r for r in blocks if r["scope"] == scope]
        if not rows:
            continue
        chart = _bar_chart(blocks, methods, behaviors, scope)
        scope_sections.append(
            f'<h3>{html.escape(SCOPE_TITLE.get(scope, scope))}</h3>'
            + _legend(methods) + chart
            + '<p class="sub">Each bar is the best (method, behavior) block in this '
              'scope — the largest behavior swing that method reached at any dose '
              'through any translator. Hover a bar for its dose and translator.</p>'
        )

    sum_cols = ["method", "scope", "mean_dP_peak", "median_dP_peak", "best_dP_peak",
                "mean_acc_swing", "mean_monotonic_to_peak", "median_dose_at_peak",
                "median_coeff_at_peak", "mean_sv_norm", "mean_retention_vs_native",
                "n_retention_refs", "n_no_native_ref",
                "n_blocks", "n_behaviors", "n_translators"]
    best_cols = ["method", "behavior", "dP_peak", "acc_swing_at_peak", "dose_at_peak",
                 "coeff_at_peak", "sv_norm", "monotonic_to_peak", "translator_type",
                 "loss", "norm_mode", "translator"]
    blk_cols = ["method", "scope", "behavior", "dP_peak", "acc_swing_at_peak",
                "dose_at_peak", "coeff_at_peak", "sv_norm", "monotonic_to_peak",
                "p_at_0", "source_layer", "target_layer", "translator"]

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cross-method steering comparison — CAA / RepE / GCAV</title>
<style>{_CSS}</style></head>
<body class="viz-root"><div class="wrap">
<h1>Cross-method steering comparison</h1>
<p>How well do translated steering vectors work, per discovery method?
Read from <code>{html.escape(", ".join(sources))}</code>.</p>

<div class="note"><p><strong>Why not raw coefficients.</strong> CAA vectors carry
their own behavior-dependent magnitude (|sv| ranges roughly 0.2&ndash;5) while RepE
and GCAV are unit-norm by construction, so the same <code>coefficient</code> is a
different physical perturbation per method. Every run is therefore read at its own
best dose: <code>dose = coefficient &times; sv_norm</code> and
<code>dP_peak = max<sub>c</sub>[ P(match|+c) &minus; P(match|&minus;c) ]</code>.
Because a collapsed model has P(match)&nbsp;&rarr;&nbsp;0 on both sides, collapsed
doses contribute ~0 swing and cannot win the peak search &mdash; no coherence
cutoff is needed or applied.</p></div>

<div class="tiles">{"".join(tiles)}</div>

<h2>Headline table — per method and scope</h2>
<p><code>mean_retention_vs_native</code> is the share of the natively-extracted
target vector's swing that this scope recovers, matched per behavior and target
layer; native rows are 1.0 by construction and source rows are blank (different
model). Behaviors whose native vector barely steers are excluded from the average
&mdash; dividing by a near-zero reference manufactures huge ratios &mdash; and
counted in <code>n_no_native_ref</code>, so weak coverage is visible rather than
hidden.</p>
{_table(summary, sum_cols)}

<h2>Peak swing by behavior</h2>
{"".join(scope_sections)}

<h2>Best translator per method and behavior (translated scope)</h2>
{_table(best, best_cols)}

<h2>All blocks</h2>
<p class="sub">One row per (method, scope, translator, behavior).</p>
{_table(blocks, blk_cols)}
</div></body></html>"""
    Path(path).write_text(doc)
    print(f"    {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+",
                    default=[str(DEFAULT_OUT / "*.jsonl")],
                    help="JSONL result file glob(s) to merge")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--methods", nargs="+", default=None, choices=list(METHODS),
                    metavar="METHOD",
                    help=f"Restrict to these methods (default: all present). "
                         f"Choices: {', '.join(METHODS)}")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.results)
    if not rows:
        raise SystemExit(f"No result rows found in {args.results}")

    blocks = analyze(rows, methods=args.methods)
    summary = summarize(blocks)
    best = best_per_method(blocks, scope="target")

    present = [m for m in METHODS if any(b["method"] == m for b in blocks)]
    extra = sorted({b["method"] for b in blocks} - set(present))
    methods = present + extra
    behaviors = sorted({b["behavior"] for b in blocks})

    print(f"  {len(rows)} rows -> {len(blocks)} blocks  "
          f"methods={methods}  behaviors={len(behaviors)}")
    print("  Wrote:")
    write_csv(out / "cross_method.csv", blocks)
    write_csv(out / "method_summary.csv", summary)
    write_csv(out / "best_per_method.csv", best)
    build_html(summary, blocks, best, methods, behaviors,
               out / "method_report.html", args.results)

    # console headline
    print("\n  Peak behavior swing (dP_peak), mean over blocks:")
    print(f"    {'method':6s} {'scope':8s} {'mean':>8s} {'best':>8s} "
          f"{'retention':>10s} {'med dose':>9s}")
    for r in summary:
        ret = r["mean_retention_vs_native"]
        print(f"    {r['method']:6s} {r['scope']:8s} {r['mean_dP_peak']:+8.3f} "
              f"{r['best_dP_peak']:+8.3f} "
              f"{(f'{ret:9.1%}' if ret == ret else '        —')} "
              f"{r['median_dose_at_peak']:9.3g}")


if __name__ == "__main__":
    main()
