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


def load_rows(patterns, methods=None):
    """Load result rows, de-duplicated across shards / re-runs.

    Rows predating the method dimension carry no `method` field and are all CAA,
    so a missing method is normalized to "CAA" — that keeps the de-dup key stable
    for old files and lets `methods` filter them.

    This report is single-method by construction: its heatmaps and effect sizes
    are indexed by (scope, translator, norm, behavior), and mixing methods into
    one page would silently overlay curves whose coefficients mean different
    physical doses. Pass `methods` to select one; cross-method comparison lives
    in method_report.py."""
    rows = []
    seen = set()
    for pat in patterns:
        for fp in glob.glob(pat):
            for line in Path(fp).read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["method"] = r.get("method") or "CAA"
                if methods and r["method"] not in methods:
                    continue
                k = (r["method"], r.get("scope"), r.get("translator"),
                     r.get("norm_mode"), r.get("behavior"), r.get("coefficient"))
                if k in seen:
                    continue
                seen.add(k)
                rows.append(r)
    return rows


def write_results_csv(rows, path):
    cols = ["method", "scope", "translator", "translator_type", "loss", "pooling",
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
    # index by block; the method is part of the block identity so a results file
    # holding several methods does not merge their curves
    blocks = defaultdict(list)
    for r in rows:
        blocks[(r.get("method") or "CAA", r["scope"], r.get("translator", ""),
                r.get("norm_mode", ""), r["behavior"])].append(r)

    # Source curves + their own response metrics, keyed by (method, source layer,
    # behavior): with layer-pair sweeps there is one baseline per source layer,
    # and each translator is judged against the baseline of ITS method and source
    # layer. Rows predating layer support carry no source_layer -> default 8.
    source_curve = {}
    source_stats = {}
    for (meth, scope, tr, nm, beh), rs in blocks.items():
        if scope == "source":
            lay = rs[0].get("source_layer", 8)
            source_curve[(meth, lay, beh)] = _curve(rs)
            source_stats[(meth, lay, beh)] = _response_metrics(
                source_curve[(meth, lay, beh)], coherent_max)

    summary = []
    for (meth, scope, tr, nm, beh), rs in blocks.items():
        if scope != "target":
            continue
        cur = _curve(rs)
        m = _response_metrics(cur, coherent_max)

        # fidelity vs source: corr of the two P(behavior) curves, but ONLY over
        # the coherent window — beyond it both collapse to ~0 and spuriously
        # "agree", which inflates the score.
        fid = float("nan")
        src_lay = rs[0].get("source_layer", 8)
        sc = source_curve.get((meth, src_lay, beh))
        if sc:
            shared = sorted(c for c in (set(cur) & set(sc)) if abs(c) <= coherent_max)
            if len(shared) >= 2:
                fid = _pearson([cur[c][0] for c in shared],
                               [sc[c][0] for c in shared])

        ex = rs[0]
        summary.append({
            "method": meth,
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
    cols = ["method", "translator_type", "loss", "pooling", "norm_mode", "behavior",
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
# The report is a single self-contained page in the same visual family as
# outputs/{layer_sweep,comparison}/report.html: system-ui body + Charter serif
# headings, CSS-variable palette that themes light/dark (media query + a
# data-theme toggle scope), stat tiles, metric-reminder cards, SVG charts drawn
# in JS, and a shared hover tooltip. Colors follow the dataviz skill's validated
# default palette. build_html stays fully generic: it discovers behaviors,
# coefficients, source baselines and every (translator × norm × layer) config
# from the rows, so it renders identically for the single l8→l12 case and for
# the ~24-translator fineweb sweep.

# dataviz default palette (validated) — categorical slots + a diverging P scale.
_CSS = """
  *, *::before, *::after { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
    --border: rgba(11, 11, 11, 0.10); --accent: #2a78d6;
    --cat-1: #2a78d6; --cat-2: #eb6834; --cat-3: #1baf7a; --cat-4: #eda100;
    --cat-5: #e87ba4; --cat-6: #008300; --cat-7: #4a3aa7; --cat-8: #e34948;
    --c-src: #0b0b0b;
    --cell-lo: #256abf; --cell-mid: #f0efec; --cell-hi: #d03b3b;
    --stat-good: #0ca30c; --stat-bad: #d03b3b;
    --best: #008300;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --axis: #383835;
      --border: rgba(255, 255, 255, 0.10); --accent: #3987e5;
      --cat-1: #3987e5; --cat-2: #d95926; --cat-3: #199e70; --cat-4: #c98500;
      --cat-5: #d55181; --cat-6: #008300; --cat-7: #9085e9; --cat-8: #e66767;
      --c-src: #ffffff;
      --cell-lo: #3987e5; --cell-mid: #383835; --cell-hi: #e66767;
      --stat-good: #0ca30c; --stat-bad: #d03b3b;
      --best: #23a559;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --border: rgba(255, 255, 255, 0.10); --accent: #3987e5;
    --cat-1: #3987e5; --cat-2: #d95926; --cat-3: #199e70; --cat-4: #c98500;
    --cat-5: #d55181; --cat-6: #008300; --cat-7: #9085e9; --cat-8: #e66767;
    --c-src: #ffffff;
    --cell-lo: #3987e5; --cell-mid: #383835; --cell-hi: #e66767;
    --stat-good: #0ca30c; --stat-bad: #d03b3b;
    --best: #23a559;
  }
  html { background: var(--page); }
  body { margin: 0; background: var(--page); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 15px; line-height: 1.55; }
  main { max-width: 1080px; margin: 0 auto; padding: 48px 24px 96px; }
  h1, h2 { font-family: Charter, "Bitstream Charter", "Iowan Old Style", Georgia, serif;
    font-weight: 600; text-wrap: balance; margin: 0; }
  h1 { font-size: 2.2rem; line-height: 1.15; letter-spacing: -0.01em; }
  h2 { font-size: 1.4rem; margin: 0 0 4px; }
  section { margin-top: 60px; }
  p { max-width: 72ch; }
  code, .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 0.92em; }
  code { background: color-mix(in srgb, var(--ink) 8%, transparent); padding: 1px 5px; border-radius: 4px; }
  a { color: var(--accent); }
  .eyebrow { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.72rem;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin: 0 0 14px; }
  .lede { color: var(--ink-2); font-size: 1.05rem; margin: 14px 0 0; }
  .section-note { color: var(--ink-2); margin: 6px 0 20px; }

  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 34px 0 0; }
  .tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px 14px; }
  .tile .label { font-size: 0.82rem; color: var(--ink-2); margin: 0; }
  .tile .value { font-size: 1.9rem; font-weight: 600; margin: 2px 0 0; line-height: 1.2; word-break: break-word; }
  .tile .value small { font-size: 0.95rem; font-weight: 500; color: var(--muted); }
  .tile .context { font-size: 0.78rem; color: var(--muted); margin: 4px 0 0; }

  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; margin-top: 20px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; }
  .card h3 { font-size: 0.95rem; margin: 0 0 6px; font-family: ui-monospace, Menlo, Consolas, monospace; font-weight: 600; }
  .card p { font-size: 0.88rem; color: var(--ink-2); margin: 0; max-width: none; }

  figure { margin: 24px 0 0; background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px 20px 16px; }
  figure .fig-title { font-weight: 600; margin: 0; font-size: 1rem; }
  figure .fig-sub { color: var(--muted); font-size: 0.85rem; margin: 2px 0 14px; }
  .chart-scroll { overflow-x: auto; }
  svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 12px; padding: 0;
    list-style: none; font-size: 0.82rem; color: var(--ink-2); }
  .legend li { display: flex; align-items: center; gap: 7px; }
  .legend .swatch { width: 11px; height: 11px; border-radius: 3px; display: inline-block; flex: none; }
  .legend .swatch.dash { width: 18px; height: 0; border-radius: 0; background: none !important;
    border-top: 2.5px dashed currentColor; }

  #tooltip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink);
    border: 1px solid var(--border); border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    padding: 9px 12px; font-size: 0.8rem; line-height: 1.45; max-width: 320px; opacity: 0; z-index: 10; }
  #tooltip .tt-title { font-weight: 600; }
  #tooltip .tt-row { display: flex; justify-content: space-between; gap: 16px; }
  #tooltip .tt-row span:last-child { font-family: ui-monospace, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
  @media (prefers-reduced-motion: no-preference) { #tooltip { transition: opacity 120ms ease; } }

  .table-scroll { overflow-x: auto; margin-top: 12px; }
  table.sum { border-collapse: collapse; font-size: 0.8rem; min-width: 720px; background: var(--surface); width: 100%; }
  table.sum th, table.sum td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  table.sum th { color: var(--ink-2); font-weight: 600; font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.72rem; position: sticky; top: 0; background: var(--surface); z-index: 2; }
  table.sum td { font-variant-numeric: tabular-nums; color: var(--ink); }
  table.sum td.lbl { text-align: left; font-weight: 600; font-family: ui-monospace, Menlo, Consolas, monospace; }
  table.sum td.cfg { text-align: left; font-family: ui-monospace, Menlo, Consolas, monospace; color: var(--ink-2); }
  table.sum tr.src td { font-weight: 700; }
  table.sum tr.src td.lbl { color: var(--ink); }
  table.sum tr.src { border-top: 2px solid var(--axis); }
  table.sum tr.best td.cfg::after { content: " ★"; color: var(--best); }
  table.sum tr.best td { background: color-mix(in srgb, var(--best) 12%, transparent); }
  .corrcell { font-weight: 600; border-radius: 3px; }

  /* heatmap */
  .hm-legend { display: flex; align-items: center; gap: 10px; font-size: 0.78rem; color: var(--ink-2); margin: 0 0 6px; }
  .hm-legend .bar { width: 150px; height: 12px; border-radius: 3px; border: 1px solid var(--border);
    background: linear-gradient(90deg, var(--cell-lo), var(--cell-mid), var(--cell-hi)); }
  table.hm { border-collapse: separate; border-spacing: 2px; font-size: 0.72rem; background: var(--surface); }
  table.hm th, table.hm td { padding: 4px 6px; text-align: center; white-space: nowrap; }
  table.hm th { color: var(--ink-2); font-weight: 600; font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.68rem; position: sticky; top: 0; background: var(--surface); z-index: 2; }
  table.hm th.lbl, table.hm td.lbl { position: sticky; left: 0; z-index: 3; background: var(--surface);
    text-align: left; font-family: ui-monospace, Menlo, Consolas, monospace; color: var(--ink-2);
    box-shadow: 1px 0 0 var(--grid); }
  table.hm th.lbl { z-index: 4; }
  table.hm tr.src td.lbl, table.hm tr.src th.lbl { color: var(--ink); font-weight: 700; }
  table.hm td.cell { border-radius: 3px; font-variant-numeric: tabular-nums; color: var(--ink-2);
    background: var(--cell-mid); min-width: 40px; }
  table.hm td.cell .acc { display: block; font-size: 0.9em; opacity: 0.75; }
  table.hm .broken { opacity: 0.42; }
  table.hm th.broken { color: var(--muted); }
  table.hm th.edge, table.hm td.edge { box-shadow: inset 2px 0 0 var(--axis); }

  footer { margin-top: 72px; color: var(--muted); font-size: 0.8rem; }
"""

# JS body (everything after the injected `const DATA = …;`). Kept static; all
# dynamic values arrive through DATA so no server-side string interpolation
# leaks into the script. Only benign SVG namespace URL is referenced.
_JS = """
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS = "http://www.w3.org/2000/svg";
const isNil = (x) => x === null || x === undefined || Number.isNaN(x);
const fmt = (x, d = 3) => isNil(x) ? "\\u2014" : (x >= 0 ? "+" : "") + x.toFixed(d);
const fmt2 = (x) => isNil(x) ? "\\u2014" : x.toFixed(2);
const gfmt = (x) => {
  const a = Math.abs(x);
  if (a >= 1000) return (x / 1000) + "k";
  return String(x);
};

// ── tooltip ──
const tip = document.getElementById("tooltip");
function showTip(e, html) { tip.innerHTML = html; tip.style.opacity = 1; moveTip(e); }
function moveTip(e) {
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.style.opacity = 0; }
const row = (k, v) => `<div class="tt-row"><span>${k}</span><span>${v}</span></div>`;

// ── color helpers ──
function hexToRgb(h) {
  h = h.trim();
  if (h.startsWith("rgb")) return h.match(/\\d+/g).map(Number);
  const s = h.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16));
}
function mix(h1, h2, t) {
  const a = hexToRgb(h1), b = hexToRgb(h2);
  return "rgb(" + a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",") + ")";
}
function inkFor(bg) {
  const m = bg.match(/\\d+/g).map(Number);
  const lum = (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) / 255;
  return lum > 0.6 ? "#0b0b0b" : "#ffffff";
}
// P(behavior) in [0,1] -> diverging blue(low) / gray(0.5) / red(high)
function pColor(p) {
  const t = Math.max(-1, Math.min(1, (p - 0.5) / 0.5));
  return t >= 0 ? mix(css("--cell-mid"), css("--cell-hi"), t)
                : mix(css("--cell-mid"), css("--cell-lo"), -t);
}
// correlation / fidelity in [-1,1] -> gray / green(+) / red(-)
function corrColor(v) {
  const t = Math.max(-1, Math.min(1, v));
  return t >= 0 ? mix(css("--cell-mid"), css("--stat-good"), t)
                : mix(css("--cell-mid"), css("--stat-bad"), -t);
}
function catColor(cid) {
  return (cid >= 0 && cid < 8) ? css("--cat-" + (cid + 1)) : null;
}

// ── svg helpers ──
function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function svgRoot(container, w, h) {
  container.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${w} ${h}`, width: "100%", role: "img" });
  svg.style.minWidth = Math.min(w, 520) + "px";
  svg.style.display = "block";
  container.appendChild(svg);
  return svg;
}
function text(svg, x, y, s, a = {}) {
  const t = el("text", { x, y, fill: a.fill || css("--ink-2"), "font-size": a.size || 11,
    "text-anchor": a.anchor || "start", "dominant-baseline": a.baseline || "middle",
    ...(a.extra || {}) }, svg);
  t.textContent = s; return t;
}

// ── per-behavior line chart: P(behavior) vs coefficient over the coherent window ──
function lineChart(idx, beh) {
  const series = DATA.charts[beh] || [];
  const container = document.getElementById("chart-" + idx);
  if (!container) return;
  const cm = DATA.coherentMax;
  const xs = DATA.coeffs.filter((c) => Math.abs(c) <= cm);
  if (!xs.length) { container.innerHTML = ""; return; }
  const M = { l: 44, r: 16, t: 14, b: 40 }, W = 720, H = 300;
  const plotW = W - M.l - M.r, plotH = H - M.t - M.b;
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = 0, y1 = 1;
  const X = (v) => M.l + (x1 === x0 ? 0.5 : (v - x0) / (x1 - x0)) * plotW;
  const Y = (v) => M.t + plotH - (v - y0) / (y1 - y0) * plotH;
  const svg = svgRoot(container, W, H);

  // y grid + ticks
  for (let v = 0; v <= 1.0001; v += 0.25) {
    el("line", { x1: M.l, x2: M.l + plotW, y1: Y(v), y2: Y(v),
      stroke: css("--grid"), "stroke-width": 1 }, svg);
    text(svg, M.l - 8, Y(v), v.toFixed(2), { anchor: "end", size: 10, fill: css("--muted"),
      extra: { "font-variant-numeric": "tabular-nums" } });
  }
  // neutral P=0.5 reference
  el("line", { x1: M.l, x2: M.l + plotW, y1: Y(0.5), y2: Y(0.5),
    stroke: css("--axis"), "stroke-width": 1, "stroke-dasharray": "3 4" }, svg);
  // x ticks
  xs.forEach((c) => text(svg, X(c), M.t + plotH + 16, gfmt(c),
    { anchor: "middle", size: 10, fill: css("--muted"), extra: { "font-variant-numeric": "tabular-nums" } }));
  text(svg, M.l + plotW / 2, H - 3, "steering coefficient", { anchor: "middle", size: 10.5, fill: css("--ink-2") });

  const legendItems = [];
  let otherCount = 0;
  series.forEach((s) => {
    const pts = [...s.points].sort((a, b) => a.c - b.c).filter((p) => Math.abs(p.c) <= cm);
    if (!pts.length) return;
    let color, dash = null, wdt = 2, other = false;
    if (s.isSrc) { color = css("--c-src"); dash = "6 4"; wdt = 2.5; }
    else {
      const c = catColor(s.cid);
      if (c) { color = c; } else { color = css("--muted"); other = true; }
    }
    const attrs = { d: pts.map((p, i) => (i ? "L" : "M") + X(p.c) + "," + Y(p.p)).join(" "),
      fill: "none", stroke: color, "stroke-width": other ? 1.25 : wdt,
      "stroke-linejoin": "round", "stroke-linecap": "round", opacity: other ? 0.5 : 1 };
    if (dash) attrs["stroke-dasharray"] = dash;
    el("path", attrs, svg);
    pts.forEach((p) => {
      if (!other) el("circle", { cx: X(p.c), cy: Y(p.p), r: s.isSrc ? 4 : 3.5,
        fill: s.isSrc ? css("--surface") : color, stroke: color, "stroke-width": s.isSrc ? 2 : 1.5 }, svg);
      const hit = el("circle", { cx: X(p.c), cy: Y(p.p), r: 11, fill: "transparent" }, svg);
      const html = `<div class="tt-title">${s.label}</div>` +
        row("behavior", beh) + row("coefficient", gfmt(p.c)) +
        row("P(behavior)", fmt2(p.p)) + row("accuracy", fmt2(p.acc));
      hit.addEventListener("pointerenter", (e) => showTip(e, html));
      hit.addEventListener("pointermove", moveTip);
      hit.addEventListener("pointerleave", hideTip);
    });
    if (s.isSrc) legendItems.push({ label: s.label, color, dash: true });
    else if (!other) legendItems.push({ label: s.label, color });
    else otherCount++;
  });
  if (otherCount) legendItems.push({ label: "+" + otherCount + " other config" + (otherCount > 1 ? "s" : ""), color: css("--muted") });

  const leg = document.getElementById("legend-" + idx);
  if (leg) leg.innerHTML = legendItems.map((it) =>
    `<li><span class="swatch${it.dash ? " dash" : ""}" style="${it.dash ? "color" : "background"}:${it.color}"></span>${it.label}</li>`).join("");
}

// ── colorize server-rendered heatmap cells + summary corr cells ──
function colorizeCells() {
  document.querySelectorAll("td.cell").forEach((td) => {
    const p = parseFloat(td.dataset.p);
    if (Number.isNaN(p)) return;
    const bg = pColor(p);
    td.style.background = bg;
    td.style.color = inkFor(bg);
  });
  document.querySelectorAll(".corrcell").forEach((td) => {
    const v = parseFloat(td.dataset.v);
    if (Number.isNaN(v)) return;
    const bg = corrColor(v);
    td.style.background = bg;
    td.style.color = inkFor(bg);
  });
}

let _tipsAttached = false;
function attachCellTips() {
  if (_tipsAttached) return;
  _tipsAttached = true;
  document.querySelectorAll("td.cell[data-p]").forEach((td) => {
    const tr = td.closest("tr");
    const label = tr ? tr.dataset.label : "";
    const p = parseFloat(td.dataset.p), acc = parseFloat(td.dataset.acc), c = td.dataset.c;
    const html = `<div class="tt-title">${label}</div>` +
      row("coefficient", c) + row("P(behavior)", fmt2(p)) + row("accuracy", fmt2(acc));
    td.addEventListener("pointerenter", (e) => showTip(e, html));
    td.addEventListener("pointermove", moveTip);
    td.addEventListener("pointerleave", hideTip);
  });
}

function renderAll() {
  colorizeCells();
  DATA.behaviors.forEach((beh, i) => lineChart(i, beh));
}
attachCellTips();
renderAll();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
new MutationObserver(renderAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
"""


def _mean(vals):
    vs = [v for v in vals if v == v]
    return sum(vs) / len(vs) if vs else float("nan")


def build_html(rows, summary, source_curve, source_stats, path, coherent_max=5.0):
    behaviors = sorted({r["behavior"] for r in rows})
    coeffs = sorted({r["coefficient"] for r in rows})

    # index: (method,scope,tr,nm,beh,coeff) -> (p,acc)
    idx = {}
    for r in rows:
        idx[(r.get("method") or "CAA", r["scope"], r.get("translator", ""),
             r.get("norm_mode", ""), r["behavior"], r["coefficient"])] = (
                 r["avg_p_match"], r["accuracy"])

    def fmt(x):
        return f"{x:+.3f}" if x == x else "—"

    def dv(x):  # data-value attribute (NaN -> empty so JS skips it)
        return f"{x:.6f}" if x == x else ""

    # ── stable global translator-config order → categorical colors ──
    # The method leads every key so a mixed-method file keeps its series apart.
    # It only appears in the visible label when several methods are present, so
    # the usual single-method report reads exactly as before.
    all_methods = sorted({r.get("method") or "CAA" for r in rows})
    multi = len(all_methods) > 1

    def _m(meth):
        return f"{meth} " if multi else ""

    tgt_keys = sorted({(r.get("method") or "CAA",
                        r.get("translator_type", ""), r.get("loss", ""),
                        r.get("pooling", ""), r.get("norm_mode", ""),
                        r.get("source_layer", 8), r.get("target_layer", 8),
                        r.get("translator", ""))
                       for r in rows if r["scope"] == "target"})
    config_label = {}
    config_cid = {}
    config_labels = []
    for k in tgt_keys:
        meth, ttype, loss, pooling, nm, slay, tlay, tr = k
        label = f"{_m(meth)}{ttype}/{loss}/{pooling}/{nm} l{slay}→l{tlay}"
        config_cid[k] = len(config_labels)
        config_label[k] = label
        config_labels.append(label)

    # ── chart payload: per behavior, coherent-window curves for source + targets ──
    charts = {}
    for beh in behaviors:
        series = []
        src_labels = sorted({(r.get("method") or "CAA", r.get("source_layer", 8),
                              r.get("translator", ""))
                             for r in rows
                             if r["scope"] == "source" and r["behavior"] == beh})
        for meth, lay, slbl in src_labels:
            pts = []
            for c in coeffs:
                if abs(c) > coherent_max:
                    continue
                v = idx.get((meth, "source", slbl, "", beh, c))
                if v is not None:
                    pts.append({"c": c, "p": round(v[0], 4), "acc": round(v[1], 4)})
            series.append({"label": f"SOURCE ({_m(meth)}Llama-1B l{lay})",
                           "isSrc": True, "points": pts})
        for k in tgt_keys:
            meth, ttype, loss, pooling, nm, slay, tlay, tr = k
            pts = []
            for c in coeffs:
                if abs(c) > coherent_max:
                    continue
                v = idx.get((meth, "target", tr, nm, beh, c))
                if v is not None:
                    pts.append({"c": c, "p": round(v[0], 4), "acc": round(v[1], 4)})
            if pts:
                series.append({"label": config_label[k], "isSrc": False,
                               "cid": config_cid[k], "points": pts})
        charts[beh] = series

    data = {
        "coherentMax": coherent_max,
        "behaviors": behaviors,
        "coeffs": coeffs,
        "configLabels": config_labels,
        "charts": charts,
    }

    # ── header stat tiles ──
    mean_fid = _mean([s["fidelity_vs_source"] for s in summary])
    beh_fid = {beh: _mean([s["fidelity_vs_source"] for s in summary if s["behavior"] == beh])
               for beh in behaviors}
    valid_fid = {b: v for b, v in beh_fid.items() if v == v}
    best_beh = max(valid_fid, key=valid_fid.get) if valid_fid else None
    worst_beh = min(valid_fid, key=valid_fid.get) if valid_fid else None
    pairs = sorted({(s["source_layer"], s["target_layer"]) for s in summary})
    n_cfg = len(config_labels)
    n_beh = len(behaviors)

    if len(pairs) == 1:
        slay, tlay = pairs[0]
        transport_val = f"l{slay}&nbsp;&rarr;&nbsp;l{tlay}"
        transport_ctx = f"{n_cfg} translated config{'s' if n_cfg != 1 else ''} &times; {n_beh} behaviors"
    elif pairs:
        transport_val = f"{len(pairs)} layer&nbsp;pairs"
        transport_ctx = f"{n_cfg} translated config{'s' if n_cfg != 1 else ''} &times; {n_beh} behaviors"
    else:
        transport_val = "&mdash;"
        transport_ctx = "no translated configs"

    def _short(b):
        return html.escape(b) if b else "&mdash;"

    tiles = [
        ("Mean fidelity vs. source",
         f"{mean_fid:.2f}" if mean_fid == mean_fid else "&mdash;",
         f"Pearson corr of P-curves over |c|&le;{coherent_max:g}, averaged across all configs &amp; behaviors"),
        ("Best-transferring behavior",
         _short(best_beh),
         (f"fidelity {valid_fid[best_beh]:.2f}" if best_beh else "n/a")),
        ("Weakest-transferring behavior",
         _short(worst_beh),
         (f"fidelity {valid_fid[worst_beh]:.2f}" if worst_beh else "n/a")),
        ("Layer transport", transport_val, transport_ctx),
    ]
    tiles_html = "".join(
        f'<div class="tile"><p class="label">{lab}</p>'
        f'<p class="value">{val}</p><p class="context">{ctx}</p></div>'
        for lab, val, ctx in tiles)

    # ── metric-reminder cards ──
    cards = [
        ("response_corr",
         "Spearman rank-correlation between the steering coefficient and P(behavior) over the "
         "coherent window. <b>+1</b> = coefficient&uarr; reliably raises the behavior (the desired "
         "CAA direction); <b>~0</b> = no monotonic response; <b>&lt;0</b> = coefficient&uarr; "
         "<i>lowers</i> the behavior (wrong direction)."),
        (f"coherent regime |c|&le;{coherent_max:g}",
         "At large |coeff| the model collapses and P(behavior)&rarr;0 on <i>both</i> sides; the dead "
         "tails spuriously agree and inflate similarity. For these Llama-3.2 layer-8 residual vectors "
         "the source still steers cleanly through <code>|c|&le;5</code> and is fully collapsed by "
         "<code>|c|=10</code>. Collapse onset is behavior-dependent (~&pm;2 to &pm;5), so the cutoff "
         "is a heuristic &mdash; set it with <code>--coherent-max</code>. Columns past it are greyed."),
        ("dP_coherent",
         "P(beh)@+max &minus; P(beh)@&minus;max within the coherent window &mdash; the signed effect "
         "size of the steer."),
        ("fidelity_vs_source",
         "Pearson correlation between the translated vector's P-curve on the 3B model and the SOURCE "
         "vector's P-curve on the 1B model, over shared coherent coefficients. High = the translation "
         "reproduces the original steering response."),
        ("SOURCE row",
         "The original Llama-3.2-1B CAA vector on the 1B model &mdash; the reference every translated "
         "vector is judged against. Drawn as a thick dashed line in each chart and the top row of each "
         "heatmap."),
        ("heatmap cells",
         "Each cell is P(behavior) (avg&nbsp;P(match)), colored blue&rarr;grey&rarr;red for "
         "0&rarr;0.5&rarr;1; accuracy is printed below in the same cell."),
    ]
    cards_html = "".join(
        f'<div class="card"><h3>{h}</h3><p>{body}</p></div>' for h, body in cards)

    # ── summary table ──
    best_by_beh = {}
    for s in summary:
        v = s["response_corr"]
        if v == v and (s["behavior"] not in best_by_beh or v > best_by_beh[s["behavior"]]):
            best_by_beh[s["behavior"]] = v

    sum_rows = []
    for beh in behaviors:
        src_layers = sorted({(meth, lay) for (meth, lay, b) in source_stats if b == beh})
        for meth, lay in src_layers:
            ss = source_stats[(meth, lay, beh)]
            rc = ss.get("response_corr", float("nan"))
            sum_rows.append(
                f'<tr class="src"><td class="lbl">{html.escape(beh)}</td>'
                f'<td class="cfg" colspan="4">SOURCE (Llama-1B original)</td>'
                f'<td>l{lay}</td>'
                f'<td class="corrcell" data-v="{dv(rc)}">{fmt(rc)}</td>'
                f'<td>{fmt(ss.get("dP_coherent", float("nan")))}</td>'
                f'<td>{fmt(ss.get("p_neg", float("nan")))}</td>'
                f'<td>{fmt(ss.get("p_at_0", float("nan")))}</td>'
                f'<td>{fmt(ss.get("p_pos", float("nan")))}</td>'
                f'<td>{fmt(ss.get("monotonic", float("nan")))}</td>'
                f'<td>—</td><td>—</td></tr>')
        for s in sorted([x for x in summary if x["behavior"] == beh],
                        key=lambda x: -_nan_key(x["response_corr"])):
            is_best = s["response_corr"] == best_by_beh.get(beh, 99)
            cls = ' class="best"' if is_best else ""
            cfg = f'{s["translator_type"]}/{s["loss"]}/{s["pooling"]}'
            sum_rows.append(
                f'<tr{cls}><td class="lbl"></td>'
                f'<td class="cfg">{html.escape(cfg)}</td>'
                f'<td>{html.escape(str(s["norm_mode"]))}</td>'
                f'<td></td><td></td>'
                f'<td>l{s["source_layer"]}→l{s["target_layer"]}</td>'
                f'<td class="corrcell" data-v="{dv(s["response_corr"])}">{fmt(s["response_corr"])}</td>'
                f'<td>{fmt(s["dP_coherent"])}</td><td>{fmt(s["p_neg"])}</td>'
                f'<td>{fmt(s["p_at_0"])}</td><td>{fmt(s["p_pos"])}</td>'
                f'<td>{fmt(s["monotonic"])}</td>'
                f'<td class="corrcell" data-v="{dv(s["fidelity_vs_source"])}">{fmt(s["fidelity_vs_source"])}</td>'
                f'<td>{s["sv_norm"]:.3f}</td></tr>')

    summary_table = (
        '<div class="table-scroll"><table class="sum"><thead><tr>'
        '<th style="text-align:left">behavior</th>'
        '<th style="text-align:left">type/loss/pool</th><th>norm</th>'
        '<th></th><th></th><th>layers</th>'
        '<th>response_corr<br>(coeff&rarr;Pbeh)</th><th>dP coh.</th>'
        '<th>P@&minus;max</th><th>P@0</th><th>P@+max</th><th>monotonic</th>'
        '<th>fidelity<br>vs src</th><th>|sv|</th></tr></thead><tbody>'
        + "".join(sum_rows) + "</tbody></table></div>")

    # ── per-behavior sections: line chart + heatmap table ──
    def cell(meth, scope, tr, nm, beh, c):
        v = idx.get((meth, scope, tr, nm, beh, c))
        bk = " broken" if abs(c) > coherent_max else ""
        edge = " edge" if c == coherent_edge else ""
        if v is None:
            return f'<td class="cell{bk}{edge}">—</td>'
        p, acc = v
        return (f'<td class="cell{bk}{edge}" data-p="{p:.6f}" data-acc="{acc:.6f}" '
                f'data-c="{c:g}">{p:.2f}<span class="acc">{acc:.2f}</span></td>')

    # first collapsed coefficient (positive side) marks the coherent boundary
    coherent_edge = None
    for c in coeffs:
        if c > coherent_max:
            coherent_edge = c
            break

    sections = []
    for i, beh in enumerate(behaviors):
        head_cells = ['<th class="lbl">config</th>']
        for c in coeffs:
            classes = []
            if abs(c) > coherent_max:
                classes.append("broken")
            if c == coherent_edge:
                classes.append("edge")
            cls = f' class="{" ".join(classes)}"' if classes else ""
            head_cells.append(f'<th{cls}>{c:g}</th>')
        header = "<tr>" + "".join(head_cells) + "</tr>"

        body_rows = []
        src_labels = sorted({(r.get("method") or "CAA", r.get("source_layer", 8),
                              r.get("translator", ""))
                             for r in rows
                             if r["scope"] == "source" and r["behavior"] == beh})
        for meth, lay, slbl in src_labels:
            lbl = f"SOURCE ({_m(meth)}Llama-1B l{lay}, original)"
            cells = "".join(cell(meth, "source", slbl, "", beh, c) for c in coeffs)
            body_rows.append(f'<tr class="src" data-label="{html.escape(lbl)}">'
                             f'<td class="lbl">{html.escape(lbl)}</td>{cells}</tr>')
        for k in tgt_keys:
            meth, ttype, loss, pooling, nm, slay, tlay, tr = k
            # only render rows that actually have data for this behavior
            if not any(idx.get((meth, "target", tr, nm, beh, c)) is not None
                       for c in coeffs):
                continue
            lbl = config_label[k]
            cells = "".join(cell(meth, "target", tr, nm, beh, c) for c in coeffs)
            body_rows.append(f'<tr data-label="{html.escape(lbl)}">'
                             f'<td class="lbl">{html.escape(lbl)}</td>{cells}</tr>')

        heatmap = ('<div class="table-scroll"><table class="hm"><thead>'
                   + header + "</thead><tbody>" + "".join(body_rows)
                   + "</tbody></table></div>")

        sections.append(f"""
<section id="beh-{i}">
  <h2>{html.escape(beh)}</h2>
  <figure aria-label="Line chart: P(behavior) vs coefficient for {html.escape(beh)}">
    <p class="fig-title">P(behavior) vs. steering coefficient</p>
    <p class="fig-sub">coherent window |coeff|&le;{coherent_max:g}; y = avg P(match); dashed = SOURCE reference, dotted line = neutral P=0.5</p>
    <ul class="legend" id="legend-{i}"></ul>
    <div class="chart-scroll"><div id="chart-{i}"></div></div>
  </figure>
  <figure aria-label="Heatmap: full coefficient sweep for {html.escape(beh)}">
    <p class="fig-title">Full coefficient sweep &mdash; P(behavior) heatmap</p>
    <p class="fig-sub">rows = config; each cell is P(match) with accuracy below; greyed columns are past the coherent cutoff</p>
    <div class="hm-legend"><span>P(behavior)</span><span>0.0</span><span class="bar"></span><span>1.0</span></div>
    {heatmap}
  </figure>
</section>""")

    # ── assemble document ──
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A/B steering sweep &mdash; does coefficient &uarr; raise P(behavior)?</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header>
  <p class="eyebrow">tests_on_translation &middot; CAA steering-vector transport</p>
  <h1>Does turning the coefficient up raise P(behavior)?</h1>
  <p class="lede">
    A/B evaluation of translated CAA steering vectors: a good translation should behave like the
    original vector &mdash; pushing the coefficient <em>more positive</em> raises P(behavior) and
    more negative lowers it. Each translated vector on Llama-3.2-3B is compared against the SOURCE
    vector on Llama-3.2-1B across a coefficient grid, per MWE behavior.
  </p>
  <div class="tiles">{tiles_html}</div>
</header>

<section id="metrics">
  <h2>How to read this report</h2>
  <p class="section-note">
    The headline metric is <b>response_corr</b>; fidelity and effect size are computed only within
    the coherent regime, where the model has not yet collapsed. Hover any chart point or heatmap
    cell for the exact numbers.
  </p>
  <div class="cards">{cards_html}</div>
</section>

<section id="summary">
  <h2>Summary &mdash; coefficient&rarr;behavior response</h2>
  <p class="section-note">
    One row per config, grouped by behavior, ranked by response_corr. The bold <b>SOURCE</b> row is
    the Llama-1B reference; the strongest translated config per behavior is marked &starf;.
    response_corr and fidelity cells are tinted green (right direction) &rarr; red (wrong direction).
  </p>
  {summary_table}
</section>
{"".join(sections)}

<footer>
  Generated by <span class="mono">ab_report.py</span> from A/B sweep results
  (<span class="mono">ab_sweep.py</span>). SOURCE = CAA steering vector on Llama-3.2-1B;
  translated vectors evaluated on Llama-3.2-3B. Metrics computed within the coherent window
  |coeff|&le;{coherent_max:g}. Colors follow the project data-viz palette; the page themes to
  your light/dark preference.
</footer>
</main>

<div id="tooltip" role="status"></div>

<script>
const DATA = {json.dumps(data)};
{_JS}
</script>
</body>
</html>
"""
    Path(path).write_text(doc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+",
                    default=[str(DEFAULT_OUT / "*.jsonl")],
                    help="JSONL result file glob(s) to merge")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--methods", nargs="+", default=None, metavar="METHOD",
                    help="Restrict to these discovery methods (CAA/RepE/GCAV). "
                         "This report is single-method by construction — a coefficient "
                         "means a different physical dose per method — so pass one "
                         "method when the results file holds several. Rows with no "
                         "method field are treated as CAA. Default: all rows present.")
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
    rows = load_rows(args.results, methods=args.methods)
    if not rows:
        raise SystemExit(f"No result rows found in {args.results}"
                         + (f" for methods {args.methods}" if args.methods else ""))
    present = sorted({r["method"] for r in rows})
    print(f"  Loaded {len(rows)} rows from {args.results}  (methods: {present})")
    if len(present) > 1:
        print(f"  !! {len(present)} methods in one report: {present}. Coefficients are "
              "NOT comparable across methods (CAA carries its own magnitude, RepE/GCAV "
              "are unit-norm), so curves will be overlaid on an axis that means "
              "different doses per method. Re-run with --methods <ONE>, or use "
              "method_report.py for the cross-method view.")

    write_results_csv(rows, out / "results.csv")
    summary, source_curve, source_stats = build_summary(rows, args.coherent_max)
    write_summary_csv(summary, out / "summary.csv")
    build_html(rows, summary, source_curve, source_stats, out / "report.html",
               args.coherent_max)

    print(f"  Wrote:\n    {out/'results.csv'}\n    {out/'summary.csv'}\n    {out/'report.html'}")


if __name__ == "__main__":
    main()
