"""
Build ONE static HTML dashboard that consolidates every A/B steering-sweep run
(same-architecture translator comparison, cross-architecture new models,
layer-pair sweep, l8->l12 confirmation) so all of them can be browsed, filtered
and compared from a single page instead of four separate report.html files.

Reads each run's already-built results.csv (written by ab_sweep.py/ab_report.py)
straight off disk -- no re-running of any sweep. Reuses the response-metric math
from ab_report.py (Pearson/Spearman/effect-size/fidelity) but extends it to also
treat scope="native" rows (a target model's OWN natively-trained CAA vector,
evaluated at the same layer -- present for the new_models and layer_sweep runs)
as a second reference curve, so every chart/table can show, side by side:

    SOURCE  -- the original Llama-3.2-1B CAA vector (the thing being translated)
    NATIVE  -- that target model's own from-scratch CAA vector (a ceiling: does
               this behavior direction even exist/transfer on this architecture
               at all, independent of translation quality?)
    target  -- the TRANSLATED vector, one line/row per translator config

Metric definitions are NOT pasted as prose on the page. Every metric name is a
dotted-underline <span data-tip="..."> that shows its definition in the shared
hover tooltip (see GLOSSARY in the JS below) -- hover to read, otherwise the
page is just the results themselves: line charts + coefficient-sweep heatmaps.

Usage:
    conda run -n acteng python ab_dashboard.py
    conda run -n acteng python ab_dashboard.py --out outputs/ab_eval/dashboard.html
"""

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ab_report import _curve, _mean, _nan_key, _pearson, _response_metrics

_HERE = Path(__file__).resolve().parent

DATASETS = [
    {"id": "same_arch", "label": "Same-architecture translators",
     "sub": "Llama-3.2-1B → Llama-3.2-3B, layer 8→8, 58 translator configs",
     "dir": _HERE / "outputs" / "ab_eval", "coherent_max": 5.0},
    {"id": "cross_arch", "label": "Cross-architecture (new models)",
     "sub": "Llama-3.2-1B → Qwen2.5-0.5B / gemma-3-1B, layer 8→{8..18}",
     "dir": _HERE / "outputs" / "ab_eval" / "new_models", "coherent_max": 5.0},
    {"id": "layer_sweep", "label": "Layer-pair sweep",
     "sub": "Llama-3.2-1B l8 → Llama-3.2-3B l{8..18}, single translator config",
     "dir": _HERE / "outputs" / "ab_eval" / "layer_sweep", "coherent_max": 5.0},
    {"id": "l8_to_l12", "label": "l8→l12 confirmation",
     "sub": "Best config re-checked at Llama-3.2-1B l8 → Llama-3.2-3B l12",
     "dir": _HERE / "outputs" / "ab_eval" / "l8_to_l12", "coherent_max": 5.0},
]


def load_rows(results_csv, methods=("CAA",)):
    """Rows of one run's results.csv, restricted to `methods`.

    This dashboard indexes blocks by (scope, translator, norm, behavior) with no
    method component, and its coefficient axis is shared across every curve it
    draws — which is only valid within one discovery method, since CAA carries its
    own magnitude while RepE/GCAV are unit-norm. It therefore loads ONE method,
    defaulting to CAA (every run in DATASETS is a CAA run). A results.csv written
    before methods existed has no `method` column, so a missing value reads as
    CAA. Pass methods=None to disable the filter. Cross-method comparison lives in
    method_report.py."""
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if methods is not None:
        rows = [r for r in rows if (r.get("method") or "CAA") in methods]
    for r in rows:
        r["method"] = r.get("method") or "CAA"
        r["coefficient"] = float(r["coefficient"])
        r["avg_p_match"] = float(r["avg_p_match"])
        r["accuracy"] = float(r["accuracy"])
        sv = r.get("sv_norm")
        r["sv_norm"] = float(sv) if sv not in (None, "") else float("nan")
        r["source_layer"] = int(float(r["source_layer"])) if r.get("source_layer") else 8
        r["target_layer"] = int(float(r["target_layer"])) if r.get("target_layer") else 8
    return rows


_MODEL_L_RE = re.compile(r"_l\d+(?:_mean)?$")


def parse_translator_models(tr):
    """'best_translator__Llama-3.2-1B-Instruct_l8__gemma-3-1B-it_l14__linear__procrustes'
    -> ('Llama-3.2-1B-Instruct', 'gemma-3-1B-it'). Also strips a trailing
    '_mean' pooling suffix (e.g. '..._l8_mean' -> model name only). Returns
    (None, None) if the translator string doesn't follow this convention
    (e.g. it's a native row)."""
    parts = tr.split("__")
    if len(parts) >= 3 and parts[0] == "best_translator":
        return _MODEL_L_RE.sub("", parts[1]), _MODEL_L_RE.sub("", parts[2])
    return None, None


def parse_native_model(tr, fallback):
    """'native_Qwen2.5-0.5B-Instruct_l10' -> 'Qwen2.5-0.5B-Instruct'.
    'native_l10' (same-architecture run, no model in the name) -> fallback."""
    rest = tr[len("native_"):] if tr.startswith("native_") else tr
    m = re.match(r"^(.*)_l\d+$", rest)
    if m and m.group(1):
        return m.group(1)
    return fallback


def build_dataset(rows, coherent_max):
    behaviors = sorted({r["behavior"] for r in rows})
    coeffs = sorted({r["coefficient"] for r in rows})

    blocks = defaultdict(list)
    for r in rows:
        blocks[(r["scope"], r.get("translator", ""), r.get("norm_mode", ""), r["behavior"])].append(r)

    source_curve, source_stats = {}, {}
    for (scope, tr, nm, beh), rs in blocks.items():
        if scope == "source":
            lay = rs[0]["source_layer"]
            source_curve[(lay, beh)] = _curve(rs)
            source_stats[(lay, beh)] = _response_metrics(source_curve[(lay, beh)], coherent_max)

    def fidelity(cur, src_lay, beh):
        sc = source_curve.get((src_lay, beh))
        if not sc:
            return float("nan")
        shared = sorted(c for c in (set(cur) & set(sc)) if abs(c) <= coherent_max)
        return _pearson([cur[c][0] for c in shared], [sc[c][0] for c in shared]) if len(shared) >= 2 else float("nan")

    target_summary, native_summary = [], []
    for (scope, tr, nm, beh), rs in blocks.items():
        if scope not in ("target", "native"):
            continue
        cur = _curve(rs)
        m = _response_metrics(cur, coherent_max)
        ex = rs[0]
        src_lay = ex["source_layer"]
        row = {
            "scope": scope, "translator": tr,
            "translator_type": ex.get("translator_type", ""), "loss": ex.get("loss", ""),
            "pooling": ex.get("pooling", ""), "norm_mode": nm, "behavior": beh,
            "response_corr": m["response_corr"], "dP_coherent": m["dP_coherent"],
            "p_neg": m["p_neg"], "p_at_0": m["p_at_0"], "p_pos": m["p_pos"],
            "monotonic": m["monotonic"], "fidelity_vs_source": fidelity(cur, src_lay, beh),
            "sv_norm": ex.get("sv_norm", float("nan")), "source_layer": src_lay,
            "target_layer": ex.get("target_layer", 8),
        }
        (target_summary if scope == "target" else native_summary).append(row)

    # dominant target model name, for native-row labels that don't carry one
    tgt_models = [parse_translator_models(r["translator"])[1] for r in rows if r["scope"] == "target"]
    tgt_models = [m for m in tgt_models if m]
    dominant_tgt_model = Counter(tgt_models).most_common(1)[0][0] if tgt_models else "target model"
    distinct_tgt_models = sorted(set(tgt_models))

    idx = {}
    for r in rows:
        idx[(r["scope"], r.get("translator", ""), r.get("norm_mode", ""), r["behavior"], r["coefficient"])] = \
            (r["avg_p_match"], r["accuracy"])

    tgt_keys = sorted({(r.get("translator_type", ""), r.get("loss", ""), r.get("pooling", ""),
                        r.get("norm_mode", ""), r["source_layer"], r["target_layer"], r.get("translator", ""))
                       for r in rows if r["scope"] == "target"})
    config_label, config_cid, config_labels = {}, {}, []
    for k in tgt_keys:
        ttype, loss, pooling, nm, slay, tlay, tr = k
        _, tgt_model = parse_translator_models(tr)
        model_bit = f"{tgt_model} " if tgt_model and len(distinct_tgt_models) > 1 else ""
        label = f"{model_bit}{ttype}/{loss}/{pooling}/{nm} l{slay}→l{tlay}"
        config_cid[k] = len(config_labels)
        config_label[k] = label
        config_labels.append(label)

    native_keys = sorted({(r.get("translator", ""), r["target_layer"]) for r in rows if r["scope"] == "native"})
    native_label = {k: f"NATIVE ({parse_native_model(k[0], dominant_tgt_model)} l{k[1]}, own CAA)" for k in native_keys}

    charts = {}
    for beh in behaviors:
        series = []
        src_labels = sorted({(r["source_layer"], r.get("translator", "")) for r in rows
                             if r["scope"] == "source" and r["behavior"] == beh})
        for lay, slbl in src_labels:
            pts = [{"c": c, "p": round(v[0], 4), "acc": round(v[1], 4)}
                   for c in coeffs if abs(c) <= coherent_max
                   for v in [idx.get(("source", slbl, "", beh, c))] if v is not None]
            series.append({"label": f"SOURCE (Llama-1B l{lay})", "kind": "src", "points": pts})
        for tr, lay in native_keys:
            pts = [{"c": c, "p": round(v[0], 4), "acc": round(v[1], 4)}
                   for c in coeffs if abs(c) <= coherent_max
                   for v in [idx.get(("native", tr, "", beh, c))] if v is not None]
            if pts:
                series.append({"label": native_label[(tr, lay)], "kind": "native", "points": pts})
        for k in tgt_keys:
            ttype, loss, pooling, nm, slay, tlay, tr = k
            pts = [{"c": c, "p": round(v[0], 4), "acc": round(v[1], 4)}
                   for c in coeffs if abs(c) <= coherent_max
                   for v in [idx.get(("target", tr, nm, beh, c))] if v is not None]
            if pts:
                series.append({"label": config_label[k], "kind": "target", "cid": config_cid[k], "points": pts})
        charts[beh] = series

    return {
        "behaviors": behaviors, "coeffs": coeffs, "coherentMax": coherent_max,
        "configLabels": config_labels, "charts": charts,
        "idx": idx, "tgt_keys": tgt_keys, "config_label": config_label,
        "native_keys": native_keys, "native_label": native_label,
        "target_summary": target_summary, "native_summary": native_summary,
        "source_curve": source_curve, "source_stats": source_stats,
        "dominant_tgt_model": dominant_tgt_model,
    }


def fmt(x):
    return f"{x:+.3f}" if x == x else "—"


def dv(x):
    return f"{x:.6f}" if x == x else ""


def cell(idx, scope, tr, nm, beh, c, coherent_max, coherent_edge):
    v = idx.get((scope, tr, nm, beh, c))
    bk = " broken" if abs(c) > coherent_max else ""
    edge = " edge" if c == coherent_edge else ""
    if v is None:
        return f'<td class="cell{bk}{edge}">—</td>'
    p, acc = v
    return f'<td class="cell{bk}{edge}" data-p="{p:.6f}" data-acc="{acc:.6f}" data-c="{c:g}">{p:.2f}<span class="acc">{acc:.2f}</span></td>'


def build_tiles(ds):
    target_summary, native_summary = ds["target_summary"], ds["native_summary"]
    behaviors = ds["behaviors"]
    mean_fid = _mean([s["fidelity_vs_source"] for s in target_summary])
    beh_fid = {b: _mean([s["fidelity_vs_source"] for s in target_summary if s["behavior"] == b]) for b in behaviors}
    valid_fid = {b: v for b, v in beh_fid.items() if v == v}
    best_beh = max(valid_fid, key=valid_fid.get) if valid_fid else None
    worst_beh = min(valid_fid, key=valid_fid.get) if valid_fid else None
    pairs = sorted({(s["source_layer"], s["target_layer"]) for s in target_summary})
    n_cfg, n_beh = len(ds["configLabels"]), len(behaviors)

    if len(pairs) == 1:
        slay, tlay = pairs[0]
        transport_val, transport_ctx = f"l{slay}&nbsp;&rarr;&nbsp;l{tlay}", f"{n_cfg} config{'s' if n_cfg != 1 else ''} &times; {n_beh} behaviors"
    elif pairs:
        transport_val, transport_ctx = f"{len(pairs)} layer&nbsp;pairs", f"{n_cfg} config{'s' if n_cfg != 1 else ''} &times; {n_beh} behaviors"
    else:
        transport_val, transport_ctx = "&mdash;", "no translated configs"

    def short(b):
        return html.escape(b) if b else "&mdash;"

    tiles = [
        ("Mean fidelity vs. source", f"{mean_fid:.2f}" if mean_fid == mean_fid else "&mdash;",
         '<span class="metric" data-tip="fidelity_vs_source">fidelity</span> averaged across all configs &amp; behaviors'),
        ("Best-transferring behavior", short(best_beh), f"fidelity {valid_fid[best_beh]:.2f}" if best_beh else "n/a"),
        ("Weakest-transferring behavior", short(worst_beh), f"fidelity {valid_fid[worst_beh]:.2f}" if worst_beh else "n/a"),
        ("Layer transport", transport_val, transport_ctx),
    ]
    if native_summary:
        mean_native_fid = _mean([s["fidelity_vs_source"] for s in native_summary])
        tiles.append(("Native ceiling",
                       f"{mean_native_fid:.2f}" if mean_native_fid == mean_native_fid else "&mdash;",
                       'target model’s <span class="metric" data-tip="native_row">own</span> CAA vector vs. SOURCE'))
    return "".join(f'<div class="tile"><p class="label">{lab}</p><p class="value">{val}</p>'
                   f'<p class="context">{ctx}</p></div>' for lab, val, ctx in tiles)


def build_summary_table(ds):
    behaviors = ds["behaviors"]
    best_by_beh = {}
    for s in ds["target_summary"]:
        v = s["response_corr"]
        if v == v and (s["behavior"] not in best_by_beh or v > best_by_beh[s["behavior"]]):
            best_by_beh[s["behavior"]] = v

    rows_html = []
    for beh in behaviors:
        src_layers = sorted({lay for (lay, b) in ds["source_stats"] if b == beh})
        for lay in src_layers:
            ss = ds["source_stats"][(lay, beh)]
            rc = ss.get("response_corr", float("nan"))
            rows_html.append(
                f'<tr class="src"><td class="lbl">{html.escape(beh)}</td>'
                f'<td class="cfg" colspan="2">SOURCE (Llama-1B original)</td>'
                f'<td>l{lay}</td>'
                f'<td class="corrcell" data-v="{dv(rc)}">{fmt(rc)}</td>'
                f'<td>{fmt(ss.get("dP_coherent", float("nan")))}</td>'
                f'<td>{fmt(ss.get("p_neg", float("nan")))}</td>'
                f'<td>{fmt(ss.get("p_at_0", float("nan")))}</td>'
                f'<td>{fmt(ss.get("p_pos", float("nan")))}</td>'
                f'<td>{fmt(ss.get("monotonic", float("nan")))}</td>'
                f'<td>—</td><td>—</td></tr>')
        for s in sorted([x for x in ds["native_summary"] if x["behavior"] == beh],
                        key=lambda x: -_nan_key(x["response_corr"])):
            lbl = ds["native_label"].get((s["translator"], s["target_layer"]), "NATIVE")
            rows_html.append(
                f'<tr class="native"><td class="lbl"></td>'
                f'<td class="cfg" colspan="2">{html.escape(lbl)}</td>'
                f'<td>l{s["source_layer"]}→l{s["target_layer"]}</td>'
                f'<td class="corrcell" data-v="{dv(s["response_corr"])}">{fmt(s["response_corr"])}</td>'
                f'<td>{fmt(s["dP_coherent"])}</td><td>{fmt(s["p_neg"])}</td>'
                f'<td>{fmt(s["p_at_0"])}</td><td>{fmt(s["p_pos"])}</td>'
                f'<td>{fmt(s["monotonic"])}</td>'
                f'<td class="corrcell" data-v="{dv(s["fidelity_vs_source"])}">{fmt(s["fidelity_vs_source"])}</td>'
                f'<td>{s["sv_norm"]:.3f}</td></tr>')
        for s in sorted([x for x in ds["target_summary"] if x["behavior"] == beh],
                        key=lambda x: -_nan_key(x["response_corr"])):
            is_best = s["response_corr"] == best_by_beh.get(beh, 99)
            cls = ' class="best"' if is_best else ""
            cfg = f'{s["translator_type"]}/{s["loss"]}/{s["pooling"]}'
            rows_html.append(
                f'<tr{cls}><td class="lbl"></td>'
                f'<td class="cfg">{html.escape(cfg)}</td>'
                f'<td>{html.escape(str(s["norm_mode"]))}</td>'
                f'<td>l{s["source_layer"]}→l{s["target_layer"]}</td>'
                f'<td class="corrcell" data-v="{dv(s["response_corr"])}">{fmt(s["response_corr"])}</td>'
                f'<td>{fmt(s["dP_coherent"])}</td><td>{fmt(s["p_neg"])}</td>'
                f'<td>{fmt(s["p_at_0"])}</td><td>{fmt(s["p_pos"])}</td>'
                f'<td>{fmt(s["monotonic"])}</td>'
                f'<td class="corrcell" data-v="{dv(s["fidelity_vs_source"])}">{fmt(s["fidelity_vs_source"])}</td>'
                f'<td>{s["sv_norm"]:.3f}</td></tr>')

    head = (
        '<tr><th style="text-align:left">behavior</th>'
        '<th style="text-align:left">type/loss/pool</th><th>norm</th><th>layers</th>'
        '<th><span class="metric" data-tip="response_corr">response_corr</span></th>'
        '<th><span class="metric" data-tip="dP_coherent">dP coh.</span></th>'
        '<th><span class="metric" data-tip="p_at_edges">P@&minus;max</span></th>'
        '<th><span class="metric" data-tip="p_at_edges">P@0</span></th>'
        '<th><span class="metric" data-tip="p_at_edges">P@+max</span></th>'
        '<th><span class="metric" data-tip="monotonic">monotonic</span></th>'
        '<th><span class="metric" data-tip="fidelity_vs_source">fidelity<br>vs src</span></th>'
        '<th><span class="metric" data-tip="sv_norm">|sv|</span></th></tr>')
    return f'<div class="table-scroll"><table class="sum"><thead>{head}</thead><tbody>{"".join(rows_html)}</tbody></table></div>'


def build_behavior_section(ds, ds_id, i, beh):
    coherent_max = ds["coherentMax"]
    coeffs = ds["coeffs"]
    idx = ds["idx"]

    coherent_edge = next((c for c in coeffs if c > coherent_max), None)
    head_cells = ['<th class="lbl">config</th>']
    for c in coeffs:
        classes = []
        if abs(c) > coherent_max:
            classes.append("broken")
        if c == coherent_edge:
            classes.append("edge")
        cls = f' class="{" ".join(classes)}"' if classes else ""
        head_cells.append(f"<th{cls}>{c:g}</th>")
    header = "<tr>" + "".join(head_cells) + "</tr>"

    body_rows = []
    src_pairs = sorted({lay for (lay, b) in ds["source_stats"] if b == beh})
    for lay in src_pairs:
        # find the source translator label used at this layer for this behavior
        src_tr = next((tr for (scope, tr, nm, b2, c), v in idx.items()
                       if scope == "source" and b2 == beh), "")
        lbl = f"SOURCE (Llama-1B l{lay}, original)"
        cells = "".join(cell(idx, "source", src_tr, "", beh, c, coherent_max, coherent_edge) for c in coeffs)
        body_rows.append(f'<tr class="src" data-label="{html.escape(lbl)}"><td class="lbl">{html.escape(lbl)}</td>{cells}</tr>')

    for tr, lay in ds["native_keys"]:
        if not any(idx.get(("native", tr, "", beh, c)) is not None for c in coeffs):
            continue
        lbl = ds["native_label"][(tr, lay)]
        cells = "".join(cell(idx, "native", tr, "", beh, c, coherent_max, coherent_edge) for c in coeffs)
        body_rows.append(f'<tr class="native" data-label="{html.escape(lbl)}"><td class="lbl">{html.escape(lbl)}</td>{cells}</tr>')

    for k in ds["tgt_keys"]:
        ttype, loss, pooling, nm, slay, tlay, tr = k
        if not any(idx.get(("target", tr, nm, beh, c)) is not None for c in coeffs):
            continue
        lbl = ds["config_label"][k]
        cells = "".join(cell(idx, "target", tr, nm, beh, c, coherent_max, coherent_edge) for c in coeffs)
        body_rows.append(f'<tr data-label="{html.escape(lbl)}"><td class="lbl">{html.escape(lbl)}</td>{cells}</tr>')

    heatmap = f'<div class="table-scroll"><table class="hm"><thead>{header}</thead><tbody>{"".join(body_rows)}</tbody></table></div>'
    active = " active" if i == 0 else ""
    return f"""
<section class="beh-section{active}" data-beh="{i}" id="{ds_id}-beh-{i}">
  <figure aria-label="Line chart: P(behavior) vs coefficient for {html.escape(beh)}">
    <p class="fig-title">P(behavior) vs. steering coefficient &mdash; {html.escape(beh)}</p>
    <p class="fig-sub">coherent window |coeff|&le;{coherent_max:g}; dashed black = SOURCE, solid grey = NATIVE ceiling, colored = translated configs</p>
    <ul class="legend" id="legend-{ds_id}-{i}"></ul>
    <div class="chart-scroll"><div id="chart-{ds_id}-{i}"></div></div>
  </figure>
  <figure aria-label="Heatmap: full coefficient sweep for {html.escape(beh)}">
    <p class="fig-title">Full coefficient sweep &mdash; <span class="metric" data-tip="heatmap">P(behavior) heatmap</span></p>
    <p class="fig-sub">rows = config; each cell is P(match) with accuracy below; greyed columns are past the coherent cutoff</p>
    <div class="hm-legend"><span>P(behavior)</span><span>0.0</span><span class="bar"></span><span>1.0</span></div>
    {heatmap}
  </figure>
</section>"""


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
    --c-src: #0b0b0b; --c-native: #7a7869;
    --cell-lo: #256abf; --cell-mid: #f0efec; --cell-hi: #d03b3b;
    --stat-good: #0ca30c; --stat-bad: #d03b3b;
    --best: #008300; --tab-bg: #eeede7;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --axis: #383835;
      --border: rgba(255, 255, 255, 0.10); --accent: #3987e5;
      --cat-1: #3987e5; --cat-2: #d95926; --cat-3: #199e70; --cat-4: #c98500;
      --cat-5: #d55181; --cat-6: #008300; --cat-7: #9085e9; --cat-8: #e66767;
      --c-src: #ffffff; --c-native: #a19f8d;
      --cell-lo: #3987e5; --cell-mid: #383835; --cell-hi: #e66767;
      --stat-good: #0ca30c; --stat-bad: #d03b3b;
      --best: #23a559; --tab-bg: #232322;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --border: rgba(255, 255, 255, 0.10); --accent: #3987e5;
    --cat-1: #3987e5; --cat-2: #d95926; --cat-3: #199e70; --cat-4: #c98500;
    --cat-5: #d55181; --cat-6: #008300; --cat-7: #9085e9; --cat-8: #e66767;
    --c-src: #ffffff; --c-native: #a19f8d;
    --cell-lo: #3987e5; --cell-mid: #383835; --cell-hi: #e66767;
    --stat-good: #0ca30c; --stat-bad: #d03b3b;
    --best: #23a559; --tab-bg: #232322;
  }
  html { background: var(--page); }
  body { margin: 0; background: var(--page); color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 15px; line-height: 1.55; }
  main { max-width: 1080px; margin: 0 auto; padding: 48px 24px 96px; }
  h1, h2 { font-family: Charter, "Bitstream Charter", "Iowan Old Style", Georgia, serif;
    font-weight: 600; text-wrap: balance; margin: 0; }
  h1 { font-size: 2.2rem; line-height: 1.15; letter-spacing: -0.01em; }
  h2 { font-size: 1.15rem; margin: 0 0 4px; }
  section { margin-top: 40px; }
  p { max-width: 72ch; }
  code, .mono { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 0.92em; }
  a { color: var(--accent); }
  .eyebrow { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.72rem;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); margin: 0 0 14px; }
  .lede { color: var(--ink-2); font-size: 1.02rem; margin: 14px 0 0; }
  .hint { color: var(--muted); font-size: 0.85rem; margin: 10px 0 0; }

  .metric { border-bottom: 1px dotted var(--muted); cursor: help; }

  /* tab bars */
  .tabbar { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 0; padding: 0; list-style: none; }
  .tabbar button { font: inherit; font-size: 0.86rem; padding: 7px 13px; border-radius: 7px;
    border: 1px solid var(--border); background: var(--tab-bg); color: var(--ink-2); cursor: pointer; }
  .tabbar button.active { background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 600; }
  .tabbar.dataset button { font-size: 0.92rem; padding: 9px 15px; }
  .dataset-panel { display: none; }
  .dataset-panel.active { display: block; }
  .beh-section { display: none; }
  .beh-section.active { display: block; }
  .ds-sub { color: var(--muted); font-size: 0.85rem; margin: 8px 0 0; }

  .tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 22px 0 0; }
  .tile { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px 14px; }
  .tile .label { font-size: 0.82rem; color: var(--ink-2); margin: 0; }
  .tile .value { font-size: 1.7rem; font-weight: 600; margin: 2px 0 0; line-height: 1.2; word-break: break-word; }
  .tile .context { font-size: 0.78rem; color: var(--muted); margin: 4px 0 0; }

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
  .legend .swatch.solidthick { width: 18px; height: 0; border-radius: 0; background: none !important;
    border-top: 3px solid currentColor; }

  #tooltip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink);
    border: 1px solid var(--border); border-radius: 6px; box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    padding: 9px 12px; font-size: 0.8rem; line-height: 1.45; max-width: 320px; opacity: 0; z-index: 10; }
  #tooltip .tt-title { font-weight: 600; }
  #tooltip .tt-body { margin-top: 4px; color: var(--ink-2); }
  #tooltip .tt-row { display: flex; justify-content: space-between; gap: 16px; }
  #tooltip .tt-row span:last-child { font-family: ui-monospace, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
  @media (prefers-reduced-motion: no-preference) { #tooltip { transition: opacity 120ms ease; } }

  .table-scroll { overflow-x: auto; margin-top: 12px; max-height: 560px; }
  table.sum { border-collapse: collapse; font-size: 0.8rem; min-width: 760px; background: var(--surface); width: 100%; }
  table.sum th, table.sum td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
  table.sum th { color: var(--ink-2); font-weight: 600; font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.72rem; position: sticky; top: 0; background: var(--surface); z-index: 2; }
  table.sum td { font-variant-numeric: tabular-nums; color: var(--ink); }
  table.sum td.lbl { text-align: left; font-weight: 600; font-family: ui-monospace, Menlo, Consolas, monospace; }
  table.sum td.cfg { text-align: left; font-family: ui-monospace, Menlo, Consolas, monospace; color: var(--ink-2); }
  table.sum tr.src td { font-weight: 700; }
  table.sum tr.src td.lbl { color: var(--ink); }
  table.sum tr.src { border-top: 2px solid var(--axis); }
  table.sum tr.native td.cfg { color: var(--c-native); font-weight: 600; }
  table.sum tr.best td.cfg::after { content: " ★"; color: var(--best); }
  table.sum tr.best td { background: color-mix(in srgb, var(--best) 12%, transparent); }
  .corrcell { font-weight: 600; border-radius: 3px; }

  table.hm { border-collapse: separate; border-spacing: 2px; font-size: 0.72rem; background: var(--surface); }
  table.hm th, table.hm td { padding: 4px 6px; text-align: center; white-space: nowrap; }
  table.hm th { color: var(--ink-2); font-weight: 600; font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.68rem; position: sticky; top: 0; background: var(--surface); z-index: 2; }
  table.hm th.lbl, table.hm td.lbl { position: sticky; left: 0; z-index: 3; background: var(--surface);
    text-align: left; font-family: ui-monospace, Menlo, Consolas, monospace; color: var(--ink-2);
    box-shadow: 1px 0 0 var(--grid); }
  table.hm th.lbl { z-index: 4; }
  table.hm tr.src td.lbl, table.hm tr.src th.lbl { color: var(--ink); font-weight: 700; }
  table.hm tr.native td.lbl { color: var(--c-native); font-weight: 700; }
  table.hm td.cell { border-radius: 3px; font-variant-numeric: tabular-nums; color: var(--ink-2);
    background: var(--cell-mid); min-width: 40px; }
  table.hm td.cell .acc { display: block; font-size: 0.9em; opacity: 0.75; }
  table.hm .broken { opacity: 0.42; }
  table.hm th.broken { color: var(--muted); }
  table.hm th.edge, table.hm td.edge { box-shadow: inset 2px 0 0 var(--axis); }
  .hm-legend { display: flex; align-items: center; gap: 10px; font-size: 0.78rem; color: var(--ink-2); margin: 0 0 6px; }
  .hm-legend .bar { width: 150px; height: 12px; border-radius: 3px; border: 1px solid var(--border);
    background: linear-gradient(90deg, var(--cell-lo), var(--cell-mid), var(--cell-hi)); }

  footer { margin-top: 60px; color: var(--muted); font-size: 0.8rem; }
"""

_JS = r"""
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const NS = "http://www.w3.org/2000/svg";
const isNil = (x) => x === null || x === undefined || Number.isNaN(x);
const fmt2 = (x) => isNil(x) ? "—" : x.toFixed(2);
const gfmt = (x) => { const a = Math.abs(x); return a >= 1000 ? (x / 1000) + "k" : String(x); };

const GLOSSARY = {
  response_corr: {title: "response_corr", body: "Spearman rank-correlation between the steering coefficient and P(behavior), over the coherent window. +1 = coefficient↑ reliably raises the behavior (the desired CAA direction); ~0 = no monotonic response; <0 = coefficient↑ lowers the behavior (wrong direction)."},
  dP_coherent: {title: "dP_coherent", body: "P(behavior)@+max − P(behavior)@−max within the coherent window — the signed effect size of the steer."},
  p_at_edges: {title: "P(behavior)", body: "Average P(match) at the largest coherent negative coefficient, at coefficient 0 (baseline, no steering), and at the largest coherent positive coefficient."},
  monotonic: {title: "monotonic", body: "Fraction of adjacent coherent steps where P(behavior) increases as the coefficient increases — 1.0 = perfectly monotonic response."},
  fidelity_vs_source: {title: "fidelity_vs_source", body: "Pearson correlation between this vector's P(behavior)-vs-coefficient curve and the SOURCE vector's curve, over shared coherent coefficients. High = this vector reproduces the original steering response shape."},
  sv_norm: {title: "|sv| — steering vector norm", body: "L2 norm of the vector actually injected at the target layer. Large norms relative to the source can mean the vector is off-manifold for that layer/model."},
  coherent: {title: "coherent window", body: "At large |coefficient| the model breaks down and P(behavior)→0 on BOTH sides — the dead tails spuriously agree and would inflate similarity if included. Metrics are restricted to |coeff| below this cutoff (default 5); columns past it are greyed in the heatmap."},
  heatmap: {title: "P(behavior) heatmap", body: "Every cell is avg P(match) at that (config, coefficient), colored blue→grey→red for 0→0.5→1; accuracy is printed below in the same cell. Hover a cell for exact numbers."},
  source_row: {title: "SOURCE row", body: "The original Llama-3.2-1B CAA vector, evaluated on Llama-3.2-1B itself — the reference every translated (and native) vector is judged against."},
  native_row: {title: "NATIVE row", body: "The target model's OWN CAA vector, trained from scratch on that model at the same layer (not translated from anywhere). This is a ceiling: how well this behavior direction works on this architecture at all, independent of translation quality."},
  target_config: {title: "translated config", body: "translator_type / loss / pooling / norm_mode of the learned map used to transport the SOURCE vector onto the target model, plus the source→target layer pair."},
};

const tip = document.getElementById("tooltip");
function showTip(e, htmlStr) { tip.innerHTML = htmlStr; tip.style.opacity = 1; moveTip(e); }
function moveTip(e) {
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + pad, y = e.clientY + pad;
  if (x + w > innerWidth - 8) x = e.clientX - w - pad;
  if (y + h > innerHeight - 8) y = e.clientY - h - pad;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.style.opacity = 0; }
const row = (k, v) => `<div class="tt-row"><span>${k}</span><span>${v}</span></div>`;

function hexToRgb(h) {
  h = h.trim();
  if (h.startsWith("rgb")) return h.match(/\d+/g).map(Number);
  const s = h.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(s.slice(i, i + 2), 16));
}
function mix(h1, h2, t) {
  const a = hexToRgb(h1), b = hexToRgb(h2);
  return "rgb(" + a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",") + ")";
}
function inkFor(bg) {
  const m = bg.match(/\d+/g).map(Number);
  const lum = (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) / 255;
  return lum > 0.6 ? "#0b0b0b" : "#ffffff";
}
function pColor(p) {
  const t = Math.max(-1, Math.min(1, (p - 0.5) / 0.5));
  return t >= 0 ? mix(css("--cell-mid"), css("--cell-hi"), t) : mix(css("--cell-mid"), css("--cell-lo"), -t);
}
function corrColor(v) {
  const t = Math.max(-1, Math.min(1, v));
  return t >= 0 ? mix(css("--cell-mid"), css("--stat-good"), t) : mix(css("--cell-mid"), css("--stat-bad"), -t);
}
function catColor(cid) { return (cid >= 0 && cid < 8) ? css("--cat-" + (cid + 1)) : null; }

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
    "text-anchor": a.anchor || "start", "dominant-baseline": a.baseline || "middle", ...(a.extra || {}) }, svg);
  t.textContent = s; return t;
}

function lineChart(dsId, idx, beh, coeffs, coherentMax, series) {
  const container = document.getElementById(`chart-${dsId}-${idx}`);
  if (!container) return;
  const xs = coeffs.filter((c) => Math.abs(c) <= coherentMax);
  if (!xs.length) { container.innerHTML = ""; return; }
  const M = { l: 44, r: 16, t: 14, b: 40 }, W = 720, H = 300;
  const plotW = W - M.l - M.r, plotH = H - M.t - M.b;
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const X = (v) => M.l + (x1 === x0 ? 0.5 : (v - x0) / (x1 - x0)) * plotW;
  const Y = (v) => M.t + plotH - v * plotH;
  const svg = svgRoot(container, W, H);

  for (let v = 0; v <= 1.0001; v += 0.25) {
    el("line", { x1: M.l, x2: M.l + plotW, y1: Y(v), y2: Y(v), stroke: css("--grid"), "stroke-width": 1 }, svg);
    text(svg, M.l - 8, Y(v), v.toFixed(2), { anchor: "end", size: 10, fill: css("--muted"), extra: { "font-variant-numeric": "tabular-nums" } });
  }
  el("line", { x1: M.l, x2: M.l + plotW, y1: Y(0.5), y2: Y(0.5), stroke: css("--axis"), "stroke-width": 1, "stroke-dasharray": "3 4" }, svg);
  xs.forEach((c) => text(svg, X(c), M.t + plotH + 16, gfmt(c), { anchor: "middle", size: 10, fill: css("--muted"), extra: { "font-variant-numeric": "tabular-nums" } }));
  text(svg, M.l + plotW / 2, H - 3, "steering coefficient", { anchor: "middle", size: 10.5, fill: css("--ink-2") });

  const legendItems = [];
  let otherCount = 0;
  series.forEach((s) => {
    const pts = [...s.points].sort((a, b) => a.c - b.c).filter((p) => Math.abs(p.c) <= coherentMax);
    if (!pts.length) return;
    let color, dash = null, wdt = 2, other = false, kindClass = s.kind;
    if (s.kind === "src") { color = css("--c-src"); dash = "6 4"; wdt = 2.5; }
    else if (s.kind === "native") { color = css("--c-native"); wdt = 3; }
    else {
      const c = catColor(s.cid);
      if (c) color = c; else { color = css("--muted"); other = true; }
    }
    const attrs = { d: pts.map((p, i) => (i ? "L" : "M") + X(p.c) + "," + Y(p.p)).join(" "),
      fill: "none", stroke: color, "stroke-width": other ? 1.25 : wdt,
      "stroke-linejoin": "round", "stroke-linecap": "round", opacity: other ? 0.5 : 1 };
    if (dash) attrs["stroke-dasharray"] = dash;
    el("path", attrs, svg);
    pts.forEach((p) => {
      if (!other) el("circle", { cx: X(p.c), cy: Y(p.p), r: s.kind === "src" ? 4 : 3.5,
        fill: s.kind === "src" ? css("--surface") : color, stroke: color, "stroke-width": s.kind === "src" ? 2 : 1.5 }, svg);
      const hit = el("circle", { cx: X(p.c), cy: Y(p.p), r: 11, fill: "transparent" }, svg);
      const htmlStr = `<div class="tt-title">${s.label}</div>` +
        row("behavior", beh) + row("coefficient", gfmt(p.c)) +
        row("P(behavior)", fmt2(p.p)) + row("accuracy", fmt2(p.acc));
      hit.addEventListener("pointerenter", (e) => showTip(e, htmlStr));
      hit.addEventListener("pointermove", moveTip);
      hit.addEventListener("pointerleave", hideTip);
    });
    if (s.kind === "src") legendItems.push({ label: s.label, color, dash: true });
    else if (s.kind === "native") legendItems.push({ label: s.label, color, thick: true });
    else if (!other) legendItems.push({ label: s.label, color });
    else otherCount++;
  });
  if (otherCount) legendItems.push({ label: "+" + otherCount + " other config" + (otherCount > 1 ? "s" : ""), color: css("--muted") });

  const leg = document.getElementById(`legend-${dsId}-${idx}`);
  if (leg) leg.innerHTML = legendItems.map((it) =>
    `<li><span class="swatch${it.dash ? " dash" : it.thick ? " solidthick" : ""}" style="${(it.dash || it.thick) ? "color" : "background"}:${it.color}"></span>${it.label}</li>`).join("");
}

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

function attachCellTips() {
  document.querySelectorAll("td.cell[data-p]").forEach((td) => {
    const tr = td.closest("tr");
    const label = tr ? tr.dataset.label : "";
    const p = parseFloat(td.dataset.p), acc = parseFloat(td.dataset.acc), c = td.dataset.c;
    const htmlStr = `<div class="tt-title">${label}</div>` +
      row("coefficient", c) + row("P(behavior)", fmt2(p)) + row("accuracy", fmt2(acc));
    td.addEventListener("pointerenter", (e) => showTip(e, htmlStr));
    td.addEventListener("pointermove", moveTip);
    td.addEventListener("pointerleave", hideTip);
  });
}

function attachGlossaryTips() {
  document.addEventListener("pointerover", (e) => {
    const t = e.target.closest(".metric[data-tip]");
    if (!t) return;
    const g = GLOSSARY[t.dataset.tip];
    if (!g) return;
    showTip(e, `<div class="tt-title">${g.title}</div><div class="tt-body">${g.body}</div>`);
  });
  document.addEventListener("pointermove", (e) => {
    if (e.target.closest(".metric[data-tip]")) moveTip(e);
  });
  document.addEventListener("pointerout", (e) => {
    if (e.target.closest(".metric[data-tip]")) hideTip();
  });
}

function attachTabs() {
  const dsTabs = [...document.querySelectorAll(".dataset-tab")];
  dsTabs.forEach((btn) => btn.addEventListener("click", () => {
    dsTabs.forEach((b) => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".dataset-panel").forEach((p) => p.classList.toggle("active", p.dataset.ds === btn.dataset.ds));
  }));
  document.querySelectorAll(".dataset-panel").forEach((panel) => {
    const tabs = [...panel.querySelectorAll(".beh-tab")];
    const secs = [...panel.querySelectorAll(".beh-section")];
    tabs.forEach((btn) => btn.addEventListener("click", () => {
      tabs.forEach((b) => b.classList.toggle("active", b === btn));
      secs.forEach((s) => s.classList.toggle("active", s.dataset.beh === btn.dataset.beh));
    }));
  });
}

function renderAll() {
  colorizeCells();
  DATA.datasets.forEach((ds) => {
    ds.behaviors.forEach((beh, i) => lineChart(ds.id, i, beh, ds.coeffs, ds.coherentMax, ds.charts[beh] || []));
  });
}
attachTabs();
attachCellTips();
attachGlossaryTips();
renderAll();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
new MutationObserver(renderAll).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
"""


def build_html(built_datasets, path):
    ds_tabs = "".join(
        f'<button class="dataset-tab{" active" if i == 0 else ""}" data-ds="{d["meta"]["id"]}">{html.escape(d["meta"]["label"])}</button>'
        for i, d in enumerate(built_datasets))

    panels = []
    js_datasets = []
    for i, d in enumerate(built_datasets):
        ds, meta = d["ds"], d["meta"]
        beh_tabs = "".join(
            f'<button class="beh-tab{" active" if j == 0 else ""}" data-beh="{j}">{html.escape(b)}</button>'
            for j, b in enumerate(ds["behaviors"]))
        sections = "".join(build_behavior_section(ds, meta["id"], j, b) for j, b in enumerate(ds["behaviors"]))
        panel = f"""
<div class="dataset-panel{" active" if i == 0 else ""}" data-ds="{meta["id"]}">
  <h2>{html.escape(meta["label"])}</h2>
  <p class="ds-sub">{html.escape(meta["sub"])}</p>
  <div class="tiles">{build_tiles(ds)}</div>
  <section id="{meta['id']}-summary">
    <h2>Summary &mdash; coefficient&rarr;behavior response</h2>
    <p class="hint">Bold <span class="metric" data-tip="source_row">SOURCE</span> row = Llama-1B reference; olive <span class="metric" data-tip="native_row">NATIVE</span> row(s) = target model's own CAA vector; best <span class="metric" data-tip="target_config">translated config</span> per behavior marked &starf;. Hover any dotted metric name for its definition.</p>
    {build_summary_table(ds)}
  </section>
  <section>
    <h2>Per-behavior charts &amp; heatmaps</h2>
    <div class="tabbar beh">{beh_tabs}</div>
    {sections}
  </section>
</div>"""
        panels.append(panel)
        js_datasets.append({
            "id": meta["id"], "behaviors": ds["behaviors"], "coeffs": ds["coeffs"],
            "coherentMax": ds["coherentMax"], "charts": ds["charts"],
        })

    data = {"datasets": js_datasets}

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A/B steering dashboard &mdash; translated vs. native vs. origin</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header>
  <p class="eyebrow">tests_on_translation &middot; CAA steering-vector transport</p>
  <h1>Translated vs. native vs. origin &mdash; one dashboard, every run</h1>
  <p class="lede">
    Every A/B steering-sweep run in this repo, in one page: pick a run, pick a behavior, read the
    chart. <b>SOURCE</b> (dashed black) is the original Llama-3.2-1B CAA vector; <b>NATIVE</b> (solid
    grey, where available) is the target model's own from-scratch CAA vector at that layer &mdash; a
    ceiling independent of translation quality; colored lines are the translated configs being judged.
  </p>
  <p class="hint">Hover any dotted-underline metric name for its definition &mdash; no separate glossary section.</p>
  <div class="tabbar dataset">{ds_tabs}</div>
</header>
{"".join(panels)}
<footer>
  Generated by <span class="mono">ab_dashboard.py</span>, consolidating
  <span class="mono">outputs/ab_eval/{{,new_models/,layer_sweep/,l8_to_l12/}}results.csv</span>
  (produced by <span class="mono">ab_sweep.py</span> + <span class="mono">ab_report.py</span>).
  Colors follow the project data-viz palette; the page themes to your light/dark preference.
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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(_HERE / "outputs" / "ab_eval" / "dashboard.html"))
    ap.add_argument("--method", default="CAA",
                    help="Discovery method to render (default CAA). This dashboard "
                         "shares one coefficient axis across all its curves, which is "
                         "only meaningful within a single method; use method_report.py "
                         "to compare methods.")
    args = ap.parse_args()

    built = []
    for meta in DATASETS:
        results_csv = meta["dir"] / "results.csv"
        if not results_csv.exists():
            print(f"  skip {meta['id']}: {results_csv} not found")
            continue
        rows = load_rows(results_csv, methods=(args.method,))
        if not rows:
            print(f"  skip {meta['id']}: no {args.method} rows in {results_csv}")
            continue
        ds = build_dataset(rows, meta["coherent_max"])
        built.append({"meta": meta, "ds": ds})
        print(f"  {meta['id']}: {len(rows)} rows, {len(ds['configLabels'])} target configs, "
              f"{len(ds['native_keys'])} native configs, {len(ds['behaviors'])} behaviors")

    if not built:
        raise SystemExit("No datasets found -- check the DATASETS paths in ab_dashboard.py")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build_html(built, out)
    print(f"  Wrote: {out}")


if __name__ == "__main__":
    main()
