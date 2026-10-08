"""
Build a generic pivot-style A/B dashboard: pick a metric, pick what to put on
the x-axis, pick ONE dimension to compare as colored lines, fix the rest with
dropdowns, and check/uncheck which values of the compare dimension to plot.

This mirrors the design of activation_engineering/evaluation/build_dashboard.py
+ dashboard_template.html (the reference the user pointed to) rather than the
fixed dataset/behavior tabs of ab_dashboard.py: instead of pre-deciding what's
shown, every A/B result across every sweep run in this repo becomes one flat
pool of "records" -- (target_model, config, behavior, layer) -> per-coefficient
P(behavior)/accuracy curves -- and the page lets you pivot freely.

"config" folds translator_type/loss/pooling/norm_mode into one label (the
direct analog of the reference's "method" dimension), plus four special
pseudo-configs:
  SOURCE  -- the original Llama-3.2-1B CAA vector. Its curve doesn't depend on
             the target model, so it's replicated across every (model, layer)
             pair that actually has data -- same trick the reference used to
             spread its all-layer GAS method across every concrete layer --
             so "compare vs. origin" is always available regardless of filters.
  NATIVE  -- the target model's own from-scratch CAA vector at that layer,
             where such a run exists (present for the new_models and
             layer_sweep sweeps).
  LOBO/plain, LOBO/whiten -- the leave-one-behavior-out study (LLaDA-8B only,
             outputs/ab_eval/llada_lobo, produced by fit_procrustes_lobo.py):
             the Procrustes map was fit on the other 6 behaviors' CAA
             difference vectors and evaluated on the excluded one. Unlike
             every other config, "config" and "behavior" are coupled here --
             each LOBO/{variant} config is really a different translator per
             behavior, the one that excluded it -- so the held-out behavior is
             just whatever the Behavior selector is set to. Compare against
             the ordinary translated config at the same (model, layer,
             mean-pooling) to see what fitting on the behavior itself bought
             you -- MAIN/plain and MAIN/whiten below are a shortcut for
             exactly that comparison.
  MAIN/plain, MAIN/whiten -- LOBO_MODEL's own mean-pooling translated config
             (fit on generic FineWeb sentences, never touching ANY behavior's
             diff vectors), aliased under a name that sits next to LOBO/plain
             and LOBO/whiten. Pure duplicate of data already present under the
             verbose translator_type/loss/pooling/norm_mode label -- not a
             separate data source -- kept only so "LOBO vs. its upper-bound
             baseline" is a 2-click comparison instead of requiring you to
             know that label.

One discovery method at a time (--method, default CAA): the page's x-axis is the
raw coefficient, which means a different physical dose per method (CAA vectors
carry their own magnitude, RepE/GCAV are unit-norm), so plotting methods as
sibling lines here would compare them on an axis they do not share. The
cross-method view is method_report.py, which reads on dose = coeff * sv_norm.

Usage:
    conda run -n acteng python ab_pivot_dashboard.py
    conda run -n acteng python ab_pivot_dashboard.py --method RepE
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ab_dashboard import DATASETS  # the dashboard-run registry lives with its CLI
from acttrans.evaluation.results_io import (
    load_rows_csv as load_rows,
    parse_native_model,
    parse_target_pooling,
    parse_translator_models,
)

_HERE = Path(__file__).resolve().parent
TEMPLATE = _HERE / "ab_pivot_template.html"
OUT = _HERE / "outputs" / "ab_eval" / "pivot_dashboard.html"
COHERENT_MAX = 5.0
LOBO_CSV = _HERE / "outputs" / "ab_eval" / "llada_lobo" / "results.csv"
LOBO_MODEL = "LLaDA-8B-Instruct"


def collect_file(rows):
    """One run's results.csv rows -> (target/native records[], source curves per behavior)."""
    tgt_models = [parse_translator_models(r["translator"])[1] for r in rows if r["scope"] == "target"]
    tgt_models = [m for m in tgt_models if m]
    dominant = Counter(tgt_models).most_common(1)[0][0] if tgt_models else None

    blocks = defaultdict(list)
    for r in rows:
        blocks[(r["scope"], r.get("translator", ""), r.get("norm_mode", ""), r["behavior"])].append(r)

    records, source_curves = [], {}
    for (scope, tr, nm, beh), rs in blocks.items():
        rs = sorted(rs, key=lambda r: r["coefficient"])
        rec = {
            "coeffs": [round(r["coefficient"], 4) for r in rs],
            "behavior_prob": [round(r["avg_p_match"], 4) for r in rs],
            "accuracy": [round(r["accuracy"], 4) for r in rs],
            "n": max((int(r["n"]) for r in rs if r.get("n")), default=0),
        }
        ex = rs[0]

        if scope == "source":
            if beh not in source_curves or len(rec["coeffs"]) > len(source_curves[beh]["coeffs"]):
                source_curves[beh] = rec
            continue

        if scope == "native":
            model = parse_native_model(tr, dominant)
            config = "NATIVE"
        else:  # target
            model = parse_translator_models(tr)[1] or dominant
            # TARGET-side pooling, read off the translator name — the `pooling`
            # column is the source side's mode, which is "last" for every arm of a
            # diffusion run and would collapse mean and mask into one label. Since
            # the record key below is (model, config, behavior, layer), that
            # collapse would drop one whole arm as a duplicate rather than merely
            # mislabel it. See results_io.parse_target_pooling.
            pooling = parse_target_pooling(tr, default=ex.get("pooling", "") or "last")
            config = f'{ex.get("translator_type", "")}/{ex.get("loss", "")}/{pooling}/{nm}'
        if not model:
            continue

        rec.update({"model": model, "config": config, "behavior": beh, "layer": ex["target_layer"]})
        records.append(rec)
    return records, source_curves


def collect_lobo(rows):
    """outputs/ab_eval/llada_lobo/results.csv rows -> (LOBO target records[],
    source curves per behavior). Tagged with the LOBO/plain|LOBO/whiten
    pseudo-config described in the module docstring, rather than run through
    collect_file's generic translator_type/loss/pooling/norm_mode label --
    each LOBO/{variant} config here is a different translator per behavior
    (the one that excluded it), which that generic label has no way to say."""
    blocks = defaultdict(list)
    for r in rows:
        blocks[(r["scope"], r.get("translator", ""), r["behavior"])].append(r)

    records, source_curves = [], {}
    for (scope, tr, beh), rs in blocks.items():
        rs = sorted(rs, key=lambda r: r["coefficient"])
        rec = {
            "coeffs": [round(r["coefficient"], 4) for r in rs],
            "behavior_prob": [round(r["avg_p_match"], 4) for r in rs],
            "accuracy": [round(r["accuracy"], 4) for r in rs],
            "n": max((int(r["n"]) for r in rs if r.get("n")), default=0),
        }
        ex = rs[0]

        if scope == "source":
            if beh not in source_curves or len(rec["coeffs"]) > len(source_curves[beh]["coeffs"]):
                source_curves[beh] = rec
            continue
        if scope != "target":
            continue

        loss = ex.get("loss", "")
        if "_lobo-" not in loss:
            continue  # not actually a LOBO fit -- don't mislabel it as one
        config = "LOBO/whiten" if "whiten" in loss else "LOBO/plain"
        rec.update({"model": LOBO_MODEL, "config": config, "behavior": beh, "layer": ex["target_layer"]})
        records.append(rec)
    return records, source_curves


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", default="CAA",
                    help="Discovery method to render (default CAA). One method per "
                         "page — see the module docstring for why.")
    ap.add_argument("--steering-mode", default="additive", choices=["additive", "projection"],
                    help="Application mode to render (default additive). One mode per "
                         "page, same reason as --method: the whole page's coefficient "
                         "framing (coherentMax, 'steering coefficient' axis) assumes one "
                         "physical dosing scheme, and an apply_projection_ablation alpha "
                         "is not that. Rows with no steering_mode field predate the "
                         "dimension and are treated as additive.")
    args = ap.parse_args()

    all_records, source_curves, seen = [], {}, set()

    for meta in DATASETS:
        results_csv = meta["dir"] / "results.csv"
        if not results_csv.exists():
            print(f"  skip {meta['id']}: {results_csv} not found")
            continue
        rows = load_rows(results_csv, methods=(args.method,))
        rows = [r for r in rows if r["steering_mode"] == args.steering_mode]
        if not rows:
            print(f"  skip {meta['id']}: no {args.method} rows in {results_csv}")
            continue
        recs, srcs = collect_file(rows)
        for r in recs:
            key = (r["model"], r["config"], r["behavior"], r["layer"])
            if key in seen:
                continue
            seen.add(key)
            all_records.append(r)
        for beh, curve in srcs.items():
            if beh not in source_curves or len(curve["coeffs"]) > len(source_curves[beh]["coeffs"]):
                source_curves[beh] = curve
        print(f"  {meta['id']}: {len(rows)} rows -> {len(recs)} target/native records")

    if LOBO_CSV.exists():
        lobo_rows = load_rows(LOBO_CSV, methods=(args.method,))
        lobo_rows = [r for r in lobo_rows if r["steering_mode"] == args.steering_mode]
        lobo_recs, lobo_srcs = collect_lobo(lobo_rows)
        for r in lobo_recs:
            key = (r["model"], r["config"], r["behavior"], r["layer"])
            if key in seen:
                continue
            seen.add(key)
            all_records.append(r)
        for beh, curve in lobo_srcs.items():
            if beh not in source_curves or len(curve["coeffs"]) > len(source_curves[beh]["coeffs"]):
                source_curves[beh] = curve
        print(f"  llada_lobo: {len(lobo_rows)} rows -> {len(lobo_recs)} LOBO target records")
    else:
        print(f"  skip llada_lobo: {LOBO_CSV} not found")

    # MAIN/plain, MAIN/whiten: a shortcut alias of LOBO_MODEL's own mean-pooling
    # translated config (the one already present under its verbose
    # translator_type/loss/pooling/norm_mode label), so it sits next to
    # LOBO/plain and LOBO/whiten for a 2-click "what did holding the behavior
    # out cost you" comparison. Pure duplicate of data already in all_records,
    # not a new data source.
    n_main = 0
    for r in list(all_records):
        if r["model"] != LOBO_MODEL:
            continue
        parts = r["config"].split("/")
        if len(parts) != 4 or parts[2] != "mean":
            continue
        alias_config = "MAIN/whiten" if "whiten" in parts[1] else "MAIN/plain"
        key = (r["model"], alias_config, r["behavior"], r["layer"])
        if key in seen:
            continue
        seen.add(key)
        alias = dict(r, config=alias_config)
        all_records.append(alias)
        n_main += 1
    if n_main:
        print(f"  {n_main} MAIN/plain+MAIN/whiten alias records added")

    # replicate SOURCE into every (model, layer) pair that actually has data
    pairs = sorted({(r["model"], r["layer"]) for r in all_records})
    for model, layer in pairs:
        for beh, curve in source_curves.items():
            key = (model, "SOURCE", beh, layer)
            if key in seen:
                continue
            seen.add(key)
            rec = dict(curve)
            rec.update({"model": model, "config": "SOURCE", "behavior": beh, "layer": layer})
            all_records.append(rec)

    models = sorted({r["model"] for r in all_records})
    configs = sorted({r["config"] for r in all_records})
    behaviors = sorted({r["behavior"] for r in all_records})
    layers = sorted({r["layer"] for r in all_records})

    # curate a readable default selection instead of checking all ~90 configs:
    # SOURCE + NATIVE + the translated configs that reach the highest accuracy
    # anywhere (a crude effect-size proxy, cheap to compute from the records
    # already in hand -- the dashboard itself lets you pick any others).
    scores = defaultdict(list)
    for r in all_records:
        if r["config"] in ("SOURCE", "NATIVE") or not r["accuracy"]:
            continue
        scores[r["config"]].append(max(r["accuracy"]))
    ranked = sorted(scores, key=lambda c: -(sum(scores[c]) / len(scores[c])))
    default_configs = [c for c in (["SOURCE", "NATIVE"] + ranked[:6]) if c in configs]

    payload = {
        "n_series": len(all_records),
        "coherentMax": COHERENT_MAX,
        "dims": {"models": models, "configs": configs, "behaviors": behaviors, "layers": layers},
        "defaultConfigs": default_configs,
        "records": all_records,
    }

    template = TEMPLATE.read_text()
    doc = template.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":")))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc)

    print(f"  {len(all_records)} records: {len(models)} target models, {len(configs)} configs, "
          f"{len(behaviors)} behaviors, {len(layers)} layers")
    print(f"  Default compare selection: {default_configs}")
    print(f"  Wrote: {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
