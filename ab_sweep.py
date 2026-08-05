"""
Full A/B steering sweep across ALL FineWeb translators.

For every translator in outputs/fineweb/, every discovery method (--methods:
CAA / RepE / GCAV), every MWE behavior, a set of transport modes
(translator-aware: linear/Procrustes checkpoints get a single procrustes floor;
other translators get restore/no_restore), and a wide coefficient grid, this:

  1. Translates the Llama-3.2-1B steering vector into Llama-3.2-3B space.
  2. Runs the closed-ended (A/B) evaluation on the TARGET model (Llama-3.2-3B)
     with the translated vector.
  3. Runs the same A/B evaluation on the SOURCE model (Llama-3.2-1B) with the
     original vector once per (method, source layer, behavior), as the reference
     baseline.

Layer pairs are read from each checkpoint's filename (`.._l{src}__.._l{tgt}__..`):
the source SV is taken from that source layer and the translated SV is injected
at that target layer, so checkpoints fitted with `fit_procrustes.py
--source-layer/--target-layer` flow through unchanged.

Results stream to a JSONL file (one row per
method/scope/translator/norm/behavior/coeff), so the run is fully resumable:
re-running skips combinations already present. Rows written before methods were a
dimension carry no `method` field and are treated as CAA, so old results files
resume correctly.

COMPARING METHODS: CAA vectors carry their own (behavior-dependent) magnitude
while RepE and GCAV are unit-norm by construction, so the same `coefficient` is
a different physical dose per method. Cross-method reading is done on
dose = coefficient * sv_norm by method_report.py; a raw-coefficient comparison
across methods is not meaningful.

The heavy A/B machinery (steering hook, prompt construction, P(match) metric)
lives in acttrans.evaluation.ab_eval — the same code ab_comparison.py runs — so
this stays faithful to the activation_engineering evaluation. All three methods
use an additive hook there, which is what ab_eval implements.

Usage:
    conda run -n acteng python ab_sweep.py                       # CAA, everything
    conda run -n acteng python ab_sweep.py --methods CAA RepE GCAV --with-native
    conda run -n acteng python ab_sweep.py --device cuda:1 \
        --translators outputs/fineweb/best_translator__*mlp__cosine.pt
    conda run -n acteng python ab_sweep.py --with-source --no-target  # baseline only
    conda run -n acteng python ab_sweep.py --limit 10            # quick smoke test

Then build the comparison tables/plots:
    conda run -n acteng python ab_report.py                      # per-method report
    conda run -n acteng python method_report.py                  # cross-method
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

# The faithful A/B machinery, the shared experiment grid and the translation routine.
from acttrans.constants import (
    BEHAVIORS,
    LAYER,
    METHOD,
    METHODS,
    MODULE,
    SOURCE_MODEL,
    TARGET_MODEL,
)
from acttrans.evaluation.ab_eval import aggregate_by_coefficient, run_ab_eval
from acttrans.utils.checkpoints import TranslatorInfo, discover_translators
from acttrans.utils.hf import load_model_and_tokenizer
from acttrans.utils.paths import sv_path
from translate_steering_vector import translate_steering_vector

_HERE = Path(__file__).resolve().parent
_ACTENG = _HERE.parent / "activation_engineering"

SOURCE_LAYER = LAYER      # CAA SVs were extracted at layer 8

# -1000 .. -0.25, 0, 0.25 .. 1000  (symmetric grid requested by the user)
_MAG = [1000, 500, 250, 100, 50, 10, 5, 2, 1, 0.75, 0.50, 0.25]
COEFFICIENTS = sorted({-m for m in _MAG} | {0.0} | {m for m in _MAG})

DEFAULT_OUT = _HERE / "outputs" / "ab_eval"


# ── Translator-aware transport modes ─────────────────────────────────────────

@dataclass
class TransportMode:
    label: str        # stable id: JSONL norm_mode column, output subdir, resume key
    norm_mode: str    # passed to translate_steering_vector
    apply_bias: bool


def transport_modes(ttype: str) -> List[TransportMode]:
    """Transport modes to evaluate for a given translator type.

    linear/Procrustes checkpoints are transported faithfully as the closed-form
    Procrustes floor: bias-free direction transport, scaled by the fitted scale
    `s` stored in the checkpoint. The gradient-trained non-linear translators
    keep the existing restore / no_restore (="none") magnitude handling. The
    "no_restore" label maps to norm_mode "none" to preserve JSONL/report
    continuity with prior runs.
    """
    if ttype == "linear":
        return [TransportMode("procrustes", "procrustes", False)]
    return [
        TransportMode("restore", "restore", False),
        TransportMode("no_restore", "none", False),
    ]


# ── Result rows + resumable JSONL ────────────────────────────────────────────
# (Checkpoint metadata parsing lives in acttrans.utils.checkpoints.)

def _combo_key(row: dict) -> str:
    """Stable identity for a (method, scope, translator, norm, behavior) eval block.

    Rows written before methods were a dimension carry no `method` field and are
    all CAA, so a missing/empty method resolves to CAA — that keeps every
    pre-existing CAA row a valid resume key."""
    method = row.get("method") or METHOD
    return "|".join([method] + [str(row.get(k, "")) for k in
                                ("scope", "translator", "norm_mode", "behavior")])


def _load_done(jsonl_path: Path) -> set:
    done = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_combo_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return done


def _append_rows(jsonl_path: Path, rows: List[dict]):
    with open(jsonl_path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _source_label(layer: int) -> str:
    """Identity of a source-baseline block in the `translator` field. The default
    layer keeps the empty label so existing JSONL rows stay valid resume keys;
    other layers get a distinct label so their blocks never collide."""
    return "" if layer == SOURCE_LAYER else f"source_l{layer}"


def _emit(scope, translator: Optional[TranslatorInfo], norm_mode, behavior,
          agg: dict, sv_norm: float, source_layer: int = SOURCE_LAYER,
          method: str = METHOD) -> List[dict]:
    if translator:
        source_layer = translator.src_layer
    base = {
        "method": method,                                 # "CAA" | "RepE" | "GCAV"
        "scope": scope,                                   # "source" | "target"
        "translator": translator.name if translator else _source_label(source_layer),
        "translator_type": translator.ttype if translator else "",
        "loss": translator.loss if translator else "",
        "pooling": translator.pooling if translator else "",
        "norm_mode": norm_mode,                    # "restore"|"no_restore"|"procrustes"|"" (source)
        "behavior": behavior,
        "sv_norm": sv_norm,
        "source_layer": source_layer,                     # layer the source SV came from
        "target_layer": translator.tgt_layer if translator else None,  # inject layer
    }
    rows = []
    for coeff in sorted(agg):
        rows.append({**base, "coefficient": coeff, **agg[coeff]})
    return rows


# ── Phases ───────────────────────────────────────────────────────────────────

def phase_translate(translators: List[TranslatorInfo], out_root: Path,
                    behaviors: List[str], methods: List[str]) -> dict:
    """Translate every (translator, transport-mode, method, behavior) SV. Returns
    nested dict of translated sv paths + norms:
    tr[name][mode.label][method][behavior] = (sv_path, sv_norm).

    The method is a path component inside the tree written by
    translate_steering_vector (it is recovered from the source path), so several
    methods share one (translator, mode) output root without colliding."""
    print("=" * 70)
    print(f"  PHASE 1 — translating {len(translators)} translators × "
          f"{len(methods)} methods × {len(behaviors)} behaviors × "
          f"(translator-aware) transport modes")
    print("=" * 70)
    table = {}
    for tr in translators:
        table[tr.name] = {}
        for mode in transport_modes(tr.ttype):
            table[tr.name][mode.label] = {}
            # unique output root per (translator, mode) avoids path collisions
            tr_out = out_root / "translated" / f"{tr.name}__{mode.label}"
            for method in methods:
                table[tr.name][mode.label][method] = {}
                for behavior in behaviors:
                    src_sv = sv_path(_ACTENG, SOURCE_MODEL, method, behavior,
                                     MODULE, tr.src_layer)
                    if not src_sv.exists():
                        print(f"  !! missing source SV: {src_sv}")
                        continue
                    translated, out_pt = translate_steering_vector(
                        src_sv, tr.path, output_dir=str(tr_out),
                        norm_mode=mode.norm_mode, apply_bias=mode.apply_bias,
                    )
                    table[tr.name][mode.label][method][behavior] = (
                        Path(out_pt), float(translated.norm().item())
                    )
    return table


def _load_test_items(behavior: str) -> List[dict]:
    p = _ACTENG / "data" / "CAA_datasets" / "test" / behavior / "test_dataset_ab.json"
    return json.loads(p.read_text())


def phase_source(source_layers, behaviors, coefficients, limit, device,
                 jsonl_path, done, methods):
    """Baseline: original SV on the SOURCE model (Llama-1B), once per
    (method, source layer, behavior). Each translator's fidelity is judged
    against the baseline of the method and layer its source vector came from."""
    todo = [(m, lay, b) for m in methods for lay in source_layers for b in behaviors
            if _combo_key({"method": m, "scope": "source",
                           "translator": _source_label(lay),
                           "norm_mode": "", "behavior": b}) not in done]
    if not todo:
        print("\n  PHASE 2 (source baseline) — all method/layer/behavior combos done, skipping")
        return
    print("\n" + "=" * 70)
    print(f"  PHASE 2 — source baseline on {SOURCE_MODEL}  "
          f"(methods {methods}, layers {sorted(source_layers)}, {len(todo)} combos)")
    print("=" * 70)
    model, tok = _load_model_on(SOURCE_MODEL, device)
    try:
        for method, layer, behavior in todo:
            src_sv = sv_path(_ACTENG, SOURCE_MODEL, method, behavior, MODULE, layer)
            if not src_sv.exists():
                print(f"  !! missing source SV: {src_sv}")
                continue
            sv = torch.load(src_sv, map_location=model.device, weights_only=True)
            items = _load_test_items(behavior)
            print(f"  [source] {method} l{layer} / {behavior}  ({len(items)} items, "
                  f"|sv|={sv.float().norm().item():.3f})")
            results = run_ab_eval(model, tok, items, sv, layer, MODULE,
                                  coefficients, limit=limit)
            rows = _emit("source", None, "", behavior, aggregate_by_coefficient(results),
                         float(sv.float().norm().item()), source_layer=layer,
                         method=method)
            _append_rows(jsonl_path, rows)
            done.add(_combo_key(rows[0]))
    finally:
        del model, tok
        torch.cuda.empty_cache()


def phase_target(translators, table, behaviors, coefficients, limit, device,
                 jsonl_path, done, methods):
    """Translated SV on each checkpoint's OWN target model, all combos.

    Translators are grouped by their target model (read from the checkpoint
    config, not assumed to be Llama-3B) so cross-family sweeps — e.g. Llama-1B ->
    Qwen and Llama-1B -> Gemma checkpoints in one glob — each load their real
    target model once. The single-target Llama-3B case is just a one-group run,
    so prior behavior is unchanged."""
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for tr in translators:
        groups[_target_model_of(tr)].append(tr)

    print("\n" + "=" * 70)
    print(f"  PHASE 3 — target eval across {len(groups)} target model(s): "
          f"{', '.join(sorted(groups))}")
    print("=" * 70)
    for target_model_name in sorted(groups):
        trs = groups[target_model_name]
        print(f"\n  -- target model {target_model_name}  ({len(trs)} translators) --")
        model, tok = _load_model_on(target_model_name, device)
        try:
            for tr in trs:
                for mode in transport_modes(tr.ttype):
                    for method in methods:
                        for behavior in behaviors:
                            key = _combo_key({"method": method, "scope": "target",
                                              "translator": tr.name,
                                              "norm_mode": mode.label,
                                              "behavior": behavior})
                            if key in done:
                                continue
                            entry = (table.get(tr.name, {}).get(mode.label, {})
                                     .get(method, {}).get(behavior))
                            if entry is None:
                                continue
                            sv_file, sv_norm = entry
                            sv = torch.load(sv_file, map_location=model.device,
                                            weights_only=True)
                            # inject layer comes from the translated SV's own path (set by
                            # the translator's config, i.e. the checkpoint's target layer)
                            tlayer = _target_layer_of(sv_file)
                            items = _load_test_items(behavior)
                            print(f"  [target] {method} / {tr.name} / {mode.label} / "
                                  f"{behavior} (L{tlayer}, |sv|={sv_norm:.3f})")
                            results = run_ab_eval(model, tok, items, sv, tlayer, MODULE,
                                                  coefficients, limit=limit)
                            rows = _emit("target", tr, mode.label, behavior,
                                         aggregate_by_coefficient(results), sv_norm,
                                         method=method)
                            _append_rows(jsonl_path, rows)
                            done.add(key)
        finally:
            del model, tok
            torch.cuda.empty_cache()


def _native_label(layer: int, model_name: Optional[str] = None) -> str:
    """Identity of a native-baseline block in the `translator` field. Model-
    qualified when a model is given, so native baselines for different target
    models (e.g. Qwen native_l12 vs Gemma native_l12) never share a resume key.
    The unqualified form is kept for the single-model Llama-3B layer_sweep run."""
    if model_name is None:
        return f"native_l{layer}"
    return f"native_{model_name.split('/')[-1]}_l{layer}"


def phase_native(translators, behaviors, coefficients, limit, device,
                 jsonl_path, done, methods):
    """Upper-bound baseline: the natively-extracted TARGET-model SV at each target
    layer, evaluated on the TARGET model itself (no transport). This is the
    reference a translated vector should be judged against — it lets us report
    how much of native steering the translated vector recovers, on the same model
    and layer, rather than only fidelity to the SOURCE-model curve.

    Like phase_target, translators are grouped by their OWN target model (read
    from each checkpoint config) and the swept target layers are taken per model,
    so a mixed Qwen+Gemma glob evaluates native SVs on the right model each."""
    from collections import defaultdict
    groups: dict = defaultdict(set)
    for tr in translators:
        groups[_target_model_of(tr)].add(tr.tgt_layer)

    print("\n" + "=" * 70)
    print(f"  PHASE — native baseline across {len(groups)} target model(s): "
          f"{', '.join(sorted(groups))}")
    print("=" * 70)
    for target_model_name in sorted(groups):
        layers = sorted(groups[target_model_name])
        todo = [(m, lay, b) for m in methods for lay in layers for b in behaviors
                if _combo_key({"method": m, "scope": "native",
                               "translator": _native_label(lay, target_model_name),
                               "norm_mode": "", "behavior": b}) not in done]
        if not todo:
            print(f"  -- {target_model_name}: all method/layer/behavior combos done, skipping")
            continue
        print(f"\n  -- native baseline on {target_model_name} "
              f"(methods {methods}, layers {layers}, {len(todo)} combos) --")
        model, tok = _load_model_on(target_model_name, device)
        try:
            for method, layer, behavior in todo:
                nsv = sv_path(_ACTENG, target_model_name, method, behavior, MODULE, layer)
                if not nsv.exists():
                    print(f"  !! missing native SV: {nsv}")
                    continue
                sv = torch.load(nsv, map_location=model.device, weights_only=True)
                items = _load_test_items(behavior)
                print(f"  [native] {method} / {target_model_name.split('/')[-1]} l{layer} / "
                      f"{behavior}  ({len(items)} items, "
                      f"|sv|={sv.float().norm().item():.3f})")
                results = run_ab_eval(model, tok, items, sv, layer, MODULE,
                                      coefficients, limit=limit)
                agg = aggregate_by_coefficient(results)
                base = {
                    "method": method,
                    "scope": "native",
                    "translator": _native_label(layer, target_model_name),
                    "translator_type": "native",
                    "loss": "", "pooling": "", "norm_mode": "",
                    "behavior": behavior,
                    "sv_norm": float(sv.float().norm().item()),
                    "source_layer": None,
                    "target_layer": layer,
                }
                rows = [{**base, "coefficient": c, **agg[c]} for c in sorted(agg)]
                _append_rows(jsonl_path, rows)
                done.add(_combo_key(base))
        finally:
            del model, tok
            torch.cuda.empty_cache()


def _target_layer_of(sv_file: Path) -> int:
    """layer_{idx} is the parent's parent dir name of sv.pt. Scan from the end
    and require an integer suffix so an ancestor output dir like 'layer_sweep'
    is ignored (only the real `layer_<int>` component matches)."""
    for part in reversed(sv_file.parts):
        if part.startswith("layer_") and part.split("_", 1)[1].isdigit():
            return int(part.split("_", 1)[1])
    return SOURCE_LAYER


# ── Model loading honoring an explicit device ────────────────────────────────

_TGT_MODEL_CACHE: dict = {}


def _target_model_of(tr: TranslatorInfo) -> str:
    """Target model name a translator maps INTO, read from its checkpoint config
    (cached per path). The filename only encodes layers, so the model identity
    comes from the saved config — this is what lets a mixed-family glob evaluate
    each checkpoint on its real target model."""
    key = str(tr.path)
    if key not in _TGT_MODEL_CACHE:
        ck = torch.load(tr.path, map_location="cpu", weights_only=False)
        _TGT_MODEL_CACHE[key] = ck["config"]["target_model"]["name"]
    return _TGT_MODEL_CACHE[key]


def _load_model_on(model_name: str, device: str):
    print(f"  Loading {model_name} on {device} …")
    return load_model_and_tokenizer(model_name, device)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--translators", nargs="+",
                   default=[str(_HERE / "outputs" / "fineweb" / "best_translator__*.pt")],
                   help="Glob(s) for translator checkpoints (default: all fineweb best_*)")
    p.add_argument("--behaviors", nargs="+", default=BEHAVIORS)
    p.add_argument("--methods", nargs="+", default=[METHOD], choices=list(METHODS),
                   metavar="METHOD",
                   help=f"Steering-vector discovery methods to sweep (default: {METHOD}). "
                        f"Choices: {', '.join(METHODS)}. Every row carries a `method` "
                        "column and the resume key is method-aware, so methods can share "
                        "one results file. NOTE: CAA vectors carry their own magnitude "
                        "while RepE/GCAV are unit-norm, so compare methods on "
                        "dose = coefficient * sv_norm (see method_report.py).")
    p.add_argument("--coefficients", type=float, nargs="+", default=COEFFICIENTS)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap test items per coeff (quick runs)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--results-name", default="results.jsonl",
                   help="JSONL filename inside out-dir (use shards for parallel runs)")
    p.add_argument("--with-source", dest="with_source", action="store_true",
                   default=True, help="Run the source-model baseline (default on)")
    p.add_argument("--no-source", dest="with_source", action="store_false")
    p.add_argument("--no-target", dest="with_target", action="store_false",
                   default=True, help="Skip the target-model eval")
    p.add_argument("--with-native", dest="with_native", action="store_true",
                   default=False,
                   help="Also eval the natively-extracted TARGET-model SV at each "
                        "target layer (upper-bound baseline for how much native "
                        "steering the translated vector recovers). Off by default.")
    p.add_argument("--skip-translate", action="store_true",
                   help="Assume translated SVs already exist on disk")
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / args.results_name

    translators = discover_translators(args.translators)
    if not translators:
        raise SystemExit(f"No translators matched: {args.translators}")

    print(f"  Translators ({len(translators)}):")
    for tr in translators:
        print(f"    - {tr.ttype:8s} {tr.loss:8s} {tr.pooling:4s} "
              f"l{tr.src_layer}->l{tr.tgt_layer}  {tr.name}")
    print(f"  Methods     : {args.methods}")
    print(f"  Behaviors   : {args.behaviors}")
    print(f"  Coefficients: {args.coefficients}")
    print(f"  Results     : {jsonl_path}")
    print(f"  Device      : {args.device}")

    # Phase 1 — translate (idempotent, cheap)
    if args.skip_translate:
        # rebuild the path table without re-translating
        table = {}
        for tr in translators:
            table[tr.name] = {}
            # target layer comes from the translator's own config, not SOURCE_LAYER
            ck = torch.load(tr.path, map_location="cpu", weights_only=False)
            tgt_layer = ck["config"]["target_model"]["layer"]
            for mode in transport_modes(tr.ttype):
                table[tr.name][mode.label] = {}
                tr_out = out_root / "translated" / f"{tr.name}__{mode.label}"
                for method in args.methods:
                    table[tr.name][mode.label][method] = {}
                    for behavior in args.behaviors:
                        sv_pt = sv_path(tr_out, TARGET_MODEL, method, behavior,
                                        MODULE, tgt_layer)
                        if sv_pt.exists():
                            v = torch.load(sv_pt, map_location="cpu", weights_only=True)
                            table[tr.name][mode.label][method][behavior] = (
                                sv_pt, float(v.float().norm())
                            )
    else:
        table = phase_translate(translators, out_root, args.behaviors, args.methods)

    done = _load_done(jsonl_path)

    if args.with_source:
        source_layers = sorted({tr.src_layer for tr in translators}) or [SOURCE_LAYER]
        phase_source(source_layers, args.behaviors, args.coefficients, args.limit,
                     args.device, jsonl_path, done, args.methods)

    if args.with_target:
        phase_target(translators, table, args.behaviors, args.coefficients,
                     args.limit, args.device, jsonl_path, done, args.methods)

    if args.with_native:
        phase_native(translators, args.behaviors, args.coefficients,
                     args.limit, args.device, jsonl_path, done, args.methods)

    print(f"\n  Done. Rows in {jsonl_path}. Build tables with: "
          f"conda run -n acteng python ab_report.py --results {jsonl_path}")


if __name__ == "__main__":
    main()
