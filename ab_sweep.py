"""
Full A/B steering sweep across ALL FineWeb translators.

For every translator in outputs/fineweb/, every MWE/CAA behavior, a set of
transport modes (translator-aware: linear/Procrustes checkpoints get a single
procrustes floor; other translators get restore/no_restore), and a wide
coefficient grid, this:

  1. Translates the Llama-3.2-1B CAA steering vector into Llama-3.2-3B space.
  2. Runs the closed-ended (A/B) evaluation on the TARGET model (Llama-3.2-3B)
     with the translated vector.
  3. Runs the same A/B evaluation on the SOURCE model (Llama-3.2-1B) with the
     original vector once per (source layer, behavior), as the reference baseline.

Layer pairs are read from each checkpoint's filename (`.._l{src}__.._l{tgt}__..`):
the source SV is taken from that source layer and the translated SV is injected
at that target layer, so checkpoints fitted with `fit_procrustes.py
--source-layer/--target-layer` flow through unchanged.

Results stream to a JSONL file (one row per scope/translator/norm/behavior/coeff),
so the run is fully resumable: re-running skips combinations already present.

The heavy A/B machinery (steering hook, prompt construction, P(match) metric) is
imported verbatim from ab_comparison.py so this stays faithful to the
activation_engineering evaluation.

Usage:
    conda run -n acteng python ab_sweep.py                       # everything
    conda run -n acteng python ab_sweep.py --device cuda:1 \
        --translators outputs/fineweb/best_translator__*mlp__cosine.pt
    conda run -n acteng python ab_sweep.py --with-source --no-target  # baseline only
    conda run -n acteng python ab_sweep.py --limit 10            # quick smoke test

Then build the comparison tables/plots:
    conda run -n acteng python ab_report.py
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch

# Reuse the faithful A/B machinery and the translation routine.
from ab_comparison import _load_model, run_ab_eval, ABResult
from translate_steering_vector import translate_steering_vector

_HERE = Path(__file__).resolve().parent
_ACTENG = _HERE.parent / "activation_engineering"

# ── Fixed experiment grid ────────────────────────────────────────────────────

SOURCE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
SOURCE_LAYER = 8          # CAA SVs were extracted at layer 8
MODULE = "residual"
METHOD = "CAA"

BEHAVIORS = [
    "coordinate-other-ais",
    "corrigible-neutral-HHH",
    "hallucination",
    "myopic-reward",
    "refusal",
    "survival-instinct",
    "sycophancy",
]

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


# ── Translator checkpoint metadata ───────────────────────────────────────────

@dataclass
class Translator:
    path: Path
    name: str          # checkpoint stem (unique id)
    ttype: str         # mlp | encoder | flow | sae | linear
    loss: str          # mse | cosine | info_nce | procrustes*
    pooling: str       # last | mean
    src_layer: int     # layer the source SV/activations were extracted at
    tgt_layer: int     # layer the translated SV is injected at on the target


_CKPT_RE = re.compile(
    r"best_translator__(?P<src>.+?)__(?P<tgt>.+?)__(?P<type>mlp|encoder|flow|sae|linear)__(?P<loss>.+)$"
)
_LAYER_RE = re.compile(r"_l(\d+)(?:_mean)?$")


def parse_translator(path: Path) -> Translator:
    stem = path.stem  # strips .pt
    m = _CKPT_RE.match(stem)
    if not m:
        raise ValueError(f"Cannot parse translator filename: {path.name}")
    layers = []
    for slug in (m.group("src"), m.group("tgt")):
        lm = _LAYER_RE.search(slug)
        if not lm:
            raise ValueError(f"Cannot parse layer from model slug {slug!r} in {path.name}")
        layers.append(int(lm.group(1)))
    pooling = "mean" if m.group("src").endswith("_mean") else "last"
    return Translator(
        path=path,
        name=stem,
        ttype=m.group("type"),
        loss=m.group("loss"),
        pooling=pooling,
        src_layer=layers[0],
        tgt_layer=layers[1],
    )


def discover_translators(patterns: List[str]) -> List[Translator]:
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in glob.glob(pat))
    # de-dup + stable order
    uniq = sorted({p.resolve() for p in paths})
    return [parse_translator(p) for p in uniq]


# ── Result rows + resumable JSONL ────────────────────────────────────────────

def _aggregate(results: List[ABResult]) -> dict:
    """coeff -> {avg_p_match, accuracy, n} for one eval run."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in results:
        by[r.coefficient].append(r)
    out = {}
    for c, items in by.items():
        out[c] = {
            "avg_p_match": sum(r.behavior_prob for r in items) / len(items),
            "accuracy": sum(1 for r in items if r.is_match) / len(items),
            "n": len(items),
        }
    return out


def _combo_key(row: dict) -> str:
    """Stable identity for a (scope, translator, norm, behavior) eval block."""
    return "|".join(str(row.get(k, "")) for k in
                    ("scope", "translator", "norm_mode", "behavior"))


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


def _emit(scope, translator: Optional[Translator], norm_mode, behavior,
          agg: dict, sv_norm: float, source_layer: int = SOURCE_LAYER) -> List[dict]:
    if translator:
        source_layer = translator.src_layer
    base = {
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

def phase_translate(translators: List[Translator], out_root: Path,
                    behaviors: List[str]) -> dict:
    """Translate every (translator, behavior, transport-mode) SV. Returns nested
    dict of translated sv paths + norms:
    tr[name][mode.label][behavior] = (sv_path, sv_norm)."""
    print("=" * 70)
    print(f"  PHASE 1 — translating {len(translators)} translators × "
          f"{len(behaviors)} behaviors × (translator-aware) transport modes")
    print("=" * 70)
    table = {}
    for tr in translators:
        table[tr.name] = {}
        for mode in transport_modes(tr.ttype):
            table[tr.name][mode.label] = {}
            # unique output root per (translator, mode) avoids path collisions
            tr_out = out_root / "translated" / f"{tr.name}__{mode.label}"
            for behavior in behaviors:
                src_sv = (
                    _ACTENG / "steering_vectors"
                    / SOURCE_MODEL.replace("/", "_")
                    / METHOD / behavior / MODULE
                    / f"layer_{tr.src_layer}" / "sv.pt"
                )
                if not src_sv.exists():
                    print(f"  !! missing source SV: {src_sv}")
                    continue
                translated, out_pt = translate_steering_vector(
                    src_sv, tr.path, output_dir=str(tr_out),
                    norm_mode=mode.norm_mode, apply_bias=mode.apply_bias,
                )
                table[tr.name][mode.label][behavior] = (
                    Path(out_pt), float(translated.norm().item())
                )
    return table


def _load_test_items(behavior: str) -> List[dict]:
    p = _ACTENG / "data" / "CAA_datasets" / "test" / behavior / "test_dataset_ab.json"
    return json.loads(p.read_text())


def phase_source(source_layers, behaviors, coefficients, limit, device,
                 jsonl_path, done):
    """Baseline: original SV on the SOURCE model (Llama-1B), once per
    (source layer, behavior). Each translator's fidelity is judged against the
    baseline of the layer its source vector came from."""
    todo = [(lay, b) for lay in source_layers for b in behaviors
            if _combo_key({"scope": "source", "translator": _source_label(lay),
                           "norm_mode": "", "behavior": b}) not in done]
    if not todo:
        print("\n  PHASE 2 (source baseline) — all layer/behavior combos done, skipping")
        return
    print("\n" + "=" * 70)
    print(f"  PHASE 2 — source baseline on {SOURCE_MODEL}  "
          f"(layers {sorted(source_layers)}, {len(todo)} combos)")
    print("=" * 70)
    model, tok = _load_model_on(SOURCE_MODEL, device)
    try:
        for layer, behavior in todo:
            src_sv = (
                _ACTENG / "steering_vectors" / SOURCE_MODEL.replace("/", "_")
                / METHOD / behavior / MODULE / f"layer_{layer}" / "sv.pt"
            )
            if not src_sv.exists():
                print(f"  !! missing source SV: {src_sv}")
                continue
            sv = torch.load(src_sv, map_location=model.device, weights_only=True)
            items = _load_test_items(behavior)
            print(f"  [source] l{layer} / {behavior}  ({len(items)} items)")
            results = run_ab_eval(model, tok, items, sv, layer, MODULE,
                                  coefficients, limit=limit)
            rows = _emit("source", None, "", behavior, _aggregate(results),
                         float(sv.norm().item()), source_layer=layer)
            _append_rows(jsonl_path, rows)
    finally:
        del model, tok
        torch.cuda.empty_cache()


def phase_target(translators, table, behaviors, coefficients, limit, device,
                 jsonl_path, done):
    """Translated SV on the TARGET model (Llama-3B), all combos."""
    print("\n" + "=" * 70)
    print(f"  PHASE 3 — target eval on {TARGET_MODEL}")
    print("=" * 70)
    model, tok = _load_model_on(TARGET_MODEL, device)
    target_layer = None  # read from first translated sv.json
    try:
        for tr in translators:
            for mode in transport_modes(tr.ttype):
                for behavior in behaviors:
                    key = _combo_key({"scope": "target", "translator": tr.name,
                                      "norm_mode": mode.label, "behavior": behavior})
                    if key in done:
                        continue
                    entry = table.get(tr.name, {}).get(mode.label, {}).get(behavior)
                    if entry is None:
                        continue
                    sv_path, sv_norm = entry
                    sv = torch.load(sv_path, map_location=model.device,
                                    weights_only=True)
                    # inject layer comes from the translated SV's own path (set by
                    # the translator's config, i.e. the checkpoint's target layer)
                    tlayer = _target_layer_of(sv_path)
                    items = _load_test_items(behavior)
                    print(f"  [target] {tr.name} / {mode.label} / {behavior} "
                          f"(L{tlayer}, |sv|={sv_norm:.3f})")
                    results = run_ab_eval(model, tok, items, sv, tlayer, MODULE,
                                          coefficients, limit=limit)
                    rows = _emit("target", tr, mode.label, behavior,
                                 _aggregate(results), sv_norm)
                    _append_rows(jsonl_path, rows)
                    done.add(key)
    finally:
        del model, tok
        torch.cuda.empty_cache()


def _target_layer_of(sv_path: Path) -> int:
    """layer_{idx} is the parent's parent dir name of sv.pt."""
    for part in sv_path.parts:
        if part.startswith("layer_"):
            return int(part.split("_", 1)[1])
    return SOURCE_LAYER


# ── Model loading honoring an explicit device ────────────────────────────────

def _load_model_on(model_name: str, device: str):
    """Like ab_comparison._load_model but onto an explicit device."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache = os.getenv("HF_CACHE_DIR")
    print(f"  Loading {model_name} on {device} …")
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, cache_dir=cache
    ).to(device)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
    return model, tok


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--translators", nargs="+",
                   default=[str(_HERE / "outputs" / "fineweb" / "best_translator__*.pt")],
                   help="Glob(s) for translator checkpoints (default: all fineweb best_*)")
    p.add_argument("--behaviors", nargs="+", default=BEHAVIORS)
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
                for behavior in args.behaviors:
                    sv_pt = (tr_out / "steering_vectors"
                             / TARGET_MODEL.replace("/", "_") / METHOD / behavior
                             / MODULE / f"layer_{tgt_layer}" / "sv.pt")
                    if sv_pt.exists():
                        v = torch.load(sv_pt, map_location="cpu", weights_only=True)
                        table[tr.name][mode.label][behavior] = (sv_pt, float(v.norm()))
    else:
        table = phase_translate(translators, out_root, args.behaviors)

    done = _load_done(jsonl_path)

    if args.with_source:
        source_layers = sorted({tr.src_layer for tr in translators}) or [SOURCE_LAYER]
        phase_source(source_layers, args.behaviors, args.coefficients, args.limit,
                     args.device, jsonl_path, done)

    if args.with_target:
        phase_target(translators, table, args.behaviors, args.coefficients,
                     args.limit, args.device, jsonl_path, done)

    print(f"\n  Done. Rows in {jsonl_path}. Build tables with: "
          f"conda run -n acteng python ab_report.py --results {jsonl_path}")


if __name__ == "__main__":
    main()
