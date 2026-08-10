"""Loading and parsing of A/B sweep results, shared by the report + dashboards.

``ab_sweep.py`` streams one JSONL row per (method, scope, translator, norm,
behavior, coefficient); ``ab_report.py`` merges those shards into a
``results.csv``. The report reads the JSONL directly (``load_rows_jsonl``); the
dashboards read the already-built ``results.csv`` (``load_rows_csv``). Both live
here so the de-dup key, the method-normalization rule, and the translator-name
parsing are defined ONCE for every consumer instead of copied between CLIs.
"""

import csv
import glob
import json
import re
from pathlib import Path


def load_rows_jsonl(patterns, methods=None):
    """Load result rows from JSONL shard glob(s), de-duplicated across shards / re-runs.

    Rows predating the method dimension carry no `method` field and are all CAA,
    so a missing method is normalized to "CAA" — that keeps the de-dup key stable
    for old files and lets `methods` filter them.

    A single report is single-method by construction: its heatmaps and effect
    sizes are indexed by (scope, translator, norm, behavior), and mixing methods
    into one page would silently overlay curves whose coefficients mean different
    physical doses. Pass `methods` to select one; cross-method comparison lives in
    method_report.py."""
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


def load_rows_csv(results_csv, methods=("CAA",)):
    """Rows of one run's results.csv, restricted to `methods`.

    The dashboards index blocks by (scope, translator, norm, behavior) with no
    method component, and their coefficient axis is shared across every curve they
    draw — which is only valid within one discovery method, since CAA carries its
    own magnitude while RepE/GCAV are unit-norm. They therefore load ONE method,
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
