"""Build the row-order manifest for behavior-contrastive-difference translator fitting.

WHY THIS EXISTS INSTEAD OF REUSING load_sentences_from_dataset("CAA", ...): that
loader (acttrans.data.dataset._load_caa + load_sentences_from_dataset) DEDUPLICATES
its flat sentence list while preserving order. Fitting a translator on per-pair
(positive - negative) DIFFERENCES requires the positive and negative activation of
the SAME item to line up by position; a loader that silently drops a duplicate
sentence shifts every index after it, and two items that happen to share an
identical question or answer string would misalign source and target without any
error. This script instead reads each behavior's generate_dataset.json directly,
keeps every item (even exact duplicates), and freezes the (behavior, item_index)
order into a manifest that both extract_caa_diff_source.py and
extract_caa_diff_target.py iterate identically — row i in every downstream .pt file
means the same underlying item, by construction, with no shared-loader involved.

WHY THE "generate" SPLIT, NOT "test": generate_dataset.json is the exact split the
ORIGINAL CAA vectors under activation_engineering/steering_vectors/ were built from
(see activation_engineering/configs/generation/CAA/config.yaml:
`dataset_kwargs: {split: generate}`, no `limit`, so every item is used). Fitting the
translator on the same split keeps its training distribution the real
behavior-contrastive distribution, not a resampling of it, and leaves
test_dataset_ab.json (the A/B eval set ab_sweep.py scores against) fully untouched —
no leakage between what the translator is fit on and what any A/B result is judged
on, independent of the leave-one-behavior-out split this manifest also supports.

Usage:
    conda run -p /mnt/C-SSD/parraga/conda_envs/act_eng python build_caa_diff_manifest.py
"""
import argparse
import hashlib
import json
from pathlib import Path

from acttrans.constants import BEHAVIORS

_HERE = Path(__file__).resolve().parent
_ACTENG_DATA = _HERE.parent / "activation_engineering" / "data"


def build_manifest(acteng_data: Path, behaviors=BEHAVIORS, split: str = "generate"):
    """Every (behavior, item) pair from each behavior's generate_dataset.json, in a
    fixed order: BEHAVIORS list order, then file order within each JSON. No
    shuffling, no dedup — see module docstring for why both matter here."""
    rows = []
    for behavior in behaviors:
        json_path = acteng_data / "CAA_datasets" / split / behavior / "generate_dataset.json"
        if not json_path.exists():
            raise FileNotFoundError(f"CAA generate-split dataset not found: {json_path}")
        items = json.loads(json_path.read_text())
        for i, item in enumerate(items):
            question = item.get("question", "").strip()
            pos_choice = item.get("answer_matching_behavior", "").strip()
            neg_choice = item.get("answer_not_matching_behavior", "").strip()
            if not (question and pos_choice and neg_choice):
                raise ValueError(
                    f"{json_path} item {i} is missing question/pos/neg — "
                    "every generate_dataset.json item is expected to have all three."
                )
            rows.append({
                "behavior": behavior,
                "item_index": i,
                "question": question,
                "pos_choice": pos_choice,
                "neg_choice": neg_choice,
            })
    return rows


def manifest_hash(rows) -> str:
    """Content hash of the manifest, stored alongside every diff-activation cache so
    a fit script can verify (cheaply, without re-reading the manifest) that a source
    cache and a target cache it's about to pair up were built from the SAME rows in
    the SAME order — the diff-cache analog of paths.py's n_sentences check."""
    blob = json.dumps(rows, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acteng-data", default=str(_ACTENG_DATA),
                    help="Root of activation_engineering/data (default: sibling repo)")
    ap.add_argument("--split", default="generate")
    ap.add_argument("--output", default=str(_HERE / "data" / "caa_diff" / "pairs_manifest.json"))
    args = ap.parse_args()

    rows = build_manifest(Path(args.acteng_data), split=args.split)
    h = manifest_hash(rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"hash": h, "rows": rows}, indent=None))

    print(f"  {len(rows)} pairs across {len(BEHAVIORS)} behaviors  ->  {out_path}")
    print(f"  manifest hash: {h}")
    from collections import Counter
    for b, n in Counter(r["behavior"] for r in rows).items():
        print(f"    {b:24s} {n:5d}")


if __name__ == "__main__":
    main()
