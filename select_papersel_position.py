"""Select the best (layer, template-relative position) direction, via the
existing closed-ended A/B metric — the counterpart of the paper's
validation-set position/layer selection (arXiv 2512.24143v2, Section 3.3),
scored here with this repo's own A/B metric instead of keyword-match/LLaMA-
Guard, per this project's scope decision.

TRAIN / VALIDATION SPLIT reuses the split this repo already keeps disjoint for
exactly this purpose (see build_caa_diff_manifest.py's docstring): the
"generate"-split manifest rows extract_target_papersel.py was run over build
each candidate direction (the mean of a behavior's per-item diffs at a given
(layer, tag)); the separate "test"-split test_dataset_ab.json items — never
touched by extraction — score it, via run_ab_eval(steering_mode="projection",
alpha=1.0): how much does orthogonally ablating this direction reduce
avg_p_match relative to the unmodified baseline.

ONE GLOBAL (layer*, tag*), SCORED ACROSS BEHAVIORS: fit_procrustes_lobo.py's
LOBO fit consumes one shared arm-cache file spanning all 7 behaviors' rows
(exactly like the existing "mean"/"mask" arms), so the winner here is the
candidate whose ablation-induced drop is largest ON AVERAGE across
`--behaviors` (default: all 7) — not the best pick for a single behavior in
isolation, which would bias the eventual LOBO fit toward whichever behavior
happened to be used for selection.

OUTPUT is the winning candidate's PER-ITEM diffs (all behaviors, all manifest
rows), written in the exact cache format extract_caa_diff_target.py's other
arms already use — so `fit_procrustes_lobo.py --arm papersel --target-layer
<winning layer>` needs no code changes at all.

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -p /mnt/C-SSD/parraga/conda_envs/act_eng python \\
        select_papersel_position.py --layers 10 12 14 16 18 20 22
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch

from acttrans.constants import BEHAVIORS
from acttrans.evaluation.ab_eval import aggregate_by_coefficient, run_ab_eval
from acttrans.utils.hf import load_model_and_tokenizer
from extract_target_papersel import DEFAULT_LAYERS, LLADA_8B, position_tags

_HERE = Path(__file__).resolve().parent
_ACTENG = _HERE.parent / "activation_engineering"
DIFF_DIR = _HERE / "data" / "caa_diff" / "activations"


def load_test_items(behavior: str):
    p = _ACTENG / "data" / "CAA_datasets" / "test" / behavior / "test_dataset_ab.json"
    return json.loads(p.read_text())


def load_candidate(short: str, layer: int, tag: str):
    path = DIFF_DIR / f"{short}_l{layer}_{tag}_diff.pt"
    return torch.load(path, weights_only=False) if path.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--behaviors", nargs="+", default=BEHAVIORS, choices=BEHAVIORS,
                    help="Behaviors to average the selection score across (default: all 7)")
    ap.add_argument("--model", default=LLADA_8B)
    ap.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    ap.add_argument("--n-pre", type=int, default=4)
    ap.add_argument("--n-post", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Cap validation items per behavior (quick runs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(DIFF_DIR))
    args = ap.parse_args()

    short = args.model.split("/")[-1]
    pre_tags, post_tags = position_tags(args.n_pre, args.n_post)
    tags = pre_tags + post_tags
    layers = sorted(args.layers)
    candidates = [(l, t) for l in layers for t in tags]

    model, tok = load_model_and_tokenizer(args.model, args.device)
    model.eval()

    drops = defaultdict(list)   # (layer, tag) -> [drop_p_match per behavior]
    scored_at = {}              # (layer, tag) -> raw avg_p_match, for logging

    for behavior in args.behaviors:
        val_items = load_test_items(behavior)
        print(f"\nBehavior {behavior!r}: {len(val_items)} validation (test-split) items")

        baseline = aggregate_by_coefficient(
            run_ab_eval(model, tok, val_items, torch.zeros(model.config.hidden_size), 0,
                       "residual", [0.0], limit=args.limit)
        )[0.0]
        print(f"  baseline avg_p_match={baseline['avg_p_match']:.4f}  accuracy={baseline['accuracy']:.4f}")

        for layer, tag in candidates:
            cache = load_candidate(short, layer, tag)
            if cache is None:
                print(f"  !! missing candidate cache for layer {layer} tag {tag}, skipping")
                continue
            mask = [b == behavior for b in cache["behaviors"]]
            beh_diffs = cache["diffs"][mask]
            if beh_diffs.shape[0] == 0:
                continue
            direction = beh_diffs.mean(dim=0)

            agg = aggregate_by_coefficient(
                run_ab_eval(model, tok, val_items, direction, layer, "residual",
                           [1.0], limit=args.limit, steering_mode="projection")
            )[1.0]
            drop = baseline["avg_p_match"] - agg["avg_p_match"]
            drops[(layer, tag)].append(drop)
            scored_at[(layer, tag)] = agg["avg_p_match"]
            print(f"  layer={layer:3d} tag={tag:6s}  ablated avg_p_match={agg['avg_p_match']:.4f}  drop={drop:+.4f}")

    if not drops:
        raise SystemExit("No candidates scored — did extract_target_papersel.py run for this model?")

    mean_drop = {k: sum(v) / len(v) for k, v in drops.items() if len(v) == len(args.behaviors)}
    if not mean_drop:
        raise SystemExit("No candidate was scored on every requested behavior — check for missing caches above.")

    best_layer, best_tag = max(mean_drop, key=mean_drop.get)
    print(f"\nWinner: layer={best_layer} tag={best_tag}  mean drop_p_match={mean_drop[(best_layer, best_tag)]:+.4f} "
          f"across {len(args.behaviors)} behaviors")

    winner_cache = load_candidate(short, best_layer, best_tag)
    out_path = Path(args.out_dir) / f"{short}_l{best_layer}_papersel_diff.pt"
    torch.save({
        "diffs": winner_cache["diffs"], "behaviors": winner_cache["behaviors"],
        "n_pairs": winner_cache["diffs"].shape[0], "manifest_hash": winner_cache["manifest_hash"],
        "pooling": "papersel", "model": args.model, "layer": best_layer, "module": "residual",
        "papersel_selected_tag": best_tag, "papersel_behaviors": args.behaviors,
        "papersel_mean_drop_p_match": mean_drop[(best_layer, best_tag)],
    }, out_path)
    print(f"Wrote {out_path}\nUse with: fit_procrustes_lobo.py --target-layer {best_layer} --arm papersel")


if __name__ == "__main__":
    main()
