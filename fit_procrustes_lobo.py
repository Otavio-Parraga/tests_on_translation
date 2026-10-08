"""Leave-one-behavior-out Procrustes fit on CAA behavior-contrastive DIFFERENCE
vectors (extract_caa_diff_source.py / extract_caa_diff_target.py), instead of on
generic FineWeb sentence activations.

WHY: fit_procrustes.py fits W on raw activations of generic FineWeb sentences —
useful as a floor baseline, but the actual object ever transported through W is a
behavior-contrastive DIFFERENCE vector (a CAA/RepE/GCAV steering vector), which is a
different distribution from "the residual stream of an arbitrary sentence." This
script fits W directly on that distribution instead, and asks the sharper question:
does the map generalize to a behavior it never saw fit on? Held out one at a time,
across all 7 behaviors, so every behavior gets exactly one fold where its own
diff vectors never touched the fit.

DATA, not files: no [dataset]/data_dir plumbing here. --source-diff/--target-diff
point directly at the .pt caches the two extract_caa_diff_*.py scripts write, each
carrying {"diffs": [N,dim], "behaviors": [N], "manifest_hash": ...}. The two files
being pairable (same N, same row order) is checked via manifest_hash before
anything is fit — a silent misalignment here would poison every fold.

CENTER=True (the fit_procrustes.py default) is kept for diff vectors too: whatever
mean offset is common to every difference regardless of behavior (for instance, any
residual contribution from the answer-choice letter token "(A)"/"(B)" itself,
present in every pos/neg pair alike) is exactly what centering removes before
fitting the direction-preserving map — arguably MORE apt here than for raw
activations, not less.

THE TRANSPORT DIAGNOSTIC (the actual answer to "does this generalize?"): after
fitting on the other 6 behaviors, the excluded behavior's REAL, already-existing
CAA vector (steering_vectors/.../CAA/<behavior>/residual/layer_8/sv.pt — the exact
file ab_sweep.py would translate) is pushed through the freshly-saved checkpoint via
TranslatorRunner.transport(...), reusing the IDENTICAL code path
translate_steering_vector.py/ab_sweep.py use at real transport time (procrustes
norm mode, bias-free) — not a hand-rolled reimplementation that could silently
diverge from it. The result is compared by cosine to the held-out behavior's own
mean target-side diff (this script's independently-extracted ground truth for "what
LLaDA's own analog of that behavior vector looks like"). Retrieval metrics
(compute_retrieval_metrics) on the held-out behavior's per-item diffs are reported
alongside as a second, complementary check (item-level structure, not just the
mean direction).

Usage (one fold; the launcher loops all 7 behaviors x layers x arms x variants):
    conda run -p /mnt/C-SSD/parraga/conda_envs/act_eng python fit_procrustes_lobo.py \\
        --config config/new_models/fineweb_llada8b.toml \\
        --target-layer 16 --arm mean --exclude-behavior refusal
"""
import argparse
import copy
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
import torch.nn.functional as F

from acttrans.constants import BEHAVIORS, LAYER, SOURCE_MODEL
from acttrans.evaluation.metrics import compute_retrieval_metrics
from acttrans.models.transport import TranslatorRunner
from acttrans.models.translator import (
    build_translator, save_translator, fit_orthogonal_procrustes, procrustes_scale,
)
from acttrans.utils.config import load_config
from acttrans.utils.paths import best_translator_path

_HERE = Path(__file__).resolve().parent
_ACTENG = _HERE.parent / "activation_engineering"
DIFF_DIR = _HERE / "data" / "caa_diff" / "activations"


def load_diff_cache(path: Path):
    d = torch.load(path, weights_only=False)
    return d["diffs"].float(), d["behaviors"], d["manifest_hash"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    help="config/new_models/fineweb_llada8b.toml (mean/papersel) or "
                         "fineweb_llada8b_mask.toml (mask) — model/layer metadata only, "
                         "no activation cache is read from it.")
    ap.add_argument("--exclude-behavior", required=True, choices=BEHAVIORS)
    ap.add_argument("--source-layer", type=int, default=LAYER)
    ap.add_argument("--target-layer", type=int, required=True)
    ap.add_argument("--arm", choices=["mean", "mask", "papersel"], required=True)
    ap.add_argument("--source-diff", default=None)
    ap.add_argument("--target-diff", default=None)
    ap.add_argument("--whiten", action="store_true", default=False)
    ap.add_argument("--no-bias", dest="bias", action="store_false", default=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--device", default="cpu",
                    help="Device for BOTH the Procrustes SVD/eigh fit and the retrieval "
                         "check. Default cpu: these matrices are tiny (<=4911 rows) and "
                         "this script is meant to run with zero GPU dependency by default, "
                         "so a busy GPU never blocks a LOBO fit. Pass e.g. cuda:0 to run the "
                         "fit itself on GPU instead — useful when the bottleneck is CPU "
                         "contention from OTHER processes on a shared machine, not the "
                         "linear algebra itself (fit_orthogonal_procrustes/procrustes_scale "
                         "are plain tensor ops with no CPU-only calls, so this is a pure "
                         "device move, not a different code path).")
    args = ap.parse_args()

    config = load_config(args.config)
    config = copy.deepcopy(config)
    config["source_model"]["layer"] = args.source_layer
    config["target_model"]["layer"] = args.target_layer
    config.setdefault("translator", {})
    config["translator"]["type"] = "linear"
    config["translator"]["bias"] = bool(args.bias)
    config.setdefault("training", {})
    base_tag = "procrustes_whiten" if args.whiten else "procrustes"
    tag = f"{base_tag}{'' if args.bias else '_nobias'}_lobo-{args.exclude_behavior}"
    config["training"]["losses"] = [tag]
    config["training"].pop("loss", None)
    config["training"]["normalize_activations"] = False  # RAW diffs — Procrustes preserves norms

    src_path = Path(args.source_diff) if args.source_diff else (
        DIFF_DIR / f"{SOURCE_MODEL.split('/')[-1]}_l{args.source_layer}_diff.pt"
    )
    tgt_path = Path(args.target_diff) if args.target_diff else (
        DIFF_DIR / f"{config['target_model']['name'].split('/')[-1]}_l{args.target_layer}_{args.arm}_diff.pt"
    )
    output_dir = Path(config["training"]["output_dir"])
    output_path = Path(args.output) if args.output else best_translator_path(output_dir, config)

    for p in (src_path, tgt_path):
        if not p.exists():
            raise SystemExit(f"Diff cache not found: {p}\nRun extract_caa_diff_*.py first.")

    src_diffs, src_beh, src_hash = load_diff_cache(src_path)
    tgt_diffs, tgt_beh, tgt_hash = load_diff_cache(tgt_path)
    if src_hash != tgt_hash:
        raise SystemExit(
            f"manifest_hash mismatch: {src_path.name}={src_hash} vs {tgt_path.name}={tgt_hash}. "
            "These two caches were built from different manifests and are NOT row-aligned — "
            "refusing to fit on them."
        )
    if src_beh != tgt_beh:
        raise SystemExit("Behavior label lists differ between source and target caches — "
                          "row alignment cannot be trusted.")

    behaviors = src_beh
    excl_mask = [b == args.exclude_behavior for b in behaviors]
    train_idx = [i for i, m in enumerate(excl_mask) if not m]
    held_idx = [i for i, m in enumerate(excl_mask) if m]
    if not held_idx:
        raise SystemExit(f"No rows for behavior {args.exclude_behavior!r} in {tgt_path}")

    train_src, train_tgt = src_diffs[train_idx], tgt_diffs[train_idx]
    held_src, held_tgt = src_diffs[held_idx], tgt_diffs[held_idx]

    # Fit on args.device (default cpu); held_src/held_tgt stay on cpu regardless —
    # compute_retrieval_metrics moves what it needs internally, and the money-metric
    # step below always runs on cpu via TranslatorRunner.
    train_src_fit = train_src.to(args.device)
    train_tgt_fit = train_tgt.to(args.device)

    input_dim, output_dim = train_src.shape[1], train_tgt.shape[1]
    print(f"Excluding {args.exclude_behavior!r}: {len(train_idx)} train pairs "
          f"(6 behaviors), {len(held_idx)} held-out pairs")
    print(f"Fitting {'whitened' if args.whiten else 'orthogonal'} Procrustes "
          f"(bias={args.bias}) on diff vectors, dims {input_dim}->{output_dim}, device={args.device}")

    W, b = fit_orthogonal_procrustes(train_src_fit, train_tgt_fit, center=True, bias=args.bias, whiten=args.whiten)
    s = procrustes_scale(train_src_fit, train_tgt_fit, W, center=True)
    config["translator"]["procrustes_scale"] = float(s)
    print(f"  W shape {tuple(W.shape)}; optimal scale s={s:.4f}")

    # build_translator() always constructs on CPU; set_weights() uses in-place
    # .copy_(), which preserves the DESTINATION's device rather than adopting W/b's
    # — so without this .to(), the model would silently stay CPU-resident while
    # W/b (and train_src_fit/train_tgt_fit) are on args.device, and the very next
    # forward pass would fail with a cuda/cpu mismatch.
    model = build_translator(config, input_dim=input_dim, output_dim=output_dim).to(args.device)
    model.set_weights(W, b)
    model.eval()

    with torch.no_grad():
        pred_tr = model(train_src_fit)
    mse = F.mse_loss(pred_tr, train_tgt_fit).item()
    cos_tr = F.cosine_similarity(pred_tr, train_tgt_fit, dim=-1).mean().item()
    print(f"Train (6 behaviors) reconstruction: MSE={mse:.6f}  cosine={cos_tr:.4f}")

    metrics = compute_retrieval_metrics(model, held_src, held_tgt, ks=[1, 5],
                                        device=args.device, max_samples=None)
    print(f"Held-out ({args.exclude_behavior}) item-level retrieval, N={len(held_idx)}:")
    for k in (1, 5):
        r = metrics[k]
        print(f"  acc@{k}: src2tgt={r['src2tgt']:.4f}  tgt2src={r['tgt2src']:.4f}")

    model = model.cpu()  # checkpoints are always saved CPU-resident, like every other one here
    save_translator(model, output_path, config, input_dim, output_dim)
    print(f"Model saved to {output_path}")

    # ── The money metric: transport the REAL, already-existing sv.pt through this
    # never-saw-this-behavior map, via the EXACT code path ab_sweep.py uses. ────
    real_sv_path = (_ACTENG / "steering_vectors" / SOURCE_MODEL.replace("/", "_") / "CAA"
                    / args.exclude_behavior / "residual" / f"layer_{args.source_layer}" / "sv.pt")
    if real_sv_path.exists():
        real_sv = torch.load(real_sv_path, map_location="cpu", weights_only=True).float()
        runner = TranslatorRunner(output_path, device="cpu")
        predicted = runner.transport(real_sv, norm_mode="procrustes", apply_bias=False)
        actual_mean_diff = held_tgt.mean(dim=0)
        cos = F.cosine_similarity(predicted, actual_mean_diff, dim=0).item()
        print(f"Money metric: cosine(translated real sv.pt, held-out mean target diff) = {cos:.4f}")
    else:
        print(f"(no real sv.pt at {real_sv_path} — skipping the money-metric check)")


if __name__ == "__main__":
    main()
