"""Fit the orthogonal Procrustes linear baseline translator (closed form).

This is the strong, closed-form "floor" baseline for activation-space translation.
An orthogonal / semi-orthogonal map ``W`` (orthonormal columns here, since the
target space is larger than the source) preserves norms and inner products exactly,
so it structurally avoids the scale blow-up and mean-collapse pathologies of the
learned translators. Any nonlinear translator has to beat this.

Usage:
    conda run -n acteng python fit_procrustes.py --config config/fineweb/mlp_mse_last.toml

The checkpoint is saved (with ``translator.type = "linear"`` injected) via
``save_translator`` so ``translate_steering_vector.py`` and the ab-sweep loader
pick it up unchanged: ``outputs/fineweb/best_translator__..__linear__..pt``.
"""
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import copy
import torch
import torch.nn.functional as F

from acttrans.data.split import preprocess_activations, split_paired_activations
from acttrans.evaluation.metrics import compute_retrieval_metrics
from acttrans.models.translator import (
    build_translator,
    save_translator,
    fit_orthogonal_procrustes,
    procrustes_scale,
)
from acttrans.utils.config import load_activations, load_config, resolve_activation_paths
from acttrans.utils.paths import best_translator_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/fineweb/mlp_mse_last.toml")
    parser.add_argument("--source-activations", default=None,
                        help="Override path to source activations")
    parser.add_argument("--target-activations", default=None,
                        help="Override path to target activations")
    parser.add_argument("--output", default=None,
                        help="Override checkpoint output path")
    parser.add_argument("--source-layer", type=int, default=None,
                        help="Override source_model.layer from the config. The activation "
                             "cache and checkpoint filename are keyed by layer, so each "
                             "(source, target) layer pair yields a distinct translator.")
    parser.add_argument("--target-layer", type=int, default=None,
                        help="Override target_model.layer from the config (see --source-layer).")
    parser.add_argument("--no-center", dest="center", action="store_false",
                        help="Do not mean-center before fitting (default: center)")
    parser.add_argument("--no-bias", dest="bias", action="store_false",
                        help="Do not fit an affine bias (default: fit bias)")
    parser.add_argument("--whiten", action="store_true", default=False,
                        help="Fit a whitened (anisotropy-aware) Procrustes map instead "
                             "of the plain orthogonal one: whiten each space, fit the "
                             "orthogonal map on whitened coords, then fold whitening back "
                             "into W. Better fit on anisotropic residual streams, but the "
                             "resulting W is no longer exactly norm-preserving "
                             "(W^T W != I).")
    args = parser.parse_args()

    config = load_config(args.config)

    # Force the translator to the linear/Procrustes baseline so the checkpoint is
    # rebuilt as a LinearTranslator by build_translator on load, and so the slug
    # (and hence filename) reads "..__linear__..". Inject BEFORE computing the path.
    config = copy.deepcopy(config)
    config.setdefault("translator", {})
    config["translator"]["type"] = "linear"
    config["translator"]["bias"] = bool(args.bias)

    # Layer-pair overrides. Model slugs embed the layer (`_l{n}`), so overriding
    # here (BEFORE paths are computed) routes to the right activation cache and
    # yields a distinct checkpoint per (source, target) layer pair. The saved
    # config carries the overridden layers, so translate_steering_vector.py and
    # ab_sweep.py inject at the correct target layer downstream.
    if args.source_layer is not None:
        config["source_model"]["layer"] = args.source_layer
    if args.target_layer is not None:
        config["target_model"]["layer"] = args.target_layer

    # Tag the checkpoint by the Procrustes variant so the with-bias and no-bias
    # fits land in DISTINCT files (..__linear__procrustes.pt vs
    # ..__linear__procrustes_nobias.pt) and both coexist in the ab-sweep glob
    # instead of clobbering each other. loss_tag() reads training.losses, so
    # override it here (BEFORE the path is computed). Skipped if --output is set.
    if args.output is None:
        config.setdefault("training", {})
        base_tag = "procrustes_whiten" if args.whiten else "procrustes"
        config["training"]["losses"] = [base_tag if args.bias else f"{base_tag}_nobias"]
        config["training"].pop("loss", None)  # ensure loss_tag uses `losses`

    output_dir = Path(config["training"]["output_dir"])
    src_path, tgt_path = resolve_activation_paths(
        config, args.source_activations, args.target_activations
    )
    output_path = Path(args.output) if args.output else best_translator_path(output_dir, config)

    for path, mcfg in ((src_path, config["source_model"]), (tgt_path, config["target_model"])):
        if not path.exists():
            raise SystemExit(
                f"Activation cache not found: {path}\n"
                f"Extract it first:  conda run -n acteng python prepare_activations.py "
                f"--config {args.config} --source-layer {config['source_model']['layer']} "
                f"--target-layer {config['target_model']['layer']}"
            )

    print(f"Loading source activations from {src_path}")
    print(f"Loading target activations from {tgt_path}")
    source = load_activations(src_path).float()
    target = load_activations(tgt_path).float()

    # Match train.py / trainer preprocessing so the baseline is comparable.
    if config.get("training", {}).get("normalize_activations", False):
        print("normalize_activations=True -> L2-normalizing activations (matches trainer)")
    source, target = preprocess_activations(source, target, config)

    input_dim = source.shape[1]
    output_dim = target.shape[1]
    print(f"Input dim: {input_dim}, Output dim: {output_dim}, N={source.shape[0]}")

    # Held-out split mirroring the trainer (same seed + ratio) so numbers are comparable.
    train_src, train_tgt, val_src, val_tgt = split_paired_activations(source, target, config)
    n_train = train_src.shape[0]

    variant = "whitened (anisotropy-aware)" if args.whiten else "orthogonal"
    print(f"Fitting {variant} Procrustes (center={args.center}, bias={args.bias}, "
          f"whiten={args.whiten}) on {n_train} train pairs...")
    W, b = fit_orthogonal_procrustes(train_src, train_tgt, center=args.center,
                                     bias=args.bias, whiten=args.whiten)
    s = procrustes_scale(train_src, train_tgt, W, center=args.center)
    print(f"  W shape {tuple(W.shape)}; optimal scale s={s:.4f} "
          f"(used by translate_steering_vector.py --norm-mode procrustes)")

    model = build_translator(config, input_dim=input_dim, output_dim=output_dim)
    model.set_weights(W, b)
    model.eval()

    # Persist the optimal least-squares scale so it round-trips inside the
    # checkpoint. The "procrustes" norm mode in translate_steering_vector.py
    # reads it back to produce the faithful transport s*(W.sv).
    # build_translator ignores unknown translator keys, so this is safe.
    config["translator"]["procrustes_scale"] = float(s)

    # ── Train-set reconstruction quality ─────────────────────────────────────
    with torch.no_grad():
        pred_tr = model(train_src)
    mse = F.mse_loss(pred_tr, train_tgt).item()
    cos = F.cosine_similarity(pred_tr, train_tgt, dim=-1).mean().item()
    # Sanity: the plain orthogonal fit has orthonormal columns => W^T W ≈ I. The
    # whitened fit deliberately gives that up, so its ||W^T W - I|| is expected ≠ 0.
    wtw = (W.t() @ W)
    ortho_err = (wtw - torch.eye(W.shape[1])).abs().max().item()
    ortho_note = "expected ≠ 0 under --whiten" if args.whiten else "orthonormal-columns check"
    print(f"Train reconstruction: MSE={mse:.6f}  cosine={cos:.4f}  "
          f"||W^T W - I||_max={ortho_err:.2e}  ({ortho_note})")

    # ── Held-out retrieval (the actual "floor" number) ───────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = compute_retrieval_metrics(
        model, val_src, val_tgt, ks=[1, 5], device=device, max_samples=2048
    )
    print("Held-out retrieval (src->tgt / tgt->src):")
    for k in (1, 5):
        r = metrics[k]
        print(f"  acc@{k}: src2tgt={r['src2tgt']:.4f}  tgt2src={r['tgt2src']:.4f}")

    save_translator(model, output_path, config, input_dim, output_dim)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
