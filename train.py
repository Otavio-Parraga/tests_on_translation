import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
import torch.nn.functional as F
from acttrans.data.split import preprocess_activations, split_paired_activations
from acttrans.models.translator import (
    build_translator,
    fit_procrustes_anchor,
    save_translator,
)
from acttrans.training.trainer import train_translator
from acttrans.utils.config import load_activations, load_config, resolve_activation_paths
from acttrans.utils.paths import translator_path


def _fit_anchor(model, source, target, config):
    """Fit the frozen closed-form Procrustes anchor BEFORE training starts.

    The anchor must be fitted on exactly the data the trainer will see: the same
    ``normalize_activations`` preprocessing and the same seeded train/val split, using
    the TRAIN half only. Fitting on all rows would leak the validation set into the
    starting point, making early-stopping's "held-out" loss no longer held out; fitting
    on differently-preprocessed data would leave the model starting somewhere other
    than the Procrustes floor it is supposed to start at.

    The optimal LS scale is stored on the config so it round-trips into the checkpoint
    (mirroring fit_procrustes.py), where it documents the anchor's magnitude. Note the
    transport layer does NOT auto-apply it to an anchored checkpoint — it is the scale
    of ``W`` alone, not of ``W + gate*base`` (see models/transport.py).
    """
    src, tgt = preprocess_activations(source.float(), target.float(), config)
    train_src, train_tgt, _, _ = split_paired_activations(src, tgt, config)
    print(f"Fitting frozen Procrustes anchor on {train_src.shape[0]} train pairs...")
    s = fit_procrustes_anchor(model, train_src, train_tgt)
    config["translator"]["procrustes_scale"] = float(s)

    # With gate initialized to 0 the model IS the anchor, so these numbers are the
    # Procrustes floor itself — printed so the log proves the run started there
    # instead of us merely asserting it.
    model.eval()
    with torch.no_grad():
        pred = model(train_src)
    print(
        f"  anchor scale s={s:.4f}  train MSE={F.mse_loss(pred, train_tgt).item():.6f}  "
        f"train cosine={F.cosine_similarity(pred, train_tgt, dim=-1).mean().item():.4f}  "
        f"(gate={model.gate.item():.3f}, so this is the Procrustes floor)"
    )


def main():
    parser = argparse.ArgumentParser(description="Train activation space translator")
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument(
        "--source-activations",
        default=None,
        help="Path to source activations (default: output_dir/activations/<source_model>.pt)",
    )
    parser.add_argument(
        "--target-activations",
        default=None,
        help="Path to target activations (default: output_dir/activations/<target_model>.pt)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save model (default: output_dir/translator__<source>__<target>.pt)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = Path(config["training"]["output_dir"])
    src_path, tgt_path = resolve_activation_paths(
        config, args.source_activations, args.target_activations
    )
    output_path = args.output or translator_path(output_dir, config)

    print(f"Loading source activations from {src_path}")
    print(f"Loading target activations from {tgt_path}")
    source = load_activations(src_path)
    target = load_activations(tgt_path)
    activations = {"source": source, "target": target}

    input_dim = source.shape[1]
    output_dim = target.shape[1]
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    model = build_translator(config, input_dim=input_dim, output_dim=output_dim)
    print(f"Translator: {sum(p.numel() for p in model.parameters()):,} parameters")

    if str(config["translator"].get("anchor", "") or "").lower() == "procrustes":
        _fit_anchor(model, source, target, config)

    model, train_losses, val_losses = train_translator(activations, config, model)

    save_translator(model, output_path, config, input_dim, output_dim)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
