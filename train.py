import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
from acttrans.models.translator import build_translator, save_translator
from acttrans.training.trainer import train_translator
from acttrans.utils.config import load_activations, load_config, resolve_activation_paths
from acttrans.utils.paths import translator_path


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

    model, train_losses, val_losses = train_translator(activations, config, model)

    save_translator(model, output_path, config, input_dim, output_dim)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
