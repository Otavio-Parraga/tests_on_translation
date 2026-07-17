import sys, argparse, tomllib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.models.translator import build_translator, save_translator
from src.training.trainer import train_translator
from src.utils.paths import data_dir_of, resolve_activation_path, translator_path


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

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    output_dir = Path(config["training"]["output_dir"])
    data_dir = data_dir_of(config)
    src_path = (
        Path(args.source_activations)
        if args.source_activations
        else resolve_activation_path(data_dir, config["source_model"], "source")
    )
    tgt_path = (
        Path(args.target_activations)
        if args.target_activations
        else resolve_activation_path(data_dir, config["target_model"], "target")
    )
    output_path = args.output or translator_path(output_dir, config)

    print(f"Loading source activations from {src_path}")
    print(f"Loading target activations from {tgt_path}")
    source = torch.load(src_path, weights_only=False)["activations"]
    target = torch.load(tgt_path, weights_only=False)["activations"]
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
