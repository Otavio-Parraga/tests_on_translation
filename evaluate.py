import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from acttrans.evaluation.evaluator import evaluate_from_checkpoint
from acttrans.utils.config import load_config, resolve_activation_paths
from acttrans.utils.paths import best_translator_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate activation translator")
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to translator checkpoint (default: output_dir/best_translator__<source>__<target>.pt)")
    parser.add_argument("--source-activations", default=None,
                        help="Path to source activations (default: output_dir/activations/<source_model>.pt)")
    parser.add_argument("--target-activations", default=None,
                        help="Path to target activations (default: output_dir/activations/<target_model>.pt)")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10],
                        help="Top-k values to evaluate (default: 1 5 10)")
    args = parser.parse_args()

    config = load_config(args.config)

    output_dir = Path(config["training"]["output_dir"])
    checkpoint_path = args.checkpoint or best_translator_path(output_dir, config)
    src_path, tgt_path = resolve_activation_paths(
        config, args.source_activations, args.target_activations
    )

    print(f"Evaluating checkpoint: {checkpoint_path}")
    results = evaluate_from_checkpoint(
        checkpoint_path=checkpoint_path,
        source_activations_path=src_path,
        target_activations_path=tgt_path,
        config_path=args.config,
        ks=args.ks,
    )
    return results


if __name__ == "__main__":
    main()
