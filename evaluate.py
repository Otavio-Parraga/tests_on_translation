import sys, argparse, tomllib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.evaluator import evaluate_from_checkpoint
from src.utils.paths import data_dir_of, resolve_activation_path, best_translator_path

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

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    output_dir = Path(config["training"]["output_dir"])
    data_dir = data_dir_of(config)
    checkpoint_path = args.checkpoint or best_translator_path(output_dir, config)
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
