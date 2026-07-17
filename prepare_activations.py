import argparse

from dotenv import load_dotenv

load_dotenv()

from acttrans.data.dataset import prepare_paired_activations
from acttrans.utils.config import load_config
from acttrans.utils.paths import activation_path, data_dir_of


def main():
    parser = argparse.ArgumentParser(description="Extract paired activations from two models")
    parser.add_argument("--config", default="config/default.toml")
    parser.add_argument("--source-layer", type=int, default=None,
                        help="Override source_model.layer from the config. Caches are "
                             "keyed by (model, layer), so each layer gets its own file "
                             "and already-complete layers are skipped.")
    parser.add_argument("--target-layer", type=int, default=None,
                        help="Override target_model.layer from the config (see --source-layer).")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.source_layer is not None:
        config["source_model"]["layer"] = args.source_layer
    if args.target_layer is not None:
        config["target_model"]["layer"] = args.target_layer

    print(f"Extracting activations for:")
    print(f"  Source: {config['source_model']['name']} (layer {config['source_model']['layer']})")
    print(f"  Target: {config['target_model']['name']} (layer {config['target_model']['layer']})")

    result = prepare_paired_activations(config)
    data_dir = data_dir_of(config)
    print(f"Done. Source: {result['source'].shape}, Target: {result['target'].shape}")
    print(f"Saved to {activation_path(data_dir, config['source_model'])}")
    print(f"     and {activation_path(data_dir, config['target_model'])}")


if __name__ == "__main__":
    main()
