import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    from src.evaluation.metrics import compute_retrieval_metrics
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.evaluation.metrics import compute_retrieval_metrics


def evaluate_translator(model, val_source, val_target, ks, device):
    metrics = compute_retrieval_metrics(
        model, val_source, val_target, ks=ks, device=device
    )
    return metrics


def evaluate_from_checkpoint(
    checkpoint_path,
    source_activations_path,
    target_activations_path,
    config_path,
    ks=None,
):
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    try:
        from src.models.translator import load_translator
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.models.translator import load_translator

    model = load_translator(checkpoint_path)

    source = torch.load(source_activations_path, weights_only=False)["activations"]
    target = torch.load(target_activations_path, weights_only=False)["activations"]
    N = source.shape[0]

    tcfg = config["training"]
    train_ratio = tcfg.get("train_ratio", 0.8)
    seed = tcfg.get("seed", 42)

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(N, generator=generator)
    source = source[perm]
    target = target[perm]

    n_train = int(N * train_ratio)
    val_source = source[n_train:]
    val_target = target[n_train:]

    ks = ks if ks is not None else tcfg.get("eval_ks", [1, 5, 10])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = evaluate_translator(model, val_source, val_target, ks=ks, device=device)

    for k, dirs in sorted(metrics.items()):
        print(
            f"Acc@{k}  src→tgt: {dirs['src2tgt']:.4f}  tgt→src: {dirs['tgt2src']:.4f}"
        )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-activations", required=True)
    parser.add_argument("--target-activations", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    evaluate_from_checkpoint(
        args.checkpoint,
        args.source_activations,
        args.target_activations,
        args.config,
    )
