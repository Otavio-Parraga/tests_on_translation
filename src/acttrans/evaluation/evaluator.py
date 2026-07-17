import torch

from ..data.split import split_paired_activations
from ..models.translator import load_translator
from ..utils.config import load_activations, load_config
from .metrics import compute_retrieval_metrics


def evaluate_from_checkpoint(
    checkpoint_path,
    source_activations_path,
    target_activations_path,
    config_path,
    ks=None,
):
    config = load_config(config_path)
    model = load_translator(checkpoint_path)

    source = load_activations(source_activations_path)
    target = load_activations(target_activations_path)

    # Same seeded split as the trainer, so these are the true held-out rows.
    _, _, val_source, val_target = split_paired_activations(source, target, config)

    ks = ks if ks is not None else config["training"].get("eval_ks", [1, 5, 10])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = compute_retrieval_metrics(model, val_source, val_target, ks=ks, device=device)

    for k, dirs in sorted(metrics.items()):
        print(
            f"Acc@{k}  src→tgt: {dirs['src2tgt']:.4f}  tgt→src: {dirs['tgt2src']:.4f}"
        )

    return metrics
