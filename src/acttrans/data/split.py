"""The train/val split used everywhere.

The trainer, the standalone evaluator and the Procrustes baseline must all see
the exact same held-out rows, or "held-out" numbers silently become train-set
numbers. This is the single implementation of that split.
"""

import torch
import torch.nn.functional as F


def preprocess_activations(source, target, config):
    """Apply the training-time activation preprocessing to a (source, target)
    pair and return them. Currently that is the optional L2-normalization
    controlled by ``training.normalize_activations`` (default False).

    The trainer and the Procrustes baseline must preprocess identically, or the
    baseline stops being comparable to the trained translators; transport-time
    preprocessing (see ``models.transport``) reads the same flag. This is the
    single implementation for the paired case so those stay locked together.
    """
    if config.get("training", {}).get("normalize_activations", False):
        source = F.normalize(source, dim=-1)
        target = F.normalize(target, dim=-1)
    return source, target


def split_paired_activations(source, target, config):
    """Deterministic (seeded) shuffle + train/val split of index-paired
    activations. Returns (train_src, train_tgt, val_src, val_tgt).

    Uses training.seed (default 42) and training.train_ratio (default 0.8),
    matching what every training run in outputs/ was produced with."""
    tcfg = config.get("training", {})
    train_ratio = tcfg.get("train_ratio", 0.8)
    seed = tcfg.get("seed", 42)

    N = source.shape[0]
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(N, generator=generator)
    source = source[perm]
    target = target[perm]

    n_train = int(N * train_ratio)
    return source[:n_train], target[:n_train], source[n_train:], target[n_train:]
