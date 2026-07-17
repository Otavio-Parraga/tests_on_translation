"""Config loading and activation-cache resolution shared by every entry point.

All CLI scripts follow the same pattern: load a TOML config, resolve the
source/target activation caches from it (with optional CLI overrides), then
load the cached tensors. This module is that pattern, written once.
"""

import tomllib
from pathlib import Path

import torch

from .paths import data_dir_of, resolve_activation_path


def load_config(path) -> dict:
    """Load a TOML experiment config."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve_activation_paths(config, source_override=None, target_override=None):
    """(source, target) activation-cache paths for a config, honoring explicit
    CLI overrides; otherwise the per-(model, layer) cache under data_dir
    (falling back to the legacy flat layout, see ``resolve_activation_path``)."""
    data_dir = data_dir_of(config)
    src = (
        Path(source_override)
        if source_override
        else resolve_activation_path(data_dir, config["source_model"], "source")
    )
    tgt = (
        Path(target_override)
        if target_override
        else resolve_activation_path(data_dir, config["target_model"], "target")
    )
    return src, tgt


def load_activations(path) -> torch.Tensor:
    """Load an activation cache: either the standard {'activations': ...} dict
    written by extraction, or a bare tensor."""
    obj = torch.load(path, weights_only=False)
    return obj["activations"] if isinstance(obj, dict) else obj
