"""
Shared plumbing for comparing translated vs native steering vectors.

Works for any single-direction method in the activation_engineering tree (CAA,
RepE, GCAV — see ``acttrans.constants.METHODS``); the method is a parameter of
the loaders and defaults to CAA.

Provides:
  - loading of native steering vectors from the activation_engineering tree
  - the mean-activation direction of the target model (for centered cosines /
    mean-collapse diagnostics), computed once from the FineWeb activation cache
    and memoized on disk
  - small metric helpers (cosine, rejection, rank correlations)

Checkpoint discovery/parsing lives in ``acttrans.utils.checkpoints`` and the
in-memory direction transport in ``acttrans.models.transport``; both are
re-exported here for the analysis modules.

Paths are relative to the repo root (run everything from there); the
activation_engineering checkout defaults to a sibling of the repo and can be
overridden with the ACTENG_ROOT environment variable.
"""

import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F

from ..constants import (  # noqa: F401
    BEHAVIORS,
    METHOD,
    METHODS,
    MODULE,
    SOURCE_MODEL,
    TARGET_MODEL,
)
from ..models.transport import TranslatorRunner  # noqa: F401
from ..utils.checkpoints import TranslatorInfo, discover_translators, parse_translator  # noqa: F401
from ..utils import paths as _paths

ACTENG_ROOT = Path(os.environ.get("ACTENG_ROOT", "../activation_engineering"))

DEFAULT_TRANSLATOR_GLOB = "outputs/fineweb/best_translator__*.pt"
DEFAULT_OUT_DIR = Path("outputs/comparison")


# ── Steering vector loading ──────────────────────────────────────────────────

def sv_path(model_name: str, behavior: str, layer: int, method: str = METHOD) -> Path:
    return _paths.sv_path(ACTENG_ROOT, model_name, method, behavior, MODULE, layer)


def load_sv(model_name: str, behavior: str, layer: int,
            method: str = METHOD) -> torch.Tensor:
    """Load a steering vector as a float32 [D] tensor.

    Squeezing dim 0 normalizes the two shapes the methods save: CAA and RepE
    write [1, D], GCAV writes a bare [D]."""
    path = sv_path(model_name, behavior, layer, method)
    if not path.exists():
        raise FileNotFoundError(f"Steering vector not found: {path}")
    vec = torch.load(path, map_location="cpu", weights_only=True).float()
    return vec.squeeze(0) if vec.dim() == 2 else vec


def available_native_layers(model_name: str, behavior: str,
                            method: str = METHOD) -> List[int]:
    """Layers at which a native SV exists for this model/behavior/method."""
    root = (ACTENG_ROOT / "steering_vectors" / model_name.replace("/", "_")
            / method / behavior / MODULE)
    layers = []
    for d in root.glob("layer_*"):
        if (d / "sv.pt").exists():
            layers.append(int(d.name.split("_", 1)[1]))
    return sorted(layers)


# ── Mean-activation direction of the target model ───────────────────────────

def mean_activation_direction(
    model_name: str = TARGET_MODEL,
    layer: int = 8,
    pooling: str = "last",
    cache_dir: Path = DEFAULT_OUT_DIR / "cache",
):
    """Unit-norm mean activation of the FineWeb cache for (model, layer, pooling).

    Used as the 'generic direction' reference: high cosine of a translated SV
    with this direction (and with everything else) is the mean-collapse failure
    mode. Returns None (with a warning) if no activation cache exists.
    """
    short = model_name.split("/")[-1]
    suffix = "_mean" if pooling == "mean" else ""
    memo = cache_dir / f"mean_dir__{short}_l{layer}{suffix}.pt"
    if memo.exists():
        return torch.load(memo, map_location="cpu", weights_only=True)

    cache = Path("data/fineweb/activations") / f"{short}_l{layer}{suffix}.pt"
    if not cache.exists():
        print(f"  !! no activation cache at {cache}; centered-cosine metrics will be skipped")
        return None

    data = torch.load(cache, map_location="cpu", weights_only=False)
    acts = data["activations"] if isinstance(data, dict) else data
    mean_dir = F.normalize(acts.float().mean(dim=0), dim=-1)

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mean_dir, memo)
    return mean_dir


# ── Small metric helpers ─────────────────────────────────────────────────────

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def reject(v: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Component of v orthogonal to `direction` (which need not be unit norm)."""
    d = F.normalize(direction, dim=-1)
    return v - (v @ d) * d


def centered_cosine(a: torch.Tensor, b: torch.Tensor, mean_dir: torch.Tensor) -> float:
    """Cosine after removing each vector's component along the mean-activation
    direction. High raw cosine + near-zero centered cosine = the agreement was
    all shared mean."""
    return cosine(reject(a, mean_dir), reject(b, mean_dir))


def random_cosine_null(dim: int, reference: torch.Tensor, n: int = 500,
                       seed: int = 0) -> Dict[str, float]:
    """Null distribution of |cos(random, reference)| for calibration."""
    g = torch.Generator().manual_seed(seed)
    rand = torch.randn(n, dim, generator=g)
    cos = F.cosine_similarity(rand, reference.unsqueeze(0).expand(n, -1), dim=-1)
    return {"mean_abs": cos.abs().mean().item(), "std": cos.std().item()}


def spearman(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation without a scipy dependency."""
    ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    ra = ta.argsort().argsort().double()
    rb = tb.argsort().argsort().double()
    return pearson(ra.tolist(), rb.tolist())


def pearson(a: List[float], b: List[float]) -> float:
    ta, tb = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    ta, tb = ta - ta.mean(), tb - tb.mean()
    denom = ta.norm() * tb.norm()
    return (ta @ tb / denom).item() if denom > 0 else float("nan")
