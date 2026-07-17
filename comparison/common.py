"""
Shared plumbing for comparing translated vs native CAA steering vectors.

Provides:
  - discovery/parsing of translator checkpoints (mirrors ab_sweep.py conventions)
  - loading of native CAA steering vectors from the activation_engineering tree
  - in-memory direction transport through a translator checkpoint (mirrors the
    logic of translate_steering_vector.py, without writing anything to disk)
  - the mean-activation direction of the target model (for centered cosines /
    mean-collapse diagnostics), computed once from the FineWeb activation cache
    and memoized on disk.
"""

import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent.parent  # repo root
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.models.translator import build_translator, LinearTranslator  # noqa: E402

_ACTENG = _HERE.parent / "activation_engineering"

SOURCE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
METHOD = "CAA"
MODULE = "residual"

BEHAVIORS = [
    "coordinate-other-ais",
    "corrigible-neutral-HHH",
    "hallucination",
    "myopic-reward",
    "refusal",
    "survival-instinct",
    "sycophancy",
]

DEFAULT_TRANSLATOR_GLOB = str(_HERE / "outputs" / "fineweb" / "best_translator__*.pt")
DEFAULT_OUT_DIR = _HERE / "outputs" / "comparison"


# ── Steering vector loading ──────────────────────────────────────────────────

def sv_path(model_name: str, behavior: str, layer: int) -> Path:
    return (
        _ACTENG / "steering_vectors" / model_name.replace("/", "_")
        / METHOD / behavior / MODULE / f"layer_{layer}" / "sv.pt"
    )


def load_sv(model_name: str, behavior: str, layer: int) -> torch.Tensor:
    """Load a CAA steering vector as a float32 [D] tensor."""
    path = sv_path(model_name, behavior, layer)
    if not path.exists():
        raise FileNotFoundError(f"Steering vector not found: {path}")
    vec = torch.load(path, map_location="cpu", weights_only=True).float()
    return vec.squeeze(0) if vec.dim() == 2 else vec


def available_native_layers(model_name: str, behavior: str) -> List[int]:
    """Layers at which a native CAA SV exists for this model/behavior."""
    root = _ACTENG / "steering_vectors" / model_name.replace("/", "_") / METHOD / behavior / MODULE
    layers = []
    for d in root.glob("layer_*"):
        if (d / "sv.pt").exists():
            layers.append(int(d.name.split("_", 1)[1]))
    return sorted(layers)


# ── Translator checkpoint metadata (same filename conventions as ab_sweep.py) ─

@dataclass
class TranslatorInfo:
    path: Path
    name: str          # checkpoint stem (unique id)
    ttype: str         # mlp | encoder | flow | sae | linear
    loss: str          # mse | cosine | info_nce | procrustes*
    pooling: str       # last | mean
    src_layer: int
    tgt_layer: int


_CKPT_RE = re.compile(
    r"best_translator__(?P<src>.+?)__(?P<tgt>.+?)__(?P<type>mlp|encoder|flow|sae|linear)__(?P<loss>.+)$"
)
_LAYER_RE = re.compile(r"_l(\d+)(?:_mean)?$")


def parse_translator(path: Path) -> TranslatorInfo:
    stem = path.stem
    m = _CKPT_RE.match(stem)
    if not m:
        raise ValueError(f"Cannot parse translator filename: {path.name}")
    layers = []
    for slug in (m.group("src"), m.group("tgt")):
        lm = _LAYER_RE.search(slug)
        if not lm:
            raise ValueError(f"Cannot parse layer from model slug {slug!r} in {path.name}")
        layers.append(int(lm.group(1)))
    pooling = "mean" if m.group("src").endswith("_mean") else "last"
    return TranslatorInfo(
        path=path,
        name=stem,
        ttype=m.group("type"),
        loss=m.group("loss"),
        pooling=pooling,
        src_layer=layers[0],
        tgt_layer=layers[1],
    )


def discover_translators(patterns: Optional[List[str]] = None) -> List[TranslatorInfo]:
    patterns = patterns or [DEFAULT_TRANSLATOR_GLOB]
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in glob.glob(pat))
    uniq = sorted({p.resolve() for p in paths})
    return [parse_translator(p) for p in uniq]


# ── In-memory direction transport ────────────────────────────────────────────

class TranslatorRunner:
    """Loads a translator checkpoint once and transports direction vectors
    in memory, faithfully mirroring translate_steering_vector.py:

      - matches training preprocessing (normalize_activations)
      - linear translators transport bias-free (the affine bias cancels for a
        difference direction)
      - norm modes: "restore" (rescale to the source SV norm), "none" (raw
        translator output), "procrustes" (bias-free linear output times the
        fitted scale s; linear-only).
    """

    def __init__(self, info: TranslatorInfo, device: str = "cpu"):
        self.info = info
        self.device = device
        ckpt = torch.load(info.path, map_location="cpu", weights_only=False)
        self.config = ckpt["config"]
        self.normalize_activations = (
            self.config.get("training", {}).get("normalize_activations", False)
        )
        self.module = build_translator(
            self.config, input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"]
        )
        self.module.load_state_dict(ckpt["state_dict"])
        self.module = self.module.to(device).eval()

    @property
    def procrustes_scale(self) -> Optional[float]:
        try:
            return float(self.config["translator"]["procrustes_scale"])
        except (KeyError, TypeError):
            return None

    def default_norm_mode(self) -> str:
        """Translator-aware default, matching ab_sweep.transport_modes():
        linear/Procrustes checkpoints -> faithful floor transport, others ->
        restore the source SV's norm."""
        if self.info.ttype == "linear" and self.procrustes_scale is not None:
            return "procrustes"
        return "restore"

    @torch.no_grad()
    def transport(self, vec: torch.Tensor, norm_mode: str = "restore") -> torch.Tensor:
        """Transport a [D_src] direction vector; returns a float32 [D_tgt] cpu tensor."""
        if norm_mode not in {"restore", "none", "procrustes"}:
            raise ValueError(f"Unknown norm_mode {norm_mode!r}")

        x = vec.float().to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        src_norm = x.norm(dim=-1, keepdim=True)

        if self.normalize_activations:
            x = F.normalize(x, dim=-1)

        if isinstance(self.module, LinearTranslator):
            # Bias-free direction transport (bias cancels for a difference).
            out = x @ self.module.W.weight.T
        else:
            out = self.module(x)
        out = out.squeeze(0).float().cpu()

        if norm_mode == "restore":
            out = F.normalize(out, dim=-1) * src_norm.squeeze().cpu()
        elif norm_mode == "procrustes":
            s = self.procrustes_scale
            if s is None:
                raise ValueError(
                    f"norm_mode='procrustes' needs config.translator.procrustes_scale, "
                    f"absent from {self.info.name}"
                )
            out = out * s
        return out


# ── Mean-activation direction of the target model ───────────────────────────

def mean_activation_direction(
    model_name: str = TARGET_MODEL,
    layer: int = 8,
    pooling: str = "last",
    cache_dir: Path = DEFAULT_OUT_DIR / "cache",
) -> Optional[torch.Tensor]:
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

    cache = _HERE / "data" / "fineweb" / "activations" / f"{short}_l{layer}{suffix}.pt"
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
