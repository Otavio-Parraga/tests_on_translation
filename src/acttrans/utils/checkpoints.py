"""Discovery + metadata parsing of translator checkpoints.

Checkpoint filenames are produced by ``paths.experiment_slug``:

    best_translator__{src_slug}__{tgt_slug}__{type}[+{anchor}]__{loss}.pt

with model slugs of the form ``{model}_l{layer}[_mean]``. This module parses
that convention back into structured metadata — the single copy shared by
ab_sweep.py and the comparison package (which previously each had their own).

The optional ``+{anchor}`` suffix marks a translator trained on top of a frozen
closed-form map (currently only ``+procrustes``). It is split off into its own
field rather than folded into ``ttype`` so ``ttype`` stays the bare architecture
name — downstream reporting groups by architecture *and* by anchor, and an
anchored run is the from-scratch run's control, not a different architecture.
"""

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TranslatorInfo:
    path: Path
    name: str          # checkpoint stem (unique id)
    ttype: str         # mlp | encoder | flow | sae | linear
    loss: str          # mse | cosine | info_nce | procrustes*
    pooling: str       # last | mean
    src_layer: int     # layer the source SV/activations were extracted at
    tgt_layer: int     # layer the translated SV is injected at on the target
    anchor: str = ""   # "" (from scratch) | procrustes (frozen closed-form anchor)


_CKPT_RE = re.compile(
    r"best_translator__(?P<src>.+?)__(?P<tgt>.+?)__"
    r"(?P<type>mlp|encoder|flow|sae|linear)(?:\+(?P<anchor>procrustes))?__"
    r"(?P<loss>.+)$"
)
_LAYER_RE = re.compile(r"_l(\d+)(?:_mean)?$")


def parse_translator(path: Path) -> TranslatorInfo:
    stem = path.stem  # strips .pt
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
        anchor=m.group("anchor") or "",
    )


def discover_translators(patterns: List[str]) -> List[TranslatorInfo]:
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in glob.glob(str(pat)))
    # de-dup + stable order
    uniq = sorted({p.resolve() for p in paths})
    return [parse_translator(p) for p in uniq]
