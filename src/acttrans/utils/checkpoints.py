"""Discovery + metadata parsing of translator checkpoints.

Checkpoint filenames are produced by ``paths.experiment_slug``:

    best_translator__{src_slug}__{tgt_slug}__{type}[+{anchor}]__{loss}.pt

with model slugs of the form ``{model}_l{layer}[_mean|_mask]``. This module parses
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

from .paths import POOLING_SUFFIX_RE, POOLING_SUFFIXES


@dataclass
class TranslatorInfo:
    path: Path
    name: str          # checkpoint stem (unique id)
    ttype: str         # mlp | encoder | flow | sae | linear
    loss: str          # mse | cosine | info_nce | procrustes*
    pooling: str       # last | mean | mask — the SOURCE side's extraction mode
    src_layer: int     # layer the source SV/activations were extracted at
    tgt_layer: int     # layer the translated SV is injected at on the target
    anchor: str = ""   # "" (from scratch) | procrustes (frozen closed-form anchor)
    # The TARGET side's extraction mode, which need not match the source's. It
    # usually does (config_gen sets both from one value), but a diffusion target
    # is read at "mean"/"mask" while its causal source stays last-token — and for
    # those runs `pooling` alone reads "last" for every arm, making the arms
    # indistinguishable in any report that groups by it. Defaulted so existing
    # callers that only care about the source side are unaffected.
    tgt_pooling: str = "last"


_CKPT_RE = re.compile(
    r"best_translator__(?P<src>.+?)__(?P<tgt>.+?)__"
    r"(?P<type>mlp|encoder|flow|sae|linear)(?:\+(?P<anchor>procrustes))?__"
    r"(?P<loss>.+)$"
)
# Built from the paths.py suffix table so a newly registered extraction mode
# parses here too, instead of failing to match and killing a whole glob.
_LAYER_RE = re.compile(r"_l(\d+)%s$" % POOLING_SUFFIX_RE)


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
    def pooling_of_slug(slug: str) -> str:
        return next((mode for mode, suf in POOLING_SUFFIXES.items() if slug.endswith(suf)), "last")

    return TranslatorInfo(
        path=path,
        name=stem,
        ttype=m.group("type"),
        loss=m.group("loss"),
        pooling=pooling_of_slug(m.group("src")),
        src_layer=layers[0],
        tgt_layer=layers[1],
        anchor=m.group("anchor") or "",
        tgt_pooling=pooling_of_slug(m.group("tgt")),
    )


def discover_translators(patterns: List[str]) -> List[TranslatorInfo]:
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(Path(p) for p in glob.glob(str(pat)))
    # de-dup + stable order
    uniq = sorted({p.resolve() for p in paths})
    return [parse_translator(p) for p in uniq]
