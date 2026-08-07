"""Checkpoint-filename parsing (the single shared parser)."""
from pathlib import Path

import pytest

from acttrans.utils.checkpoints import parse_translator


def test_parse_last_token_single_loss():
    p = Path("outputs/fineweb/best_translator__"
             "Llama-3.2-1B-Instruct_l8__Llama-3.2-3B-Instruct_l8__mlp__mse.pt")
    info = parse_translator(p)
    assert info.ttype == "mlp"
    assert info.loss == "mse"
    assert info.pooling == "last"
    assert info.src_layer == 8 and info.tgt_layer == 8
    assert info.name == p.stem
    assert info.anchor == ""   # from-scratch names carry no anchor


def test_parse_mean_pooled_compound_loss():
    p = Path("best_translator__"
             "Llama-3.2-1B-Instruct_l8_mean__Llama-3.2-3B-Instruct_l8_mean__flow__mse+cosine+info_nce.pt")
    info = parse_translator(p)
    assert info.ttype == "flow"
    assert info.loss == "mse+cosine+info_nce"
    assert info.pooling == "mean"
    assert info.src_layer == 8 and info.tgt_layer == 8


def test_parse_anchored_keeps_bare_ttype():
    # `+procrustes` is split into its own field: ttype stays the architecture name
    # so an anchored run groups with (and is the control for) the from-scratch one.
    p = Path("outputs/anchored/best_translator__"
             "Llama-3.2-1B-Instruct_l8__Llama-3.2-3B-Instruct_l12__mlp+procrustes__"
             "cosine+info_nce.pt")
    info = parse_translator(p)
    assert info.ttype == "mlp"
    assert info.anchor == "procrustes"
    assert info.loss == "cosine+info_nce"
    assert info.pooling == "last"
    assert info.src_layer == 8 and info.tgt_layer == 12


def test_parse_rejects_bad_name():
    with pytest.raises(ValueError):
        parse_translator(Path("not_a_translator.pt"))
