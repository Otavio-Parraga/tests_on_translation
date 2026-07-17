"""Translator training losses on toy tensors."""
import pytest
import torch

from acttrans.training.losses import make_loss_fn


def test_mse_zero_when_equal():
    x = torch.randn(8, 16)
    assert make_loss_fn("mse", 0.07)(x, x).item() == pytest.approx(0.0, abs=1e-7)


def test_cosine_zero_when_aligned():
    x = torch.randn(8, 16)
    # cosine_embedding_loss with y=1 is 1 - cos; identical vectors → 0
    assert make_loss_fn("cosine", 0.07)(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_info_nce_lower_when_aligned():
    # well-separated rows so each row's best match is itself
    tgt = torch.eye(6) * 10
    fn = make_loss_fn("info_nce", 0.07)
    aligned = fn(tgt.clone(), tgt.clone()).item()
    shuffled = fn(tgt[torch.tensor([1, 2, 3, 4, 5, 0])].clone(), tgt.clone()).item()
    assert aligned < shuffled
    assert torch.isfinite(torch.tensor(aligned))


def test_vsp_zero_when_equal_and_singleton():
    fn = make_loss_fn("vsp", 0.07)
    x = torch.randn(5, 16)
    assert fn(x, x).item() == pytest.approx(0.0, abs=1e-6)
    # a single vector has no pairwise structure → exactly zero
    assert fn(torch.randn(1, 16), torch.randn(1, 16)).item() == 0.0


def test_unknown_loss_raises():
    with pytest.raises(ValueError):
        make_loss_fn("nope", 0.07)
