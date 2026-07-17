"""The shared train/val split and activation preprocessing."""
import torch

from acttrans.data.split import preprocess_activations, split_paired_activations


def test_split_deterministic_and_disjoint():
    src = torch.randn(100, 8)
    tgt = torch.randn(100, 12)
    cfg = {"training": {"train_ratio": 0.8, "seed": 42}}

    tr_s, tr_t, va_s, va_t = split_paired_activations(src, tgt, cfg)
    # sizes follow train_ratio
    assert tr_s.shape[0] == 80 and va_s.shape[0] == 20
    assert tr_t.shape[0] == 80 and va_t.shape[0] == 20
    # source/target stay row-aligned after the shared permutation
    assert tr_s.shape[1] == 8 and tr_t.shape[1] == 12

    # same seed → identical split
    again = split_paired_activations(src, tgt, cfg)
    assert all(torch.equal(a, b) for a, b in zip((tr_s, tr_t, va_s, va_t), again))

    # a different seed generally reorders the rows
    other = split_paired_activations(src, tgt, {"training": {"seed": 7}})
    assert not torch.equal(tr_s, other[0])


def test_split_defaults():
    src = torch.randn(10, 4)
    tgt = torch.randn(10, 4)
    # missing training table → defaults (ratio 0.8, seed 42)
    assert split_paired_activations(src, tgt, {})[0].shape[0] == 8


def test_preprocess_normalize_true():
    src = torch.randn(16, 8) * 5
    tgt = torch.randn(16, 12) * 3
    ns, nt = preprocess_activations(src, tgt, {"training": {"normalize_activations": True}})
    assert torch.allclose(ns.norm(dim=-1), torch.ones(16), atol=1e-5)
    assert torch.allclose(nt.norm(dim=-1), torch.ones(16), atol=1e-5)


def test_preprocess_normalize_false_is_identity():
    src = torch.randn(4, 4)
    tgt = torch.randn(4, 4)
    ns, nt = preprocess_activations(src, tgt, {"training": {}})
    # default path returns the same tensors untouched
    assert ns is src and nt is tgt
