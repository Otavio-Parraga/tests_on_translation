"""Retrieval metrics on known similarity matrices."""
import torch
import torch.nn as nn

from acttrans.evaluation.metrics import compute_retrieval_metrics, retrieval_accuracy


def test_retrieval_accuracy_identity():
    # perfect similarity: the true match is on the diagonal → acc@1 == 1.0
    sim = torch.eye(5)
    acc = retrieval_accuracy(sim, ks=[1])
    assert acc[1]["src2tgt"] == 1.0
    assert acc[1]["tgt2src"] == 1.0


def test_retrieval_accuracy_topk():
    # row i's true match (col i) is only the 2nd-highest → misses @1, hits @2
    sim = torch.tensor([
        [0.9, 1.0, 0.0],
        [0.0, 0.9, 1.0],
        [1.0, 0.0, 0.9],
    ])
    acc = retrieval_accuracy(sim, ks=[1, 2])
    assert acc[1]["src2tgt"] == 0.0
    assert acc[2]["src2tgt"] == 1.0


def test_compute_retrieval_metrics_identity_model():
    # identity model + source == target → every prediction matches its own target
    val = torch.randn(64, 32)
    acc = compute_retrieval_metrics(nn.Identity(), val, val.clone(), ks=[1], device="cpu")
    assert acc[1]["src2tgt"] == 1.0
    assert acc[1]["tgt2src"] == 1.0
