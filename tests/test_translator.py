"""Structural properties of the translator models and the Procrustes fit."""
import torch

from acttrans.models.translator import (
    FlowTranslator,
    fit_orthogonal_procrustes,
    procrustes_scale,
)


def test_flow_inverse_recovers_input():
    # Equal dims → the flow is an exact bijection: inverse(forward(x)) == x.
    torch.manual_seed(0)
    model = FlowTranslator(input_dim=16, output_dim=16, num_blocks=2, coupling_hidden=32)
    model.eval()
    x = torch.randn(8, 16)
    with torch.no_grad():
        y = model(x)          # runs data-dependent ActNorm init on the way through
        x_rec = model.inverse(y)
    assert torch.allclose(x, x_rec, atol=1e-4)


def test_procrustes_recovers_known_rotation():
    torch.manual_seed(0)
    D, N = 12, 400
    # a known orthogonal Q from the QR of a random matrix
    Q, _ = torch.linalg.qr(torch.randn(D, D))
    X = torch.randn(N, D)
    Y = X @ Q.T   # so Y = X @ W.T with the true W == Q

    W, b = fit_orthogonal_procrustes(X, Y, center=False, bias=False)
    assert b is None
    assert torch.allclose(W, Q, atol=1e-4)
    assert torch.allclose(X @ W.T, Y, atol=1e-4)
    # W is orthogonal here, so the optimal LS scale is ~1
    assert abs(procrustes_scale(X, Y, W, center=False) - 1.0) < 1e-3


def test_procrustes_semiorthogonal_preserves_norm():
    # out > in → W has orthonormal columns, so ||W v|| == ||v||.
    torch.manual_seed(0)
    X = torch.randn(200, 8)
    Y = torch.randn(200, 16)
    W, _ = fit_orthogonal_procrustes(X, Y, center=True, bias=True)
    v = torch.randn(8)
    assert torch.allclose((W @ v).norm(), v.norm(), atol=1e-4)
