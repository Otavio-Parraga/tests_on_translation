"""Structural properties of the translator models and the Procrustes fit."""
import pytest
import torch
import torch.nn.functional as F

from acttrans.models.translator import (
    AnchoredTranslator,
    FlowTranslator,
    MLPTranslator,
    build_translator,
    fit_orthogonal_procrustes,
    fit_procrustes_anchor,
    load_translator,
    procrustes_scale,
    save_translator,
)


def _anchored(input_dim=8, output_dim=12, seed=0):
    """A small AnchoredTranslator with a random (non-zero) anchor loaded."""
    torch.manual_seed(seed)
    base = MLPTranslator(input_dim, output_dim, hidden_dims=[16], dropout=0.0,
                         activation="gelu")
    model = AnchoredTranslator(base, input_dim=input_dim, output_dim=output_dim)
    model.set_anchor(torch.randn(output_dim, input_dim), torch.randn(output_dim))
    return model.eval()


def _anchored_config(anchor="procrustes", ttype="mlp"):
    cfg = {
        "source_model": {"hidden_dim": 8},
        "target_model": {"hidden_dim": 12},
        "translator": {"type": ttype, "hidden_dims": [16], "dropout": 0.0},
    }
    if anchor is not None:
        cfg["translator"]["anchor"] = anchor
    return cfg


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


# ── AnchoredTranslator ───────────────────────────────────────────────────────

def test_anchored_starts_exactly_at_its_anchor():
    # gate init = 0 → the wrapper IS the anchor, so the whole point of the design
    # (training starts at the Procrustes floor, never below it) holds at step 0.
    model = _anchored()
    x = torch.randn(5, 8)
    with torch.no_grad():
        assert model.gate.item() == 0.0
        assert torch.allclose(model(x), model.anchor(x), atol=1e-6)
        # forward_direction drops the anchor bias (the bias cancels on a difference)
        assert torch.allclose(
            model.forward_direction(x), F.linear(x, model.anchor_W), atol=1e-6
        )


def test_anchor_gets_no_gradient_but_base_and_gate_do():
    model = _anchored()
    # a non-zero gate so the base branch is actually on the graph
    with torch.no_grad():
        model.gate.fill_(0.5)
    model(torch.randn(5, 8)).pow(2).sum().backward()

    # buffers have no .grad slot at all — frozen by construction, not by convention
    assert not model.anchor_W.requires_grad
    assert model.anchor_W.grad is None and model.anchor_b.grad is None
    assert model.gate.grad is not None and model.gate.grad.abs() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.base.parameters())


def test_anchored_state_dict_roundtrips_through_save_load(tmp_path):
    # The anchor lives in buffers, so it must survive save_translator/load_translator
    # like a parameter would — otherwise a reloaded checkpoint would silently lose
    # the entire closed-form map and keep only the residual.
    cfg = _anchored_config()
    model = build_translator(cfg, input_dim=8, output_dim=12)
    model.set_anchor(torch.randn(12, 8), torch.randn(12))
    with torch.no_grad():
        model.gate.fill_(0.3)
    model.eval()

    path = tmp_path / "best_translator__a_l8__b_l8__mlp+procrustes__cosine.pt"
    save_translator(model, path, cfg, input_dim=8, output_dim=12)
    reloaded = load_translator(path).eval()

    assert "anchor_W" in model.state_dict()
    x = torch.randn(4, 8)
    with torch.no_grad():
        assert torch.allclose(model(x), reloaded(x), atol=1e-6)
        assert torch.allclose(reloaded.anchor_W, model.anchor_W)


def test_fit_procrustes_anchor_recovers_a_rotation():
    torch.manual_seed(0)
    D, N = 12, 400
    Q, _ = torch.linalg.qr(torch.randn(D, D))
    X = torch.randn(N, D)
    Y = X @ Q.T

    model = AnchoredTranslator(
        MLPTranslator(D, D, hidden_dims=[16], dropout=0.0, activation="gelu"),
        input_dim=D, output_dim=D,
    ).eval()
    s = fit_procrustes_anchor(model, X, Y)

    assert abs(s - 1.0) < 1e-2          # an orthogonal map needs no rescaling
    with torch.no_grad():
        cos = F.cosine_similarity(model(X), Y, dim=-1).mean().item()
    assert cos > 0.99                    # gate=0 → this is the anchor's own cosine


def test_build_translator_rejects_bad_anchors():
    with pytest.raises(ValueError, match="Unknown translator anchor"):
        build_translator(_anchored_config(anchor="orthogonal"), 8, 12)
    # anchoring a linear translator is an affine map plus an affine map — a no-op
    with pytest.raises(ValueError, match="redundant"):
        build_translator(_anchored_config(ttype="linear"), 8, 12)
    # unset / "none" / "" leave the base model untouched
    for anchor in (None, "none", ""):
        assert not isinstance(
            build_translator(_anchored_config(anchor=anchor), 8, 12), AnchoredTranslator
        )
