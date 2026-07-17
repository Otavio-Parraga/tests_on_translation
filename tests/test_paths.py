"""Slug / path round-trips and loss-tag resolution — pure, no I/O."""
from pathlib import Path

from acttrans.utils.paths import (
    best_translator_path,
    experiment_slug,
    loss_tag,
    model_slug,
    parse_sv_path,
    resolve_losses,
    sv_path,
)


def test_model_slug_last_vs_mean():
    cfg = {"name": "meta-llama/Llama-3.2-3B-Instruct", "layer": 8}
    assert model_slug(cfg) == "Llama-3.2-3B-Instruct_l8"
    assert model_slug({**cfg, "token_position": "mean"}) == "Llama-3.2-3B-Instruct_l8_mean"
    # last-token (default) keeps the bare slug
    assert model_slug({**cfg, "token_position": -1}) == "Llama-3.2-3B-Instruct_l8"


def test_sv_path_roundtrip():
    p = sv_path("root", "meta-llama/Llama-3.2-3B-Instruct", "CAA", "sycophancy", "residual", 8)
    # model name '/' is slugged to '_' in the tree
    assert p == Path("root/steering_vectors/meta-llama_Llama-3.2-3B-Instruct/"
                     "CAA/sycophancy/residual/layer_8/sv.pt")
    model, method, behavior, module, layer = parse_sv_path(p)
    assert model == "meta-llama_Llama-3.2-3B-Instruct"
    assert (method, behavior, module, layer) == ("CAA", "sycophancy", "residual", 8)


def test_resolve_losses_legacy_and_list():
    # legacy single-string `loss`
    assert resolve_losses({"loss": "cosine"}) == (["cosine"], [1.0])
    # default when nothing set
    assert resolve_losses({}) == (["mse"], [1.0])
    # `losses` list with default per-loss weights
    names, weights = resolve_losses({"losses": ["mse", "info_nce"]})
    assert names == ["mse", "info_nce"] and weights == [1.0, 1.0]
    # explicit weights honored
    assert resolve_losses({"losses": ["mse", "vsp"], "loss_weights": [1.0, 0.5]}) == (
        ["mse", "vsp"], [1.0, 0.5]
    )


def test_loss_tag_and_experiment_slug():
    cfg = {
        "source_model": {"name": "a/Llama-1B", "layer": 8},
        "target_model": {"name": "a/Llama-3B", "layer": 8},
        "translator": {"type": "mlp"},
        "training": {"losses": ["mse", "cosine"]},
    }
    assert loss_tag(cfg) == "mse+cosine"
    assert experiment_slug(cfg) == "Llama-1B_l8__Llama-3B_l8__mlp__mse+cosine"
    assert best_translator_path("outputs/fineweb", cfg) == Path(
        "outputs/fineweb/best_translator__Llama-1B_l8__Llama-3B_l8__mlp__mse+cosine.pt"
    )
