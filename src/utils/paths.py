"""Path conventions.

Each *sentence set* is split across two roots:

    data/<sentence_set>/                 # inputs + derived data (data_dir)
        sentences.json
        activations/
            <model_slug>.pt              # one cache per (model, layer) on these sentences

    outputs/<sentence_set>/              # artifacts worth keeping (output_dir)
        best_translator__<src>__<tgt>.pt
        translator__<src>__<tgt>.pt
        tensorboard/

Sentences and activations live under ``data_dir`` (cheap to regenerate, often
large); only translator checkpoints and TensorBoard logs land in ``output_dir``.

Keying activation caches by model means switching the target model never
overwrites the source cache (or another target's), and keying translator
checkpoints by the model pair keeps multiple translators side-by-side in the
same sentence-set directory without clobbering each other.

These helpers take no heavy imports so every script can share them cheaply.
"""

from pathlib import Path


def data_dir_of(config) -> Path:
    """Root for sentences + activations (defaults to ``data/``)."""
    return Path(config.get("training", {}).get("data_dir", "data/"))


def sentences_path(data_dir) -> Path:
    """The sentence set persisted for a run: data_dir/sentences.json."""
    return Path(data_dir) / "sentences.json"


def model_slug(model_cfg) -> str:
    """e.g. {"name": "meta-llama/Llama-3.2-3B-Instruct", "layer": 8} -> "Llama-3.2-3B-Instruct_l8".

    Mean-pooled extraction gets a ``_mean`` suffix so its activation cache and
    translator checkpoints never collide with the last-token ones on the same
    sentence set. Last-token (the default) keeps the bare slug for backward
    compatibility with existing caches."""
    short = str(model_cfg.get("name", "model")).split("/")[-1]
    base = f"{short}_l{model_cfg.get('layer', '?')}"
    if str(model_cfg.get("token_position", -1)) == "mean":
        base += "_mean"
    return base


def activation_path(data_dir, model_cfg) -> Path:
    """Per-(model, sentence-set) activation cache: data_dir/activations/<model_slug>.pt."""
    return Path(data_dir) / "activations" / f"{model_slug(model_cfg)}.pt"


def legacy_activation_path(data_dir, role: str) -> Path:
    """Pre-refactor flat layout: data_dir/{source,target}_activations.pt."""
    return Path(data_dir) / f"{role}_activations.pt"


def resolve_activation_path(data_dir, model_cfg, role: str) -> Path:
    """Path to read activations from: prefer the per-model cache, fall back to the
    legacy flat file if it's the only one present, otherwise default to the
    per-model path (which extraction will create)."""
    new = activation_path(data_dir, model_cfg)
    if new.exists():
        return new
    legacy = legacy_activation_path(data_dir, role)
    if legacy.exists():
        return legacy
    return new


def translator_pair_slug(source_cfg, target_cfg) -> str:
    return f"{model_slug(source_cfg)}__{model_slug(target_cfg)}"


def loss_tag(config) -> str:
    """Loss identifier, mirroring the trainer: '+'-joined loss names (e.g. 'mse', 'mse+info_nce')."""
    tcfg = config.get("training", {})
    raw = tcfg.get("losses") or [tcfg.get("loss", "mse")]
    names = raw if isinstance(raw, list) else [raw]
    return "+".join(names)


def experiment_slug(config) -> str:
    """Identifies one translator experiment: model pair + translator type + loss.

    Activations are keyed by model alone (shared across experiments on the same
    sentences); translators add type+loss so an MLP/MSE run, a cosine run, and an
    SAE run on the same model pair never overwrite each other's checkpoint."""
    s = config.get("source_model", {})
    t = config.get("target_model", {})
    tr_type = config.get("translator", {}).get("type", "mlp")
    return f"{translator_pair_slug(s, t)}__{tr_type}__{loss_tag(config)}"


def best_translator_path(output_dir, config) -> Path:
    return Path(output_dir) / f"best_translator__{experiment_slug(config)}.pt"


def translator_path(output_dir, config) -> Path:
    return Path(output_dir) / f"translator__{experiment_slug(config)}.pt"
