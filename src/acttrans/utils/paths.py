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


# The non-default extraction modes, and the filename suffix each one earns. This
# is the single source of truth for the convention: model_slug() writes these
# suffixes, and checkpoints.py / results_io.py / comparison.common read them back.
# Keeping one dict means adding a mode can never leave a reader classifying it as
# last-token (the failure mode that silently reuses the wrong activation cache).
#
#   mean -> mask-weighted mean over every real token
#   mask -> hidden state at an appended [MASK] position (masked-diffusion LMs)
#
# Last-token extraction (token_position = -1, the default) is deliberately absent:
# it keeps the bare slug so every pre-existing cache stays valid.
POOLING_SUFFIXES = {"mean": "_mean", "mask": "_mask"}

# For readers that parse a slug back apart: "(?:_mean|_mask)?"
POOLING_SUFFIX_RE = "(?:%s)?" % "|".join(POOLING_SUFFIXES.values())


def pooling_of(model_cfg) -> str:
    """The extraction mode of a model block: 'last' | 'mean' | 'mask'.

    Any int token_position reads as 'last' for labelling purposes — only -1 is
    used in practice, and a fixed index is still a single-position readout."""
    tp = str(model_cfg.get("token_position", -1))
    return tp if tp in POOLING_SUFFIXES else "last"


def pooling_suffix(model_cfg) -> str:
    """The filename suffix earned by this model block's extraction mode ('' for last)."""
    return POOLING_SUFFIXES.get(str(model_cfg.get("token_position", -1)), "")


def model_slug(model_cfg) -> str:
    """e.g. {"name": "meta-llama/Llama-3.2-3B-Instruct", "layer": 8} -> "Llama-3.2-3B-Instruct_l8".

    Non-default extraction modes get a suffix (``_mean``, ``_mask``) so their
    activation caches and translator checkpoints never collide with the
    last-token ones on the same sentence set. Last-token (the default) keeps the
    bare slug for backward compatibility with existing caches.

    The collision matters more than it looks: extraction skips any cache whose
    ``n_sentences`` already matches, so two modes sharing a slug would not merely
    overwrite each other — the second would silently *reuse the first's
    activations* and report success."""
    short = str(model_cfg.get("name", "model")).split("/")[-1]
    return f"{short}_l{model_cfg.get('layer', '?')}{pooling_suffix(model_cfg)}"


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


def resolve_losses(tcfg) -> tuple:
    """Resolve the training losses from a [training] table: supports `losses`
    (list) and the legacy `loss` (single string). Returns (names, weights);
    weights default to 1.0 per loss. Single source of truth for the trainer
    and for the loss tag baked into checkpoint filenames."""
    raw = tcfg.get("losses") or [tcfg.get("loss", "mse")]
    names = raw if isinstance(raw, list) else [raw]
    weights = tcfg.get("loss_weights", [1.0] * len(names))
    return names, weights


def loss_tag(config) -> str:
    """Loss identifier: '+'-joined loss names (e.g. 'mse', 'mse+info_nce')."""
    names, _ = resolve_losses(config.get("training", {}))
    return "+".join(names)


def experiment_slug(config) -> str:
    """Identifies one translator experiment: model pair + translator type + loss.

    Activations are keyed by model alone (shared across experiments on the same
    sentences); translators add type+loss so an MLP/MSE run, a cosine run, and an
    SAE run on the same model pair never overwrite each other's checkpoint.

    An anchored translator (``translator.anchor``, e.g. a frozen Procrustes anchor
    with a trained residual on top) reads as ``{type}+{anchor}`` — e.g.
    ``mlp+procrustes`` — because it is a genuinely different model from the
    from-scratch ``mlp`` on the same pair and loss, and the two are trained as each
    other's control. Without the suffix they would overwrite one another."""
    s = config.get("source_model", {})
    t = config.get("target_model", {})
    tcfg = config.get("translator", {})
    tr_type = tcfg.get("type", "mlp")
    anchor = str(tcfg.get("anchor", "") or "").lower()
    if anchor and anchor != "none":
        tr_type = f"{tr_type}+{anchor}"
    return f"{translator_pair_slug(s, t)}__{tr_type}__{loss_tag(config)}"


def best_translator_path(output_dir, config) -> Path:
    return Path(output_dir) / f"best_translator__{experiment_slug(config)}.pt"


def translator_path(output_dir, config) -> Path:
    return Path(output_dir) / f"translator__{experiment_slug(config)}.pt"


def sv_path(root, model_name: str, method: str, behavior: str, module: str, layer: int) -> Path:
    """A steering vector inside an activation_engineering-style tree:

        {root}/steering_vectors/{model}/{method}/{behavior}/{module}/layer_{idx}/sv.pt

    where {model} is the HF name with '/' replaced by '_'. This layout is shared
    by the source vectors (activation_engineering repo) and the translated ones
    written by translate_steering_vector.py."""
    return (
        Path(root)
        / "steering_vectors"
        / model_name.replace("/", "_")
        / method
        / behavior
        / module
        / f"layer_{layer}"
        / "sv.pt"
    )


def parse_sv_path(sv_path: Path) -> tuple:
    """Inverse of ``sv_path``: extract (model_slug, method, behavior, module,
    layer_idx) from an sv.pt path inside a steering_vectors/ tree."""
    parts = Path(sv_path).parts
    try:
        root_idx = parts.index("steering_vectors")
    except ValueError:
        raise ValueError(f"'steering_vectors' not found in path: {sv_path}")

    rel = parts[root_idx + 1:]
    # Expected: {model}/{method}/{behavior}/{module}/layer_{idx}/sv.pt
    if len(rel) < 6:
        raise ValueError(
            "Unexpected path structure. Expected:\n"
            "  steering_vectors/{model}/{method}/{behavior}/{module}/layer_{idx}/sv.pt\n"
            f"Got: {sv_path}"
        )

    model_slug, method, behavior, module, layer_part = rel[0], rel[1], rel[2], rel[3], rel[4]

    if not layer_part.startswith("layer_"):
        raise ValueError(f"Expected 'layer_{{idx}}', got: {layer_part!r}")
    layer_idx = int(layer_part.split("_", 1)[1])

    return model_slug, method, behavior, module, layer_idx
