#!/usr/bin/env python3
"""Generate the anchored-translator study configs (config/anchored/).

The study asks one question: does a trained translator that STARTS from the
closed-form orthogonal Procrustes solution (``translator.anchor = "procrustes"``,
see models/translator.AnchoredTranslator) beat that floor, given that every
from-scratch translator measured so far LOSES to it?

Two stages, because GPU time is the scarce resource here (one 11 GB GTX 1080 Ti):

  Stage A — closed form, no training. One config per target model; the target
    layer is overridden on the CLI, so a single config covers the whole cached
    layer sweep:

        for k in <LAYERS>; do
          conda run -n acteng python fit_procrustes.py \
            --config config/anchored/procrustes_<tag>.toml \
            --source-layer 8 --target-layer $k
        done

    Scoring those with compare_translated_and_original.py geometric picks each
    pair's best target layer by mean cosine to the native SV.

  Stage B — the actual training, ONLY at each pair's best layer (plus Llama l8 as
    a control that separates a layer effect from an anchor effect). Two runs per
    (pair, layer): mlp + cosine+info_nce, once anchored and once from scratch.
    They are each other's control, and ``experiment_slug`` keeps them in separate
    files (``mlp+procrustes`` vs ``mlp``).

WHY normalize_activations = false EVERYWHERE here, including the trained runs.
The historical comparison was confounded: the trained sweeps used
normalize_activations = true while Procrustes used raw activations, so "trained
loses to Procrustes" mixed an architecture effect with a preprocessing effect.
Both losses used here (cosine, info_nce) are scale-invariant, so raw is safe for
them — and it puts the anchor, its residual and the baseline in one preprocessing
regime. It is also *required* for the anchored runs: the anchor is fitted on the
trainer's own data, and an anchor fitted on L2-normalized rows is not the
Procrustes floor these numbers are compared against.

Re-run this script any time to regenerate the configs from scratch.
"""

from pathlib import Path

from acttrans.config_gen.common import LIMIT, TRANSLATOR_BLOCKS, toml_list

from acttrans.constants import LAYER as SOURCE_LAYER, SOURCE_MODEL

OUT_DIR = Path(__file__).resolve().parent / "anchored"

# The four target models with BOTH cached FineWeb activations under
# data/fineweb/activations/ and (Phi excepted, see below) native CAA steering
# vectors in the activation_engineering tree. `layers` lists exactly the cached
# target layers — this study never extracts new activations.
#
# `batch_size` is the activation-EXTRACTION batch size carried over from the
# per-model configs in config/new_models/ (unused here, since activations are
# already cached, but kept so these configs stay drop-in for extraction too).
PAIRS = {
    "llama3b": {
        "target": "meta-llama/Llama-3.2-3B-Instruct",
        "layers": [8, 10, 12, 14, 16, 18],
        "batch_size": 32,
        "note": "same family, 2048 -> 3072. The reference pair: Procrustes peaks at l12.",
    },
    "qwen0.5b": {
        "target": "Qwen/Qwen2.5-0.5B-Instruct",
        "layers": [8, 10, 12, 14, 16],
        "batch_size": 32,
        "note": "cross-family, 2048 -> 896 (target SMALLER than source).",
    },
    "gemma1b": {
        "target": "google/gemma-3-1B-it",
        "layers": [8, 10, 12, 14, 16, 18],
        "batch_size": 32,
        "note": ("cross-family, 2048 -> 1152. Known pathological in this repo "
                 "(residual scale ~550x + direction collapse onto the mean), so "
                 "near-zero cosines are expected, not a bug."),
    },
    "phitinymoe": {
        "target": "microsoft/Phi-tiny-MoE-instruct",
        "layers": [10, 12, 14, 16, 18, 20, 22],
        "batch_size": 16,
        "note": ("cross-family sparse MoE, 2048 -> 4096. CAVEAT: the tree has native "
                 "CAA SVs for this model at layer 8 / myopic-reward ONLY, and no "
                 "cached Phi l8 activations, so this pair cannot be scored against "
                 "native SVs at all — it is judged on held-out reconstruction only, "
                 "at the mid-depth layer (l16 = 50% of 32 layers)."),
    },
}

# Stage-B target layers per pair: each pair's BEST layer as measured in Stage A by
# mean cosine to the native SV (outputs/anchored/cmp_stageA/geometric/
# per_translator.csv), except Phi, which has no native SVs to score and therefore
# uses its mid-depth layer. Llama additionally keeps l8 as the control that tells a
# layer effect apart from an anchor effect.
#
# Measured Stage-A mean_cos (1B l8 -> target lk, 7 behaviors, CAA), best in brackets:
#   llama3b     l8 .255  l10 .298  [l12 .372]  l14 .333  l16 .263  l18 .220
#   qwen0.5b    l8 .044  l10 .150  l12 .150  [l14 .227]  l16 .166
#   gemma1b     l8 .047  l10 .165  [l12 .425]  l14 .317  l16 .184  l18 .129
#   phitinymoe  not scoreable (see PAIRS note) -> l16 by depth
#
# Note the l12 = .372 for llama3b reproduces the historical layer-sweep number
# exactly, which is the regression check on this whole pipeline. Note also that
# gemma1b is NOT near-zero here: its residual-scale pathology shows up in the A/B
# steering effect, but the raw-activation Procrustes DIRECTION at l12 is the
# strongest of all four pairs.
TRAIN_LAYERS = {
    "llama3b": [12, 8],
    "qwen0.5b": [14],
    "gemma1b": [12],
    "phitinymoe": [16],
}

# The single trained recipe: the best-performing compound loss from the loss-combo
# sweep, on the mlp architecture. Both terms are scale-invariant, which is what
# makes normalize_activations = false safe here.
LOSSES = ["cosine", "info_nce"]
LOSS_WEIGHTS = [1.0, 1.0]

EPOCHS = 1000               # repo default; early stopping usually ends runs far sooner
EARLY_STOPPING_PATIENCE = 50

OUTPUT_DIR = "outputs/anchored"
DATA_DIR = "data/fineweb"   # shared sentences.json + activation caches (read-only here)


def model_blocks(pair: dict, target_layer) -> str:
    """The [source_model]/[target_model]/[dataset] body shared by both stages.

    Not reused from acttrans.config_gen.common.base_blocks: that one hard-codes
    the Llama-3B target, and this study sweeps four different target models."""
    return f"""batch_size = {pair['batch_size']}   # activation-extraction batch size

[source_model]
name = "{SOURCE_MODEL}"
layer = {SOURCE_LAYER}
module = "residual"
token_position = -1

[target_model]
name = "{pair['target']}"
layer = {target_layer}
module = "residual"
token_position = -1

[dataset]
name = "FINEWEB"
behaviors = []          # unused for FINEWEB
split = "train"         # "train"/"generate" -> 80% ; "test" -> 20% (seed 42, doc-level)
data_root = ""          # unused for FINEWEB; subset fixed to fineweb-edu/sample-10BT
limit = {LIMIT}
"""


def procrustes_config(tag: str, pair: dict) -> str:
    """Stage A: the closed-form floor, fitted per target layer via --target-layer."""
    layers = " ".join(str(k) for k in pair["layers"])
    return f"""# AUTO-GENERATED by config/generate_anchored_configs.py — do not edit by hand.
# Stage A of the anchored-translator study: the closed-form orthogonal Procrustes
# FLOOR for {SOURCE_MODEL} l{SOURCE_LAYER} -> {pair['target']}.
# {pair['note']}
#
# `layer` below is a placeholder — fit every cached target layer by overriding it:
#   for k in {layers}; do
#     conda run -n acteng python fit_procrustes.py \\
#       --config config/anchored/procrustes_{tag}.toml --source-layer {SOURCE_LAYER} --target-layer $k
#   done
#
# normalize_activations = false: an orthogonal map is norm/inner-product
# preserving, so it must see RAW activations; L2-normalizing first would distort
# exactly the geometry the map exists to preserve. The trained configs in this
# directory match it, so the floor and the trained runs are finally comparable.

{model_blocks(pair, pair['layers'][0])}
[translator]
type = "linear"
bias = true

[training]
data_dir = "{DATA_DIR}"        # sentences.json + activation caches (shared, read-only)
output_dir = "{OUTPUT_DIR}"    # Procrustes baselines land beside the trained runs
seed = 42
train_ratio = 0.8
normalize_activations = false     # RAW activations — see header
"""


def trained_config(tag: str, pair: dict, target_layer: int, anchored: bool) -> str:
    """Stage B: mlp + cosine+info_nce, anchored or from scratch (its own control)."""
    if anchored:
        anchor_block = """anchor = "procrustes"    # frozen closed-form anchor; gate starts at 0,
                         # so the run STARTS at the Procrustes floor and can
                         # only add what the orthogonal map misses
gate_init = 0.0"""
        what = ("ANCHORED on the frozen closed-form Procrustes map "
                "(y = W x + b + gate * mlp(x))")
    else:
        anchor_block = 'anchor = "none"          # from-scratch CONTROL for the anchored run'
        what = "FROM SCRATCH — the control the anchored run is measured against"
    return f"""# AUTO-GENERATED by config/generate_anchored_configs.py — do not edit by hand.
# Stage B of the anchored-translator study: {what}.
# {SOURCE_MODEL} l{SOURCE_LAYER} -> {pair['target']} l{target_layer}, mlp, losses={'+'.join(LOSSES)}.
#
# Target layer {target_layer} comes from Stage A (see TRAIN_LAYERS in the generator).
# The anchored and from-scratch checkpoints differ only in [translator].anchor and
# land in separate files ({'mlp+procrustes' if anchored else 'mlp'}), so they are a clean pair.
#
# normalize_activations = false, unlike the historical trained sweeps: cosine and
# info_nce are both scale-invariant, and raw activations put this run in the SAME
# preprocessing as the Procrustes baseline it is being compared to (the old
# comparison confounded architecture with preprocessing). It is also required for
# the anchored variant — an anchor fitted on L2-normalized rows is not the floor.

{model_blocks(pair, target_layer)}
[translator]
{TRANSLATOR_BLOCKS['mlp']}
{anchor_block}

[training]
epochs = {EPOCHS}
batch_size = 2048
lr = 1e-4
weight_decay = 1e-4
train_ratio = 0.8
seed = 42
output_dir = "{OUTPUT_DIR}"    # keeps this study out of outputs/fineweb's A/B globs
data_dir = "{DATA_DIR}"        # sentences.json + activation caches (shared, read-only)
normalize_activations = false     # RAW — see header
losses = {toml_list(LOSSES)}
loss_weights = {toml_list(LOSS_WEIGHTS)}
temperature = 0.1
grad_clip = 1.0
lr_warmup_epochs = 10
early_stopping_patience = {EARLY_STOPPING_PATIENCE}
early_stopping_min_delta = 1e-4
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for tag, pair in PAIRS.items():
        name = f"procrustes_{tag}.toml"
        (OUT_DIR / name).write_text(procrustes_config(tag, pair))
        written.append(name)

        for layer in TRAIN_LAYERS.get(tag, []):
            for anchored in (True, False):
                suffix = "anchored" if anchored else "scratch"
                name = f"mlp_{tag}_l{layer}_{suffix}.toml"
                (OUT_DIR / name).write_text(
                    trained_config(tag, pair, layer, anchored)
                )
                written.append(name)

    print(f"Wrote {len(written)} configs to {OUT_DIR}/")
    for n in sorted(written):
        print(f"  {n}")


if __name__ == "__main__":
    main()
