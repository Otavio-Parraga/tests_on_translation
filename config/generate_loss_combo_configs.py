#!/usr/bin/env python3
"""Generate the compound-loss training sweep configs.

Matrix: translator arch {mlp, encoder, flow, sae} x loss COMBINATION
        x token pooling {last}  =  4 x len(LOSS_COMBOS) configs,
        written to config/loss_combos/.

This is the "compound loss" baseline: instead of training each translator on a
single coordinate loss (mse | cosine | info_nce, as in the FineWeb sweep), we
train on WEIGHTED COMBINATIONS of them. The trainer already supports this via
`losses = [...]` + `loss_weights = [...]`; this generator just emits the configs.

Everything else (models, layer, data_dir, output_dir, epochs, ...) is IDENTICAL
to config/fineweb so:
  - the SAME activation caches under data/fineweb/activations/ are reused
    (no re-extraction needed), and
  - the checkpoints land in outputs/fineweb/ with names like
        best_translator__..__mlp__mse+cosine+info_nce.pt
    (the loss tag is the '+'-joined loss names — see src/utils/paths.py), so
    run_ab_sweep.py's glob `outputs/fineweb/best_translator__*.pt` picks them up
    alongside the single-loss checkpoints automatically.

Only last-token pooling is emitted here to keep the matrix small. To also sweep
mean-pooling, add "mean" to POOLINGS below (this DOUBLES the config count and
requires the mean activation caches to exist).

Re-run this script any time to regenerate the configs from scratch.
"""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "loss_combos"

SOURCE_NAME = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_NAME = "meta-llama/Llama-3.2-3B-Instruct"
LAYER = 8
LIMIT = 30000

# ---------------------------------------------------------------------------
# The compound-loss matrix. Each entry is (list_of_loss_names, list_of_weights).
# Weights are per-loss coefficients applied inside the trainer's `criterion`
# (loss = sum(w_i * loss_i)). To add a future relational loss (e.g. "vsp"),
# simply append a new tuple here, e.g.
#     (["mse", "vsp"], [1.0, 1.0]),
# once the trainer's _make_loss_fn knows how to build it. Nothing else changes:
# the checkpoint name / A/B glob pick up the new '+'-joined tag automatically.
#
# Weight rationale: mse is on a raw-magnitude scale while cosine/info_nce are
# bounded ~O(1); on L2-normalized activations (normalize_activations = true)
# all three sit at comparable scales, so equal 1.0 weights are a sane baseline.
# ---------------------------------------------------------------------------
LOSS_COMBOS = [
    (["mse", "cosine"], [1.0, 1.0]),
    (["mse", "info_nce"], [1.0, 1.0]),
    (["cosine", "info_nce"], [1.0, 1.0]),
    (["mse", "cosine", "info_nce"], [1.0, 1.0, 1.0]),
]

# Token pooling. Keep to {last} by default; add "mean" to also sweep mean-pooled
# (doubles the matrix and needs the *_mean activation caches).
POOLINGS = {"last": "-1"}  # token_position value, verbatim TOML

# Per-architecture [translator] block (everything below the `type` line).
# Kept byte-identical to config/generate_fineweb_configs.py.
TRANSLATOR_BLOCKS = {
    "mlp": """type = "mlp"
hidden_dims = [2048, 2048, 2048]
dropout = 0.1
activation = "gelu"
use_residual = false""",
    "encoder": """type = "encoder"
d_model = 512
nhead = 8
num_layers = 4
dropout = 0.1""",
    "flow": """type = "flow"
num_blocks = 8           # stacked [ActNorm -> mixing -> AffineCoupling] blocks
coupling_hidden = 1024
activation = "gelu"
mixing = "permutation"   # "permutation" (stable) | "inv1x1" (Glow, can blow up at large D)""",
    "sae": """type = "sae"
latent_dim = 16384         # 8x overcomplete bottleneck (8 * 2048)
sae_activation = "topk"    # "topk" (architectural sparsity) | "l1" (penalty-based)
k = 64
l1_coeff = 1e-3
normalize_decoder = true""",
}


def _toml_list(values) -> str:
    """Render a python list of strings/numbers as a TOML inline array."""
    parts = []
    for v in values:
        parts.append(f'"{v}"' if isinstance(v, str) else str(v))
    return "[" + ", ".join(parts) + "]"


def combo_slug(loss_names) -> str:
    """Filename-safe tag: '+'-joined loss names (matches the checkpoint loss_tag)."""
    return "+".join(loss_names)


def training_block(loss_names, loss_weights, arch: str) -> str:
    extra = ""
    if arch == "flow":
        # A reversible flow learns target->source for free; supervise it too.
        extra = "\ninverse_loss_weight = 1.0"
    return f"""[training]
epochs = 1000
batch_size = 2048
lr = 1e-4
weight_decay = 1e-4
train_ratio = 0.8
seed = 42
output_dir = "outputs/fineweb"   # checkpoints land beside the single-loss ones (shared A/B glob)
data_dir = "data/fineweb"        # sentences.json + activation caches (shared with the fineweb sweep)
normalize_activations = true
losses = {_toml_list(loss_names)}
loss_weights = {_toml_list(loss_weights)}
temperature = 0.1
grad_clip = 1.0
lr_warmup_epochs = 10
early_stopping_patience = 50
early_stopping_min_delta = 1e-4{extra}
"""


def make_config(arch: str, loss_names, loss_weights, pool: str) -> str:
    tp = POOLINGS[pool]
    tag = combo_slug(loss_names)
    return f"""# AUTO-GENERATED by config/generate_loss_combo_configs.py — do not edit by hand.
# Compound-loss sweep: translator={arch}  losses={tag}  token={pool}
# Shared FineWeb split (sample-10BT, seed 42, limit {LIMIT}); reuses data/fineweb activation caches.

batch_size = 32   # activation-extraction batch size

[source_model]
name = "{SOURCE_NAME}"
layer = {LAYER}
module = "residual"
token_position = {tp}

[target_model]
name = "{TARGET_NAME}"
layer = {LAYER}
module = "residual"
token_position = {tp}

[dataset]
name = "FINEWEB"
behaviors = []          # unused for FINEWEB
split = "train"         # "train"/"generate" -> 80% ; "test" -> 20% (seed 42, doc-level)
data_root = ""          # unused for FINEWEB; subset fixed to fineweb-edu/sample-10BT
limit = {LIMIT}

[translator]
{TRANSLATOR_BLOCKS[arch]}

{training_block(loss_names, loss_weights, arch)}"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for arch in TRANSLATOR_BLOCKS:
        for loss_names, loss_weights in LOSS_COMBOS:
            for pool in POOLINGS:
                name = f"{arch}_{combo_slug(loss_names)}_{pool}.toml"
                (OUT_DIR / name).write_text(
                    make_config(arch, loss_names, loss_weights, pool)
                )
                written.append(name)
    print(f"Wrote {len(written)} configs to {OUT_DIR}/")
    for n in written:
        print(f"  {n}")


if __name__ == "__main__":
    main()
