"""Shared building blocks for the sweep config generators.

Both generators (generate_fineweb_configs.py, generate_loss_combo_configs.py)
emit configs over the SAME FineWeb split and model pair; the model/dataset TOML
blocks and per-architecture [translator] blocks live here so the two matrices
can never drift apart. The experiment grid itself (model names, layer, limit)
comes from acttrans.constants.
"""

from acttrans.constants import (
    FINEWEB_LIMIT as LIMIT,
    LAYER,
    SOURCE_MODEL as SOURCE_NAME,
    TARGET_MODEL as TARGET_NAME,
)

# Per-architecture [translator] block (everything below the `type` line).
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


def toml_list(values) -> str:
    """Render a python list of strings/numbers as a TOML inline array."""
    parts = []
    for v in values:
        parts.append(f'"{v}"' if isinstance(v, str) else str(v))
    return "[" + ", ".join(parts) + "]"


def base_blocks(tp: str) -> str:
    """The shared body of every sweep config: extraction batch size, the two
    model blocks and the FineWeb dataset block. `tp` is the token_position
    value, verbatim TOML (e.g. '-1' or '\"mean\"')."""
    return f"""batch_size = 32   # activation-extraction batch size

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
"""


def training_block(loss_names, loss_weights, arch: str,
                   output_dir_comment: str, data_dir_comment: str) -> str:
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
output_dir = "outputs/fineweb"   # {output_dir_comment}
data_dir = "data/fineweb"        # {data_dir_comment}
normalize_activations = true
losses = {toml_list(loss_names)}
loss_weights = {toml_list(loss_weights)}
temperature = 0.1
grad_clip = 1.0
lr_warmup_epochs = 10
early_stopping_patience = 50
early_stopping_min_delta = 1e-4{extra}
"""
