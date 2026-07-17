"""Fixed experiment grid shared across sweeps, comparisons and config generators.

Single source of truth for the Llama-3.2 1B -> 3B FineWeb experiment: the model
pair, the CAA steering-vector location (method/module/layer) and the seven MWE
behaviors evaluated everywhere. Import from here instead of redeclaring.
"""

SOURCE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

LAYER = 8            # CAA SVs were extracted at layer 8
METHOD = "CAA"
MODULE = "residual"

FINEWEB_LIMIT = 30000  # sentence cap of the shared FineWeb split (sample-10BT, seed 42)

BEHAVIORS = [
    "coordinate-other-ais",
    "corrigible-neutral-HHH",
    "hallucination",
    "myopic-reward",
    "refusal",
    "survival-instinct",
    "sycophancy",
]
