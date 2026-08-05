"""Fixed experiment grid shared across sweeps, comparisons and config generators.

Single source of truth for the Llama-3.2 1B -> 3B FineWeb experiment: the model
pair, the steering-vector location (method/module/layer) and the seven MWE
behaviors evaluated everywhere. Import from here instead of redeclaring.
"""

SOURCE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

LAYER = 8            # SVs were extracted at layer 8 on the 1B source
METHOD = "CAA"       # default method (the original single-method experiment)
MODULE = "residual"

# Steering-vector discovery methods this repo can translate and evaluate.
#
# All three live at the same place in the activation_engineering tree
# (steering_vectors/<model>/<METHOD>/<behavior>/residual/layer_<N>/sv.pt), are
# built from the same CAA contrastive dataset, produce ONE flat direction per
# (model, behavior, layer) in the residual stream, and are applied with an
# additive hook — which is what makes them drop-in for the CAA tooling:
#
#   CAA   mean of per-pair (positive - negative) activation differences.
#         Norm is meaningful and behavior-dependent (~0.2 to ~5).
#   RepE  first principal component of the sign-alternated, per-pair-normalized
#         differences (LAT). Saved as [1, D]; unit norm by construction.
#   GCAV  normal of a logistic-regression boundary separating positive from
#         negative activations. Saved as [D]; explicitly unit-normalized.
#
# NORM CONVENTION — the one thing that is NOT comparable across methods.
# CAA carries its own magnitude while RepE/GCAV are unit vectors, so the same
# coefficient is a different physical dose per method. Compare methods on
# `dose = coefficient * sv_norm` (see method_report.py), not on raw coefficient.
METHODS = ("CAA", "RepE", "GCAV")

# Methods whose vectors are unit-norm by construction, so their `sv_norm` column
# carries no behavior information and their useful coefficient window sits at
# larger raw coefficients than CAA's.
UNIT_NORM_METHODS = ("RepE", "GCAV")

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
