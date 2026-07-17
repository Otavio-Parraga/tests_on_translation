#!/usr/bin/env bash
# Translate one Llama-3.2-1B-Instruct steering vector (CAA / corrigible-neutral-HHH / layer_8)
# to Llama-3.2-3B-Instruct space using the trained SAE translator (1B -> 3B, 30k set).
#
# The target model, layer and module are read from the checkpoint's stored config,
# so the output lands at:
#   steering_vectors/meta-llama_Llama-3.2-3B-Instruct/CAA/corrigible-neutral-HHH/residual/layer_8/sv.pt
#
# The folder structure mirrors activation_engineering so the vectors can be used
# there without renaming — just point its evaluation config to this repo's
# steering_vectors/ directory.
#
# Usage:
#   bash bash/translate_test.sh [--no-restore-norm]
#
# Run from the project root:
#   cd /home/parraga/projects/tests_on_translation
#   bash bash/translate_test.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACTENG_ROOT="$(cd "${REPO_ROOT}/../activation_engineering" && pwd)"

# Source steering vector: layer_8 to match the translator's source layer (l8).
SOURCE_SV="${ACTENG_ROOT}/steering_vectors/meta-llama_Llama-3.2-1B-Instruct/CAA/corrigible-neutral-HHH/residual/layer_8/sv.pt"
# SAE translator produced by: train.py --config config/sae.toml
CHECKPOINT="${REPO_ROOT}/outputs/generated_30k/best_translator__Llama-3.2-1B-Instruct_l8__Llama-3.2-3B-Instruct_l8__sae__mse.pt"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: translator checkpoint not found:" >&2
    echo "  $CHECKPOINT" >&2
    echo "Training may not have finished yet. Run: conda run -n acteng python train.py --config config/sae.toml" >&2
    exit 1
fi

echo "=== Translation test: Llama-3.2-1B -> Llama-3.2-3B (SAE) ==="
echo "Source     : ${SOURCE_SV}"
echo "Checkpoint : ${CHECKPOINT}"
echo "Output     : ${REPO_ROOT}/steering_vectors/  (this repo)"
echo ""

conda run -n acteng python "${REPO_ROOT}/translate_steering_vector.py" \
    "${SOURCE_SV}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${REPO_ROOT}" \
    "$@"
