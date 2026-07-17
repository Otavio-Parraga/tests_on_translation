#!/usr/bin/env bash
# Experiment: Llama-3.2-3B-Instruct as the TARGET model, trained against the
# Llama-3.2-1B-Instruct source on the existing 30k generated sentence set.
#
# Reuses, never overwrites:
#   - outputs/generated_30k/sentences.json                 (same sentences)
#   - outputs/generated_30k/activations/Llama-3.2-1B-Instruct_l8.pt  (cached source)
# Produces, side-by-side with the existing gemma run:
#   - outputs/generated_30k/activations/Llama-3.2-3B-Instruct_l8.pt  (new target acts)
#   - outputs/generated_30k/best_translator__Llama-3.2-1B-Instruct_l8__Llama-3.2-3B-Instruct_l8__mlp__mse.pt
#   - outputs/generated_30k/translator__...__Llama-3.2-3B-Instruct_l8__mlp__mse.pt
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
CONFIG="$ROOT/config/llama3b_30k.toml"
SENTENCES="$ROOT/outputs/generated_30k/sentences.json"

cd "$ROOT"

if [ ! -f "$SENTENCES" ]; then
    echo "ERROR: expected sentence set not found at $SENTENCES" >&2
    exit 1
fi

# Step 1 — extract activations (Llama-1B source is reused from cache; only the
# new Llama-3B target is actually computed). Idempotent: re-running skips work
# that is already complete.
echo "[1/2] Extracting paired activations (target = Llama-3.2-3B-Instruct)..."
conda run -n acteng python prepare_activations.py --config "$CONFIG"

# Step 2 — train the translator with the default-config settings.
echo "[2/2] Training translator..."
conda run -n acteng python train.py --config "$CONFIG"

echo "Done. Outputs in outputs/generated_30k/ (Llama-3.2-3B target)."
