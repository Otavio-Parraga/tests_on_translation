#!/usr/bin/env bash
# Experiment: train translator on 15k sentences sampled from each model.
# All outputs are isolated under outputs/generated_15k/.
# Does not touch any existing files in outputs/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
CONFIG="$ROOT/config/generated_30k.toml"
SENTENCES="$ROOT/outputs/generated_30k/sentences.json"

cd "$ROOT"

mkdir -p outputs/generated_30k

# Step 1 — sample sentences (skip if already done)
if [ -f "$SENTENCES" ]; then
    echo "[1/3] sentences already exist at $SENTENCES, skipping generation."
else
    echo "[1/3] Sampling 300k sentences from each model..."
    python sample_sentences.py \
        --config "$CONFIG" \
        --n 300000 \
        --output "$SENTENCES" \
        --temperature 1.0 \
        --max_new_tokens 128
fi

# Step 2 — extract paired activations (skip if both already exist)
SRC="$ROOT/outputs/generated_30k/activations/Llama-3.2-1B-Instruct_l8.pt"
TGT="$ROOT/outputs/generated_30k/activations/gemma-3-1B-it_l8.pt"

if [ -f "$SRC" ] && [ -f "$TGT" ]; then
    echo "[2/3] activations already exist, skipping extraction."
else
    echo "[2/3] Extracting paired activations..."
    python prepare_activations.py --config "$CONFIG"
fi

# Step 3 — train
echo "[3/3] Training translator..."
python train.py --config "$CONFIG"

echo "Done. Outputs in outputs/generated_30k/"
