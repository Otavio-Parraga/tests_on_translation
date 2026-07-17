#!/usr/bin/env bash
#
# Set up and run the full FineWeb translator sweep.
#
#   24 experiments = translator {mlp, encoder, flow, sae}
#                  x loss        {mse, cosine, info_nce}
#                  x token       {last, mean}
#
# Every run trains on the SAME FineWeb split (sample-10BT, seed 42, limit 30000).
# Activations are extracted ONCE per token-pooling and reused by all runs:
#   data/fineweb/activations/<model>_l8.pt        (last token)
#   data/fineweb/activations/<model>_l8_mean.pt   (mean pooled)
# Translator checkpoints + TensorBoard logs land in outputs/fineweb/.
#
# Usage:
#   ./run_fineweb_sweep.sh            # extract (both poolings) + train+eval all 24
#   ./run_fineweb_sweep.sh extract    # only extract the shared activations
#   ./run_fineweb_sweep.sh train      # only train+eval (assumes activations exist)

set -euo pipefail
cd "$(dirname "$0")"   # repo root (so .env / load_dotenv finds HF_CACHE_DIR)

PY="conda run -n acteng python"

# HuggingFace Xet streaming backend has been network-flaky here; the classic
# resolver is slower but reliable. Remove this line to re-enable Xet.
export HF_HUB_DISABLE_XET=1

CONFIG_DIR="config/fineweb"
DATA_DIR="data/fineweb"
ACT_DIR="$DATA_DIR/activations"

# The two configs used purely to drive extraction (translator/loss is irrelevant
# for extraction — only the token_position differs between them).
LAST_CFG="$CONFIG_DIR/mlp_mse_last.toml"
MEAN_CFG="$CONFIG_DIR/mlp_mse_mean.toml"

# Activation caches every training run depends on.
REQUIRED_ACTS=(
  "$ACT_DIR/Llama-3.2-1B-Instruct_l8.pt"
  "$ACT_DIR/Llama-3.2-3B-Instruct_l8.pt"
  "$ACT_DIR/Llama-3.2-1B-Instruct_l8_mean.pt"
  "$ACT_DIR/Llama-3.2-3B-Instruct_l8_mean.pt"
)

generate_configs() {
  echo "==> Regenerating sweep configs"
  $PY config/generate_fineweb_configs.py
}

extract() {
  echo "==> Extracting LAST-token activations (shared FineWeb split)"
  $PY prepare_activations.py --config "$LAST_CFG"
  echo "==> Extracting MEAN-pooled activations (same FineWeb split)"
  $PY prepare_activations.py --config "$MEAN_CFG"
}

preflight() {
  echo "==> Pre-flight: verifying everything needed for last-token AND mean runs"
  local ok=1
  if [[ ! -f "$DATA_DIR/sentences.json" ]]; then
    echo "   MISSING: $DATA_DIR/sentences.json"; ok=0
  fi
  for f in "${REQUIRED_ACTS[@]}"; do
    if [[ ! -f "$f" ]]; then echo "   MISSING: $f"; ok=0; else echo "   ok: $f"; fi
  done
  if [[ $ok -ne 1 ]]; then
    echo "Pre-flight FAILED — run './run_fineweb_sweep.sh extract' first." >&2
    exit 1
  fi
  echo "   Pre-flight OK."
}

train_all() {
  local total=0 done=0
  total=$(ls -1 "$CONFIG_DIR"/*.toml | wc -l)
  echo "==> Training + evaluating $total configs"
  for cfg in "$CONFIG_DIR"/*.toml; do
    done=$((done + 1))
    echo
    echo "---- [$done/$total] $cfg ----"
    $PY train.py --config "$cfg"
    $PY evaluate.py --config "$cfg"
  done
  echo
  echo "==> Sweep complete. Checkpoints in outputs/fineweb/  (tensorboard --logdir outputs/fineweb/tensorboard)"
}

MODE="${1:-all}"
case "$MODE" in
  extract)
    generate_configs; extract; preflight ;;
  train)
    preflight; train_all ;;
  all)
    generate_configs; extract; preflight; train_all ;;
  *)
    echo "Unknown mode: $MODE (expected: extract | train | all)" >&2; exit 1 ;;
esac
