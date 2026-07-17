#!/usr/bin/env bash
#
# Set up and run the COMPOUND-LOSS translator sweep (Topic 1 of the KD analysis).
#
#   N experiments = translator {mlp, encoder, flow, sae}
#                 x loss COMBO  {mse+cosine, mse+info_nce, cosine+info_nce,
#                                mse+cosine+info_nce}
#                 x token       {last}
#
# These reuse the EXACT SAME FineWeb split + activation caches as the single-loss
# FineWeb sweep (data/fineweb/), and their checkpoints land in outputs/fineweb/
# with '+'-joined loss tags, e.g.
#     best_translator__..__mlp__mse+cosine+info_nce.pt
# so run_ab_sweep.sh's glob picks them up alongside the single-loss checkpoints.
#
# Nothing here re-extracts activations if the caches already exist.
#
# Usage (run from anywhere; the script cd's to the repo root itself):
#   scripts/run_loss_combo_sweep.sh            # extract-if-missing + train+eval all combos
#   scripts/run_loss_combo_sweep.sh extract    # only ensure the shared activations exist
#   scripts/run_loss_combo_sweep.sh train      # only train+eval (assumes activations exist)

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (so .env / load_dotenv finds HF_CACHE_DIR)

PY="conda run -n acteng python"

# HuggingFace Xet streaming backend has been network-flaky here; the classic
# resolver is slower but reliable. Remove this line to re-enable Xet.
export HF_HUB_DISABLE_XET=1

CONFIG_DIR="config/loss_combos"
DATA_DIR="data/fineweb"
ACT_DIR="$DATA_DIR/activations"

# The compound-loss configs are last-token only, so we only need the last-token
# caches. (Add the *_mean caches here if you extend POOLINGS to include "mean".)
REQUIRED_ACTS=(
  "$ACT_DIR/Llama-3.2-1B-Instruct_l8.pt"
  "$ACT_DIR/Llama-3.2-3B-Instruct_l8.pt"
)

# A config used purely to drive extraction if the caches are missing (only its
# token_position matters for extraction — translator/loss is irrelevant).
LAST_CFG="$CONFIG_DIR/mlp_mse+cosine_last.toml"

generate_configs() {
  echo "==> Regenerating compound-loss sweep configs"
  $PY config/generate_loss_combo_configs.py
}

acts_present() {
  for f in "${REQUIRED_ACTS[@]}"; do
    [[ -f "$f" ]] || return 1
  done
  return 0
}

extract() {
  if acts_present; then
    echo "==> Activation caches already present — reusing the FineWeb caches:"
    for f in "${REQUIRED_ACTS[@]}"; do echo "   ok: $f"; done
    return 0
  fi
  echo "==> Some activation caches missing — extracting LAST-token activations"
  echo "    (shared with the FineWeb sweep; run scripts/run_fineweb_sweep.sh extract instead"
  echo "     if you also want the mean-pooled caches.)"
  $PY prepare_activations.py --config "$LAST_CFG"
}

preflight() {
  echo "==> Pre-flight: verifying everything needed for the compound-loss runs"
  local ok=1
  if [[ ! -f "$DATA_DIR/sentences.json" ]]; then
    echo "   MISSING: $DATA_DIR/sentences.json"; ok=0
  fi
  for f in "${REQUIRED_ACTS[@]}"; do
    if [[ ! -f "$f" ]]; then echo "   MISSING: $f"; ok=0; else echo "   ok: $f"; fi
  done
  if [[ $ok -ne 1 ]]; then
    echo "Pre-flight FAILED — run 'scripts/run_loss_combo_sweep.sh extract' first." >&2
    exit 1
  fi
  echo "   Pre-flight OK."
}

train_all() {
  local total=0 done=0
  total=$(ls -1 "$CONFIG_DIR"/*.toml | wc -l)
  echo "==> Training + evaluating $total compound-loss configs"
  for cfg in "$CONFIG_DIR"/*.toml; do
    done=$((done + 1))
    echo
    echo "---- [$done/$total] $cfg ----"
    $PY train.py --config "$cfg"
    $PY evaluate.py --config "$cfg"
  done
  echo
  echo "==> Compound-loss sweep complete. Checkpoints in outputs/fineweb/"
  echo "    (tensorboard --logdir outputs/fineweb/tensorboard)"
  echo
  echo "==> Next: run the A/B steering eval + report over ALL translators"
  echo "    (single-loss + compound-loss checkpoints are picked up by the same glob):"
  echo
  echo "        scripts/run_ab_sweep.sh"
  echo
  echo "    (or 'scripts/run_ab_sweep.sh 10' for a 10-item-per-coefficient smoke test)"
}

MODE="${1:-all}"
case "$MODE" in
  extract)
    generate_configs; extract; preflight ;;
  train)
    generate_configs; preflight; train_all ;;
  all)
    generate_configs; extract; preflight; train_all ;;
  *)
    echo "Unknown mode: $MODE (expected: extract | train | all)" >&2; exit 1 ;;
esac
