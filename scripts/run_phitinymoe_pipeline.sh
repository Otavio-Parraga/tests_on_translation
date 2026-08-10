#!/usr/bin/env bash
# Full Llama-1B -> Phi-tiny-MoE-instruct pipeline, run sequentially and detached:
#   STEP 1  Fit raw + whitened Procrustes translators (CPU, fast) -- run_fit_phitinymoe.sh
#   STEP 2  A/B steering eval on the real target model (GPU 1)    -- run_ab_phitinymoe.sh
# FineWeb activation extraction (the long GPU step) already completed separately.
#
# Failure-tolerant: step 2 still runs even if step 1 has a partial failure (any
# checkpoints that did fit are still evaluated); each script is independently
# resumable/idempotent, so re-running this after a kill/crash just picks up
# where it left off.
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)

declare -a STATUS

step() {
  local label="$1"; shift
  echo
  echo "############################################################"
  echo "# $label"
  echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "############################################################"
  if "$@"; then
    STATUS+=("OK    $label")
  else
    STATUS+=("FAIL  $label (exit $?)")
    echo "!! step failed: $label — continuing to next step"
  fi
}

step "STEP 1/2  Fit Procrustes translators (raw + whitened)" scripts/run_fit_phitinymoe.sh
step "STEP 2/2  A/B steering eval on Phi-tiny-MoE-instruct"   scripts/run_ab_phitinymoe.sh

echo
echo "############################################################"
echo "# PIPELINE SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
