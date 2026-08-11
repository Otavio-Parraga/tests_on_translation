#!/usr/bin/env bash
# Sequential, GPU-2-only LLaDA extraction pipeline. Runs one model after the other:
#
#   STEP 1  GSAI-ML/LLaDA-8B-Instruct  (32 blocks) layers {10,12,14,16,18,20,22}
#   STEP 2  inclusionAI/LLaDA2.0-mini  (20 layers) layers {6,8,10,12,14}
#
# SEQUENTIAL IS REQUIRED, not just tidy: the two models are ~16 GB and ~32.5 GB
# resident, so running them concurrently would OOM a single 48 GB card.
#
# Both steps reuse data/fineweb/sentences.json (the exact 30k the Llama caches were
# built on) so target activations stay row-aligned with the Llama-1B source cache
# and no network is touched. Both use token_position = "mean" (bidirectional
# attention leaves no privileged last token), so caches carry a _mean suffix.
#
# This is the MEAN arm. scripts/run_llada_mask.sh is its counterpart, extracting at
# an appended [MASK] position over the same models and layers (caches suffixed
# _mask, so the two arms coexist). Run them one after the other, never at once —
# either model alone already dominates a 48 GB card.
#
# Failure-tolerant: the two steps are independent, so a failure in one is logged
# and the pipeline continues to the next (this is meant to be run detached — you
# want as much done as possible on return). Both steps are individually
# resumable/idempotent: complete layer caches are skipped, interrupted ones resume
# mid-batch, so re-running this script after any failure is safe and cheap.
#
# LAUNCH DETACHED (survives closing the terminal):
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_llada_all.sh > /dev/null 2>&1 &
#
# It self-logs to outputs/llada_extraction_<timestamp>.log regardless of how it is
# launched, so follow progress with:
#   tail -f "$(ls -t outputs/llada_extraction_*.log | head -1)"
#
# Override the GPU with GPU=n, the env with CONDA_ENV_PATH=/path, the log with LOG=path.
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

mkdir -p outputs/new_models

# Always leave a log behind, even when launched detached with output to /dev/null.
LOG="${LOG:-outputs/llada_extraction_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LLaDA extraction pipeline (sequential)  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
preflight || exit 1

declare -a STATUS

step() {  # $1 = human label; $2... = command
  local label="$1"; shift
  echo
  echo "############################################################"
  echo "# $label"
  echo "#   $(date '+%Y-%m-%d %H:%M:%S')  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> physical GPU $GPU)"
  echo "############################################################"
  if "$@"; then
    STATUS+=("OK    $label")
  else
    STATUS+=("FAIL  $label (exit $?)")
    echo "!! step failed: $label — continuing to next step"
  fi
}

# --- STEP 1: LLaDA-8B-Instruct activations ------------------------------------
step "STEP 1/2  LLaDA-8B-Instruct activations (layers 10 12 14 16 18 20 22)" \
  $RUN extract_activations_sweep.py \
    --config config/new_models/fineweb_llada8b.toml --layers 10 12 14 16 18 20 22

# --- STEP 2: LLaDA2.0-mini activations ----------------------------------------
step "STEP 2/2  LLaDA2.0-mini activations (layers 6 8 10 12 14)" \
  $RUN extract_activations_sweep.py \
    --config config/new_models/fineweb_llada2mini.toml --layers 6 8 10 12 14

# --- integrity: extraction exiting 0 is NOT evidence the caches are usable ------
# A corrupt model yields structurally perfect output (right shape, count, dtype),
# so every signal above reports success. This is the step that actually looks at
# the numbers; without it a sweep can and did report OK over caches of NaN.
step "VERIFY  LLaDA-8B-Instruct caches" \
  $RUN scripts/verify_activations.py \
    --config config/new_models/fineweb_llada8b.toml --layers 10 12 14 16 18 20 22
step "VERIFY  LLaDA2.0-mini caches" \
  $RUN scripts/verify_activations.py \
    --config config/new_models/fineweb_llada2mini.toml --layers 6 8 10 12 14

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Artifacts:"
echo "  LLaDA-8B activations   : data/fineweb/activations/LLaDA-8B-Instruct_l{10,12,14,16,18,20,22}_mean.pt"
echo "  LLaDA2.0-mini          : data/fineweb/activations/LLaDA2.0-mini_l{6,8,10,12,14}_mean.pt"
echo "  Log                    : $LOG"
