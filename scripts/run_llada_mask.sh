#!/usr/bin/env bash
# Sequential, GPU-2-only LLaDA extraction at the appended [MASK] position — the
# mask-readout arm of the diffusion-target study. Exact counterpart of
# run_llada_all.sh (which runs the mean arm); same models, same layers, same
# batch sizes, only token_position differs:
#
#   STEP 1  GSAI-ML/LLaDA-8B-Instruct  (32 blocks) layers {10,12,14,16,18,20,22}
#   STEP 2  inclusionAI/LLaDA2.0-mini  (20 layers) layers {6,8,10,12,14}
#
# WHY a second arm: mean-pooling describes text that already exists, but a steering
# vector is injected mid-denoising, where the sequence is mostly [MASK]. The mask
# position is the only slot these models were trained to fill, and it is where the
# A/B eval reads its logits — so a vector fit here is fit in the space it is applied
# in. Running both arms over identical layers makes "where to read" a controlled
# comparison rather than a guess.
#
# Caches land with a _mask suffix (POOLING_SUFFIXES in utils/paths.py), so this arm
# never collides with the _mean caches and BOTH can be kept. That suffix is
# load-bearing: extraction skips any cache whose n_sentences already matches, so a
# shared slug would make this run silently reuse the mean activations.
#
# SEQUENTIAL IS REQUIRED, not just tidy: the two models are ~16 GB and ~32.5 GB
# resident, so running them concurrently would OOM a single 48 GB card. For the same
# reason, do NOT run this at the same time as run_llada_all.sh.
#
# Both steps reuse data/fineweb/sentences.json (the exact 30k the Llama caches were
# built on) so target activations stay row-aligned with the Llama-1B source cache and
# no network is touched. The source side stays token_position = -1 (Llama is causal
# and has no mask token), so the existing Llama-3.2-1B-Instruct_l8.pt cache is reused
# and this arm costs target extraction only.
#
# Failure-tolerant: the two steps are independent, so a failure in one is logged and
# the pipeline continues. Both are individually resumable/idempotent — complete layer
# caches are skipped, interrupted ones resume mid-batch — so re-running after any
# failure is safe and cheap.
#
# LAUNCH DETACHED (survives closing the terminal):
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_llada_mask.sh > /dev/null 2>&1 &
#
# It self-logs to outputs/llada_mask_extraction_<timestamp>.log regardless of how it
# is launched, so follow progress with:
#   tail -f "$(ls -t outputs/llada_mask_extraction_*.log | head -1)"
#
# Override the GPU with GPU=n, the env with CONDA_ENV_PATH=/path, the log with LOG=path.
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

mkdir -p outputs/new_models

# Always leave a log behind, even when launched detached with output to /dev/null.
LOG="${LOG:-outputs/llada_mask_extraction_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LLaDA extraction pipeline — MASK position (sequential)  ->  $LOG"
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

# --- STEP 1: LLaDA-8B-Instruct, mask position ---------------------------------
step "STEP 1/2  LLaDA-8B-Instruct mask-position activations (layers 10 12 14 16 18 20 22)" \
  $RUN extract_activations_sweep.py \
    --config config/new_models/fineweb_llada8b_mask.toml --layers 10 12 14 16 18 20 22

# --- STEP 2: LLaDA2.0-mini, mask position -------------------------------------
step "STEP 2/2  LLaDA2.0-mini mask-position activations (layers 6 8 10 12 14)" \
  $RUN extract_activations_sweep.py \
    --config config/new_models/fineweb_llada2mini_mask.toml --layers 6 8 10 12 14

# --- integrity: see the equivalent block in run_llada_all.sh --------------------
step "VERIFY  LLaDA-8B-Instruct mask caches" \
  $RUN scripts/verify_activations.py \
    --config config/new_models/fineweb_llada8b_mask.toml --layers 10 12 14 16 18 20 22
step "VERIFY  LLaDA2.0-mini mask caches" \
  $RUN scripts/verify_activations.py \
    --config config/new_models/fineweb_llada2mini_mask.toml --layers 6 8 10 12 14

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Artifacts:"
echo "  LLaDA-8B activations   : data/fineweb/activations/LLaDA-8B-Instruct_l{10,12,14,16,18,20,22}_mask.pt"
echo "  LLaDA2.0-mini          : data/fineweb/activations/LLaDA2.0-mini_l{6,8,10,12,14}_mask.pt"
echo "  Log                    : $LOG"
