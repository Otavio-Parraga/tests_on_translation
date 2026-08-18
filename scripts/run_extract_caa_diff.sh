#!/usr/bin/env bash
# Extract behavior-contrastive (positive - negative) activation DIFFERENCES for
# the leave-one-behavior-out translator study — build_caa_diff_manifest.py's
# manifest, run through both the SOURCE model (Llama-3.2-1B-Instruct, layer 8,
# token_position -2, the exact recipe the real CAA sv.pt files were built with —
# see extract_caa_diff_source.py's docstring) and the TARGET model
# (GSAI-ML/LLaDA-8B-Instruct, layers 10..22, mean AND mask arms in one load — see
# extract_caa_diff_target.py's docstring).
#
# SEQUENTIAL, GPU-2-ONLY, LIKE EVERY OTHER LLaDA LAUNCHER HERE. Source extraction
# is small and fast (a ~1B model); target extraction loads the ~16 GB LLaDA-8B.
# Never run this alongside another LLaDA process on the same GPU.
#
# COST (estimate, not yet measured on real hardware): source side is ~9.8k tiny
# forward passes on a 1B model, a few minutes. Target side is ~4.9k items x 2
# forward passes (one plain, one mask-appended), each hooking all 7 layers at once
# so BOTH arms come from those same two forwards — NOT 14 separate passes (see the
# script's docstring for why). At the ~50-90ms/forward ballpark measured for
# single-sequence LLaDA-8B calls elsewhere in this project, that is roughly
# 4.9k x 2 x 0.07s =~ 700s =~ 12-20 minutes. Both scripts checkpoint every 200
# items and resume automatically, so an interruption costs only the chunk it died
# in, not the whole pass.
#
# LAUNCH DETACHED (survives closing the terminal):
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_extract_caa_diff.sh > /dev/null 2>&1 &
#   tail -f "$(ls -t outputs/caa_diff_extraction_*.log | head -1)"
#
# Then fit the LOBO translators (scripts/run_fit_llada_lobo.sh — CPU only, no
# GPU dependency, safe to run any time after this completes) and evaluate
# (scripts/run_ab_llada_lobo.sh — GPU, LLaDA-8B again).
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

MANIFEST="data/caa_diff/pairs_manifest.json"
LOG="${LOG:-outputs/caa_diff_extraction_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# CAA behavior-diff extraction (source + target)  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
preflight || exit 1

if [[ ! -f "$MANIFEST" ]]; then
  echo "Building manifest (CPU-only, no model load)..."
  $RUN build_caa_diff_manifest.py || exit 1
else
  echo "Manifest already present: $MANIFEST"
fi

declare -a STATUS

echo
echo "=================================================================="
echo "  SOURCE  Llama-3.2-1B-Instruct l8 (token -2)   $(date '+%H:%M:%S')"
echo "=================================================================="
if $RUN extract_caa_diff_source.py --device cuda; then
  STATUS+=("OK    source")
else
  STATUS+=("FAIL  source (exit $?)")
  echo "!! source extraction failed — continuing to target anyway"
fi

echo
echo "=================================================================="
echo "  TARGET  LLaDA-8B-Instruct l{10,12,14,16,18,20,22} mean+mask   $(date '+%H:%M:%S')"
echo "=================================================================="
if $RUN extract_caa_diff_target.py --layers 10 12 14 16 18 20 22 --device cuda; then
  STATUS+=("OK    target")
else
  STATUS+=("FAIL  target (exit $?)")
  echo "!! target extraction failed"
fi

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Artifacts in data/caa_diff/activations/:"
ls -1 data/caa_diff/activations/*_diff.pt 2>/dev/null | sed 's#.*/##'
echo
echo "Next: ./scripts/run_fit_llada_lobo.sh"
