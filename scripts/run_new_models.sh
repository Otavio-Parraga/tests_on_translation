#!/usr/bin/env bash
# Sequential, GPU-1-only pipeline. Runs one thing after the other:
#
#   STEP 1  Native-3B baseline A/B eval (the missing control for the layer sweep):
#           evaluate the natively-extracted Llama-3.2-3B CAA SVs on 3B itself, at
#           target layers {8,10,12,14,16,18}, then rebuild the layer-sweep report
#           so "fraction of native steering recovered" can be read off.
#
#   STEP 2  Extract FineWeb activations for Qwen2.5-0.5B-Instruct (24 layers) at
#           the middle-third sweep {8,10,12,14,16}.
#
#   STEP 3  Extract FineWeb activations for gemma-3-1B-it (26 layers) at the
#           middle-third sweep {8,10,12,14,16,18}.
#
# Steps 2-3 reuse data/fineweb/sentences.json (the exact 30k the Llama caches were
# built on) so target activations are row-aligned with the Llama-1B source cache
# and no network is touched. Everything is pinned to GPU 1.
#
# Failure-tolerant: the three steps are independent, so a failure in one is logged
# and the pipeline continues to the next (the user runs this detached and wants as
# much as possible done on return). All steps are individually resumable/idempotent:
# complete activation caches and already-done A/B (scope,behavior) combos are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)

export CUDA_VISIBLE_DEVICES=1          # <- pin the whole pipeline to GPU 1
export TOKENIZERS_PARALLELISM=false
RUN="conda run -n acteng python"

AB_OUT="outputs/ab_eval/layer_sweep"
mkdir -p "$AB_OUT" outputs/new_models

declare -a STATUS

step() {  # $1 = human label; $2... = command
  local label="$1"; shift
  echo
  echo "############################################################"
  echo "# $label"
  echo "#   $(date '+%Y-%m-%d %H:%M:%S')  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> physical GPU 1)"
  echo "############################################################"
  if "$@"; then
    STATUS+=("OK    $label")
  else
    STATUS+=("FAIL  $label (exit $?)")
    echo "!! step failed: $label — continuing to next step"
  fi
}

# --- STEP 1: native-3B baseline A/B eval -------------------------------------
step "STEP 1/3  Native-3B baseline A/B eval" \
  $RUN ab_sweep.py \
    --translators "outputs/layer_sweep/task1_1b_to_3b/*.pt" \
    --with-native --no-source --no-target --skip-translate \
    --out-dir "$AB_OUT" --device cuda:0

# report rebuild is cosmetic — never let it block the extraction steps
step "STEP 1b   Rebuild layer-sweep A/B report (now includes native baseline)" \
  $RUN ab_report.py --results "$AB_OUT/results.jsonl" --out-dir "$AB_OUT"

# --- STEP 2: Qwen activations ------------------------------------------------
step "STEP 2/3  Qwen2.5-0.5B-Instruct activations (layers 8 10 12 14 16)" \
  $RUN extract_activations_sweep.py \
    --config config/new_models/fineweb_qwen0.5b.toml --layers 8 10 12 14 16

# --- STEP 3: Gemma activations -----------------------------------------------
step "STEP 3/3  gemma-3-1B-it activations (layers 8 10 12 14 16 18)" \
  $RUN extract_activations_sweep.py \
    --config config/new_models/fineweb_gemma1b.toml --layers 8 10 12 14 16 18

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Artifacts:"
echo "  Native baseline + report : $AB_OUT/report.html"
echo "  Qwen activations         : data/fineweb/activations/Qwen2.5-0.5B-Instruct_l{8,10,12,14,16}.pt"
echo "  Gemma activations        : data/fineweb/activations/gemma-3-1B-it_l{8,10,12,14,16,18}.pt"
