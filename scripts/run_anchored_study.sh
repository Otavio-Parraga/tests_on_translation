#!/usr/bin/env bash
#
# The anchored-translator study, end to end.
#
# Question: closed-form orthogonal Procrustes is the best translator this repo has
# measured, and every gradient-trained translator loses to it. If a trained
# translator STARTS from the Procrustes solution instead of from noise
# (translator.anchor = "procrustes", gate initialized to 0 so step 0 IS the floor),
# does it beat that floor?
#
# Three stages, in order:
#
#   A  fit the closed-form floor at EVERY cached target layer of all four pairs
#      (no training, seconds each), score them against native steering vectors,
#      and read off each pair's best target layer. Cheap enough to be exhaustive,
#      which is what keeps stage B small.
#   B  train, ONLY at each pair's best layer, two mlp translators on
#      cosine+info_nce: one anchored, one from scratch (its control). Llama also
#      gets l8, which separates a layer effect from an anchor effect.
#   C  score every stage-B checkpoint plus the stage-A floors into one table.
#
# Stage B's layers come from stage A and are baked into TRAIN_LAYERS in
# config/generate_anchored_configs.py — re-run stage A first if the caches change,
# then update that table and regenerate the configs.
#
# Single-GPU, strictly sequential: this is one 11 GB GTX 1080 Ti. Set GPU= to pick
# the device (never launch onto a GPU another job is already using).
#
# Usage:
#   scripts/run_anchored_study.sh            # all three stages
#   scripts/run_anchored_study.sh B C        # only the stages named
#   GPU=1 scripts/run_anchored_study.sh

set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)

GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
RUN="conda run -n acteng python"

OUT="outputs/anchored"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

STAGES="${*:-A B C}"
has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

# Cached target layers per pair — must match PAIRS in the config generator.
PAIR_LAYERS=(
  "llama3b:8 10 12 14 16 18"
  "qwen0.5b:8 10 12 14 16"
  "gemma1b:8 10 12 14 16 18"
  "phitinymoe:10 12 14 16 18 20 22"
)

# Stage-B runs — must match TRAIN_LAYERS in the config generator.
TRAINED=(
  mlp_llama3b_l12_anchored   mlp_llama3b_l12_scratch
  mlp_llama3b_l8_anchored    mlp_llama3b_l8_scratch
  mlp_qwen0.5b_l14_anchored  mlp_qwen0.5b_l14_scratch
  mlp_gemma1b_l12_anchored   mlp_gemma1b_l12_scratch
  mlp_phitinymoe_l16_anchored mlp_phitinymoe_l16_scratch
)

echo "############ anchored-translator study — GPU $GPU — stages: $STAGES"

if has_stage A; then
  echo "==== Stage A: closed-form Procrustes at every cached layer  $(date '+%H:%M:%S')"
  : > "$LOGS/stageA_fit.log"
  for spec in "${PAIR_LAYERS[@]}"; do
    tag="${spec%%:*}"; layers="${spec#*:}"
    for k in $layers; do
      echo "######## $tag -> l$k" | tee -a "$LOGS/stageA_fit.log"
      $RUN fit_procrustes.py --config "config/anchored/procrustes_$tag.toml" \
        --source-layer 8 --target-layer "$k" >> "$LOGS/stageA_fit.log" 2>&1
      echo "  exit=$?"
    done
  done
  $RUN compare_translated_and_original.py geometric \
    --translators "$OUT/best_translator__*linear__procrustes.pt" \
    --device cuda:0 --out "$OUT/cmp_stageA" --no-layer-sweep \
    > "$LOGS/stageA_cmp.log" 2>&1
  echo "  stage A scored -> $OUT/cmp_stageA/geometric/per_translator.csv"
fi

if has_stage B; then
  echo "==== Stage B: 10 training runs at the stage-A best layers  $(date '+%H:%M:%S')"
  for cfg in "${TRAINED[@]}"; do
    echo "######## $cfg  $(date '+%H:%M:%S')"
    s=$(date +%s)
    $RUN train.py --config "config/anchored/$cfg.toml" \
      > "$LOGS/train_${cfg#mlp_}.log" 2>&1
    echo "  exit=$? elapsed=$(( $(date +%s) - s ))s"
  done
fi

if has_stage C; then
  echo "==== Stage C: score everything + summary table  $(date '+%H:%M:%S')"
  $RUN compare_translated_and_original.py geometric \
    --translators "$OUT/best_translator__*.pt" \
    --device cuda:0 --out "$OUT/cmp" --no-layer-sweep \
    > "$LOGS/stageC_cmp.log" 2>&1
  echo "  exit=$?"
  $RUN anchored_summary.py --cmp "$OUT/cmp" \
    --translators "$OUT/best_translator__*.pt" --out "$OUT/summary.csv" \
    --device cuda:0 2>&1 | tee "$LOGS/summary.log"
fi

echo "DONE $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Summary : $OUT/summary.csv"
