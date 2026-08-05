#!/usr/bin/env bash
# Native-baseline A/B eval for the new target models (Qwen2.5-0.5B, gemma-3-1B):
# evaluate each model's OWN natively-extracted CAA SV, on that model, at the swept
# target layers {8,10,12,14,16(,18)}. Appends scope="native" rows to the SAME
# results as the translated run so the report can report, per model/layer/behavior,
# the fraction of native steering the translated vector recovers.
#
# Uses the parameterized phase_native (groups translators by their own target model).
# Resumable (JSONL keyed by scope|translator|norm|behavior) and pinned to GPU 1.
set -uo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
RUN="conda run -n acteng python"
OUT="outputs/ab_eval/new_models"

echo "############################################################"
echo "# Native baseline — Qwen + Gemma   $(date '+%Y-%m-%d %H:%M:%S')  (GPU 1)"
echo "############################################################"
$RUN ab_sweep.py \
  --translators "outputs/new_models/best_translator__*procrustes*.pt" \
  --with-native --no-source --no-target --skip-translate \
  --out-dir "$OUT" --device cuda:0
echo "ab_sweep(native) exit=$?"

echo "--- rebuild report (non-fatal) ---"
$RUN ab_report.py --results "$OUT/results.jsonl" --out-dir "$OUT" \
  || echo "!! report build failed (non-fatal); results.jsonl intact"

echo
echo "DONE $(date '+%Y-%m-%d %H:%M:%S')."
echo "  Report : $OUT/report.html"
echo "  Rows   : $(wc -l < "$OUT/results.jsonl" 2>/dev/null) in $OUT/results.jsonl"
