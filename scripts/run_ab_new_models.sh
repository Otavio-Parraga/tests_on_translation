#!/usr/bin/env bash
# A/B steering eval of the cross-family translated vectors on their REAL target
# models (Qwen2.5-0.5B, gemma-3-1B), using the parameterized phase_target.
#
#   Translators : all 22 in outputs/new_models/ (raw + whitened Procrustes),
#                 Llama-3.2-1B l8 -> {Qwen l8..16, Gemma l8..18}.
#   Baseline    : --with-source evaluates the Llama-1B CAA SVs on Llama-1B (source
#                 layer 8) once per behavior — the reference curve fidelity_vs_source
#                 in the report is measured against.
#   Contrast    : raw (__procrustes) vs whitened (__procrustes_whiten) shows whether
#                 the anisotropy-aware map that fixed retrieval also steers behavior.
#
# Resumable (JSONL keyed by scope|translator|norm|behavior) and pinned to GPU 1.
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
RUN="conda run -n acteng python"
OUT="outputs/ab_eval/new_models"
mkdir -p "$OUT"

echo "############################################################"
echo "# A/B eval — cross-family translated vectors   $(date '+%Y-%m-%d %H:%M:%S')"
echo "#   target models loaded per-checkpoint (GPU 1)"
echo "############################################################"
$RUN ab_sweep.py \
  --translators "outputs/new_models/best_translator__*procrustes*.pt" \
  --with-source \
  --out-dir "$OUT" --device cuda:0
SWEEP_RC=$?
echo "ab_sweep exit=$SWEEP_RC"

echo "--- build report (non-fatal) ---"
$RUN ab_report.py --results "$OUT/results.jsonl" --out-dir "$OUT" \
  || echo "!! report build failed (non-fatal); results.jsonl is intact"

echo
echo "DONE $(date '+%Y-%m-%d %H:%M:%S')."
echo "  Report : $OUT/report.html"
echo "  Rows   : $(wc -l < "$OUT/results.jsonl" 2>/dev/null) in $OUT/results.jsonl"
