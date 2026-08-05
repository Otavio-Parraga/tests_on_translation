#!/usr/bin/env bash
# A/B steering eval of the Llama-1B -> Phi-tiny-MoE-instruct translated vectors
# on the REAL target model, mirroring run_ab_new_models.sh (Qwen/Gemma) but
# scoped to just the Phi-tiny-MoE checkpoints so this doesn't re-touch the
# already-evaluated Qwen/Gemma combos (harmless either way since ab_sweep.py
# is resumable, but scoping keeps this run fast and focused).
#
#   Translators : all Phi-tiny-MoE checkpoints in outputs/new_models/ (raw +
#                 whitened Procrustes), Llama-3.2-1B l8 -> Phi l{10..22}.
#   Baseline    : --with-source evaluates the Llama-1B CAA SVs on Llama-1B
#                 (source layer 8) once per behavior — the reference curve
#                 fidelity_vs_source in the report is measured against.
#   No native baseline yet: no native CAA SV has been extracted for
#   Phi-tiny-MoE itself (would need a run in the sibling activation_engineering
#   repo first) — that's a separate follow-up, not blocking this eval.
#
# Resumable (JSONL keyed by scope|translator|norm|behavior) and pinned to GPU 1.
set -uo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
RUN="conda run -n acteng python"
OUT="outputs/ab_eval/new_models"
mkdir -p "$OUT"

echo "############################################################"
echo "# A/B eval — Llama-1B -> Phi-tiny-MoE-instruct   $(date '+%Y-%m-%d %H:%M:%S')"
echo "#   target model loaded once (GPU 1)"
echo "############################################################"
$RUN ab_sweep.py \
  --translators "outputs/new_models/best_translator__*Phi-tiny-MoE-instruct*procrustes*.pt" \
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
