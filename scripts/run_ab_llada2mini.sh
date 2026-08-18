#!/usr/bin/env bash
# A/B steering sweep for the LLaDA2.0-mini target: translated Llama-3.2-1B l8
# steering vectors injected into inclusionAI/LLaDA2.0-mini across the 20
# Procrustes translators built by scripts/run_fit_llada2mini.sh (5 layers x
# mean/mask x plain/whiten), all 7 MWE behaviors.
#
# READOUT POSITION — same fix as LLaDA-8B (see run_ab_llada.sh): ab_eval scores
# A/B at an appended [MASK] for masked-diffusion models (config.model_type in
# {"llada","llada2_moe"} — see evaluation/ab_eval.py's MASKED_DIFFUSION_MODEL_TYPES),
# auto-detected, no flag needed.
#
# BLOCK-DIFFUSION ATTENTION MASK — LLaDA2.0-mini's forward() rejects a plain
# (B,T) padding mask and requires a pre-built (B,1,T,T) bias. This is handled
# transparently: utils/hf.py's _patch_forward_for_block_attention_mask patches
# the MODEL CLASS's forward() the moment load_model_and_tokenizer sees
# config.model_type == "llada2_moe" (done once, inside ab_sweep.py's normal model
# load — no change needed here or in ab_eval.py). Same for the rotary-table
# repair (_repair_rope_buffers, unconditional on every load) and mask-token
# resolution (resolve_mask_token_id checks the TOKENIZER's '<|mask|>' for this
# model, config.mask_token_id for LLaDA-8B — both covered by one function).
#
# COEFFICIENT GRID — 29 points: the LLaDA-8B study's 17-point grid (dropping
# ab_sweep.py's default +-100/250/500/1000 magnitudes, which only recorded
# collapse there — see run_ab_llada.sh) PLUS a denser probe of the 50-100 range
# (60,70,80,90,100) and +-200, added because LLaDA2.0-mini's own collapse
# threshold relative to |sv| is not yet measured (unlike the 8B target, this
# model has no prior A/B run to read a residual-norm-vs-|sv| ratio off of) — so
# this grid is deliberately wider rather than assuming the 8B's saturation point
# transfers to a different architecture (MoE, 20 layers vs 32, different scale).
# Passed on the CLI, so ab_sweep.py's own default grid is untouched.
#
# CAA ONLY (see run_ab_llada.sh for why methods aren't mixed on one coefficient axis).
#
# COST — see run_ab_llada.sh for the LLaDA-8B baseline (~88ms/forward,
# ~4.1h for 28 translators x 7 behaviors x 17 coefficients x 50 items). This
# sweep has fewer translators (20 vs 28) but a wider grid (29 vs 17
# coefficients) — 20 x 7 x 29 = 4060 (translator,behavior,coeff) x 50-item
# blocks vs the 8B run's 28 x 7 x 17 = 3332. Per-item latency for this MoE
# model is NOT yet measured (only ~1.4B of its ~16B params are active per
# token, which could cut latency, but the vendored/non-optimized modeling code
# and routing overhead could offset that) — budget accordingly and watch the
# first few blocks' timing in the log before assuming a total.
#
# LAUNCH DETACHED (survives closing the terminal):
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_ab_llada2mini.sh > /dev/null 2>&1 &
#   tail -f "$(ls -t outputs/llada2mini_ab_*.log | head -1)"
#
# Then refresh the dashboards:
#   conda run -p $CONDA_ENV_PATH python ab_report.py --results 'outputs/ab_eval/llada2mini/*.jsonl' \
#       --out-dir outputs/ab_eval/llada2mini
#   conda run -p $CONDA_ENV_PATH python ab_pivot_dashboard.py
#
# Override with METHODS="RepE GCAV", GPU=n, COEFFS="...".
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

OUT_DIR="outputs/ab_eval/llada2mini"
TRANSLATORS="outputs/new_models/best_translator__*LLaDA2.0-mini_l*__linear__procrustes*.pt"
METHODS="${METHODS:-CAA}"
COEFFS="${COEFFS:--200 -100 -90 -80 -70 -60 -50 -10 -5 -2 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 2 5 10 50 60 70 80 90 100 200}"

mkdir -p "$OUT_DIR"
LOG="${LOG:-outputs/llada2mini_ab_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LLaDA2.0-mini A/B steering sweep  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "#   methods: $METHODS   results: $OUT_DIR/results.jsonl"
echo "############################################################"
preflight || exit 1

n_ckpt=$(ls -1 $TRANSLATORS 2>/dev/null | wc -l)
if (( n_ckpt == 0 )); then
  echo "!! No LLaDA2.0-mini translators matched: $TRANSLATORS" >&2
  echo "   Run scripts/run_fit_llada2mini.sh first." >&2
  exit 1
fi
echo "Translators: $n_ckpt"
echo "Coefficients ($(echo $COEFFS | wc -w) points): $COEFFS"

declare -a STATUS
for method in $METHODS; do
  echo
  echo "############################################################"
  echo "# METHOD $method   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "############################################################"
  if $RUN ab_sweep.py \
      --translators "$TRANSLATORS" \
      --methods "$method" \
      --coefficients $COEFFS \
      --device cuda:0 \
      --out-dir "$OUT_DIR"; then
    STATUS+=("OK    $method")
  else
    STATUS+=("FAIL  $method (exit $?)")
    echo "!! method $method failed — continuing to the next"
  fi
done

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo "  rows in results.jsonl: $(wc -l < "$OUT_DIR/results.jsonl" 2>/dev/null || echo 0)"
echo
echo "Next — build results.csv and refresh the dashboards:"
echo "  $RUN ab_report.py --results '$OUT_DIR/*.jsonl' --out-dir $OUT_DIR"
echo "  $RUN ab_pivot_dashboard.py"
