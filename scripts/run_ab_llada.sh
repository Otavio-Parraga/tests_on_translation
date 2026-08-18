#!/usr/bin/env bash
# A/B steering sweep for the DIFFUSION target: translated Llama-3.2-1B l8 steering
# vectors injected into GSAI-ML/LLaDA-8B-Instruct across the 28 Procrustes
# translators built by scripts/run_fit_llada.sh (7 layers x mean/mask x
# plain/whiten), all 7 MWE behaviors.
#
# READOUT POSITION — the reason this sweep is only now meaningful. acttrans's
# ab_eval scores A/B at an appended [MASK] for masked-diffusion models instead of
# at the final position (auto-detected from config.model_type; see the module
# docstring in evaluation/ab_eval.py). Measured on LLaDA-8B over the same items,
# mean P(A)+P(B) at the readout position:
#     final position (pre-fix) : 0.0004     <- metric has no mass to move
#     appended [MASK]          : 0.998
# Before that fix this sweep returned avg_p_match ~2e-4, flat across every
# coefficient — indistinguishable from a translator that transports nothing.
#
# COEFFICIENT GRID — 17 points, ab_sweep.py's 25-point default minus the four
# largest magnitudes (+-100, 250, 500, 1000). Those only ever record collapse
# here: the translated vectors carry |sv| = 8.8-12.5 into a residual stream whose
# mean ||h|| is 104, so even +-50 is already ~4x the residual norm and +-10 is
# ~0.85x. Keeping +-50 preserves a visible saturation shoulder while dropping the
# four magnitudes that cost 32% of the runtime to show nothing. Passed on the CLI,
# so ab_sweep.py's default grid is untouched for other studies.
#
# CAA ONLY. RepE/GCAV vectors are present under
# ../activation_engineering/steering_vectors/meta-llama_Llama-3.2-1B-Instruct/ and
# this script takes METHODS="RepE GCAV" to add them later into the SAME resumable
# results file (the resume key is method-aware). Coefficients are NOT comparable
# across methods — CAA carries its own magnitude, RepE/GCAV are unit-norm — which
# is why ab_pivot_dashboard.py is single-method and cross-method reading happens in
# method_report.py on dose = coefficient * sv_norm.
#
# COST, measured on this machine: ~88 ms per LLaDA-8B forward pass (8B params,
# bidirectional, no KV cache) => ~75 s per (translator, behavior) block at 17
# coefficients x 50 items. 28 x 7 = 196 blocks ~= 4.1 h, plus a few minutes for
# the Llama-1B source baseline. Fully resumable: re-running skips
# (method, scope, translator, norm, behavior) combos already in the JSONL, so an
# interrupted run costs only the block it died in.
#
# No --with-native: that arm needs natively-extracted vectors for LLaDA-8B itself,
# and the steering_vectors/ tree covers the five causal models only. The source
# baseline (--with-source, on by default) still gives every curve its reference.
#
# LAUNCH DETACHED (survives closing the terminal):
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_ab_llada.sh > /dev/null 2>&1 &
#   tail -f "$(ls -t outputs/llada_ab_*.log | head -1)"
#
# Then refresh the dashboards:
#   conda run -p $CONDA_ENV_PATH python ab_report.py --results 'outputs/ab_eval/llada/*.jsonl' \
#       --out-dir outputs/ab_eval/llada
#   conda run -p $CONDA_ENV_PATH python ab_pivot_dashboard.py
#
# Override with METHODS="RepE GCAV", GPU=n, COEFFS="...".
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

OUT_DIR="outputs/ab_eval/llada"
TRANSLATORS="outputs/new_models/best_translator__*LLaDA-8B-Instruct_l*__linear__procrustes*.pt"
METHODS="${METHODS:-CAA}"
COEFFS="${COEFFS:--50 -10 -5 -2 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 2 5 10 50}"

mkdir -p "$OUT_DIR"
LOG="${LOG:-outputs/llada_ab_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LLaDA-8B A/B steering sweep  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "#   methods: $METHODS   results: $OUT_DIR/results.jsonl"
echo "############################################################"
preflight || exit 1

n_ckpt=$(ls -1 $TRANSLATORS 2>/dev/null | wc -l)
if (( n_ckpt == 0 )); then
  echo "!! No LLaDA translators matched: $TRANSLATORS" >&2
  echo "   Run scripts/run_fit_llada.sh first." >&2
  exit 1
fi
echo "Translators: $n_ckpt"
echo "Coefficients: $COEFFS"

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
