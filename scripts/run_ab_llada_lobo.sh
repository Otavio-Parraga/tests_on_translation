#!/usr/bin/env bash
# A/B steering eval for the leave-one-behavior-out translators
# (scripts/run_fit_llada_lobo.sh's checkpoints): for each behavior, evaluate ONLY
# its own excluded-fold translator, on ONLY that behavior — the genuinely
# held-out test, as opposed to the main diffusion sweep's translators, which were
# fit on generic FineWeb sentences and then evaluated on all 7 behaviors freely
# (nothing behavior-specific to hold out there).
#
# ONE ab_sweep.py INVOCATION PER EXCLUDED BEHAVIOR, not one big invocation across
# all of them. ab_sweep.py's --translators glob and --behaviors list are
# independent axes — passing all 7 LOBO checkpoints alongside all 7 behaviors in
# one call would ALSO evaluate every mismatched pairing (e.g. the
# excludes-refusal translator on sycophancy), which is harmless but not what a
# leave-one-out design asks for and would waste most of the run on pairings this
# study isn't about. Looping per-behavior keeps every row in the output file the
# behavior's own held-out fold, and only that one.
#
# Writes to outputs/ab_eval/llada_lobo/ — a SEPARATE directory from
# outputs/ab_eval/llada/ (the main diffusion sweep), so this never overwrites or
# comingles with those results.
#
# CAA ONLY, matching the main study's scoping choice and the same trimmed 17-point
# coefficient grid (see scripts/run_ab_llada.sh for the collapse-region rationale).
# Cost: 98 LOBO checkpoints (default ARMS=mean grid) x 1 behavior each = 98 blocks
# total (not 98 x 7 — each translator is evaluated on exactly one behavior), same
# per-block cost as the main sweep (~28-150s depending on load), so budget
# roughly 45 min - 4 h depending on measured throughput. Widening
# scripts/run_fit_llada_lobo.sh's grid to ARMS="mean mask" doubles this.
#
# LAUNCH DETACHED:
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_ab_llada_lobo.sh > /dev/null 2>&1 &
#   tail -f "$(ls -t outputs/llada_ab_lobo_*.log | head -1)"
#
# Then build the report (no pivot-dashboard registration yet — this is a
# diagnostic study, not a dimension of the main pivot's DATASETS registry):
#   conda run -p $CONDA_ENV_PATH python ab_report.py \
#       --results 'outputs/ab_eval/llada_lobo/*.jsonl' --out-dir outputs/ab_eval/llada_lobo
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

BEHAVIORS=(coordinate-other-ais corrigible-neutral-HHH hallucination myopic-reward refusal survival-instinct sycophancy)
OUT_DIR="outputs/ab_eval/llada_lobo"
COEFFS="${COEFFS:--50 -10 -5 -2 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 2 5 10 50}"

mkdir -p "$OUT_DIR"
LOG="${LOG:-outputs/llada_ab_lobo_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LLaDA-8B LOBO A/B eval  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
preflight || exit 1

n_ckpt=$(ls -1 outputs/new_models/best_translator__*_lobo-*.pt 2>/dev/null | wc -l)
if (( n_ckpt == 0 )); then
  echo "!! No LOBO translators found in outputs/new_models/." >&2
  echo "   Run scripts/run_fit_llada_lobo.sh first." >&2
  exit 1
fi
echo "LOBO checkpoints available: $n_ckpt"
echo "Coefficients: $COEFFS"

declare -a STATUS
for behavior in "${BEHAVIORS[@]}"; do
  glob="outputs/new_models/best_translator__*LLaDA-8B-Instruct_l*__linear__procrustes*_lobo-${behavior}.pt"
  n=$(ls -1 $glob 2>/dev/null | wc -l)
  echo
  echo "############################################################"
  echo "# BEHAVIOR $behavior   ($n translators)   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "############################################################"
  if (( n == 0 )); then
    echo "  skip: no translators matching $glob"
    STATUS+=("SKIP  $behavior (no checkpoints)")
    continue
  fi
  if $RUN ab_sweep.py \
      --translators "$glob" \
      --methods CAA \
      --behaviors "$behavior" \
      --coefficients $COEFFS \
      --device cuda:0 \
      --out-dir "$OUT_DIR"; then
    STATUS+=("OK    $behavior")
  else
    STATUS+=("FAIL  $behavior (exit $?)")
    echo "!! behavior $behavior failed — continuing to the next"
  fi
done

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo "  rows in results.jsonl: $(wc -l < "$OUT_DIR/results.jsonl" 2>/dev/null || echo 0)"
echo
echo "Next — build the report:"
echo "  $RUN ab_report.py --results '$OUT_DIR/*.jsonl' --out-dir $OUT_DIR"
