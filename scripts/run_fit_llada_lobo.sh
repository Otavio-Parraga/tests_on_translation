#!/usr/bin/env bash
# Leave-one-behavior-out Procrustes fits on the CAA behavior-diff activations
# (scripts/run_extract_caa_diff.sh's output): for every behavior, fit on the
# other 6 and hold that one out entirely, across every target layer and both
# Procrustes variants.
#
# CPU-ONLY, NO GPU DEPENDENCY. fit_procrustes_lobo.py's retrieval check defaults
# to --device cpu (these matrices are at most ~1000 held-out rows — tiny), so the
# fits themselves never touch the GPU and are safe to run regardless of what is on
# GPU 2. That said, this script still requires the diff-cache FILES that
# scripts/run_extract_caa_diff.sh writes, so in practice it must run AFTER that
# extraction has completed, not merely "not touch the GPU while it runs."
#
# GRID, and why it's narrower than the main diffusion study's 7 layers x 2 arms x
# 2 variants = 28 configs per behavior. Defaulting ARMS to "mean" only (not
# "mean mask") because the main study already found mask-position translators
# underperform mean-pooled ones for STEERING specifically (3/49 and 1/49 monotone
# cells vs 13/49 and 6/49 for mean, plain and whitened respectively) even though
# mask activations are the better-conditioned representation by every geometric
# measure. Re-running the full 28-config grid here would cost 4x more for the arm
# this project's own prior result says is the weaker bet. Override with
# ARMS="mean mask" to widen it back to parity with the main study.
#
# 7 excluded-behaviors x 7 layers x 1 arm(default) x 2 variants = 98 fits. Each
# fit is closed-form (CPU SVD) on <=4911 rows total — smaller and faster than the
# FineWeb Procrustes fits (30k rows), which took ~15s each; expect this to be
# faster per-fit, so budget ~20-25 min for the default grid, ~40-50 min for
# ARMS="mean mask".
#
# LAUNCH DETACHED:
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_fit_llada_lobo.sh > /dev/null 2>&1 &
#   tail -f "$(ls -t outputs/llada_lobo_fit_*.log | head -1)"
#
# Override CONDA_ENV_PATH, ARMS, VARIANTS="plain whiten", LAYERS="10 12 ...".
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/C-SSD/parraga/conda_envs/act_eng}"
RUN="conda run -p $CONDA_ENV_PATH --no-capture-output python"
BEHAVIORS=(coordinate-other-ais corrigible-neutral-HHH hallucination myopic-reward refusal survival-instinct sycophancy)
LAYERS=(${LAYERS:-10 12 14 16 18 20 22})
ARMS=(${ARMS:-mean})
VARIANTS=(${VARIANTS:-plain whiten})
OUT="outputs/new_models"

LOG="${LOG:-outputs/llada_lobo_fit_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LOBO Procrustes fits  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "#   layers=${LAYERS[*]}  arms=${ARMS[*]}  variants=${VARIANTS[*]}"
echo "#   total fits: $((7 * ${#LAYERS[@]} * ${#ARMS[@]} * ${#VARIANTS[@]}))"
echo "############################################################"

if [[ ! -f data/caa_diff/pairs_manifest.json ]]; then
  echo "!! data/caa_diff/pairs_manifest.json missing." >&2
  echo "   Run scripts/run_extract_caa_diff.sh first." >&2
  exit 1
fi
missing=0
for arm in "${ARMS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    f="data/caa_diff/activations/LLaDA-8B-Instruct_l${layer}_${arm}_diff.pt"
    [[ -f "$f" ]] || { echo "!! missing: $f" >&2; missing=1; }
  done
done
[[ -f data/caa_diff/activations/Llama-3.2-1B-Instruct_l8_diff.pt ]] || { echo "!! missing source diff cache" >&2; missing=1; }
if (( missing )); then
  echo "!! One or more diff caches are missing — run scripts/run_extract_caa_diff.sh first." >&2
  exit 1
fi
mkdir -p "$OUT"

declare -a STATUS
config_for() {  # $1=arm -> matching model/layer metadata config
  [[ "$1" == "mask" ]] && echo "config/new_models/fineweb_llada8b_mask.toml" || echo "config/new_models/fineweb_llada8b.toml"
}

n=0
for excl in "${BEHAVIORS[@]}"; do
  for layer in "${LAYERS[@]}"; do
    for arm in "${ARMS[@]}"; do
      for variant in "${VARIANTS[@]}"; do
        n=$((n+1))
        flag=""; [[ "$variant" == "whiten" ]] && flag="--whiten"
        echo
        echo "[$n]  excl=${excl}  L${layer} ${arm} ${variant}   $(date '+%H:%M:%S')"
        if $RUN fit_procrustes_lobo.py \
            --config "$(config_for "$arm")" \
            --target-layer "$layer" --arm "$arm" \
            --exclude-behavior "$excl" $flag; then
          STATUS+=("OK    $excl L${layer} ${arm} ${variant}")
        else
          STATUS+=("FAIL  $excl L${layer} ${arm} ${variant} (exit $?)")
          echo "  !! fit failed — continuing"
        fi
      done
    done
  done
done

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
n_fail=$(printf '%s\n' "${STATUS[@]}" | grep -c '^FAIL' || true)
echo "  ${#STATUS[@]} fits, ${n_fail} failed"
printf '%s\n' "${STATUS[@]}" | grep '^FAIL' || true
echo
echo "Checkpoints ($(ls -1 "$OUT"/best_translator__*_lobo-*.pt 2>/dev/null | wc -l)):"
ls -1 "$OUT"/best_translator__*_lobo-*.pt 2>/dev/null | sed 's#.*/##' | head -20
echo
echo "Next: ./scripts/run_ab_llada_lobo.sh"
