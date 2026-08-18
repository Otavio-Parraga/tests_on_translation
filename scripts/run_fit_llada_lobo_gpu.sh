#!/usr/bin/env bash
# GPU-accelerated counterpart to run_fit_llada_lobo.sh, for when the CPU-only
# default is bottlenecked not by the linear algebra itself but by CPU contention
# from OTHER processes on a shared machine (fit_orthogonal_procrustes and
# procrustes_scale are plain tensor ops with no CPU-only calls — see
# fit_procrustes_lobo.py's --device docstring — so pointing them at a free GPU is
# a pure device move, not a different code path or a different result).
#
# WHY A SEPARATE BEHAVIOR LIST, NOT JUST "--device cuda:0 on the same script":
# this is meant to run CONCURRENTLY with an already-in-progress
# run_fit_llada_lobo.sh (the default CPU version), not replace it. Two processes
# fitting the SAME (behavior, layer, arm, variant) at once would race to write the
# SAME checkpoint path. Checkpoints are named by their held-out behavior, never by
# which device fit them, so the only safe split is by BEHAVIOR: this script's
# BEHAVIORS list must be the complement of whatever the concurrent CPU run has
# already reached or is about to reach next — check that run's log before
# launching this, and edit the list below accordingly. There is no automatic
# coordination between the two scripts; that is a deliberate simplicity choice,
# not an oversight, and it means YOU are responsible for keeping the two lists
# disjoint each time you run this alongside a CPU pass.
#
# Everything else mirrors run_fit_llada_lobo.sh exactly (same grid, same
# config_for() mapping, same output directory) — only --device and the behavior
# list differ.
#
# LAUNCH DETACHED:
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_fit_llada_lobo_gpu.sh > /dev/null 2>&1 &
#   tail -f "$(ls -t outputs/llada_lobo_fit_gpu_*.log | head -1)"
#
# Override BEHAVIORS="hallucination myopic-reward", LAYERS, ARMS, VARIANTS, GPU=n.
set -uo pipefail
cd "$(dirname "$0")/.."   # scripts/ lives one level below the repo root
source scripts/llada_env.sh

BEHAVIORS=(${BEHAVIORS:-hallucination myopic-reward refusal survival-instinct sycophancy})
LAYERS=(${LAYERS:-10 12 14 16 18 20 22})
ARMS=(${ARMS:-mean})
VARIANTS=(${VARIANTS:-plain whiten})
OUT="outputs/new_models"

LOG="${LOG:-outputs/llada_lobo_fit_gpu_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LOBO Procrustes fits (GPU)  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> physical GPU $GPU)"
echo "#   behaviors=${BEHAVIORS[*]}"
echo "#   layers=${LAYERS[*]}  arms=${ARMS[*]}  variants=${VARIANTS[*]}"
echo "#   total fits: $((${#BEHAVIORS[@]} * ${#LAYERS[@]} * ${#ARMS[@]} * ${#VARIANTS[@]}))"
echo "############################################################"
preflight || exit 1

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
            --exclude-behavior "$excl" --device cuda:0 $flag; then
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
echo "Checkpoints for this run's behaviors:"
for b in "${BEHAVIORS[@]}"; do
  ls -1 "$OUT"/best_translator__*_lobo-"$b".pt 2>/dev/null | sed 's#.*/##'
done
