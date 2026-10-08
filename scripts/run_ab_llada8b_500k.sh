#!/usr/bin/env bash
# A/B steering sweep for the 500k-scale Procrustes translators: translated
# Llama-3.2-1B l8 steering vectors injected into GSAI-ML/LLaDA-8B-Instruct,
# across the 14 translators fit by scripts/run_procrustes_500k.sh's LLaDA-8B arm
# (7 layers x plain/whiten), all 7 MWE behaviors.
#
# Mirror of scripts/run_ab_llada.sh (the 30k-scale version) with ONLY the
# translator glob / output dir swapped to the 500k checkpoints:
#   30k  outputs/new_models/best_translator__*LLaDA-8B-Instruct_l*__linear__procrustes*.pt
#   500k outputs/new_models_500k/best_translator__*LLaDA-8B-Instruct_l*__linear__procrustes*.pt
#
# LLaDA2.0-mini is DELIBERATELY NOT covered here — its 500k Procrustes fits came
# back weak (cosine ~0.14-0.19 across every layer, same order as the 30k fit),
# so an A/B sweep on it would burn GPU time transporting a vector through a map
# that likely isn't capturing real structure. Revisit after diagnose_alignment.py.
#
# Does NOT source scripts/llada_env.sh's preflight() — that function hard-checks
# data/fineweb/sentences.json for exactly 30000 rows, a path that no longer
# exists (the 30k corpus now lives at data/fineweb_30k/). A/B eval never reads
# the FineWeb sentence set at all (it uses MWE behavior prompts + the CAA
# steering vectors under ../activation_engineering/steering_vectors/), so that
# check was never actually relevant here — this script does its own minimal
# preflight instead.
#
# COEFFICIENT GRID — reused verbatim from run_ab_llada.sh: same model
# (LLaDA-8B-Instruct), same source steering vector (Llama-3.2-1B CAA, layer 8),
# so the residual-norm-vs-|sv| saturation point measured there should transfer;
# revisit if this run's low/high coefficients show unexpected behavior.
#
# CAA ONLY, GPU pinned to whatever $GPU is set to (default 5 here, since GPU 2
# is shared with other users' jobs on this machine — override with GPU=n).
#
# LAUNCH DETACHED (survives closing the terminal):
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   GPU=5 setsid nohup ./scripts/run_ab_llada8b_500k.sh > /dev/null 2>&1 &
#   tail -f "$(ls -t outputs/llada8b_500k_ab_*.log | head -1)"
#
# Then refresh the dashboards:
#   conda run -p $CONDA_ENV_PATH python ab_report.py --results 'outputs/ab_eval/llada_500k/*.jsonl' \
#       --out-dir outputs/ab_eval/llada_500k
#   conda run -p $CONDA_ENV_PATH python ab_pivot_dashboard.py
#
# Override with METHODS="RepE GCAV", GPU=n, COEFFS="...".
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/C-SSD/parraga/conda_envs/act_eng}"
GPU="${GPU:-5}"
export CUDA_VISIBLE_DEVICES="$GPU"
export TOKENIZERS_PARALLELISM=false
export HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/E-SSD/model_cache/hf}"
RUN="conda run -p $CONDA_ENV_PATH --no-capture-output python"

OUT_DIR="outputs/ab_eval/llada_500k"
TRANSLATORS="outputs/new_models_500k/best_translator__*LLaDA-8B-Instruct_l*__linear__procrustes*.pt"
METHODS="${METHODS:-CAA}"
COEFFS="${COEFFS:--50 -10 -5 -2 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 2 5 10 50}"

mkdir -p "$OUT_DIR"
LOG="${LOG:-outputs/llada8b_500k_ab_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LLaDA-8B-Instruct 500k A/B steering sweep  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "#   methods: $METHODS   results: $OUT_DIR/results.jsonl"
echo "############################################################"

# --- minimal preflight (see header for why llada_env.sh's is not reused) -----
rc=0
if [[ ! -x "$CONDA_ENV_PATH/bin/python" ]]; then
  echo "!! conda env python not found: $CONDA_ENV_PATH/bin/python" >&2; rc=1
fi
if [[ ! -d "$HF_CACHE_DIR" ]]; then
  echo "!! HF_CACHE_DIR does not exist: $HF_CACHE_DIR" >&2; rc=1
fi
n_ckpt=$(ls -1 $TRANSLATORS 2>/dev/null | wc -l)
if (( n_ckpt == 0 )); then
  echo "!! No LLaDA-8B 500k translators matched: $TRANSLATORS" >&2
  echo "   Run the 500k Procrustes fits first." >&2
  rc=1
fi
if (( rc != 0 )); then
  echo "preflight FAILED — nothing was run." >&2
  exit 1
fi
echo "preflight OK  |  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (-> physical GPU $GPU)"
echo "              |  HF_CACHE_DIR=$HF_CACHE_DIR"
echo "              |  Translators: $n_ckpt"
echo "              |  Coefficients: $COEFFS"

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
