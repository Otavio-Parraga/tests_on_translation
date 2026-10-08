#!/usr/bin/env bash
# Extract FineWeb activations (mean pooling, target models only) for BOTH LLaDA
# models at the 500k-sentence scale, mirroring run_llada_all.sh's structure but
# against the *_500k.toml configs / data/fineweb_500k data_dir instead of the
# existing 30k ones:
#
#   STEP 1  GSAI-ML/LLaDA-8B-Instruct  (32 blocks) layers {10,12,14,16,18,20,22}
#   STEP 2  inclusionAI/LLaDA2.0-mini  (20 layers) layers {6,8,10,12,14}
#
# NOTHING EXISTING IS TOUCHED: this reads config/new_models/fineweb_llada8b_500k.toml
# and fineweb_llada2mini_500k.toml (new files, data_dir = data/fineweb_500k), never
# the original fineweb_llada8b.toml/fineweb_llada2mini.toml (data_dir =
# data/fineweb_30k). data/fineweb_500k/sentences.json already exists (500000
# sentences, prepared separately) and is reused network-free — this script never
# writes to data/fineweb_30k or any other existing data_dir.
#
# MEAN ARM ONLY (no _mask companion at this scale): the full mean+mask sweep at
# 500k would need ~145GB, more than this shared disk currently has free. See the
# per-layer disk guard below, which refuses to start a layer that would leave the
# shared disk critically full rather than filling it and affecting other users.
#
# GPU 2 ONLY, sequential: same reasoning as run_llada_all.sh — the two models are
# ~16GB and ~32.5GB resident, so running them concurrently would OOM a single
# 48GB card; GPU is pinned via scripts/llada_env.sh (override with GPU=2, though
# this script refuses any other value — see below).
#
# Resumable/idempotent: each layer is its own extract_activations_sweep.py
# invocation (unmodified, existing script) with its own per-batch-resumable
# cache — already-complete layer caches are skipped, interrupted ones resume
# mid-batch, so re-running this script after any failure (including a disk-guard
# abort) is safe and cheap.
#
# LAUNCH DETACHED (survives closing the terminal):
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_llada_500k.sh > /dev/null 2>&1 &
#
# It self-logs to outputs/llada_500k_extraction_<timestamp>.log regardless of how
# it is launched, so follow progress with:
#   tail -f "$(ls -t outputs/llada_500k_extraction_*.log | head -1)"
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

# This task was explicitly scoped to GPU 2 only (the one idle card on a shared
# 6-GPU machine) — refuse to run on anything else rather than silently
# defaulting elsewhere if GPU is set in the environment.
if [[ "$GPU" != "2" ]]; then
  echo "!! GPU=$GPU requested, but this run is pinned to GPU 2 only. Refusing." >&2
  exit 1
fi

DATA_DIR="data/fineweb_500k"
MIN_FREE_GB_8B="${MIN_FREE_GB_8B:-20}"    # ~7.63GB/layer cache + temp batch dir + margin
MIN_FREE_GB_MINI="${MIN_FREE_GB_MINI:-12}" # ~3.81GB/layer cache + temp batch dir + margin

mkdir -p outputs/new_models_500k

LOG="${LOG:-outputs/llada_500k_extraction_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LLaDA 500k FineWeb extraction pipeline (mean arm, sequential)  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"

# --- preflight specific to the 500k dir (llada_env.sh's own preflight hard-
# checks data/fineweb/sentences.json for exactly 30000 rows, which is the WRONG
# invariant here — this checks the 500k corpus instead). ----------------------
preflight_500k() {
  local rc=0
  if [[ ! -x "$CONDA_ENV_PATH/bin/python" ]]; then
    echo "!! conda env python not found: $CONDA_ENV_PATH/bin/python" >&2
    rc=1
  fi
  if [[ ! -d "$HF_CACHE_DIR" ]]; then
    echo "!! HF_CACHE_DIR does not exist: $HF_CACHE_DIR" >&2
    rc=1
  fi
  if [[ ! -f "$DATA_DIR/sentences.json" ]]; then
    echo "!! $DATA_DIR/sentences.json is missing." >&2
    echo "   Refusing to run: expected the pre-existing 500k FineWeb sentence set." >&2
    rc=1
  fi
  (( rc != 0 )) && { echo "preflight FAILED — nothing was run." >&2; return 1; }

  local probe
  probe=$("$CONDA_ENV_PATH/bin/python" -c \
    "import json,sys,acttrans; print(sys.executable, len(json.load(open('$DATA_DIR/sentences.json'))))" 2>&1) || {
      echo "!! env python failed to run, 'import acttrans' failed, or sentences.json is unreadable:" >&2
      echo "   $probe" >&2
      echo "preflight FAILED — nothing was run." >&2
      return 1
    }
  local interp="${probe%% *}" n_sent="${probe##* }"
  if [[ "$n_sent" != "500000" ]]; then
    echo "!! $DATA_DIR/sentences.json holds $n_sent sentences, expected 500000." >&2
    echo "preflight FAILED — nothing was run." >&2
    return 1
  fi
  echo "preflight OK  |  interpreter=$interp"
  echo "              |  HF_CACHE_DIR=$HF_CACHE_DIR"
  echo "              |  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (-> physical GPU $GPU)"
  echo "              |  corpus=$DATA_DIR/sentences.json ($n_sent sentences)"
}
preflight_500k || exit 1

free_gb() { df --output=avail -BG /mnt/C-SSD | tail -1 | tr -dc '0-9'; }

# --- per-layer disk guard: check BEFORE each layer, not once at the top, so a
# disk that fills from OTHER users' concurrent activity on this shared machine
# stops this pipeline before it writes a file it can't fit, rather than after. -
extract_layer() {  # $1 = config, $2 = layer, $3 = min_free_gb, $4 = human label
  local cfg="$1" layer="$2" min_free="$3" label="$4"
  local free; free=$(free_gb)
  if (( free < min_free )); then
    echo "!! only ${free}GB free on /mnt/C-SSD (need >= ${min_free}GB for $label layer $layer)." >&2
    echo "!! refusing to start this layer — shared disk is too full. Stopping the pipeline here." >&2
    return 1
  fi
  echo
  echo "############################################################"
  echo "# $label layer $layer   (${free}GB free before this layer)"
  echo "#   $(date '+%Y-%m-%d %H:%M:%S')  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> physical GPU $GPU)"
  echo "############################################################"
  $RUN extract_activations_sweep.py --config "$cfg" --layers "$layer"
}

declare -a STATUS

# --- STEP 1: LLaDA-8B-Instruct, mean position, one layer at a time -----------
LLADA8B_CFG="config/new_models/fineweb_llada8b_500k.toml"
for layer in 10 12 14 16 18 20 22; do
  if extract_layer "$LLADA8B_CFG" "$layer" "$MIN_FREE_GB_8B" "LLaDA-8B-Instruct 500k"; then
    STATUS+=("OK    LLaDA-8B-Instruct layer $layer")
  else
    STATUS+=("FAIL  LLaDA-8B-Instruct layer $layer (exit $?)")
    echo "!! stopping STEP 1 early — see disk/error above." >&2
    break
  fi
done

# --- STEP 2: LLaDA2.0-mini, mean position, one layer at a time ---------------
LLADA2MINI_CFG="config/new_models/fineweb_llada2mini_500k.toml"
for layer in 6 8 10 12 14; do
  if extract_layer "$LLADA2MINI_CFG" "$layer" "$MIN_FREE_GB_MINI" "LLaDA2.0-mini 500k"; then
    STATUS+=("OK    LLaDA2.0-mini layer $layer")
  else
    STATUS+=("FAIL  LLaDA2.0-mini layer $layer (exit $?)")
    echo "!! stopping STEP 2 early — see disk/error above." >&2
    break
  fi
done

# --- integrity: exiting 0 is NOT evidence the caches are usable --------------
echo
echo "############################################################"
echo "# VERIFY"
echo "############################################################"
$RUN scripts/verify_activations.py --config "$LLADA8B_CFG" --layers 10 12 14 16 18 20 22
STATUS+=("verify LLaDA-8B-Instruct: exit $?")
$RUN scripts/verify_activations.py --config "$LLADA2MINI_CFG" --layers 6 8 10 12 14
STATUS+=("verify LLaDA2.0-mini: exit $?")

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Artifacts:"
echo "  LLaDA-8B activations   : data/fineweb_500k/activations/LLaDA-8B-Instruct_l{10,12,14,16,18,20,22}_mean.pt"
echo "  LLaDA2.0-mini          : data/fineweb_500k/activations/LLaDA2.0-mini_l{6,8,10,12,14}_mean.pt"
echo "  Log                    : $LOG"
echo
echo "NOTE: this covers TARGET-model activations only. A row-aligned"
echo "Llama-3.2-1B-Instruct source cache at data/fineweb_500k (needed before any"
echo "Procrustes fit can use this data) was intentionally NOT extracted — it was"
echo "outside what was asked. Say the word if you want that added too (~3.8GB,"
echo "one layer, fast — the source model is 1B params)."
