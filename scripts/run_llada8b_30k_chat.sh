#!/usr/bin/env bash
# CHAT-TEMPLATED 30k FineWeb experiment, end to end, LLaDA-8B-Instruct ONLY:
#
#   STEP 1  extract SOURCE activations  Llama-3.2-1B-Instruct  layer 8   (token -1)
#   STEP 2  extract TARGET activations  GSAI-ML/LLaDA-8B-Instruct  layers {10..22}  ("mean")
#   STEP 3  verify every cache is finite / right shape
#   STEP 4  fit orthogonal Procrustes per target layer, plain AND whitened
#   STEP 5  full A/B steering sweep over the resulting translators
#
# THE QUESTION THIS ANSWERS: the Procrustes map is applied to CAA steering vectors
# extracted from CHAT-TEMPLATED prompts (extract_caa_diff_source.py builds
# apply_chat_template(...) + "Answer: (A)"), but has so far been FITTED on raw
# FineWeb sentences — i.e. fitted in a different region of activation space than
# the one it is used in. This run templates both sides of the pair and measures
# whether closing that gap produces a better map.
#
# ONE VARIABLE: read positions are unchanged from the raw run (source -1, target
# "mean") and data/fineweb_30k_chat/sentences.json is a byte-for-byte copy of
# data/fineweb_30k/sentences.json, so the ONLY difference from the existing
# baseline in outputs/new_models/ is the chat template.
#
# NOTHING EXISTING IS TOUCHED. Everything lands in new, separate locations:
#   data/fineweb_30k_chat/    outputs/new_models_30k_chat/    outputs/ab_eval_30k_chat/
# The raw baseline (data/fineweb_30k, outputs/new_models, outputs/ab_eval) is
# read-only here. The templated caches also carry a chat_template=true flag that
# extract_activations_with_resume cross-checks, so a templated run can never
# silently reuse a raw cache even if pointed at the wrong data_dir.
#
# LLaDA2.0-mini is deliberately NOT part of this experiment — decide on the 8B
# result first, then extend if it pays off.
#
# GPU 2 ONLY: this script refuses any other value (see below).
#
# Resumable/idempotent at every step: extraction skips complete layer caches and
# resumes interrupted ones mid-batch; fit_procrustes is cheap to redo; ab_sweep.py
# streams to JSONL and skips combinations already present. Re-running after any
# failure is safe.
#
# LAUNCH DETACHED (survives closing the terminal):
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_llada8b_30k_chat.sh > /dev/null 2>&1 &
#
# It self-logs to outputs/llada8b_30k_chat_<timestamp>.log regardless of how it is
# launched, so follow progress with:
#   tail -f "$(ls -t outputs/llada8b_30k_chat_*.log | head -1)"
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

# Pinned to GPU 2 (the one idle card on a shared 6-GPU machine) — refuse anything
# else rather than silently grabbing another user's card if GPU is set elsewhere.
if [[ "$GPU" != "2" ]]; then
  echo "!! GPU=$GPU requested, but this run is pinned to GPU 2 only. Refusing." >&2
  exit 1
fi

CFG="config/new_models/fineweb_llada8b_30k_chat.toml"
DATA_DIR="data/fineweb_30k_chat"
TRANSLATOR_DIR="outputs/new_models_30k_chat"
AB_OUT="outputs/ab_eval_30k_chat"
SOURCE_LAYER=8
TARGET_LAYERS=(10 12 14 16 18 20 22)

mkdir -p "$TRANSLATOR_DIR" "$AB_OUT"

LOG="${LOG:-outputs/llada8b_30k_chat_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LLaDA-8B 30k CHAT-TEMPLATED pipeline  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"

# --- preflight (llada_env.sh's own preflight hard-checks data/fineweb, the wrong
# path for this run — this checks the templated dir's corpus instead) -----------
preflight_chat() {
  local rc=0
  [[ -x "$CONDA_ENV_PATH/bin/python" ]] || { echo "!! conda env python not found: $CONDA_ENV_PATH/bin/python" >&2; rc=1; }
  [[ -d "$HF_CACHE_DIR" ]] || { echo "!! HF_CACHE_DIR does not exist: $HF_CACHE_DIR" >&2; rc=1; }
  [[ -f "$CFG" ]] || { echo "!! config not found: $CFG" >&2; rc=1; }
  if [[ ! -f "$DATA_DIR/sentences.json" ]]; then
    echo "!! $DATA_DIR/sentences.json is missing — extraction would re-stream FineWeb" >&2
    echo "   and break row alignment with the raw 30k caches. Refusing." >&2
    rc=1
  fi
  (( rc != 0 )) && { echo "preflight FAILED — nothing was run." >&2; return 1; }

  # The templated corpus MUST be the same 30k sentences as the raw one, or the
  # whole template-vs-raw comparison is confounded by a different sentence set.
  local probe
  probe=$("$CONDA_ENV_PATH/bin/python" -c "
import hashlib, json, sys, acttrans
raw = open('data/fineweb_30k/sentences.json','rb').read()
chat = open('$DATA_DIR/sentences.json','rb').read()
same = hashlib.sha256(raw).hexdigest() == hashlib.sha256(chat).hexdigest()
print(sys.executable, len(json.loads(chat)), same)" 2>&1) || {
      echo "!! env python failed to run, 'import acttrans' failed, or sentences.json is unreadable:" >&2
      echo "   $probe" >&2
      echo "preflight FAILED — nothing was run." >&2
      return 1
    }
  read -r interp n_sent same <<<"$probe"
  if [[ "$n_sent" != "30000" ]]; then
    echo "!! $DATA_DIR/sentences.json holds $n_sent sentences, expected 30000." >&2
    echo "preflight FAILED — nothing was run." >&2
    return 1
  fi
  if [[ "$same" != "True" ]]; then
    echo "!! $DATA_DIR/sentences.json differs from data/fineweb_30k/sentences.json." >&2
    echo "   The template-vs-raw comparison requires an IDENTICAL sentence set." >&2
    echo "preflight FAILED — nothing was run." >&2
    return 1
  fi
  local free; free=$(df --output=avail -BG /mnt/C-SSD | tail -1 | tr -dc '0-9')
  if (( free < 10 )); then
    echo "!! only ${free}GB free on /mnt/C-SSD; this run needs ~2GB plus headroom. Refusing." >&2
    return 1
  fi
  echo "preflight OK  |  interpreter=$interp"
  echo "              |  HF_CACHE_DIR=$HF_CACHE_DIR"
  echo "              |  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (-> physical GPU $GPU)"
  echo "              |  corpus=$DATA_DIR/sentences.json ($n_sent sentences, identical to raw 30k)"
  echo "              |  free disk=${free}GB"
}
preflight_chat || exit 1

declare -a STATUS
step() {  # $1 = human label; rest = command
  local label="$1"; shift
  echo
  echo "############################################################"
  echo "# $label"
  echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "############################################################"
  if "$@"; then
    STATUS+=("OK    $label")
    return 0
  fi
  STATUS+=("FAIL  $label (exit $?)")
  return 1
}

# --- STEP 1: SOURCE side (Llama-3.2-1B-Instruct, layer 8, templated) ----------
# extract_activations_sweep.py --role source, NOT prepare_activations.py: the
# latter re-streams FineWeb from the network and rewrites sentences.json, which
# would break row alignment with the copied corpus.
step "STEP 1  source Llama-3.2-1B-Instruct layer $SOURCE_LAYER (templated)" \
  $RUN extract_activations_sweep.py --config "$CFG" --role source --layers "$SOURCE_LAYER" \
  || { echo "!! source extraction failed — nothing downstream can be fitted. Stopping." >&2
       for s in "${STATUS[@]}"; do echo "  $s"; done; exit 1; }

# --- STEP 2: TARGET side (LLaDA-8B-Instruct, layer sweep, templated) ---------
# One invocation per layer so a failure costs only that layer, and so each layer's
# resume state stays independent.
declare -a GOOD_LAYERS
for layer in "${TARGET_LAYERS[@]}"; do
  if step "STEP 2  target LLaDA-8B-Instruct layer $layer (templated)" \
       $RUN extract_activations_sweep.py --config "$CFG" --layers "$layer"; then
    GOOD_LAYERS+=("$layer")
  else
    echo "!! layer $layer failed — continuing with the remaining layers." >&2
  fi
done

if (( ${#GOOD_LAYERS[@]} == 0 )); then
  echo "!! no target layer extracted successfully. Stopping before fit/sweep." >&2
  for s in "${STATUS[@]}"; do echo "  $s"; done
  exit 1
fi

# --- STEP 3: integrity — exiting 0 is NOT evidence the caches are usable ------
step "STEP 3  verify source cache" \
  $RUN scripts/verify_activations.py --config "$CFG" --role source_model --layers "$SOURCE_LAYER" || true
step "STEP 3  verify target caches (${GOOD_LAYERS[*]})" \
  $RUN scripts/verify_activations.py --config "$CFG" --layers "${GOOD_LAYERS[@]}" || true

# --- STEP 4: fit Procrustes per target layer, plain AND whitened --------------
# Plain is exactly norm-preserving; whitened fits anisotropic residual streams
# better but gives up exact norm preservation. Both are cheap and both already
# exist for the raw baseline, so fitting both keeps the comparison like-for-like.
for layer in "${GOOD_LAYERS[@]}"; do
  step "STEP 4  fit Procrustes (plain)    l$SOURCE_LAYER -> l$layer" \
    $RUN fit_procrustes.py --config "$CFG" --source-layer "$SOURCE_LAYER" --target-layer "$layer" || true
  step "STEP 4  fit Procrustes (whitened) l$SOURCE_LAYER -> l$layer" \
    $RUN fit_procrustes.py --config "$CFG" --source-layer "$SOURCE_LAYER" --target-layer "$layer" --whiten || true
done

# --- STEP 5: A/B steering sweep over every translator just fitted -------------
# Separate --out-dir so these rows never merge with the raw baseline's results.
# ab_sweep.py reads the layer pair from each checkpoint filename and injects at the
# right target layer, so one glob covers the whole sweep. Resumable: it skips rows
# already in results.jsonl, so re-running this script continues rather than redoing.
shopt -s nullglob
TRANSLATORS=("$TRANSLATOR_DIR"/best_translator__*.pt)
shopt -u nullglob
if (( ${#TRANSLATORS[@]} == 0 )); then
  echo "!! no translator checkpoints in $TRANSLATOR_DIR — skipping the A/B sweep." >&2
  STATUS+=("SKIP  STEP 5 A/B sweep (no translators)")
else
  echo "  ${#TRANSLATORS[@]} translator checkpoints to sweep"
  step "STEP 5  A/B sweep (${#TRANSLATORS[@]} translators)" \
    $RUN ab_sweep.py \
      --translators "$TRANSLATOR_DIR/best_translator__*.pt" \
      --out-dir "$AB_OUT" \
      --device cuda:0 || true
fi

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Artifacts:"
echo "  templated source cache : $DATA_DIR/activations/Llama-3.2-1B-Instruct_l${SOURCE_LAYER}.pt"
echo "  templated target caches: $DATA_DIR/activations/LLaDA-8B-Instruct_l{${GOOD_LAYERS[*]// /,}}_mean.pt"
echo "  translators            : $TRANSLATOR_DIR/best_translator__*.pt"
echo "  A/B results            : $AB_OUT/results.jsonl"
echo "  Log                    : $LOG"
echo
echo "NEXT — the comparison this run exists for. Same metric, raw vs templated:"
echo "  $RUN ab_report.py --results $AB_OUT/results.jsonl        # templated"
echo "  $RUN ab_report.py --results outputs/ab_eval/results.jsonl # raw baseline"
echo "If the templated fit does not beat the raw one, the raw pipeline stands and"
echo "the 500k extraction can resume as it was (l10/l12 already on disk)."
