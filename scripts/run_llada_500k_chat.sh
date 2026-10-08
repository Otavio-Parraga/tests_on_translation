#!/usr/bin/env bash
# CHAT-TEMPLATED 500k FineWeb extraction for BOTH LLaDA models, TWO read positions
# per layer, ONE forward pass per model:
#
#   STEP 1  GSAI-ML/LLaDA-8B-Instruct  layers {10,12,14,16,18,20,22,24,26}  x  {mean, -1}
#   STEP 2  inclusionAI/LLaDA2.0-mini  layers {6,8,10,12,14,16}             x  {mean, -1}
#   STEP 3  verify every cache (both positions, both models)
#
# WHAT THIS IS FOR: the Procrustes map is applied to CAA steering vectors that
# were extracted from CHAT-TEMPLATED prompts, but the existing 500k caches
# (data/fineweb_500k, complete and untouched) were extracted from RAW sentences
# with mean pooling. This run re-extracts the same 500k sentences templated, and
# reads each layer at two positions from the same forward pass:
#   mean  -> mask-weighted mean over the real tokens            ..._l{L}_mean.pt
#   -1    -> the last real token: the final token of the assistant-turn opener,
#            i.e. the position immediately before generation    ..._l{L}.pt
# so "does the template help" and "is mean pooling enough" are answered by ONE
# extraction against the templated Llama-3.2-1B source caches being built on
# another machine over the same sentence set.
#
# ROW ALIGNMENT IS THE WHOLE GAME. data/fineweb_500k_chat/sentences.json must be
# byte-identical to data/fineweb_500k/sentences.json (preflight checks the hash);
# the Llama source caches extracted elsewhere must be built from that same file
# with chat_template = true and dropped into data/fineweb_500k_chat/activations/.
#
# NOTHING EXISTING IS TOUCHED: new data_dir, new configs; the raw 500k caches and
# every config pointing at data/fineweb_500k are read-only here. Templated caches
# carry a chat_template=true flag that both extractors cross-check, so a templated
# run can never silently reuse a raw cache.
#
# GPU 2 ONLY, sequential: the two models are ~16GB and ~33GB resident; this script
# refuses any other GPU value (see below).
#
# Resumable: extract_activations_multi.py checkpoints every 64 batches and skips
# complete caches, so re-running after any failure continues where it stopped.
#
# LAUNCH DETACHED (survives closing the terminal):
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_llada_500k_chat.sh > /dev/null 2>&1 &
# Follow with:
#   tail -f "$(ls -t outputs/llada_500k_chat_*.log | head -1)"
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/llada_env.sh

if [[ "$GPU" != "2" ]]; then
  echo "!! GPU=$GPU requested, but this run is pinned to GPU 2 only. Refusing." >&2
  exit 1
fi

DATA_DIR="data/fineweb_500k_chat"
LLADA8B_CFG="config/new_models/fineweb_llada8b_500k_chat.toml"
LLADA2MINI_CFG="config/new_models/fineweb_llada2mini_500k_chat.toml"
LLADA8B_LAYERS=(10 12 14 16 18 20 22 24 26)
LLADA2MINI_LAYERS=(6 8 10 12 14 16)
POSITIONS=(mean -1)
MIN_FREE_GB="${MIN_FREE_GB:-40}"   # ~75GB (8B) + ~25GB (mini) total; guard re-checked inside the extractor every flush

mkdir -p outputs/new_models_500k_chat
LOG="${LOG:-outputs/llada_500k_chat_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LLaDA 500k CHAT-TEMPLATED extraction (mean + -1, one pass/model)  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"

preflight_chat() {
  local rc=0
  [[ -x "$CONDA_ENV_PATH/bin/python" ]] || { echo "!! conda env python not found: $CONDA_ENV_PATH/bin/python" >&2; rc=1; }
  [[ -d "$HF_CACHE_DIR" ]] || { echo "!! HF_CACHE_DIR does not exist: $HF_CACHE_DIR" >&2; rc=1; }
  [[ -f "$LLADA8B_CFG" && -f "$LLADA2MINI_CFG" ]] || { echo "!! a config is missing" >&2; rc=1; }
  [[ -f "$DATA_DIR/sentences.json" ]] || { echo "!! $DATA_DIR/sentences.json is missing — refusing (row alignment)." >&2; rc=1; }
  (( rc != 0 )) && { echo "preflight FAILED — nothing was run." >&2; return 1; }

  local probe
  probe=$("$CONDA_ENV_PATH/bin/python" -c "
import hashlib, json, sys, acttrans
raw = open('data/fineweb_500k/sentences.json','rb').read()
chat = open('$DATA_DIR/sentences.json','rb').read()
print(sys.executable, len(json.loads(chat)), hashlib.sha256(raw).hexdigest() == hashlib.sha256(chat).hexdigest())" 2>&1) || {
      echo "!! env python failed, 'import acttrans' failed, or sentences.json unreadable:" >&2
      echo "   $probe" >&2; echo "preflight FAILED — nothing was run." >&2; return 1; }
  read -r interp n_sent same <<<"$probe"
  [[ "$n_sent" == "500000" ]] || { echo "!! $DATA_DIR/sentences.json holds $n_sent sentences, expected 500000." >&2; return 1; }
  [[ "$same" == "True" ]] || { echo "!! $DATA_DIR/sentences.json differs from data/fineweb_500k/sentences.json — NOT row-aligned. Refusing." >&2; return 1; }
  local free; free=$(df --output=avail -BG /mnt/C-SSD | tail -1 | tr -dc '0-9')
  (( free >= MIN_FREE_GB )) || { echo "!! only ${free}GB free on /mnt/C-SSD (need >= ${MIN_FREE_GB}GB). Refusing." >&2; return 1; }
  echo "preflight OK  |  interpreter=$interp"
  echo "              |  HF_CACHE_DIR=$HF_CACHE_DIR"
  echo "              |  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (-> physical GPU $GPU)"
  echo "              |  corpus=$DATA_DIR/sentences.json ($n_sent sentences, identical to raw 500k)"
  echo "              |  free disk=${free}GB"
}
preflight_chat || exit 1

declare -a STATUS
step() {  # $1 = label; rest = command
  local label="$1"; shift
  echo; echo "############################################################"
  echo "# $label"; echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "############################################################"
  if "$@"; then STATUS+=("OK    $label"); return 0; fi
  local rc=$?; STATUS+=("FAIL  $label (exit $rc)"); return "$rc"
}

# --- STEP 1: LLaDA-8B-Instruct, all layers x both positions, one pass ---------
step "STEP 1  LLaDA-8B-Instruct 500k templated  layers ${LLADA8B_LAYERS[*]}  x  {${POSITIONS[*]}}" \
  $RUN extract_activations_multi.py --config "$LLADA8B_CFG" \
    --layers "${LLADA8B_LAYERS[@]}" --positions "${POSITIONS[@]}" --min-free-gb 15 || true

# --- STEP 2: LLaDA2.0-mini, all layers x both positions, one pass -------------
step "STEP 2  LLaDA2.0-mini 500k templated  layers ${LLADA2MINI_LAYERS[*]}  x  {${POSITIONS[*]}}" \
  $RUN extract_activations_multi.py --config "$LLADA2MINI_CFG" \
    --layers "${LLADA2MINI_LAYERS[@]}" --positions "${POSITIONS[@]}" --min-free-gb 15 || true

# --- STEP 3: integrity — exit 0 is NOT evidence the caches are usable ----------
for pos in "${POSITIONS[@]}"; do
  step "STEP 3  verify LLaDA-8B-Instruct  position=$pos" \
    $RUN scripts/verify_activations.py --config "$LLADA8B_CFG" --token-position "$pos" --layers "${LLADA8B_LAYERS[@]}" || true
  step "STEP 3  verify LLaDA2.0-mini      position=$pos" \
    $RUN scripts/verify_activations.py --config "$LLADA2MINI_CFG" --token-position "$pos" --layers "${LLADA2MINI_LAYERS[@]}" || true
done

echo; echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Artifacts (data/fineweb_500k_chat/activations/):"
echo "  LLaDA-8B-Instruct_l{${LLADA8B_LAYERS[*]// /,}}_mean.pt   and   LLaDA-8B-Instruct_l{...}.pt   (-1 arm)"
echo "  LLaDA2.0-mini_l{${LLADA2MINI_LAYERS[*]// /,}}_mean.pt         and   LLaDA2.0-mini_l{...}.pt         (-1 arm)"
echo "  Log: $LOG"
echo
echo "SOURCE SIDE (other machine) must produce, from THIS branch's code and THIS sentences.json:"
echo "  extract_activations_multi.py --config $LLADA8B_CFG --role source --layers 8 --positions mean -1"
echo "  -> data/fineweb_500k_chat/activations/Llama-3.2-1B-Instruct_l8.pt  and  _l8_mean.pt"
echo "Then fit: fit_procrustes.py --config <cfg> --source-layer 8 --target-layer <L>  (both arms; see config header)."
