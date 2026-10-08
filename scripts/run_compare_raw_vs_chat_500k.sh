#!/usr/bin/env bash
# Build the CHAT-TEMPLATED 500k arm end to end and compare it against the RAW arm
# that already exists on this machine (fitted + A/B-swept in outputs/new_models_500k
# and outputs/ab_eval/llada_500k). LLaDA-8B-Instruct only: the mini's A/B readout
# is broken (P(match) ~ 4e-6), so it cannot be compared by A/B yet.
#
#   STEP 1  extract templated Llama-3.2-1B SOURCE caches, layer 8, {mean, -1}   (GPU, ~40 min)
#   STEP 2  verify them
#   STEP 3  fit Procrustes, source l8 (-1) -> LLaDA-8B templated, for
#           layers {10..26} x target {mean, -1} x {plain, whiten}  = 36 translators
#   STEP 4  A/B sweep over those 36 (same coefficient grid / method as the raw run) (GPU, ~2.5 h)
#   STEP 5  ab_report on BOTH arms' results dirs (raw has no summary.csv yet)
#   STEP 6  compare_raw_vs_chat_500k.py -> side-by-side table + CSV
#
# The templated LLaDA-8B target caches were extracted by scripts/run_llada_500k_chat.sh
# (complete, verified). Source-side templated caches are the one missing input.
#
# WHY THE ARMS STAY IN SEPARATE DIRS: templated translators get the SAME filename
# as raw ones (..._l22_mean__linear__procrustes.pt), and results_io dedups on that
# name, so one merged results.jsonl would collapse the arms into each other. The
# raw arm is read-only here; everything templated lands in *_500k_chat dirs.
#
# GPU 2 ONLY (refuses otherwise). Meant to be launched through
# scripts/wait_gpu_then_run.sh so it starts only once GPU 2 has been idle for a
# while. Every step is resumable / idempotent: extraction and ab_sweep skip
# finished work, fit_procrustes is cheap to redo.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/llada_env.sh

if [[ "$GPU" != "2" ]]; then
  echo "!! GPU=$GPU requested, but this run is pinned to GPU 2 only. Refusing." >&2
  exit 1
fi

CFG="config/new_models/fineweb_llada8b_500k_chat.toml"
DATA_DIR="data/fineweb_500k_chat"
TR_DIR="outputs/new_models_500k_chat"
AB_RAW="outputs/ab_eval/llada_500k"
AB_CHAT="outputs/ab_eval/llada_500k_chat"
SOURCE_LAYER=8
TARGET_LAYERS=(10 12 14 16 18 20 22 24 26)
TARGET_POSITIONS=(mean -1)
# Identical to scripts/run_ab_llada8b_500k.sh (the raw arm), so the arms are comparable.
COEFFS="${COEFFS:--50 -10 -5 -2 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 2 5 10 50}"
METHODS="${METHODS:-CAA}"

mkdir -p "$TR_DIR" "$AB_CHAT"
LOG="${LOG:-outputs/compare_raw_vs_chat_500k_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# RAW vs CHAT-TEMPLATED 500k comparison (LLaDA-8B)  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"

preflight() {
  local rc=0
  [[ -x "$CONDA_ENV_PATH/bin/python" ]] || { echo "!! conda env python not found" >&2; rc=1; }
  [[ -d "$HF_CACHE_DIR" ]] || { echo "!! HF_CACHE_DIR does not exist: $HF_CACHE_DIR" >&2; rc=1; }
  [[ -f "$CFG" ]] || { echo "!! config missing: $CFG" >&2; rc=1; }
  [[ -f "$DATA_DIR/sentences.json" ]] || { echo "!! $DATA_DIR/sentences.json missing" >&2; rc=1; }
  [[ -f "$AB_RAW/results.jsonl" ]] || { echo "!! raw arm results missing: $AB_RAW/results.jsonl" >&2; rc=1; }
  local n_tgt; n_tgt=$(ls "$DATA_DIR"/activations/LLaDA-8B-Instruct_l*.pt 2>/dev/null | wc -l)
  (( n_tgt >= 18 )) || { echo "!! expected 18 templated LLaDA-8B target caches in $DATA_DIR/activations, found $n_tgt" >&2; rc=1; }
  (( rc != 0 )) && { echo "preflight FAILED — nothing was run." >&2; return 1; }
  local probe
  probe=$("$CONDA_ENV_PATH/bin/python" -c "
import hashlib, json, acttrans
raw = open('data/fineweb_500k/sentences.json','rb').read(); chat = open('$DATA_DIR/sentences.json','rb').read()
print(len(json.loads(chat)), hashlib.sha256(raw).hexdigest() == hashlib.sha256(chat).hexdigest())" 2>&1) || {
      echo "!! env python / import acttrans / sentences.json failed: $probe" >&2; return 1; }
  read -r n_sent same <<<"$probe"
  [[ "$n_sent" == "500000" && "$same" == "True" ]] || { echo "!! sentence set is not the 500k raw set ($n_sent, identical=$same)" >&2; return 1; }
  echo "preflight OK  |  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (-> physical GPU $GPU)  |  HF_CACHE_DIR=$HF_CACHE_DIR"
  echo "              |  templated target caches: $n_tgt  |  corpus: $n_sent rows, identical to raw"
  echo "              |  coefficients: $COEFFS  |  methods: $METHODS"
}
preflight || exit 1

declare -a STATUS
step() {
  local label="$1"; shift
  echo; echo "############################################################"
  echo "# $label"; echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "############################################################"
  if "$@"; then STATUS+=("OK    $label"); return 0; fi
  local rc=$?; STATUS+=("FAIL  $label (exit $rc)"); return "$rc"
}
fail_summary() { echo; echo "# ABORTED  $(date '+%Y-%m-%d %H:%M:%S')"; for s in "${STATUS[@]}"; do echo "  $s"; done; exit 1; }

# --- STEP 1: templated SOURCE caches (both positions, one pass) --------------
step "STEP 1  Llama-3.2-1B source l$SOURCE_LAYER templated x {${TARGET_POSITIONS[*]}}" \
  $RUN extract_activations_multi.py --config "$CFG" --role source --layers "$SOURCE_LAYER" \
    --positions "${TARGET_POSITIONS[@]}" --batch-size 32 --min-free-gb 10 || fail_summary

# --- STEP 2: verify ----------------------------------------------------------
for pos in "${TARGET_POSITIONS[@]}"; do
  step "STEP 2  verify source cache position=$pos" \
    $RUN scripts/verify_activations.py --config "$CFG" --role source_model --token-position "$pos" --layers "$SOURCE_LAYER" \
    || fail_summary
done

# --- STEP 3: fits — source -1 (the raw arm's convention) x target {mean,-1} x {plain,whiten}
n_fit_fail=0
for layer in "${TARGET_LAYERS[@]}"; do
  for pos in "${TARGET_POSITIONS[@]}"; do
    step "STEP 3  fit plain    l$SOURCE_LAYER -> l$layer tgt=$pos" \
      $RUN fit_procrustes.py --config "$CFG" --source-layer "$SOURCE_LAYER" --target-layer "$layer" \
        --source-token-position -1 --target-token-position "$pos" || ((n_fit_fail++))
    step "STEP 3  fit whitened l$SOURCE_LAYER -> l$layer tgt=$pos" \
      $RUN fit_procrustes.py --config "$CFG" --source-layer "$SOURCE_LAYER" --target-layer "$layer" \
        --source-token-position -1 --target-token-position "$pos" --whiten || ((n_fit_fail++))
  done
done
n_tr=$(ls "$TR_DIR"/best_translator__*LLaDA-8B-Instruct_l*__linear__procrustes*.pt 2>/dev/null | wc -l)
echo; echo "  translators in $TR_DIR: $n_tr  (fit failures: $n_fit_fail)"
(( n_tr > 0 )) || fail_summary

# --- STEP 4: A/B sweep, separate results dir from the raw arm ----------------
for method in $METHODS; do
  step "STEP 4  A/B sweep $method over $n_tr templated translators" \
    $RUN ab_sweep.py --translators "$TR_DIR/best_translator__*LLaDA-8B-Instruct_l*__linear__procrustes*.pt" \
      --methods "$method" --coefficients $COEFFS --device cuda:0 --out-dir "$AB_CHAT" || true
done

# --- STEP 5: summaries for BOTH arms (raw dir has only results.jsonl so far) --
step "STEP 5  ab_report raw arm"  $RUN ab_report.py --results "$AB_RAW/*.jsonl"  --out-dir "$AB_RAW"  || true
step "STEP 5  ab_report chat arm" $RUN ab_report.py --results "$AB_CHAT/*.jsonl" --out-dir "$AB_CHAT" || true

# --- STEP 6: the comparison ----------------------------------------------------
step "STEP 6  compare raw vs chat" $RUN compare_raw_vs_chat_500k.py || true

echo; echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Comparison table : outputs/ab_eval/compare_raw_vs_chat_500k.csv  (also printed above)"
echo "Templated fits   : $TR_DIR/    A/B rows: $(wc -l < "$AB_CHAT/results.jsonl" 2>/dev/null || echo 0)"
echo "Log              : $LOG"
