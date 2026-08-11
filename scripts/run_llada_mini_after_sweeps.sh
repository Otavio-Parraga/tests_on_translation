#!/usr/bin/env bash
# Re-extract the LLaDA2.0-mini activations, but only AFTER the A/B sweeps finish.
#
# WHY THIS IS GATED rather than launched directly: the mini needs ~33 GB and the
# running sweep holds ~16 GB of the same 48 GB card. Starting now would either OOM
# this job or, far worse, OOM the sweep — which is 3 methods x 7 behaviors x 196
# blocks of work. So this waits for the sweep to exit and for the memory to
# actually come back before touching the GPU.
#
# WHY A RERUN IS NEEDED: every mini cache on disk is corrupt. LLaDA2MoeRotaryEmbedding
# registers inv_freq with persistent=False, so it is absent from the checkpoint and
# never restored; it is computed in __init__, which under transformers 5.x fast init
# runs on the meta device, so .to(cuda) produced UNINITIALIZED memory instead of a
# frequency table. Fixed in utils/hf.py (_repair_rope_buffers + a corrected
# _default_rope_parameters_compat). The LLaDA-8B caches were never affected — that
# model has no ROPE_INIT_FUNCTIONS-based rotary module — and are NOT touched here.
#
# The corrupt caches are MOVED ASIDE, not deleted: extraction skips any cache whose
# n_sentences already matches, so leaving them in place would make this a silent
# no-op. Quarantining keeps them recoverable while forcing a real re-extraction.
# The 3 mask layers that happen to be finite go too — the same uninitialized table
# produced them, so "finite" is not evidence of "correct".
#
# LAUNCH DETACHED:
#   setsid nohup ./scripts/run_llada_mini_after_sweeps.sh > /dev/null 2>&1 &
#
# Follow with: tail -f "$(ls -t outputs/llada_mini_rerun_*.log | head -1)"
# Override: GPU=n, CONDA_ENV_PATH=/path, NEED_GB=n, MAX_WAIT_H=n
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/llada_env.sh

mkdir -p outputs
LOG="${LOG:-outputs/llada_mini_rerun_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

NEED_GB="${NEED_GB:-36}"      # mini peaks ~33 GB; leave headroom
MAX_WAIT_H="${MAX_WAIT_H:-10}"
DEADLINE=$(( $(date +%s) + MAX_WAIT_H * 3600 ))

echo "############################################################"
echo "# LLaDA2.0-mini re-extraction (gated on the A/B sweeps)"
echo "#   started waiting $(date '+%Y-%m-%d %H:%M:%S')"
echo "#   will not start until: no ab_sweep running AND >= ${NEED_GB}GB free on GPU $GPU"
echo "############################################################"

# --- 1. wait for the sweeps ---------------------------------------------------
# Bracket in the pattern so pgrep cannot match this script's own command line,
# and drop our own PID as a second guard — a self-match would make this wait
# forever for itself and the extraction would never run.
sweeps_running() {
  pgrep -f "ab_[s]weep.py|run_ab_[l]lada.sh" 2>/dev/null | grep -vqx "$$"
}
while sweeps_running; do
  if (( $(date +%s) > DEADLINE )); then
    echo "!! sweeps still running after ${MAX_WAIT_H}h — giving up without touching the GPU."
    exit 1
  fi
  sleep 120
done
echo "sweeps finished $(date '+%Y-%m-%d %H:%M:%S') — waiting for GPU memory to be released"

# --- 2. wait for the memory to actually come back -----------------------------
# Process exit and memory release are not simultaneous, and the finalizer may
# still be spinning up. Require the headroom to hold across consecutive checks so
# a transient dip cannot trigger a start.
STABLE=0
while (( STABLE < 3 )); do
  if (( $(date +%s) > DEADLINE )); then
    echo "!! GPU never freed ${NEED_GB}GB within ${MAX_WAIT_H}h — giving up."
    exit 1
  fi
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU")
  FREE_GB=$(( (TOTAL - USED) / 1024 ))
  if (( FREE_GB >= NEED_GB )); then
    STABLE=$(( STABLE + 1 ))
  else
    (( STABLE > 0 )) && echo "  headroom dropped to ${FREE_GB}GB — restarting the stability count"
    STABLE=0
  fi
  sleep 30
done
echo "GPU $GPU has ${FREE_GB}GB free and steady — starting $(date '+%Y-%m-%d %H:%M:%S')"

preflight || exit 1

# --- 3. quarantine the corrupt caches -----------------------------------------
QDIR="data/fineweb/activations/corrupt_pre_rope_fix"
mkdir -p "$QDIR"
cat > "$QDIR/README.md" <<'NOTE'
LLaDA2.0-mini activation caches produced BEFORE the rotary-embedding fix.

Every one of these is invalid. The model's inv_freq buffer was never
materialized (registered persistent=False, computed on the meta device), so the
forward pass ran with uninitialized memory as its rotary frequency table. All 5
mean caches are NaN in every row. The 3 finite mask caches are NOT usable
either: the same uninitialized table produced them, and uninitialized memory
varies run to run, so "finite" here means "happened not to overflow", not
"correct".

Kept only so the failure is inspectable. Do not use them for any fit or eval.
Superseded by the caches of the same name in the parent directory.
NOTE

MOVED=0
for f in data/fineweb/activations/LLaDA2.0-mini_l*_mean.pt data/fineweb/activations/LLaDA2.0-mini_l*_mask.pt; do
  [[ -e "$f" ]] || continue
  mv -n "$f" "$QDIR/" && MOVED=$(( MOVED + 1 ))
done
# Partial batch dirs would be resumed from, re-merging corrupt batches into a
# fresh cache. Move any of those aside too.
for d in data/fineweb/activations/.LLaDA2.0-mini_*_batches; do
  [[ -d "$d" ]] || continue
  mv -n "$d" "$QDIR/" && echo "  quarantined partial batch dir $(basename "$d")"
done
echo "quarantined $MOVED corrupt mini caches -> $QDIR"

# --- 4. re-extract (8B is untouched: its caches are valid) ---------------------
declare -a STATUS
step() {
  local label="$1"; shift
  echo
  echo "############################################################"
  echo "# $label"
  echo "#   $(date '+%Y-%m-%d %H:%M:%S')  (GPU $GPU)"
  echo "############################################################"
  if "$@"; then STATUS+=("OK    $label"); else
    STATUS+=("FAIL  $label (exit $?)"); echo "!! step failed: $label — continuing"
  fi
}

step "STEP 1/2  LLaDA2.0-mini MEAN activations (layers 6 8 10 12 14)" \
  $RUN extract_activations_sweep.py \
    --config config/new_models/fineweb_llada2mini.toml --layers 6 8 10 12 14

step "STEP 2/2  LLaDA2.0-mini MASK activations (layers 6 8 10 12 14)" \
  $RUN extract_activations_sweep.py \
    --config config/new_models/fineweb_llada2mini_mask.toml --layers 6 8 10 12 14

# --- 5. verify: exiting 0 is not evidence the caches are usable ----------------
step "VERIFY  mini mean caches" \
  $RUN scripts/verify_activations.py \
    --config config/new_models/fineweb_llada2mini.toml --layers 6 8 10 12 14
step "VERIFY  mini mask caches" \
  $RUN scripts/verify_activations.py \
    --config config/new_models/fineweb_llada2mini_mask.toml --layers 6 8 10 12 14

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "  A VERIFY step failing means those caches are still unusable — check the log."
echo "  Quarantined originals: $QDIR"
echo "  Log: $LOG"
