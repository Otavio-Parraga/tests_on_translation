#!/usr/bin/env bash
# Wait until a GPU has been IDLE for a continuous window, then run a command.
#
#   GPU=2 MINUTES=30 ./scripts/wait_gpu_then_run.sh ./scripts/run_compare_raw_vs_chat_500k.sh
#
# "Idle" = no compute process on that GPU AND used memory below IDLE_MIB (default
# 1000 MiB — X/display contexts and nvidia-smi itself sit far below that). The
# window resets to zero the moment anyone touches the card, so the command only
# starts after MINUTES of uninterrupted idleness — the point being not to jump onto
# a shared card in the gap between someone's jobs.
#
# If the command exits non-zero (e.g. another job grabbed the card right after we
# started and we OOM'd), the gate RE-ARMS and tries again, up to MAX_RETRIES. The
# pipelines this fronts are resumable, so a retry continues rather than redoing.
#
# LAUNCH DETACHED:
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   GPU=2 MINUTES=30 setsid nohup ./scripts/wait_gpu_then_run.sh ./scripts/run_compare_raw_vs_chat_500k.sh > /dev/null 2>&1 &
#   tail -f "$(ls -t outputs/gpu_gate_*.log | head -1)"
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-2}"
MINUTES="${MINUTES:-30}"
POLL_SECONDS="${POLL_SECONDS:-60}"
IDLE_MIB="${IDLE_MIB:-1000}"
MAX_RETRIES="${MAX_RETRIES:-3}"

(( $# > 0 )) || { echo "usage: GPU=n MINUTES=m $0 <command> [args...]" >&2; exit 2; }

LOG="${LOG:-outputs/gpu_gate_$(date '+%Y%m%d_%H%M%S').log}"
mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

need_polls=$(( MINUTES * 60 / POLL_SECONDS ))
echo "############################################################"
echo "# GPU gate  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')   waiting for GPU $GPU idle for ${MINUTES} min"
echo "#   (poll every ${POLL_SECONDS}s, idle = no compute apps and < ${IDLE_MIB} MiB used)"
echo "#   then: $*"
echo "############################################################"

gpu_idle() {
  local apps mem
  apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  mem=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')
  [[ -z "$mem" ]] && return 1          # nvidia-smi failed: treat as busy
  (( apps == 0 && mem < IDLE_MIB ))
}

wait_for_window() {
  local streak=0 last_report=0
  while (( streak < need_polls )); do
    if gpu_idle; then
      (( streak++ ))
    else
      if (( streak > 0 )); then
        echo "$(date '+%H:%M:%S')  GPU $GPU busy again after $(( streak * POLL_SECONDS / 60 )) idle min — window reset"
      fi
      streak=0
    fi
    # progress line every 10 min, so the log shows the gate is alive without spamming
    local now; now=$(date +%s)
    if (( now - last_report >= 600 )); then
      local mem; mem=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
      echo "$(date '+%H:%M:%S')  idle streak: $(( streak * POLL_SECONDS / 60 ))/${MINUTES} min   (GPU $GPU used: ${mem:-?})"
      last_report=$now
    fi
    (( streak < need_polls )) && sleep "$POLL_SECONDS"
  done
}

attempt=0
while :; do
  (( attempt++ ))
  wait_for_window
  echo; echo "$(date '+%Y-%m-%d %H:%M:%S')  GPU $GPU idle for ${MINUTES} min — launching (attempt $attempt/$MAX_RETRIES): $*"
  echo "############################################################"
  if GPU="$GPU" "$@"; then
    echo; echo "$(date '+%Y-%m-%d %H:%M:%S')  command finished OK."
    exit 0
  fi
  rc=$?
  echo; echo "$(date '+%Y-%m-%d %H:%M:%S')  command exited $rc."
  if (( attempt >= MAX_RETRIES )); then
    echo "!! giving up after $MAX_RETRIES attempts. The pipeline is resumable — re-run this gate to continue."
    exit "$rc"
  fi
  echo "   re-arming the gate (waiting for another ${MINUTES}-min idle window before retrying)."
done
