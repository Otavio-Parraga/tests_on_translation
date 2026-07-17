#!/usr/bin/env bash
#
# Run the full A/B steering sweep across all 24 FineWeb translators, split over
# the two GPUs. Each GPU process writes its own results shard; ab_report.py then
# merges the shards into results.csv / summary.csv / report.html.
#
#   GPU 0 : first half of the translators + the SOURCE baseline (Llama-1B)
#   GPU 1 : second half of the translators
#
# Fully resumable: re-running skips (translator,norm,behavior) blocks already in
# the shard. Safe to Ctrl-C and restart.
#
# Usage (run from anywhere; the script cd's to the repo root itself):
#   scripts/run_ab_sweep.sh              # full sweep (background-friendly)
#   scripts/run_ab_sweep.sh 10           # smoke test: 10 test items per coefficient

set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)

PY="conda run -n acteng python"
export HF_HUB_DISABLE_XET=1

OUT="outputs/ab_eval"
mkdir -p "$OUT"

LIMIT_ARG=""
if [[ $# -ge 1 ]]; then
  LIMIT_ARG="--limit $1"
fi

# Enumerate the 24 best translators and split in half.
mapfile -t ALL < <(ls outputs/fineweb/best_translator__*.pt | sort)
N=${#ALL[@]}
HALF=$(( (N + 1) / 2 ))
GPU0=( "${ALL[@]:0:$HALF}" )
GPU1=( "${ALL[@]:$HALF}" )

echo "==> $N translators: $HALF on cuda:0 (+source baseline), $((N-HALF)) on cuda:1"

# NOTE: activation_engineering/.env pins CUDA_VISIBLE_DEVICES="1" (loaded by
# ab_comparison.py via os.environ.setdefault). We export it explicitly per
# process so setdefault is a no-op and each process sees exactly ONE physical
# GPU as logical cuda:0. This is what actually splits the work across both GPUs.

# Physical GPU 0 — first half + source baseline
CUDA_VISIBLE_DEVICES=0 $PY ab_sweep.py \
    --device cuda:0 \
    --translators "${GPU0[@]}" \
    --with-source \
    --results-name results_gpu0.jsonl \
    $LIMIT_ARG \
    > "$OUT/log_gpu0.txt" 2>&1 &
PID0=$!

# Physical GPU 1 — second half, no source baseline (computed once on GPU 0)
CUDA_VISIBLE_DEVICES=1 $PY ab_sweep.py \
    --device cuda:0 \
    --translators "${GPU1[@]}" \
    --no-source \
    --results-name results_gpu1.jsonl \
    $LIMIT_ARG \
    > "$OUT/log_gpu1.txt" 2>&1 &
PID1=$!

echo "==> launched PID0=$PID0 (cuda:0), PID1=$PID1 (cuda:1)"
echo "    tail -f $OUT/log_gpu0.txt   /   $OUT/log_gpu1.txt"

# `|| R=$?` keeps `set -e` from aborting before we build the report when a GPU
# job exits nonzero (e.g. one GPU OOMs but the other finished its shard).
R0=0; wait $PID0 || R0=$?
R1=0; wait $PID1 || R1=$?
echo "==> gpu0 exit=$R0  gpu1 exit=$R1"

echo "==> building report from shards"
$PY ab_report.py --results "$OUT"/results_gpu*.jsonl --out-dir "$OUT"
echo "==> done. Open $OUT/report.html"
