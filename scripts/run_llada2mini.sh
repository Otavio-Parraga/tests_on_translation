#!/usr/bin/env bash
# Extract FineWeb activations for inclusionAI/LLaDA2.0-mini (20 decoder layers) at
# the middle-third sweep {6,8,10,12,14}. Reuses data/fineweb/sentences.json
# (network-free, row-aligned with the existing Llama-1B source cache). Pinned to
# GPU 2 by default (override with GPU=n).
#
# Same diffusion-LM caveat as run_llada8b.sh: token_position = "mean", so caches
# land with a _mean suffix. Two extra notes specific to this model:
#   * ~32.5 GB resident — 16B total params, all experts held on-device regardless
#     of the ~1.4B routed per token. Do NOT run this concurrently with the 8B
#     sweep on one 48 GB card; run_llada_all.sh sequences them for that reason.
#   * Model load is slow (~14.8k weight tensors materialized per load) and
#     extract_activations_sweep.py reloads per layer, so budget ~15-25 min of
#     pure loading across the 5 layers on top of compute.
#
# Resumable/idempotent: already-complete layer caches are skipped, interrupted
# ones resume mid-batch.
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

preflight || exit 1
mkdir -p outputs/new_models

echo "############################################################"
echo "# LLaDA2.0-mini activations (layers 6 8 10 12 14)"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> physical GPU $GPU)"
echo "############################################################"
$RUN extract_activations_sweep.py \
  --config config/new_models/fineweb_llada2mini.toml --layers 6 8 10 12 14
RC=$?

echo
echo "DONE $(date '+%Y-%m-%d %H:%M:%S')  exit=$RC"
echo "Activations: data/fineweb/activations/LLaDA2.0-mini_l{6,8,10,12,14}_mean.pt"
exit $RC
