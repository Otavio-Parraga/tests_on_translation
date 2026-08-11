#!/usr/bin/env bash
# Extract FineWeb activations for GSAI-ML/LLaDA-8B-Instruct (32 blocks) at the
# middle-third sweep {10,12,14,16,18,20,22}, mirroring run_phitinymoe.sh. Reuses
# data/fineweb/sentences.json (network-free, row-aligned with the existing
# Llama-1B source cache). Pinned to GPU 2 by default (override with GPU=n).
#
# This is a DIFFUSION LM, so extraction differs from the causal targets in one
# way that matters: token_position = "mean" in the config, because bidirectional
# attention leaves no privileged "last token". Caches therefore land with a
# _mean suffix. Extraction still uses a single clean forward pass — the iterative
# denoising loop is never invoked.
#
# Resumable/idempotent: already-complete layer caches are skipped, interrupted
# ones resume mid-batch.
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

preflight || exit 1
mkdir -p outputs/new_models

echo "############################################################"
echo "# LLaDA-8B-Instruct activations (layers 10 12 14 16 18 20 22)"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')  (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES -> physical GPU $GPU)"
echo "############################################################"
$RUN extract_activations_sweep.py \
  --config config/new_models/fineweb_llada8b.toml --layers 10 12 14 16 18 20 22
RC=$?

echo
echo "DONE $(date '+%Y-%m-%d %H:%M:%S')  exit=$RC"
echo "Activations: data/fineweb/activations/LLaDA-8B-Instruct_l{10,12,14,16,18,20,22}_mean.pt"
exit $RC
