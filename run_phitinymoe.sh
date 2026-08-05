#!/usr/bin/env bash
# Extract FineWeb activations for microsoft/Phi-tiny-MoE-instruct (32 layers) at
# the middle-third sweep {10,12,14,16,18,20,22}, mirroring run_new_models.sh's
# Qwen/Gemma steps. Reuses data/fineweb/sentences.json (network-free, row-aligned
# with the existing Llama-1B source cache). Pinned to GPU 1.
#
# Resumable/idempotent: already-complete layer caches are skipped, interrupted
# ones resume mid-batch.
set -uo pipefail
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
RUN="conda run -n acteng python"

mkdir -p outputs/new_models

echo "############################################################"
echo "# Phi-tiny-MoE-instruct activations (layers 10 12 14 16 18 20 22)"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')  (CUDA_VISIBLE_DEVICES=1 -> physical GPU 1)"
echo "############################################################"
$RUN extract_activations_sweep.py \
  --config config/new_models/fineweb_phitinymoe.toml --layers 10 12 14 16 18 20 22
RC=$?

echo
echo "DONE $(date '+%Y-%m-%d %H:%M:%S')  exit=$RC"
echo "Activations: data/fineweb/activations/Phi-tiny-MoE-instruct_l{10,12,14,16,18,20,22}.pt"
