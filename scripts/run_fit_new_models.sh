#!/usr/bin/env bash
# Fit the closed-form orthogonal-Procrustes "floor" translator for each layer pair
# of the cross-family study: Llama-3.2-1B l8 (source) -> {Qwen2.5-0.5B, gemma-3-1B}
# at each swept target layer. RAW activations (normalize_activations=false in the
# configs) so the norm/inner-product-preserving geometry is faithful.
#
# Checkpoints land (auto-named) in outputs/new_models/ as
#   best_translator__Llama-3.2-1B-Instruct_l8__<TGT>_l<k>__linear__procrustes.pt
# discoverable by the geometric comparison and the A/B sweep downstream.
#
# The fit is closed-form (CPU SVD); only the held-out retrieval check uses the GPU.
# Pinned to GPU 1. Failure-tolerant + idempotent-friendly (skips a pair whose
# checkpoint already exists).
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)

export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
RUN="conda run -n acteng python"
OUT="outputs/new_models"
mkdir -p "$OUT"

declare -a STATUS

fit() {  # $1=config  $2=tgt_slug  $3=target_layer
  local cfg="$1" slug="$2" k="$3"
  local ckpt="$OUT/best_translator__Llama-3.2-1B-Instruct_l8__${slug}_l${k}__linear__procrustes.pt"
  echo
  echo "=================================================================="
  echo "  FIT  Llama-1B l8 -> ${slug} l${k}   $(date '+%H:%M:%S')"
  echo "=================================================================="
  if [[ -f "$ckpt" ]]; then
    echo "  skip (exists): $ckpt"; STATUS+=("OK(skip) ${slug} l${k}"); return
  fi
  if $RUN fit_procrustes.py --config "$cfg" --source-layer 8 --target-layer "$k"; then
    STATUS+=("OK       ${slug} l${k}")
  else
    STATUS+=("FAIL     ${slug} l${k} (exit $?)")
    echo "  !! fit failed: ${slug} l${k} — continuing"
  fi
}

for k in 8 10 12 14 16;    do fit config/new_models/fineweb_qwen0.5b.toml Qwen2.5-0.5B-Instruct "$k"; done
for k in 8 10 12 14 16 18; do fit config/new_models/fineweb_gemma1b.toml  gemma-3-1B-it          "$k"; done

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Checkpoints in $OUT/ :"
ls -1 "$OUT"/best_translator__*procrustes.pt 2>/dev/null | sed 's#.*/##'
