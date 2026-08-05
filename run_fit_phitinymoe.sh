#!/usr/bin/env bash
# Fit both the raw and whitened orthogonal-Procrustes "floor" translators for
# each swept layer of Llama-3.2-1B l8 (source) -> Phi-tiny-MoE-instruct (target).
# Mirrors run_fit_new_models.sh's approach for Qwen/Gemma, but does raw AND
# whitened up front (Gemma's raw Procrustes collapsed on retrieval until
# whitened — see [[gemma-residual-scale]] / new-models-translation memory —
# so both floors are captured now rather than discovered as a follow-up).
#
# Checkpoints land (auto-named) in outputs/new_models/ as
#   best_translator__Llama-3.2-1B-Instruct_l8__Phi-tiny-MoE-instruct_l<k>__linear__procrustes[_whiten].pt
# discoverable by the geometric comparison and the A/B sweep downstream.
#
# Closed-form (CPU SVD for the raw fit; --whiten adds a CPU whitening step) —
# no GPU needed for this stage, unlike extraction/A-B-eval. Failure-tolerant +
# skips a (layer, variant) pair whose checkpoint already exists.
set -uo pipefail
cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
RUN="conda run -n acteng python"
OUT="outputs/new_models"
CFG="config/new_models/fineweb_phitinymoe.toml"
SLUG="Phi-tiny-MoE-instruct"
mkdir -p "$OUT"

declare -a STATUS

fit() {  # $1=extra_flag ("" or "--whiten")  $2=tag  $3=target_layer
  local flag="$1" tag="$2" k="$3"
  local ckpt="$OUT/best_translator__Llama-3.2-1B-Instruct_l8__${SLUG}_l${k}__linear__${tag}.pt"
  echo
  echo "=================================================================="
  echo "  FIT  Llama-1B l8 -> ${SLUG} l${k}  (${tag})   $(date '+%H:%M:%S')"
  echo "=================================================================="
  if [[ -f "$ckpt" ]]; then
    echo "  skip (exists): $ckpt"; STATUS+=("OK(skip) l${k} ${tag}"); return
  fi
  if $RUN fit_procrustes.py --config "$CFG" --source-layer 8 --target-layer "$k" $flag; then
    STATUS+=("OK       l${k} ${tag}")
  else
    STATUS+=("FAIL     l${k} ${tag} (exit $?)")
    echo "  !! fit failed: l${k} ${tag} — continuing"
  fi
}

for k in 10 12 14 16 18 20 22; do
  fit ""         "procrustes"        "$k"
  fit "--whiten" "procrustes_whiten" "$k"
done

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Checkpoints in $OUT/ :"
ls -1 "$OUT"/best_translator__*"${SLUG}"*procrustes*.pt 2>/dev/null | sed 's#.*/##'
