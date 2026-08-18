#!/usr/bin/env bash
# Fit the closed-form Procrustes "floor" translators for the DIFFUSION study:
# Llama-3.2-1B l8 (source, last-token) -> GSAI-ML/LLaDA-8B-Instruct at each swept
# target layer {10,12,14,16,18,20,22}, for BOTH extraction arms and BOTH variants:
#
#   arm      mean  -> LLaDA-8B-Instruct_l{k}_mean.pt   (mask-weighted mean over real tokens)
#            mask  -> LLaDA-8B-Instruct_l{k}_mask.pt   (hidden state at an appended [MASK])
#   variant  plain    orthogonal Procrustes            (..__linear__procrustes.pt)
#            whiten   anisotropy-aware Procrustes      (..__linear__procrustes_whiten.pt)
#
# 7 layers x 2 arms x 2 variants = 28 checkpoints in outputs/new_models/.
#
# WHY BOTH VARIANTS, and why --whiten is not optional here. LLaDA's mean-pooled
# residual stream is dominated by one shared direction (||mu||/mean||h|| = 0.97,
# mean pairwise cosine 0.94), so plain-Procrustes held-out retrieval reads 0.006
# @1 — near chance — while the whitened fit on the SAME activations reads 0.922.
# Its centered effective dimensionality is 93, HIGHER than Phi-tiny-MoE's 9.6, so
# the representation is rich and the plain number is an anisotropy artifact, not a
# fact about diffusion LMs. Keeping the plain fit alongside it preserves the
# documented floor baseline and makes that gap visible in the sweep instead of
# hiding it behind a preprocessing choice.
#
# LLaDA2.0-mini is deliberately ABSENT. Its activation caches are unusable: 5/5
# mean layers and 2/5 mask layers came out 100% NaN, non-deterministically (the
# finite ones are excellent — ||mu||/mean||h|| = 0.14, effective dim 226-721 — so
# this is a load-time weight-materialization flake, not a shim incompatibility).
# Add it here once extraction gains a load-then-verify-finite retry.
#
# The fit is closed-form (CPU SVD); only the held-out retrieval check touches the
# GPU, and only with a 2048-sample cosine matrix, so this is safe to run beside
# other GPU work. Failure-tolerant and idempotent (skips existing checkpoints).
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

OUT="outputs/new_models"
mkdir -p "$OUT"
LOG="${LOG:-outputs/llada_fit_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LLaDA-8B Procrustes fits (7 layers x mean/mask x plain/whiten)  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
preflight || exit 1

declare -a STATUS

fit() {  # $1=config  $2=arm(mean|mask)  $3=target_layer  $4=variant(plain|whiten)
  local cfg="$1" arm="$2" k="$3" variant="$4"
  local tag="procrustes"; local flag=""
  if [[ "$variant" == "whiten" ]]; then tag="procrustes_whiten"; flag="--whiten"; fi
  local ckpt="$OUT/best_translator__Llama-3.2-1B-Instruct_l8__LLaDA-8B-Instruct_l${k}_${arm}__linear__${tag}.pt"
  echo
  echo "=================================================================="
  echo "  FIT  Llama-1B l8 -> LLaDA-8B l${k} ${arm} [${variant}]   $(date '+%H:%M:%S')"
  echo "=================================================================="
  if [[ -f "$ckpt" ]]; then
    echo "  skip (exists): ${ckpt##*/}"; STATUS+=("OK(skip) l${k} ${arm} ${variant}"); return
  fi
  if $RUN fit_procrustes.py --config "$cfg" --source-layer 8 --target-layer "$k" $flag; then
    STATUS+=("OK       l${k} ${arm} ${variant}")
  else
    STATUS+=("FAIL     l${k} ${arm} ${variant} (exit $?)")
    echo "  !! fit failed: l${k} ${arm} ${variant} — continuing"
  fi
}

for k in 10 12 14 16 18 20 22; do
  fit config/new_models/fineweb_llada8b.toml      mean "$k" plain
  fit config/new_models/fineweb_llada8b.toml      mean "$k" whiten
  fit config/new_models/fineweb_llada8b_mask.toml mask "$k" plain
  fit config/new_models/fineweb_llada8b_mask.toml mask "$k" whiten
done

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Checkpoints ($(ls -1 "$OUT"/best_translator__*LLaDA-8B*.pt 2>/dev/null | wc -l)):"
ls -1 "$OUT"/best_translator__*LLaDA-8B*.pt 2>/dev/null | sed 's#.*/##'
