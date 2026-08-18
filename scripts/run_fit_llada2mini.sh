#!/usr/bin/env bash
# Fit the closed-form Procrustes "floor" translators for the LLaDA2.0-mini target:
# Llama-3.2-1B l8 (source, last-token) -> inclusionAI/LLaDA2.0-mini at each swept
# target layer {6,8,10,12,14}, for BOTH extraction arms and BOTH variants:
#
#   arm      mean  -> LLaDA2.0-mini_l{k}_mean.pt   (mask-weighted mean over real tokens)
#            mask  -> LLaDA2.0-mini_l{k}_mask.pt   (hidden state at an appended [MASK])
#   variant  plain    orthogonal Procrustes            (..__linear__procrustes.pt)
#            whiten   anisotropy-aware Procrustes      (..__linear__procrustes_whiten.pt)
#
# 5 layers x 2 arms x 2 variants = 20 checkpoints in outputs/new_models/.
#
# WHY THIS SCRIPT IS SEPARATE from run_fit_llada.sh (the LLaDA-8B counterpart),
# rather than folding this model into it: run_fit_llada.sh's own header explicitly
# excluded LLaDA2.0-mini because its activation caches were unusable (5/5 mean
# layers and most mask layers came out 100% NaN — an uninitialized rotary table,
# see utils/hf.py::_repair_rope_buffers's docstring). That has since been fixed
# and the caches re-extracted + verified (scripts/run_llada_mini_after_sweeps.sh,
# completed 2026-08-12 12:55; corrupt originals quarantined to
# data/fineweb/activations/corrupt_pre_rope_fix/). Keeping a dedicated script
# mirrors this repo's existing one-script-per-target-model convention (e.g.
# run_fit_llada_lobo.sh vs run_fit_llada.sh) rather than complicating the 8B
# script's now-stale exclusion note.
#
# WHY BOTH VARIANTS: LLaDA-8B's mean-pooled residual stream turned out to be
# dominated by one shared direction (plain-Procrustes held-out retrieval ~chance,
# whitened ~0.92) — see run_fit_llada.sh for the full writeup. Whether the same
# anisotropy holds for this MoE/mini architecture is exactly the kind of thing
# this sweep is meant to find out, so both variants are fit here too rather than
# assuming the answer transfers.
#
# The fit is closed-form (CPU SVD); only the held-out retrieval check touches the
# GPU, and only with a small cosine matrix, so this is safe to run beside other
# GPU work (see run_fit_llada.sh). Failure-tolerant and idempotent (skips existing
# checkpoints).
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)
source scripts/llada_env.sh

OUT="outputs/new_models"
mkdir -p "$OUT"
LOG="${LOG:-outputs/llada2mini_fit_$(date '+%Y%m%d_%H%M%S').log}"
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# LLaDA2.0-mini Procrustes fits (5 layers x mean/mask x plain/whiten)  ->  $LOG"
echo "#   $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
preflight || exit 1

missing=0
for arm in mean mask; do
  for layer in 6 8 10 12 14; do
    f="data/fineweb/activations/LLaDA2.0-mini_l${layer}_${arm}.pt"
    [[ -f "$f" ]] || { echo "!! missing: $f" >&2; missing=1; }
  done
done
if (( missing )); then
  echo "!! One or more LLaDA2.0-mini activation caches are missing." >&2
  echo "   Extract with: CUDA_VISIBLE_DEVICES=\$GPU \$RUN extract_activations_sweep.py \\" >&2
  echo "     --config config/new_models/fineweb_llada2mini.toml --layers 6 8 10 12 14" >&2
  echo "   (and again with fineweb_llada2mini_mask.toml for the mask arm)" >&2
  exit 1
fi

declare -a STATUS

fit() {  # $1=config  $2=arm(mean|mask)  $3=target_layer  $4=variant(plain|whiten)
  local cfg="$1" arm="$2" k="$3" variant="$4"
  local tag="procrustes"; local flag=""
  if [[ "$variant" == "whiten" ]]; then tag="procrustes_whiten"; flag="--whiten"; fi
  local ckpt="$OUT/best_translator__Llama-3.2-1B-Instruct_l8__LLaDA2.0-mini_l${k}_${arm}__linear__${tag}.pt"
  echo
  echo "=================================================================="
  echo "  FIT  Llama-1B l8 -> LLaDA2.0-mini l${k} ${arm} [${variant}]   $(date '+%H:%M:%S')"
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

for k in 6 8 10 12 14; do
  fit config/new_models/fineweb_llada2mini.toml      mean "$k" plain
  fit config/new_models/fineweb_llada2mini.toml      mean "$k" whiten
  fit config/new_models/fineweb_llada2mini_mask.toml mask "$k" plain
  fit config/new_models/fineweb_llada2mini_mask.toml mask "$k" whiten
done

echo
echo "############################################################"
echo "# SUMMARY  $(date '+%Y-%m-%d %H:%M:%S')"
echo "############################################################"
for s in "${STATUS[@]}"; do echo "  $s"; done
echo
echo "Checkpoints ($(ls -1 "$OUT"/best_translator__*LLaDA2.0-mini*.pt 2>/dev/null | wc -l)):"
ls -1 "$OUT"/best_translator__*LLaDA2.0-mini*.pt 2>/dev/null | sed 's#.*/##'
echo
echo "Next: ./scripts/run_ab_llada2mini.sh"
