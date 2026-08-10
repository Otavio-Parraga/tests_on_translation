#!/usr/bin/env bash
# Layer-transfer study: how does closed-form Procrustes translation quality vary
# with the TARGET layer, when the SOURCE layer is held fixed?
#
#   Task 1 (forward, 1B -> 3B): source Llama-3.2-1B layer 8 (its ~midpoint),
#           target Llama-3.2-3B layer in {8,10,12,14,16,18} (middle third of 28).
#   Task 2 (reverse, 3B -> 1B): source Llama-3.2-3B layer 14 (its ~midpoint),
#           target Llama-3.2-1B layer in {4,6,8,10,12} (middle of 16).
#
# Method is the closed-form orthogonal Procrustes "floor" (one fit per layer pair):
# it was the strongest transport in the l8->l8 study and needs no GPU training, so a
# whole layer sweep is cheap. Everything is comparison in native activation space via
# compare_translated_and_original.py, which reads the model pair from each checkpoint.
#
# Idempotent: activation caches and (with a guard) fitted checkpoints are reused.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)

RUN="conda run -n acteng python"
FWD_CFG="config/fineweb/linear_procrustes_raw.toml"      # 1B (source) -> 3B (target)
REV_CFG="config/layer_sweep/procrustes_3b_to_1b.toml"    # 3B (source) -> 1B (target)

T1_DIR="outputs/layer_sweep/task1_1b_to_3b"
T2_DIR="outputs/layer_sweep/task2_3b_to_1b"
mkdir -p "$T1_DIR" "$T2_DIR"

echo "############################################################"
echo "# STEP 1/3  Extract FineWeb activations for the new layers"
echo "#   3B (target) needs: 10 12 14 16 18   (l8 already cached)"
echo "#   1B (source) needs: 4 6 10 12        (l8 already cached)"
echo "# Each call runs source-extraction on GPU:0 and target on GPU:1 in"
echo "# parallel; complete caches are skipped. New layers are paired so both"
echo "# GPUs stay busy per call."
echo "############################################################"
# pairs: (1B src layer, 3B tgt layer)
extract() {  # $1=src_layer(1B)  $2=tgt_layer(3B)
  echo "--- extract 1B l$1 (GPU:0) + 3B l$2 (GPU:1) ---"
  $RUN prepare_activations.py --config "$FWD_CFG" --source-layer "$1" --target-layer "$2"
}
extract 4  10
extract 6  12
extract 10 14
extract 12 16
extract 8  18   # 1B l8 cached -> effectively only 3B l18 is extracted

echo
echo "############################################################"
echo "# STEP 2/3  Fit one Procrustes translator per layer pair"
echo "############################################################"
fit() {  # $1=config $2=src_layer $3=tgt_layer $4=output_path
  if [[ -f "$4" ]]; then echo "--- skip (exists): $4"; return; fi
  echo "--- fit $4 ---"
  $RUN fit_procrustes.py --config "$1" --source-layer "$2" --target-layer "$3" --output "$4"
}

echo "== Task 1: 1B l8 -> 3B l{8,10,12,14,16,18} =="
for k in 8 10 12 14 16 18; do
  fit "$FWD_CFG" 8 "$k" \
    "$T1_DIR/best_translator__Llama-3.2-1B-Instruct_l8__Llama-3.2-3B-Instruct_l${k}__linear__procrustes.pt"
done

echo "== Task 2: 3B l14 -> 1B l{4,6,8,10,12} =="
for k in 4 6 8 10 12; do
  fit "$REV_CFG" 14 "$k" \
    "$T2_DIR/best_translator__Llama-3.2-3B-Instruct_l14__Llama-3.2-1B-Instruct_l${k}__linear__procrustes.pt"
done

echo
echo "############################################################"
echo "# STEP 3/3  Geometric comparison vs native SVs (per task)"
echo "############################################################"
$RUN compare_translated_and_original.py geometric \
  --translators "$T1_DIR/*.pt" --out "outputs/layer_sweep/task1_cmp" --device cuda:0
$RUN compare_translated_and_original.py geometric \
  --translators "$T2_DIR/*.pt" --out "outputs/layer_sweep/task2_cmp" --device cuda:0

echo
echo "DONE. Per-target-layer quality is in:"
echo "  outputs/layer_sweep/task1_cmp/geometric/per_translator.csv   (1B l8 -> 3B l?)"
echo "  outputs/layer_sweep/task2_cmp/geometric/per_translator.csv   (3B l14 -> 1B l?)"
