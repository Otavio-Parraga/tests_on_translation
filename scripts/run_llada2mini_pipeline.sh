#!/usr/bin/env bash
# Chains the full LLaDA2.0-mini study: fit -> A/B sweep -> report, each gated on
# the previous step succeeding. Mirrors the LOBO study's chained-launcher
# pattern (build_caa_diff_manifest.py -> extract -> fit -> eval -> report) so
# a single detached launch is all that's needed.
#
# PREREQUISITE (already satisfied as of 2026-08-12): LLaDA2.0-mini's FineWeb
# activation caches must exist and be verified — see
# scripts/run_llada_mini_after_sweeps.sh, which re-extracted and verified all 10
# caches (5 layers x mean/mask) after fixing the uninitialized-rotary-table bug.
# This pipeline does NOT re-extract; it only fits translators on what's already
# on disk and refuses to start (via run_fit_llada2mini.sh's own check) if any
# cache is missing.
#
# LAUNCH DETACHED (survives closing the terminal):
#   cd /mnt/C-SSD/parraga/tests_on_translation
#   setsid nohup ./scripts/run_llada2mini_pipeline.sh > /dev/null 2>&1 &
#   tail -f "$(ls -t outputs/llada2mini_pipeline_*.log | head -1)"
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root (scripts/ lives one level down)

LOG="${LOG:-outputs/llada2mini_pipeline_$(date '+%Y%m%d_%H%M%S').log}"
mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

echo "=== [1/3] LLaDA2.0-mini Procrustes fits  $(date '+%Y-%m-%d %H:%M:%S') ==="
./scripts/run_fit_llada2mini.sh
rc=$?
echo "=== fits exit=$rc  $(date '+%Y-%m-%d %H:%M:%S') ==="
if [[ $rc -ne 0 ]]; then echo "!! fits failed, aborting chain" >&2; exit $rc; fi

echo "=== [2/3] LLaDA2.0-mini A/B sweep  $(date '+%Y-%m-%d %H:%M:%S') ==="
./scripts/run_ab_llada2mini.sh
rc=$?
echo "=== eval exit=$rc  $(date '+%Y-%m-%d %H:%M:%S') ==="
if [[ $rc -ne 0 ]]; then echo "!! eval failed, aborting chain before the report" >&2; exit $rc; fi

echo "=== [3/3] building report + dashboards  $(date '+%Y-%m-%d %H:%M:%S') ==="
source scripts/llada_env.sh
$RUN ab_report.py --results 'outputs/ab_eval/llada2mini/*.jsonl' --out-dir outputs/ab_eval/llada2mini
$RUN ab_pivot_dashboard.py

echo "=== PIPELINE DONE  $(date '+%Y-%m-%d %H:%M:%S') ==="
