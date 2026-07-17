#!/usr/bin/env bash
# Minimal end-to-end run with the default config: extract activations, then train.
#
# Usage (run from anywhere; the script cd's to the repo root itself):
#   scripts/run_prepare_and_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

PY="conda run -n acteng python"
$PY prepare_activations.py --config config/default.toml
$PY train.py --config config/default.toml
