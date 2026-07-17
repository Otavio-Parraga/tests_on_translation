#!/usr/bin/env bash
# Run the A/B comparison between the original Llama steering vector and the
# translated Gemma steering vector for a given behavior.
#
# Usage:
#   scripts/ab_comparison.sh                        # full run, sycophancy
#   scripts/ab_comparison.sh --limit 30             # quick test (~1 min)
#   scripts/ab_comparison.sh --behavior refusal     # different behavior
#   scripts/ab_comparison.sh --coefficients -20 0 20
#
# The translated vector must exist in this repo's steering_vectors/ dir.
# If it doesn't, run:
#   scripts/translate_test.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda run -n acteng python "${REPO_ROOT}/ab_comparison.py" "$@"
