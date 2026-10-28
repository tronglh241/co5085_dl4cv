#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

# Needs the prepared Food-101 copy (scripts/a1/prepare_data.py).
python scripts/a1/eda.py

# Needs the A1 checkpoints; skip until the runs have finished.
if compgen -G "checkpoints/a1/*/*/best_*.pt" > /dev/null; then
  python scripts/a1/analyze.py
else
  echo "no A1 checkpoints yet, skipping scripts/a1/analyze.py"
fi
