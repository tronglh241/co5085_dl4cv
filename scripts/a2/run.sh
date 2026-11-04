#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ ! -d data/celeba_112 ]; then
  echo "data/celeba_112 is missing; run scripts/a2/prepare_data.py first"
  exit 1
fi

python scripts/a2/eda.py

for config in configs/a2/train_*.yml; do
  name="$(basename "$config" .yml)"
  if compgen -G "checkpoints/a2/$name/*/best_*.pt" > /dev/null; then
    python scripts/a2/evaluate.py --config "$config"
  else
    echo "no checkpoint for $name, skipping"
  fi
done

python scripts/a2/analyze.py
