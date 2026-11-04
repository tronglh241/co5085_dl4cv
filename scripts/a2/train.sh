#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

if [ ! -d data/celeba_112 ]; then
  echo "data/celeba_112 is missing; run scripts/a2/prepare_data.py first"
  exit 1
fi

for config in configs/a2/train_*.yml; do
  echo "training $config"
  python scripts/train.py "$config"
done
