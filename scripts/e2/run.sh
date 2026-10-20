#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/e2/check_msa.py
python scripts/e2/analyze.py --configs configs/e2/test_*.yml
for config in configs/e2/test_*.yml; do
  python scripts/e2/attention_maps.py --config "$config"
done
