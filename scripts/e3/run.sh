#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/e3/analyze.py --configs configs/e3/test_*.yml
