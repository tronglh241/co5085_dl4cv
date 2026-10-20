#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

python scripts/e1/analyze.py
