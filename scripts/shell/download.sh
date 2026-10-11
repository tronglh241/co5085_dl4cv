#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

folder_id="${1:-1g8PldLUgOxWfs4BraPLj8DtoGJ04GWZf}"

if ! command -v gdown >/dev/null 2>&1; then
  echo 'gdown not found, install it with: pip install -r requirements.txt' >&2
  exit 1
fi

gdown --folder "$folder_id"

mv artifacts/checkpoints checkpoints
mv artifacts/logs logs
rm -r artifacts
