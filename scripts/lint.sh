#!/usr/bin/env bash
# lint.sh [dirs...] — style gate, RUN INSIDE the container shell (see shell/dev.sh).
# isort check + flake8 over our code only.

set -euo pipefail
cd "$(dirname "$0")/.."

paths=()

if [ "$#" -eq 0 ]; then
  for p in common datasets models scripts; do
    [ -d "$p" ] && paths+=("$p")
  done
else
  for p in "$@"; do
    if [ -d "$p" ]; then
      paths+=("$p")
    else
      echo "lint.sh: skip missing '$p'" >&2
    fi
  done
fi

if [ "${#paths[@]}" -eq 0 ]; then
  echo "lint.sh: nothing to lint" >&2
  exit 0
fi

python3 -m isort "${paths[@]}"
python3 -m flake8 "${paths[@]}"
