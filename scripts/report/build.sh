#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

texfiles=(
  reports/exercises.tex
  reports/assignment_1.tex
  reports/assignment_2.tex
)

if [ "$#" -gt 0 ]; then
  texfiles=("$@")
fi

for texfile in "${texfiles[@]}"; do
  if [ ! -f "$texfile" ]; then
    echo "skip missing $texfile" >&2
    continue
  fi
  latexindent -y='defaultIndent: "    "' -s -wd "$texfile" && rm -f "${texfile%.*}".bak*
  latexmk -lualatex -shell-escape -silent -interaction=batchmode -cd "$texfile"
done
