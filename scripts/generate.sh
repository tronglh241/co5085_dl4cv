#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

step="${1:-all}"

common() {
  python scripts/common/eda.py
}

case "$step" in
  common) common ;;
  e1) bash scripts/e1/run.sh ;;
  e2) bash scripts/e2/run.sh ;;
  a1) bash scripts/a1/run.sh ;;
  e3) echo "E3 assets deferred" ;;
  all) common; bash scripts/e1/run.sh; bash scripts/e2/run.sh; bash scripts/a1/run.sh ;;
  *) echo "usage: $0 [common|e1|e2|a1|all|e3]" >&2; exit 2 ;;
esac
