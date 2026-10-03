#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
tensorboard --logdir "${1:-logs}" --host 0.0.0.0 --port 6006
