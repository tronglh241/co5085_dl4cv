#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
service="latex"
cid="$(docker compose ps -q "$service")"

if [ -z "$cid" ] || [ "$(docker inspect -f '{{.State.Running}}' "$cid")" != "true" ]; then
  docker compose up -d "$service"
fi

docker compose exec "$service" bash
