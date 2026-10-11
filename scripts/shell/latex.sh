#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
service="latex"
container_id="$(docker compose ps -q -a "$service")"

if [ -z "$container_id" ]; then
  docker compose up -d "$service"
elif [ "$(docker inspect -f '{{.State.Running}}' "$container_id")" != "true" ]; then
  docker compose start "$service"
fi

docker compose exec "$service" bash
