#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

targets=(checkpoints logs)
manifest_dir=checksums

find_artifacts() {
    case "$1" in
        checkpoints) find checkpoints -type f \( -name '*.pt' -o -name '*.yml' \) -print0 ;;
        logs) find logs -type f -name 'events.out.tfevents.*' -print0 ;;
    esac
}

generate() {
    mkdir -p "$manifest_dir"
    for target in "${targets[@]}"; do
        if [ ! -d "$target" ]; then
            echo "missing $target/" >&2
            exit 1
        fi
        find_artifacts "$target" \
            | LC_ALL=C sort -z \
            | xargs -0 -r md5sum > "$manifest_dir/$target.md5"
        echo "wrote $manifest_dir/$target.md5"
    done
}

verify() {
    for target in "${targets[@]}"; do
        manifest="$manifest_dir/$target.md5"
        if [ ! -f "$manifest" ]; then
            echo "missing $manifest" >&2
            exit 1
        fi
        md5sum --quiet -c "$manifest"
    done
    echo "all artifacts verified"
}

case "${1:-verify}" in
    generate) generate ;;
    verify) verify ;;
    *) echo "usage: $0 [generate|verify]" >&2; exit 2 ;;
esac
