#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."
cd docs && bundle exec jekyll build
