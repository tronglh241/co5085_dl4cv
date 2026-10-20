#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

checkpoints_folder_url="https://drive.google.com/drive/folders/1wJBqLOWrRHm_ihNl6K5uDO_mrIULgJmH"
logs_folder_url="https://drive.google.com/drive/folders/1fldkQaFTqDzF-L6q1on37cckQQUPMNhF"

gdown -c "$checkpoints_folder_url"
gdown -c "$logs_folder_url"
