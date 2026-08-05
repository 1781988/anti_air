#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run.sh <radar.mat> <infrared.mp4> [result.json]" >&2
  exit 2
fi

MODEL_PATH="${ANTI_AIR_MODEL:-model.pt}"
OUTPUT_PATH="${3:-result.json}"
python main.py infer \
  --radar "$1" \
  --ir "$2" \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_PATH"
