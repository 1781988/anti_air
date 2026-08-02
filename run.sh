#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash run.sh <radar.mat> <infrared.mp4> [result.json]" >&2
  exit 2
fi

RADAR="$1"
INFRARED="$2"
OUTPUT="${3:-result.json}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${ANTI_AIR_MODEL:-$ROOT/model/model.joblib}"
if [[ ! -f "$MODEL" ]]; then
  MODEL="$ROOT/outputs/model/model.joblib"
fi
if [[ ! -f "$MODEL" ]]; then
  echo "Model not found. Set ANTI_AIR_MODEL or train/package the model first." >&2
  exit 3
fi
python "$ROOT/infer.py" --radar "$RADAR" --ir "$INFRARED" --model "$MODEL" --output "$OUTPUT"
