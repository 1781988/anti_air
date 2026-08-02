#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: bash run_demo.sh <radar.mat> <infrared.mp4> <model.joblib>"
  exit 2
fi

python infer.py --radar "$1" --ir "$2" --model "$3" --output result.json
