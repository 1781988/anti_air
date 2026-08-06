#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run.sh <radar.mat> <infrared.mp4> [result.json]" >&2
  exit 2
fi

ENV_NAME="${ANTI_AIR_ENV:-anti-air}"
MODEL_PATH="${ANTI_AIR_MODEL:-model.pt}"
OUTPUT_PATH="${3:-result.json}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
  PYTHON_CMD=(python)
elif command -v conda >/dev/null 2>&1 && conda run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
  PYTHON_CMD=(conda run --no-capture-output -n "$ENV_NAME" python)
else
  echo "Conda environment '$ENV_NAME' is not ready. Run: bash setup.sh" >&2
  exit 1
fi

"${PYTHON_CMD[@]}" main.py infer \
  --radar "$1" \
  --ir "$2" \
  --model "$MODEL_PATH" \
  --output "$OUTPUT_PATH"
