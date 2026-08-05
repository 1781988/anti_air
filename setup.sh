#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
ENV_DIR="${ENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11 is required. Install it first, or set PYTHON_BIN=/path/to/python3.11" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "FFmpeg is required for robust MP4 decoding." >&2
  echo "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ffmpeg" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m compileall -q anti_air main.py
python -m pytest

echo
echo "Environment ready."
echo "Activate: source $ENV_DIR/bin/activate"
echo "Run all: python main.py all"
