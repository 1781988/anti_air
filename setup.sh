#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python was not found. Install Python 3.11 or 3.12, or set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi
PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "$PYTHON_VERSION" in
  3.11|3.12) ;;
  *)
    echo "Python 3.11 or 3.12 is required; detected $PYTHON_VERSION." >&2
    exit 1
    ;;
esac
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
if [[ -d tests ]]; then
  python -m pytest
fi

echo
echo "Environment ready."
echo "Activate: source $ENV_DIR/bin/activate"
echo "Run all: python main.py all"
