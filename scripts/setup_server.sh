#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-.venv}"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  cat <<'EOF'
WARNING: system FFmpeg/FFprobe was not found.
OpenCV will be tried first, but MP4 fallback decoding requires FFmpeg.
Ubuntu/Debian installation command:
  sudo apt-get update && sudo apt-get install -y ffmpeg
EOF
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
python -m compileall -q .
python -m pytest -q

echo "Environment ready. Activate with: source $ENV_DIR/bin/activate"
echo "Video diagnostic example:"
echo "  python scripts/check_video.py --video 'data/train/ir_339_class-B_16：18.mp4'"
