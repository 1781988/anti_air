#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ANTI_AIR_ENV:-anti-air}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PROFILE="${1:-auto}"
case "$PROFILE" in
  auto|quick|cpu|competition) ;;
  *)
    echo "Usage: bash train.sh [auto|quick|cpu|competition] [--rebuild-cache]" >&2
    exit 2
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda was not found. Run bash setup.sh after installing Miniconda." >&2
  exit 1
fi
if ! conda run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
  echo "Conda environment '$ENV_NAME' does not exist. Run: bash setup.sh" >&2
  exit 1
fi

if ! find data/train -maxdepth 1 -type f -name '*.mat' -print -quit | grep -q .; then
  echo "No radar MAT files found in data/train/. Copy the competition train folder there first." >&2
  exit 1
fi
if ! find data/train -maxdepth 1 -type f -name '*.mp4' -print -quit | grep -q .; then
  echo "No infrared MP4 files found in data/train/. Copy the competition train folder there first." >&2
  exit 1
fi

ARGS=(main.py all --profile "$PROFILE")
if [[ "${2:-}" == "--rebuild-cache" || ! -d .cache/anti_air ]]; then
  ARGS+=(--rebuild-cache)
fi

exec conda run --no-capture-output -n "$ENV_NAME" python "${ARGS[@]}"
