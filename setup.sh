#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ANTI_AIR_ENV:-anti-air}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Conda was not found.
Install Miniconda first, reopen the shell, and rerun: bash setup.sh
Official installer: https://docs.conda.io/projects/miniconda/en/latest/
EOF
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
  echo "Updating existing Conda environment: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f environment.yml --prune
else
  echo "Creating Conda environment: $ENV_NAME"
  conda env create -n "$ENV_NAME" -f environment.yml
fi

conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip install --upgrade pip setuptools wheel

if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
  conda run --no-capture-output -n "$ENV_NAME" \
    python -m pip install --upgrade torch torchvision --index-url "$TORCH_INDEX_URL"
fi

conda run --no-capture-output -n "$ENV_NAME" \
  python -m pip install -e ".[dev]"
conda run --no-capture-output -n "$ENV_NAME" \
  python -m compileall -q anti_air main.py
conda run --no-capture-output -n "$ENV_NAME" \
  python -m pytest

conda run --no-capture-output -n "$ENV_NAME" \
  python -c 'import shutil,sys,torch; print("\nEnvironment verification"); print("Python:",sys.version.split()[0]); print("PyTorch:",torch.__version__); print("CUDA available:",torch.cuda.is_available()); print("GPU:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print("FFmpeg:",shutil.which("ffmpeg"))'

cat <<EOF

Environment ready: $ENV_NAME
No .venv is used.
No activation is required for the wrapper scripts.

Next steps:
  1. Copy the competition train folder to: $ROOT_DIR/data/train
  2. Quick check:    bash train.sh quick
  3. Full pipeline:  bash train.sh

Optional manual activation:
  conda activate $ENV_NAME
EOF
