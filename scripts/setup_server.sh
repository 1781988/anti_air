#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENV_DIR="${ENV_DIR:-.venv}"

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
python -m pytest -q

echo "Environment ready. Activate with: source $ENV_DIR/bin/activate"
