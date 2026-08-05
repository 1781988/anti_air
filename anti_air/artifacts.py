from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pid": os.getpid(),
    }


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def prepare_run_dir(path: str | Path) -> Path:
    run = Path(path).expanduser().resolve()
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True, exist_ok=True)
    return run


def create_submission(
    *,
    repository_root: str | Path,
    model_path: str | Path,
    result_path: str | Path,
    output_path: str | Path,
) -> Path:
    root = Path(repository_root).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    include = [
        "anti_air",
        "main.py",
        "config.yaml",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
        "run.sh",
        "setup.sh",
        "tests",
    ]
    with tempfile.TemporaryDirectory(prefix="anti_air_submission_") as directory:
        staging = Path(directory) / "anti_air_submission"
        staging.mkdir(parents=True)
        for relative in include:
            source = root / relative
            destination = staging / relative
            if source.is_dir():
                shutil.copytree(
                    source,
                    destination,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            elif source.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        shutil.copy2(model_path, staging / "model.pt")
        shutil.copy2(result_path, staging / "result.json")
        (staging / "SUBMISSION_README.txt").write_text(
            "Install: bash setup.sh\n"
            "Infer: bash run.sh <radar.mat> <infrared.mp4> [result.json]\n"
            "The model is fully offline after dependencies are installed.\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in staging.rglob("*"):
                if file.is_file():
                    archive.write(file, file.relative_to(staging.parent))
    return output
