from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path


INCLUDE = [
    "anti_air",
    "configs",
    "docs",
    "scripts",
    "infer.py",
    "train.py",
    "evaluate.py",
    "extract_features.py",
    "requirements.txt",
    "pyproject.toml",
    "run.sh",
    "README.md",
    "Dockerfile",
    "Makefile",
]


def _copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_required(source: Path, destination: Path, description: str) -> None:
    if not source.exists():
        raise FileNotFoundError(f"{description} not found: {source}")
    _copy(source, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(staging: Path) -> None:
    files = []
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "ARTIFACT_MANIFEST.json":
            files.append(
                {
                    "path": str(path.relative_to(staging)),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    (staging / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps({"file_count": len(files), "files": files}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the final offline competition submission package")
    parser.add_argument("--model", default="outputs/model/model.joblib")
    parser.add_argument("--training-summary", default="outputs/model/training_summary.json")
    parser.add_argument("--model-card", default="outputs/model/MODEL_CARD.md")
    parser.add_argument("--evaluation-dir", default="outputs/evaluation")
    parser.add_argument("--inspection-dir", default="outputs/inspection")
    parser.add_argument("--report", default="outputs/report/test_report.md")
    parser.add_argument("--output", default="outputs/submission/anti_air_submission.zip")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model = Path(args.model).resolve()
    training_summary = Path(args.training_summary).resolve()
    model_card = Path(args.model_card).resolve()
    evaluation_dir = Path(args.evaluation_dir).resolve()
    inspection_dir = Path(args.inspection_dir).resolve()
    report = Path(args.report).resolve()
    output = Path(args.output).resolve()
    staging = output.parent / "anti_air_submission"

    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for relative in INCLUDE:
        source = root / relative
        _copy_required(source, staging / relative, "Submission source")

    _copy_required(model, staging / "model" / "model.joblib", "Trained model")
    _copy_required(training_summary, staging / "model" / "training_summary.json", "Training summary")
    _copy_required(model_card, staging / "model" / "MODEL_CARD.md", "Model card")
    _copy_required(evaluation_dir, staging / "evaluation", "Evaluation artifacts")
    if inspection_dir.is_dir():
        _copy(inspection_dir, staging / "inspection")
    _copy_required(report, staging / "report" / "test_report.md", "Test report")

    (staging / "SUBMISSION_README.txt").write_text(
        "Install: python -m pip install -e .\n"
        "Run: bash run.sh <radar.mat> <infrared.mp4> [result.json]\n"
        "Performance validity: read report/test_report.md and evaluation/metrics.json.\n"
        "The package is offline and does not read class labels from input filenames.\n",
        encoding="utf-8",
    )
    _write_manifest(staging)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in staging.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(staging.parent))
    print(f"Submission package written to {output}")


if __name__ == "__main__":
    main()
