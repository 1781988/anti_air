from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


INCLUDE = [
    "anti_air",
    "configs",
    "docs",
    "infer.py",
    "requirements.txt",
    "pyproject.toml",
    "run.sh",
    "README.md",
]


def _copy(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the final offline competition submission package")
    parser.add_argument("--model", default="outputs/model/model.joblib")
    parser.add_argument("--report", default="outputs/report/test_report.md")
    parser.add_argument("--output", default="outputs/submission/anti_air_submission.zip")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    model = Path(args.model).resolve()
    report = Path(args.report).resolve()
    if not model.is_file():
        raise FileNotFoundError(f"Trained model not found: {model}")
    output = Path(args.output).resolve()
    staging = output.parent / "anti_air_submission"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for relative in INCLUDE:
        source = root / relative
        if source.exists():
            _copy(source, staging / relative)
    _copy(model, staging / "model" / "model.joblib")
    if report.is_file():
        _copy(report, staging / "report" / "test_report.md")
    (staging / "SUBMISSION_README.txt").write_text(
        "Run: bash run.sh <radar.mat> <infrared.mp4> [result.json]\n"
        "The package is offline and does not read class labels from input filenames.\n",
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in staging.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(staging.parent))
    print(f"Submission package written to {output}")


if __name__ == "__main__":
    main()
