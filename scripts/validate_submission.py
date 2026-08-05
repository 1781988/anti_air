from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


REQUIRED = {
    "anti_air_submission/run.sh",
    "anti_air_submission/infer.py",
    "anti_air_submission/model/model.joblib",
    "anti_air_submission/model/training_summary.json",
    "anti_air_submission/model/MODEL_CARD.md",
    "anti_air_submission/evaluation/metrics.json",
    "anti_air_submission/evaluation/folds.json",
    "anti_air_submission/evaluation/record_predictions.csv",
    "anti_air_submission/evaluation/confusion_matrix.csv",
    "anti_air_submission/report/test_report.md",
    "anti_air_submission/ARTIFACT_MANIFEST.json",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the generated competition ZIP")
    parser.add_argument("archive")
    args = parser.parse_args()
    archive = Path(args.archive).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)

    with zipfile.ZipFile(archive) as handle:
        names = set(handle.namelist())
        missing = sorted(REQUIRED - names)
        if missing:
            raise RuntimeError(f"Submission archive is missing required files: {missing}")
        manifest = json.loads(handle.read("anti_air_submission/ARTIFACT_MANIFEST.json"))
        for item in manifest.get("files", []):
            archive_name = f"anti_air_submission/{item['path']}"
            if archive_name not in names:
                raise RuntimeError(f"Manifest references a missing file: {archive_name}")
            payload = handle.read(archive_name)
            if len(payload) != int(item["size_bytes"]):
                raise RuntimeError(f"Size mismatch: {archive_name}")
            if _sha256(payload) != item["sha256"]:
                raise RuntimeError(f"Checksum mismatch: {archive_name}")
        metrics = json.loads(handle.read("anti_air_submission/evaluation/metrics.json"))
        summary = json.loads(handle.read("anti_air_submission/model/training_summary.json"))

    print(
        json.dumps(
            {
                "status": "ok",
                "archive": str(archive),
                "file_count": len(names),
                "evaluation_status": metrics.get("status"),
                "eligible_for_primary_score": metrics.get("eligible_for_primary_score"),
                "model_version": summary.get("model_version"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
