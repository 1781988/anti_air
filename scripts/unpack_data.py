from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract .7z or .zip competition data")
    parser.add_argument("archive")
    parser.add_argument("--output", default="data/train")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    archive = Path(args.archive).resolve()
    output = Path(args.output).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists() and args.force:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    suffix = archive.suffix.lower()
    if suffix == ".7z":
        try:
            import py7zr
        except ImportError as exc:
            raise RuntimeError("Install dependencies first: pip install -r requirements.txt") from exc
        with py7zr.SevenZipFile(archive, mode="r") as handle:
            handle.extractall(output)
    elif suffix == ".zip":
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(output)
    else:
        raise ValueError(f"Unsupported archive format: {archive.suffix}")
    print(f"Extracted {archive} to {output}")


if __name__ == "__main__":
    main()
