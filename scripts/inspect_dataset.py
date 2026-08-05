from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from anti_air.dataset import resolve_samples, write_manifest
from anti_air.infrared import video_metadata
from anti_air.radar import load_radar_frame, numeric_radar_frame
from anti_air.utils import json_dump


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect competition data and generate a normalized manifest"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root")
    source.add_argument("--manifest")
    parser.add_argument("--output-dir", default="outputs/inspection")
    parser.add_argument("--require-labels", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = resolve_samples(
        data_root=args.data_root,
        manifest=args.manifest,
        require_labels=args.require_labels,
        strict_pairs=True,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    inventory: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] inspect batch={sample.batch_id}")
        item: dict[str, Any] = sample.to_dict()
        try:
            raw = load_radar_frame(sample.radar_path)
            numeric = numeric_radar_frame(raw)
            item["radar"] = {
                "raw_shape": list(raw.shape),
                "raw_columns": [str(column) for column in raw.columns],
                "numeric_shape": list(numeric.shape),
                "numeric_columns": [str(column) for column in numeric.columns],
                "matlab_table_fallback": bool(
                    raw.attrs.get("matlab_table_fallback", False)
                ),
            }
        except Exception as exc:
            error = {
                "batch_id": sample.batch_id,
                "modality": "radar",
                "error": repr(exc),
            }
            item["radar_error"] = error["error"]
            errors.append(error)
            print(f"  ERROR radar batch={sample.batch_id}: {error['error']}")

        try:
            item["infrared"] = video_metadata(sample.infrared_path)
        except Exception as exc:
            error = {
                "batch_id": sample.batch_id,
                "modality": "infrared",
                "error": repr(exc),
            }
            item["infrared_error"] = error["error"]
            errors.append(error)
            print(f"  ERROR infrared batch={sample.batch_id}: {error['error']}")
        inventory.append(item)

    labels = Counter(sample.label for sample in samples if sample.label is not None)
    summary = {
        "sample_count": len(samples),
        "class_counts": dict(sorted(labels.items())),
        "error_count": len(errors),
        "errors": errors,
        "inventory": inventory,
    }
    json_dump(summary, output / "dataset_inventory.json")
    write_manifest(samples, output / "resolved_manifest.csv")
    print(
        f"Inspection complete: samples={len(samples)} "
        f"errors={len(errors)} output={output}"
    )
    if errors:
        print("Inspection failed. Detailed errors were written to:")
        print(f"  {output / 'dataset_inventory.json'}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
