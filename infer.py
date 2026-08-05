from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from anti_air.dataset import Sample, resolve_samples
from anti_air.feature_store import build_feature_cache, tables_from_extractions
from anti_air.modeling import ModelBundle, aggregate_record_probabilities, predict_window_probabilities
from anti_air.pipeline import extract_record
from anti_air.utils import json_dump


def _record_to_result(record: pd.Series, classes: list[str], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "batch_id": str(record["batch_id"]),
        "label": str(record["predicted_label"]),
        "confidence": float(record["confidence"]),
        "class_probabilities": {label: float(record[f"prob__{label}"]) for label in classes},
        "window_count": int(record["window_count"]),
    }
    if manifest and manifest.get("records"):
        item = manifest["records"][0]
        infrared_metadata = item.get("infrared_metadata", {})
        result["alignment"] = item.get("alignment")
        result["quality"] = {
            "radar": item.get("radar_metadata", {}).get("valid_ratio"),
            "infrared": infrared_metadata.get("quality"),
        }
        result["video_decoder"] = {
            "backend": infrared_metadata.get("decoder_backend"),
            "codec": infrared_metadata.get("decoder_codec"),
        }
    return result


def predict(
    radar_path: str,
    infrared_path: str,
    model_path: str,
    *,
    batch_id: str = "inference",
) -> dict[str, Any]:
    model = Path(model_path).expanduser().resolve()
    radar = Path(radar_path).expanduser().resolve()
    infrared = Path(infrared_path).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"Model file not found: {model}")
    if not radar.is_file():
        raise FileNotFoundError(f"Radar file not found: {radar}")
    if not infrared.is_file():
        raise FileNotFoundError(f"Infrared video not found: {infrared}")
    bundle: ModelBundle = joblib.load(model)
    sample = Sample(
        batch_id=batch_id,
        radar_path=radar,
        infrared_path=infrared,
        label=None,
    )
    extraction = extract_record(sample, bundle.config)
    tables = tables_from_extractions([extraction], bundle.config)
    windows = predict_window_probabilities(bundle, tables)
    records = aggregate_record_probabilities(windows, bundle.classes)
    result = _record_to_result(records.iloc[0], bundle.classes, tables.manifest)
    result["window_predictions"] = [
        {
            "window_id": str(row["window_id"]),
            "ir_start_seconds": float(row["ir_start_seconds"]),
            "ir_end_seconds": float(row["ir_end_seconds"]),
            "predicted_label": str(row["predicted_label"]),
            "confidence": float(row["window_confidence"]),
        }
        for _, row in windows.iterrows()
    ]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-pair or directory inference")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--radar", help="Single radar MAT file; also provide --ir")
    mode.add_argument("--data-root", help="Directory containing paired test files")
    mode.add_argument("--manifest", help="CSV manifest containing test pairs")
    parser.add_argument("--ir", help="Single infrared MP4 file")
    parser.add_argument("--batch-id", default="inference")
    parser.add_argument("--model", default="outputs/model/model.joblib")
    parser.add_argument("--output", default="outputs/predictions/result.json")
    parser.add_argument("--windows-output", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.radar:
        if not args.ir:
            raise ValueError("--ir is required when --radar is used")
        result = predict(args.radar, args.ir, args.model, batch_id=args.batch_id)
        json_dump(result, output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"Model file not found: {model}")
    bundle: ModelBundle = joblib.load(model)
    samples = resolve_samples(
        data_root=args.data_root,
        manifest=args.manifest,
        require_labels=False,
        strict_pairs=True,
    )
    with tempfile.TemporaryDirectory(prefix="anti_air_infer_") as temporary:
        tables = build_feature_cache(samples, bundle.config, temporary, force=True)
        window_predictions = predict_window_probabilities(bundle, tables)
        record_predictions = aggregate_record_probabilities(window_predictions, bundle.classes)
    results = [
        _record_to_result(row, bundle.classes)
        for _, row in record_predictions.iterrows()
    ]
    json_dump({"model_version": bundle.version, "predictions": results}, output)
    csv_path = Path(args.windows_output) if args.windows_output else output.with_suffix(".windows.csv")
    window_predictions.to_csv(csv_path, index=False)
    record_predictions.to_csv(output.with_suffix(".records.csv"), index=False)
    print(f"Predicted {len(results)} records; output={output}")


if __name__ == "__main__":
    main()
