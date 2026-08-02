from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .dataset import Sample
from .pipeline import RecordExtraction, extract_record
from .utils import json_dump, stable_hash


META_COLUMNS = [
    "window_id",
    "batch_id",
    "label",
    "ir_start_seconds",
    "ir_end_seconds",
    "radar_start_seconds",
    "radar_end_seconds",
    "radar_quality",
    "infrared_quality",
]


@dataclass
class FeatureTables:
    radar: pd.DataFrame
    infrared: pd.DataFrame
    fusion: pd.DataFrame
    manifest: dict[str, Any]


def _sample_fingerprint(sample: Sample, config: dict[str, Any]) -> str:
    payload = {
        "batch_id": sample.batch_id,
        "radar": [str(sample.radar_path), sample.radar_path.stat().st_size, sample.radar_path.stat().st_mtime_ns],
        "infrared": [
            str(sample.infrared_path),
            sample.infrared_path.stat().st_size,
            sample.infrared_path.stat().st_mtime_ns,
        ],
        "config": config,
    }
    return stable_hash(payload)


def _row(window: Any, features: dict[str, float]) -> dict[str, Any]:
    return {
        "window_id": window.window_id,
        "batch_id": window.batch_id,
        "label": window.label,
        "ir_start_seconds": window.ir_start_seconds,
        "ir_end_seconds": window.ir_end_seconds,
        "radar_start_seconds": window.radar_start_seconds,
        "radar_end_seconds": window.radar_end_seconds,
        "radar_quality": window.quality["radar"],
        "infrared_quality": window.quality["infrared"],
        **features,
    }


def tables_from_extractions(extractions: list[RecordExtraction], config: dict[str, Any]) -> FeatureTables:
    radar_rows: list[dict[str, Any]] = []
    infrared_rows: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    for extraction in extractions:
        for window in extraction.windows:
            radar_rows.append(_row(window, window.radar))
            infrared_rows.append(_row(window, window.infrared))
            fusion_rows.append(_row(window, window.fusion))
    return FeatureTables(
        radar=pd.DataFrame(radar_rows),
        infrared=pd.DataFrame(infrared_rows),
        fusion=pd.DataFrame(fusion_rows),
        manifest={
            "cache_version": config.get("runtime", {}).get("cache_version", 1),
            "config_hash": stable_hash(config),
            "sample_count": len(extractions),
            "window_count": len(fusion_rows),
            "classes": sorted(
                {str(extraction.sample.label) for extraction in extractions if extraction.sample.label is not None}
            ),
            "records": [extraction.manifest() for extraction in extractions],
        },
    )


def build_feature_cache(
    samples: list[Sample],
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    force: bool = False,
) -> FeatureTables:
    output = Path(output_dir)
    records_dir = output / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    extractions: list[RecordExtraction] = []

    for index, sample in enumerate(samples, start=1):
        fingerprint = _sample_fingerprint(sample, config)
        cache_path = records_dir / f"{sample.batch_id}_{fingerprint}.joblib"
        stale = list(records_dir.glob(f"{sample.batch_id}_*.joblib"))
        if cache_path.exists() and not force:
            print(f"[{index}/{len(samples)}] cache hit batch={sample.batch_id}")
            extraction = joblib.load(cache_path)
        else:
            print(f"[{index}/{len(samples)}] extracting batch={sample.batch_id} label={sample.label}")
            extraction = extract_record(sample, config)
            joblib.dump(extraction, cache_path, compress=3)
            for old in stale:
                if old != cache_path:
                    old.unlink(missing_ok=True)
        extractions.append(extraction)

    tables = tables_from_extractions(extractions, config)
    output.mkdir(parents=True, exist_ok=True)
    tables.radar.to_csv(output / "radar_features.csv", index=False)
    tables.infrared.to_csv(output / "infrared_features.csv", index=False)
    tables.fusion.to_csv(output / "fusion_features.csv", index=False)
    json_dump(tables.manifest, output / "manifest.json")
    return tables


def load_feature_cache(path: str | Path, *, require_labels: bool = True) -> FeatureTables:
    root = Path(path)
    radar_path = root / "radar_features.csv"
    infrared_path = root / "infrared_features.csv"
    fusion_path = root / "fusion_features.csv"
    manifest_path = root / "manifest.json"
    for required in (radar_path, infrared_path, fusion_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"Feature cache file not found: {required}")
    radar = pd.read_csv(radar_path)
    infrared = pd.read_csv(infrared_path)
    fusion = pd.read_csv(fusion_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_labels:
        for name, frame in (("radar", radar), ("infrared", infrared), ("fusion", fusion)):
            if "label" not in frame or frame["label"].isna().any():
                raise ValueError(f"{name} cache contains missing labels")
    window_ids = set(radar["window_id"])
    if window_ids != set(infrared["window_id"]) or window_ids != set(fusion["window_id"]):
        raise ValueError("Feature cache branches have inconsistent window IDs")
    return FeatureTables(radar=radar, infrared=infrared, fusion=fusion, manifest=manifest)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(column for column in frame.columns if column not in META_COLUMNS)
