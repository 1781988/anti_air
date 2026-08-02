from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .alignment import AlignmentResult, estimate_alignment
from .dataset import Sample
from .infrared import extract_infrared_features
from .radar import extract_radar_features


@dataclass
class FeatureBundle:
    radar: dict[str, float]
    infrared: dict[str, float]
    alignment: AlignmentResult
    quality: dict[str, float]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def extract_sample(sample: Sample, config: dict[str, Any]) -> FeatureBundle:
    radar_cfg = config.get("radar", {})
    ir_cfg = config.get("infrared", {})
    align_cfg = config.get("alignment", {})

    radar_features, radar_activity = extract_radar_features(
        sample.radar_path,
        max_numeric_columns=int(radar_cfg.get("max_numeric_columns", 64)),
    )
    ir_features, ir_activity, ir_quality = extract_infrared_features(
        sample.infrared_path,
        sample_fps=float(ir_cfg.get("sample_fps", 3.0)),
        resize_width=int(ir_cfg.get("resize_width", 640)),
        max_samples=int(ir_cfg.get("max_samples", 2000)),
        motion_percentile=float(ir_cfg.get("motion_percentile", 99.5)),
        min_blob_area=int(ir_cfg.get("min_blob_area", 2)),
        max_blob_area_ratio=float(ir_cfg.get("max_blob_area_ratio", 0.002)),
    )

    duration = max(float(ir_features.get("meta__duration_seconds", 0.0)), 1e-6)
    radar_rate = len(radar_activity) / duration
    ir_rate = float(ir_cfg.get("sample_fps", 3.0))
    if bool(align_cfg.get("enabled", True)):
        alignment = estimate_alignment(
            radar_activity,
            ir_activity,
            radar_rate_hz=radar_rate,
            infrared_rate_hz=ir_rate,
            common_rate_hz=float(align_cfg.get("common_rate_hz", 5.0)),
            max_lag_seconds=float(align_cfg.get("max_lag_seconds", 30.0)),
        )
    else:
        alignment = AlignmentResult(0.0, 0.0, float(align_cfg.get("common_rate_hz", 5.0)))

    radar_features["sync__offset_seconds"] = alignment.offset_seconds
    radar_features["sync__score"] = alignment.score
    ir_features["sync__offset_seconds"] = alignment.offset_seconds
    ir_features["sync__score"] = alignment.score

    quality = {
        "radar": float(radar_features.get("quality__valid_ratio", 0.0)),
        "infrared": float(ir_quality.get("tracking_rate", 0.0)),
    }
    return FeatureBundle(radar=radar_features, infrared=ir_features, alignment=alignment, quality=quality)
