from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .alignment import AlignmentResult, estimate_alignment
from .dataset import Sample
from .infrared import InfraredSequence, extract_infrared_sequence, infrared_window_features, video_metadata
from .radar import RadarSequence, extract_radar_sequence, radar_window_features


@dataclass
class WindowFeature:
    window_id: str
    batch_id: str
    label: str | None
    ir_start_seconds: float
    ir_end_seconds: float
    radar_start_seconds: float
    radar_end_seconds: float
    radar: dict[str, float]
    infrared: dict[str, float]
    fusion: dict[str, float]
    quality: dict[str, float]


@dataclass
class RecordExtraction:
    sample: Sample
    alignment: AlignmentResult
    radar_metadata: dict[str, Any]
    infrared_metadata: dict[str, Any]
    windows: list[WindowFeature]

    def manifest(self) -> dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "alignment": self.alignment.to_dict(),
            "radar_metadata": self.radar_metadata,
            "infrared_metadata": self.infrared_metadata,
            "window_count": len(self.windows),
            "windows": [
                {
                    key: value
                    for key, value in asdict(window).items()
                    if key not in {"radar", "infrared", "fusion"}
                }
                for window in self.windows
            ],
        }


def _prefix(features: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}__{key}": value for key, value in features.items()}


def _build_windows(start: float, end: float, length: float, stride: float, maximum: int) -> list[tuple[float, float]]:
    if end <= start:
        return [(start, max(start + 1e-3, end))]
    if end - start <= length:
        return [(start, end)]
    windows: list[tuple[float, float]] = []
    current = start
    while current + length <= end + 1e-9 and len(windows) < maximum:
        windows.append((current, min(current + length, end)))
        current += stride
    if windows and windows[-1][1] < end - max(1e-6, 0.25 * length) and len(windows) < maximum:
        windows.append((max(start, end - length), end))
    return windows


def _window_activity_correlation(
    radar: RadarSequence,
    infrared: InfraredSequence,
    alignment: AlignmentResult,
    ir_start: float,
    ir_end: float,
) -> float:
    radar_start = alignment.infrared_to_radar(ir_start)
    radar_end = alignment.infrared_to_radar(ir_end)
    r_mask = (radar.times >= radar_start) & (radar.times < radar_end)
    i_mask = (infrared.times >= ir_start) & (infrared.times < ir_end)
    r = radar.activity[r_mask]
    i = infrared.activity[i_mask]
    if len(r) < 3 or len(i) < 3:
        return math.nan
    count = min(100, max(3, min(len(r), len(i))))
    common = np.linspace(0.0, 1.0, count)
    r_resampled = np.interp(common, np.linspace(0.0, 1.0, len(r)), r)
    i_resampled = np.interp(common, np.linspace(0.0, 1.0, len(i)), i)
    if np.std(r_resampled) < 1e-8 or np.std(i_resampled) < 1e-8:
        return 0.0
    return float(np.corrcoef(r_resampled, i_resampled)[0, 1])


def extract_record(sample: Sample, config: dict[str, Any]) -> RecordExtraction:
    ir_config = config["infrared"]
    radar_config = config["radar"]
    alignment_config = config["alignment"]
    window_config = config["window"]

    metadata = video_metadata(sample.infrared_path)
    radar = extract_radar_sequence(
        sample.radar_path,
        duration_hint=float(metadata["duration_seconds"]) or None,
        max_numeric_columns=int(radar_config["max_numeric_columns"]),
        max_vector_expansion=int(radar_config["max_vector_expansion"]),
    )
    infrared = extract_infrared_sequence(
        sample.infrared_path,
        sample_fps=float(ir_config["sample_fps"]),
        resize_width=int(ir_config["resize_width"]),
        max_samples=int(ir_config["max_samples"]),
        clahe_clip_limit=float(ir_config["clahe_clip_limit"]),
        motion_percentile=float(ir_config["motion_percentile"]),
        contrast_percentile=float(ir_config["contrast_percentile"]),
        min_blob_area=int(ir_config["min_blob_area"]),
        max_blob_area_ratio=float(ir_config["max_blob_area_ratio"]),
        max_track_jump_ratio=float(ir_config["max_track_jump_ratio"]),
        max_missed_frames=int(ir_config["max_missed_frames"]),
    )

    if bool(alignment_config["enabled"]):
        alignment = estimate_alignment(
            radar.activity,
            infrared.activity,
            radar_rate_hz=radar.rate_hz,
            infrared_rate_hz=infrared.sample_rate_hz,
            common_rate_hz=float(alignment_config["common_rate_hz"]),
            max_lag_seconds=float(alignment_config["max_lag_seconds"]),
            estimate_drift=bool(alignment_config["estimate_drift"]),
            drift_segments=int(alignment_config["drift_segments"]),
            max_drift_ppm=float(alignment_config["max_drift_ppm"]),
        )
    else:
        alignment = AlignmentResult(0.0, 1.0, 0.0, 0.0, float(alignment_config["common_rate_hz"]))

    ir_overlap_start = max(0.0, alignment.radar_to_infrared(0.0))
    ir_overlap_end = min(infrared.duration_seconds, alignment.radar_to_infrared(radar.duration_seconds))
    if ir_overlap_end - ir_overlap_start < 1.0:
        ir_overlap_start = 0.0
        ir_overlap_end = min(infrared.duration_seconds, radar.duration_seconds)

    ranges = _build_windows(
        ir_overlap_start,
        ir_overlap_end,
        float(window_config["length_seconds"]),
        float(window_config["stride_seconds"]),
        int(window_config["max_windows_per_record"]),
    )
    minimum_coverage = float(window_config["min_coverage_ratio"])
    windows: list[WindowFeature] = []
    for index, (ir_start, ir_end) in enumerate(ranges):
        radar_start = max(0.0, alignment.infrared_to_radar(ir_start))
        radar_end = min(radar.duration_seconds, alignment.infrared_to_radar(ir_end))
        if radar_end <= radar_start:
            continue
        radar_features, radar_coverage = radar_window_features(
            radar,
            radar_start,
            radar_end,
            spectral_bins=int(radar_config["spectral_bins"]),
        )
        infrared_features, infrared_coverage = infrared_window_features(infrared, ir_start, ir_end)
        if min(radar_coverage, infrared_coverage) < minimum_coverage:
            continue
        correlation = _window_activity_correlation(radar, infrared, alignment, ir_start, ir_end)
        radar_quality = float(
            np.clip(
                radar_coverage * float(radar.metadata.get("valid_ratio", 0.0)),
                0.0,
                1.0,
            )
        )
        infrared_quality = float(
            np.clip(
                infrared_coverage
                * (0.35 + 0.65 * float(infrared_features.get("quality__detected_rate", 0.0))),
                0.0,
                1.0,
            )
        )
        sync_features = {
            "sync__offset_seconds": alignment.offset_seconds,
            "sync__drift_ppm": alignment.drift_ppm,
            "sync__score": alignment.score,
            "sync__window_activity_corr": correlation,
            "quality__radar": radar_quality,
            "quality__infrared": infrared_quality,
        }
        radar_features.update({key: value for key, value in sync_features.items() if key.startswith("sync__")})
        infrared_features.update({key: value for key, value in sync_features.items() if key.startswith("sync__")})
        fusion = {
            **_prefix(radar_features, "radar"),
            **_prefix(infrared_features, "infrared"),
            **sync_features,
        }
        windows.append(
            WindowFeature(
                window_id=f"{sample.batch_id}__{index:04d}",
                batch_id=sample.batch_id,
                label=sample.label,
                ir_start_seconds=float(ir_start),
                ir_end_seconds=float(ir_end),
                radar_start_seconds=float(radar_start),
                radar_end_seconds=float(radar_end),
                radar=radar_features,
                infrared=infrared_features,
                fusion=fusion,
                quality={"radar": radar_quality, "infrared": infrared_quality},
            )
        )

    if not windows:
        raise ValueError(
            f"No valid aligned windows for batch {sample.batch_id}. "
            "Lower window.min_coverage_ratio or inspect the input files."
        )
    return RecordExtraction(
        sample=sample,
        alignment=alignment,
        radar_metadata=radar.metadata,
        infrared_metadata=infrared.metadata,
        windows=windows,
    )
