from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.signal import correlate, correlation_lags


@dataclass(frozen=True)
class AlignmentPoint:
    radar_time_seconds: float
    offset_seconds: float
    score: float


@dataclass(frozen=True)
class AlignmentResult:
    offset_seconds: float
    scale: float
    drift_ppm: float
    score: float
    common_rate_hz: float
    points: list[AlignmentPoint] = field(default_factory=list)
    mapping: str = "t_ir = scale * t_radar + offset"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def infrared_to_radar(self, time_seconds: float) -> float:
        return (time_seconds - self.offset_seconds) / max(self.scale, 1e-9)

    def radar_to_infrared(self, time_seconds: float) -> float:
        return self.scale * time_seconds + self.offset_seconds


def _resample(values: np.ndarray, source_rate_hz: float, target_rate_hz: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2 or source_rate_hz <= 0 or target_rate_hz <= 0:
        return values
    duration = (values.size - 1) / source_rate_hz
    target_count = max(2, int(round(duration * target_rate_hz)) + 1)
    source_time = np.arange(values.size, dtype=float) / source_rate_hz
    target_time = np.arange(target_count, dtype=float) / target_rate_hz
    target_time = np.clip(target_time, 0.0, source_time[-1])
    return np.interp(target_time, source_time, values)


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    median = float(np.nanmedian(values))
    scale = float(np.nanmedian(np.abs(values - median)) * 1.4826)
    if scale < 1e-8:
        scale = float(np.nanstd(values))
    if scale < 1e-8:
        return np.zeros_like(values)
    output = (np.where(finite, values, median) - median) / scale
    return np.clip(output, -10.0, 10.0)


def _estimate_lag(
    radar: np.ndarray,
    infrared: np.ndarray,
    *,
    common_rate_hz: float,
    max_lag_seconds: float,
) -> tuple[float, float]:
    if radar.size < 3 or infrared.size < 3:
        return 0.0, 0.0
    radar = _normalize(radar)
    infrared = _normalize(infrared)
    if np.std(radar) < 1e-8 or np.std(infrared) < 1e-8:
        return 0.0, 0.0
    correlation = correlate(infrared, radar, mode="full", method="fft")
    lags = correlation_lags(infrared.size, radar.size, mode="full")
    overlap = np.minimum(
        np.minimum(np.arange(1, correlation.size + 1), np.arange(correlation.size, 0, -1)),
        min(infrared.size, radar.size),
    )
    correlation = correlation / np.maximum(overlap, 1)
    max_lag = int(round(max_lag_seconds * common_rate_hz))
    valid = np.abs(lags) <= max_lag
    if not np.any(valid):
        return 0.0, 0.0
    valid_corr = correlation[valid]
    valid_lags = lags[valid]
    index = int(np.argmax(valid_corr))
    lag_seconds = float(valid_lags[index] / common_rate_hz)
    score = float(np.clip(valid_corr[index], -1.0, 1.0))
    return lag_seconds, score


def estimate_alignment(
    radar_activity: np.ndarray,
    infrared_activity: np.ndarray,
    *,
    radar_rate_hz: float,
    infrared_rate_hz: float,
    common_rate_hz: float = 5.0,
    max_lag_seconds: float = 45.0,
    estimate_drift: bool = True,
    drift_segments: int = 5,
    max_drift_ppm: float = 5000.0,
) -> AlignmentResult:
    radar = _resample(radar_activity, radar_rate_hz, common_rate_hz)
    infrared = _resample(infrared_activity, infrared_rate_hz, common_rate_hz)
    offset, global_score = _estimate_lag(
        radar,
        infrared,
        common_rate_hz=common_rate_hz,
        max_lag_seconds=max_lag_seconds,
    )
    points: list[AlignmentPoint] = []
    scale = 1.0
    drift_ppm = 0.0

    common_duration = min(len(radar), len(infrared)) / max(common_rate_hz, 1e-9)
    if estimate_drift and drift_segments >= 3 and common_duration >= max(60.0, 4.0 * max_lag_seconds):
        segment_duration = common_duration / drift_segments
        segment_samples = int(round(segment_duration * common_rate_hz))
        for segment_index in range(drift_segments):
            center = (segment_index + 0.5) * segment_duration
            radar_start = max(0, segment_index * segment_samples)
            radar_end = min(len(radar), (segment_index + 1) * segment_samples)
            expected_ir_start = int(round(radar_start + offset * common_rate_hz))
            expected_ir_end = int(round(radar_end + offset * common_rate_hz))
            margin = int(round(max_lag_seconds * common_rate_hz))
            infrared_start = max(0, expected_ir_start - margin)
            infrared_end = min(len(infrared), expected_ir_end + margin)
            if radar_end - radar_start < 10 or infrared_end - infrared_start < 10:
                continue
            local_offset, local_score = _estimate_lag(
                radar[radar_start:radar_end],
                infrared[infrared_start:infrared_end],
                common_rate_hz=common_rate_hz,
                max_lag_seconds=max_lag_seconds,
            )
            local_offset += (infrared_start - radar_start) / common_rate_hz
            points.append(AlignmentPoint(center, local_offset, local_score))

        reliable = [point for point in points if point.score >= max(0.05, global_score * 0.35)]
        if len(reliable) >= 3:
            x = np.asarray([point.radar_time_seconds for point in reliable], dtype=float)
            y = np.asarray([point.offset_seconds for point in reliable], dtype=float)
            weights = np.asarray([max(point.score, 0.01) for point in reliable], dtype=float)
            design = np.column_stack([x, np.ones_like(x)])
            weighted_design = design * np.sqrt(weights[:, None])
            weighted_y = y * np.sqrt(weights)
            slope, intercept = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)[0]
            max_slope = max_drift_ppm / 1e6
            slope = float(np.clip(slope, -max_slope, max_slope))
            scale = 1.0 + slope
            offset = float(intercept)
            drift_ppm = slope * 1e6
            global_score = float(np.average([point.score for point in reliable], weights=weights))

    return AlignmentResult(
        offset_seconds=float(offset),
        scale=float(scale),
        drift_ppm=float(drift_ppm),
        score=float(global_score),
        common_rate_hz=float(common_rate_hz),
        points=points,
    )
