from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class Alignment:
    offset_seconds: float
    score: float
    common_rate_hz: float
    status: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)

    def ir_to_radar(self, ir_seconds: float) -> float:
        return ir_seconds - self.offset_seconds

    def radar_to_ir(self, radar_seconds: float) -> float:
        return radar_seconds + self.offset_seconds


def _resample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    finite = np.isfinite(times) & np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros_like(grid)
    order = np.argsort(times[finite])
    x = times[finite][order]
    y = values[finite][order]
    unique, indices = np.unique(x, return_index=True)
    y = y[indices]
    if len(unique) < 2:
        return np.zeros_like(grid)
    return np.interp(grid, unique, y, left=y[0], right=y[-1])


def estimate_alignment(
    radar_times: np.ndarray,
    radar_activity: np.ndarray,
    ir_times: np.ndarray,
    ir_activity: np.ndarray,
    *,
    max_lag_seconds: float,
    common_rate_hz: float = 2.0,
) -> Alignment:
    if len(radar_times) < 4 or len(ir_times) < 4:
        return Alignment(0.0, 0.0, common_rate_hz, "insufficient_activity")
    duration = min(float(np.nanmax(radar_times)), float(np.nanmax(ir_times)))
    if not math.isfinite(duration) or duration <= 2.0:
        return Alignment(0.0, 0.0, common_rate_hz, "insufficient_duration")
    grid = np.arange(0.0, duration, 1.0 / common_rate_hz)
    radar = _resample(radar_times, radar_activity, grid)
    infrared = _resample(ir_times, ir_activity, grid)
    radar = signal.detrend(radar)
    infrared = signal.detrend(infrared)
    radar_std = float(np.std(radar))
    ir_std = float(np.std(infrared))
    if radar_std < 1e-8 or ir_std < 1e-8:
        return Alignment(0.0, 0.0, common_rate_hz, "flat_activity")
    radar = radar / radar_std
    infrared = infrared / ir_std
    correlation = signal.correlate(infrared, radar, mode="full", method="fft")
    lags = signal.correlation_lags(len(infrared), len(radar), mode="full")
    max_lag = int(round(max_lag_seconds * common_rate_hz))
    mask = np.abs(lags) <= max_lag
    if not np.any(mask):
        return Alignment(0.0, 0.0, common_rate_hz, "no_valid_lag")
    selected = correlation[mask]
    selected_lags = lags[mask]
    best = int(np.argmax(selected))
    lag_samples = int(selected_lags[best])
    overlap = max(1, len(grid) - abs(lag_samples))
    score = float(selected[best] / overlap)
    score = float(np.clip(score, -1.0, 1.0))
    return Alignment(lag_samples / common_rate_hz, score, common_rate_hz, "ok")
