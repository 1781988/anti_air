from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import correlate, correlation_lags


@dataclass(frozen=True)
class AlignmentResult:
    offset_seconds: float
    score: float
    common_rate_hz: float
    convention: str = "positive means infrared activity lags radar activity"

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _resample(values: np.ndarray, source_rate_hz: float, target_rate_hz: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2 or source_rate_hz <= 0:
        return values
    duration = (values.size - 1) / source_rate_hz
    target_n = max(2, int(round(duration * target_rate_hz)) + 1)
    source_t = np.linspace(0.0, duration, values.size)
    target_t = np.linspace(0.0, duration, target_n)
    return np.interp(target_t, source_t, values)


def estimate_alignment(
    radar_activity: np.ndarray,
    infrared_activity: np.ndarray,
    *,
    radar_rate_hz: float,
    infrared_rate_hz: float,
    common_rate_hz: float = 5.0,
    max_lag_seconds: float = 30.0,
) -> AlignmentResult:
    r = _resample(radar_activity, radar_rate_hz, common_rate_hz)
    i = _resample(infrared_activity, infrared_rate_hz, common_rate_hz)
    if r.size < 3 or i.size < 3:
        return AlignmentResult(0.0, 0.0, common_rate_hz)

    r = (r - np.mean(r)) / (np.std(r) + 1e-8)
    i = (i - np.mean(i)) / (np.std(i) + 1e-8)
    corr = correlate(i, r, mode="full", method="fft")
    lags = correlation_lags(i.size, r.size, mode="full")
    denom = max(1.0, float(min(i.size, r.size)))
    corr = corr / denom

    max_lag = int(round(max_lag_seconds * common_rate_hz))
    mask = np.abs(lags) <= max_lag
    if not np.any(mask):
        return AlignmentResult(0.0, 0.0, common_rate_hz)
    idx_local = int(np.argmax(corr[mask]))
    valid_lags = lags[mask]
    valid_corr = corr[mask]
    return AlignmentResult(
        offset_seconds=float(valid_lags[idx_local] / common_rate_hz),
        score=float(valid_corr[idx_local]),
        common_rate_hz=float(common_rate_hz),
    )
