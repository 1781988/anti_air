from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal


_TIME_HINT = re.compile(r"(^|_)(time|timestamp|datetime|date|second|sec|frame|时刻|时间|秒)($|_)", re.I)


@dataclass
class RadarSequence:
    frame: pd.DataFrame
    times: np.ndarray
    activity: np.ndarray
    rate_hz: float
    duration_seconds: float
    metadata: dict[str, Any]


def _load_mat_dictionary(path: Path) -> dict[str, Any]:
    matio_error: Exception | None = None
    try:
        from matio import load_from_mat

        return load_from_mat(path, raw_data=False, add_table_attrs=True)
    except Exception as exc:
        matio_error = exc
    try:
        from scipy.io import loadmat

        return loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as scipy_error:
        raise RuntimeError(
            "Unable to load radar MAT file. If it contains a MATLAB table, install "
            "the 'mat-io' package or run scripts/convert_matlab_tables.m. "
            f"mat-io error={matio_error!r}; scipy error={scipy_error!r}"
        ) from scipy_error


def _candidate_to_frame(value: Any, name: str) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame.columns = [str(column) for column in frame.columns]
        return frame
    if isinstance(value, pd.Series):
        return value.to_frame(name=name)
    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return pd.DataFrame.from_records(value.reshape(-1))
        if value.ndim == 0:
            return None
        if value.ndim == 1:
            return pd.DataFrame({name: list(value)})
        if value.ndim == 2 and min(value.shape) > 0:
            return pd.DataFrame(value, columns=[f"{name}_{idx}" for idx in range(value.shape[1])])
    if hasattr(value, "__dict__"):
        fields = {key: val for key, val in vars(value).items() if not key.startswith("_")}
        if fields:
            try:
                return pd.DataFrame(fields)
            except Exception:
                return None
    return None


def load_radar_frame(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    raw = _load_mat_dictionary(input_path)
    candidates: list[pd.DataFrame] = []
    for name, value in raw.items():
        if str(name).startswith("__"):
            continue
        frame = _candidate_to_frame(value, str(name))
        if frame is not None and not frame.empty:
            candidates.append(frame)
    if not candidates:
        raise ValueError(f"No table or matrix found in radar file: {input_path}")
    frame = max(candidates, key=lambda item: item.shape[0] * max(1, item.shape[1])).reset_index(drop=True)
    if frame.shape[0] < frame.shape[1] and frame.shape[0] <= 8:
        frame = frame.T.reset_index(drop=True)
        frame.columns = [f"signal_{idx}" for idx in range(frame.shape[1])]
    return frame


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff]+", "_", str(name)).strip("_")
    return cleaned or "unnamed"


def _object_scalar(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (bool, int, float, np.number)):
        if np.iscomplexobj(value):
            return float(abs(value))
        return float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return float(pd.Timestamp(value).timestamp())
    array = np.asarray(value)
    if array.size == 1 and np.issubdtype(array.dtype, np.number):
        scalar = array.reshape(-1)[0]
        return float(abs(scalar)) if np.iscomplexobj(scalar) else float(scalar)
    return math.nan


def _expand_column(series: pd.Series, name: str, max_vector_expansion: int) -> dict[str, pd.Series]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return {name: pd.Series(series.astype("int64") / 1e9, index=series.index, dtype=float)}
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy()
        if np.iscomplexobj(values):
            return {
                f"{name}_abs": pd.Series(np.abs(values), index=series.index, dtype=float),
                f"{name}_real": pd.Series(np.real(values), index=series.index, dtype=float),
                f"{name}_imag": pd.Series(np.imag(values), index=series.index, dtype=float),
            }
        return {name: pd.to_numeric(series, errors="coerce")}

    scalar = series.map(_object_scalar)
    if scalar.notna().mean() >= 0.25:
        return {name: scalar.astype(float)}

    arrays: list[np.ndarray | None] = []
    lengths: list[int] = []
    for value in series:
        try:
            array = np.asarray(value)
            if array.ndim == 0 or not np.issubdtype(array.dtype, np.number):
                arrays.append(None)
                continue
            flat = array.reshape(-1)
            arrays.append(flat)
            lengths.append(len(flat))
        except Exception:
            arrays.append(None)
    if not lengths:
        return {}

    result: dict[str, pd.Series] = {}
    common_length = max(set(lengths), key=lengths.count)
    if common_length <= max_vector_expansion and lengths.count(common_length) >= max(2, len(series) // 2):
        for idx in range(common_length):
            values = [
                float(abs(array[idx])) if array is not None and len(array) > idx else math.nan
                for array in arrays
            ]
            result[f"{name}_{idx}"] = pd.Series(values, index=series.index, dtype=float)
    else:
        for statistic in ("mean", "std", "max", "energy"):
            values: list[float] = []
            for array in arrays:
                if array is None or array.size == 0:
                    values.append(math.nan)
                    continue
                magnitude = np.abs(array.astype(np.complex128))
                if statistic == "mean":
                    values.append(float(np.mean(magnitude)))
                elif statistic == "std":
                    values.append(float(np.std(magnitude)))
                elif statistic == "max":
                    values.append(float(np.max(magnitude)))
                else:
                    values.append(float(np.mean(magnitude**2)))
            result[f"{name}_{statistic}"] = pd.Series(values, index=series.index, dtype=float)
    return result


def numeric_radar_frame(
    frame: pd.DataFrame,
    *,
    max_columns: int = 96,
    max_vector_expansion: int = 16,
) -> pd.DataFrame:
    converted: dict[str, pd.Series] = {}
    for column in frame.columns:
        name = _clean_name(str(column))
        for expanded_name, expanded in _expand_column(frame[column], name, max_vector_expansion).items():
            if expanded.notna().any():
                unique_name = expanded_name
                suffix = 1
                while unique_name in converted:
                    unique_name = f"{expanded_name}_{suffix}"
                    suffix += 1
                converted[unique_name] = expanded
            if len(converted) >= max_columns:
                break
        if len(converted) >= max_columns:
            break
    if not converted:
        raise ValueError("Radar table contains no usable numeric columns")
    return pd.DataFrame(converted).replace([np.inf, -np.inf], np.nan)


def _infer_times(frame: pd.DataFrame, numeric: pd.DataFrame, duration_hint: float | None) -> tuple[np.ndarray, float, str]:
    for column in frame.columns:
        if not _TIME_HINT.search(_clean_name(str(column))):
            continue
        series = frame[column]
        try:
            if pd.api.types.is_datetime64_any_dtype(series) or series.dtype == object:
                parsed = pd.to_datetime(series, errors="coerce")
                valid = parsed.notna()
                if valid.sum() >= 3:
                    values = parsed.astype("int64").to_numpy(dtype=np.float64) / 1e9
                    values = values - np.nanmin(values[valid.to_numpy()])
                    diffs = np.diff(values[np.isfinite(values)])
                    step = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else math.nan
                    if math.isfinite(step) and step > 0:
                        return values, 1.0 / step, str(column)
        except Exception:
            pass
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size >= 3 and np.all(np.diff(finite) >= 0):
            diffs = np.diff(finite)
            positive = diffs[diffs > 0]
            if positive.size:
                step = float(np.median(positive))
                scale = 1.0
                if step > 1e6:
                    scale = 1e9
                elif step > 1e3:
                    scale = 1e3
                times = (values - finite[0]) / scale
                rate = 1.0 / max(step / scale, 1e-9)
                if 1e-4 <= rate <= 1e6:
                    return times, rate, str(column)

    rows = len(numeric)
    if duration_hint is not None and duration_hint > 0 and rows > 1:
        times = np.linspace(0.0, float(duration_hint), rows, endpoint=False)
        return times, rows / float(duration_hint), "video_duration_hint"
    return np.arange(rows, dtype=float), 1.0, "row_index"


def _fill_numeric(numeric: pd.DataFrame) -> np.ndarray:
    values = numeric.to_numpy(dtype=np.float64)
    medians = np.nanmedian(values, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    return np.where(np.isfinite(values), values, medians)


def extract_radar_sequence(
    path: str | Path,
    *,
    duration_hint: float | None = None,
    max_numeric_columns: int = 96,
    max_vector_expansion: int = 16,
) -> RadarSequence:
    raw_frame = load_radar_frame(path)
    numeric = numeric_radar_frame(
        raw_frame,
        max_columns=max_numeric_columns,
        max_vector_expansion=max_vector_expansion,
    )
    times, rate_hz, time_source = _infer_times(raw_frame, numeric, duration_hint)
    values = _fill_numeric(numeric)
    center = np.median(values, axis=0)
    scale = np.median(np.abs(values - center), axis=0) * 1.4826
    scale[scale < 1e-8] = np.std(values[:, scale < 1e-8], axis=0) if np.any(scale < 1e-8) else scale[scale < 1e-8]
    scale[scale < 1e-8] = 1.0
    standardized = (values - center) / scale
    delta = np.vstack([np.zeros((1, standardized.shape[1])), np.diff(standardized, axis=0)])
    activity = np.sqrt(np.mean(delta**2, axis=1))
    duration = float(times[-1] - times[0] + 1.0 / max(rate_hz, 1e-9)) if len(times) else 0.0
    return RadarSequence(
        frame=numeric.reset_index(drop=True),
        times=np.asarray(times, dtype=np.float64),
        activity=activity.astype(np.float32),
        rate_hz=float(rate_hz),
        duration_seconds=duration,
        metadata={
            "rows": int(len(numeric)),
            "raw_columns": int(raw_frame.shape[1]),
            "numeric_columns": int(numeric.shape[1]),
            "valid_ratio": float(np.isfinite(numeric.to_numpy(dtype=float)).mean()),
            "time_source": time_source,
        },
    )


def _spectral_features(values: np.ndarray, rate_hz: float, prefix: str, bins: int) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    result: dict[str, float] = {}
    if x.size < 4 or rate_hz <= 0:
        return {f"{prefix}__spectral_entropy": math.nan, f"{prefix}__dominant_hz": math.nan}
    x = signal.detrend(x)
    frequencies, power = signal.periodogram(x, fs=rate_hz)
    if power.size <= 1 or float(np.sum(power[1:])) <= 0:
        return {f"{prefix}__spectral_entropy": 0.0, f"{prefix}__dominant_hz": 0.0}
    frequencies = frequencies[1:]
    power = power[1:]
    probabilities = power / np.sum(power)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12))) / math.log(len(probabilities))
    result[f"{prefix}__spectral_entropy"] = entropy
    result[f"{prefix}__dominant_hz"] = float(frequencies[int(np.argmax(power))])
    edges = np.linspace(0.0, max(rate_hz / 2.0, 1e-9), bins + 1)
    total = float(np.sum(power)) + 1e-12
    for idx in range(bins):
        mask = (frequencies >= edges[idx]) & (frequencies < edges[idx + 1])
        result[f"{prefix}__band_{idx}"] = float(np.sum(power[mask]) / total)
    return result


def _series_features(values: np.ndarray, prefix: str, rate_hz: float, spectral_bins: int) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    finite = x[np.isfinite(x)]
    keys = ["mean", "std", "min", "max", "median", "q10", "q90", "iqr", "rms", "diff_std", "diff_abs_max"]
    if finite.size == 0:
        return {f"{prefix}__{key}": math.nan for key in keys}
    q10, q25, q50, q75, q90 = np.quantile(finite, [0.1, 0.25, 0.5, 0.75, 0.9])
    diffs = np.diff(finite)
    result = {
        f"{prefix}__mean": float(np.mean(finite)),
        f"{prefix}__std": float(np.std(finite)),
        f"{prefix}__min": float(np.min(finite)),
        f"{prefix}__max": float(np.max(finite)),
        f"{prefix}__median": float(q50),
        f"{prefix}__q10": float(q10),
        f"{prefix}__q90": float(q90),
        f"{prefix}__iqr": float(q75 - q25),
        f"{prefix}__rms": float(np.sqrt(np.mean(finite**2))),
        f"{prefix}__diff_std": float(np.std(diffs)) if diffs.size else 0.0,
        f"{prefix}__diff_abs_max": float(np.max(np.abs(diffs))) if diffs.size else 0.0,
    }
    result.update(_spectral_features(finite, rate_hz, prefix, spectral_bins))
    return result


def radar_window_features(
    sequence: RadarSequence,
    start_seconds: float,
    end_seconds: float,
    *,
    spectral_bins: int = 8,
) -> tuple[dict[str, float], float]:
    mask = (sequence.times >= start_seconds) & (sequence.times < end_seconds)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        indices = np.array([int(np.argmin(np.abs(sequence.times - (start_seconds + end_seconds) / 2.0)))])
    frame = sequence.frame.iloc[indices]
    features: dict[str, float] = {
        "meta__rows": float(len(frame)),
        "meta__duration": float(max(end_seconds - start_seconds, 0.0)),
        "quality__valid_ratio": float(np.isfinite(frame.to_numpy(dtype=float)).mean()),
    }
    for column in frame.columns:
        features.update(
            _series_features(
                frame[column].to_numpy(dtype=float),
                f"col__{_clean_name(str(column))}",
                sequence.rate_hz,
                spectral_bins,
            )
        )
    activity = sequence.activity[indices]
    features.update(_series_features(activity, "activity", sequence.rate_hz, spectral_bins))
    coverage = min(1.0, len(indices) / max(1.0, (end_seconds - start_seconds) * sequence.rate_hz))
    return features, float(coverage)
