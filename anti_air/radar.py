from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _load_mat_dictionary(path: Path) -> dict[str, Any]:
    try:
        from matio import load_from_mat

        return load_from_mat(path, raw_data=False, add_table_attrs=True)
    except Exception as matio_error:
        try:
            from scipy.io import loadmat

            return loadmat(path, squeeze_me=True, struct_as_record=False)
        except Exception as scipy_error:
            raise RuntimeError(
                "Unable to load radar MAT file. The competition files may contain a "
                "MATLAB table; install the 'mat-io' package or export the table to CSV. "
                f"mat-io error={matio_error!r}; scipy error={scipy_error!r}"
            ) from scipy_error


def _candidate_to_frame(value: Any, name: str) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame.columns = [str(c) for c in frame.columns]
        return frame

    if isinstance(value, pd.Series):
        return value.to_frame(name=name)

    if isinstance(value, np.ndarray):
        if value.dtype.names:
            return pd.DataFrame.from_records(value)
        if value.ndim == 1 and np.issubdtype(value.dtype, np.number):
            return pd.DataFrame({name: value})
        if value.ndim == 2 and np.issubdtype(value.dtype, np.number):
            return pd.DataFrame(value, columns=[f"{name}_{i}" for i in range(value.shape[1])])
    return None


def load_radar_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    raw = _load_mat_dictionary(path)
    candidates: list[pd.DataFrame] = []
    for name, value in raw.items():
        if str(name).startswith("__"):
            continue
        frame = _candidate_to_frame(value, str(name))
        if frame is not None and not frame.empty:
            candidates.append(frame)

    if not candidates:
        raise ValueError(f"No numeric matrix or MATLAB table found in {path}")

    frame = max(candidates, key=lambda x: x.shape[0] * max(1, x.shape[1])).reset_index(drop=True)
    return frame


def _clean_name(name: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    return name or "unnamed"


def _series_stats(values: pd.Series, prefix: str) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = x[np.isfinite(x)]
    result: dict[str, float] = {
        f"{prefix}__missing_ratio": float(1.0 - len(finite) / max(len(x), 1)),
    }
    if finite.size == 0:
        for key in ("mean", "std", "min", "max", "median", "q10", "q90", "iqr", "rms", "diff_std", "diff_abs_max"):
            result[f"{prefix}__{key}"] = math.nan
        return result

    q10, q25, q50, q75, q90 = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90])
    diffs = np.diff(finite)
    result.update(
        {
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
    )
    return result


def numeric_radar_frame(frame: pd.DataFrame, max_columns: int = 64) -> pd.DataFrame:
    converted: dict[str, pd.Series] = {}
    for col in frame.columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        if series.notna().any():
            converted[_clean_name(str(col))] = series
        if len(converted) >= max_columns:
            break
    if not converted:
        raise ValueError("Radar table contains no scalar numeric columns")
    return pd.DataFrame(converted)


def extract_radar_features(
    path: str | Path,
    *,
    max_numeric_columns: int = 64,
) -> tuple[dict[str, float], np.ndarray]:
    frame = load_radar_frame(path)
    numeric = numeric_radar_frame(frame, max_columns=max_numeric_columns)

    features: dict[str, float] = {
        "meta__rows": float(len(numeric)),
        "meta__numeric_columns": float(numeric.shape[1]),
        "quality__valid_ratio": float(np.isfinite(numeric.to_numpy(dtype=float)).mean()),
    }
    for col in numeric.columns:
        features.update(_series_stats(numeric[col], f"col__{col}"))

    values = numeric.to_numpy(dtype=np.float64)
    medians = np.nanmedian(values, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    values = np.where(np.isfinite(values), values, medians)
    scales = np.std(values, axis=0)
    scales[scales < 1e-8] = 1.0
    standardized = (values - np.mean(values, axis=0)) / scales
    activity = np.sqrt(np.mean(standardized**2, axis=1))
    return features, activity.astype(np.float32)
