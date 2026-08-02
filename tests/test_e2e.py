from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from scipy.io import savemat

from anti_air.config import load_config
from anti_air.dataset import discover_samples
from anti_air.evaluation import evaluate_grouped_cv
from anti_air.feature_store import build_feature_cache, tables_from_extractions
from anti_air.modeling import aggregate_record_probabilities, fit_model_bundle, predict_window_probabilities
from anti_air.pipeline import extract_record


def _write_video(path: Path, *, class_index: int, seed: int) -> None:
    width, height, fps, frames = 128, 96, 20.0, 100
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MP4 writer is unavailable")
    rng = np.random.default_rng(seed)
    for frame_index in range(frames):
        background = np.full((height, width), 70, dtype=np.uint8)
        noise = rng.normal(0, 3, size=(height, width))
        background = np.clip(background + noise, 0, 255).astype(np.uint8)
        if class_index == 0:
            x = 15 + frame_index
            y = 30 + int(5 * np.sin(frame_index / 8))
        else:
            x = 15 + frame_index // 2
            y = 20 + int(20 * np.sin(frame_index / 12))
        x %= width - 5
        cv2.circle(background, (x, y), 2 + class_index, 230, -1)
        bgr = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        writer.write(bgr)
    writer.release()


def _write_record(root: Path, batch_id: str, label: str, class_index: int, seed: int) -> None:
    samples = 200
    t = np.linspace(0, 5, samples, endpoint=False)
    rng = np.random.default_rng(seed)
    if class_index == 0:
        matrix = np.column_stack(
            [
                np.sin(2 * np.pi * 1.0 * t),
                1.5 + 0.2 * np.cos(2 * np.pi * 0.5 * t),
                np.linspace(0, 1, samples),
            ]
        )
    else:
        matrix = np.column_stack(
            [
                np.sin(2 * np.pi * 2.5 * t),
                3.0 + 0.5 * np.cos(2 * np.pi * 1.5 * t),
                np.sin(2 * np.pi * 0.25 * t),
            ]
        )
    matrix += rng.normal(0, 0.03, size=matrix.shape)
    savemat(root / f"radar_{batch_id}_{label}_12：00.mat", {"radar_data": matrix})
    _write_video(root / f"ir_{batch_id}_{label}_12：00.mp4", class_index=class_index, seed=seed)


def _test_config() -> dict:
    config = load_config(None)
    config["infrared"].update({"sample_fps": 5.0, "resize_width": 128, "max_samples": 100})
    config["radar"].update({"max_numeric_columns": 8, "spectral_bins": 3})
    config["alignment"].update({"estimate_drift": False, "max_lag_seconds": 2.0})
    config["window"].update(
        {"length_seconds": 2.0, "stride_seconds": 1.0, "min_coverage_ratio": 0.3, "max_windows_per_record": 10}
    )
    config["model"].update({"n_estimators": 40, "min_samples_leaf": 1})
    config["evaluation"].update({"n_splits": 3, "min_valid_folds": 2})
    config["runtime"]["n_jobs"] = 1
    return config


def test_end_to_end_training_and_grouped_evaluation(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for index in range(3):
        _write_record(data, f"A{index}", "class-A", 0, 100 + index)
        _write_record(data, f"B{index}", "class-B", 1, 200 + index)

    samples = discover_samples(data, require_labels=True, strict_pairs=True)
    assert len(samples) == 6
    config = _test_config()
    tables = build_feature_cache(samples, config, tmp_path / "features")
    assert tables.fusion["batch_id"].nunique() == 6
    assert len(tables.fusion) > 6

    evaluation = evaluate_grouped_cv(tables, config)
    assert evaluation.metrics["status"] in {"ok", "limited_folds"}
    assert evaluation.metrics["valid_folds"] >= 2
    assert 0.0 <= evaluation.metrics["macro_f1"] <= 1.0

    bundle = fit_model_bundle(tables, config)
    model_path = tmp_path / "model.joblib"
    joblib.dump(bundle, model_path)
    loaded = joblib.load(model_path)
    extraction = extract_record(samples[0], config)
    single = tables_from_extractions([extraction], config)
    window = predict_window_probabilities(loaded, single)
    record = aggregate_record_probabilities(window, loaded.classes)
    assert len(record) == 1
    assert record.iloc[0]["predicted_label"] in {"class-A", "class-B"}
    assert 0.0 <= float(record.iloc[0]["confidence"]) <= 1.0
