from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def _aggregate(values: list[float], prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"{prefix}__{k}": math.nan for k in ("mean", "std", "q50", "q90", "max")}
    return {
        f"{prefix}__mean": float(np.mean(x)),
        f"{prefix}__std": float(np.std(x)),
        f"{prefix}__q50": float(np.quantile(x, 0.50)),
        f"{prefix}__q90": float(np.quantile(x, 0.90)),
        f"{prefix}__max": float(np.max(x)),
    }


def extract_infrared_features(
    path: str | Path,
    *,
    sample_fps: float = 3.0,
    resize_width: int = 640,
    max_samples: int = 2000,
    motion_percentile: float = 99.5,
    min_blob_area: int = 2,
    max_blob_area_ratio: float = 0.002,
) -> tuple[dict[str, float], np.ndarray, dict[str, float]]:
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open infrared video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0:
        fps = 30.0
    stride = max(1, int(round(fps / max(sample_fps, 1e-6))))

    intensity_mean: list[float] = []
    intensity_std: list[float] = []
    intensity_p99: list[float] = []
    gradient_energy: list[float] = []
    motion_mean: list[float] = []
    motion_p99: list[float] = []
    candidate_count: list[float] = []
    largest_blob_ratio: list[float] = []
    centroid_speeds: list[float] = []
    activity: list[float] = []

    prev_gray: np.ndarray | None = None
    prev_centroid: tuple[float, float] | None = None
    sampled = 0
    frame_index = 0
    tracked = 0

    try:
        while sampled < max_samples:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride != 0:
                frame_index += 1
                continue
            frame_index += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if resize_width > 0 and gray.shape[1] > resize_width:
                scale = resize_width / gray.shape[1]
                gray = cv2.resize(gray, (resize_width, max(1, int(round(gray.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
            gray_f = gray.astype(np.float32)

            intensity_mean.append(float(np.mean(gray_f)))
            intensity_std.append(float(np.std(gray_f)))
            intensity_p99.append(float(np.quantile(gray_f, 0.99)))
            gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
            gradient_energy.append(float(np.mean(np.hypot(gx, gy))))

            centroid: tuple[float, float] | None = None
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray).astype(np.float32)
                mm = float(np.mean(diff))
                mp = float(np.quantile(diff, 0.99))
                motion_mean.append(mm)
                motion_p99.append(mp)

                threshold = float(np.quantile(diff, motion_percentile / 100.0))
                binary = (diff >= max(threshold, 1.0)).astype(np.uint8)
                binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                n_labels, _, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
                frame_area = float(gray.shape[0] * gray.shape[1])
                max_area = max(float(min_blob_area), frame_area * max_blob_area_ratio)

                valid: list[tuple[float, tuple[float, float]]] = []
                for idx in range(1, n_labels):
                    area = float(stats[idx, cv2.CC_STAT_AREA])
                    if min_blob_area <= area <= max_area:
                        valid.append((area, (float(centers[idx, 0]), float(centers[idx, 1]))))
                candidate_count.append(float(len(valid)))
                largest_blob_ratio.append(max((a for a, _ in valid), default=0.0) / frame_area)
                if valid:
                    _, centroid = max(valid, key=lambda item: item[0])
                    tracked += 1

                if centroid is not None and prev_centroid is not None:
                    dx = centroid[0] - prev_centroid[0]
                    dy = centroid[1] - prev_centroid[1]
                    centroid_speeds.append(float(math.hypot(dx, dy) * sample_fps))
                if centroid is not None:
                    prev_centroid = centroid
                activity.append(mm + 0.1 * len(valid))
            else:
                activity.append(0.0)

            prev_gray = gray
            sampled += 1
    finally:
        cap.release()

    if sampled == 0:
        raise ValueError(f"No readable frames in {path}")

    features: dict[str, float] = {
        "meta__fps": fps,
        "meta__total_frames": float(total_frames),
        "meta__width": float(width),
        "meta__height": float(height),
        "meta__sampled_frames": float(sampled),
        "meta__duration_seconds": float(total_frames / fps) if total_frames > 0 else float(sampled / sample_fps),
        "quality__tracking_rate": float(tracked / max(sampled - 1, 1)),
    }
    for series, prefix in (
        (intensity_mean, "intensity_mean"),
        (intensity_std, "intensity_std"),
        (intensity_p99, "intensity_p99"),
        (gradient_energy, "gradient_energy"),
        (motion_mean, "motion_mean"),
        (motion_p99, "motion_p99"),
        (candidate_count, "candidate_count"),
        (largest_blob_ratio, "largest_blob_ratio"),
        (centroid_speeds, "centroid_speed"),
    ):
        features.update(_aggregate(series, prefix))

    quality = {
        "tracking_rate": features["quality__tracking_rate"],
        "sampled_frames": float(sampled),
        "duration_seconds": features["meta__duration_seconds"],
    }
    return features, np.asarray(activity, dtype=np.float32), quality
