from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy import signal

from .video_io import VideoReader


@dataclass
class InfraredSequence:
    frame: pd.DataFrame
    times: np.ndarray
    activity: np.ndarray
    sample_rate_hz: float
    duration_seconds: float
    metadata: dict[str, Any]


@dataclass
class _TrackState:
    x: float | None = None
    y: float | None = None
    vx: float = 0.0
    vy: float = 0.0
    missed: int = 0

    def predict(self) -> tuple[float, float] | None:
        if self.x is None or self.y is None:
            return None
        return self.x + self.vx, self.y + self.vy

    def update(self, position: tuple[float, float] | None, max_missed: int) -> None:
        if position is None:
            self.missed += 1
            if self.missed > max_missed:
                self.x = self.y = None
                self.vx = self.vy = 0.0
            return
        x, y = position
        if self.x is not None and self.y is not None:
            new_vx = x - self.x
            new_vy = y - self.y
            self.vx = 0.6 * self.vx + 0.4 * new_vx
            self.vy = 0.6 * self.vy + 0.4 * new_vy
        self.x, self.y = x, y
        self.missed = 0


def video_metadata(path: str | Path) -> dict[str, float | str | None]:
    with VideoReader(path) as reader:
        return reader.metadata()


def _candidate_components(
    response: np.ndarray,
    threshold: float,
    *,
    min_blob_area: int,
    max_blob_area_ratio: float,
) -> list[dict[str, float]]:
    binary = (response >= threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8)
    frame_area = float(response.shape[0] * response.shape[1])
    max_area = max(float(min_blob_area), frame_area * max_blob_area_ratio)
    candidates: list[dict[str, float]] = []
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if not (min_blob_area <= area <= max_area):
            continue
        width = float(stats[index, cv2.CC_STAT_WIDTH])
        height = float(stats[index, cv2.CC_STAT_HEIGHT])
        aspect = max(width, height) / max(min(width, height), 1.0)
        if aspect > 8.0:
            continue
        mask = labels == index
        score = float(np.mean(response[mask]) * math.sqrt(area))
        candidates.append(
            {
                "x": float(centers[index, 0]),
                "y": float(centers[index, 1]),
                "area": area,
                "score": score,
                "aspect": aspect,
            }
        )
    return candidates


def _choose_candidate(
    candidates: list[dict[str, float]],
    state: _TrackState,
    *,
    frame_diagonal: float,
    max_track_jump_ratio: float,
) -> dict[str, float] | None:
    if not candidates:
        return None
    predicted = state.predict()
    if predicted is None:
        return max(candidates, key=lambda item: item["score"])
    max_jump = frame_diagonal * max_track_jump_ratio * max(1.0, state.missed + 1.0)
    scored: list[tuple[float, dict[str, float]]] = []
    for candidate in candidates:
        distance = math.hypot(candidate["x"] - predicted[0], candidate["y"] - predicted[1])
        if distance <= max_jump:
            continuity = math.exp(-distance / max(max_jump, 1e-6))
            scored.append((candidate["score"] * (0.5 + continuity), candidate))
    if scored:
        return max(scored, key=lambda item: item[0])[1]
    if state.missed >= 2:
        return max(candidates, key=lambda item: item["score"])
    return None


def extract_infrared_sequence(
    path: str | Path,
    *,
    sample_fps: float = 5.0,
    resize_width: int = 960,
    max_samples: int = 5000,
    clahe_clip_limit: float = 2.0,
    motion_percentile: float = 99.65,
    contrast_percentile: float = 99.75,
    min_blob_area: int = 2,
    max_blob_area_ratio: float = 0.0015,
    max_track_jump_ratio: float = 0.10,
    max_missed_frames: int = 5,
) -> InfraredSequence:
    input_path = Path(path).expanduser().resolve()
    rows: list[dict[str, float]] = []
    activity: list[float] = []
    state = _TrackState()
    previous: np.ndarray | None = None
    previous_speed = 0.0
    previous_heading: float | None = None
    clahe = cv2.createCLAHE(clipLimit=max(0.1, clahe_clip_limit), tileGridSize=(8, 8))

    with VideoReader(input_path) as reader:
        if reader.info is None:
            raise RuntimeError(f"Video reader did not provide metadata: {input_path}")
        info = reader.info
        source_fps = float(info.fps or 30.0)
        total_frames = int(info.frames)
        source_width = int(info.width)
        source_height = int(info.height)
        requested_rate = min(max(float(sample_fps), 1e-6), source_fps)

        for current_time, gray in reader.iter_gray_frames(
            sample_fps=requested_rate,
            resize_width=resize_width,
            max_samples=max_samples,
        ):
            enhanced = clahe.apply(gray)
            smooth = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3.0)
            contrast = cv2.absdiff(enhanced, smooth).astype(np.float32)
            if previous is None:
                motion = np.zeros_like(contrast)
            else:
                motion = cv2.absdiff(enhanced, previous).astype(np.float32)
                motion = cv2.GaussianBlur(motion, (3, 3), 0)
            response = motion + 0.65 * contrast
            response_threshold = max(
                float(np.quantile(motion, motion_percentile / 100.0))
                + 0.65 * float(np.quantile(contrast, contrast_percentile / 100.0)),
                2.0,
            )
            candidates = _candidate_components(
                response,
                response_threshold,
                min_blob_area=min_blob_area,
                max_blob_area_ratio=max_blob_area_ratio,
            )
            diagonal = math.hypot(gray.shape[1], gray.shape[0])
            selected = _choose_candidate(
                candidates,
                state,
                frame_diagonal=diagonal,
                max_track_jump_ratio=max_track_jump_ratio,
            )
            position = None if selected is None else (selected["x"], selected["y"])
            old_x, old_y = state.x, state.y
            state.update(position, max_missed_frames)

            detected = float(selected is not None)
            x_norm = y_norm = area_ratio = target_score = math.nan
            speed = acceleration = turn_rate = math.nan
            if selected is not None:
                x_norm = selected["x"] / max(gray.shape[1], 1)
                y_norm = selected["y"] / max(gray.shape[0], 1)
                area_ratio = selected["area"] / max(gray.size, 1)
                target_score = selected["score"]
                if old_x is not None and old_y is not None:
                    dx = selected["x"] - old_x
                    dy = selected["y"] - old_y
                    speed = math.hypot(dx, dy) * requested_rate / max(diagonal, 1e-6)
                    acceleration = (speed - previous_speed) * requested_rate
                    heading = math.atan2(dy, dx)
                    if previous_heading is not None:
                        delta = math.atan2(math.sin(heading - previous_heading), math.cos(heading - previous_heading))
                        turn_rate = delta * requested_rate
                    previous_speed = speed
                    previous_heading = heading

            motion_mean = float(np.mean(motion))
            motion_p99 = float(np.quantile(motion, 0.99))
            contrast_p99 = float(np.quantile(contrast, 0.99))
            rows.append(
                {
                    "time_s": float(current_time),
                    "intensity_mean": float(np.mean(enhanced)),
                    "intensity_std": float(np.std(enhanced)),
                    "intensity_p99": float(np.quantile(enhanced, 0.99)),
                    "gradient_mean": float(np.mean(np.abs(cv2.Laplacian(enhanced, cv2.CV_32F)))),
                    "motion_mean": motion_mean,
                    "motion_p99": motion_p99,
                    "contrast_p99": contrast_p99,
                    "candidate_count": float(len(candidates)),
                    "detected": detected,
                    "track_x": x_norm,
                    "track_y": y_norm,
                    "track_area_ratio": area_ratio,
                    "track_score": target_score,
                    "track_speed": speed,
                    "track_acceleration": acceleration,
                    "track_turn_rate": turn_rate,
                }
            )
            activity.append(
                motion_p99
                + 0.25 * contrast_p99
                + (0.5 * target_score if math.isfinite(target_score) else 0.0)
            )
            previous = enhanced

        decoder_backend = info.backend
        decoder_codec = info.codec
        metadata_duration = float(info.duration_seconds)

    if not rows:
        raise ValueError(f"No readable frames in {input_path}")
    descriptor = pd.DataFrame(rows)
    times = descriptor.pop("time_s").to_numpy(dtype=float)
    if len(times) >= 2:
        positive_steps = np.diff(times)
        positive_steps = positive_steps[positive_steps > 0]
        actual_sample_rate = 1.0 / float(np.median(positive_steps)) if len(positive_steps) else requested_rate
    else:
        actual_sample_rate = requested_rate
    duration = metadata_duration or float(times[-1] + 1.0 / max(actual_sample_rate, 1e-9))
    detected_rate = float(descriptor["detected"].mean())
    return InfraredSequence(
        frame=descriptor,
        times=times,
        activity=np.asarray(activity, dtype=np.float32),
        sample_rate_hz=float(actual_sample_rate),
        duration_seconds=duration,
        metadata={
            "source_fps": source_fps,
            "source_frames": total_frames,
            "source_width": source_width,
            "source_height": source_height,
            "sampled_frames": len(rows),
            "detected_rate": detected_rate,
            "decoder_backend": decoder_backend,
            "decoder_codec": decoder_codec,
            "quality": float(min(1.0, detected_rate * 1.5 + min(0.25, len(rows) / 1000.0))),
        },
    )


def _spectral_features(values: np.ndarray, rate_hz: float, prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4 or rate_hz <= 0:
        return {f"{prefix}__dominant_hz": math.nan, f"{prefix}__spectral_entropy": math.nan}
    frequencies, power = signal.periodogram(signal.detrend(x), fs=rate_hz)
    frequencies, power = frequencies[1:], power[1:]
    if power.size == 0 or float(np.sum(power)) <= 0:
        return {f"{prefix}__dominant_hz": 0.0, f"{prefix}__spectral_entropy": 0.0}
    probability = power / np.sum(power)
    entropy = -float(np.sum(probability * np.log(probability + 1e-12))) / max(math.log(len(probability)), 1.0)
    return {
        f"{prefix}__dominant_hz": float(frequencies[int(np.argmax(power))]),
        f"{prefix}__spectral_entropy": entropy,
    }


def _aggregate(values: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    names = ("mean", "std", "min", "max", "q10", "q50", "q90", "iqr")
    if x.size == 0:
        return {f"{prefix}__{name}": math.nan for name in names}
    q10, q25, q50, q75, q90 = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        f"{prefix}__mean": float(np.mean(x)),
        f"{prefix}__std": float(np.std(x)),
        f"{prefix}__min": float(np.min(x)),
        f"{prefix}__max": float(np.max(x)),
        f"{prefix}__q10": float(q10),
        f"{prefix}__q50": float(q50),
        f"{prefix}__q90": float(q90),
        f"{prefix}__iqr": float(q75 - q25),
    }


def infrared_window_features(
    sequence: InfraredSequence,
    start_seconds: float,
    end_seconds: float,
) -> tuple[dict[str, float], float]:
    mask = (sequence.times >= start_seconds) & (sequence.times < end_seconds)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        indices = np.array([int(np.argmin(np.abs(sequence.times - (start_seconds + end_seconds) / 2.0)))])
    frame = sequence.frame.iloc[indices]
    features: dict[str, float] = {
        "meta__frames": float(len(frame)),
        "meta__duration": float(max(end_seconds - start_seconds, 0.0)),
        "quality__detected_rate": float(frame["detected"].mean()),
    }
    for column in frame.columns:
        features.update(_aggregate(frame[column].to_numpy(dtype=float), column))
    for column in ("motion_p99", "contrast_p99", "track_speed", "track_acceleration", "track_turn_rate"):
        features.update(_spectral_features(frame[column].to_numpy(dtype=float), sequence.sample_rate_hz, column))

    valid_xy = frame[["track_x", "track_y"]].dropna().to_numpy(dtype=float)
    if len(valid_xy) >= 2:
        segments = np.linalg.norm(np.diff(valid_xy, axis=0), axis=1)
        path_length = float(np.sum(segments))
        displacement = float(np.linalg.norm(valid_xy[-1] - valid_xy[0]))
        features["trajectory__path_length"] = path_length
        features["trajectory__straightness"] = displacement / max(path_length, 1e-8)
        features["trajectory__x_span"] = float(np.ptp(valid_xy[:, 0]))
        features["trajectory__y_span"] = float(np.ptp(valid_xy[:, 1]))
    else:
        features.update(
            {
                "trajectory__path_length": math.nan,
                "trajectory__straightness": math.nan,
                "trajectory__x_span": math.nan,
                "trajectory__y_span": math.nan,
            }
        )
    activity = sequence.activity[indices]
    features.update(_aggregate(activity, "activity"))
    coverage = min(1.0, len(indices) / max(1.0, (end_seconds - start_seconds) * sequence.sample_rate_hz))
    return features, float(coverage)
