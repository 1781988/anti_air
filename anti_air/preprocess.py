from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .alignment import Alignment, estimate_alignment
from .config import config_hash
from .data import Sample
from .radar import RadarSequence, extract_radar_sequence
from .video_io import VideoReader


@dataclass(frozen=True)
class CachedRecord:
    batch_id: str
    label: str
    path: Path
    windows: int
    alignment: dict[str, Any]
    radar_metadata: dict[str, Any]
    infrared_metadata: dict[str, Any]
    cache_hit: bool

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


@dataclass(frozen=True)
class PreprocessResult:
    records: list[CachedRecord]
    radar_schema: list[str]
    elapsed_seconds: float
    cache_dir: Path

    @property
    def window_count(self) -> int:
        return sum(record.windows for record in self.records)

    def summary(self) -> dict[str, Any]:
        return {
            "records": len(self.records),
            "windows": self.window_count,
            "cache_hits": sum(int(record.cache_hit) for record in self.records),
            "radar_schema": self.radar_schema,
            "elapsed_seconds": self.elapsed_seconds,
            "items": [record.to_dict() for record in self.records],
        }


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    payload = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _record_cache_key(sample: Sample, config: dict[str, Any], schema: list[str]) -> str:
    payload = {
        "batch": sample.batch_id,
        "radar": _file_fingerprint(sample.radar_path),
        "infrared": _file_fingerprint(sample.infrared_path),
        "config": config_hash(config),
        "schema": schema,
        "version": 3,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _video_info(path: Path) -> dict[str, Any]:
    with VideoReader(path) as reader:
        return reader.metadata()


def fit_radar_schema(samples: list[Sample], config: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    coverage: dict[str, int] = {}
    variance: dict[str, list[float]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for sample in samples:
        video = _video_info(sample.infrared_path)
        sequence = extract_radar_sequence(
            sample.radar_path,
            duration_hint=float(video["duration_seconds"]) or None,
            max_numeric_columns=max(96, int(config["radar_channels"]) * 3),
            max_vector_expansion=int(config["radar_vector_expansion"]),
        )
        metadata[sample.batch_id] = {
            "video": video,
            "radar_columns": list(sequence.frame.columns),
            "radar_metadata": sequence.metadata,
        }
        for column in sequence.frame.columns:
            values = sequence.frame[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            coverage[column] = coverage.get(column, 0) + 1
            variance.setdefault(column, []).append(float(np.var(finite)))
    if not coverage:
        raise ValueError("No usable radar numeric columns were found")
    required = max(1, math.ceil(len(samples) * 0.5))
    candidates = [name for name, count in coverage.items() if count >= required]
    if not candidates:
        candidates = list(coverage)
    candidates.sort(
        key=lambda name: (
            -coverage[name],
            -float(np.median(variance.get(name, [0.0]))),
            name,
        )
    )
    selected = candidates[: int(config["radar_channels"])]
    while len(selected) < int(config["radar_channels"]):
        selected.append(f"__padding_{len(selected)}")
    return selected, metadata


def _normalise_map(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    low, high = np.quantile(image, [0.01, 0.995])
    if high <= low + 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = np.clip((image - low) * 255.0 / (high - low), 0, 255)
    return scaled.astype(np.uint8)


def _target_position(response: np.ndarray, previous: tuple[float, float] | None) -> tuple[tuple[float, float], bool]:
    threshold = float(np.quantile(response, 0.9975))
    binary = (response >= max(threshold, 2.0)).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, _, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
    candidates: list[tuple[float, float, float]] = []
    max_area = max(4.0, response.size * 0.0015)
    for index in range(1, count):
        area = float(stats[index, cv2.CC_STAT_AREA])
        if not 2.0 <= area <= max_area:
            continue
        x, y = float(centers[index, 0]), float(centers[index, 1])
        score = float(response[int(round(y)) % response.shape[0], int(round(x)) % response.shape[1]])
        if previous is not None:
            distance = math.hypot(x - previous[0], y - previous[1])
            score *= math.exp(-distance / max(0.15 * math.hypot(*response.shape), 1.0))
        candidates.append((score, x, y))
    if candidates:
        _, x, y = max(candidates)
        return (x, y), True
    y, x = np.unravel_index(int(np.argmax(response)), response.shape)
    return (float(x), float(y)), False


def _crop(image: np.ndarray, center: tuple[float, float], side: int) -> np.ndarray:
    x, y = center
    half = side // 2
    x0, x1 = int(round(x)) - half, int(round(x)) + half
    y0, y1 = int(round(y)) - half, int(round(y)) + half
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - image.shape[1])
    pad_bottom = max(0, y1 - image.shape[0])
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        x0 += pad_left
        x1 += pad_left
        y0 += pad_top
        y1 += pad_top
    return image[y0:y1, x0:x1]


def extract_video_timeline(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    image_size = int(config["image_size"])
    sample_fps = float(config["video_sample_fps"])
    max_samples = int(config["max_video_samples"])
    frames: list[np.ndarray] = []
    times: list[float] = []
    activities: list[float] = []
    detections: list[float] = []
    previous_enhanced: np.ndarray | None = None
    previous_position: tuple[float, float] | None = None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    with VideoReader(path) as reader:
        info = reader.metadata()
        for timestamp, gray in reader.iter_gray_frames(
            sample_fps=sample_fps,
            resize_width=640,
            max_samples=max_samples,
        ):
            enhanced = clahe.apply(gray)
            background = cv2.GaussianBlur(enhanced, (0, 0), 4.0)
            contrast = cv2.absdiff(enhanced, background).astype(np.float32)
            if previous_enhanced is None:
                motion = np.zeros_like(contrast)
            else:
                motion = cv2.absdiff(enhanced, previous_enhanced).astype(np.float32)
                motion = cv2.GaussianBlur(motion, (3, 3), 0)
            response = contrast + 0.8 * motion
            position, detected = _target_position(response, previous_position)
            if detected or previous_position is None:
                previous_position = position
            crop_side = max(48, int(round(min(gray.shape) * 0.20)))
            intensity_crop = _crop(enhanced, position, crop_side)
            contrast_crop = _crop(_normalise_map(contrast), position, crop_side)
            motion_crop = _crop(_normalise_map(motion), position, crop_side)
            channels = [
                cv2.resize(channel, (image_size, image_size), interpolation=cv2.INTER_AREA)
                for channel in (intensity_crop, contrast_crop, motion_crop)
            ]
            frames.append(np.stack(channels, axis=0).astype(np.uint8))
            times.append(float(timestamp))
            activities.append(float(np.quantile(response, 0.995)))
            detections.append(float(detected))
            previous_enhanced = enhanced
        backend = info.get("backend")
        codec = info.get("codec")
        duration = float(info.get("duration_seconds") or 0.0)
        source = info

    if not frames:
        raise ValueError(f"No infrared frames decoded from {path}")
    return {
        "frames": np.stack(frames),
        "times": np.asarray(times, dtype=np.float32),
        "activity": np.asarray(activities, dtype=np.float32),
        "detected": np.asarray(detections, dtype=np.float32),
        "duration": duration or float(times[-1] + 1.0 / sample_fps),
        "metadata": {
            **source,
            "sampled_frames": len(frames),
            "detected_rate": float(np.mean(detections)),
            "decoder_backend": backend,
            "decoder_codec": codec,
        },
    }


def _window_ranges(start: float, end: float, config: dict[str, Any]) -> list[tuple[float, float]]:
    length = float(config["window_seconds"])
    stride = float(config["window_stride_seconds"])
    maximum = int(config["max_windows_per_record"])
    if end <= start:
        return []
    if end - start <= length:
        return [(start, end)]
    result: list[tuple[float, float]] = []
    current = start
    while current + length <= end + 1e-6 and len(result) < maximum:
        result.append((current, current + length))
        current += stride
    if result and result[-1][1] < end - 0.25 * length and len(result) < maximum:
        result.append((max(start, end - length), end))
    return result


def _select_frames(timeline: dict[str, Any], start: float, end: float, count: int) -> tuple[np.ndarray, float, float]:
    times = timeline["times"]
    indices = np.flatnonzero((times >= start) & (times < end))
    if indices.size == 0:
        indices = np.asarray([int(np.argmin(np.abs(times - (start + end) / 2.0)))])
    positions = np.linspace(0, len(indices) - 1, count).round().astype(int)
    selected = indices[positions]
    coverage = min(1.0, len(indices) / max(1.0, (end - start) * max(1e-6, len(times) / timeline["duration"])))
    detected = float(np.mean(timeline["detected"][selected]))
    return timeline["frames"][selected], coverage, detected


def _radar_window(
    sequence: RadarSequence,
    schema: list[str],
    start: float,
    end: float,
    steps: int,
) -> tuple[np.ndarray, float]:
    grid = np.linspace(start, end, steps, endpoint=False, dtype=np.float64)
    output = np.zeros((len(schema), steps), dtype=np.float32)
    finite_time = np.isfinite(sequence.times)
    for channel, name in enumerate(schema):
        if name.startswith("__padding_") or name not in sequence.frame:
            continue
        values = sequence.frame[name].to_numpy(dtype=float)
        finite = finite_time & np.isfinite(values)
        if finite.sum() < 2:
            continue
        output[channel] = np.interp(
            grid,
            sequence.times[finite],
            values[finite],
            left=float(values[finite][0]),
            right=float(values[finite][-1]),
        ).astype(np.float32)
    center = np.median(output, axis=1, keepdims=True)
    scale = np.median(np.abs(output - center), axis=1, keepdims=True) * 1.4826
    scale[scale < 1e-6] = 1.0
    output = np.clip((output - center) / scale, -8.0, 8.0)
    inside = (sequence.times >= start) & (sequence.times < end)
    expected = max(1.0, (end - start) * sequence.rate_hz)
    coverage = min(1.0, float(np.sum(inside)) / expected)
    return output.astype(np.float32), coverage


def preprocess_record(
    sample: Sample,
    config: dict[str, Any],
    schema: list[str],
    cache_dir: Path,
    *,
    rebuild: bool,
) -> CachedRecord:
    key = _record_cache_key(sample, config, schema)
    path = cache_dir / f"{sample.batch_id}-{key}.npz"
    metadata_path = path.with_suffix(".json")
    if path.is_file() and metadata_path.is_file() and not rebuild:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return CachedRecord(
            batch_id=sample.batch_id,
            label=sample.label,
            path=path,
            windows=int(metadata["windows"]),
            alignment=metadata["alignment"],
            radar_metadata=metadata["radar_metadata"],
            infrared_metadata=metadata["infrared_metadata"],
            cache_hit=True,
        )

    timeline = extract_video_timeline(sample.infrared_path, config)
    radar = extract_radar_sequence(
        sample.radar_path,
        duration_hint=float(timeline["duration"]),
        max_numeric_columns=max(96, int(config["radar_channels"]) * 3),
        max_vector_expansion=int(config["radar_vector_expansion"]),
    )
    alignment = estimate_alignment(
        radar.times,
        radar.activity,
        timeline["times"],
        timeline["activity"],
        max_lag_seconds=float(config["max_alignment_lag_seconds"]),
        common_rate_hz=2.0,
    )
    ir_start = max(0.0, alignment.radar_to_ir(0.0))
    ir_end = min(float(timeline["duration"]), alignment.radar_to_ir(radar.duration_seconds))
    if ir_end - ir_start < 1.0:
        ir_start = 0.0
        ir_end = min(float(timeline["duration"]), radar.duration_seconds)
    ranges = _window_ranges(ir_start, ir_end, config)
    radar_windows: list[np.ndarray] = []
    ir_windows: list[np.ndarray] = []
    qualities: list[np.ndarray] = []
    starts: list[float] = []
    ends: list[float] = []
    for ir_window_start, ir_window_end in ranges:
        radar_start = max(0.0, alignment.ir_to_radar(ir_window_start))
        radar_end = min(radar.duration_seconds, alignment.ir_to_radar(ir_window_end))
        if radar_end <= radar_start:
            continue
        radar_tensor, radar_coverage = _radar_window(
            radar,
            schema,
            radar_start,
            radar_end,
            int(config["radar_steps"]),
        )
        ir_tensor, ir_coverage, detected_rate = _select_frames(
            timeline,
            ir_window_start,
            ir_window_end,
            int(config["frames_per_window"]),
        )
        radar_windows.append(radar_tensor.astype(np.float16))
        ir_windows.append(ir_tensor)
        qualities.append(
            np.asarray(
                [
                    radar_coverage,
                    ir_coverage,
                    detected_rate,
                    max(0.0, alignment.score),
                ],
                dtype=np.float32,
            )
        )
        starts.append(ir_window_start)
        ends.append(ir_window_end)
    if not radar_windows:
        raise ValueError(f"No aligned windows were produced for batch {sample.batch_id}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        radar=np.stack(radar_windows),
        infrared=np.stack(ir_windows),
        quality=np.stack(qualities),
        starts=np.asarray(starts, dtype=np.float32),
        ends=np.asarray(ends, dtype=np.float32),
        label=np.asarray(sample.label),
        batch_id=np.asarray(sample.batch_id),
    )
    metadata = {
        "batch_id": sample.batch_id,
        "label": sample.label,
        "windows": len(radar_windows),
        "alignment": alignment.to_dict(),
        "radar_metadata": radar.metadata,
        "infrared_metadata": timeline["metadata"],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return CachedRecord(
        batch_id=sample.batch_id,
        label=sample.label,
        path=path,
        windows=len(radar_windows),
        alignment=alignment.to_dict(),
        radar_metadata=radar.metadata,
        infrared_metadata=timeline["metadata"],
        cache_hit=False,
    )


def preprocess_dataset(
    samples: list[Sample],
    config: dict[str, Any],
    cache_dir: str | Path,
    *,
    rebuild: bool = False,
) -> PreprocessResult:
    started = time.perf_counter()
    cache = Path(cache_dir).expanduser().resolve() / config_hash(config)
    cache.mkdir(parents=True, exist_ok=True)
    schema_path = cache / "radar_schema.json"
    if schema_path.is_file() and not rebuild:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))["columns"]
    else:
        schema, _ = fit_radar_schema(samples, config)
        schema_path.write_text(
            json.dumps({"columns": schema}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    records: list[CachedRecord] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[preprocess {index}/{len(samples)}] batch={sample.batch_id}", flush=True)
        records.append(preprocess_record(sample, config, schema, cache, rebuild=rebuild))
    return PreprocessResult(
        records=records,
        radar_schema=schema,
        elapsed_seconds=time.perf_counter() - started,
        cache_dir=cache,
    )
