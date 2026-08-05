from __future__ import annotations

import time
from collections import Counter
from typing import Any

from .data import Sample
from .radar import extract_radar_sequence
from .video_io import VideoReader


def inspect_samples(samples: list[Sample], config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    items: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, sample in enumerate(samples, start=1):
        print(f"[inspect {index}/{len(samples)}] batch={sample.batch_id}", flush=True)
        item = sample.to_dict()
        try:
            with VideoReader(sample.infrared_path) as reader:
                video = reader.metadata()
            radar = extract_radar_sequence(
                sample.radar_path,
                duration_hint=float(video["duration_seconds"]) or None,
                max_numeric_columns=max(96, int(config["radar_channels"]) * 3),
                max_vector_expansion=int(config["radar_vector_expansion"]),
            )
            item["infrared"] = video
            item["radar"] = {
                **radar.metadata,
                "columns": list(radar.frame.columns),
                "rate_hz": radar.rate_hz,
                "duration_seconds": radar.duration_seconds,
            }
        except Exception as exc:
            item["error"] = repr(exc)
            errors.append({"batch_id": sample.batch_id, "error": repr(exc)})
        items.append(item)
    return {
        "status": "ok" if not errors else "error",
        "sample_count": len(samples),
        "class_record_counts": dict(sorted(Counter(sample.label for sample in samples).items())),
        "errors": errors,
        "items": items,
        "elapsed_seconds": time.perf_counter() - started,
    }
