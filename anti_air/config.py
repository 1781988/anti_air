from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 2026,
    "paths": {
        "data": "data/train",
        "cache": ".cache/anti_air",
        "run": "runs/latest",
    },
    "profile": "auto",
    "common": {
        "window_seconds": 12.0,
        "window_stride_seconds": 6.0,
        "max_windows_per_record": 240,
        "max_alignment_lag_seconds": 45.0,
        "radar_channels": 32,
        "radar_steps": 128,
        "radar_vector_expansion": 8,
        "vision_backbone": "mobilenet_v3_small",
        "pretrained_vision": True,
        "num_workers": 2,
        "label_smoothing": 0.05,
        "weight_decay": 1.0e-4,
        "learning_rate": 3.0e-4,
        "backbone_learning_rate": 3.0e-5,
        "early_stop_patience": 5,
        "evaluation_folds": 5,
    },
    "profiles": {
        "quick": {
            "image_size": 96,
            "frames_per_window": 4,
            "video_sample_fps": 1.0,
            "max_video_samples": 500,
            "epochs": 2,
            "freeze_backbone_epochs": 2,
            "batch_size": 8,
        },
        "cpu": {
            "image_size": 112,
            "frames_per_window": 6,
            "video_sample_fps": 1.5,
            "max_video_samples": 1600,
            "epochs": 8,
            "freeze_backbone_epochs": 8,
            "batch_size": 8,
        },
        "competition": {
            "image_size": 160,
            "frames_per_window": 12,
            "video_sample_fps": 3.0,
            "max_video_samples": 5000,
            "epochs": 24,
            "freeze_backbone_epochs": 4,
            "batch_size": 12,
        },
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path = "config.yaml", *, profile: str | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    source = Path(path)
    if source.is_file():
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Configuration root must be a mapping: {source}")
        config = _deep_update(config, payload)

    requested = profile or str(config.get("profile", "auto"))
    if requested == "auto":
        requested = "competition" if torch.cuda.is_available() else "cpu"
    profiles = config.get("profiles", {})
    if requested not in profiles:
        raise ValueError(f"Unknown profile {requested!r}; choices={sorted(profiles)}")

    resolved = _deep_update(config.get("common", {}), profiles[requested])
    resolved.update(
        {
            "seed": int(config["seed"]),
            "profile": requested,
            "paths": copy.deepcopy(config["paths"]),
        }
    )
    validate_config(resolved)
    return resolved


def validate_config(config: dict[str, Any]) -> None:
    positive = [
        "window_seconds",
        "window_stride_seconds",
        "radar_channels",
        "radar_steps",
        "image_size",
        "frames_per_window",
        "video_sample_fps",
        "epochs",
        "batch_size",
    ]
    for key in positive:
        if float(config[key]) <= 0:
            raise ValueError(f"Configuration value {key} must be positive")
    if int(config["frames_per_window"]) < 2:
        raise ValueError("frames_per_window must be at least 2")
    if int(config["radar_channels"]) > 256:
        raise ValueError("radar_channels is unexpectedly large")


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
