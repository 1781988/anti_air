from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if path is not None:
        custom_path = Path(path)
        with custom_path.open("r", encoding="utf-8") as handle:
            custom = yaml.safe_load(handle) or {}
        config = _deep_merge(config, custom)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    ir = config["infrared"]
    radar = config["radar"]
    window = config["window"]
    model = config["model"]
    if float(ir["sample_fps"]) <= 0:
        raise ValueError("infrared.sample_fps must be positive")
    if int(ir["resize_width"]) < 0:
        raise ValueError("infrared.resize_width must be non-negative")
    if float(radar["target_rate_hz"]) <= 0:
        raise ValueError("radar.target_rate_hz must be positive")
    if float(window["length_seconds"]) <= 0 or float(window["stride_seconds"]) <= 0:
        raise ValueError("window length and stride must be positive")
    weights = model["branch_weights"]
    if sum(max(0.0, float(v)) for v in weights.values()) <= 0:
        raise ValueError("At least one model branch weight must be positive")
