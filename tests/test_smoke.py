from pathlib import Path

import numpy as np

from anti_air.alignment import estimate_alignment
from anti_air.dataset import parse_competition_filename


def test_parse_filename() -> None:
    meta = parse_competition_filename(Path("ir_339_class-B_16：18.mp4"))
    assert meta["modality"] == "ir"
    assert meta["batch_id"] == "339"
    assert meta["label"] == "class-B"
    assert meta["start_time"] == "16:18"


def test_alignment_recovers_positive_lag() -> None:
    radar = np.zeros(100)
    radar[20:30] = 1.0
    infrared = np.zeros(100)
    infrared[25:35] = 1.0
    result = estimate_alignment(
        radar,
        infrared,
        radar_rate_hz=10.0,
        infrared_rate_hz=10.0,
        common_rate_hz=10.0,
        max_lag_seconds=2.0,
    )
    assert abs(result.offset_seconds - 0.5) <= 0.11
