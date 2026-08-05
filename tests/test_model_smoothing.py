from __future__ import annotations

from anti_air.modeling import _effective_probability_smoothing


def test_small_sample_smoothing_increases_when_one_record_per_class() -> None:
    config = {
        "model": {
            "probability_smoothing": 0.02,
            "small_sample_probability_smoothing": 0.15,
            "confidence_record_target_per_class": 5,
        }
    }
    smoothing, reliability = _effective_probability_smoothing(
        config,
        {"class-A": 2, "class-B": 1},
    )
    assert reliability == 0.2
    assert abs(smoothing - 0.14) < 1e-12
