from __future__ import annotations

import pandas as pd

from anti_air.evaluation import evaluate_grouped_cv
from anti_air.feature_store import FeatureTables


def _tables() -> FeatureTables:
    rows = []
    labels = {"A1": "class-A", "A2": "class-A", "B1": "class-B"}
    values = {"A1": 0.0, "A2": 1.0, "B1": 10.0}
    for batch_id, label in labels.items():
        for window in range(2):
            rows.append(
                {
                    "window_id": f"{batch_id}-{window}",
                    "batch_id": batch_id,
                    "label": label,
                    "ir_start_seconds": float(window),
                    "ir_end_seconds": float(window + 1),
                    "radar_start_seconds": float(window),
                    "radar_end_seconds": float(window + 1),
                    "radar_quality": 1.0,
                    "infrared_quality": 1.0,
                    "sync__score": 1.0,
                    "feature": values[batch_id] + window * 0.01,
                }
            )
    frame = pd.DataFrame(rows)
    return FeatureTables(
        radar=frame.copy(),
        infrared=frame.copy(),
        fusion=frame.copy(),
        manifest={},
    )


def test_partial_class_coverage_is_not_reported_as_ok() -> None:
    config = {
        "seed": 7,
        "runtime": {"n_jobs": 1},
        "model": {
            "estimator": "extra_trees",
            "n_estimators": 10,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "probability_smoothing": 0.0,
            "small_sample_probability_smoothing": 0.0,
            "confidence_record_target_per_class": 5,
            "branch_weights": {"radar": 0.3, "infrared": 0.2, "fusion": 0.5},
        },
        "evaluation": {
            "n_splits": 3,
            "min_valid_folds": 1,
            "min_records_for_confidence_interval": 10,
        },
    }
    result = evaluate_grouped_cv(_tables(), config)
    assert result.metrics["status"] == "insufficient_class_coverage"
    assert result.metrics["eligible_for_primary_score"] is False
    assert result.metrics["class_coverage"] == 0.5
    assert result.metrics["unevaluated_classes"] == ["class-B"]
    assert result.metrics["macro_f1_95ci"]["lower"] is None
    assert set(result.metrics["branch_metrics"]) == {"radar", "infrared", "fusion"}
    assert "radar_predicted_label" in result.predictions
