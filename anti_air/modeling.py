from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass
class BranchArtifact:
    model: Any
    columns: list[str]


def make_classifier(*, seed: int, n_estimators: int, min_samples_leaf: int, class_weight: str) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    class_weight=class_weight,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_branch(frame: pd.DataFrame, labels: list[str], **classifier_kwargs: Any) -> BranchArtifact:
    model = make_classifier(**classifier_kwargs)
    columns = sorted(frame.columns)
    model.fit(frame.reindex(columns=columns), labels)
    return BranchArtifact(model=model, columns=columns)


def predict_branch(artifact: BranchArtifact, features: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    x = pd.DataFrame([features]).reindex(columns=artifact.columns)
    proba = artifact.model.predict_proba(x)[0]
    classes = artifact.model.named_steps["classifier"].classes_
    return np.asarray(classes), np.asarray(proba, dtype=float)


def quality_aware_fusion(
    radar_classes: np.ndarray,
    radar_proba: np.ndarray,
    ir_classes: np.ndarray,
    ir_proba: np.ndarray,
    *,
    radar_quality: float,
    infrared_quality: float,
    base_radar_weight: float,
    base_infrared_weight: float,
) -> tuple[list[str], np.ndarray, dict[str, float]]:
    classes = sorted(set(map(str, radar_classes)) | set(map(str, ir_classes)))
    radar_map = {str(c): float(p) for c, p in zip(radar_classes, radar_proba, strict=True)}
    ir_map = {str(c): float(p) for c, p in zip(ir_classes, ir_proba, strict=True)}

    wr = max(1e-6, base_radar_weight * max(0.05, min(1.0, radar_quality)))
    wi = max(1e-6, base_infrared_weight * max(0.05, min(1.0, infrared_quality)))
    normalizer = wr + wi
    wr, wi = wr / normalizer, wi / normalizer

    fused = np.asarray([wr * radar_map.get(c, 0.0) + wi * ir_map.get(c, 0.0) for c in classes])
    fused = fused / max(float(np.sum(fused)), 1e-12)
    return classes, fused, {"radar": float(wr), "infrared": float(wi)}
