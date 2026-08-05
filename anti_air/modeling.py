from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .feature_store import FeatureTables, feature_columns


@dataclass
class BranchArtifact:
    model: Any
    columns: list[str]
    classes: list[str]


@dataclass
class ModelBundle:
    version: str
    config: dict[str, Any]
    classes: list[str]
    radar: BranchArtifact
    infrared: BranchArtifact
    fusion: BranchArtifact
    training_summary: dict[str, Any]


def _make_classifier(config: dict[str, Any], *, seed: int, n_jobs: int) -> Pipeline:
    model_config = config["model"]
    common = dict(
        n_estimators=int(model_config["n_estimators"]),
        min_samples_leaf=int(model_config["min_samples_leaf"]),
        max_features=model_config.get("max_features", "sqrt"),
        class_weight=model_config.get("class_weight", "balanced"),
        random_state=seed,
        n_jobs=n_jobs,
    )
    estimator_name = str(model_config.get("estimator", "extra_trees")).lower()
    if estimator_name == "random_forest":
        classifier = RandomForestClassifier(**common)
    elif estimator_name == "extra_trees":
        classifier = ExtraTreesClassifier(**common)
    else:
        raise ValueError(f"Unsupported estimator: {estimator_name}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("classifier", classifier),
        ]
    )


def _balanced_record_weights(groups: pd.Series) -> np.ndarray:
    counts = groups.value_counts().to_dict()
    weights = groups.map(lambda group: 1.0 / max(int(counts[group]), 1)).to_numpy(dtype=float)
    return weights * len(weights) / max(float(np.sum(weights)), 1e-12)


def fit_branch(
    frame: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int,
    n_jobs: int,
) -> BranchArtifact:
    columns = feature_columns(frame)
    if not columns:
        raise ValueError("No feature columns available for model training")
    labels = frame["label"].astype(str)
    if labels.nunique() < 2:
        raise ValueError("At least two classes are required for training")
    model = _make_classifier(config, seed=seed, n_jobs=n_jobs)
    weights = _balanced_record_weights(frame["batch_id"].astype(str))
    model.fit(frame[columns], labels, classifier__sample_weight=weights)
    classes = [str(value) for value in model.named_steps["classifier"].classes_]
    return BranchArtifact(model=model, columns=columns, classes=classes)


def _effective_probability_smoothing(
    config: dict[str, Any],
    class_record_counts: dict[str, int],
) -> tuple[float, float]:
    model_config = config["model"]
    base = max(0.0, float(model_config.get("probability_smoothing", 0.0)))
    extra_max = max(0.0, float(model_config.get("small_sample_probability_smoothing", 0.0)))
    target = max(1, int(model_config.get("confidence_record_target_per_class", 5)))
    minimum = min(class_record_counts.values()) if class_record_counts else 0
    reliability = min(1.0, minimum / target)
    effective = min(0.95, base + extra_max * (1.0 - reliability))
    return effective, reliability


def fit_model_bundle(tables: FeatureTables, config: dict[str, Any]) -> ModelBundle:
    seed = int(config["seed"])
    n_jobs = int(config.get("runtime", {}).get("n_jobs", -1))
    classes = sorted(tables.fusion["label"].astype(str).unique().tolist())
    class_record_counts = (
        tables.fusion[["batch_id", "label"]]
        .drop_duplicates()["label"]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    effective_smoothing, record_reliability = _effective_probability_smoothing(config, class_record_counts)
    summary = {
        "records": int(tables.fusion["batch_id"].nunique()),
        "windows": int(len(tables.fusion)),
        "classes": classes,
        "class_record_counts": class_record_counts,
        "class_window_counts": tables.fusion["label"].astype(str).value_counts().sort_index().to_dict(),
        "minimum_records_per_class": int(min(class_record_counts.values())) if class_record_counts else 0,
        "record_level_confidence_reliability": float(record_reliability),
        "effective_probability_smoothing": float(effective_smoothing),
        "training_data_warning": (
            "Independent record counts are below the configured confidence target; "
            "probabilities are regularized and must not be interpreted as calibrated risk."
            if record_reliability < 1.0
            else None
        ),
    }
    return ModelBundle(
        version="1.1.0",
        config=config,
        classes=classes,
        radar=fit_branch(tables.radar, config, seed=seed + 11, n_jobs=n_jobs),
        infrared=fit_branch(tables.infrared, config, seed=seed + 23, n_jobs=n_jobs),
        fusion=fit_branch(tables.fusion, config, seed=seed + 37, n_jobs=n_jobs),
        training_summary=summary,
    )


def _aligned_probabilities(artifact: BranchArtifact, frame: pd.DataFrame, classes: list[str]) -> np.ndarray:
    probabilities = artifact.model.predict_proba(frame.reindex(columns=artifact.columns))
    output = np.zeros((len(frame), len(classes)), dtype=float)
    class_index = {label: index for index, label in enumerate(classes)}
    for source_index, label in enumerate(artifact.classes):
        if label in class_index:
            output[:, class_index[label]] = probabilities[:, source_index]
    return output


def predict_window_probabilities(bundle: ModelBundle, tables: FeatureTables) -> pd.DataFrame:
    order = tables.fusion["window_id"].astype(str).tolist()
    radar = tables.radar.set_index("window_id").loc[order].reset_index()
    infrared = tables.infrared.set_index("window_id").loc[order].reset_index()
    fusion = tables.fusion.set_index("window_id").loc[order].reset_index()

    radar_probability = _aligned_probabilities(bundle.radar, radar, bundle.classes)
    infrared_probability = _aligned_probabilities(bundle.infrared, infrared, bundle.classes)
    fusion_probability = _aligned_probabilities(bundle.fusion, fusion, bundle.classes)

    branch_weights = bundle.config["model"]["branch_weights"]
    radar_quality = np.clip(fusion["radar_quality"].to_numpy(dtype=float), 0.02, 1.0)
    infrared_quality = np.clip(fusion["infrared_quality"].to_numpy(dtype=float), 0.02, 1.0)
    sync_score = pd.to_numeric(
        fusion.get("sync__score", pd.Series(np.zeros(len(fusion)))), errors="coerce"
    ).fillna(0.0)
    sync_quality = np.clip((sync_score.to_numpy(dtype=float) + 1.0) / 2.0, 0.05, 1.0)

    wr = float(branch_weights.get("radar", 0.30)) * radar_quality
    wi = float(branch_weights.get("infrared", 0.15)) * infrared_quality
    wf = float(branch_weights.get("fusion", 0.55)) * np.sqrt(radar_quality * infrared_quality) * sync_quality
    weight_sum = np.maximum(wr + wi + wf, 1e-12)
    wr, wi, wf = wr / weight_sum, wi / weight_sum, wf / weight_sum
    probability = wr[:, None] * radar_probability + wi[:, None] * infrared_probability + wf[:, None] * fusion_probability

    smoothing = float(
        bundle.training_summary.get(
            "effective_probability_smoothing",
            bundle.config["model"].get("probability_smoothing", 0.0),
        )
    )
    if smoothing > 0:
        probability = (1.0 - smoothing) * probability + smoothing / len(bundle.classes)
    probability = probability / np.maximum(probability.sum(axis=1, keepdims=True), 1e-12)

    output = fusion[
        [
            column
            for column in ["window_id", "batch_id", "label", "ir_start_seconds", "ir_end_seconds"]
            if column in fusion
        ]
    ].copy()
    for index, label in enumerate(bundle.classes):
        output[f"prob__{label}"] = probability[:, index]
        output[f"radar_prob__{label}"] = radar_probability[:, index]
        output[f"infrared_prob__{label}"] = infrared_probability[:, index]
        output[f"fusion_prob__{label}"] = fusion_probability[:, index]
    output["weight__radar"] = wr
    output["weight__infrared"] = wi
    output["weight__fusion"] = wf
    output["effective_probability_smoothing"] = smoothing
    output["window_confidence"] = probability.max(axis=1)
    output["predicted_label"] = [bundle.classes[index] for index in probability.argmax(axis=1)]
    return output


def aggregate_record_probabilities(window_predictions: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch_id, group in window_predictions.groupby("batch_id", sort=False):
        probabilities = group[[f"prob__{label}" for label in classes]].to_numpy(dtype=float)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1) / max(np.log(len(classes)), 1.0)
        confidence_weight = np.clip(1.0 - entropy, 0.1, 1.0)
        mean_probability = np.average(probabilities, axis=0, weights=confidence_weight)
        mean_probability = mean_probability / max(float(mean_probability.sum()), 1e-12)
        best = int(np.argmax(mean_probability))
        row: dict[str, Any] = {
            "batch_id": str(batch_id),
            "label": str(group["label"].dropna().iloc[0]) if "label" in group and group["label"].notna().any() else None,
            "predicted_label": classes[best],
            "confidence": float(mean_probability[best]),
            "window_count": int(len(group)),
        }
        for index, label in enumerate(classes):
            row[f"prob__{label}"] = float(mean_probability[index])
        rows.append(row)
    return pd.DataFrame(rows)
