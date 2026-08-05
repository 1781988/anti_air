from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

from .feature_store import FeatureTables
from .modeling import aggregate_record_probabilities, fit_model_bundle, predict_window_probabilities


@dataclass
class EvaluationResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    window_predictions: pd.DataFrame
    confusion: pd.DataFrame
    folds: list[dict[str, Any]]


def subset_tables(tables: FeatureTables, batch_ids: Iterable[str]) -> FeatureTables:
    selected = set(map(str, batch_ids))
    return FeatureTables(
        radar=tables.radar[tables.radar["batch_id"].astype(str).isin(selected)].reset_index(drop=True),
        infrared=tables.infrared[tables.infrared["batch_id"].astype(str).isin(selected)].reset_index(drop=True),
        fusion=tables.fusion[tables.fusion["batch_id"].astype(str).isin(selected)].reset_index(drop=True),
        manifest=tables.manifest,
    )


def _record_table(tables: FeatureTables) -> pd.DataFrame:
    records = tables.fusion[["batch_id", "label"]].drop_duplicates().copy()
    records["batch_id"] = records["batch_id"].astype(str)
    records["label"] = records["label"].astype(str)
    if records["batch_id"].duplicated().any():
        raise ValueError("A batch_id maps to multiple labels")
    return records.reset_index(drop=True)


def _fold_plan(
    records: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[tuple[np.ndarray, np.ndarray, str]], list[dict[str, Any]]]:
    labels = records["label"].to_numpy()
    class_counts = records["label"].value_counts()
    requested = int(config["evaluation"]["n_splits"])
    seed = int(config["seed"])
    valid: list[tuple[np.ndarray, np.ndarray, str]] = []
    diagnostics: list[dict[str, Any]] = []

    if len(records) >= 4 and int(class_counts.min()) >= 2:
        n_splits = min(requested, int(class_counts.min()), len(records))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        candidates = [
            (train_index, test_index, "stratified_group")
            for train_index, test_index in splitter.split(records, labels)
        ]
    else:
        splitter = LeaveOneOut()
        candidates = [
            (train_index, test_index, "leave_one_batch")
            for train_index, test_index in splitter.split(records)
        ]

    for attempted_fold, (train_index, test_index, strategy) in enumerate(candidates):
        train_classes = set(labels[train_index])
        test_classes = set(labels[test_index])
        reason: str | None = None
        if len(train_classes) < 2:
            reason = "training_fold_has_fewer_than_two_classes"
        elif not test_classes.issubset(train_classes):
            reason = "test_class_absent_from_training_fold"
        retained = reason is None
        diagnostics.append(
            {
                "attempted_fold": attempted_fold,
                "strategy": strategy,
                "retained": retained,
                "skip_reason": reason,
                "train_batches": records.iloc[train_index]["batch_id"].tolist(),
                "test_batches": records.iloc[test_index]["batch_id"].tolist(),
                "train_classes": sorted(train_classes),
                "test_classes": sorted(test_classes),
            }
        )
        if retained:
            valid.append((train_index, test_index, strategy))

    return valid, diagnostics


def _bootstrap_macro_f1(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    seed: int,
    minimum_records: int,
    full_class_coverage: bool,
    iterations: int = 1000,
) -> dict[str, float | str | None]:
    if not full_class_coverage:
        return {
            "lower": None,
            "upper": None,
            "reason": "not_computed_without_full_class_coverage",
        }
    if len(truth) < minimum_records:
        return {
            "lower": None,
            "upper": None,
            "reason": f"requires_at_least_{minimum_records}_independent_records",
        }
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(truth), size=len(truth))
        scores.append(float(f1_score(truth[indices], prediction[indices], average="macro", zero_division=0)))
    return {
        "lower": float(np.quantile(scores, 0.025)),
        "upper": float(np.quantile(scores, 0.975)),
        "reason": None,
    }


def _aggregate_branch_records(
    windows: pd.DataFrame,
    classes: list[str],
    branch: str,
) -> pd.DataFrame:
    prefix = f"{branch}_prob__"
    rows: list[dict[str, Any]] = []
    columns = [f"{prefix}{label}" for label in classes]
    for batch_id, group in windows.groupby("batch_id", sort=False):
        probability = group[columns].to_numpy(dtype=float)
        entropy = -np.sum(probability * np.log(probability + 1e-12), axis=1) / max(np.log(len(classes)), 1.0)
        weights = np.clip(1.0 - entropy, 0.1, 1.0)
        mean_probability = np.average(probability, axis=0, weights=weights)
        mean_probability = mean_probability / max(float(mean_probability.sum()), 1e-12)
        best = int(np.argmax(mean_probability))
        row: dict[str, Any] = {
            "batch_id": str(batch_id),
            "label": str(group["label"].dropna().iloc[0]) if group["label"].notna().any() else None,
            "predicted_label": classes[best],
            "confidence": float(mean_probability[best]),
        }
        for index, label in enumerate(classes):
            row[f"prob__{label}"] = float(mean_probability[index])
        rows.append(row)
    return pd.DataFrame(rows)


def _score_predictions(
    predictions: pd.DataFrame,
    classes: list[str],
) -> dict[str, Any]:
    if predictions.empty:
        return {"records": 0}
    truth = predictions["label"].astype(str).to_numpy()
    predicted = predictions["predicted_label"].astype(str).to_numpy()
    probability_columns = [f"prob__{label}" for label in classes]
    for column in probability_columns:
        if column not in predictions:
            predictions[column] = 0.0
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    result: dict[str, Any] = {
        "records": int(len(predictions)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(recall_score(truth, predicted, average="macro", labels=classes, zero_division=0)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", labels=classes, zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
    }
    try:
        result["log_loss"] = float(log_loss(truth, probabilities, labels=classes))
    except ValueError:
        result["log_loss"] = None
    return result


def _merge_branch_predictions(
    ensemble: pd.DataFrame,
    branch_records: dict[str, pd.DataFrame],
    classes: list[str],
) -> pd.DataFrame:
    output = ensemble.copy()
    for branch, frame in branch_records.items():
        rename = {
            "predicted_label": f"{branch}_predicted_label",
            "confidence": f"{branch}_confidence",
            **{f"prob__{label}": f"{branch}_prob__{label}" for label in classes},
        }
        selected = frame[["batch_id", *rename.keys()]].rename(columns=rename)
        output = output.merge(selected, on="batch_id", how="left", validate="one_to_one")
    return output


def evaluate_grouped_cv(tables: FeatureTables, config: dict[str, Any]) -> EvaluationResult:
    records = _record_table(tables)
    classes = sorted(records["label"].unique().tolist())
    folds, fold_diagnostics = _fold_plan(records, config)
    record_predictions: list[pd.DataFrame] = []
    window_predictions: list[pd.DataFrame] = []
    branch_predictions: dict[str, list[pd.DataFrame]] = {
        "radar": [],
        "infrared": [],
        "fusion": [],
    }

    retained_index = 0
    for train_index, test_index, strategy in folds:
        train_batches = records.iloc[train_index]["batch_id"].tolist()
        test_batches = records.iloc[test_index]["batch_id"].tolist()
        train_tables = subset_tables(tables, train_batches)
        test_tables = subset_tables(tables, test_batches)
        model = fit_model_bundle(train_tables, config)
        windows = predict_window_probabilities(model, test_tables)
        ensemble = aggregate_record_probabilities(windows, model.classes)
        ensemble["fold"] = retained_index
        windows["fold"] = retained_index
        train_majority = records.iloc[train_index]["label"].value_counts().sort_values(ascending=False).index[0]
        ensemble["majority_baseline_label"] = str(train_majority)
        record_predictions.append(ensemble)
        window_predictions.append(windows)
        for branch in branch_predictions:
            branch_frame = _aggregate_branch_records(windows, model.classes, branch)
            branch_frame["fold"] = retained_index
            branch_predictions[branch].append(branch_frame)
        retained_index += 1

    class_record_counts = records["label"].value_counts().sort_index().to_dict()
    if not record_predictions:
        empty = pd.DataFrame(columns=["batch_id", "label", "predicted_label"])
        confusion = pd.DataFrame(0, index=classes, columns=classes)
        return EvaluationResult(
            metrics={
                "status": "insufficient_grouped_data",
                "eligible_for_primary_score": False,
                "metric_interpretation": "not_available",
                "reason": "No leakage-safe fold retained every test class in the training set.",
                "records": int(len(records)),
                "classes": classes,
                "class_record_counts": class_record_counts,
                "attempted_folds": len(fold_diagnostics),
                "valid_folds": 0,
                "evaluated_records": 0,
                "total_records": int(len(records)),
                "evaluation_coverage": 0.0,
                "evaluated_classes": [],
                "unevaluated_classes": classes,
                "class_coverage": 0.0,
                "unevaluated_batches": records["batch_id"].tolist(),
                "branch_metrics": {},
            },
            predictions=empty,
            window_predictions=empty,
            confusion=confusion,
            folds=fold_diagnostics,
        )

    predictions = pd.concat(record_predictions, ignore_index=True)
    windows = pd.concat(window_predictions, ignore_index=True)
    branch_records = {
        branch: pd.concat(frames, ignore_index=True)
        for branch, frames in branch_predictions.items()
    }
    predictions = _merge_branch_predictions(predictions, branch_records, classes)

    truth = predictions["label"].astype(str).to_numpy()
    predicted = predictions["predicted_label"].astype(str).to_numpy()
    evaluated_classes = sorted(set(truth))
    evaluated_batches = set(predictions["batch_id"].astype(str))
    total_batches = set(records["batch_id"].astype(str))
    unevaluated_classes = sorted(set(classes) - set(evaluated_classes))
    unevaluated_batches = sorted(total_batches - evaluated_batches)
    record_coverage = len(evaluated_batches) / max(len(total_batches), 1)
    class_coverage = len(evaluated_classes) / max(len(classes), 1)
    minimum_folds = int(config["evaluation"].get("min_valid_folds", 2))

    if class_coverage < 1.0:
        status = "insufficient_class_coverage"
        reason = "At least one class has no leakage-safe held-out prediction. Metrics are diagnostic only."
    elif record_coverage < 1.0:
        status = "partial_record_coverage"
        reason = "Not every independent batch could be evaluated without class leakage."
    elif len(folds) < minimum_folds:
        status = "limited_folds"
        reason = "The number of valid grouped folds is below the configured minimum."
    else:
        status = "ok"
        reason = None
    eligible = status == "ok"

    probability_columns = [f"prob__{label}" for label in classes]
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    matrix = confusion_matrix(truth, predicted, labels=classes)
    confusion = pd.DataFrame(matrix, index=classes, columns=classes)
    report = classification_report(truth, predicted, labels=classes, output_dict=True, zero_division=0)
    ensemble_metrics = _score_predictions(predictions.copy(), classes)
    branch_metrics = {
        branch: _score_predictions(frame.copy(), classes)
        for branch, frame in branch_records.items()
    }

    baseline_prediction = predictions["majority_baseline_label"].astype(str).to_numpy()
    baseline = {
        "strategy": "training_fold_majority_class",
        "accuracy": float(accuracy_score(truth, baseline_prediction)),
        "macro_f1": float(f1_score(truth, baseline_prediction, average="macro", labels=classes, zero_division=0)),
    }
    minimum_ci_records = int(config["evaluation"].get("min_records_for_confidence_interval", 10))
    metrics: dict[str, Any] = {
        "status": status,
        "eligible_for_primary_score": eligible,
        "metric_interpretation": "primary" if eligible else "diagnostic_only",
        "reason": reason,
        "attempted_folds": len(fold_diagnostics),
        "valid_folds": len(folds),
        "evaluated_records": int(len(evaluated_batches)),
        "total_records": int(len(total_batches)),
        "evaluation_coverage": float(record_coverage),
        "evaluated_classes": evaluated_classes,
        "unevaluated_classes": unevaluated_classes,
        "class_coverage": float(class_coverage),
        "unevaluated_batches": unevaluated_batches,
        "classes": classes,
        "class_record_counts": class_record_counts,
        **{key: value for key, value in ensemble_metrics.items() if key != "records"},
        "classification_report": report,
        "macro_f1_95ci": _bootstrap_macro_f1(
            truth,
            predicted,
            seed=int(config["seed"]),
            minimum_records=minimum_ci_records,
            full_class_coverage=class_coverage == 1.0,
        ),
        "branch_metrics": branch_metrics,
        "majority_baseline": baseline,
    }
    return EvaluationResult(metrics, predictions, windows, confusion, fold_diagnostics)
