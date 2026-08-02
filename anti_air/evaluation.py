from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
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


def _fold_indices(records: pd.DataFrame, config: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray, str]]:
    labels = records["label"].to_numpy()
    class_counts = records["label"].value_counts()
    requested = int(config["evaluation"]["n_splits"])
    seed = int(config["seed"])
    folds: list[tuple[np.ndarray, np.ndarray, str]] = []
    if len(records) >= 4 and int(class_counts.min()) >= 2:
        n_splits = min(requested, int(class_counts.min()), len(records))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_index, test_index in splitter.split(records, labels):
            folds.append((train_index, test_index, "stratified_group"))
    else:
        splitter = LeaveOneOut()
        for train_index, test_index in splitter.split(records):
            train_classes = set(labels[train_index])
            test_classes = set(labels[test_index])
            if len(train_classes) < 2 or not test_classes.issubset(train_classes):
                continue
            folds.append((train_index, test_index, "leave_one_batch_partial"))
        if not folds and len(records) >= 2:
            for test_index in np.array_split(np.arange(len(records)), min(2, len(records))):
                train_index = np.asarray([idx for idx in range(len(records)) if idx not in set(test_index)])
                if len(set(labels[train_index])) >= 2 and set(labels[test_index]).issubset(set(labels[train_index])):
                    folds.append((train_index, np.asarray(test_index), "group_holdout_partial"))
    return folds


def _bootstrap_macro_f1(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    seed: int,
    iterations: int = 1000,
) -> dict[str, float | None]:
    if len(truth) < 2:
        return {"lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(truth), size=len(truth))
        scores.append(float(f1_score(truth[indices], prediction[indices], average="macro", zero_division=0)))
    return {"lower": float(np.quantile(scores, 0.025)), "upper": float(np.quantile(scores, 0.975))}


def evaluate_grouped_cv(tables: FeatureTables, config: dict[str, Any]) -> EvaluationResult:
    records = _record_table(tables)
    classes = sorted(records["label"].unique().tolist())
    folds = _fold_indices(records, config)
    fold_records: list[dict[str, Any]] = []
    record_predictions: list[pd.DataFrame] = []
    window_predictions: list[pd.DataFrame] = []

    for fold_index, (train_index, test_index, strategy) in enumerate(folds):
        train_batches = records.iloc[train_index]["batch_id"].tolist()
        test_batches = records.iloc[test_index]["batch_id"].tolist()
        train_tables = subset_tables(tables, train_batches)
        test_tables = subset_tables(tables, test_batches)
        model = fit_model_bundle(train_tables, config)
        window = predict_window_probabilities(model, test_tables)
        record = aggregate_record_probabilities(window, model.classes)
        record["fold"] = fold_index
        window["fold"] = fold_index
        record_predictions.append(record)
        window_predictions.append(window)
        fold_records.append(
            {
                "fold": fold_index,
                "strategy": strategy,
                "train_batches": train_batches,
                "test_batches": test_batches,
                "train_classes": sorted(set(records.iloc[train_index]["label"])),
                "test_classes": sorted(set(records.iloc[test_index]["label"])),
            }
        )

    minimum_folds = int(config["evaluation"].get("min_valid_folds", 2))
    if not record_predictions:
        empty = pd.DataFrame(columns=["batch_id", "label", "predicted_label"])
        confusion = pd.DataFrame(0, index=classes, columns=classes)
        return EvaluationResult(
            metrics={
                "status": "insufficient_grouped_data",
                "reason": "No leakage-safe fold retained every test class in the training set.",
                "records": len(records),
                "classes": classes,
                "class_record_counts": records["label"].value_counts().sort_index().to_dict(),
                "valid_folds": 0,
            },
            predictions=empty,
            window_predictions=empty,
            confusion=confusion,
            folds=[],
        )

    predictions = pd.concat(record_predictions, ignore_index=True)
    windows = pd.concat(window_predictions, ignore_index=True)
    truth = predictions["label"].astype(str).to_numpy()
    predicted = predictions["predicted_label"].astype(str).to_numpy()
    probability_columns = [f"prob__{label}" for label in classes]
    for column in probability_columns:
        if column not in predictions:
            predictions[column] = 0.0
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    probabilities = probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)

    matrix = confusion_matrix(truth, predicted, labels=classes)
    confusion = pd.DataFrame(matrix, index=classes, columns=classes)
    report = classification_report(truth, predicted, labels=classes, output_dict=True, zero_division=0)
    metrics: dict[str, Any] = {
        "status": "ok" if len(fold_records) >= minimum_folds else "limited_folds",
        "valid_folds": len(fold_records),
        "evaluated_records": len(predictions),
        "total_records": len(records),
        "evaluation_coverage": len(predictions) / max(len(records), 1),
        "classes": classes,
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "classification_report": report,
        "macro_f1_95ci": _bootstrap_macro_f1(truth, predicted, seed=int(config["seed"])),
    }
    try:
        metrics["log_loss"] = float(log_loss(truth, probabilities, labels=classes))
    except ValueError:
        metrics["log_loss"] = None
    return EvaluationResult(metrics, predictions, windows, confusion, fold_records)
