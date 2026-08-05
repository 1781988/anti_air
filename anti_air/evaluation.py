from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .preprocess import CachedRecord
from .torch_data import WindowDataset, make_refs
from .trainer import load_checkpoint, resolve_device, train_model


def _record_probabilities(
    model: torch.nn.Module,
    records: list[CachedRecord],
    classes: list[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    label_to_index = {label: index for index, label in enumerate(classes)}
    refs = make_refs(records)
    dataset = WindowDataset(refs, label_to_index, augment=False, seed=int(config["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    device = next(model.parameters()).device
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    gates: dict[str, list[np.ndarray]] = defaultdict(list)
    labels: dict[str, str] = {}
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["radar"].to(device),
                batch["infrared"].to(device),
                batch["quality"].to(device),
            )
            probabilities = torch.softmax(output["logits"], dim=-1).cpu().numpy()
            gate_values = output["gates"].cpu().numpy()
            for batch_id, target, probability, gate in zip(
                batch["batch_id"],
                batch["label"].tolist(),
                probabilities,
                gate_values,
            ):
                batch_id = str(batch_id)
                grouped[batch_id].append(probability)
                gates[batch_id].append(gate)
                labels[batch_id] = classes[int(target)]
    result: list[dict[str, Any]] = []
    for batch_id in sorted(grouped):
        probability = np.mean(np.stack(grouped[batch_id]), axis=0)
        probability /= max(float(np.sum(probability)), 1e-12)
        predicted = classes[int(np.argmax(probability))]
        mean_gate = np.mean(np.stack(gates[batch_id]), axis=0)
        result.append(
            {
                "batch_id": batch_id,
                "label": labels[batch_id],
                "prediction": predicted,
                "confidence": float(np.max(probability)),
                "probabilities": {label: float(value) for label, value in zip(classes, probability)},
                "mean_modality_gate": {
                    "radar": float(mean_gate[0]),
                    "infrared": float(mean_gate[1]),
                },
                "windows": len(grouped[batch_id]),
            }
        )
    return result


def _folds(records: list[CachedRecord], requested: int, seed: int) -> tuple[list[tuple[list[int], list[int]]], str]:
    labels = np.asarray([record.label for record in records])
    counts = Counter(labels)
    minimum = min(counts.values())
    if len(records) >= 4 and minimum >= 2:
        splits = min(requested, minimum)
        splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
        return [(train.tolist(), test.tolist()) for train, test in splitter.split(np.arange(len(records)), labels)], "stratified_record_cv"

    partial: list[tuple[list[int], list[int]]] = []
    for test_index in range(len(records)):
        train = [index for index in range(len(records)) if index != test_index]
        train_classes = {records[index].label for index in train}
        if len(train_classes) >= 2 and records[test_index].label in train_classes:
            partial.append((train, [test_index]))
    return partial, "partial_leave_one_record_out"


def evaluate_records(
    records: list[CachedRecord],
    config: dict[str, Any],
    radar_schema: list[str],
) -> dict[str, Any]:
    folds, strategy = _folds(records, int(config["evaluation_folds"]), int(config["seed"]))
    all_classes = sorted({record.label for record in records})
    if not folds:
        return {
            "status": "insufficient_independent_records",
            "reason": "No leakage-safe validation fold can train on every class represented in its test fold.",
            "strategy": strategy,
            "record_count": len(records),
            "class_record_counts": dict(Counter(record.label for record in records)),
            "coverage": 0.0,
            "predictions": [],
        }

    predictions: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    cv_config = dict(config)
    cv_config["epochs"] = min(int(config["epochs"]), 8)
    cv_config["early_stop_patience"] = min(int(config["early_stop_patience"]), 3)
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        train_records = [records[index] for index in train_indices]
        test_records = [records[index] for index in test_indices]
        print(
            f"[cv {fold_index + 1}/{len(folds)}] train={[item.batch_id for item in train_records]} "
            f"test={[item.batch_id for item in test_records]}",
            flush=True,
        )
        with tempfile.TemporaryDirectory(prefix="anti_air_cv_") as directory:
            checkpoint = Path(directory) / "model.pt"
            _, history = train_model(
                train_records,
                cv_config,
                radar_schema,
                validation_records=test_records,
                output_path=checkpoint,
            )
            model, checkpoint_payload = load_checkpoint(checkpoint, device=resolve_device())
            fold_predictions = _record_probabilities(
                model,
                test_records,
                checkpoint_payload["classes"],
                cv_config,
            )
        for item in fold_predictions:
            item["fold"] = fold_index
        predictions.extend(fold_predictions)
        fold_summaries.append(
            {
                "fold": fold_index,
                "train_batches": [item.batch_id for item in train_records],
                "test_batches": [item.batch_id for item in test_records],
                "epochs": history["epochs_completed"],
                "seconds": history["elapsed_seconds"],
            }
        )

    truth = np.asarray([item["label"] for item in predictions])
    predicted = np.asarray([item["prediction"] for item in predictions])
    probabilities = np.asarray(
        [[item["probabilities"].get(label, 0.0) for label in all_classes] for item in predictions],
        dtype=float,
    )
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    evaluated_batches = {item["batch_id"] for item in predictions}
    evaluated_classes = set(truth.tolist())
    coverage = len(evaluated_batches) / max(1, len(records))
    statistically_valid = coverage == 1.0 and evaluated_classes == set(all_classes) and min(Counter(record.label for record in records).values()) >= 2
    status = "valid" if statistically_valid else "diagnostic_only"
    matrix = confusion_matrix(truth, predicted, labels=all_classes)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            truth,
            predicted,
            labels=all_classes,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": all_classes,
            "values": matrix.tolist(),
        },
    }
    try:
        metrics["log_loss"] = float(log_loss(truth, probabilities, labels=all_classes))
    except ValueError:
        metrics["log_loss"] = None
    return {
        "status": status,
        "warning": None if statistically_valid else (
            "The validation does not cover every independent record and class; metrics are diagnostic and must not be reported as final competition performance."
        ),
        "strategy": strategy,
        "record_count": len(records),
        "class_record_counts": dict(Counter(record.label for record in records)),
        "coverage": coverage,
        "evaluated_classes": sorted(evaluated_classes),
        "folds": fold_summaries,
        "metrics": metrics,
        "predictions": predictions,
    }


def predict_cached_records(
    checkpoint_path: str | Path,
    records: list[CachedRecord],
) -> list[dict[str, Any]]:
    model, checkpoint = load_checkpoint(checkpoint_path, device=resolve_device())
    return _record_probabilities(model, records, checkpoint["classes"], checkpoint["config"])
