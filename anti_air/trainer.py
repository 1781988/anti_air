from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from .model import MultiModalClassifier, build_model
from .preprocess import CachedRecord
from .torch_data import WindowRef, WindowDataset, make_refs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sampler(refs: list[WindowRef]) -> WeightedRandomSampler:
    record_windows: dict[str, int] = {}
    class_records: dict[str, set[str]] = {}
    for ref in refs:
        record_windows[ref.batch_id] = record_windows.get(ref.batch_id, 0) + 1
        class_records.setdefault(ref.label, set()).add(ref.batch_id)
    weights = [
        1.0 / max(1, record_windows[ref.batch_id]) / max(1, len(class_records[ref.label]))
        for ref in refs
    ]
    return WeightedRandomSampler(weights, num_samples=len(refs), replacement=True)


def _loader(
    refs: list[WindowRef],
    label_to_index: dict[str, int],
    config: dict[str, Any],
    *,
    train: bool,
) -> DataLoader:
    dataset = WindowDataset(
        refs,
        label_to_index,
        augment=train,
        seed=int(config["seed"]),
        cache_files=2,
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(config["batch_size"]),
        "num_workers": int(config["num_workers"]),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": train and len(dataset) >= int(config["batch_size"]),
    }
    if train:
        kwargs["sampler"] = _sampler(refs)
    else:
        kwargs["shuffle"] = False
    return DataLoader(**kwargs)


def _parameter_groups(model: MultiModalClassifier, config: dict[str, Any]) -> list[dict[str, Any]]:
    backbone_ids = {id(parameter) for parameter in model.infrared.backbone.parameters()}
    backbone = [parameter for parameter in model.parameters() if id(parameter) in backbone_ids]
    other = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    return [
        {"params": other, "lr": float(config["learning_rate"])},
        {"params": backbone, "lr": float(config["backbone_learning_rate"])},
    ]


def _run_epoch(
    model: MultiModalClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    autocast_enabled = device.type == "cuda"
    for batch in loader:
        radar = batch["radar"].to(device, non_blocking=True)
        infrared = batch["infrared"].to(device, non_blocking=True)
        quality = batch["quality"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                output = model(radar, infrared, quality)
                loss = criterion(output["logits"], target)
            if training:
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
        total_loss += float(loss.detach().cpu()) * len(target)
        labels.extend(target.detach().cpu().tolist())
        predictions.extend(output["logits"].argmax(dim=-1).detach().cpu().tolist())
    denominator = max(1, len(labels))
    return {
        "loss": total_loss / denominator,
        "accuracy": float(accuracy_score(labels, predictions)) if labels else math.nan,
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)) if labels else math.nan,
    }


def train_model(
    train_records: list[CachedRecord],
    config: dict[str, Any],
    radar_schema: list[str],
    *,
    validation_records: list[CachedRecord] | None = None,
    output_path: str | Path | None = None,
    pretrained: bool | None = None,
) -> tuple[MultiModalClassifier, dict[str, Any]]:
    started = time.perf_counter()
    set_seed(int(config["seed"]))
    device = resolve_device()
    classes = sorted({record.label for record in train_records})
    if len(classes) < 2:
        raise ValueError("At least two classes are required in the training records")
    label_to_index = {label: index for index, label in enumerate(classes)}
    train_refs = make_refs(train_records)
    validation_refs = make_refs(validation_records or [])
    train_loader = _loader(train_refs, label_to_index, config, train=True)
    validation_loader = (
        _loader(validation_refs, label_to_index, config, train=False) if validation_refs else None
    )
    model = build_model(
        config,
        num_classes=len(classes),
        pretrained=bool(config["pretrained_vision"] if pretrained is None else pretrained),
    ).to(device)
    freeze_epochs = int(config["freeze_backbone_epochs"])
    model.infrared.set_backbone_trainable(freeze_epochs <= 0)
    optimizer = AdamW(_parameter_groups(model, config), weight_decay=float(config["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, int(config["epochs"])))

    record_counts: dict[str, int] = {}
    for record in train_records:
        record_counts[record.label] = record_counts.get(record.label, 0) + 1
    class_weights = torch.tensor(
        [len(train_records) / max(1, len(classes) * record_counts[label]) for label in classes],
        device=device,
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(config["label_smoothing"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, Any]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_score = -math.inf
    no_improvement = 0

    for epoch in range(int(config["epochs"])):
        if epoch == freeze_epochs:
            model.infrared.set_backbone_trainable(True)
        epoch_started = time.perf_counter()
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )
        validation_metrics = None
        if validation_loader is not None:
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
            )
            score = float(validation_metrics["macro_f1"])
        else:
            score = -float(train_metrics["loss"])
        scheduler.step()
        entry = {
            "epoch": epoch + 1,
            "seconds": time.perf_counter() - epoch_started,
            "backbone_trainable": epoch >= freeze_epochs,
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
        }
        history.append(entry)
        print(
            f"[epoch {epoch + 1:02d}/{int(config['epochs']):02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_f1={train_metrics['macro_f1']:.4f} "
            + (
                f"val_f1={validation_metrics['macro_f1']:.4f} "
                if validation_metrics is not None
                else ""
            )
            + f"seconds={entry['seconds']:.1f}",
            flush=True,
        )
        if score > best_score + 1e-6:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
        if validation_loader is not None and no_improvement >= int(config["early_stop_patience"]):
            break

    model.load_state_dict(best_state)
    summary = {
        "device": str(device),
        "classes": classes,
        "train_records": [record.batch_id for record in train_records],
        "validation_records": [record.batch_id for record in validation_records or []],
        "train_windows": len(train_refs),
        "validation_windows": len(validation_refs),
        "epochs_completed": len(history),
        "best_score": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_version": "2.0.0",
                "state_dict": model.state_dict(),
                "classes": classes,
                "radar_schema": radar_schema,
                "config": config,
                "training": summary,
            },
            destination,
        )
    return model, summary


def load_checkpoint(path: str | Path, *, device: torch.device | None = None) -> tuple[MultiModalClassifier, dict[str, Any]]:
    device = device or resolve_device()
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config, num_classes=len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint
