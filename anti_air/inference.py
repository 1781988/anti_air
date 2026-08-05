from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Sample
from .preprocess import preprocess_record
from .torch_data import WindowDataset, make_refs
from .trainer import load_checkpoint, resolve_device


def predict_pair(
    *,
    radar_path: str | Path,
    infrared_path: str | Path,
    model_path: str | Path,
    batch_id: str = "inference",
) -> dict[str, Any]:
    device = resolve_device()
    model, checkpoint = load_checkpoint(model_path, device=device)
    classes = list(checkpoint["classes"])
    sample = Sample(
        batch_id=batch_id,
        label=classes[0],
        radar_path=Path(radar_path).expanduser().resolve(),
        infrared_path=Path(infrared_path).expanduser().resolve(),
        start_time="unknown",
    )
    if not sample.radar_path.is_file():
        raise FileNotFoundError(f"Radar file not found: {sample.radar_path}")
    if not sample.infrared_path.is_file():
        raise FileNotFoundError(f"Infrared file not found: {sample.infrared_path}")
    with tempfile.TemporaryDirectory(prefix="anti_air_infer_") as directory:
        cached = preprocess_record(
            sample,
            checkpoint["config"],
            list(checkpoint["radar_schema"]),
            Path(directory),
            rebuild=True,
        )
        dataset = WindowDataset(
            make_refs([cached]),
            {label: index for index, label in enumerate(classes)},
            augment=False,
            seed=int(checkpoint["config"]["seed"]),
        )
        loader = DataLoader(
            dataset,
            batch_size=int(checkpoint["config"]["batch_size"]),
            shuffle=False,
            num_workers=0,
        )
        probabilities: list[np.ndarray] = []
        gates: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                output = model(
                    batch["radar"].to(device),
                    batch["infrared"].to(device),
                    batch["quality"].to(device),
                )
                probabilities.extend(torch.softmax(output["logits"], -1).cpu().numpy())
                gates.extend(output["gates"].cpu().numpy())
    probability = np.mean(np.stack(probabilities), axis=0)
    probability /= max(float(probability.sum()), 1e-12)
    gate = np.mean(np.stack(gates), axis=0)
    return {
        "batch_id": batch_id,
        "prediction": classes[int(np.argmax(probability))],
        "confidence": float(np.max(probability)),
        "probabilities": {label: float(value) for label, value in zip(classes, probability)},
        "windows": cached.windows,
        "alignment": cached.alignment,
        "mean_modality_gate": {"radar": float(gate[0]), "infrared": float(gate[1])},
        "model_version": checkpoint.get("model_version", "unknown"),
    }
