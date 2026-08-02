from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from anti_air.dataset import discover_samples
from anti_air.modeling import fit_branch
from anti_air.pipeline import extract_sample, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multimodal anti-air baseline")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/baseline")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(args.data_root, require_labels=True, strict_pairs=True)

    labels: list[str] = []
    radar_rows: list[dict[str, float]] = []
    ir_rows: list[dict[str, float]] = []
    manifest: list[dict[str, object]] = []

    for index, sample in enumerate(samples, start=1):
        print(f"[{index}/{len(samples)}] extracting batch={sample.batch_id} label={sample.label}")
        bundle = extract_sample(sample, config)
        labels.append(str(sample.label))
        radar_rows.append(bundle.radar)
        ir_rows.append(bundle.infrared)
        manifest.append(
            {
                "batch_id": sample.batch_id,
                "label": sample.label,
                "alignment": bundle.alignment.to_dict(),
                "quality": bundle.quality,
            }
        )

    if len(set(labels)) < 2:
        raise ValueError("At least two target classes are required for training")

    radar_frame = pd.DataFrame(radar_rows)
    ir_frame = pd.DataFrame(ir_rows)
    model_cfg = config.get("model", {})
    classifier_kwargs = {
        "seed": int(config.get("seed", 2026)),
        "n_estimators": int(model_cfg.get("n_estimators", 400)),
        "min_samples_leaf": int(model_cfg.get("min_samples_leaf", 1)),
        "class_weight": str(model_cfg.get("class_weight", "balanced_subsample")),
    }
    radar_model = fit_branch(radar_frame, labels, **classifier_kwargs)
    ir_model = fit_branch(ir_frame, labels, **classifier_kwargs)

    artifact = {
        "version": "0.1.0",
        "config": config,
        "radar": radar_model,
        "infrared": ir_model,
        "classes": sorted(set(labels)),
        "training_batches": [sample.batch_id for sample in samples],
    }
    joblib.dump(artifact, output_dir / "model.joblib")
    radar_frame.assign(batch_id=[s.batch_id for s in samples], label=labels).to_csv(
        output_dir / "radar_features.csv", index=False
    )
    ir_frame.assign(batch_id=[s.batch_id for s in samples], label=labels).to_csv(
        output_dir / "infrared_features.csv", index=False
    )
    (output_dir / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Model and feature tables written to {output_dir}")


if __name__ == "__main__":
    main()
