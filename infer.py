from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from anti_air.dataset import Sample
from anti_air.modeling import predict_branch, quality_aware_fusion
from anti_air.pipeline import extract_sample


def predict(radar_path: str, infrared_path: str, model_path: str) -> dict[str, object]:
    artifact = joblib.load(model_path)
    sample = Sample(batch_id="inference", radar_path=Path(radar_path), infrared_path=Path(infrared_path))
    bundle = extract_sample(sample, artifact["config"])

    radar_classes, radar_proba = predict_branch(artifact["radar"], bundle.radar)
    ir_classes, ir_proba = predict_branch(artifact["infrared"], bundle.infrared)
    model_cfg = artifact["config"].get("model", {})
    classes, fused, weights = quality_aware_fusion(
        radar_classes,
        radar_proba,
        ir_classes,
        ir_proba,
        radar_quality=bundle.quality["radar"],
        infrared_quality=bundle.quality["infrared"],
        base_radar_weight=float(model_cfg.get("radar_weight", 0.60)),
        base_infrared_weight=float(model_cfg.get("infrared_weight", 0.40)),
    )
    best = int(fused.argmax())
    return {
        "label": classes[best],
        "confidence": float(fused[best]),
        "class_probabilities": {c: float(p) for c, p in zip(classes, fused, strict=True)},
        "branch_probabilities": {
            "radar": {str(c): float(p) for c, p in zip(radar_classes, radar_proba, strict=True)},
            "infrared": {str(c): float(p) for c, p in zip(ir_classes, ir_proba, strict=True)},
        },
        "fusion_weights": weights,
        "alignment": bundle.alignment.to_dict(),
        "quality": bundle.quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run radar-infrared target classification")
    parser.add_argument("--radar", required=True)
    parser.add_argument("--ir", required=True)
    parser.add_argument("--model", default="outputs/baseline/model.joblib")
    parser.add_argument("--output", default="result.json")
    args = parser.parse_args()

    result = predict(args.radar, args.ir, args.model)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
