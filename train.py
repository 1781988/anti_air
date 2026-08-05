from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from anti_air.config import load_config
from anti_air.dataset import resolve_samples
from anti_air.feature_store import build_feature_cache, load_feature_cache
from anti_air.modeling import fit_model_bundle
from anti_air.utils import json_dump, set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the final radar-infrared multimodal model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features", help="Directory created by extract_features.py")
    source.add_argument("--data-root", help="Training data directory; features are extracted automatically")
    source.add_argument("--manifest", help="CSV manifest; features are extracted automatically")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/model")
    parser.add_argument("--force-features", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    set_global_seed(int(config["seed"]))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.features:
        tables = load_feature_cache(args.features, require_labels=True)
        feature_source = str(Path(args.features).resolve())
    else:
        samples = resolve_samples(
            data_root=args.data_root,
            manifest=args.manifest,
            require_labels=True,
            strict_pairs=True,
        )
        feature_dir = output / "features"
        tables = build_feature_cache(samples, config, feature_dir, force=args.force_features)
        feature_source = str(feature_dir.resolve())

    bundle = fit_model_bundle(tables, config)
    joblib.dump(bundle, output / "model.joblib", compress=3)
    summary = {
        **bundle.training_summary,
        "model_version": bundle.version,
        "feature_source": feature_source,
        "config": config,
    }
    json_dump(summary, output / "training_summary.json")
    warning = summary.get("training_data_warning") or "None"
    (output / "MODEL_CARD.md").write_text(
        "# Anti-Air Model Card\n\n"
        f"- Model version: `{bundle.version}`\n"
        f"- Independent records: `{summary['records']}`\n"
        f"- Derived windows: `{summary['windows']}`\n"
        f"- Classes: `{', '.join(summary['classes'])}`\n"
        f"- Class record counts: `{summary['class_record_counts']}`\n"
        f"- Minimum records per class: `{summary.get('minimum_records_per_class')}`\n"
        f"- Record-level confidence reliability: `{summary.get('record_level_confidence_reliability')}`\n"
        f"- Effective probability smoothing: `{summary.get('effective_probability_smoothing')}`\n"
        f"- Training data warning: `{warning}`\n\n"
        "The model contains radar, infrared and feature-level fusion ExtraTrees branches. "
        "Window probabilities are quality-weighted and aggregated at record level. "
        "Derived windows are not independent samples; independent record counts determine evaluation validity.\n",
        encoding="utf-8",
    )
    print(f"Training complete: model={output / 'model.joblib'}")
    print(summary)


if __name__ == "__main__":
    main()
