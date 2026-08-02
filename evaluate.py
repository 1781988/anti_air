from __future__ import annotations

import argparse
from pathlib import Path

from anti_air.config import load_config
from anti_air.evaluation import evaluate_grouped_cv
from anti_air.feature_store import load_feature_cache
from anti_air.utils import json_dump, set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leakage-safe batch-grouped model evaluation")
    parser.add_argument("--features", default="outputs/features")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    set_global_seed(int(config["seed"]))
    tables = load_feature_cache(args.features, require_labels=True)
    result = evaluate_grouped_cv(tables, config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_dump(result.metrics, output / "metrics.json")
    json_dump(result.folds, output / "folds.json")
    result.predictions.to_csv(output / "record_predictions.csv", index=False)
    result.window_predictions.to_csv(output / "window_predictions.csv", index=False)
    result.confusion.to_csv(output / "confusion_matrix.csv", index=True)
    print(f"Evaluation status={result.metrics['status']} output={output}")
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "log_loss"):
        if key in result.metrics:
            print(f"{key}: {result.metrics[key]}")


if __name__ == "__main__":
    main()
