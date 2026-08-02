#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-data/train}"
CONFIG="${2:-configs/default.yaml}"

python scripts/inspect_dataset.py --data-root "$DATA_ROOT" --require-labels --output-dir outputs/inspection
python extract_features.py --data-root "$DATA_ROOT" --config "$CONFIG" --output-dir outputs/features
python evaluate.py --features outputs/features --config "$CONFIG" --output-dir outputs/evaluation
python train.py --features outputs/features --config "$CONFIG" --output-dir outputs/model
python scripts/generate_report.py \
  --metrics outputs/evaluation/metrics.json \
  --training-summary outputs/model/training_summary.json \
  --output outputs/report/test_report.md
python scripts/package_submission.py \
  --model outputs/model/model.joblib \
  --report outputs/report/test_report.md \
  --output outputs/submission/anti_air_submission.zip

echo "Complete. Submission: outputs/submission/anti_air_submission.zip"
