PYTHON ?= python
DATA_ROOT ?= data/train
CONFIG ?= configs/default.yaml

.PHONY: setup test inspect extract evaluate train report package all

setup:
	bash scripts/setup_server.sh

test:
	$(PYTHON) -m pytest -q

inspect:
	$(PYTHON) scripts/inspect_dataset.py --data-root $(DATA_ROOT) --require-labels --output-dir outputs/inspection

extract:
	$(PYTHON) extract_features.py --data-root $(DATA_ROOT) --config $(CONFIG) --output-dir outputs/features

evaluate:
	$(PYTHON) evaluate.py --features outputs/features --config $(CONFIG) --output-dir outputs/evaluation

train:
	$(PYTHON) train.py --features outputs/features --config $(CONFIG) --output-dir outputs/model

report:
	$(PYTHON) scripts/generate_report.py --metrics outputs/evaluation/metrics.json --training-summary outputs/model/training_summary.json --output outputs/report/test_report.md

package:
	$(PYTHON) scripts/package_submission.py --model outputs/model/model.joblib --report outputs/report/test_report.md --output outputs/submission/anti_air_submission.zip

all:
	bash scripts/run_all.sh $(DATA_ROOT) $(CONFIG)
