# 提交检查清单

- [ ] `python -m pytest -q` 通过；
- [ ] `outputs/inspection/dataset_inventory.json` 中 `error_count=0`；
- [ ] `outputs/evaluation/metrics.json` 已检查评价状态、记录覆盖率和类别覆盖率；
- [ ] 当 `eligible_for_primary_score=false` 时，报告没有把诊断指标描述为正式泛化性能；
- [ ] `outputs/model/model.joblib`、`training_summary.json`、`MODEL_CARD.md` 均存在；
- [ ] `outputs/report/test_report.md` 已生成；
- [ ] 提交包包含 `evaluation/metrics.json`、`folds.json`、记录级预测和混淆矩阵；
- [ ] `python scripts/validate_submission.py outputs/submission/anti_air_submission.zip` 通过；
- [ ] 在全新虚拟环境中解压提交包并执行 `bash run.sh <radar.mat> <infrared.mp4> result.json`；
- [ ] 推理程序不依赖文件名中的类别字段，不访问互联网，不需要人工交互。
