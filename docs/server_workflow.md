# 服务器训练、测试与交付全流程

## 1. 服务器建议

- 操作系统：Ubuntu 20.04/22.04/24.04；
- Python：3.11；
- CPU：8 核以上；
- 内存：16 GB 以上；
- 磁盘：训练数据体积的 2–3 倍；
- GPU：当前 ExtraTrees 基线不依赖 GPU，红外特征提取主要使用 CPU。

## 2. 下载代码

```bash
git clone https://github.com/1781988/anti_air.git
cd anti_air
git checkout main
```

## 3. 创建环境

```bash
bash scripts/setup_server.sh
source .venv/bin/activate
```

也可手动安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
python -m pytest -q
```

## 4. 上传并解压数据

将 `初赛数据.7z` 上传到仓库根目录后：

```bash
python scripts/unpack_data.py 初赛数据.7z --output data/train
```

若服务器已安装 7-Zip，也可：

```bash
7z x 初赛数据.7z -odata/train
```

解压后的 `.mat` 和 `.mp4` 可以位于任意子目录，程序会递归搜索并按批号配对。

## 5. 数据检查

```bash
python scripts/inspect_dataset.py \
  --data-root data/train \
  --require-labels \
  --output-dir outputs/inspection
```

重点检查：

```text
outputs/inspection/dataset_inventory.json
outputs/inspection/resolved_manifest.csv
```

若雷达 MATLAB table 无法解析，执行 MATLAB：

```matlab
convert_matlab_tables('data/train', 'data/train_converted')
```

随后改用 `data/train_converted` 中的雷达文件，并在 manifest 中保持与红外视频配对。

## 6. 快速联调

快速配置降低抽帧率和分辨率，仅用于验证流程：

```bash
python extract_features.py \
  --data-root data/train \
  --config configs/quick.yaml \
  --output-dir outputs/features_quick

python train.py \
  --features outputs/features_quick \
  --config configs/quick.yaml \
  --output-dir outputs/model_quick
```

## 7. 正式特征提取

```bash
python extract_features.py \
  --data-root data/train \
  --config configs/default.yaml \
  --output-dir outputs/features
```

首次执行会处理全部视频。每个批次结果缓存于 `outputs/features/records/`，中断后重新执行会自动复用已完成批次。配置或原始文件变化后，对应批次会自动重新提取。

## 8. 无泄漏评价

```bash
python evaluate.py \
  --features outputs/features \
  --config configs/default.yaml \
  --output-dir outputs/evaluation
```

输出：

```text
metrics.json
folds.json
record_predictions.csv
window_predictions.csv
confusion_matrix.csv
```

样例数据若只有少数批次，可能得到 `insufficient_grouped_data` 或 `limited_folds`。这是数据量不足，不是程序失败。正式训练数据增加后会自动执行分层批次交叉验证。

## 9. 训练最终模型

```bash
python train.py \
  --features outputs/features \
  --config configs/default.yaml \
  --output-dir outputs/model
```

最终模型：

```text
outputs/model/model.joblib
```

## 10. 单对数据测试

```bash
python infer.py \
  --radar 'data/train/radar_339_class-B_16：18.mat' \
  --ir 'data/train/ir_339_class-B_16：18.mp4' \
  --model outputs/model/model.joblib \
  --output outputs/predictions/batch_339.json
```

也可使用官方风格入口：

```bash
ANTI_AIR_MODEL=outputs/model/model.joblib \
  bash run.sh radar_test.mat ir_test.mp4 result.json
```

## 11. 批量测试

```bash
python infer.py \
  --data-root data/test \
  --model outputs/model/model.joblib \
  --output outputs/predictions/test_results.json
```

输出记录级 JSON、记录级 CSV 和窗口级 CSV。

## 12. 生成测试报告

```bash
python scripts/generate_report.py \
  --metrics outputs/evaluation/metrics.json \
  --training-summary outputs/model/training_summary.json \
  --output outputs/report/test_report.md
```

## 13. 生成比赛提交包

```bash
python scripts/package_submission.py \
  --model outputs/model/model.joblib \
  --report outputs/report/test_report.md \
  --output outputs/submission/anti_air_submission.zip
```

提交包中包括模型、源代码、算法设计、运行脚本、依赖清单和测试报告。

## 14. 一键执行

完成环境和数据解压后：

```bash
bash scripts/run_all.sh data/train configs/default.yaml
```

该命令依次完成数据检查、特征提取、评价、最终训练、报告生成和提交包构建。
