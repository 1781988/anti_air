# anti_air：复杂环境下雷达—红外目标类型智能识别

本仓库面向“探测识别赛道—科目1：复杂环境下的目标类型智能识别”。任务是在无人机超低空、超低速突防以及鸟、风筝等近地目标干扰条件下，综合利用雷达信号级、点迹级、航迹级信息和红外视频，识别无人机、空飘物及其他非无人机目标。

仓库已经覆盖完整工程流程：

```text
数据解压 → 配对检查 → MATLAB table解析 → 红外小目标跟踪
       → 雷达/红外时间对准 → 窗口级多模态特征
       → 批号级无泄漏交叉验证 → 最终模型训练
       → 单对/批量推理 → 测试报告 → 比赛提交压缩包
```

## 1. 当前算法

### 雷达分支

- 支持普通数值矩阵和 MATLAB `table/timetable`；
- 支持复数信号和向量型字段；
- 自动识别时间字段，缺少时间戳时使用视频时长估算采样率；
- 提取统计、差分、频谱、缺失率和活动曲线特征；
- 正式数据中出现距离、速度、SNR、RCS、方位、俯仰、点迹和航迹字段时可自动纳入。

### 红外分支

- CLAHE局部对比增强；
- 帧差与中心—周围局部对比联合响应；
- 自适应高分位阈值小目标候选；
- 常速度预测、距离门控和短时漏检保持；
- 提取亮度、运动、候选、速度、加速度、转向率、轨迹直线度和频域特征。

### 时间同步

数据说明指出雷达和红外为成对采集但未完成精确同步。本项目估计：

```text
t_ir = scale × t_radar + offset
```

既支持固定时间偏移，也支持长记录的线性时钟漂移估计。

### 三分支模型

- 雷达 ExtraTrees；
- 红外 ExtraTrees；
- 雷达—红外特征级融合 ExtraTrees；
- 根据雷达有效率、红外跟踪率和同步质量动态融合三分支概率；
- 将窗口结果以不确定度加权方式汇总为整段记录类别。

## 2. 防止标签泄漏

训练文件名中可能包含 `class-A/class-B`。类别字段只用于生成训练标签，不进入模型特征。推理接口不解析文件名标签。

评价严格按批号划分。禁止把同一视频或同一雷达记录切出的窗口随机分配到训练集和验证集。

## 3. 仓库结构

```text
anti_air/
├── anti_air/
│   ├── alignment.py       # 固定偏移和时钟漂移估计
│   ├── config.py          # 配置加载与校验
│   ├── dataset.py         # 文件配对、manifest和标签隔离
│   ├── evaluation.py      # 批号级无泄漏评价
│   ├── feature_store.py   # 特征缓存和断点复用
│   ├── infrared.py        # 红外小目标检测、跟踪和特征
│   ├── modeling.py        # 三分支模型和质量感知融合
│   ├── pipeline.py        # 对准窗口构建和多模态特征
│   ├── radar.py           # MATLAB table/矩阵/复数/向量解析
│   └── utils.py
├── configs/
│   ├── default.yaml       # 正式配置
│   └── quick.yaml         # 快速联调配置
├── docs/
│   ├── algorithm_design.md
│   ├── server_workflow.md
│   └── submission_checklist.md
├── scripts/
│   ├── convert_matlab_tables.m
│   ├── generate_report.py
│   ├── inspect_dataset.py
│   ├── package_submission.py
│   ├── run_all.sh
│   ├── setup_server.sh
│   └── unpack_data.py
├── extract_features.py
├── evaluate.py
├── train.py
├── infer.py
├── run.sh
├── Dockerfile
└── Makefile
```

## 4. 服务器安装

```bash
git clone https://github.com/1781988/anti_air.git
cd anti_air
git checkout main

bash scripts/setup_server.sh
source .venv/bin/activate
```

手动方式：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
python -m pytest -q
```

## 5. 解压比赛数据

将 `初赛数据.7z` 上传至仓库根目录：

```bash
python scripts/unpack_data.py 初赛数据.7z --output data/train
```

推荐目录：

```text
data/train/
├── radar_339_class-B_16：18.mat
├── ir_339_class-B_16：18.mp4
├── radar_357_class-A_16：32.mat
├── ir_357_class-A_16：32.mp4
└── ...
```

程序递归查找文件并按批号配对，不依赖文件排列顺序。

## 6. 数据检查

```bash
python scripts/inspect_dataset.py \
  --data-root data/train \
  --require-labels \
  --output-dir outputs/inspection
```

输出：

```text
outputs/inspection/dataset_inventory.json
outputs/inspection/resolved_manifest.csv
```

如果 MATLAB table 仍无法解析，可在 MATLAB 中执行：

```matlab
convert_matlab_tables('data/train', 'data/train_converted')
```

## 7. 快速联调

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

快速配置只用于确认环境和接口，不用于最终比赛结果。

## 8. 正式特征提取

```bash
python extract_features.py \
  --data-root data/train \
  --config configs/default.yaml \
  --output-dir outputs/features
```

首次处理长视频耗时较长。每个批次会独立缓存至：

```text
outputs/features/records/
```

重新执行时自动复用已完成批次；原始文件或配置变化后自动重新提取。

## 9. 评价性能

```bash
python evaluate.py \
  --features outputs/features \
  --config configs/default.yaml \
  --output-dir outputs/evaluation
```

输出：

```text
outputs/evaluation/
├── metrics.json
├── folds.json
├── record_predictions.csv
├── window_predictions.csv
└── confusion_matrix.csv
```

评价指标包括 Accuracy、Balanced Accuracy、Macro-F1、Weighted-F1、Log Loss、逐类指标、混淆矩阵和Macro-F1置信区间。

当前提供的样例数据如果只有极少批次，程序可能报告：

```text
insufficient_grouped_data
```

这表示无法形成无泄漏且训练折包含全部类别的验证折。程序不会使用随机窗口划分制造虚高指标。完整训练数据增加后会自动执行分层批号交叉验证。

## 10. 训练最终模型

```bash
python train.py \
  --features outputs/features \
  --config configs/default.yaml \
  --output-dir outputs/model
```

得到：

```text
outputs/model/
├── model.joblib
├── training_summary.json
└── MODEL_CARD.md
```

也可直接从原始数据完成特征提取和训练：

```bash
python train.py \
  --data-root data/train \
  --config configs/default.yaml \
  --output-dir outputs/model
```

## 11. 单对数据推理

```bash
python infer.py \
  --radar 'data/train/radar_339_class-B_16：18.mat' \
  --ir 'data/train/ir_339_class-B_16：18.mp4' \
  --model outputs/model/model.joblib \
  --output outputs/predictions/batch_339.json
```

统一Python接口：

```python
from infer import predict

result = predict(
    radar_path="radar_test.mat",
    infrared_path="ir_test.mp4",
    model_path="outputs/model/model.joblib",
    batch_id="test-001",
)
```

输出：

```json
{
  "batch_id": "test-001",
  "label": "class-A",
  "confidence": 0.8731,
  "class_probabilities": {
    "class-A": 0.8731,
    "class-B": 0.1269
  },
  "window_count": 21,
  "alignment": {
    "offset_seconds": 2.4,
    "scale": 1.0002,
    "drift_ppm": 200.0,
    "score": 0.71
  }
}
```

## 12. 批量测试

测试文件名不需要包含类别：

```bash
python infer.py \
  --data-root data/test \
  --model outputs/model/model.joblib \
  --output outputs/predictions/test_results.json
```

也支持CSV manifest：

```csv
batch_id,radar_path,infrared_path,label,start_time
001,/data/radar_001.mat,/data/ir_001.mp4,,
```

```bash
python infer.py \
  --manifest data/test_manifest.csv \
  --model outputs/model/model.joblib \
  --output outputs/predictions/test_results.json
```

## 13. 官方评测风格入口

```bash
ANTI_AIR_MODEL=outputs/model/model.joblib \
  bash run.sh radar_test.mat infrared_test.mp4 result.json
```

提交包内部模型路径固定后可直接执行：

```bash
bash run.sh radar_test.mat infrared_test.mp4 result.json
```

## 14. 生成测试报告

```bash
python scripts/generate_report.py \
  --metrics outputs/evaluation/metrics.json \
  --training-summary outputs/model/training_summary.json \
  --output outputs/report/test_report.md
```

## 15. 生成比赛提交包

```bash
python scripts/package_submission.py \
  --model outputs/model/model.joblib \
  --report outputs/report/test_report.md \
  --output outputs/submission/anti_air_submission.zip
```

压缩包包含：

- 算法模型；
- 模型源代码；
- 推理入口；
- 依赖文件；
- 算法设计方案；
- 测试报告；
- 一键运行脚本。

## 16. 一键完成全部流程

环境安装和数据解压完成后：

```bash
bash scripts/run_all.sh data/train configs/default.yaml
```

最终结果：

```text
outputs/submission/anti_air_submission.zip
```

## 17. 重要说明

截图和现有备注没有给出组委会最终评分公式及精确输出协议，因此仓库默认使用常见分类指标并输出通用JSON。组委会发布正式评测接口后，只需调整 `infer.py` 的输入/输出适配层和 `evaluate.py` 的主指标，不需要重写特征与模型主流程。

详细设计见 [docs/algorithm_design.md](docs/algorithm_design.md)，服务器完整操作见 [docs/server_workflow.md](docs/server_workflow.md)。
