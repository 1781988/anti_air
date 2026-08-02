# anti_air：复杂环境下雷达—红外目标类型智能识别基线

本仓库面向“探测识别赛道—科目1：复杂环境下的目标类型智能识别”。赛题关注无人机在超低空、超低速、突防等复杂场景下与鸟、风筝、漂浮物等近地目标难以区分的问题，要求综合利用跟踪目标的信号级、点迹级、航迹级及红外视频信息，构建目标类别识别算法，并提交可运行模型、源代码、算法方案和测试报告。

当前版本提供一套**可运行、可扩展、避免标签泄漏**的雷达—红外双模态基线。它不是最终高分模型，而是用于先打通数据解析、异步对准、特征提取、训练、推理和提交接口的工程骨架。

## 1. 数据结论与建模原则

根据已提供的样例数据：

- 每个批号包含一份雷达 `.mat` 和一份红外 `.mp4`；
- 文件名格式示例：`radar_339_class-B_16：18.mat`、`ir_339_class-B_16：18.mp4`；
- 雷达与红外为成对采样，但**未完成精确时间同步**；
- 红外目标通常只占整帧很小区域，云层边缘和背景运动容易形成伪目标；
- 雷达 `.mat` 可能保存 MATLAB `table`，普通 `scipy.io.loadmat` 不一定能直接展开；
- 类别出现在训练文件名中，只允许用于生成训练标签，推理时不得读取文件名中的类别字段。

因此本项目采用如下路线：

```text
成对文件发现
  ├─ 雷达：MATLAB table/矩阵解析 → 数值列统计、差分与活动曲线
  ├─ 红外：稀疏抽帧 → 帧差小目标候选 → 运动/亮度/轨迹统计
  ├─ 同步：雷达活动曲线 × 红外活动曲线互相关估计时间偏移
  └─ 双分支分类器 → 质量感知后融合 → 整段类别与置信度
```

首版优先使用后融合，原因是异步数据下直接做特征级交互容易将错误时间对应关系学习成噪声。完成稳定对准和目标级标注后，再升级为时窗级跨模态网络。

## 2. 仓库结构

```text
anti_air/
├── anti_air/
│   ├── alignment.py      # 雷达—红外活动曲线互相关对准
│   ├── dataset.py        # 文件命名解析与批号级成对发现
│   ├── infrared.py       # 红外抽帧、小目标候选与运动特征
│   ├── modeling.py       # 双分支随机森林与质量感知融合
│   ├── pipeline.py       # 统一特征提取管线
│   └── radar.py          # MATLAB table/普通矩阵读取与雷达特征
├── configs/default.yaml  # 抽帧、对准和模型参数
├── scripts/
│   └── inspect_dataset.py
├── tests/test_smoke.py
├── train.py              # 训练入口
├── infer.py              # 官方评测风格推理入口
├── run_demo.sh
├── requirements.txt
└── pyproject.toml
```

## 3. 环境安装

推荐 Python 3.11。

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

开发和测试环境：

```bash
pip install -e ".[dev]"
python -m pytest -q
```

### MATLAB table 读取

依赖 `mat-io` 用于读取 MATLAB `table`、`timetable` 和其他 MCOS 对象。代码会先调用：

```python
from matio import load_from_mat
```

若文件本身是普通数值矩阵，则会回退到 `scipy.io.loadmat`。如果比赛环境不允许安装 `mat-io`，可预先在 MATLAB 中将表格导出为普通矩阵、CSV 或 Parquet，再替换 `anti_air/radar.py` 的读取层。

## 4. 数据目录

建议目录如下：

```text
data/train/
├── ir_339_class-B_16：18.mp4
├── radar_339_class-B_16：18.mat
├── ir_357_class-A_16：32.mp4
├── radar_357_class-A_16：32.mat
└── ...
```

程序按 `批号` 配对，而不是依赖文件排序。中文全角冒号 `：` 和英文冒号 `:` 均可解析。

测试文件可以不带类别，只要仍能通过批号或显式命令行参数配对。`infer.py` 完全不读取文件名标签。

## 5. 先检查数据

在训练前运行：

```bash
python scripts/inspect_dataset.py \
  --data-root data/train \
  --output outputs/dataset_inventory.json
```

输出内容包括：

- 批号和训练标签；
- 雷达表格尺寸及字段名；
- 红外视频帧率、帧数、分辨率；
- 文件配对情况。

优先检查雷达字段是否包含距离、径向速度、方位、俯仰、SNR、RCS、点迹编号、航迹编号或时间戳。当前框架会自动提取最多64个可转换为标量数值的字段，但高分方案应按真实字段语义建立专门的信号级、点迹级和航迹级特征。

## 6. 训练基线

```bash
python train.py \
  --data-root data/train \
  --config configs/default.yaml \
  --output-dir outputs/baseline
```

输出：

```text
outputs/baseline/
├── model.joblib
├── radar_features.csv
├── infrared_features.csv
└── training_manifest.json
```

`training_manifest.json` 会记录每个批次的估计时间偏移、对准分数和模态质量。

> 样例数据只有极少批次时，模型分数没有统计意义。当前阶段应先验证读取、对准、特征和接口，不能把同一视频切出的窗口随机分到训练集和验证集，否则会产生严重背景泄漏。正式验证必须按批号、采集架次或场景分组。

## 7. 单对数据推理

```bash
python infer.py \
  --radar path/to/radar_test.mat \
  --ir path/to/ir_test.mp4 \
  --model outputs/baseline/model.joblib \
  --output result.json
```

也可使用：

```bash
bash run_demo.sh path/to/radar_test.mat path/to/ir_test.mp4 outputs/baseline/model.joblib
```

输出示例：

```json
{
  "label": "class-A",
  "confidence": 0.8731,
  "class_probabilities": {
    "class-A": 0.8731,
    "class-B": 0.1269
  },
  "branch_probabilities": {
    "radar": {
      "class-A": 0.91,
      "class-B": 0.09
    },
    "infrared": {
      "class-A": 0.79,
      "class-B": 0.21
    }
  },
  "fusion_weights": {
    "radar": 0.64,
    "infrared": 0.36
  },
  "alignment": {
    "offset_seconds": 2.4,
    "score": 0.71,
    "common_rate_hz": 5.0,
    "convention": "positive means infrared activity lags radar activity"
  },
  "quality": {
    "radar": 0.98,
    "infrared": 0.42
  }
}
```

统一 Python 接口位于 `infer.py`：

```python
from infer import predict

result = predict(
    radar_path="radar_test.mat",
    infrared_path="ir_test.mp4",
    model_path="outputs/baseline/model.joblib",
)
```

## 8. 当前特征设计

### 8.1 雷达分支

对每个可用数值字段计算：

- 缺失比例；
- 均值、标准差、极值、中位数和分位数；
- 四分位距和均方根；
- 一阶差分标准差和最大绝对变化；
- 全字段标准化后的行级活动能量，用于跨模态时间对准。

这些是无字段假设的通用基线。获得正式字段定义后，应增加：

- 信号级：SNR、RCS、微多普勒谱、谱熵、周期旋翼分量；
- 点迹级：点迹密度、凝聚度、虚警率、速度/加速度稳定性；
- 航迹级：速度、高度、转弯率、曲率、悬停比例、机动段持续时间。

### 8.2 红外分支

按配置稀疏抽帧，并提取：

- 灰度均值、方差、99%分位数；
- Sobel梯度能量；
- 帧差均值和高分位运动强度；
- 高运动响应小连通域数量和最大面积；
- 候选质心速度和跟踪成功率。

这套实现不依赖人工框标注，适合先做工程验证。正式高分方案建议人工修正一部分轨迹，训练红外小目标检测器，并对目标局部序列建模，而不是对整帧直接分类。

### 8.3 对准与融合

活动曲线统一重采样后做受限互相关：

```text
Δt* = argmax Corr(E_ir(t), E_radar(t - Δt))
```

当前输出固定偏移量。长序列后续应升级为分段偏移或线性时钟模型：

```text
t_ir = a · t_radar + b
```

双模态融合权重同时考虑基础权重和质量：

```text
w_radar ∝ base_radar_weight × radar_valid_ratio
w_ir    ∝ base_ir_weight    × infrared_tracking_rate
```

## 9. 配置参数

主要参数位于 `configs/default.yaml`：

```yaml
infrared:
  sample_fps: 3.0
  resize_width: 640
  max_samples: 2000

alignment:
  common_rate_hz: 5.0
  max_lag_seconds: 30.0

model:
  radar_weight: 0.60
  infrared_weight: 0.40
  n_estimators: 400
```

调参建议：

- 快速联调：`sample_fps=1`、`resize_width=320`、`max_samples=300`；
- 正式提取：适当提高抽帧率和宽度，避免小目标被过度缩小；
- 若目标运动很快，不应把抽帧率降得过低；
- `max_lag_seconds` 应根据设备启动偏移范围设置。

## 10. 防止训练/评测泄漏

必须遵守：

1. 文件名中的 `class-*` 只用于训练标签生成；
2. 模型特征不得包含文件名、目录名或批号编码；
3. 同一批次切出的所有窗口必须位于同一数据折；
4. 不得将同一场景的近邻连续片段随机分到训练和验证；
5. 归一化、特征选择和模型选择只能在训练折内完成；
6. 测试推理必须在无标签文件名条件下仍可运行。

## 11. 推荐迭代路线

### 阶段A：打通数据和可复现实验

- 确认MATLAB表格字段及采样频率；
- 生成雷达字段统计和视频样例可视化；
- 人工核验若干对准结果；
- 建立按批号/场景分组的评测脚本。

### 阶段B：建立强单模态模型

- 雷达：微多普勒时频图 + TCN/1D-CNN/轻量Transformer；
- 红外：小目标检测与连续轨迹提取 + 局部视频编码；
- 传统树模型作为可靠对照组。

### 阶段C：时窗级多模态融合

- 将对准后的雷达窗口和红外目标片段形成同一训练样本；
- 使用门控融合或Cross-Attention；
- 对低质量/缺失模态进行随机失活训练；
- 同时输出类别、置信度和未知类/OOD分数。

### 阶段D：比赛提交封装

- 固化依赖和权重；
- 支持CPU/GPU自动选择；
- 提供批量推理和单样本接口；
- 输出耗时、内存和异常日志；
- 完成算法方案、消融实验、混淆矩阵和测试报告。

## 12. 已知限制

- 当前红外检测是无监督帧差候选，不等于精确目标检测；
- 当前对准只估计固定偏移，未显式估计时钟漂移；
- 当前雷达特征尚未利用正式字段语义；
- 当前随机森林用于建立可复现基线，不代表最终网络选择；
- 极少样例下无法可靠估计泛化性能。

这些限制均在模块边界内，可在不改动训练和推理接口的前提下逐步替换。
