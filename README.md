# anti_air 2.0：雷达—红外多模态目标类型识别

本仓库面向“探测识别赛道—科目1：复杂环境下的目标类型智能识别”。任务是在低空、低速、复杂背景和鸟类/风筝等干扰条件下，结合雷达时序与红外视频，输出整段记录的目标类别和置信度。

版本2.0不再采用“手工统计特征 + ExtraTrees”作为最终模型，而是使用可训练的PyTorch多模态网络：

```text
雷达MAT/table → 固定时长窗口 → 1D TCN雷达编码器 ───────┐
                                                     ├→ 质量门控融合 → 类别概率
红外MP4 → 小目标候选/跟踪裁剪 → MobileNet + 时序注意力 ─┘
                  ↑
         雷达—红外活动曲线互相关对准
```

## 1. 为什么旧版本训练很快

旧提交包中的模型实际是3个ExtraTrees分类器。上传结果显示只有：

- 3个独立记录；
- 259个滑动窗口；
- `class-A` 2个记录，`class-B` 1个记录；
- 评价覆盖率只有2/3；
- Accuracy为0.5，Macro-F1为0.3333。

ExtraTrees在几百个样本上几秒完成是正常现象，不代表深度学习训练已充分进行。更关键的问题是：唯一的`class-B`记录不能同时出现在训练和独立验证中，因此旧报告中的两折验证只覆盖了部分记录/类别，不能作为正式比赛性能。

新版本做了三项修正：

1. 使用雷达TCN、红外预训练MobileNet和时序Transformer进行真实梯度训练；
2. 每个epoch打印耗时、损失和F1，不再把特征提取时间与模型训练混为一谈；
3. 当独立记录不足时，评价状态明确写为`diagnostic_only`，不会把不完整验证标成正式有效结果。

> 训练速度仍取决于数据量、GPU、配置和缓存。第一次运行需解码视频和构建缓存；相同数据再次运行会复用`.cache/`，明显加快是正常的。使用`--rebuild-cache`可强制重新处理。

## 2. 开源框架调研与选型

### BasicIRSTD

`XinyiYing/BasicIRSTD`是面向红外小目标检测的PyTorch工具箱，包含多种IRSTD模型、训练和指标实现。它与本赛题红外小目标部分最相关。但其标准训练依赖像素级目标掩码，而当前数据只提供整段类别，因此本仓库没有直接复制其训练器，而是保留“小目标检测器可替换接口”。后续获得目标掩码或预训练权重后，可将当前候选提取模块替换为BasicIRSTD模型。

### MMDetection

OpenMMLab的MMDetection适合目标框、实例分割和半监督检测。如果后续人工标注红外目标框/掩码，可用MMDetection训练检测器，再把轨迹裁剪送入本仓库的时序分类器。当前无目标框时直接套用会缺少监督信号。

### MMAction2、PyTorchVideo与VideoMAE

这些框架提供TSM、SlowFast、X3D、VideoSwin和VideoMAE等视频模型。它们适合替换本仓库的红外时序编码器，但完整框架依赖较重，而且通用动作预训练与“云背景中的几像素红外目标”存在明显域差异。本版本采用更轻的MobileNetV3 + 时序注意力，后续可在完整数据上比较VideoMAE/TSM。

### OpenRadar

OpenRadar提供距离、Doppler、角度、CFAR和跟踪等毫米波雷达DSP模块。如果组委会后续提供原始ADC或雷达立方体，可以接入OpenRadar。当前`.mat`是MATLAB table/时序字段，不是原始ADC，因此直接引入OpenRadar不会增加有效信息。

### 当前选择

本仓库采用自包含PyTorch工程，而不是整体依赖大型外部框架，原因是：

- 兼容未知MATLAB table字段；
- 同时处理视频级标签、雷达时序和异步对准；
- 提交包更小，评测接口更简单；
- 红外编码器、雷达编码器和融合层均可独立替换。

## 3. 仓库结构

```text
anti_air/
├── anti_air/              # 数据、解码、对准、预处理、模型、训练、评价
├── data/
│   ├── README.md
│   └── train/.gitkeep     # 将本地train文件夹覆盖/复制到这里
├── tests/
├── config.yaml            # 唯一配置文件，包含quick/cpu/competition三套配置
├── main.py                # 唯一Python入口
├── setup.sh               # 一键环境配置
├── run.sh                 # 比赛单对推理入口
├── requirements.txt
└── README.md
```

## 4. 服务器环境

推荐：

- Ubuntu 20.04/22.04/24.04；
- Python 3.11或3.12；
- CUDA GPU，显存建议8GB以上；
- 纯CPU也可运行，但会自动使用较轻的`cpu`配置；
- FFmpeg用于稳定读取MP4。

安装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg python3 python3-venv
```

下载和配置：

```bash
git clone https://github.com/1781988/anti_air.git
cd anti_air
bash setup.sh
source .venv/bin/activate
```

`setup.sh`会创建虚拟环境、安装依赖、编译检查并运行单元测试。

## 5. 配置数据集

把本地整个`train`文件夹复制到仓库的`data/`下：

```text
anti_air/
└── data/
    └── train/
        ├── radar_339_class-B_16：18.mat
        ├── ir_339_class-B_16：18.mp4
        ├── radar_357_class-A_*.mat
        ├── ir_357_class-A_*.mp4
        └── ...
```

文件必须直接位于`data/train/`第一层。程序不会递归猜路径。

检查文件：

```bash
find data/train -maxdepth 1 -type f \( -name '*.mat' -o -name '*.mp4' \) -printf '%f\n' | sort
```

## 6. 最简完整运行方式

正式运行只需要一条命令：

```bash
python main.py all
```

该命令自动完成：

```text
数据检查 → 视频/雷达预处理 → 异步对准 → 窗口缓存
→ 独立批次交叉验证 → 全量最终训练 → 训练集诊断预测
→ result.json → submission.zip
```

配置选择规则：

- 有CUDA：自动使用`competition`；
- 无CUDA：自动使用`cpu`；
- 首次只想验证环境：

```bash
python main.py all --profile quick
```

强制重新解码并重建缓存：

```bash
python main.py all --rebuild-cache
```

数据不在默认位置时：

```bash
python main.py all --data /absolute/path/to/train
```

## 7. 输出文件

为了减少文件数量，正常运行只在`runs/latest/`保留3个文件：

```text
runs/latest/
├── model.pt        # 最终多模态模型、类别、雷达字段schema和完整配置
├── result.json     # 数据检查、训练历史、评价指标、混淆矩阵、预测和环境信息
└── submission.zip  # 比赛交付压缩包
```

中间张量缓存在`.cache/anti_air/`，不属于比赛输出。删除缓存：

```bash
python main.py clean-cache
```

## 8. 分析结果

查看概要：

```bash
python - <<'PY'
import json
r = json.load(open('runs/latest/result.json', encoding='utf-8'))
print('profile:', r['profile'])
print('records:', r['data']['records'])
print('classes:', r['data']['class_record_counts'])
print('evaluation status:', r['evaluation']['status'])
print('coverage:', r['evaluation'].get('coverage'))
print('metrics:', r['evaluation'].get('metrics'))
print('training seconds:', r['final_training']['elapsed_seconds'])
PY
```

评价状态：

- `valid`：每个类别至少有2个独立记录，评价覆盖全部记录和类别；
- `diagnostic_only`：只能形成部分无泄漏折，指标只用于排查；
- `insufficient_independent_records`：无法形成合法验证折。

必须优先看记录级`macro_f1`、`balanced_accuracy`、`coverage`和混淆矩阵，不能用训练集回代结果作为比赛性能。

## 9. 单对推理

仓库环境下：

```bash
python main.py infer \
  --radar 'radar_test.mat' \
  --ir 'ir_test.mp4' \
  --model runs/latest/model.pt \
  --output prediction.json
```

比赛提交包中：

```bash
bash run.sh radar_test.mat ir_test.mp4 result.json
```

输出包括类别、置信度、各类别概率、窗口数、时间对准结果以及雷达/红外门控权重。

## 10. 比赛提交包

`python main.py all`会自动生成：

```text
runs/latest/submission.zip
```

压缩包包含：

- `model.pt`：完整模型；
- `result.json`：测试/评价报告；
- 算法源代码；
- 配置文件；
- 环境依赖；
- `run.sh`统一推理入口；
- README与安装脚本。

独立验证提交包：

```bash
mkdir -p /tmp/anti_air_submit
cd /tmp/anti_air_submit
unzip /path/to/anti_air/runs/latest/submission.zip
cd anti_air_submission
bash setup.sh
source .venv/bin/activate
bash run.sh /path/radar_test.mat /path/ir_test.mp4 result.json
cat result.json
```

## 11. 重要限制

1. 当前样例只有3个独立记录，任何深度网络都可能过拟合；增加滑动窗口不能等价于增加独立样本。
2. 文件名中的`class-*`只作为训练标签，不进入模型输入。
3. 现有材料没有给出组委会最终评分公式和精确输出协议；本仓库使用记录级Macro-F1、Balanced Accuracy等标准指标，并提供通用JSON推理结果。正式接口发布后只需调整输出适配层。
4. 首次运行若下载MobileNet预训练权重失败，会自动使用随机初始化并在终端警告；比赛提交模型本身不需要联网。
