# anti_air 2.1：雷达—红外多模态目标类型识别

本仓库面向“探测识别赛道—科目1：复杂环境下的目标类型智能识别”。任务是在低空、低速、复杂背景以及鸟类、风筝等干扰条件下，综合利用雷达时序和红外视频，输出整段记录的目标类别及置信度。

当前版本采用可训练的PyTorch多模态网络：

```text
雷达MAT/table → 固定时长窗口 → 1D TCN雷达编码器 ───────┐
                                                     ├→ 质量门控融合 → 类别概率
红外MP4 → 小目标候选/跟踪裁剪 → MobileNet + 时序注意力 ─┘
                  ↑
         雷达—红外活动曲线互相关对准
```

## 1. 环境管理原则

项目只使用Conda环境：

```text
环境名称：anti-air
Python版本：3.11
环境定义：environment.yml
```

不再创建或使用`.venv`。服务器`base`环境中的Python 3.13不会参与项目运行。

下面三种方式均可运行项目：

```bash
# 推荐：完全不需要激活环境
bash train.sh

# 也可以手动激活
conda activate anti-air
python main.py all

# 或直接使用conda run
conda run -n anti-air python main.py all
```

## 2. 仓库结构

```text
anti_air/
├── anti_air/              # 数据、解码、对准、模型、训练和评价
├── data/
│   └── train/.gitkeep     # 将本地train文件夹放到这里
├── tests/
├── config.yaml            # quick/cpu/competition三套配置
├── environment.yml        # Conda环境定义
├── main.py                # Python统一入口
├── setup.sh               # 一键创建或更新Conda环境
├── train.sh               # 一键训练、评价和打包
├── run.sh                 # 比赛单对推理入口
├── pyproject.toml
└── README.md
```

## 3. 从零配置服务器

### 3.1 已安装Conda

确认：

```bash
conda --version
```

下载代码：

```bash
cd ~/GMY
rm -rf anti_air
git clone https://github.com/1781988/anti_air.git
cd anti_air
```

创建环境：

```bash
bash setup.sh
```

`setup.sh`会自动完成：

```text
创建或更新anti-air环境
→ 固定Python 3.11
→ 安装Conda版FFmpeg
→ 安装项目和PyTorch依赖
→ 编译检查
→ 单元测试
→ 输出Python、PyTorch、CUDA和FFmpeg状态
```

整个过程不依赖当前shell是否显示`(base)`，也不需要执行`conda deactivate`。

### 3.2 未安装Conda

安装Miniconda：

```bash
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
```

重新打开终端后：

```bash
cd ~/GMY/anti_air
bash setup.sh
```

### 3.3 验证环境

```bash
conda run -n anti-air python --version
conda run -n anti-air python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
conda run -n anti-air ffmpeg -version | head
```

应看到Python 3.11。

如果服务器有NVIDIA GPU但`torch.cuda.is_available()`为`False`，按照PyTorch官方安装选择器取得适配该服务器的安装命令，并在`anti-air`环境中执行。也可设置官方轮子索引后重新运行安装脚本：

```bash
TORCH_INDEX_URL='<PyTorch官方选择器给出的index-url>' bash setup.sh
```

不要自行猜测CUDA轮子版本。

### 3.4 环境损坏时重建

```bash
conda env remove -n anti-air -y
bash setup.sh
```

## 4. 配置数据集

把本地完整的`train`文件夹复制到仓库的`data/`下：

```text
anti_air/
└── data/
    └── train/
        ├── radar_339_class-B_16：18.mat
        ├── ir_339_class-B_16：18.mp4
        ├── radar_357_class-A_*.mat
        ├── ir_357_class-A_*.mp4
        ├── radar_897_class-A_*.mat
        └── ir_897_class-A_*.mp4
```

文件必须直接位于`data/train/`第一层。

服务器已有数据时：

```bash
rm -rf data/train
cp -a /你的数据路径/train data/train
```

检查：

```bash
find data/train -maxdepth 1 -type f \( -name '*.mat' -o -name '*.mp4' \) -printf '%f\n' | sort
```

## 5. 最简运行方法

### 5.1 快速联调

```bash
bash train.sh quick
```

用于验证：

```text
数据读取
视频解码
雷达MAT/table解析
跨模态对准
模型前向和反向传播
评价与提交包生成
```

`quick`只用于检查流程，不作为正式比赛性能。

### 5.2 正式训练

自动选择GPU或CPU配置：

```bash
bash train.sh
```

明确使用GPU比赛配置：

```bash
bash train.sh competition
```

纯CPU：

```bash
bash train.sh cpu
```

第一次运行会自动重建缓存。数据或预处理配置发生变化后，强制重建：

```bash
bash train.sh competition --rebuild-cache
```

### 5.3 不使用包装脚本

等价命令：

```bash
conda run --no-capture-output -n anti-air \
  python main.py all --profile competition --rebuild-cache
```

## 6. 完整流程

`bash train.sh`会自动执行：

```text
数据检查
→ 雷达MAT/table读取
→ 红外视频解码和小目标轨迹裁剪
→ 雷达—红外时间对准
→ 窗口缓存
→ 独立批次无泄漏评价
→ 全量最终训练
→ 结果汇总
→ 比赛提交ZIP生成
```

相同数据再次运行会复用`.cache/anti_air/`，因此第二次明显更快是正常的。删除缓存：

```bash
conda run -n anti-air python main.py clean-cache
```

## 7. 输出文件

正常运行只保留三个主要文件：

```text
runs/latest/
├── model.pt
├── result.json
└── submission.zip
```

- `model.pt`：最终模型、类别映射、雷达字段Schema和配置；
- `result.json`：数据检查、训练历史、评价指标、混淆矩阵、环境和预测；
- `submission.zip`：比赛交付压缩包。

## 8. 查看性能

```bash
conda run -n anti-air python - <<'PY'
import json

with open('runs/latest/result.json', encoding='utf-8') as f:
    result = json.load(f)

print('profile:', result['profile'])
print('records:', result['data']['records'])
print('classes:', result['data']['class_record_counts'])
print('evaluation status:', result['evaluation']['status'])
print('coverage:', result['evaluation'].get('coverage'))
print('metrics:', result['evaluation'].get('metrics'))
print('training seconds:', result['final_training']['elapsed_seconds'])
print('epochs:', result['final_training']['epochs_completed'])
PY
```

评价状态：

- `valid`：各类别具有足够独立批次，评价覆盖完整；
- `diagnostic_only`：只能形成部分合法无泄漏验证折；
- `insufficient_independent_records`：无法形成合法验证折。

当前样例只有3个独立批次，其中`class-B`只有一个批次，因此指标通常只能作为工程诊断，不能代表正式泛化性能。

## 9. 单对推理

不需要激活环境：

```bash
ANTI_AIR_MODEL=runs/latest/model.pt \
bash run.sh \
  '/测试数据/radar_test.mat' \
  '/测试数据/ir_test.mp4' \
  prediction.json
```

手动运行：

```bash
conda run --no-capture-output -n anti-air \
  python main.py infer \
  --radar '/测试数据/radar_test.mat' \
  --ir '/测试数据/ir_test.mp4' \
  --model runs/latest/model.pt \
  --output prediction.json
```

## 10. 比赛提交包

完整训练后自动生成：

```text
runs/latest/submission.zip
```

压缩包包含：

```text
model.pt
result.json
算法源代码
config.yaml
environment.yml
pyproject.toml
setup.sh
run.sh
README.md
单元测试
```

独立验证：

```bash
rm -rf /tmp/anti_air_submit
mkdir -p /tmp/anti_air_submit
cd /tmp/anti_air_submit
unzip ~/GMY/anti_air/runs/latest/submission.zip
cd anti_air_submission

bash setup.sh
bash run.sh \
  '/测试数据/radar_test.mat' \
  '/测试数据/ir_test.mp4' \
  result.json
cat result.json
```

## 11. 常见环境问题

### 当前终端一直显示`(base)`

不影响运行。包装脚本始终调用：

```text
conda run -n anti-air ...
```

因此不会使用`base`中的Python 3.13。

### `.venv/bin/activate`不存在

正常。新版不再创建`.venv`，不要再执行：

```bash
source .venv/bin/activate
```

### 查看项目实际使用的Python

```bash
conda run -n anti-air python -c "import sys; print(sys.executable); print(sys.version)"
```

### 更新代码后同步环境

```bash
git pull --ff-only origin main
bash setup.sh
```

`setup.sh`会更新已有环境，而不是重复创建。
