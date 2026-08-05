# 数据目录

将比赛提供的整个 `train` 文件夹复制到本目录，最终结构必须为：

```text
data/
└── train/
    ├── radar_339_class-B_16：18.mat
    ├── ir_339_class-B_16：18.mp4
    ├── radar_357_class-A_*.mat
    ├── ir_357_class-A_*.mp4
    └── ...
```

程序只读取 `data/train/` 第一层的 `.mat` 和 `.mp4`，不会递归猜测路径。
原始数据不会提交到 Git。
