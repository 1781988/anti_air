from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate competition test report in Markdown")
    parser.add_argument("--metrics", default="outputs/evaluation/metrics.json")
    parser.add_argument("--training-summary", default="outputs/model/training_summary.json")
    parser.add_argument("--output", default="outputs/report/test_report.md")
    args = parser.parse_args()
    metrics = _load(Path(args.metrics))
    training = _load(Path(args.training_summary))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    class_counts = training.get("class_record_counts", {})
    status = metrics.get("status", "not_evaluated")
    lines = [
        "# 复杂环境下目标类型智能识别算法测试报告",
        "",
        "## 1. 测试对象",
        "",
        "本报告对应 `anti_air` 雷达—红外双模态目标识别算法。算法由雷达分支、红外小目标运动分支和特征级融合分支组成，输出整段记录的目标类别及置信度。",
        "",
        "## 2. 数据与划分",
        "",
        f"- 训练记录数：{training.get('records', '未知')}",
        f"- 训练窗口数：{training.get('windows', '未知')}",
        f"- 类别：{training.get('classes', [])}",
        f"- 各类别记录数：{class_counts}",
        "- 验证原则：以批号为分组单位，禁止同一批次窗口跨训练集和验证集。",
        "",
        "## 3. 评价结果",
        "",
        f"- 评价状态：`{status}`",
        f"- 有效折数：{metrics.get('valid_folds', 0)}",
        f"- 评价覆盖率：{metrics.get('evaluation_coverage', '不适用')}",
        f"- Accuracy：{metrics.get('accuracy', '不适用')}",
        f"- Balanced Accuracy：{metrics.get('balanced_accuracy', '不适用')}",
        f"- Macro-F1：{metrics.get('macro_f1', '不适用')}",
        f"- Weighted-F1：{metrics.get('weighted_f1', '不适用')}",
        f"- Log Loss：{metrics.get('log_loss', '不适用')}",
        f"- Macro-F1 95%置信区间：{metrics.get('macro_f1_95ci', '不适用')}",
        "",
        "## 4. 结果解释",
        "",
    ]
    if status == "insufficient_grouped_data":
        lines.extend(
            [
                "当前批次数不足，无法形成同时满足‘训练折包含全部类别’和‘批号完全隔离’的验证折。系统拒绝使用随机窗口划分，以避免背景和航迹泄漏造成虚高结果。",
                "正式数据增加后重新运行 `evaluate.py` 即可生成有效指标。",
            ]
        )
    else:
        lines.append("上述结果来自批号级隔离验证。应结合 `record_predictions.csv` 与 `confusion_matrix.csv` 分析具体易混类别。")
    lines.extend(
        [
            "",
            "## 5. 复现命令",
            "",
            "```bash",
            "python extract_features.py --data-root data/train --config configs/default.yaml --output-dir outputs/features",
            "python evaluate.py --features outputs/features --config configs/default.yaml --output-dir outputs/evaluation",
            "python train.py --features outputs/features --config configs/default.yaml --output-dir outputs/model",
            "```",
            "",
            "## 6. 交付文件",
            "",
            "- 算法设计：`docs/algorithm_design.md`",
            "- 模型：`outputs/model/model.joblib`",
            "- 源代码：仓库全部代码",
            "- 指标：`outputs/evaluation/metrics.json`",
            "- 预测明细：`outputs/evaluation/record_predictions.csv`",
            "- 混淆矩阵：`outputs/evaluation/confusion_matrix.csv`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
