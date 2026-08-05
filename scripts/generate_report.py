from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "不适用"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _branch_table(branch_metrics: dict[str, Any]) -> list[str]:
    if not branch_metrics:
        return ["未生成分支消融指标。"]
    lines = [
        "| 分支 | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | Log Loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    names = {"radar": "雷达", "infrared": "红外", "fusion": "特征级融合"}
    for key in ("radar", "infrared", "fusion"):
        item = branch_metrics.get(key, {})
        lines.append(
            "| {name} | {accuracy} | {balanced} | {macro} | {weighted} | {loss} |".format(
                name=names[key],
                accuracy=_format(item.get("accuracy")),
                balanced=_format(item.get("balanced_accuracy")),
                macro=_format(item.get("macro_f1")),
                weighted=_format(item.get("weighted_f1")),
                loss=_format(item.get("log_loss")),
            )
        )
    return lines


def _status_explanation(status: str, metrics: dict[str, Any]) -> list[str]:
    reason = metrics.get("reason")
    if status == "insufficient_grouped_data":
        return [
            "当前独立批次数不足，无法构造任何同时满足‘批号隔离’与‘训练折包含测试类别’的验证折。",
            "系统拒绝随机拆分同一记录的时间窗口，以避免背景、轨迹和场景泄漏。",
        ]
    if status == "insufficient_class_coverage":
        return [
            "本次交叉验证没有覆盖全部类别，因此下面的数值只能作为诊断结果，不能作为正式比赛泛化性能。",
            f"未被独立测试的类别：{metrics.get('unevaluated_classes', [])}。",
            "当前样例中某类别只有一个独立批次；将其作为测试集时，训练集中不存在该类别，因此该折必须剔除。",
        ]
    if status == "partial_record_coverage":
        return [
            "部分独立批次无法在不发生类别泄漏的条件下参与验证，当前结果仅覆盖部分记录。",
            f"未参与验证的批次：{metrics.get('unevaluated_batches', [])}。",
        ]
    if status == "limited_folds":
        return [
            "有效分组折数低于配置要求，结果方差较大，只能作为阶段性诊断。",
        ]
    if status == "ok":
        return [
            "验证覆盖全部类别和全部独立批次，且有效折数满足配置要求，可将主指标作为当前数据条件下的主要评价结果。",
        ]
    return [str(reason or "尚未完成有效评价。")]


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
    status = str(metrics.get("status", "not_evaluated"))
    eligible = bool(metrics.get("eligible_for_primary_score", False))
    confidence_interval = metrics.get("macro_f1_95ci", {})
    lines = [
        "# 复杂环境下目标类型智能识别算法测试报告",
        "",
        "## 1. 测试对象",
        "",
        "本报告对应 `anti_air` 雷达—红外双模态目标识别算法。算法由雷达分支、红外小目标运动分支和特征级融合分支组成，输出整段记录的目标类别及置信度。",
        "",
        "## 2. 数据与独立样本规模",
        "",
        f"- 独立训练记录数：{training.get('records', '未知')}",
        f"- 由记录切分得到的训练窗口数：{training.get('windows', '未知')}",
        f"- 类别：{training.get('classes', [])}",
        f"- 各类别独立记录数：{class_counts}",
        f"- 每类最少独立记录数：{training.get('minimum_records_per_class', '未知')}",
        f"- 置信度可靠性系数：{_format(training.get('record_level_confidence_reliability'))}",
        f"- 有效概率平滑系数：{_format(training.get('effective_probability_smoothing'))}",
        "- 验证原则：以批号为分组单位，禁止同一批次窗口跨训练集和验证集。",
        "",
        "## 3. 评价有效性",
        "",
        f"- 评价状态：`{status}`",
        f"- 是否可作为主性能指标：`{eligible}`",
        f"- 指标解释级别：`{metrics.get('metric_interpretation', '未知')}`",
        f"- 尝试折数：{metrics.get('attempted_folds', 0)}",
        f"- 有效折数：{metrics.get('valid_folds', 0)}",
        f"- 独立记录覆盖率：{_format(metrics.get('evaluation_coverage'))}",
        f"- 类别覆盖率：{_format(metrics.get('class_coverage'))}",
        f"- 已评价类别：{metrics.get('evaluated_classes', [])}",
        f"- 未评价类别：{metrics.get('unevaluated_classes', [])}",
        f"- 未评价批次：{metrics.get('unevaluated_batches', [])}",
        "",
        "### 有效性结论",
        "",
        *_status_explanation(status, metrics),
        "",
        "## 4. 集成模型诊断指标",
        "",
        "> 当“是否可作为主性能指标”为 `False` 时，本节数值仅用于发现问题，不代表完整比赛泛化性能。",
        "",
        f"- Accuracy：{_format(metrics.get('accuracy'))}",
        f"- Balanced Accuracy：{_format(metrics.get('balanced_accuracy'))}",
        f"- Macro-F1：{_format(metrics.get('macro_f1'))}",
        f"- Weighted-F1：{_format(metrics.get('weighted_f1'))}",
        f"- Log Loss：{_format(metrics.get('log_loss'))}",
        f"- Macro-F1 95%置信区间：{confidence_interval}",
        f"- 训练折多数类基线：{metrics.get('majority_baseline', {})}",
        "",
        "## 5. 模态分支消融",
        "",
        *_branch_table(metrics.get("branch_metrics", {})),
        "",
        "分支消融用于判断误差主要来自雷达特征、红外目标跟踪还是跨模态融合。只有在类别覆盖完整时，才可据此调节融合权重。",
        "",
        "## 6. 当前结果对应的改进结论",
        "",
    ]
    if eligible:
        lines.extend(
            [
                "- 当前评价已具备完整覆盖，可优先比较集成模型与三个单模态/融合分支的 Macro-F1。",
                "- 若某单分支持续优于集成模型，应重新估计 `model.branch_weights`，而不是继续使用固定经验权重。",
            ]
        )
    else:
        lines.extend(
            [
                "- 当前独立记录规模不足，不能通过增加窗口数量替代增加独立批次；窗口数不是独立样本数。",
                "- 在补充数据前，模型概率已启用小样本平滑，输出置信度不应解释为校准概率。",
                "- 至少应保证每个类别有多个可独立留出的批次，再根据分支消融结果调整模型结构与融合权重。",
            ]
        )
    lines.extend(
        [
            "",
            "## 7. 复现命令",
            "",
            "```bash",
            "python extract_features.py --data-root data/train --config configs/default.yaml --output-dir outputs/features --force",
            "python evaluate.py --features outputs/features --config configs/default.yaml --output-dir outputs/evaluation",
            "python train.py --features outputs/features --config configs/default.yaml --output-dir outputs/model",
            "python scripts/generate_report.py --metrics outputs/evaluation/metrics.json --training-summary outputs/model/training_summary.json --output outputs/report/test_report.md",
            "```",
            "",
            "## 8. 交付文件",
            "",
            "- 算法设计：`docs/algorithm_design.md`",
            "- 最终模型：`model/model.joblib`",
            "- 模型摘要：`model/training_summary.json`",
            "- 模型卡：`model/MODEL_CARD.md`",
            "- 总体指标：`evaluation/metrics.json`",
            "- 分折明细：`evaluation/folds.json`",
            "- 记录级预测：`evaluation/record_predictions.csv`",
            "- 窗口级预测：`evaluation/window_predictions.csv`",
            "- 混淆矩阵：`evaluation/confusion_matrix.csv`",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report written to {output}")


if __name__ == "__main__":
    main()
