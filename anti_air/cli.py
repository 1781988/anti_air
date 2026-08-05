from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from .artifacts import create_submission, environment_info, prepare_run_dir, sha256_file, write_json
from .config import load_config
from .data import class_counts, discover_samples
from .evaluation import evaluate_records, predict_cached_records
from .inference import predict_pair
from .inspect import inspect_samples
from .preprocess import preprocess_dataset
from .trainer import train_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Radar-infrared competition pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml")
    subcommands = parser.add_subparsers(dest="command", required=True)

    for name in ("all", "inspect"):
        command = subcommands.add_parser(name)
        command.add_argument("--data", default=None)
        command.add_argument("--profile", choices=["quick", "cpu", "competition", "auto"], default=None)
        if name == "all":
            command.add_argument("--rebuild-cache", action="store_true")

    infer = subcommands.add_parser("infer")
    infer.add_argument("--radar", required=True)
    infer.add_argument("--ir", required=True)
    infer.add_argument("--model", default="runs/latest/model.pt")
    infer.add_argument("--batch-id", default="inference")
    infer.add_argument("--output", default="prediction.json")

    clean = subcommands.add_parser("clean-cache")
    clean.add_argument("--cache", default=None)
    return parser


def _resolve_path(root: Path, configured: str, override: str | None) -> Path:
    value = Path(override or configured).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def command_inspect(args: argparse.Namespace, root: Path) -> int:
    config = load_config(args.config, profile=args.profile)
    data = _resolve_path(root, config["paths"]["data"], args.data)
    samples = discover_samples(data)
    inspection = inspect_samples(samples, config)
    print(json.dumps(inspection, ensure_ascii=False, indent=2))
    return 0 if inspection["status"] == "ok" else 2


def command_all(args: argparse.Namespace, root: Path) -> int:
    total_started = time.perf_counter()
    config = load_config(args.config, profile=args.profile)
    data = _resolve_path(root, config["paths"]["data"], args.data)
    cache = _resolve_path(root, config["paths"]["cache"], None)
    run = prepare_run_dir(_resolve_path(root, config["paths"]["run"], None))
    print(f"profile={config['profile']} data={data}")
    samples = discover_samples(data)
    inspection = inspect_samples(samples, config)
    if inspection["status"] != "ok":
        result = {"status": "failed_data_inspection", "inspection": inspection}
        write_json(result, run / "result.json")
        raise RuntimeError("Data inspection failed; see runs/latest/result.json")

    preprocessing = preprocess_dataset(
        samples,
        config,
        cache,
        rebuild=bool(args.rebuild_cache),
    )
    evaluation = evaluate_records(preprocessing.records, config, preprocessing.radar_schema)
    model_path = run / "model.pt"
    _, training = train_model(
        preprocessing.records,
        config,
        preprocessing.radar_schema,
        output_path=model_path,
    )
    training_predictions = predict_cached_records(model_path, preprocessing.records)
    result: dict[str, Any] = {
        "schema_version": "2.0",
        "status": "complete",
        "profile": config["profile"],
        "environment": environment_info(),
        "configuration": config,
        "data": {
            "directory": str(data),
            "records": len(samples),
            "class_record_counts": class_counts(samples),
            "inspection": inspection,
        },
        "preprocessing": preprocessing.summary(),
        "evaluation": evaluation,
        "final_training": training,
        "training_set_predictions": {
            "warning": "Resubstitution predictions are for pipeline diagnostics only, not generalization performance.",
            "items": training_predictions,
        },
        "limitations": [
            "Metrics are statistically valid only when every class has at least two independent recording batches.",
            "The current labels in filenames are used only as training targets and are excluded from model inputs.",
            "Official organizer scoring/output specifications were not present in the supplied material; adapt the final output adapter when released.",
        ],
        "runtime_seconds": time.perf_counter() - total_started,
    }
    result_path = run / "result.json"
    write_json(result, result_path)
    result["model_sha256"] = sha256_file(model_path)
    write_json(result, result_path)
    submission = create_submission(
        repository_root=root,
        model_path=model_path,
        result_path=result_path,
        output_path=run / "submission.zip",
    )
    print("\nComplete")
    print(f"  model      : {model_path}")
    print(f"  result     : {result_path}")
    print(f"  submission : {submission}")
    print(f"  evaluation : {evaluation['status']}")
    return 0


def command_infer(args: argparse.Namespace, root: Path) -> int:
    result = predict_pair(
        radar_path=args.radar,
        infrared_path=args.ir,
        model_path=args.model,
        batch_id=args.batch_id,
    )
    write_json(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.command == "inspect":
        return command_inspect(args, root)
    if args.command == "all":
        return command_all(args, root)
    if args.command == "infer":
        return command_infer(args, root)
    if args.command == "clean-cache":
        config = load_config(args.config)
        cache = _resolve_path(root, config["paths"]["cache"], args.cache)
        shutil.rmtree(cache, ignore_errors=True)
        print(f"Removed cache: {cache}")
        return 0
    raise AssertionError(args.command)
