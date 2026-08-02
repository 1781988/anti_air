from __future__ import annotations

import argparse
from pathlib import Path

from anti_air.config import load_config
from anti_air.dataset import resolve_samples, write_manifest
from anti_air.feature_store import build_feature_cache
from anti_air.utils import set_global_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract aligned radar/infrared window features")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root")
    source.add_argument("--manifest")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/features")
    parser.add_argument("--unlabeled", action="store_true", help="Allow files without class labels")
    parser.add_argument("--force", action="store_true", help="Ignore per-record cache")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    set_global_seed(int(config["seed"]))
    samples = resolve_samples(
        data_root=args.data_root,
        manifest=args.manifest,
        require_labels=not args.unlabeled,
        strict_pairs=True,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(samples, output / "resolved_manifest.csv")
    tables = build_feature_cache(samples, config, output, force=args.force)
    print(
        f"Feature extraction complete: records={tables.manifest['sample_count']} "
        f"windows={tables.manifest['window_count']} output={output}"
    )


if __name__ == "__main__":
    main()
