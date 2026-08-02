from __future__ import annotations

import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_FILENAME_RE = re.compile(
    r"^(?P<modality>ir|radar)_(?P<batch>[^_]+)(?:_(?P<label>class-[^_]+))?(?:_(?P<time>\d{1,2}[:：]\d{2}(?::\d{2})?))?$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Sample:
    batch_id: str
    radar_path: Path
    infrared_path: Path
    label: str | None = None
    start_time: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        result = asdict(self)
        result["radar_path"] = str(self.radar_path)
        result["infrared_path"] = str(self.infrared_path)
        return result


def parse_competition_filename(path: str | Path) -> dict[str, str | None]:
    p = Path(path)
    match = _FILENAME_RE.match(p.stem)
    if not match:
        raise ValueError(f"Unsupported competition filename: {p.name}")
    groups = match.groupdict()
    return {
        "modality": groups["modality"].lower(),
        "batch_id": groups["batch"],
        "label": groups.get("label"),
        "start_time": groups.get("time").replace("：", ":") if groups.get("time") else None,
    }


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_manifest(path: str | Path, *, require_labels: bool) -> list[Sample]:
    manifest_path = Path(path).resolve()
    root = manifest_path.parent
    samples: list[Sample] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"batch_id", "radar_path", "infrared_path"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for row in reader:
            batch_id = str(row["batch_id"]).strip()
            if not batch_id:
                raise ValueError("Manifest contains an empty batch_id")
            if batch_id in seen:
                raise ValueError(f"Duplicate batch_id in manifest: {batch_id}")
            seen.add(batch_id)
            label = str(row.get("label", "")).strip() or None
            if require_labels and label is None:
                raise ValueError(f"Training sample {batch_id} has no label")
            radar_path = _resolve_path(root, str(row["radar_path"]).strip())
            infrared_path = _resolve_path(root, str(row["infrared_path"]).strip())
            if not radar_path.is_file():
                raise FileNotFoundError(f"Radar file not found: {radar_path}")
            if not infrared_path.is_file():
                raise FileNotFoundError(f"Infrared file not found: {infrared_path}")
            samples.append(
                Sample(
                    batch_id=batch_id,
                    radar_path=radar_path,
                    infrared_path=infrared_path,
                    label=label,
                    start_time=str(row.get("start_time", "")).strip() or None,
                )
            )
    if not samples:
        raise ValueError(f"No samples found in manifest: {manifest_path}")
    return samples


def discover_samples(
    root: str | Path,
    *,
    require_labels: bool,
    strict_pairs: bool = True,
) -> list[Sample]:
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    records: dict[str, dict[str, object]] = {}
    candidates: Iterable[Path] = [*root.rglob("*.mat"), *root.rglob("*.mp4")]
    for path in sorted(candidates):
        try:
            meta = parse_competition_filename(path)
        except ValueError:
            continue
        batch_id = str(meta["batch_id"])
        record = records.setdefault(
            batch_id,
            {"radar": None, "ir": None, "label": None, "start_time": None},
        )
        modality = str(meta["modality"])
        if record[modality] is not None:
            raise ValueError(f"Duplicate {modality} file for batch {batch_id}")
        record[modality] = path.resolve()
        if meta["label"] is not None:
            if record["label"] not in {None, meta["label"]}:
                raise ValueError(f"Inconsistent labels for batch {batch_id}")
            record["label"] = meta["label"]
        record["start_time"] = record["start_time"] or meta["start_time"]

    samples: list[Sample] = []
    incomplete: list[str] = []
    for batch_id, record in sorted(records.items()):
        if record["radar"] is None or record["ir"] is None:
            incomplete.append(batch_id)
            continue
        label = str(record["label"]) if record["label"] is not None else None
        if require_labels and label is None:
            raise ValueError(f"Training sample {batch_id} has no class label in filename")
        samples.append(
            Sample(
                batch_id=batch_id,
                radar_path=Path(record["radar"]),
                infrared_path=Path(record["ir"]),
                label=label,
                start_time=str(record["start_time"]) if record["start_time"] else None,
            )
        )
    if strict_pairs and incomplete:
        raise ValueError(f"Unpaired batches found: {', '.join(incomplete)}")
    if not samples:
        raise ValueError(f"No paired radar/infrared samples found under {root}")
    return samples


def resolve_samples(
    *,
    data_root: str | Path | None = None,
    manifest: str | Path | None = None,
    require_labels: bool,
    strict_pairs: bool = True,
) -> list[Sample]:
    if bool(data_root) == bool(manifest):
        raise ValueError("Provide exactly one of data_root or manifest")
    if manifest:
        return load_manifest(manifest, require_labels=require_labels)
    return discover_samples(data_root, require_labels=require_labels, strict_pairs=strict_pairs)  # type: ignore[arg-type]


def write_manifest(samples: list[Sample], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["batch_id", "label", "radar_path", "infrared_path", "start_time"],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(sample.to_dict())
