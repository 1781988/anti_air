from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_FILENAME_RE = re.compile(
    r"^(?P<modality>ir|radar)_(?P<batch>[^_]+)_(?P<label>class-[^_]+)_(?P<time>.+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Sample:
    batch_id: str
    label: str
    radar_path: Path
    infrared_path: Path
    start_time: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["radar_path"] = str(self.radar_path)
        result["infrared_path"] = str(self.infrared_path)
        return result


def parse_filename(path: str | Path) -> dict[str, str]:
    p = Path(path)
    match = _FILENAME_RE.match(p.stem)
    if not match:
        raise ValueError(
            f"Unsupported filename {p.name!r}; expected "
            "ir_批号_class-X_起始时间.mp4 or radar_批号_class-X_起始时间.mat"
        )
    groups = match.groupdict()
    return {
        "modality": groups["modality"].lower(),
        "batch_id": groups["batch"],
        "label": groups["label"],
        "start_time": groups["time"].replace("：", ":"),
    }


def discover_samples(data_dir: str | Path) -> list[Sample]:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {root}. Put the competition train folder at data/train/."
        )
    records: dict[str, dict[str, Any]] = {}
    candidates = sorted([*root.glob("*.mat"), *root.glob("*.mp4")])
    for path in candidates:
        try:
            meta = parse_filename(path)
        except ValueError:
            continue
        batch_id = meta["batch_id"]
        record = records.setdefault(
            batch_id,
            {"label": None, "start_time": None, "radar": None, "ir": None},
        )
        modality = meta["modality"]
        if record[modality] is not None:
            raise ValueError(f"Duplicate {modality} file for batch {batch_id}")
        record[modality] = path.resolve()
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
        samples.append(
            Sample(
                batch_id=batch_id,
                label=str(record["label"]),
                radar_path=Path(record["radar"]),
                infrared_path=Path(record["ir"]),
                start_time=str(record["start_time"]),
            )
        )
    if incomplete:
        raise ValueError(f"Unpaired batches in {root}: {', '.join(incomplete)}")
    if not samples:
        raise ValueError(f"No paired competition files found directly under {root}")
    return samples


def class_counts(samples: list[Sample]) -> dict[str, int]:
    result: dict[str, int] = {}
    for sample in samples:
        result[sample.label] = result.get(sample.label, 0) + 1
    return dict(sorted(result.items()))
