from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Sample:
    batch_id: str
    radar_path: Path
    infrared_path: Path
    label: str | None = None
    start_time: str | None = None


def parse_competition_filename(path: str | Path) -> dict[str, str | None]:
    """Parse filenames such as ``ir_339_class-B_16：18.mp4``.

    The parser accepts both Chinese and ASCII colons. It intentionally returns
    the label only as metadata; inference code never uses that value.
    """

    p = Path(path)
    stem = p.stem.replace("：", ":")
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] not in {"ir", "radar"}:
        raise ValueError(f"Unsupported competition filename: {p.name}")

    modality = parts[0]
    batch_id = parts[1]
    label = next((x for x in parts[2:] if x.lower().startswith("class-")), None)
    start_time = parts[-1] if ":" in parts[-1] else None
    return {
        "modality": modality,
        "batch_id": batch_id,
        "label": label,
        "start_time": start_time,
    }


def discover_samples(
    root: str | Path,
    *,
    require_labels: bool,
    strict_pairs: bool = True,
) -> list[Sample]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")

    records: dict[str, dict[str, object]] = {}
    candidates: Iterable[Path] = list(root.rglob("*.mat")) + list(root.rglob("*.mp4"))

    for path in sorted(candidates):
        try:
            meta = parse_competition_filename(path)
        except ValueError:
            continue

        batch_id = str(meta["batch_id"])
        rec = records.setdefault(
            batch_id,
            {"batch_id": batch_id, "radar": None, "ir": None, "label": None, "start_time": None},
        )
        rec[str(meta["modality"])] = path
        if meta["label"] is not None:
            if rec["label"] not in {None, meta["label"]}:
                raise ValueError(f"Inconsistent labels for batch {batch_id}")
            rec["label"] = meta["label"]
        rec["start_time"] = rec["start_time"] or meta["start_time"]

    samples: list[Sample] = []
    incomplete: list[str] = []
    for batch_id, rec in sorted(records.items()):
        if rec["radar"] is None or rec["ir"] is None:
            incomplete.append(batch_id)
            continue
        if require_labels and rec["label"] is None:
            raise ValueError(f"Training sample {batch_id} has no label in filename")
        samples.append(
            Sample(
                batch_id=batch_id,
                radar_path=Path(rec["radar"]),
                infrared_path=Path(rec["ir"]),
                label=str(rec["label"]) if rec["label"] is not None else None,
                start_time=str(rec["start_time"]) if rec["start_time"] is not None else None,
            )
        )

    if strict_pairs and incomplete:
        raise ValueError(f"Unpaired batches found: {', '.join(incomplete)}")
    if not samples:
        raise ValueError(f"No paired radar/infrared samples found under {root}")
    return samples
