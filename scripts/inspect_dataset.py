from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from anti_air.dataset import discover_samples
from anti_air.radar import load_radar_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect paired radar/infrared competition data")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="outputs/dataset_inventory.json")
    args = parser.parse_args()

    samples = discover_samples(args.data_root, require_labels=False, strict_pairs=False)
    inventory = []
    for sample in samples:
        radar = load_radar_frame(sample.radar_path)
        cap = cv2.VideoCapture(str(sample.infrared_path))
        video = {
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
        cap.release()
        inventory.append(
            {
                "batch_id": sample.batch_id,
                "label": sample.label,
                "radar_path": str(sample.radar_path),
                "infrared_path": str(sample.infrared_path),
                "radar_shape": list(radar.shape),
                "radar_columns": [str(x) for x in radar.columns],
                "video": video,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote inventory for {len(inventory)} paired samples to {output}")


if __name__ == "__main__":
    main()
