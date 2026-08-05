from __future__ import annotations

import argparse
import json
from pathlib import Path

from anti_air.video_io import VideoReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose infrared video path and decoder availability")
    parser.add_argument("--video", required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()

    video = Path(args.video).expanduser().resolve()
    with VideoReader(video) as reader:
        metadata = reader.metadata()
        decoded = []
        for timestamp, frame in reader.iter_gray_frames(
            sample_fps=args.sample_fps,
            resize_width=320,
            max_samples=max(1, args.frames),
        ):
            decoded.append(
                {
                    "time_seconds": float(timestamp),
                    "shape": [int(frame.shape[0]), int(frame.shape[1])],
                    "mean": float(frame.mean()),
                }
            )
        if not decoded:
            raise RuntimeError(f"Decoder opened but returned no frames: {video}")

    result = {
        "status": "ok",
        "video": str(video),
        "metadata": metadata,
        "decoded_frames": decoded,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
