from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from anti_air.video_io import VideoReader, validate_video_path


def test_missing_video_path_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Infrared video file not found"):
        validate_video_path(tmp_path / "missing.mp4")


def test_video_reader_handles_unicode_filename(tmp_path: Path) -> None:
    ascii_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(ascii_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV build cannot create the test MJPEG video")
    for index in range(6):
        frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    unicode_path = tmp_path / "ir_339_class-B_16：18.avi"
    ascii_path.rename(unicode_path)

    with VideoReader(unicode_path) as reader:
        metadata = reader.metadata()
        frames = list(
            reader.iter_gray_frames(
                sample_fps=2.0,
                resize_width=32,
                max_samples=2,
            )
        )

    assert metadata["width"] == 64.0
    assert metadata["height"] == 48.0
    assert len(frames) == 2
    assert frames[0][1].shape[1] == 32
