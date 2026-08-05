from pathlib import Path

import cv2
import numpy as np
import pytest

from anti_air.video_io import VideoReader


def test_unicode_video_path(tmp_path: Path) -> None:
    path = tmp_path / "ir_1_class-A_10：00.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot create MJPEG test video")
    for index in range(6):
        writer.write(np.full((48, 64, 3), index * 20, np.uint8))
    writer.release()
    with VideoReader(path) as reader:
        frames = list(reader.iter_gray_frames(sample_fps=2.0, resize_width=32, max_samples=2))
    assert len(frames) == 2
    assert frames[0][1].shape[1] == 32
