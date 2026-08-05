from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frames: int
    width: int
    height: int
    duration_seconds: float
    backend: str
    codec: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "fps": float(self.fps),
            "frames": float(self.frames),
            "width": float(self.width),
            "height": float(self.height),
            "duration_seconds": float(self.duration_seconds),
            "backend": self.backend,
            "codec": self.codec,
        }


def _ratio(value: object) -> float:
    text = str(value or "").strip()
    if not text or text in {"N/A", "0/0"}:
        return 0.0
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _integer(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> float:
    try:
        result = float(str(value))
        return result if math.isfinite(result) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _ffmpeg_enabled_in_opencv() -> bool:
    try:
        build = cv2.getBuildInformation()
    except Exception:
        return False
    return any("FFMPEG" in line and "YES" in line for line in build.splitlines())


def validate_video_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Infrared video file not found: {resolved}")
    size = resolved.stat().st_size
    if size <= 0:
        raise ValueError(f"Infrared video is empty: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise PermissionError(f"Infrared video is not readable: {resolved}")
    return resolved


def _ffprobe(path: Path) -> VideoInfo | None:
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    command = [
        executable,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
        stream = (payload.get("streams") or [])[0]
    except (json.JSONDecodeError, IndexError, TypeError):
        return None
    fps = _ratio(stream.get("avg_frame_rate")) or _ratio(stream.get("r_frame_rate"))
    duration = _float(stream.get("duration")) or _float((payload.get("format") or {}).get("duration"))
    frames = _integer(stream.get("nb_frames"))
    if frames <= 0 and fps > 0 and duration > 0:
        frames = int(round(fps * duration))
    if duration <= 0 and frames > 0 and fps > 0:
        duration = frames / fps
    return VideoInfo(
        path=path,
        fps=fps or 30.0,
        frames=frames,
        width=_integer(stream.get("width")),
        height=_integer(stream.get("height")),
        duration_seconds=duration,
        backend="ffmpeg_pipe",
        codec=str(stream.get("codec_name")) if stream.get("codec_name") else None,
    )


def _capture_metadata(cap: cv2.VideoCapture, path: Path, backend: str) -> VideoInfo:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        probed = _ffprobe(path)
        if probed is not None:
            fps = fps or probed.fps
            frames = frames or probed.frames
            width = width or probed.width
            height = height or probed.height
    fps = fps or 30.0
    duration = frames / fps if frames > 0 else 0.0
    return VideoInfo(
        path=path,
        fps=fps,
        frames=frames,
        width=width,
        height=height,
        duration_seconds=duration,
        backend=backend,
    )


def _open_capture_once(path: Path, api: int, backend: str) -> tuple[cv2.VideoCapture, VideoInfo] | None:
    cap = cv2.VideoCapture(str(path), api)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        cap.release()
        return None
    if not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
        cap.release()
        cap = cv2.VideoCapture(str(path), api)
        if not cap.isOpened():
            cap.release()
            return None
    return cap, _capture_metadata(cap, path, backend)


class VideoReader:
    """Open a competition video through OpenCV or an FFmpeg raw-frame pipe."""

    def __init__(self, path: str | Path) -> None:
        self.path = validate_video_path(path)
        self._capture: cv2.VideoCapture | None = None
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self.info: VideoInfo | None = None
        self._open()

    def _try_opencv(self, path: Path, suffix: str) -> bool:
        attempts: list[tuple[int, str]] = []
        if hasattr(cv2, "CAP_FFMPEG"):
            attempts.append((cv2.CAP_FFMPEG, f"opencv_ffmpeg{suffix}"))
        attempts.append((cv2.CAP_ANY, f"opencv_auto{suffix}"))
        for api, backend in attempts:
            opened = _open_capture_once(path, api, backend)
            if opened is not None:
                self._capture, info = opened
                self.info = VideoInfo(
                    path=self.path,
                    fps=info.fps,
                    frames=info.frames,
                    width=info.width,
                    height=info.height,
                    duration_seconds=info.duration_seconds,
                    backend=info.backend,
                    codec=info.codec,
                )
                return True
        return False

    def _open(self) -> None:
        if self._try_opencv(self.path, ""):
            return

        # Some OpenCV/FFmpeg builds mishandle non-ASCII paths. A temporary ASCII
        # hard link or symlink preserves the original data without copying it.
        if any(ord(character) > 127 for character in str(self.path)):
            self._temporary = tempfile.TemporaryDirectory(prefix="anti_air_video_")
            alias = Path(self._temporary.name) / f"input{self.path.suffix.lower()}"
            try:
                os.link(self.path, alias)
            except OSError:
                try:
                    alias.symlink_to(self.path)
                except OSError:
                    alias = self.path
            if alias != self.path and self._try_opencv(alias, "_ascii_alias"):
                return

        probed = _ffprobe(self.path)
        if probed is not None and shutil.which("ffmpeg") is not None:
            self.info = probed
            return

        message = self.diagnostic_message()
        self.close()
        raise RuntimeError(message)

    def diagnostic_message(self) -> str:
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        return (
            f"Cannot decode infrared video: {self.path}\n"
            f"file_size_bytes={self.path.stat().st_size if self.path.exists() else 0}\n"
            f"opencv_version={cv2.__version__}\n"
            f"opencv_ffmpeg_enabled={_ffmpeg_enabled_in_opencv()}\n"
            f"ffmpeg={ffmpeg or 'not found'}\n"
            f"ffprobe={ffprobe or 'not found'}\n"
            "Install system FFmpeg with: sudo apt-get update && sudo apt-get install -y ffmpeg\n"
            "Then run: python scripts/check_video.py --video '<video.mp4>'"
        )

    def metadata(self) -> dict[str, float | str | None]:
        if self.info is None:
            raise RuntimeError("Video reader is not initialized")
        return self.info.to_dict()

    def _iter_opencv(
        self,
        *,
        sample_fps: float,
        resize_width: int,
        max_samples: int,
    ) -> Iterator[tuple[float, np.ndarray]]:
        if self._capture is None or self.info is None:
            return
        cap = self._capture
        source_fps = self.info.fps or 30.0
        stride = max(1, int(round(source_fps / max(sample_fps, 1e-6))))
        frame_index = 0
        sampled = 0
        while sampled < max_samples:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            current = frame_index
            frame_index += 1
            if current % stride != 0:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if resize_width > 0 and gray.shape[1] > resize_width:
                scale = resize_width / gray.shape[1]
                gray = cv2.resize(
                    gray,
                    (resize_width, max(1, int(round(gray.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            yield current / source_fps, gray
            sampled += 1

    def _iter_ffmpeg(
        self,
        *,
        sample_fps: float,
        resize_width: int,
        max_samples: int,
    ) -> Iterator[tuple[float, np.ndarray]]:
        if self.info is None:
            return
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError(self.diagnostic_message())
        source_width = self.info.width
        source_height = self.info.height
        if source_width <= 0 or source_height <= 0:
            raise RuntimeError(f"FFprobe did not return valid video dimensions for {self.path}")
        output_width = source_width
        output_height = source_height
        if resize_width > 0 and source_width > resize_width:
            output_width = resize_width
            output_height = max(1, int(round(source_height * output_width / source_width)))
        actual_fps = min(max(sample_fps, 1e-6), self.info.fps or sample_fps)
        filter_graph = f"fps={actual_fps:.12g},scale={output_width}:{output_height},format=gray"
        command = [
            executable,
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(self.path),
            "-an",
            "-sn",
            "-dn",
            "-vf",
            filter_graph,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdout is None:
            raise RuntimeError("Failed to open FFmpeg stdout pipe")
        frame_bytes = output_width * output_height
        sampled = 0
        stderr = b""
        try:
            while sampled < max_samples:
                chunk = process.stdout.read(frame_bytes)
                if len(chunk) != frame_bytes:
                    break
                frame = np.frombuffer(chunk, dtype=np.uint8).reshape(output_height, output_width)
                yield sampled / actual_fps, frame
                sampled += 1
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            if process.stderr is not None:
                stderr = process.stderr.read()
                process.stderr.close()
            if sampled == 0 and process.returncode not in {0, None}:
                detail = stderr.decode("utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"FFmpeg could not decode {self.path}:\n{detail}")

    def iter_gray_frames(
        self,
        *,
        sample_fps: float,
        resize_width: int,
        max_samples: int,
    ) -> Iterator[tuple[float, np.ndarray]]:
        if self._capture is not None:
            yield from self._iter_opencv(
                sample_fps=sample_fps,
                resize_width=resize_width,
                max_samples=max_samples,
            )
        else:
            yield from self._iter_ffmpeg(
                sample_fps=sample_fps,
                resize_width=resize_width,
                max_samples=max_samples,
            )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        self.close()
