from collections.abc import Generator
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from typing import Literal, TypedDict

import cv2


class VideoSummary(TypedDict):
    video: str
    width: int
    height: int
    fps: float
    frames_read: int
    duration_seconds: float
    status: Literal["completed"]


@contextmanager
def _open_capture(video_path: Path) -> Generator[cv2.VideoCapture, None, None]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        yield capture
    finally:
        capture.release()


def read_video(video_path: Path) -> VideoSummary:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Video input is not a file: {video_path}")

    with _open_capture(video_path) as capture:
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not all(isfinite(value) and value > 0 for value in (width, height)):
            raise ValueError(f"Invalid video dimensions: {video_path}")
        if not isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video FPS ({fps}): {video_path}")

        frames_read = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None:
                raise ValueError(f"Video returned an empty frame: {video_path}")
            frames_read += 1

        if frames_read == 0:
            raise ValueError(f"Video contains no readable frames: {video_path}")

    return {
        "video": video_path.name,
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "frames_read": frames_read,
        "duration_seconds": frames_read / fps,
        "status": "completed",
    }