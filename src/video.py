from collections.abc import Callable, Generator
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import cv2
from cv2.typing import MatLike


class VideoSummary(TypedDict):
    video: str
    width: int
    height: int
    fps: float
    frames_read: int
    frames_expected: int | None
    duration_seconds: float
    status: Literal["completed", "incomplete"]
    warning: NotRequired[str]


@contextmanager
def _open_capture(video_path: Path) -> Generator[cv2.VideoCapture, None, None]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        yield capture
    finally:
        capture.release()


def _read_metadata(
    capture: cv2.VideoCapture, video_path: Path
) -> tuple[int, int, float, int | None]:
    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not all(isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError(f"Invalid video dimensions: {video_path}")
    if not isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video FPS ({fps}): {video_path}")

    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frames_expected = (
        int(frame_count)
        if frame_count is not None
        and isfinite(frame_count)
        and frame_count > 0
        and frame_count.is_integer()
        else None
    )
    return int(width), int(height), fps, frames_expected


def read_video(
    video_path: Path,
    frame_handler: Callable[[MatLike, int, float, float], None] | None = None,
) -> VideoSummary:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Video input is not a file: {video_path}")

    with _open_capture(video_path) as capture:
        width, height, fps, frames_expected = _read_metadata(capture, video_path)

        frames_read = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None:
                raise ValueError(f"Video returned an empty frame: {video_path}")
            if frame_handler is not None:
                frame_handler(frame, frames_read, frames_read / fps, fps)
            frames_read += 1

        if frames_read == 0:
            raise ValueError(f"Video contains no readable frames: {video_path}")

    summary: VideoSummary = {
        "video": video_path.name,
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "frames_read": frames_read,
        "frames_expected": frames_expected,
        "duration_seconds": frames_read / fps,
        "status": "completed",
    }
    if frames_expected is not None and frames_read < frames_expected:
        summary["status"] = "incomplete"
        summary["warning"] = "Video reading stopped before the expected frame count."
    return summary