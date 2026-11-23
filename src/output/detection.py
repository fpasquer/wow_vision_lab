import json
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Self, TextIO

import cv2
from cv2.typing import MatLike

from detector import Detection
from video import VideoSummary

from .annotation import _annotate_frame
from .summary import write_summary


class DetectionOutput:
    def __init__(
        self, summary_path: Path, model_path: Path, input_path: Path
    ) -> None:
        self.summary_path = summary_path
        self.detection_path = summary_path.with_name(
            f"{summary_path.stem}-detections.json"
        )
        self.video_path = summary_path.with_name(f"{summary_path.stem}-annotated.mp4")
        paths = (summary_path, self.detection_path, self.video_path)
        resolved = [path.resolve() for path in paths]
        output_root = Path("data/output").resolve()
        protected = {input_path.resolve(), model_path.resolve()}
        if any(not path.is_relative_to(output_root) for path in resolved):
            raise ValueError("Detection outputs must be under data/output.")
        if len(set(resolved)) != len(paths) or protected.intersection(resolved):
            raise ValueError("Output paths must not overwrite the input or model.")
        if any(path.exists() and not path.is_file() for path in paths):
            raise ValueError("Detection output paths must be files, not directories.")
        self._summary_target = resolved[0]
        self.model_name = model_path.name
        self._directory: TemporaryDirectory[str] | None = None
        self._json: TextIO | None = None
        self._video: cv2.VideoWriter | None = None
        self._has_record = False

    def __enter__(self) -> Self:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        for path in (self._summary_target, self.detection_path, self.video_path):
            path.unlink(missing_ok=True)
        self._directory = TemporaryDirectory(
            prefix=".detection-", dir=self.summary_path.parent
        )
        try:
            self._json = (Path(self._directory.name) / "detections.json").open(
                "w", encoding="utf-8"
            )
            self._json.write(
                '{"model": ' + json.dumps(self.model_name) + ', "frames": [\n'
            )
        except OSError:
            try:
                self._close()
            finally:
                self._directory.cleanup()
            raise
        return self

    def write_frame(
        self, frame: MatLike, frame_index: int, timestamp_seconds: float,
        fps: float, detections: Sequence[Detection] | None,
    ) -> None:
        if self._directory is None or self._json is None:
            raise RuntimeError("Detection outputs are not open.")
        if self._video is None:
            self._video = cv2.VideoWriter(
                str(Path(self._directory.name) / "annotated.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), fps,
                (frame.shape[1], frame.shape[0]),
            )
            if not self._video.isOpened():
                raise ValueError(f"Cannot create annotated video: {self.video_path}")
        if detections is not None:
            content = json.dumps(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "detections": detections,
                },
                allow_nan=False,
            )
            self._json.write((",\n" if self._has_record else "") + content)
            self._has_record = True
            _annotate_frame(frame, detections)
        self._video.write(frame)

    def finalize(self, summary: VideoSummary) -> None:
        if self._directory is None or self._json is None or self._video is None:
            raise RuntimeError("Detection outputs contain no processed frames.")
        completed = summary["status"] == "completed"
        if completed:
            self._json.write("\n]}\n")
        self._close()
        directory = Path(self._directory.name)
        staged_summary = directory / "summary.json"
        write_summary(summary, staged_summary)
        if completed:
            (directory / "annotated.mp4").replace(self.video_path)
            (directory / "detections.json").replace(self.detection_path)
        staged_summary.replace(self._summary_target)

    def _close(self) -> None:
        json_file, self._json = self._json, None
        video, self._video = self._video, None
        try:
            if json_file is not None:
                json_file.close()
        finally:
            if video is not None:
                video.release()

    def __exit__(
        self, exception_type: type[BaseException] | None,
        exception: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        try:
            self._close()
        finally:
            if self._directory is not None:
                self._directory.cleanup()