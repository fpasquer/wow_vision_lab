import argparse
import sys
from collections.abc import Sequence
from math import isfinite
from pathlib import Path

import cv2
from cv2.typing import MatLike

from detector import DEFAULT_CONFIDENCE, Detector
from output import DetectionOutput, write_summary
from video import VideoSummary, read_video

DEFAULT_FRAME_STEP = 5


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read an MP4 video, optionally detect objects, and write outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the MP4 video, usually under data/input/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/summary.json"),
        help="JSON output path (default: data/output/summary.json).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        nargs="?",
        const=Path("data/models/yolo11n.pt"),
        help="Enable detection with local weights (default: data/models/yolo11n.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Detection confidence from 0 to 1 (default: %(default)s).",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=DEFAULT_FRAME_STEP,
        help="Detect one frame every N decoded frames (default: %(default)s).",
    )
    arguments = parser.parse_args(argv)
    if not isfinite(arguments.confidence) or not 0 <= arguments.confidence <= 1:
        parser.error("--confidence must be a finite number from 0 to 1.")
    if arguments.frame_step <= 0:
        parser.error("--frame-step must be a positive integer.")
    return arguments


def _detect_video(arguments: argparse.Namespace) -> VideoSummary:
    outputs = DetectionOutput(arguments.output, arguments.model, arguments.input)
    detector = Detector(arguments.model, arguments.confidence)

    def process_frame(
        frame: MatLike, frame_index: int, timestamp_seconds: float, fps: float
    ) -> None:
        detections = (
            detector.detect(frame)
            if frame_index % arguments.frame_step == 0
            else None
        )
        outputs.write_frame(frame, frame_index, timestamp_seconds, fps, detections)

    with outputs:
        summary = read_video(arguments.input, process_frame)
        write_summary(summary, arguments.output)
        if summary["status"] == "completed":
            outputs.finalize()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        if arguments.output.resolve() == arguments.input.resolve():
            raise ValueError("Output path must not overwrite the input video.")
        if arguments.model is None:
            summary = read_video(arguments.input)
            write_summary(summary, arguments.output)
        else:
            summary = _detect_video(arguments)
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    warning = summary.get("warning")
    if warning is not None:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Video summary written to: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())