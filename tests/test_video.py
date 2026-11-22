import unittest
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch

import cv2
import numpy as np

from video import read_video


@contextmanager
def generated_video() -> Generator[Path, None, None]:
    with TemporaryDirectory() as directory:
        video_path = Path(directory) / "sample.mp4"
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48)
        )
        try:
            if not writer.isOpened():
                raise RuntimeError("Cannot create the MP4 test fixture.")
            for index in range(12):
                writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
        finally:
            writer.release()
        yield video_path


def make_capture(frames_expected: float | None = 1000.0) -> MagicMock:
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.get.side_effect = {
        cv2.CAP_PROP_FRAME_WIDTH: 64.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 48.0,
        cv2.CAP_PROP_FPS: 12.0,
        cv2.CAP_PROP_FRAME_COUNT: frames_expected,
    }.get
    capture.read.side_effect = [(True, object())] * 3 + [(False, None)]
    return capture


class TestVideoReader(unittest.TestCase):
    def test_reads_generated_mp4(self) -> None:
        with generated_video() as video_path:
            summary = read_video(video_path)

        self.assertEqual(
            summary,
            {
                "video": "sample.mp4",
                "width": 64,
                "height": 48,
                "fps": 12.0,
                "frames_read": 12,
                "frames_expected": 12,
                "duration_seconds": 1.0,
                "status": "completed",
            },
        )

    def test_missing_file_fails_before_opening_capture(self) -> None:
        with TemporaryDirectory() as directory:
            with (
                patch("video.cv2.VideoCapture") as capture,
                self.assertRaisesRegex(
                    FileNotFoundError, "Video file not found"
                ),
            ):
                read_video(Path(directory) / "missing.mp4")
            capture.assert_not_called()

    def test_directory_is_not_a_video(self) -> None:
        with (
            TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "Video input is not a file"),
        ):
            read_video(Path(directory))

    def test_invalid_video_fails(self) -> None:
        with TemporaryDirectory() as directory:
            video_path = Path(directory) / "invalid.mp4"
            video_path.write_bytes(b"This is not an MP4 video.")
            with self.assertRaisesRegex(ValueError, "Cannot open video"):
                read_video(video_path)

    def test_counts_decoded_frames_and_releases_capture(self) -> None:
        with generated_video() as video_path:
            capture = make_capture()
            with patch("video.cv2.VideoCapture", return_value=capture):
                summary = read_video(video_path)

        self.assertEqual(summary["frames_read"], 3)
        self.assertEqual(summary["duration_seconds"], 0.25)
        self.assertEqual(capture.read.call_count, 4)
        self.assertIn(call(cv2.CAP_PROP_FRAME_COUNT), capture.get.call_args_list)
        capture.release.assert_called_once_with()

    def test_premature_end_reports_incomplete_reading(self) -> None:
        with generated_video() as video_path:
            capture = make_capture()
            with patch("video.cv2.VideoCapture", return_value=capture):
                summary = read_video(video_path)

        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["frames_read"], 3)
        self.assertEqual(summary["frames_expected"], 1000)
        self.assertEqual(summary["duration_seconds"], 0.25)
        self.assertEqual(
            summary["warning"],
            "Video reading stopped before the expected frame count.",
        )
        self.assertEqual(capture.read.call_count, 4)
        capture.release.assert_called_once_with()

    def test_equal_or_underreported_frame_count_is_completed(self) -> None:
        with generated_video() as video_path:
            for frames_expected in (3.0, 2.0):
                with self.subTest(frames_expected=frames_expected):
                    capture = make_capture(frames_expected)
                    with patch("video.cv2.VideoCapture", return_value=capture):
                        summary = read_video(video_path)
                    self.assertEqual(summary["status"], "completed")
                    self.assertEqual(summary["frames_read"], 3)
                    self.assertEqual(
                        summary["frames_expected"], int(frames_expected)
                    )
                    self.assertNotIn("warning", summary)
                    self.assertEqual(capture.read.call_count, 4)
                    capture.release.assert_called_once_with()

    def test_unavailable_frame_count_is_completed(self) -> None:
        with generated_video() as video_path:
            capture = make_capture(None)
            with patch("video.cv2.VideoCapture", return_value=capture):
                summary = read_video(video_path)

        self.assertIsNone(summary["frames_expected"])
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["frames_read"], 3)
        self.assertNotIn("warning", summary)
        capture.release.assert_called_once_with()

    def test_invalid_frame_counts_are_unavailable(self) -> None:
        invalid_counts = (
            0.0, -1.0, float("nan"), float("inf"), float("-inf"), 3.5, 0.5,
        )
        with generated_video() as video_path:
            for frames_expected in invalid_counts:
                with self.subTest(frames_expected=frames_expected):
                    capture = make_capture(frames_expected)
                    with patch("video.cv2.VideoCapture", return_value=capture):
                        summary = read_video(video_path)
                    self.assertIsNone(summary["frames_expected"])
                    self.assertEqual(summary["status"], "completed")
                    self.assertEqual(summary["frames_read"], 3)
                    self.assertNotIn("warning", summary)
                    self.assertEqual(capture.read.call_count, 4)
                    capture.release.assert_called_once_with()

    def test_frame_count_exception_releases_capture(self) -> None:
        with generated_video() as video_path:
            capture = make_capture()
            capture.get.side_effect = [
                64.0, 48.0, 12.0, cv2.error("Frame count unavailable."),
            ]
            with (
                patch("video.cv2.VideoCapture", return_value=capture),
                self.assertRaises(cv2.error),
            ):
                read_video(video_path)
            capture.read.assert_not_called()
            capture.release.assert_called_once_with()

    def test_unopened_capture_is_released(self) -> None:
        with generated_video() as video_path:
            capture = make_capture()
            capture.isOpened.return_value = False
            with (
                patch("video.cv2.VideoCapture", return_value=capture),
                self.assertRaisesRegex(ValueError, "Cannot open video"),
            ):
                read_video(video_path)
            capture.release.assert_called_once_with()

    def test_invalid_fps_releases_capture_without_reading(self) -> None:
        with generated_video() as video_path:
            for fps in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
                with self.subTest(fps=fps):
                    capture = make_capture()
                    capture.get.side_effect = [64.0, 48.0, fps]
                    with (
                        patch("video.cv2.VideoCapture", return_value=capture),
                        self.assertRaisesRegex(ValueError, "Invalid video FPS"),
                    ):
                        read_video(video_path)
                    capture.read.assert_not_called()
                    capture.release.assert_called_once_with()

    def test_invalid_dimensions_release_capture(self) -> None:
        with generated_video() as video_path:
            dimensions = ((0.0, 48.0), (64.0, 0.0), (float("nan"), 48.0))
            for width, height in dimensions:
                with self.subTest(width=width, height=height):
                    capture = make_capture()
                    capture.get.side_effect = [width, height, 12.0]
                    with (
                        patch("video.cv2.VideoCapture", return_value=capture),
                        self.assertRaisesRegex(
                            ValueError, "Invalid video dimensions"
                        ),
                    ):
                        read_video(video_path)
                    capture.release.assert_called_once_with()

    def test_video_without_frames_releases_capture(self) -> None:
        with generated_video() as video_path:
            capture = make_capture()
            capture.read.side_effect = [(False, None)]
            with (
                patch("video.cv2.VideoCapture", return_value=capture),
                self.assertRaisesRegex(
                    ValueError, "Video contains no readable frames"
                ),
            ):
                read_video(video_path)
            capture.release.assert_called_once_with()

    def test_empty_decoded_frame_releases_capture(self) -> None:
        with generated_video() as video_path:
            capture = make_capture()
            capture.read.side_effect = [(True, None)]
            with (
                patch("video.cv2.VideoCapture", return_value=capture),
                self.assertRaisesRegex(
                    ValueError, "Video returned an empty frame"
                ),
            ):
                read_video(video_path)
            capture.release.assert_called_once_with()

    def test_decoder_exception_releases_capture(self) -> None:
        with generated_video() as video_path:
            capture = make_capture()
            capture.read.side_effect = cv2.error("Decode failed.")
            with (
                patch("video.cv2.VideoCapture", return_value=capture),
                self.assertRaises(cv2.error),
            ):
                read_video(video_path)
            capture.release.assert_called_once_with()
