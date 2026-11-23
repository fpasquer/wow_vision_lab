import json
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from test_video import generated_video, make_capture

from main import main
from video import read_video


class TestMain(unittest.TestCase):
    def test_writes_summary_to_configured_path(self) -> None:
        with generated_video() as video_path:
            output_path = video_path.parent / "results" / "summary.json"
            errors = StringIO()
            with redirect_stderr(errors), redirect_stdout(StringIO()):
                exit_code = main(
                    ["--input", str(video_path), "--output", str(output_path)]
                )
            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(errors.getvalue(), "")
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

    def test_cli_defaults_to_data_output(self) -> None:
        with generated_video() as video_path:
            result = subprocess.run(
                [sys.executable, "-m", "main", "--input", str(video_path)],
                cwd=video_path.parent,
                env={
                    **os.environ,
                    "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            output_path = video_path.parent / "data" / "output" / "summary.json"
            summary = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["frames_read"], 12)
            self.assertEqual(summary["frames_expected"], 12)
            self.assertEqual(summary["status"], "completed")
            self.assertNotIn("warning", summary)
            self.assertNotIn("Warning:", result.stderr)

    def test_incomplete_reading_writes_summary_and_warns(self) -> None:
        with generated_video() as video_path:
            output_path = video_path.parent / "results" / "summary.json"
            capture = make_capture()
            errors = StringIO()
            with (
                patch("video.cv2.VideoCapture", return_value=capture),
                redirect_stderr(errors),
                redirect_stdout(StringIO()),
            ):
                exit_code = main(
                    ["--input", str(video_path), "--output", str(output_path)]
                )
            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            summary,
            {
                "video": "sample.mp4",
                "width": 64,
                "height": 48,
                "fps": 12.0,
                "frames_read": 3,
                "frames_expected": 1000,
                "duration_seconds": 0.25,
                "status": "incomplete",
                "warning": "Video reading stopped before the expected frame count.",
            },
        )
        self.assertEqual(
            errors.getvalue(),
            "Warning: Video reading stopped before the expected frame count.\n",
        )
        capture.release.assert_called_once_with()

    def test_unavailable_frame_counts_write_null_without_warning(self) -> None:
        frame_counts = (
            None, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), 3.5,
        )
        with generated_video() as video_path:
            for frames_expected in frame_counts:
                with self.subTest(frames_expected=frames_expected):
                    output_path = video_path.parent / "summary.json"
                    capture = make_capture(frames_expected)
                    errors = StringIO()
                    with (
                        patch("video.cv2.VideoCapture", return_value=capture),
                        redirect_stderr(errors),
                        redirect_stdout(StringIO()),
                    ):
                        exit_code = main(
                            [
                                "--input", str(video_path),
                                "--output", str(output_path),
                            ]
                        )
                    summary = json.loads(output_path.read_text(encoding="utf-8"))
                    self.assertEqual(exit_code, 0)
                    self.assertIsNone(summary["frames_expected"])
                    self.assertEqual(summary["status"], "completed")
                    self.assertNotIn("warning", summary)
                    self.assertEqual(errors.getvalue(), "")
                    capture.release.assert_called_once_with()

    def test_missing_and_invalid_inputs_report_errors(self) -> None:
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            invalid_path = directory_path / "invalid.mp4"
            invalid_path.write_bytes(b"This is not an MP4 video.")
            for input_path in (directory_path / "missing.mp4", invalid_path):
                with self.subTest(input=input_path.name):
                    output_path = directory_path / "summary.json"
                    errors = StringIO()
                    with redirect_stderr(errors), redirect_stdout(StringIO()):
                        exit_code = main(
                            [
                                "--input", str(input_path),
                                "--output", str(output_path),
                            ]
                        )
                    self.assertEqual(exit_code, 1)
                    self.assertIn("Error:", errors.getvalue())
                    self.assertIn(str(input_path), errors.getvalue())
                    self.assertFalse(output_path.exists())

    def test_reader_errors_are_reported_without_writing_output(self) -> None:
        reader_errors = (
            ValueError("Invalid video FPS (0.0)."),
            cv2.error("Decode failed."),
        )
        for error in reader_errors:
            with self.subTest(error=type(error).__name__):
                errors = StringIO()
                with (
                    patch("main.read_video", side_effect=error),
                    patch("main.write_summary") as write_summary,
                    redirect_stderr(errors),
                    redirect_stdout(StringIO()),
                ):
                    exit_code = main(["--input", "sample.mp4"])
                self.assertEqual(exit_code, 1)
                self.assertIn(str(error), errors.getvalue())
                write_summary.assert_not_called()

    def test_output_failure_is_reported(self) -> None:
        with generated_video() as video_path:
            blocked_path = video_path.parent / "blocked"
            blocked_path.write_text("Not a directory.", encoding="utf-8")
            errors = StringIO()
            with redirect_stderr(errors), redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "--input", str(video_path),
                        "--output", str(blocked_path / "summary.json"),
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertIn("Error:", errors.getvalue())
            self.assertIn(str(blocked_path), errors.getvalue())

    def test_output_failure_for_incomplete_reading_is_reported(self) -> None:
        with generated_video() as video_path:
            blocked_path = video_path.parent / "blocked"
            blocked_path.write_text("Not a directory.", encoding="utf-8")
            capture = make_capture()
            errors = StringIO()
            output = StringIO()
            with (
                patch("video.cv2.VideoCapture", return_value=capture),
                redirect_stderr(errors),
                redirect_stdout(output),
            ):
                exit_code = main(
                    [
                        "--input", str(video_path),
                        "--output", str(blocked_path / "summary.json"),
                    ]
                )
            self.assertEqual(exit_code, 1)
            self.assertIn("Error:", errors.getvalue())
            self.assertIn(str(blocked_path), errors.getvalue())
            self.assertEqual(output.getvalue(), "")
            capture.release.assert_called_once_with()

    def test_output_cannot_overwrite_input(self) -> None:
        with generated_video() as video_path:
            original_bytes = video_path.read_bytes()
            errors = StringIO()
            with redirect_stderr(errors), redirect_stdout(StringIO()):
                exit_code = main(
                    ["--input", str(video_path), "--output", str(video_path)]
                )
            self.assertEqual(exit_code, 1)
            self.assertIn("must not overwrite", errors.getvalue())
            self.assertEqual(video_path.read_bytes(), original_bytes)

    def test_input_argument_is_required(self) -> None:
        errors = StringIO()
        with redirect_stderr(errors), self.assertRaises(SystemExit) as error:
            main([])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--input", errors.getvalue())


class TestDetectionMain(unittest.TestCase):
    def setUp(self) -> None:
        Path("data/output").mkdir(parents=True, exist_ok=True)
        self.directory = TemporaryDirectory(prefix="t05-main-", dir="data/output")
        self.addCleanup(self.directory.cleanup)
        self.output_path = Path(self.directory.name) / "summary.json"
        self.detection_path = self.output_path.with_name("summary-detections.json")
        self.video_path = self.output_path.with_name("summary-annotated.mp4")
        patcher = patch("main.Detector")
        self.detector_type = patcher.start()
        self.addCleanup(patcher.stop)
        self.detections = [
            {
                "class_id": 0, "class_name": "person", "confidence": 0.8,
                "bbox": [8.0, 8.0, 40.0, 32.0],
            }
        ]
        self.detector_type.return_value.detect.return_value = self.detections

    def _run(self, input_path: Path, *options: str) -> tuple[int, str]:
        errors = StringIO()
        with redirect_stderr(errors), redirect_stdout(StringIO()):
            exit_code = main(
                [
                    "--input", str(input_path), "--output", str(self.output_path),
                    "--model", *options,
                ]
            )
        return exit_code, errors.getvalue()

    def _assert_no_detection_outputs(self) -> None:
        self.assertFalse(self.detection_path.exists())
        self.assertFalse(self.video_path.exists())
        self.assertEqual(list(Path(self.directory.name).glob(".detection-*")), [])

    def test_default_sampling_metadata_and_unchanged_summary(self) -> None:
        with generated_video() as input_path:
            expected_summary = read_video(input_path)
            exit_code, errors = self._run(input_path)
        self.assertEqual(exit_code, 0, errors)
        self.assertEqual(errors, "")
        self.assertEqual(
            json.loads(self.output_path.read_text(encoding="utf-8")), expected_summary
        )
        self.detector_type.assert_called_once_with(Path("data/models/yolo11n.pt"), 0.25)
        metadata = json.loads(self.detection_path.read_text(encoding="utf-8"))
        self.assertEqual(
            metadata,
            {
                "model": "yolo11n.pt",
                "frames": [
                    {
                        "frame_index": index, "timestamp_seconds": index / 12.0,
                        "detections": self.detections,
                    }
                    for index in (0, 5, 10)
                ],
            },
        )
        self.assertEqual(self.detector_type.return_value.detect.call_count, 3)
        self.assertEqual(list(Path(self.directory.name).glob(".detection-*")), [])

    def test_annotated_video_keeps_all_frames_and_only_marks_sampled_frames(self) -> None:
        with generated_video() as input_path:
            exit_code, errors = self._run(input_path)
        self.assertEqual(exit_code, 0, errors)
        capture = cv2.VideoCapture(str(self.video_path))
        try:
            self.assertTrue(capture.isOpened())
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 12.0)
            index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                self.assertEqual(frame.shape, (48, 64, 3))
                blue, green, red = (int(value) for value in frame[8, 8])
                if index % 5 == 0:
                    self.assertGreater(green, max(blue, red) + 50)
                else:
                    self.assertLessEqual(abs(green - red), 5)
                index += 1
            self.assertEqual(index, 12)
        finally:
            capture.release()

    def test_custom_model_confidence_and_frame_steps(self) -> None:
        model_path = Path(self.directory.name) / "custom.pt"
        with generated_video() as input_path:
            for step in (1, 2, 20):
                with self.subTest(step=step):
                    self.detector_type.reset_mock()
                    exit_code, errors = self._run(
                        input_path, str(model_path), "--confidence", "0.7",
                        "--frame-step", str(step),
                    )
                    self.assertEqual(exit_code, 0, errors)
                    self.detector_type.assert_called_once_with(model_path, 0.7)
                    metadata = json.loads(
                        self.detection_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(metadata["model"], "custom.pt")
                    self.assertEqual(
                        [frame["frame_index"] for frame in metadata["frames"]],
                        list(range(0, 12, step)),
                    )

    def test_empty_detections_still_write_sampled_frame_records(self) -> None:
        self.detector_type.return_value.detect.return_value = []
        with generated_video() as input_path:
            exit_code, errors = self._run(input_path)
        self.assertEqual(exit_code, 0, errors)
        metadata = json.loads(self.detection_path.read_text(encoding="utf-8"))
        self.assertEqual(len(metadata["frames"]), 3)
        self.assertTrue(all(frame["detections"] == [] for frame in metadata["frames"]))

    def test_incomplete_read_keeps_summary_warning_and_discards_detection_outputs(self) -> None:
        capture = make_capture()
        capture.read.side_effect = [
            (True, np.zeros((48, 64, 3), dtype=np.uint8)) for _ in range(3)
        ] + [(False, None)]
        with (
            generated_video() as input_path,
            patch("video.cv2.VideoCapture", return_value=capture),
        ):
            exit_code, errors = self._run(input_path)
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            errors, "Warning: Video reading stopped before the expected frame count.\n"
        )
        summary = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["frames_read"], 3)
        self.assertEqual(summary["frames_expected"], 1000)
        self.assertEqual(summary["duration_seconds"], 0.25)
        self._assert_no_detection_outputs()
        capture.release.assert_called_once_with()

    def test_unknown_frame_count_keeps_existing_completed_behavior(self) -> None:
        capture = make_capture(None)
        capture.read.side_effect = [
            (True, np.zeros((48, 64, 3), dtype=np.uint8)) for _ in range(3)
        ] + [(False, None)]
        with (
            generated_video() as input_path,
            patch("video.cv2.VideoCapture", return_value=capture),
        ):
            exit_code, errors = self._run(input_path)
        self.assertEqual(exit_code, 0, errors)
        self.assertEqual(errors, "")
        summary = json.loads(self.output_path.read_text(encoding="utf-8"))
        self.assertIsNone(summary["frames_expected"])
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(self.detection_path.is_file())
        self.assertTrue(self.video_path.is_file())

    def test_inference_failure_cleans_up_without_summary(self) -> None:
        self.detector_type.return_value.detect.side_effect = [
            [], RuntimeError("Inference failed.")
        ]
        with generated_video() as input_path:
            exit_code, errors = self._run(input_path)
        self.assertEqual(exit_code, 1)
        self.assertIn("Inference failed.", errors)
        self.assertFalse(self.output_path.exists())
        self._assert_no_detection_outputs()

    def test_summary_failure_does_not_publish_detection_outputs(self) -> None:
        with (
            generated_video() as input_path,
            patch("main.write_summary", side_effect=OSError("Summary write failed.")),
        ):
            exit_code, errors = self._run(input_path)
        self.assertEqual(exit_code, 1)
        self.assertIn("Summary write failed.", errors)
        self._assert_no_detection_outputs()

    def test_video_writer_failure_is_clear_and_releases_writer(self) -> None:
        with (
            generated_video() as input_path,
            patch("output.cv2.VideoWriter") as create_writer,
        ):
            create_writer.return_value.isOpened.return_value = False
            exit_code, errors = self._run(input_path)
        self.assertEqual(exit_code, 1)
        self.assertIn("Cannot create annotated video", errors)
        create_writer.return_value.release.assert_called()
        self._assert_no_detection_outputs()

    def test_json_publish_failure_restores_previous_video(self) -> None:
        self.video_path.write_bytes(b"Previous video.")
        self.detection_path.write_text("Previous detections.", encoding="utf-8")
        replace = Path.replace

        def fail_detection_publish(source: Path, target: Path) -> Path:
            if source.name == "detections.json":
                raise OSError("Detection publish failed.")
            return replace(source, target)

        with (
            generated_video() as input_path,
            patch("output.Path.replace", autospec=True, side_effect=fail_detection_publish),
        ):
            exit_code, errors = self._run(input_path)
        self.assertEqual(exit_code, 1)
        self.assertIn("Detection publish failed.", errors)
        self.assertEqual(self.video_path.read_bytes(), b"Previous video.")
        self.assertEqual(
            self.detection_path.read_text(encoding="utf-8"), "Previous detections."
        )
        self.assertEqual(list(Path(self.directory.name).glob(".detection-*")), [])

    def test_missing_and_invalid_model_errors_are_reported(self) -> None:
        for error in (
            FileNotFoundError("Model file not found: missing.pt"),
            ValueError("Cannot load model invalid.pt: Invalid weights."),
        ):
            with self.subTest(error=type(error).__name__):
                self.detector_type.side_effect = error
                with generated_video() as input_path:
                    exit_code, errors = self._run(input_path)
                self.assertEqual(exit_code, 1)
                self.assertIn(str(error), errors)
                self.assertFalse(self.output_path.exists())
                self._assert_no_detection_outputs()

    def test_non_finite_json_values_fail_without_publishing(self) -> None:
        self.detections[0]["confidence"] = float("nan")
        with generated_video() as input_path:
            exit_code, errors = self._run(input_path)
        self.assertEqual(exit_code, 1)
        self.assertIn("Out of range float values", errors)
        self._assert_no_detection_outputs()

    def test_invalid_detection_options_are_cli_errors(self) -> None:
        options = (
            ("--confidence", "nan"), ("--confidence", "inf"),
            ("--confidence", "-0.1"), ("--confidence", "1.1"),
            ("--confidence", "invalid"), ("--frame-step", "0"),
            ("--frame-step", "-1"), ("--frame-step", "1.5"),
        )
        for option, value in options:
            with (
                self.subTest(option=option, value=value),
                self.assertRaises(SystemExit) as raised,
            ):
                self._run(Path("sample.mp4"), option, value)
            self.assertEqual(raised.exception.code, 2)
        self.detector_type.assert_not_called()
        self._assert_no_detection_outputs()

    def test_outputs_outside_data_output_and_symlink_escape_are_rejected(self) -> None:
        with generated_video() as input_path:
            link = Path(self.directory.name) / "escape"
            link.symlink_to(input_path.parent, target_is_directory=True)
            for output_path in (
                input_path.parent / "outside.json", link / "escaped.json",
            ):
                with self.subTest(output=output_path):
                    exit_code, errors = self._run(
                        input_path, "--output", str(output_path)
                    )
                    self.assertEqual(exit_code, 1)
                    self.assertIn("must be under data/output", errors)
                    self.assertFalse(output_path.exists())
        self.detector_type.assert_not_called()

    def test_sidecar_collision_with_input_is_rejected(self) -> None:
        self.video_path.write_bytes(b"Input video sentinel.")
        exit_code, errors = self._run(self.video_path)
        self.assertEqual(exit_code, 1)
        self.assertIn("must not overwrite the input or model", errors)
        self.assertEqual(self.video_path.read_bytes(), b"Input video sentinel.")
        self.detector_type.assert_not_called()

    def test_output_cannot_overwrite_model_or_alias_another_output(self) -> None:
        model_path = self.output_path.with_suffix(".pt")
        model_path.write_bytes(b"Model sentinel.")
        exit_code, errors = self._run(
            Path("input.mp4"), str(model_path), "--output", str(model_path)
        )
        self.assertEqual(exit_code, 1)
        self.assertIn("must not overwrite the input or model", errors)
        self.assertEqual(model_path.read_bytes(), b"Model sentinel.")
        self.detection_path.symlink_to(self.output_path.resolve())
        exit_code, errors = self._run(Path("input.mp4"))
        self.assertEqual(exit_code, 1)
        self.assertIn("must not overwrite the input or model", errors)
        self.detector_type.assert_not_called()
