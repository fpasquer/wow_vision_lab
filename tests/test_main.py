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
from test_video import generated_video

from main import main


class TestMain(unittest.TestCase):
    def test_writes_summary_to_configured_path(self) -> None:
        with generated_video() as video_path:
            output_path = video_path.parent / "results" / "summary.json"
            with redirect_stdout(StringIO()):
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
                "frames_read": 12,
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
            self.assertEqual(summary["status"], "completed")

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
