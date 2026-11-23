import hashlib
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import torch
from test_video import generated_video

from detector import Detector
from main import main


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION_TESTS") == "1",
    "Real weights require the separate RUN_INTEGRATION_TESTS=1 command.",
)
class TestDetectorIntegration(unittest.TestCase):
    def test_real_weights_run_on_cpu_without_changing_weights(self) -> None:
        model_path = Path("data/models/yolo11n.pt")
        before = hashlib.sha256(model_path.read_bytes()).digest()
        detector = Detector(model_path)
        detections = detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))
        self.assertEqual(next(detector.model.model.parameters()).device.type, "cpu")
        self.assertIsNone(torch.version.cuda)
        self.assertEqual(len(detector.model.names), 80)
        self.assertIsInstance(detections, list)
        for detection in detections:
            self.assertIn(detection["class_id"], detector.model.names)
            self.assertGreaterEqual(detection["confidence"], 0.25)
            self.assertEqual(len(detection["bbox"]), 4)
        self.assertEqual(hashlib.sha256(model_path.read_bytes()).digest(), before)

    def test_real_video_outputs_and_sampling(self) -> None:
        Path("data/output").mkdir(parents=True, exist_ok=True)
        errors = StringIO()
        with (
            generated_video() as video_path,
            TemporaryDirectory(prefix="t05-integration-", dir="data/output") as directory,
            redirect_stderr(errors),
            redirect_stdout(StringIO()),
        ):
            output_path = Path(directory) / "summary.json"
            exit_code = main(
                ["--input", str(video_path), "--output", str(output_path), "--model"]
            )
            self.assertEqual(exit_code, 0, errors.getvalue())
            self.assertEqual(errors.getvalue(), "")
            summary = json.loads(output_path.read_text(encoding="utf-8"))
            metadata = json.loads(
                output_path.with_name("summary-detections.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["frames_read"], 12)
            self.assertEqual(summary["status"], "completed")
            self.assertNotIn("warning", summary)
            self.assertEqual(metadata["model"], "yolo11n.pt")
            self.assertEqual(
                [frame["frame_index"] for frame in metadata["frames"]], [0, 5, 10]
            )
            for frame in metadata["frames"]:
                self.assertEqual(
                    frame["timestamp_seconds"], frame["frame_index"] / 12.0
                )
                self.assertIsInstance(frame["detections"], list)
            self._assert_video(output_path.with_name("summary-annotated.mp4"))
            self.assertEqual(list(Path(directory).glob(".detection-*")), [])

    def _assert_video(self, video_path: Path) -> None:
        capture = cv2.VideoCapture(str(video_path))
        try:
            self.assertTrue(capture.isOpened())
            self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 12.0)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_WIDTH), 64.0)
            self.assertEqual(capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 48.0)
            frames_read = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                self.assertEqual(frame.shape, (48, 64, 3))
                frames_read += 1
            self.assertEqual(frames_read, 12)
        finally:
            capture.release()
