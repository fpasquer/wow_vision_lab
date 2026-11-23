import pickle
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from detector import CACHE_PATH, Detector


class TestDetector(unittest.TestCase):
    def setUp(self) -> None:
        CACHE_PATH.mkdir(parents=True, exist_ok=True)
        self.directory = TemporaryDirectory(prefix="t05-detector-", dir="data/output")
        self.addCleanup(self.directory.cleanup)
        self.model_path = Path(self.directory.name) / "custom.pt"
        self.model_path.write_bytes(b"Mock model weights.")
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        self.model = MagicMock()
        self.model.task = "detect"
        self.model.predict.return_value = [
            SimpleNamespace(
                names={0: "person", 16: "dog"},
                boxes=SimpleNamespace(
                    cls=np.array([0.0, 16.0]),
                    conf=np.array([0.8, 0.6]),
                    xyxy=np.array([[1, 2, 30, 40], [10, 12, 50, 45]]),
                ),
            )
        ]

    def test_loads_local_model_once_and_converts_results(self) -> None:
        with patch("ultralytics.YOLO", return_value=self.model) as load:
            detector = Detector(self.model_path)
            detections = detector.detect(self.frame)
            detector.detect(self.frame)
        load.assert_called_once_with(str(self.model_path.resolve()), task="detect")
        self.assertEqual(
            detections,
            [
                {
                    "class_id": 0, "class_name": "person", "confidence": 0.8,
                    "bbox": [1.0, 2.0, 30.0, 40.0],
                },
                {
                    "class_id": 16, "class_name": "dog", "confidence": 0.6,
                    "bbox": [10.0, 12.0, 50.0, 45.0],
                },
            ],
        )
        self.assertEqual(self.model.predict.call_count, 2)

    def test_forces_cpu_confidence_and_no_automatic_saving(self) -> None:
        with patch("ultralytics.YOLO", return_value=self.model):
            detector = Detector(self.model_path, 0.7)
            detector.detect(self.frame)
        arguments = self.model.predict.call_args.kwargs
        self.assertIs(arguments["source"], self.frame)
        self.assertEqual(arguments["device"], "cpu")
        self.assertEqual(arguments["conf"], 0.7)
        self.assertFalse(arguments["save"])
        self.assertFalse(arguments["save_txt"])
        self.assertFalse(arguments["verbose"])
        self.assertEqual(arguments["project"], str(CACHE_PATH.resolve()))
        self.assertTrue(arguments["exist_ok"])

    def test_empty_detections_are_an_empty_list(self) -> None:
        boxes = self.model.predict.return_value[0].boxes
        boxes.cls = np.array([])
        boxes.conf = np.array([])
        boxes.xyxy = np.empty((0, 4))
        with patch("ultralytics.YOLO", return_value=self.model):
            self.assertEqual(Detector(self.model_path).detect(self.frame), [])

    def test_missing_model_fails_without_loading_or_downloading(self) -> None:
        missing_path = self.model_path.with_name("missing.pt")
        with (
            patch("ultralytics.YOLO") as load,
            self.assertRaisesRegex(FileNotFoundError, "Model file not found"),
        ):
            Detector(missing_path)
        load.assert_not_called()

    def test_directory_and_non_pt_models_are_rejected(self) -> None:
        wrong_extension = self.model_path.with_suffix(".yaml")
        wrong_extension.write_text("Not weights.", encoding="utf-8")
        for path in (self.model_path.parent, wrong_extension):
            with (
                self.subTest(path=path),
                patch("ultralytics.YOLO") as load,
                self.assertRaisesRegex(ValueError, "local .pt file"),
            ):
                Detector(path)
            load.assert_not_called()

    def test_invalid_weights_have_path_aware_errors(self) -> None:
        errors = (
            OSError("Unreadable weights."), ValueError("Invalid weights."),
            RuntimeError("Invalid checkpoint."), EOFError("Empty checkpoint."),
            pickle.UnpicklingError("Invalid pickle."), KeyError("model"),
            TypeError("Invalid checkpoint type."),
            AttributeError("Invalid checkpoint attributes."),
            ImportError("Unsupported checkpoint module."),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with (
                    patch("ultralytics.YOLO", side_effect=error),
                    self.assertRaises(ValueError) as raised,
                ):
                    Detector(self.model_path)
                self.assertIn("Cannot load model", str(raised.exception))
                self.assertIn(str(self.model_path), str(raised.exception))
                self.assertIs(raised.exception.__cause__, error)

    def test_corrupt_local_weights_fail_with_a_clear_error(self) -> None:
        with self.assertRaises(ValueError) as raised:
            Detector(self.model_path)
        self.assertIn("Cannot load model", str(raised.exception))
        self.assertIn(str(self.model_path), str(raised.exception))

    def test_cache_cannot_escape_data_output(self) -> None:
        with (
            patch("detector.CACHE_PATH", Path("/tmp")),
            self.assertRaisesRegex(ValueError, "cache must be under data/output"),
        ):
            Detector(self.model_path)

    def test_non_detection_weights_are_rejected(self) -> None:
        self.model.task = "classify"
        with (
            patch("ultralytics.YOLO", return_value=self.model),
            self.assertRaisesRegex(ValueError, "not an object detector"),
        ):
            Detector(self.model_path)

    def test_invalid_confidence_is_rejected_before_model_loading(self) -> None:
        for confidence in (-0.1, 1.1, float("nan"), float("inf")):
            with (
                self.subTest(confidence=confidence),
                patch("ultralytics.YOLO") as load,
                self.assertRaisesRegex(ValueError, "Confidence"),
            ):
                Detector(self.model_path, confidence)
            load.assert_not_called()

    def test_invalid_results_are_not_silently_accepted(self) -> None:
        for results in ([], [SimpleNamespace(boxes=None)]):
            self.model.predict.return_value = results
            with (
                self.subTest(results=results),
                patch("ultralytics.YOLO", return_value=self.model),
                self.assertRaisesRegex(ValueError, "invalid bounding-box results"),
            ):
                Detector(self.model_path).detect(self.frame)

    def test_mismatched_result_lengths_fail(self) -> None:
        self.model.predict.return_value[0].boxes.conf = np.array([0.8])
        with (
            patch("ultralytics.YOLO", return_value=self.model),
            self.assertRaises(ValueError),
        ):
            Detector(self.model_path).detect(self.frame)
