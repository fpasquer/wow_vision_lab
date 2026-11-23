import pickle
from math import isfinite
from pathlib import Path
from typing import TypedDict

from cv2.typing import MatLike

CACHE_PATH = Path("data/output/.cache")
DEFAULT_CONFIDENCE = 0.25


class Detection(TypedDict):
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class Detector:
    def __init__(self, model_path: Path, confidence: float = DEFAULT_CONFIDENCE) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not model_path.is_file() or model_path.suffix.lower() != ".pt":
            raise ValueError(f"Model must be a local .pt file: {model_path}")
        if not isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Confidence must be a finite number from 0 to 1.")
        if not CACHE_PATH.resolve().is_relative_to(Path("data/output").resolve()):
            raise ValueError("Detector cache must be under data/output.")

        CACHE_PATH.mkdir(parents=True, exist_ok=True)
        # Import only for detection, after the writable config parent exists.
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(2)
        try:
            self.model = YOLO(str(model_path.resolve()), task="detect")
        except (
            OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError,
            KeyError, TypeError, AttributeError, ImportError,
        ) as error:
            raise ValueError(f"Cannot load model {model_path}: {error}") from error
        if self.model.task != "detect":
            raise ValueError(f"Model is not an object detector: {model_path}")
        self.confidence = confidence

    def detect(self, frame: MatLike) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            device="cpu",
            conf=self.confidence,
            save=False,
            save_txt=False,
            verbose=False,
            project=str(CACHE_PATH.resolve()),
            name="predict",
            exist_ok=True,
        )
        if len(results) != 1:
            raise ValueError("Detection model returned invalid bounding-box results.")
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            raise ValueError("Detection model returned invalid bounding-box results.")
        return [
            {
                "class_id": int(class_id),
                "class_name": result.names[int(class_id)],
                "confidence": float(confidence),
                "bbox": [float(value) for value in coordinates],
            }
            for class_id, confidence, coordinates in zip(
                boxes.cls.tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.tolist(),
                strict=True,
            )
        ]