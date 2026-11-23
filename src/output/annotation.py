from collections.abc import Sequence

import cv2
from cv2.typing import MatLike

from detector import Detection


def _annotate_frame(frame: MatLike, detections: Sequence[Detection]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(
            frame, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 255, 0), 1,
        )
