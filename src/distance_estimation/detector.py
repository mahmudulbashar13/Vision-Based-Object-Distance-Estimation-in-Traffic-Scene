"""Detector using YOLO 11 via Ultralytics.

Supports user-supplied YOLO weights (YOLOv8, YOLOv5, YOLOv11, or compatible).
"""
from typing import List, Tuple
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class YOLODetector:
    def __init__(self, weights: str, device: str = "cpu"):
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. Install with `pip install ultralytics`.")
        self.model = YOLO(weights)
        self.device = device

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.25) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """Detect objects in a BGR frame.

        Returns list of tuples: (class_id, confidence, (x1,y1,x2,y2))
        """
        results = self.model(frame, imgsz=640, conf=conf_thresh)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls.cpu().numpy()[0]) if hasattr(box, "cls") else int(box.cls[0])
                conf = float(box.conf.cpu().numpy()[0]) if hasattr(box, "conf") else float(box.conf[0])
                xyxy = box.xyxy.cpu().numpy()[0] if hasattr(box, "xyxy") else box.xyxy[0]
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                detections.append((cls, conf, (x1, y1, x2, y2)))
        return detections
