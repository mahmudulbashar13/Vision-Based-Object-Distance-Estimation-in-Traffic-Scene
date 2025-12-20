"""Main pipeline: video -> YOLO detection -> tracking -> depth -> distance estimation.

Outputs annotated video (with bounding boxes, IDs, distances) and CSV with per-frame distances.
"""
from typing import Optional
import cv2
import numpy as np
import csv
from .detector import YOLODetector
from .tracker import SimpleTracker
from .depth import DepthEstimator


class DistanceEstimationPipeline:
    def __init__(self, detector_weights: str, depth_model_type: str = "midas", depth_checkpoint: Optional[str] = None, device: str = "cpu"):
        self.detector = YOLODetector(detector_weights, device=device)
        self.tracker = SimpleTracker()
        self.depth = DepthEstimator(model_type=depth_model_type, checkpoint=depth_checkpoint, device=device)

    def estimate_distance_for_bbox(self, depth_map: np.ndarray, bbox: tuple) -> float:
        """Estimate distance (meters if using Depth Anything v2) for a bounding box."""
        x1, y1, x2, y2 = bbox
        h, w = depth_map.shape[:2]
        x1c = max(0, min(w - 1, x1))
        x2c = max(0, min(w - 1, x2))
        y1c = max(0, min(h - 1, y1))
        y2c = max(0, min(h - 1, y2))
        if x2c <= x1c or y2c <= y1c:
            return float('nan')
        crop = depth_map[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            return float('nan')
        return float(np.median(crop))

    def run(self, input_video: str, output_video: str, output_csv: str, conf: float = 0.25):
        """Process video and produce annotated output + distance CSV."""
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {input_video}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

        csv_file = open(output_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'track_id', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2', 'distance_m'])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = self.detector.detect(frame, conf_thresh=conf)
            tracks = self.tracker.update(detections)

            # compute depth once per frame
            depth_map = self.depth.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            for tr_id, bbox in tracks:
                # find best matching detection for class info
                best = None
                best_iou = 0.0
                for det in detections:
                    cls, confd, dbbox = det
                    xa1, ya1, xa2, ya2 = bbox
                    xb1, yb1, xb2, yb2 = dbbox
                    inter_x1 = max(xa1, xb1)
                    inter_y1 = max(ya1, yb1)
                    inter_x2 = min(xa2, xb2)
                    inter_y2 = min(ya2, yb2)
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h
                    areaA = (xa2 - xa1) * (ya2 - ya1)
                    areaB = (xb2 - xb1) * (yb2 - yb1)
                    iou_val = inter_area / (areaA + areaB - inter_area + 1e-8) if (areaA+areaB-inter_area)>0 else 0
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best = (cls, confd, dbbox)
                cls, confd, dbbox = best if best is not None else (-1, 0.0, bbox)
                distance = self.estimate_distance_for_bbox(depth_map, bbox)
                
                # annotate frame
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{tr_id} C:{cls} D:{distance:.2f}m"
                cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # write csv
                csv_writer.writerow([frame_idx, tr_id, cls, confd, x1, y1, x2, y2, distance])

            out.write(frame)
            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"Processed {frame_idx} frames...")

        cap.release()
        out.release()
        csv_file.close()
        print(f"Completed: {output_video} and {output_csv}")
