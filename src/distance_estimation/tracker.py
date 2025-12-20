"""Simple IoU-based tracker for persistent object IDs.

For production, consider SORT, DeepSORT, or ByteTrack.
"""
from typing import List, Tuple


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


class Track:
    def __init__(self, tid: int, bbox: Tuple[int, int, int, int]):
        self.id = tid
        self.bbox = bbox
        self.missed = 0


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 10):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, detections: List[Tuple[int, float, Tuple[int, int, int, int]]]):
        bboxes = [d[2] for d in detections]
        assigned = set()
        new_tracks: List[Track] = []

        for tr in self.tracks:
            best_iou = 0.0
            best_idx = -1
            for i, bbox in enumerate(bboxes):
                if i in assigned:
                    continue
                val = iou(tr.bbox, bbox)
                if val > best_iou:
                    best_iou = val
                    best_idx = i
            if best_idx != -1 and best_iou >= self.iou_threshold:
                tr.bbox = bboxes[best_idx]
                tr.missed = 0
                assigned.add(best_idx)
                new_tracks.append(tr)
            else:
                tr.missed += 1
                if tr.missed <= self.max_missed:
                    new_tracks.append(tr)
        for i, bbox in enumerate(bboxes):
            if i not in assigned:
                tr = Track(self.next_id, bbox)
                self.next_id += 1
                new_tracks.append(tr)
        self.tracks = new_tracks
        return [(tr.id, tr.bbox) for tr in self.tracks]
