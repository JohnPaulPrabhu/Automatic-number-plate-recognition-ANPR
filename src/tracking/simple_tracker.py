import numpy as np
from collections import deque

def iou(boxA, boxB):
    """
    Compute IoU between two boxes: [x1, y1, x2, y2]
    """

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

    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val


class SimpleTracker:
    """
    A simple IoU-based tracker.
    Each detection is assigned to an existing track if IoU > iou_thres.
    
    Track Format:
        {
            "id": int,
            "bbox": [x1, y1, x2, y2],
            "score": float,
            "class": int
        }

    Input Format per detection:
        [x1, y1, x2, y2, score, class_id]
    """

    def __init__(self, iou_thres=0.3, max_age=15):
        self.iou_thres = iou_thres
        self.max_age = max_age

        self.next_track_id = 0
        self.tracks = {}           # track_id → track dict
        self.track_age = {}        # track_id → age since last update

    def update(self, detections):
        """
        Update tracker with new detections.
        
        Returns:
            List of active tracks:
                [{
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "score": score,
                    "class": cls_id
                }, ...]
        """

        assigned_dets = set()
        updated_tracks = {}

        # --- 1. Try to match detections to existing tracks ---
        for track_id, track in self.tracks.items():
            best_iou = 0
            best_det_idx = -1

            for i, det in enumerate(detections):
                if i in assigned_dets:
                    continue

                det_box = det[:4]
                track_box = track["bbox"]

                iou_val = iou(track_box, det_box)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det_idx = i

            if best_iou >= self.iou_thres:
                # Match found → update track
                det = detections[best_det_idx]
                updated_tracks[track_id] = {
                    "id": track_id,
                    "bbox": det[:4],
                    "score": det[4],
                    "class": det[5],
                }
                assigned_dets.add(best_det_idx)
                self.track_age[track_id] = 0  # reset age
            else:
                # No match → age increases
                self.track_age[track_id] += 1
                if self.track_age[track_id] <= self.max_age:
                    updated_tracks[track_id] = track

        # --- 2. Create new tracks for unmatched detections ---
        for i, det in enumerate(detections):
            if i in assigned_dets:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1

            updated_tracks[track_id] = {
                "id": track_id,
                "bbox": det[:4],
                "score": det[4],
                "class": det[5],
            }
            self.track_age[track_id] = 0

        # --- 3. Save updated tracks ---
        self.tracks = updated_tracks

        # Output sorted by track_id for stability
        output_tracks = list(self.tracks.values())
        output_tracks.sort(key=lambda t: t["id"])

        return output_tracks
