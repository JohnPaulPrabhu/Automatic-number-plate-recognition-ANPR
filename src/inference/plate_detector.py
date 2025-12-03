from pathlib import Path
from typing import List, Tuple

import torch
from ultralytics import YOLO


class PlateDetector:
    """
    YOLO-based number plate detector (FP32).
    Assumes the model is trained to detect license plates.
    """

    def __init__(
        self,
        weights_path: str = "model_training/runs/anpr_yolo11/yolo11n_anpr4/weights/best.pt",
        device: str | None = None,
        conf_thres: float = 0.4,
        allowed_class_names: Tuple[str, ...] = ("license_plate", "plate", "number_plate"),
    ):
        """
        :param weights_path: Path to YOLO weights (.pt)
        :param device: 'cuda', 'cpu', or None for auto
        :param conf_thres: confidence threshold
        :param allowed_class_names: class names to treat as plates
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.conf_thres = conf_thres

        # Load YOLO model
        self.model = YOLO(weights_path)
        self.model.to(self.device)

        # Find plate class IDs based on class names from the model
        self.class_names = self.model.names  # dict: {id: name}
        self.plate_class_ids = []

        for cid, name in self.class_names.items():
            name_lower = name.lower()
            if any(key in name_lower for key in allowed_class_names):
                self.plate_class_ids.append(cid)

        if not self.plate_class_ids:
            print("[Warning] No plate class IDs detected from model.names. "
                  "All classes will be treated as plates.")
            # If no match, fall back to all classes
            self.plate_class_ids = list(self.class_names.keys())

        print(f"[PlateDetector] Using plate class IDs: {self.plate_class_ids}")

    def infer(self, frame_bgr) -> List[list]:
        """
        Run detection on a single BGR frame (H x W x 3, uint8).

        Returns:
            List of plate detections:
            [x1, y1, x2, y2, score]
        """
        results = self.model(
            frame_bgr,
            verbose=False,
            device=self.device,
        )[0]

        detections: List[list] = []

        if results.boxes is None:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, scores, classes):
            if score < self.conf_thres:
                continue

            cls_id = int(cls)
            if cls_id not in self.plate_class_ids:
                continue

            x1, y1, x2, y2 = box.tolist()
            detections.append([x1, y1, x2, y2, float(score)])

        return detections
