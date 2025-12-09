from pathlib import Path
from typing import List, Tuple

import torch
from ultralytics import YOLO


class PlateDetector:
    """
    YOLO-based number plate detector.

    Supports:
      - PyTorch .pt
      - ONNX .onnx
      - TensorRT .engine
    All via Ultralytics YOLO wrapper (handles letterbox, postprocess, etc.).
    """

    def __init__(
        self,
        weights_path: str = "runs/anpr_yolo11/yolo11n_anpr/weights/best.pt",
        device: str | None = None,
        conf_thres: float = 0.4,
        allowed_class_names: Tuple[str, ...] = ("license_plate", "plate", "number_plate"),
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.conf_thres = conf_thres
        self.weights_path = weights_path

        suffix = Path(weights_path).suffix.lower()
        self.is_pytorch = suffix == ".pt"

        print(f"[PlateDetector] Loading model: {weights_path}")
        self.model = YOLO(weights_path)

        # Only PyTorch models support .to()
        if self.is_pytorch:
            self.model.to(self.device)

        self.class_names = self.model.names  # dict {id: name}

        # Identify plate class IDs
        self.plate_class_ids = []
        for cid, name in self.class_names.items():
            name_lower = name.lower()
            if any(key in name_lower for key in allowed_class_names):
                self.plate_class_ids.append(cid)

        if not self.plate_class_ids:
            print("[PlateDetector] Warning: no specific plate class IDs found, using all classes.")
            self.plate_class_ids = list(self.class_names.keys())

        # print(f"[PlateDetector] Using plate class IDs: {self.plate_class_ids}")
        print(f"[PlateDetector] Backend: {'PyTorch' if self.is_pytorch else 'Exported (ONNX/engine/etc.)'}")

    def _infer_pytorch(self, frame_bgr) -> List[list]:
        results = self.model(
            frame_bgr,
            verbose=False,
            device=self.device,
        )[0]
        return self._results_to_plate_dets(results)

    def _infer_exported(self, frame_bgr) -> List[list]:
        # For ONNX / engine, we pass device index or 'cpu'
        if "cuda" in str(self.device):
            dev_arg = 0
        else:
            dev_arg = "cpu"

        results = self.model.predict(
            frame_bgr,
            verbose=False,
            device=dev_arg,
        )[0]
        return self._results_to_plate_dets(results)

    def _results_to_plate_dets(self, results) -> List[list]:
        detections: List[list] = []
        if results.boxes is None:
            return detections

        # Ultralytics already returns xyxy in original image coords
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

    def infer(self, frame_bgr) -> List[list]:
        if self.is_pytorch:
            return self._infer_pytorch(frame_bgr)
        else:
            return self._infer_exported(frame_bgr)
