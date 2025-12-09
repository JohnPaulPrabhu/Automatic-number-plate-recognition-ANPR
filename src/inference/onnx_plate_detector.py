# src/inference/onnx_plate_detector.py
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnxruntime as ort


class OnnxPlateDetector:
    """
    ONNX Runtime-based plate detector.
    Works with FP32 / FP16 / INT8 ONNX models.
    """

    def __init__(
        self,
        onnx_path: str = "models/plate_fp32.onnx",
        conf_thres: float = 0.4,
        iou_thres: float = 0.5,
        providers: Tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider"),
    ):
        self.onnx_path = onnx_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        print(f"[OnnxPlateDetector] Loading ONNX model: {onnx_path}")
        so = ort.SessionOptions()
        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=list(providers))

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1,3,640,640] likely
        self.output_names = [o.name for o in self.session.get_outputs()]

        print(f"[OnnxPlateDetector] Input: {self.input_name}, shape={self.input_shape}")
        print(f"[OnnxPlateDetector] Outputs: {self.output_names}")

        # You need to adapt post-processing based on YOLO ONNX export format.
        # Typical Ultralytics ONNX gives one output with [batch, num_boxes, 85] (xywh + obj + classes).

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # HWC -> CHW
        img = np.expand_dims(img, 0)         # 1x3x640x640
        return img

    def _postprocess(self, outputs, orig_shape) -> List[list]:
        """
        Parse YOLO ONNX outputs -> [x1, y1, x2, y2, score] for plate class only.
        You must adapt to your exported ONNX format.

        Pseudocode-ish; you'll need to align with your exact output tensor structure.
        """
        h0, w0 = orig_shape[:2]

        preds = outputs[0]  # [1, num_boxes, n_attrs]
        preds = np.squeeze(preds, axis=0)

        boxes_scores = []
        for det in preds:
            # Example YOLO format: [cx, cy, w, h, obj_conf, class1_conf, ...]
            cx, cy, w, h = det[0:4]
            obj_conf = det[4]
            class_scores = det[5:]
            cls_id = int(np.argmax(class_scores))
            cls_conf = class_scores[cls_id]
            score = obj_conf * cls_conf
            if score < self.conf_thres:
                continue

            # If this model has only one class (plate), cls_id is always 0
            # Convert from cx,cy,w,h in [0,640] to x1,y1,x2,y2 in original image space
            x1 = (cx - w / 2) / 640.0 * w0
            y1 = (cy - h / 2) / 640.0 * h0
            x2 = (cx + w / 2) / 640.0 * w0
            y2 = (cy + h / 2) / 640.0 * h0

            boxes_scores.append([x1, y1, x2, y2, float(score)])

        # Optionally apply NMS in Python if needed

        return boxes_scores

    def infer(self, frame_bgr: np.ndarray) -> List[list]:
        inp = self._preprocess(frame_bgr)
        outputs = self.session.run(self.output_names, {self.input_name: inp})
        # breakpoint()
        dets = self._postprocess(outputs, frame_bgr.shape)
        return dets
