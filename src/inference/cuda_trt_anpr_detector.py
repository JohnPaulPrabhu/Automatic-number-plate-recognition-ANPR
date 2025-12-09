import ctypes
from pathlib import Path
from typing import List

import cv2
import numpy as np


class CudaTrtAnprDetector:
    def __init__(
        self,
        engine_path: str = "models/plate_fp16.engine",
        lib_path: str = "cpp/anpr_trt_cuda.dll",
        input_w: int = 640,
        input_h: int = 640,
        max_detections: int = 200,
        conf_thres: float = 0.4,
    ):
        self.engine_path = str(engine_path)
        self.input_w = input_w
        self.input_h = input_h
        self.max_detections = max_detections
        self.conf_thres = conf_thres

        # Load shared library
        lib_full = str(Path(lib_path).resolve())
        self.lib = ctypes.cdll.LoadLibrary(lib_full)

        # Declare signatures
        self.lib.anpr_create.restype = ctypes.c_void_p
        self.lib.anpr_create.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.anpr_infer.restype = ctypes.c_int
        self.lib.anpr_infer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]

        self.lib.anpr_destroy.restype = None
        self.lib.anpr_destroy.argtypes = [ctypes.c_void_p]

        # Create engine handle
        self.handle = self.lib.anpr_create(
            self.engine_path.encode("utf-8"),
            self.input_w,
            self.input_h,
            self.max_detections,
        )

        if not self.handle:
            raise RuntimeError("Failed to create ANPR TensorRT + CUDA engine")

        print("[CudaTrtAnprDetector] Created engine for:", self.engine_path)

    def __del__(self):
        try:
            if getattr(self, "handle", None):
                self.lib.anpr_destroy(self.handle)
        except Exception:
            pass

    def infer(self, frame_bgr: np.ndarray) -> List[list]:
        """
        frame_bgr: HxWx3 uint8 (OpenCV image)
        Returns: list of [x1,y1,x2,y2,score] in original frame coordinates.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)

        h, w, c = frame_bgr.shape
        assert c == 3

        # Ensure contiguous
        frame_bgr_c = np.ascontiguousarray(frame_bgr)

        # Output: [max_detections, 5]
        out_boxes = np.zeros((self.max_detections, 5), dtype=np.float32)

        num = self.lib.anpr_infer(
            self.handle,
            frame_bgr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w, h,
            ctypes.c_float(self.conf_thres),
            out_boxes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.max_detections,
        )

        if num <= 0:
            return []

        dets = out_boxes[:num].tolist()
        return dets
