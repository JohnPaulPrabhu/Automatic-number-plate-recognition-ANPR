# src/cuda_preproc.py
import ctypes
from pathlib import Path
from typing import Tuple

import numpy as np


class CudaPreprocessor:
    """
    Thin wrapper around cuda_preproc.dll.

    Expects the DLL to export:
      - void* preproc_create(int max_src_w, int max_src_h, int dst_w, int dst_h)
      - int   preproc_run(void* handle, const uint8_t* src_bgr, int src_w, int src_h, float* dst_chw)
      - void  preproc_destroy(void* handle)

    Output: numpy.float32 array with shape (1, 3, dst_h, dst_w), RGB, normalized [0,1].
    """

    def __init__(
        self,
        dll_path: str = "cpp/cuda_preproc.dll",
        max_src_size: Tuple[int, int] = (1920, 1080),
        dst_size: Tuple[int, int] = (640, 640),
    ):
        self.max_src_w, self.max_src_h = max_src_size
        self.dst_w, self.dst_h = dst_size

        lib_full = str(Path(dll_path).resolve())
        self.lib = ctypes.cdll.LoadLibrary(lib_full)

        # C signatures
        self.lib.preproc_create.restype = ctypes.c_void_p
        self.lib.preproc_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]

        self.lib.preproc_run.restype = ctypes.c_int
        self.lib.preproc_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]

        self.lib.preproc_destroy.restype = None
        self.lib.preproc_destroy.argtypes = [ctypes.c_void_p]

        # Create context
        self.handle = self.lib.preproc_create(
            self.max_src_w,
            self.max_src_h,
            self.dst_w,
            self.dst_h,
        )
        if not self.handle:
            raise RuntimeError("Failed to create CUDA preproc context")

        print("[CudaPreprocessor] Created context, dst size =", self.dst_w, self.dst_h)

    def __del__(self):
        try:
            if getattr(self, "handle", None):
                self.lib.preproc_destroy(self.handle)
        except Exception:
            pass

    def preprocess(self, frame_bgr) -> np.ndarray:
        """
        frame_bgr: HxWx3 uint8 (OpenCV frame, BGR)
        Returns: numpy.float32, shape (1, 3, dst_h, dst_w), RGB, normalized [0,1].
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame")

        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)

        h, w, c = frame_bgr.shape
        assert c == 3

        frame_bgr_c = np.ascontiguousarray(frame_bgr)

        out = np.empty((1, 3, self.dst_h, self.dst_w), dtype=np.float32)
        flat = out.reshape(-1)

        ret = self.lib.preproc_run(
            self.handle,
            frame_bgr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w,
            h,
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if ret != 0:
            raise RuntimeError(f"preproc_run failed with code {ret}")

        return out
