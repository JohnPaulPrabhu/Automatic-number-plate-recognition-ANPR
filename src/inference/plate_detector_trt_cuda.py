# src/inference/plate_detector_trt_cuda.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import os
os.environ["TRT_CUTENSOR_DISABLE"] = "1"

import numpy as np
# import pycuda.autoinit  # noqa: F401 - initializes CUDA context
import pycuda.driver as cuda
import tensorrt as trt

from src.cuda_preproc import CudaPreprocessor


@dataclass
class HostDeviceMem:
    host: np.ndarray
    device: cuda.DeviceAllocation


class PlateDetectorTrtCuda:
    """
    TensorRT + custom CUDA preprocessing plate detector.

    Assumptions:
      - Engine has 1 input ('images') and 1 output ('output0')
      - Output shape is (1, 5, N) or (1, N, 5) where attrs = [cx, cy, w, h, conf]
      - Single-class ANPR (all detections are plates)

    This class:
      - Uses cuda_preproc.dll to do GPU preprocessing: BGR HWC -> RGB CHW 640x640 float32 [0,1]
      - Copies preprocessed tensor to TensorRT input buffer
      - Runs inference with execute_async_v3
      - Decodes output on CPU and does NMS
    """

    def __init__(
        self,
        engine_path: str = "models/plate_fp16.engine",
        conf_thres: float = 0.4,
        nms_iou_thres: float = 0.5,
        max_detections: int = 200,
        input_size: Tuple[int, int] = (640, 640),
        max_src_size: Tuple[int, int] = (1920, 1080),
        cuda_preproc_dll: str = "cpp/cuda_preproc.dll",
    ):
        cuda.init()
        self.ctx = cuda.Device(0).retain_primary_context()
        self.ctx.push()
        self.engine_path = str(engine_path)
        self.conf_thres = conf_thres
        self.nms_iou_thres = nms_iou_thres
        self.max_detections = max_detections
        self.input_w, self.input_h = input_size
        # self.cfx = pycuda.autoinit.context

        # 1. CUDA preprocessor
        self.preproc = CudaPreprocessor(
            dll_path=cuda_preproc_dll,
            max_src_size=max_src_size,
            dst_size=input_size,
        )

        # 2. TensorRT runtime + engine + context
        logger = trt.Logger(trt.Logger.WARNING)
        self.logger = logger

        # engine_bytes = Path(self.engine_path).read_bytes()
        # self.runtime = trt.Runtime(logger)
        # self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        # if self.engine is None:
        #     raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")

        engine_file = Path(self.engine_path)
        if not engine_file.exists():
            raise FileNotFoundError(f"Engine file not found: {engine_file.resolve()}")

        engine_bytes = engine_file.read_bytes()
        print(f"[PlateDetectorTrtCuda] TensorRT Python version: {trt.__version__}")
        print(f"[PlateDetectorTrtCuda] Engine path: {engine_file.resolve()}, size={len(engine_bytes)/1024/1024:.2f} MB")

        self.runtime = trt.Runtime(logger)
        try:
            self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        except Exception as e:
            raise RuntimeError(f"TensorRT failed to deserialize engine: {e}") from e
    
        print("[TRT] Engine tensors:")
        for name in self.engine:
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            print("   ", name,
                "| input" if is_input else "| output",
                "| shape:", self.engine.get_tensor_shape(name),
                "| dtype:", self.engine.get_tensor_dtype(name))


        if self.engine is None:
            raise RuntimeError(
                "deserialize_cuda_engine returned None. "
                "This usually means the engine was built with a different TensorRT major "
                "version than the one used by this Python runtime."
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        print(f"[PlateDetectorTrtCuda] Loaded engine: {self.engine_path}")
        print(f"[PlateDetectorTrtCuda] Engine has {self.engine.num_io_tensors} I/O tensors")

        # 3. Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        # Cache output shape for reshape
        self.output_shape = self.engine.get_tensor_shape(self._output_name())
        print(f"[PlateDetectorTrtCuda] Output shape: {self.output_shape}")

    # ---------- Internal helpers ----------

    def _input_name(self) -> str:
        # Assumes single input; if names differ, adjust here.
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                return name
        raise RuntimeError("No input tensor found in engine")

    def _output_name(self) -> str:
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                return name
        raise RuntimeError("No output tensor found in engine")

    def _allocate_buffers(self):
        inputs: List[HostDeviceMem] = []
        outputs: List[HostDeviceMem] = []
        bindings: List[int] = []
        stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(trt.volume(shape))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
                print(f"[PlateDetectorTrtCuda] Input tensor '{name}' shape={shape}, dtype={dtype}")
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
                print(f"[PlateDetectorTrtCuda] Output tensor '{name}' shape={shape}, dtype={dtype}")

        return inputs, outputs, bindings, stream

    # ---------- Public API ----------

    # def infer(self, frame_bgr) -> List[list]:
    #     """
    #     frame_bgr: HxWx3 BGR uint8 from cv2.VideoCapture
    #     Returns: list of [x1, y1, x2, y2, score] in original image coordinates.
    #     """
    #     if frame_bgr is None or frame_bgr.size == 0:
    #         return []

    #     orig_h, orig_w, _ = frame_bgr.shape

    #     # 1. CUDA preprocessing (BGR -> RGB, resize 640x640, normalize, CHW)
    #     img_chw = self.preproc.preprocess(frame_bgr)  # (1,3,H,W) float32
    #     # Flatten to 1D
    #     img_flat = img_chw.ravel()

    #     # 2. Copy to host input buffer
    #     # Assume single input
    #     host_in = self.inputs[0].host
    #     device_in = self.inputs[0].device
    #     host_out = self.outputs[0].host
    #     device_out = self.outputs[0].device

    #     np.copyto(host_in, img_flat.astype(host_in.dtype, copy=False))

    #     # H2D
    #     cuda.memcpy_htod_async(device_in, host_in, self.stream)

    #     # 3. Set tensor addresses and run inference
    #     for i in range(self.engine.num_io_tensors):
    #         name = self.engine.get_tensor_name(i)
    #         self.context.set_tensor_address(name, self.bindings[i])

    #     # If dynamic shapes are involved, set input shape once:
    #     # self.context.set_input_shape(self._input_name(), img_chw.shape)

    #     # Run
    #     self.context.execute_async_v3(stream_handle=self.stream.handle)

    #     # 4. D2H for outputs
    #     cuda.memcpy_dtoh_async(host_out, device_out, self.stream)
    #     self.stream.synchronize()

    #     # 5. Decode
    #     out_np = np.array(host_out, copy=True)
    #     out_np = out_np.reshape(self.output_shape)  # e.g., (1, 5, 8400) or (1, 8400, 5)

    #     detections = self._decode_yolo_output(out_np, orig_w, orig_h)
    #     return detections
    def infer(self, frame_bgr) -> List[list]:
        """
            frame_bgr: HxWx3 BGR uint8 from cv2.VideoCapture
            Returns: list of [x1, y1, x2, y2, score] in original image coordinates.
        """
        # self.cfx.push()
        # try:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        orig_h, orig_w, _ = frame_bgr.shape

        # 1. CUDA preprocessing (BGR -> RGB, resize 640x640, normalize, CHW)
        img_chw = self.preproc.preprocess(frame_bgr)  # (1, 3, 640, 640) float32
        assert img_chw.shape == (1, 3, self.input_h, self.input_w), \
            f"Preproc shape mismatch: {img_chw.shape}"

        # ---- 🔴 IMPORTANT: tell TensorRT what the input shape is ----
        ok = self.context.set_input_shape(self._input_name(), img_chw.shape)
        if not ok:
            raise RuntimeError(
                f"Failed to set input shape {img_chw.shape} "
                f"for tensor '{self._input_name()}'"
            )

        # 2. Prepare host/device buffers
        img_flat = img_chw.ravel()

        host_in = self.inputs[0].host
        device_in = self.inputs[0].device
        host_out = self.outputs[0].host
        device_out = self.outputs[0].device

        # Ensure types match (float32 or float16 depending on engine IO)
        np.copyto(host_in, img_flat.astype(host_in.dtype, copy=False))

        # H2D
        cuda.memcpy_htod_async(device_in, host_in, self.stream)

        # 3. Bind tensor addresses and run inference
        # ---- 🔴 IMPORTANT: clear previous addresses before re-binding ----
        # self.context.clear_tensor_addresses()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])

        # Run
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 4. D2H for outputs
        cuda.memcpy_dtoh_async(host_out, device_out, self.stream)
        self.stream.synchronize()

        # 5. Decode
        out_np = np.array(host_out, copy=True)
        out_np = out_np.reshape(self.output_shape)

        detections = self._decode_yolo_output(out_np, orig_w, orig_h)
        return detections
        # finally:
        #     self.cfx.pop()



    # ---------- YOLO decode + NMS ----------

    def _decode_yolo_output(self, out, img_w: int, img_h: int) -> List[list]:
        """
        out: numpy array with shape (1, 5, N) or (1, N, 5)
        Returns: list of [x1, y1, x2, y2, score]
        """
        if out.ndim != 3:
            print(f"[PlateDetectorTrtCuda] Unexpected output dims: {out.shape}")
            return []

        b, d1, d2 = out.shape
        assert b == 1, "Batch >1 not supported in this demo"

        if d1 == 5:
            # (1, 5, N)
            N = d2
            cx = out[0, 0, :]
            cy = out[0, 1, :]
            w = out[0, 2, :]
            h = out[0, 3, :]
            conf = out[0, 4, :]
        elif d2 == 5:
            # (1, N, 5)
            N = d1
            cx = out[0, :, 0]
            cy = out[0, :, 1]
            w = out[0, :, 2]
            h = out[0, :, 3]
            conf = out[0, :, 4]
        else:
            print(f"[PlateDetectorTrtCuda] Cannot interpret output shape={out.shape}")
            return []

        # Filter by confidence
        mask = conf >= self.conf_thres
        cx = cx[mask]
        cy = cy[mask]
        w = w[mask]
        h = h[mask]
        conf = conf[mask]

        if cx.size == 0:
            return []

        # Convert to xyxy in model input space (640x640)
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        # Map back to original image size (we used simple resize to 640x640)
        scale_x = img_w / float(self.input_w)
        scale_y = img_h / float(self.input_h)

        x1 *= scale_x
        x2 *= scale_x
        y1 *= scale_y
        y2 *= scale_y

        boxes = np.stack([x1, y1, x2, y2, conf], axis=-1)  # [N, 5]
        boxes = self._nms(boxes, self.nms_iou_thres, self.max_detections)
        return boxes.tolist()

    @staticmethod
    def _nms(boxes: np.ndarray, iou_thres: float, max_dets: int) -> np.ndarray:
        """
        boxes: [N, 5] = [x1, y1, x2, y2, score]
        Returns: filtered boxes [M, 5]
        """
        if boxes.size == 0:
            return boxes

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        scores = boxes[:, 4]

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0 and len(keep) < max_dets:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (x2[i] - x1[i]) * (y2[i] - y1[i])
            area_others = (x2[order[1:]] - x1[order[1:]]) * (
                y2[order[1:]] - y1[order[1:]]
            )
            union = area_i + area_others - inter + 1e-6
            iou = inter / union

            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]

        return boxes[keep]
