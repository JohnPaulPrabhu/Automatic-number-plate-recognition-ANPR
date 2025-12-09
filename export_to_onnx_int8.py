import os
import cv2
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    QuantType,
    QuantFormat,
)


class AnprCalibDataReader(CalibrationDataReader):
    """
    Data reader for ONNX Runtime INT8 calibration.
    Feeds calibration images into the model's input tensor.
    """

    def __init__(self, calib_images_dir, input_name, max_samples=200):
        self.input_name = input_name
        self.max_samples = max_samples

        self.image_paths = [
            os.path.join(calib_images_dir, f)
            for f in os.listdir(calib_images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.image_paths = self.image_paths[:max_samples]
        self.index = 0

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.index]
        self.index += 1

        img = cv2.imread(image_path)
        if img is None:
            return self.get_next()

        # Preprocess similar to YOLO (simple version)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)   # 1x3x640x640

        return {self.input_name: img}


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    fp32_model_path = os.path.join(base_dir, "models", "plate_fp32.onnx")
    int8_model_path = os.path.join(base_dir, "models", "plate_int8.onnx")
    calib_dir = os.path.join(base_dir, "datasets", "anpr_calib", "images")

    print("FP32 model:", fp32_model_path)
    print("Calibration images:", calib_dir)

    # 1. Load ONNX model and get input name
    model = onnx.load(fp32_model_path)
    input_name = model.graph.input[0].name
    print("Model input tensor:", input_name)

    # 2. Build calibration reader
    dr = AnprCalibDataReader(
        calib_images_dir=calib_dir,
        input_name=input_name,
        max_samples=300,
    )

    print("\n[INT8] Starting static quantization (QDQ, per-tensor)...")

    quantize_static(
        model_input=fp32_model_path,
        model_output=int8_model_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,   # 🔹 use QDQ instead of QOperator
        per_channel=False,              # 🔹 per-tensor to avoid 'axis' on DequantizeLinear
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        # optimize_model=True,
        op_types_to_quantize=["Conv", "MatMul"],  # safe default
    )

    print(f"[INT8] Saved INT8 quantized ONNX model:")
    print("       →", int8_model_path)


if __name__ == "__main__":
    main()


# import os
# import cv2
# import numpy as np
# import onnx
# from onnxruntime.quantization import (
#     CalibrationDataReader,
#     quantize_static,
#     QuantType,
# )


# # --------------------------
# # Calibration Data Reader
# # --------------------------
# class AnprCalibDataReader(CalibrationDataReader):
#     """
#     Data reader for ONNX Runtime INT8 calibration.
#     Feeds calibration images into the model's input tensor.
#     """

#     def __init__(self, calib_images_dir, input_name, max_samples=200):
#         self.input_name = input_name
#         self.max_samples = max_samples

#         self.image_paths = [
#             os.path.join(calib_images_dir, f)
#             for f in os.listdir(calib_images_dir)
#             if f.lower().endswith((".jpg", ".jpeg", ".png"))
#         ]
#         self.image_paths = self.image_paths[:max_samples]
#         self.index = 0

#     def get_next(self):
#         if self.index >= len(self.image_paths):
#             return None

#         image_path = self.image_paths[self.index]
#         self.index += 1

#         img = cv2.imread(image_path)
#         if img is None:
#             return self.get_next()

#         # --- Preprocess same as YOLO export ---
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
#         img = img.astype(np.float32) / 255.0
#         img = np.transpose(img, (2, 0, 1))  # HWC → CHW
#         img = np.expand_dims(img, axis=0)   # 1x3x640x640

#         return {self.input_name: img}


# # --------------------------
# # Main INT8 Quantization
# # --------------------------
# def main():
#     base_dir = os.path.dirname(os.path.abspath(__file__))

#     fp32_model_path = os.path.join(base_dir, "models", "plate_fp32.onnx")
#     int8_model_path = os.path.join(base_dir, "models", "plate_int8.onnx")
#     calib_dir = os.path.join(base_dir, "datasets", "anpr_calib", "images")

#     print("FP32 model:", fp32_model_path)
#     print("Calibration images:", calib_dir)

#     # Load ONNX model to introspect input name
#     model = onnx.load(fp32_model_path)
#     input_name = model.graph.input[0].name
#     print("Model input tensor:", input_name)

#     # Build calibration reader
#     dr = AnprCalibDataReader(
#         calib_images_dir=calib_dir,
#         input_name=input_name,
#         max_samples=300,
#     )

#     print("\n[INT8] Starting static quantization...")

#     quantize_static(
#         model_input=fp32_model_path,
#         model_output=int8_model_path,
#         calibration_data_reader=dr,
#         quant_format="QOperator",        # Recommended format
#         per_channel=True,                # Better accuracy
#         activation_type=QuantType.QUInt8,
#         weight_type=QuantType.QInt8,
#         # optimize_model=True,             # Runs ORT graph optimizer
#     )

#     print(f"[INT8] Saved INT8 quantized ONNX model:")
#     print("       →", int8_model_path)


# if __name__ == "__main__":
#     main()





# # # quantize_plate_int8_onnx.py
# # import os
# # import cv2
# # import numpy as np

# # from onnxruntime.quantization import (
# #     CalibrationDataReader,
# #     quantize_static,
# #     QuantType,
# # )


# # class AnprCalibDataReader(CalibrationDataReader):
# #     """
# #     Simple data reader for calibration: feeds images as model input.
# #     Assumes model input is 1x3x640x640, normalized 0-1 or 0-255 depending on your choice.
# #     """

# #     def __init__(self, calib_images_dir, input_name, max_samples=200):
# #         self.input_name = input_name
# #         self.max_samples = max_samples
# #         self.image_paths = [
# #             os.path.join(calib_images_dir, f)
# #             for f in os.listdir(calib_images_dir)
# #             if f.lower().endswith((".jpg", ".jpeg", ".png"))
# #         ]
# #         self.image_paths = self.image_paths[:max_samples]
# #         self._iter = iter(self.image_paths)

# #     def get_next(self):
# #         try:
# #             image_path = next(self._iter)
# #         except StopIteration:
# #             return None

# #         img = cv2.imread(image_path)
# #         if img is None:
# #             return self.get_next()

# #         # Preprocess: BGR -> RGB, resize to 640, normalize, CHW, add batch
# #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
# #         img = img.astype(np.float32) / 255.0
# #         img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
# #         img = np.expand_dims(img, axis=0)   # 1x3x640x640

# #         return {self.input_name: img}


# # def main():
# #     project_root = os.path.dirname(os.path.abspath(__file__))

# #     fp32_path = os.path.join(project_root, "models", "plate_fp32.onnx")
# #     int8_path = os.path.join(project_root, "models", "plate_int8.onnx")
# #     calib_dir = os.path.join(project_root, "datasets", "anpr_calib", "images")

# #     # You need to know the model input name. You can inspect with onnx or onnxruntime:
# #     # Example: "images" or "input_0", etc.
# #     input_name = "images"  # TODO: change to your actual input name

# #     print(f"[INT8] Using FP32 ONNX: {fp32_path}")
# #     print(f"[INT8] Calibration images: {calib_dir}")

# #     dr = AnprCalibDataReader(calib_dir, input_name=input_name, max_samples=200)

# #     quantize_static(
# #         model_input=fp32_path,
# #         model_output=int8_path,
# #         calibration_data_reader=dr,
# #         quant_format="QOperator",     # or "QDQ"
# #         per_channel=True,
# #         reduce_range=False,
# #         activation_type=QuantType.QUInt8,
# #         weight_type=QuantType.QInt8,
# #     )

# #     print(f"[INT8] Saved INT8 ONNX to: {int8_path}")


# # if __name__ == "__main__":
# #     main()
