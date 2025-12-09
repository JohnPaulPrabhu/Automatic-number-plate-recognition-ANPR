from ultralytics import YOLO
import torch
import os


def main():
    # project_root = os.path.dirname(os.path.abspath(__file__))
    weights_pt = "model_training/runs/anpr_yolo11/yolo11n_anpr4/weights/best.pt"

    print(f"[Export] Loading plate model from: {weights_pt}")
    model = YOLO(weights_pt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    breakpoint()
    model.to(device)

    print("[Export] Exporting FP16 ONNX...")
    model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,
        simplify=True,
        half=True,     # 🔥 ask Ultralytics to export FP16 directly
        opset=18,
        device=0
    )

    print("[Export] Done. Rename the generated .onnx to models/plate_fp16.onnx")


# # convert_fp32_to_fp16.py
# import os
# from onnxconverter_common import float16



# import onnx


# def main():
#     project_root = os.path.dirname(os.path.abspath(__file__))
#     # fp32_path = "model_training/runs/anpr_yolo11/yolo11n_anpr4/weights/best.pt"
#     fp32_path = os.path.join(project_root, "models", "plate_fp32.onnx")
#     fp16_path = os.path.join(project_root, "models", "plate_fp16.onnx")

#     print(f"[FP16] Loading FP32 ONNX: {fp32_path}")
#     model = onnx.load(fp32_path)

#     print("[FP16] Converting to FP16...")
#     # keep_io_types: True → input/output stay FP32, internal becomes FP16
#     model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)

#     onnx.save(model_fp16, fp16_path)
#     print(f"[FP16] Saved FP16 ONNX to: {fp16_path}")


if __name__ == "__main__":
    main()
