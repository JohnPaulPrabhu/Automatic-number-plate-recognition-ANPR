# export_plate_onnx.py
from ultralytics import YOLO
import torch
import os


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_pt="model_training/runs/anpr_yolo11/yolo11n_anpr4/weights/best.pt"


    print(f"[Export] Loading plate model from: {weights_pt}")
    model = YOLO(weights_pt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    onnx_path = os.path.join(project_root, "models", "plate_fp32.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    print("[Export] Exporting FP32 ONNX...")
    model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,
        simplify=True,
        opset=12,
        # Will save to best.onnx by default in same dir;
        # we'll just move/rename manually if needed.
    )

    print("[Export] Done. Check generated .onnx and rename to plate_fp32.onnx as needed.")


if __name__ == "__main__":
    main()
