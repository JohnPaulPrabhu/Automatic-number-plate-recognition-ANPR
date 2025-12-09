from ultralytics import YOLO
import torch
import os


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_pt = "model_training/runs/anpr_yolo11/yolo11n_anpr4/weights/best.pt"

    print(f"[Export] Loading plate model from: {weights_pt}")
    model = YOLO(weights_pt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("[Export] Exporting TensorRT FP16 engine...")
    model.export(
        format="engine",   # TensorRT
        imgsz=640,         # must match training/inference size
        # half=True,         # FP16
        dynamic=False,     # fixed 1x3x640x640
        device=0,          # GPU 0
    )

    print("[Export] Done. Check for '.engine' file next to best.pt.")


if __name__ == "__main__":
    main()
