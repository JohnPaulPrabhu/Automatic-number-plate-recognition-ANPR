from ultralytics import YOLO
import torch
import os


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    weights_pt = "models/best.pt"
    calib_yaml = os.path.join(
        project_root,
        "datasets",
        "anpr_calib",
        "data_calib.yaml",
    )

    print(f"[Export-INT8] Loading plate model from: {weights_pt}")
    model = YOLO(weights_pt)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"[Export-INT8] Using calibration data: {calib_yaml}")

    model.export(
        format="engine",   # TensorRT
        imgsz=640,
        half=False,        # INT8, not FP16
        int8=True,         # enable INT8
        dynamic=False,
        data=calib_yaml,   # calibration images
        device=0,          # GPU 0
    )

    print("[Export-INT8] Done. An INT8 engine should be created next to best.pt")


if __name__ == "__main__":
    main()
