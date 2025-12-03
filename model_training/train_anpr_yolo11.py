import os
from ultralytics import YOLO
import torch


def main():
    # ----------------------------
    # 1. Paths and basic config
    # ----------------------------
    # Adjust this to your actual dataset location
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_yaml = os.path.join(project_root, "../License_Plate_Recognition", "data.yaml")
    # data_yaml = os.path.join(project_root, "License_Plate_Recognition", "anpr", "data.yaml")

    # Choose a base model:
    # "yolo11n.pt"  - nano  (fastest, least accurate)
    # "yolo11s.pt"  - small
    # "yolo11m.pt"  - medium
    # For plates, n or s is usually enough.
    base_model = "yolo11n.pt"

    # Where to save training runs
    save_dir = os.path.join(project_root, "runs", "anpr_yolo11")

    # ----------------------------
    # 2. Load model
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Using device: {device}")
    breakpoint() 

    model = YOLO(base_model)

    # ----------------------------
    # 3. Training hyperparameters
    # ----------------------------
    # You can tune these later
    epochs = 150
    batch_size = 32  # tune based on GPU memory
    img_size = 640   # input resolution

    # ----------------------------
    # 4. Start training
    # ----------------------------
    print("[Train] Starting YOLO11 training for license plate detection...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        patience=15,
        imgsz=img_size,
        batch=batch_size,
        device=0 if device == "cuda" else "cpu",
        project=save_dir,
        name="yolo11n_anpr",
        workers=6,             # adjust if CPU is weaker
        optimizer="AdamW",       # or "SGD"
        lr0=0.01,              # initial LR
        lrf=0.01,              # final LR fraction
        weight_decay=0.0005,
        mosaic=1.0,            # you can reduce later if plates are tiny
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,            # horizontal flips often okay for plates
        verbose=True,
    )

    print("[Train] Training complete.")
    print("[Train] Best weights should be saved at:")
    print(f"        {os.path.join(save_dir, 'yolo11n_anpr', 'weights', 'best.pt')}")


if __name__ == "__main__":
    main()
