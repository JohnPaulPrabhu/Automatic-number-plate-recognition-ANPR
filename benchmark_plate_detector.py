import time
import cv2
import os

from src.inference.plate_detector_old import PlateDetector


def benchmark(weights_path, video_path, num_frames=300):
    detector = PlateDetector(
        weights_path=weights_path,
        device="cuda",
        conf_thres=0.4,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Bench] Could not open video: {video_path}")
        return

    frame_count = 0
    t_start = time.time()

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        _ = detector.infer(frame)
        frame_count += 1

    t_end = time.time()
    total = t_end - t_start
    fps = frame_count / total if total > 0 else 0.0

    print(f"[Bench] {weights_path}")
    print(f"        {frame_count} frames in {total:.3f}s → {fps:.2f} FPS")

    cap.release()


if __name__ == "__main__":
    video = "a.mp4"  # adjust

    # # FP32
    # benchmark("models/plate_fp32.onnx", video)

    # # FP16
    # benchmark("models/plate_fp16.onnx", video)

    # # INT8 (if exported)
    # benchmark("models/plate_int8.onnx", video)

    # FP32
    
    benchmark("model_training/runs/anpr_yolo11/yolo11n_anpr4/weights/best.pt", video)
    # benchmark("models/plate_fp32.pt", video)

    # FP16
    benchmark("models/plate_fp16_trt10.engine", video)

    # INT8 (if exported)
    benchmark("models/plate_int8_trt10.engine", video)
