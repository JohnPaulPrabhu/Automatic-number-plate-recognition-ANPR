import time
import cv2

from src.inference.plate_detector import PlateDetector
from src.ocr.ocr_engine import OCREngine
from src.utils.profiling import FPSCounter


def main():
    # Use a video file for now to avoid webcam FPS cap
    # Replace with 0 for webcam if needed
    video_source = "a.mp4"  # <- put your video here
    # video_source = "sample.mp4"  # <- put your video here

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[Error] Cannot open video source: {video_source}")
        return

    # FP32 plate detector
    plate_detector = PlateDetector(
        weights_path="model_training/runs/anpr_yolo11/yolo11n_anpr4/weights/best.pt",  # TODO: replace with your plate model
        # weights_path="model_training/runs/anpr_yolo11/yolo11n_anpr4/weights/best.pt",  # TODO: replace with your plate model
        device=None,
        conf_thres=0.2,
    )
    # breakpoint()
    # OCR engine
    ocr_engine = OCREngine(languages=("en",), gpu=True)

    fps_counter = FPSCounter(window=30)

    print("[ANPR] Starting ANPR demo...")
    print("[ANPR] Press 'q' to quit.")

    frame_idx = 0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[ANPR] End of stream or cannot read frame.")
            break
        t1 = time.time()

        orig_h, orig_w, _ = frame.shape

        # 1. Plate detection on original frame
        detections = plate_detector.infer(frame)
        t2 = time.time()

        # 2. OCR for each plate on original frame
        plate_results = []  # will store (x1, y1, x2, y2, text, conf)
        for det in detections:
            x1, y1, x2, y2, score = det
            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

            # clamp
            x1_i = max(0, min(orig_w - 1, x1_i))
            x2_i = max(0, min(orig_w - 1, x2_i))
            y1_i = max(0, min(orig_h - 1, y1_i))
            y2_i = max(0, min(orig_h - 1, y2_i))

            plate_crop = frame[y1_i:y2_i, x1_i:x2_i]
            if plate_crop.size == 0:
                continue

            text, conf = ocr_engine.read_plate(plate_crop)
            plate_results.append((x1, y1, x2, y2, text, conf, score))

        # 3. Resize frame ONLY for display
        max_display_w = 1280
        max_display_h = 720

        scale = min(max_display_w / orig_w, max_display_h / orig_h, 1.0)
        if scale < 1.0:
            display_frame = cv2.resize(
                frame,
                (int(orig_w * scale), int(orig_h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            display_frame = frame.copy()

        disp_h, disp_w, _ = display_frame.shape

        # 4. Draw boxes + text on the *resized* frame
        annotated = display_frame

        # Choose font scale & thickness based on display size (not original)
        base_font_scale = 0.5
        base_thickness = 1
        # Optionally boost a bit for large displays
        font_scale = base_font_scale * max(disp_w, disp_h) / 640.0
        thickness = max(1, int(base_thickness * max(disp_w, disp_h) / 640.0))

        for (x1, y1, x2, y2, text, conf, score) in plate_results:
            # scale coords to display frame
            x1_s = int(x1 * scale)
            y1_s = int(y1 * scale)
            x2_s = int(x2 * scale)
            y2_s = int(y2 * scale)

            cv2.rectangle(annotated, (x1_s, y1_s), (x2_s, y2_s),
                          (0, 255, 0), thickness)

            if text is not None:
                label = f"{text} ({conf:.2f})"
            else:
                label = f"score={score:.2f}"

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(
                annotated,
                (x1_s, max(0, y1_s - th - baseline)),
                (x1_s + tw, y1_s),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1_s, y1_s - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        t3 = time.time()

        # 5. FPS text on resized frame
        fps = fps_counter.update()
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 1.2,
            (0, 0, 255),
            thickness + 1,
            cv2.LINE_AA,
        )

        cv2.imshow("ANPR - FP32 Baseline", annotated)
        t4 = time.time()

        # (timing prints + key handling same as before...)


        if frame_idx % 60 == 0:
            print(
                f"[Timing] capture={1000*(t1-t0):5.1f} ms, "
                f"detect={1000*(t2-t1):5.1f} ms, "
                f"ocr+draw={1000*(t3-t2):5.1f} ms, "
                f"show={1000*(t4-t3):5.1f} ms, "
                f"total={1000*(t4-t0):5.1f} ms, "
                f"FPS≈{fps:4.1f}"
            )

        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[ANPR] Stopped.")


if __name__ == "__main__":
    main()
