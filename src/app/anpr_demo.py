import time
from collections import defaultdict, Counter

import cv2

from src.inference.plate_detector import PlateDetector
from src.ocr.ocr_engine import OCREngine
from src.tracking.simple_tracker import SimpleTracker
from src.utils.profiling import FPSCounter
from src.utils.plate_postprocess import clean_and_validate_plate


def main():
    # 0 = webcam, or path to your ANPR video
    video_source = "a.mp4"  # <- change as needed

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[Error] Cannot open video source: {video_source}")
        return

    # Plate detector (FP32, your trained model)
    plate_detector = PlateDetector(
        weights_path="model_training/runs/anpr_yolo11/yolo11n_anpr4/weights/best.pt",
        device=None,
        conf_thres=0.4,
    )

    # OCR engine
    ocr_engine = OCREngine(languages=("en",), gpu=True)

    # Tracker (IoU-based) for plates
    tracker = SimpleTracker(iou_thres=0.3, max_age=15)

    fps_counter = FPSCounter(window=30)

    # Track OCR state per plate ID
    # track_id -> dict with OCR aggregation state
    track_states = defaultdict(
        lambda: {
            "history": [],     # list of (clean_text, conf)
            "best_text": None,
            "best_conf": 0.0,
            "stable_text": None,
            "is_valid": False,
        }
    )

    # Hyperparameters for temporal smoothing
    min_conf_for_vote = 0.6
    min_votes_for_stable = 3

    cv2.namedWindow("ANPR - Phase 2 (Tracking + Smoothing)", cv2.WINDOW_NORMAL)

    print("[ANPR] Starting ANPR Phase 2 demo (tracking + temporal OCR)...")
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

        # 1. Plate detection (FP32)
        dets = plate_detector.infer(frame)
        # Convert to tracker format: [x1, y1, x2, y2, score, class_id]
        tracker_inputs = []
        for x1, y1, x2, y2, score in dets:
            tracker_inputs.append([x1, y1, x2, y2, score, 0])  # class_id=0 (plate)

        # 2. Update tracker
        tracks = tracker.update(tracker_inputs)
        t2 = time.time()

        # 3. For each active track: crop current bbox, run OCR, update history
        active_ids = set()

        for t in tracks:
            track_id = t["id"]
            x1, y1, x2, y2 = t["bbox"]
            active_ids.add(track_id)

            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])

            # clamp to frame
            x1_i = max(0, min(orig_w - 1, x1_i))
            x2_i = max(0, min(orig_w - 1, x2_i))
            y1_i = max(0, min(orig_h - 1, y1_i))
            y2_i = max(0, min(orig_h - 1, y2_i))

            plate_crop = frame[y1_i:y2_i, x1_i:x2_i]
            if plate_crop.size == 0:
                continue

            raw_text, conf = ocr_engine.read_plate(plate_crop)

            if raw_text is None:
                continue

            clean_text, is_valid = clean_and_validate_plate(raw_text, country="IN")
            if not clean_text:
                continue

            state = track_states[track_id]
            state["history"].append((clean_text, conf))

            # Update best text
            if conf > state["best_conf"]:
                state["best_conf"] = conf
                state["best_text"] = clean_text
                state["is_valid"] = is_valid

            # Majority vote over history above a confidence threshold
            votes = Counter(
                txt for (txt, c) in state["history"] if c >= min_conf_for_vote
            )
            if votes:
                candidate, count = votes.most_common(1)[0]
                if count >= min_votes_for_stable:
                    state["stable_text"] = candidate
                    # If any candidate that wins is valid, mark valid
                    _, valid_candidate = clean_and_validate_plate(candidate, "IN")
                    state["is_valid"] = valid_candidate

        # Optional: prune track_states for tracks that disappeared long ago
        # (simple version: keep only currently active IDs)
        for tid in list(track_states.keys()):
            if tid not in active_ids:
                # You can keep them for logging instead of deleting
                # del track_states[tid]
                pass

        # 4. Resize frame for display
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
        annotated = display_frame

        # Font scaling based on display size
        base_font_scale = 0.5
        base_thickness = 1
        font_scale = base_font_scale * max(disp_w, disp_h) / 640.0
        thickness = max(1, int(base_thickness * max(disp_w, disp_h) / 640.0))

        # 5. Draw tracks + stable plate text
        for t in tracks:
            track_id = t["id"]
            x1, y1, x2, y2 = t["bbox"]

            x1_s = int(x1 * scale)
            y1_s = int(y1 * scale)
            x2_s = int(x2 * scale)
            y2_s = int(y2 * scale)

            state = track_states.get(track_id, None)
            if state is not None:
                display_text = state["stable_text"] or state["best_text"]
                display_conf = state["best_conf"]
                is_valid = state["is_valid"]
            else:
                display_text = None
                display_conf = 0.0
                is_valid = False

            # Color: green if valid, yellow otherwise
            color = (0, 255, 0) if is_valid else (0, 255, 255)

            cv2.rectangle(annotated, (x1_s, y1_s), (x2_s, y2_s), color, thickness)

            if display_text is not None:
                label = f"ID {track_id} | {display_text} ({display_conf:.2f})"
            else:
                label = f"ID {track_id}"

            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            cv2.rectangle(
                annotated,
                (x1_s, max(0, y1_s - th - baseline)),
                (x1_s + tw, y1_s),
                color,
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

        # 6. FPS
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

        cv2.imshow("ANPR - Phase 2 (Tracking + Smoothing)", annotated)
        t4 = time.time()

        if frame_idx % 60 == 0:
            print(
                f"[Timing] capture={1000*(t1-t0):5.1f} ms, "
                f"det+track={1000*(t2-t1):5.1f} ms, "
                f"ocr+agg+draw={1000*(t3-t2):5.1f} ms, "
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
