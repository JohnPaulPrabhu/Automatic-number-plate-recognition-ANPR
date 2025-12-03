import cv2


def draw_tracks(frame_bgr, tracks, class_names=None):
    """
    Draw bounding boxes, scores, IDs (and class names if available).
    :param frame_bgr: input frame (numpy array)
    :param tracks: list of {"id", "bbox", "score", "class_id"}
    :param class_names: list/dict of class names indexed by class_id
    """
    out = frame_bgr.copy()

    for t in tracks:
        x1, y1, x2, y2 = map(int, t["bbox"])
        track_id = t["id"]
        score = t["score"]
        cls_id = t["class_id"]

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"ID {track_id} | {score:.2f}"
        if class_names is not None and cls_id in class_names:
            label = f"{class_names[cls_id]} {score:.2f} | ID {track_id}"

        # Label bg
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            out,
            (x1, max(0, y1 - th - baseline)),
            (x1 + tw, y1),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return out


def put_fps_text(frame_bgr, fps: float):
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame_bgr,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return frame_bgr
