import cv2
import os


def extract_frames(video_paths, out_dir, every_n_frames=10, max_frames=500):
    os.makedirs(out_dir, exist_ok=True)
    count = 0

    for vid_idx, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Calib] Could not open {video_path}")
            continue

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % every_n_frames == 0:
                out_path = os.path.join(out_dir, f"calib_{vid_idx}_{frame_idx:06d}.jpg")
                cv2.imwrite(out_path, frame)
                count += 1
                if count >= max_frames:
                    break

            frame_idx += 1

        cap.release()
        if count >= max_frames:
            break

    print(f"[Calib] Extracted {count} frames into {out_dir}")


if __name__ == "__main__":
    # TODO: replace with your actual video paths
    videos = [
        "a.mp4",
        "sample.mp4",
    ]
    extract_frames(videos, out_dir="datasets/anpr_calib/images", every_n_frames=10, max_frames=500)
