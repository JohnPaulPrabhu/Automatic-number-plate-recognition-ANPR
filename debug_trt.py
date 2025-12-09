import numpy as np
import cv2
import os
import sys
import torch # Testing context conflict

# Ensure src is in path
sys.path.append(os.getcwd())

from src.inference.plate_detector_trt_cuda import PlateDetectorTrtCuda

def main():
    try:
        print("Initializing PlateDetectorTrtCuda...")
        # Note: Using absolute path to engine if possible or relative
        detector = PlateDetectorTrtCuda(
            engine_path="models/plate_fp16_trt10.engine",
            cuda_preproc_dll="cpp/cuda_preproc.dll"
        )
        print("Initialized.")
        
        # Simulate OCREngine loading (PyTorch GPU usage)
        print("Initializing PyTorch on GPU...")
        x = torch.ones((100, 100), device="cuda")
        print("PyTorch tensor created:", x.shape)
        
        # Create dummy frame with the weird resolution from a.mp4
        h, w = 720, 1114
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        print(f"Running inference on dummy frame {frame.shape}...")
        
        dets = detector.infer(frame)
        print("Inference successful. Detections:", dets)
        
    except Exception as e:
        print("Caught exception:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
