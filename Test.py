import numpy as np
import matplotlib.pyplot as plt

def visualize_npy(path):
    arr = np.load(path)
    print("Shape:", arr.shape)
    print("Min:", arr.min(), "Max:", arr.max())
    print("Mean:", arr.mean(), "Std:", arr.std())

    # ---- Detect format ----
    if arr.ndim == 4:
        N, A, B, C = arr.shape
        
        # Case 1: NCHW -> arr[0] shape = (C,H,W)
        if C <= 4:  # Most images have 1 or 3 channels
            layout = "NCHW"
            img = np.transpose(arr[0], (1, 2, 0))   # C,H,W → H,W,C
        
        # Case 2: NHWC -> arr[0] shape = (H,W,C)
        else:
            layout = "NHWC"
            img = arr[0]

        print("Detected layout:", layout)

    elif arr.ndim == 3:
        # Could be CHW or HWC
        if arr.shape[0] <= 4:  
            layout = "CHW"
            img = np.transpose(arr, (1, 2, 0))
        else:
            layout = "HWC"
            img = arr
        print("Detected layout:", layout)

    elif arr.ndim == 2:
        layout = "HW"
        img = arr
        print("Detected layout:", layout)

    else:
        raise ValueError("Unsupported array shape for visualization")

    # ---- Visualization ----
    plt.figure(figsize=(6,6))
    if img.ndim == 2 or img.shape[2] == 1:
        plt.imshow(img.squeeze(), cmap="gray")
    else:
        plt.imshow(img)
    plt.title(f"Visualization ({layout})")
    plt.colorbar()
    plt.show()

    # ---- Histogram ----
    plt.figure(figsize=(6,4))
    plt.hist(arr.flatten(), bins=100)
    plt.title("Value Distribution Histogram")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.show()
