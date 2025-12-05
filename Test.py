import numpy as np
import matplotlib.pyplot as plt
import math

def save_feature_map_grid(arr, output_path="feature_maps.png", max_channels=64):
    """
    arr: 3D numpy array (H, W, C) or (C, H, W) or (D, H, W)
    Saves a single montage image.
    """

    # --- Normalize dimensions ---
    if arr.ndim != 3:
        raise ValueError("Input must be a 3D array (H,W,C) or (C,H,W) or (D,H,W)")

    # Case: CHW
    if arr.shape[0] > 4 and arr.shape[-1] <= 4:
        arr = np.transpose(arr, (1, 2, 0))  # CHW → HWC

    # Case: D,H,W (volume) → treat as feature maps
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)

    # If now 3D but not HWC, assume first dimension is channels
    if arr.ndim == 3 and arr.shape[-1] > 4:
        H, W, C = arr.shape
    else:
        raise ValueError("Could not interpret array as feature map tensor.")

    # Limit channels
    C = min(C, max_channels)

    # Grid size (square)
    cols = int(math.sqrt(C))
    rows = math.ceil(C / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c] if rows > 1 else axes[c]
            ax.axis("off")

            if idx < C:
                img = arr[:, :, idx]
                ax.imshow(img, cmap="gray")
                ax.set_title(f"ch {idx}", fontsize=6)
            idx += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved montage image to: {output_path}")













import numpy as np
import imageio
import os

def save_tensor_as_images(arr, out_dir="saved_images"):
    os.makedirs(out_dir, exist_ok=True)

    print("Input shape:", arr.shape)

    # Remove batch dimension if exists (N, ...)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]

    # Case 1: HWC image (1,3,4 channels)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        img = arr
        if img.shape[-1] == 1:
            img = img[:, :, 0]  # drop channel dim
        imageio.imwrite(f"{out_dir}/image.png", normalize_to_uint8(img))
        print("Saved: image.png")
        return

    # Case 2: CHW image
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        img = np.transpose(arr, (1, 2, 0))
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        imageio.imwrite(f"{out_dir}/image.png", normalize_to_uint8(img))
        print("Saved: image.png")
        return

    # Case 3: Feature maps: (H, W, C) with C > 4
    if arr.ndim == 3 and arr.shape[-1] > 4:
        H, W, C = arr.shape
        print(f"Detected feature maps: {C} channels")

        for c in range(C):
            channel_img = arr[:, :, c]
            imageio.imwrite(
                f"{out_dir}/channel_{c:03d}.png",
                normalize_to_uint8(channel_img)
            )
        print(f"Saved {C} channel images in {out_dir}")
        return

    # Case 4: Volume data (D, H, W)
    if arr.ndim == 3:
        D, H, W = arr.shape
        print(f"Detected 3D volume: {D} slices")

        for d in range(D):
            imageio.imwrite(
                f"{out_dir}/slice_{d:03d}.png",
                normalize_to_uint8(arr[d])
            )
        print(f"Saved {D} slices in {out_dir}")
        return

    raise ValueError("Unsupported tensor shape for image saving.")


def normalize_to_uint8(img):
    img = img.astype(np.float32)
    min_val = img.min()
    max_val = img.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - min_val) / (max_val - min_val)
    return (img * 255).astype(np.uint8)
















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
