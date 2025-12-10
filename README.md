# Automatic Number Plate Recognition (ANPR)

A high-performance ANPR system optimized for NVIDIA GPUs using TensorRT and custom CUDA preprocessing. This project combines YOLO-based plate detection, object tracking, and OCR to achieve real-time license plate recognition.

## Features

-   **High-Speed Detection**: Uses YOLOv11 optimized with TensorRT (FP16/INT8 support).
-   **Hardware Acceleration**: Custom CUDA kernels for image preprocessing (resize, normalization, layout conversion).
-   **Robust Tracking**: IoU-based tracking to maintain plate identities across frames.
-   **Temporal OCR Smoothing**: Aggregates OCR results over time to improve accuracy and stability (e.g., majority voting).
-   **OCR**: Integrated with EasyOCR for text recognition.

## Requirements

-   **OS**: Windows (tested) or Linux.
-   **GPU**: NVIDIA GPU with CUDA capability.
-   **Software**:
    -   CUDA Toolkit (compatible with your TensorRT version).
    -   TensorRT 10.x.
    -   Anaconda or Miniconda.

## internal modules
-   `src.app.anpr_demo`: Main application entry point.
-   `src.inference`: different detectors (TensorRT, ONNX, PyTorch, etc.).
-   `src.cuda_preproc`: Python wrapper for the CUDA preprocessing DLL.
-   `src.ocr`: OCR engine wrapper.
-   `src.tracking`: Simple IoU tracker.
-   `cpp/`: Source code for the CUDA preprocessing DLL.

## Usage

### 1. Environment Setup

Create a Conda environment and install dependencies:

```bash
conda create -n ANPR python=3.10
conda activate ANPR
pip install -r requirements.txt
```

*Note: You may need to install `pytorch` and `tensorrt` specifically matching your CUDA version.*

### 2. Prepare Models

Ensure your TensorRT engine file is placed in `models/`.
Example: `models/plate_fp16_trt10.engine`.

To export a YOLO model to TensorRT:
```bash
python export_plate_trt_fp16.py
```

### 3. Run the Demo

Run the main ANPR application:

```bash
python -m src.app.anpr_demo
```

### 4. Benchmarking

To benchmark the inference speed of different models:

```bash
python benchmark_plate_detector.py
```

## Configuration

-   **Video Source**: Update `video_source` in `src/app/anpr_demo.py` to point to your video file or webcam index.
-   **Model Path**: Update `engine_path` in `src/app/anpr_demo.py` to match your engine filename.
-   **Detection Thresholds**: Adjust `conf_thres` and `nms_iou_thres` in the detector initialization.

## Troubleshooting

### CUDA Context Issues
If you encounter `[TRT] [E] ... CuTensor` errors or crashes when using PyTorch and TensorRT together, ensure you use the **Primary Context** method in your TensorRT wrapper, as PyTorch and PyCUDA can conflict if they manage separate contexts.

## License

[MIT](LICENSE)