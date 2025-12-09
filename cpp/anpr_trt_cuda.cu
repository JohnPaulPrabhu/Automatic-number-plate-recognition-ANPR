#include <NvInfer.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace nvinfer1;

// --------------------
// Simple TensorRT logger
// --------------------
class TrtLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::fprintf(stderr, "[TRT][%d] %s\n", int(severity), msg);
        }
    }
};

static TrtLogger gLogger;

// --------------------
// CUDA helpers
// --------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, cudaGetErrorString(err));         \
        }                                                                      \
    } while (0)


// --------------------
// CUDA kernel: preprocess
// Input:  BGR uint8, H x W x 3 (tightly packed)
// Output: CHW float32, RGB, normalized 0–1, dst_w x dst_h
// --------------------
__global__ void preprocess_bgr_to_chw_kernel(
    const uint8_t* __restrict__ src,
    int src_w, int src_h, int src_stride,
    float* __restrict__ dst,
    int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;  // output x
    int dy = blockIdx.y * blockDim.y + threadIdx.y;  // output y
    if (dx >= dst_w || dy >= dst_h) return;

    float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

    float sx = (dx + 0.5f) * scale_x - 0.5f;
    float sy = (dy + 0.5f) * scale_y - 0.5f;

    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    x0 = max(0, min(x0, src_w - 1));
    y0 = max(0, min(y0, src_h - 1));
    x1 = max(0, min(x1, src_w - 1));
    y1 = max(0, min(y1, src_h - 1));

    float fx = sx - x0;
    float fy = sy - y0;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w10 = fx * (1.0f - fy);
    float w11 = fx * fy;

    float b00, g00, r00;
    float b01, g01, r01;
    float b10, g10, r10;
    float b11, g11, r11;

    // Manually load each pixel (no lambda)
    const uint8_t* p00 = src + y0 * src_stride + x0 * 3;
    const uint8_t* p01 = src + y1 * src_stride + x0 * 3;
    const uint8_t* p10 = src + y0 * src_stride + x1 * 3;
    const uint8_t* p11 = src + y1 * src_stride + x1 * 3;

    b00 = static_cast<float>(p00[0]);
    g00 = static_cast<float>(p00[1]);
    r00 = static_cast<float>(p00[2]);

    b01 = static_cast<float>(p01[0]);
    g01 = static_cast<float>(p01[1]);
    r01 = static_cast<float>(p01[2]);

    b10 = static_cast<float>(p10[0]);
    g10 = static_cast<float>(p10[1]);
    r10 = static_cast<float>(p10[2]);

    b11 = static_cast<float>(p11[0]);
    g11 = static_cast<float>(p11[1]);
    r11 = static_cast<float>(p11[2]);

    float b = w00 * b00 + w01 * b01 + w10 * b10 + w11 * b11;
    float g = w00 * g00 + w01 * g01 + w10 * g10 + w11 * g11;
    float r = w00 * r00 + w01 * r01 + w10 * r10 + w11 * r11;

    const float inv255 = 1.0f / 255.0f;
    b *= inv255;
    g *= inv255;
    r *= inv255;

    int dst_index_yx = dy * dst_w + dx;
    int plane_size = dst_w * dst_h;

    // RGB, CHW
    dst[0 * plane_size + dst_index_yx] = r;
    dst[1 * plane_size + dst_index_yx] = g;
    dst[2 * plane_size + dst_index_yx] = b;
}

// __global__ void preprocess_bgr_to_chw_kernel(
//     const uint8_t* __restrict__ src,
//     int src_w, int src_h, int src_stride,
//     float* __restrict__ dst,
//     int dst_w, int dst_h)
// {
//     int dx = blockIdx.x * blockDim.x + threadIdx.x;  // output x
//     int dy = blockIdx.y * blockDim.y + threadIdx.y;  // output y
//     if (dx >= dst_w || dy >= dst_h) return;

//     float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
//     float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

//     float sx = (dx + 0.5f) * scale_x - 0.5f;
//     float sy = (dy + 0.5f) * scale_y - 0.5f;

//     int x0 = static_cast<int>(floorf(sx));
//     int y0 = static_cast<int>(floorf(sy));
//     int x1 = x0 + 1;
//     int y1 = y0 + 1;

//     x0 = max(0, min(x0, src_w - 1));
//     y0 = max(0, min(y0, src_h - 1));
//     x1 = max(0, min(x1, src_w - 1));
//     y1 = max(0, min(y1, src_h - 1));

//     float fx = sx - x0;
//     float fy = sy - y0;

//     float w00 = (1.0f - fx) * (1.0f - fy);
//     float w01 = (1.0f - fx) * fy;
//     float w10 = fx * (1.0f - fy);
//     float w11 = fx * fy;

//     auto load_bgr = [&](int x, int y, float& b, float& g, float& r) {
//         const uint8_t* p = src + y * src_stride + x * 3;
//         b = static_cast<float>(p[0]);
//         g = static_cast<float>(p[1]);
//         r = static_cast<float>(p[2]);
//     };

//     float b00, g00, r00;
//     float b01, g01, r01;
//     float b10, g10, r10;
//     float b11, g11, r11;

//     load_bgr(x0, y0, b00, g00, r00);
//     load_bgr(x0, y1, b01, g01, r01);
//     load_bgr(x1, y0, b10, g10, r10);
//     load_bgr(x1, y1, b11, g11, r11);

//     float b = w00 * b00 + w01 * b01 + w10 * b10 + w11 * b11;
//     float g = w00 * g00 + w01 * g01 + w10 * g10 + w11 * g11;
//     float r = w00 * r00 + w01 * r01 + w10 * r10 + w11 * r11;

//     const float inv255 = 1.0f / 255.0f;
//     b *= inv255;
//     g *= inv255;
//     r *= inv255;

//     int dst_index_yx = dy * dst_w + dx;
//     int plane_size = dst_w * dst_h;

//     // RGB, CHW
//     dst[0 * plane_size + dst_index_yx] = r;
//     dst[1 * plane_size + dst_index_yx] = g;
//     dst[2 * plane_size + dst_index_yx] = b;
// }

static void launch_preprocess(
    const uint8_t* d_src,
    int src_w, int src_h,
    float* d_dst,
    int dst_w, int dst_h,
    cudaStream_t stream)
{
    int src_stride = src_w * 3;
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x,
              (dst_h + block.y - 1) / block.y);
    preprocess_bgr_to_chw_kernel<<<grid, block, 0, stream>>>(
        d_src, src_w, src_h, src_stride,
        d_dst, dst_w, dst_h
    );
}


// --------------------
// ANPR Engine wrapper
// --------------------
struct AnprEngine {
    std::unique_ptr<IRuntime> runtime;
    std::shared_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;

    // int input_index;
    // int output_index;
    int input_w = 640;
    int input_h = 640;
    int output_num_vals = 0; // 5 * 8400 = 42000

    // Device buffers
    uint8_t* d_src = nullptr;
    float* d_input = nullptr;
    float* d_output = nullptr;

    // Host buffer for output
    std::vector<float> h_output;

    cudaStream_t stream = nullptr;

    int max_detections = 200;

    ~AnprEngine() {
        if (d_src) cudaFree(d_src);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (stream) cudaStreamDestroy(stream);
    }
};


// Read engine file
static std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "Failed to open engine file: %s\n", path.c_str());
        return {};
    }
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(size);
    f.read(buf.data(), size);
    return buf;
}


// --------------------
// Simple YOLO decode + NMS (CPU)
// Assumes output: (1, 5, N) = [cx, cy, w, h, conf]
// Returns list of [x1, y1, x2, y2, conf] in out_boxes (host)
// --------------------
static int decode_yolo_output(
    const float* out, int N,
    int img_w, int img_h,
    float conf_thres,
    float* out_boxes, int max_detections)
{
    struct Box { float x1,y1,x2,y2,score; };
    std::vector<Box> boxes;
    boxes.reserve(N);

    const float* cx = out + 0 * N;
    const float* cy = out + 1 * N;
    const float* w  = out + 2 * N;
    const float* h  = out + 3 * N;
    const float* cf = out + 4 * N;

    for (int i = 0; i < N; ++i) {
        float score = cf[i];
        if (score < conf_thres) continue;

        float x1 = cx[i] - w[i] / 2.0f;
        float y1 = cy[i] - h[i] / 2.0f;
        float x2 = cx[i] + w[i] / 2.0f;
        float y2 = cy[i] + h[i] / 2.0f;

        // scale from 640x640 to image w/h
        float scale_x = static_cast<float>(img_w) / 640.0f;
        float scale_y = static_cast<float>(img_h) / 640.0f;

        x1 *= scale_x;
        x2 *= scale_x;
        y1 *= scale_y;
        y2 *= scale_y;

        boxes.push_back({x1,y1,x2,y2,score});
    }

    // NMS (greedy IoU)
    auto iou = [](const Box& a, const Box& b){
        float xx1 = std::max(a.x1, b.x1);
        float yy1 = std::max(a.y1, b.y1);
        float xx2 = std::min(a.x2, b.x2);
        float yy2 = std::min(a.y2, b.y2);
        float w = std::max(0.0f, xx2 - xx1);
        float h = std::max(0.0f, yy2 - yy1);
        float inter = w * h;
        float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
        float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
        return inter / (areaA + areaB - inter + 1e-6f);
    };

    std::sort(boxes.begin(), boxes.end(),
              [](const Box& a, const Box& b){ return a.score > b.score; });

    std::vector<Box> kept;
    float iou_thres = 0.5f;

    for (const auto& b : boxes) {
        bool keep = true;
        for (const auto& k : kept) {
            if (iou(b, k) > iou_thres) {
                keep = false;
                break;
            }
        }
        if (keep) {
            kept.push_back(b);
            if ((int)kept.size() >= max_detections) break;
        }
    }

    int n = static_cast<int>(kept.size());
    for (int i = 0; i < n; ++i) {
        out_boxes[5*i + 0] = kept[i].x1;
        out_boxes[5*i + 1] = kept[i].y1;
        out_boxes[5*i + 2] = kept[i].x2;
        out_boxes[5*i + 3] = kept[i].y2;
        out_boxes[5*i + 4] = kept[i].score;
    }
    return n;
}


// --------------------
// C ABI for Python
// --------------------
extern "C" {

// Create engine handle
// engine_path: path to .engine
// input_w, input_h: usually 640,640
// max_detections: capacity of output arrays
__declspec(dllexport)
void* anpr_create(const char* engine_path, int input_w, int input_h, int max_detections)
{
    auto engine = new AnprEngine();
    engine->input_w = input_w;
    engine->input_h = input_h;
    engine->max_detections = max_detections;

    // 1. Load engine file
    std::vector<char> data = read_file(engine_path);
    if (data.empty()) {
        delete engine;
        return nullptr;
    }

    // 2. Create runtime + engine
    engine->runtime.reset(createInferRuntime(gLogger));
    if (!engine->runtime) {
        delete engine;
        return nullptr;
    }

    engine->engine.reset(engine->runtime->deserializeCudaEngine(data.data(), data.size()));
    if (!engine->engine) {
        delete engine;
        return nullptr;
    }

    engine->context.reset(engine->engine->createExecutionContext());
    if (!engine->context) {
        delete engine;
        return nullptr;
    }

    // engine->input_index  = engine->engine->getBindingIndex("images");  // typical Ultralytics name
    // engine->output_index = engine->engine->getBindingIndex("output0"); // might differ, check with Netron

    // if (engine->input_index < 0 || engine->output_index < 0) {
    //     std::fprintf(stderr, "Failed to find bindings 'images' or 'output0'\n");
    //     delete engine;
    //     return nullptr;
    // }

    // // 3. Figure out output size
    // Dims out_dims = engine->engine->getBindingDimensions(engine->output_index);
    // // Expect (1, 5, N)
    // int out0 = out_dims.d[0];
    // int out1 = out_dims.d[1];
    // int out2 = out_dims.d[2];
    // engine->output_num_vals = out0 * out1 * out2;  // should be 1*5*8400 = 42000


    // 3. Figure out output size using name-based API
    // NOTE: Make sure these names match your engine.
    // Ultralytics YOLO engines usually have "images" as input and "output0" as output.
    // const char* inputName  = "images";
    const char* outputName = "output0";

    // Get output tensor shape: typically (1, 5, 8400) for your ANPR YOLO
    nvinfer1::Dims out_dims = engine->engine->getTensorShape(outputName);
    if (out_dims.nbDims != 3) {
        std::fprintf(stderr, "Unexpected output dims nbDims=%d\n", out_dims.nbDims);
        delete engine;
        return nullptr;
    }

    int64_t out0 = out_dims.d[0];
    int64_t out1 = out_dims.d[1];
    int64_t out2 = out_dims.d[2];
    engine->output_num_vals = static_cast<int>(out0 * out1 * out2);  // 1*5*8400 = 42000


    // 4. Allocate device buffers & host output
    size_t input_bytes  = 1LL * 3 * input_h * input_w * sizeof(float);
    size_t output_bytes = 1LL * engine->output_num_vals * sizeof(float);

    size_t src_bytes = 1LL * input_h * input_w * 3 * sizeof(uint8_t); // enough for resized image (safe upper bound)

    CUDA_CHECK(cudaMalloc(&engine->d_src, src_bytes));
    CUDA_CHECK(cudaMalloc(&engine->d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&engine->d_output, output_bytes));

    engine->h_output.resize(engine->output_num_vals);

    CUDA_CHECK(cudaStreamCreate(&engine->stream));

    return engine;
}


// Run inference
// handle: from anpr_create
// bgr_data: host pointer to uint8 BGR image, shape HxW x3
// src_w, src_h: original width/height
// conf_thres: confidence threshold
// out_boxes: host float array of shape [max_detections, 5]
// returns: number of detections
__declspec(dllexport)
int anpr_infer(
    void* handle,
    const uint8_t* bgr_data,
    int src_w, int src_h,
    float conf_thres,
    float* out_boxes,
    int max_detections)
{
    if (!handle) return -1;
    AnprEngine* engine = reinterpret_cast<AnprEngine*>(handle);

    // 1. Copy source image to device (note: here we assume src_w*src_h*3 fits allocated buffer)
    size_t src_bytes = static_cast<size_t>(src_w) * src_h * 3 * sizeof(uint8_t);
    CUDA_CHECK(cudaMemcpyAsync(engine->d_src, bgr_data, src_bytes, cudaMemcpyHostToDevice, engine->stream));

    // // 2. Launch preprocess kernel -> d_input
    // launch_preprocess(
    //     engine->d_src, src_w, src_h,
    //     engine->d_input,
    //     engine->input_w, engine->input_h,
    //     engine->stream
    // );

    // // 3. Run TensorRT inference
    // void* bindings[2];
    // bindings[engine->input_index]  = engine->d_input;
    // bindings[engine->output_index] = engine->d_output;

    // if (!engine->context->enqueueV2(bindings, engine->stream, nullptr)) {
    //     std::fprintf(stderr, "enqueueV2 failed\n");
    //     return -1;
    // }


    // 2. Launch preprocess kernel -> d_input
    launch_preprocess(
        engine->d_src, src_w, src_h,
        engine->d_input,
        engine->input_w, engine->input_h,
        engine->stream
    );

    // 3. Set tensor addresses and run TensorRT inference (enqueueV3)
    const char* inputName  = "images";
    const char* outputName = "output0";

    // Tell the context where input/output buffers live on GPU
    if (!engine->context->setTensorAddress(inputName,  engine->d_input)) {
        std::fprintf(stderr, "setTensorAddress for input failed\n");
        return -1;
    }
    if (!engine->context->setTensorAddress(outputName, engine->d_output)) {
        std::fprintf(stderr, "setTensorAddress for output failed\n");
        return -1;
    }

    // Enqueue inference
    if (!engine->context->enqueueV3(engine->stream)) {
        std::fprintf(stderr, "enqueueV3 failed\n");
        return -1;
    }


    // 4. Copy output back
    size_t output_bytes = static_cast<size_t>(engine->output_num_vals) * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(engine->h_output.data(), engine->d_output, output_bytes,
                               cudaMemcpyDeviceToHost, engine->stream));

    cudaStreamSynchronize(engine->stream);

    // 5. Decode on CPU
    int N = engine->output_num_vals / 5;
    int num = decode_yolo_output(
        engine->h_output.data(), N,
        src_w, src_h,
        conf_thres,
        out_boxes,
        std::min(max_detections, engine->max_detections)
    );

    return num;
}


// Destroy engine
__declspec(dllexport)
void anpr_destroy(void* handle)
{
    if (!handle) return;
    AnprEngine* engine = reinterpret_cast<AnprEngine*>(handle);
    delete engine;
}

} // extern "C"

// nvcc -std=c++17 -I "C:/TensorRT-10.14.1.48/include" -L "C:/TensorRT-10.14.1.48/lib"  -lnvinfer -shared -Xcompiler "/MD" -o anpr_trt_cuda.dll anpr_trt_cuda.cu
