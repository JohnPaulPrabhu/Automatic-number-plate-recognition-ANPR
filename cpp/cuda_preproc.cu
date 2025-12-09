// cpp/cuda_preproc.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
                         __FILE__, __LINE__, cudaGetErrorString(err));         \
        }                                                                      \
    } while (0)

struct PreprocContext {
    int dst_w;
    int dst_h;
    size_t max_src_bytes;

    uint8_t* d_src = nullptr;  // device buffer for input BGR
    float*   d_dst = nullptr;  // device buffer for output CHW
    cudaStream_t stream = nullptr;
};

// -------------------------------------------------------------
// CUDA kernel: BGR uint8 HxWx3 -> RGB float32 CHW, normalized 0–1
// Resize with bilinear.
// -------------------------------------------------------------
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

// -------------------------------------------------------------
// C API (for Python ctypes)
// -------------------------------------------------------------
extern "C" {

// Create context: allocate GPU buffers & stream
// max_src_w, max_src_h: max possible frame size (e.g. 1920x1080)
// dst_w, dst_h: YOLO input size (e.g. 640x640)
__declspec(dllexport)
void* preproc_create(int max_src_w, int max_src_h, int dst_w, int dst_h)
{
    auto ctx = new PreprocContext();
    ctx->dst_w = dst_w;
    ctx->dst_h = dst_h;
    ctx->max_src_bytes = static_cast<size_t>(max_src_w) * max_src_h * 3;

    CUDA_CHECK(cudaMalloc(&ctx->d_src, ctx->max_src_bytes * sizeof(uint8_t)));

    size_t dst_bytes = static_cast<size_t>(3) * dst_w * dst_h * sizeof(float);
    CUDA_CHECK(cudaMalloc(&ctx->d_dst, dst_bytes));

    CUDA_CHECK(cudaStreamCreate(&ctx->stream));

    return ctx;
}

// Run preprocessing for one frame
// src_bgr: host pointer to uint8 BGR HxWx3
// src_w, src_h: actual frame size
// dst_chw: host pointer to float array of size 3*dst_h*dst_w
// returns 0 on success, <0 on failure
__declspec(dllexport)
int preproc_run(
    void* handle,
    const uint8_t* src_bgr,
    int src_w, int src_h,
    float* dst_chw)
{
    if (!handle) return -1;
    auto ctx = reinterpret_cast<PreprocContext*>(handle);

    size_t src_bytes = static_cast<size_t>(src_w) * src_h * 3 * sizeof(uint8_t);
    if (src_bytes > ctx->max_src_bytes * sizeof(uint8_t)) {
        std::fprintf(stderr, "[preproc_run] src frame too large!\n");
        return -2;
    }

    // 1. Copy frame to GPU (async)
    CUDA_CHECK(cudaMemcpyAsync(ctx->d_src, src_bgr, src_bytes,
                               cudaMemcpyHostToDevice, ctx->stream));

    // 2. Launch kernel
    dim3 block(16, 16);
    dim3 grid((ctx->dst_w + block.x - 1) / block.x,
              (ctx->dst_h + block.y - 1) / block.y);

    preprocess_bgr_to_chw_kernel<<<grid, block, 0, ctx->stream>>>(
        ctx->d_src,
        src_w, src_h,
        src_w * 3,
        ctx->d_dst,
        ctx->dst_w, ctx->dst_h
    );

    // 3. Copy result back to host as CHW float32
    size_t dst_bytes = static_cast<size_t>(3) * ctx->dst_w * ctx->dst_h * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(dst_chw, ctx->d_dst, dst_bytes,
                               cudaMemcpyDeviceToHost, ctx->stream));

    cudaStreamSynchronize(ctx->stream);
    return 0;
}

// Destroy context
__declspec(dllexport)
void preproc_destroy(void* handle)
{
    if (!handle) return;
    auto ctx = reinterpret_cast<PreprocContext*>(handle);
    if (ctx->d_src) cudaFree(ctx->d_src);
    if (ctx->d_dst) cudaFree(ctx->d_dst);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    delete ctx;
}

} // extern "C"
