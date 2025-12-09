#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Simple CUDA check macro for debugging (optional)
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                         __FILE__, __LINE__, cudaGetErrorString(err));           \
        }                                                                        \
    } while (0)


// Kernel: BGR uint8 (H x W x 3) -> RGB float32 (3 x dst_h x dst_w), normalized 0–1
// with bilinear resize
__global__ void preprocess_bgr_to_chw_kernel(
    const uint8_t* __restrict__ src, int src_w, int src_h, int src_stride,
    float* __restrict__ dst, int dst_w, int dst_h)
{
    int dx = blockIdx.x * blockDim.x + threadIdx.x;  // output x
    int dy = blockIdx.y * blockDim.y + threadIdx.y;  // output y

    if (dx >= dst_w || dy >= dst_h) return;

    // scale factors (from dst to src)
    float scale_x = static_cast<float>(src_w) / static_cast<float>(dst_w);
    float scale_y = static_cast<float>(src_h) / static_cast<float>(dst_h);

    // map center of dst pixel to src coordinates
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

    // Helper to read BGR pixel at (x,y)
    auto load_bgr = [&](int x, int y, float& b, float& g, float& r) {
        // assuming src is tightly packed: stride = src_w * 3
        const uint8_t* p = src + y * src_stride + x * 3;
        b = static_cast<float>(p[0]);
        g = static_cast<float>(p[1]);
        r = static_cast<float>(p[2]);
    };

    float b00, g00, r00;
    float b01, g01, r01;
    float b10, g10, r10;
    float b11, g11, r11;

    load_bgr(x0, y0, b00, g00, r00);
    load_bgr(x0, y1, b01, g01, r01);
    load_bgr(x1, y0, b10, g10, r10);
    load_bgr(x1, y1, b11, g11, r11);

    // bilinear interpolate each channel
    float b = w00 * b00 + w01 * b01 + w10 * b10 + w11 * b11;
    float g = w00 * g00 + w01 * g01 + w10 * g10 + w11 * g11;
    float r = w00 * r00 + w01 * r01 + w10 * r10 + w11 * r11;

    // normalize 0–1
    const float scale = 1.0f / 255.0f;
    b *= scale;
    g *= scale;
    r *= scale;

    // write to CHW layout: [C, H, W]
    int dst_index_yx = dy * dst_w + dx;
    int plane_size   = dst_w * dst_h;

    // YOLO expects RGB → channels in order [0]=R, [1]=G, [2]=B
    dst[0 * plane_size + dst_index_yx] = r;
    dst[1 * plane_size + dst_index_yx] = g;
    dst[2 * plane_size + dst_index_yx] = b;
}


// Host helper: launch preprocess kernel
void preprocess_bgr_to_chw(
    const uint8_t* d_src, int src_w, int src_h,
    float* d_dst, int dst_w, int dst_h,
    cudaStream_t stream = 0)
{
    // src_stride in bytes (BGR, tightly packed)
    int src_stride = src_w * 3;

    dim3 block(16, 16);
    dim3 grid(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y
    );

    preprocess_bgr_to_chw_kernel<<<grid, block, 0, stream>>>(
        d_src, src_w, src_h, src_stride,
        d_dst, dst_w, dst_h
    );
}
