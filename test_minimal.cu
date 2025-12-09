// test_minimal.cu
__global__ void dummy(float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    x[i] = 1.0f;
}
int main() { return 0; }
