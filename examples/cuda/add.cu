__global__ void add(const float *left, const float *right, float *result, unsigned int count) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        result[i] = left[i] + right[i];
    }
}
