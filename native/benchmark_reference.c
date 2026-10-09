/* Explicit benchmark baseline only; caller supplies already validated buffers. */
#include <stddef.h>

void hl_benchmark_weighted_relu(const float *x, const float *w, float *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float value = x[i] * w[i];
        out[i] = value > 0 ? value : 0;
    }
}
