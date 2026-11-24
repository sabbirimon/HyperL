/*
 * HyperL Native Kernel Implementations
 * Copyright 2026 Sabbir Hassan Imon and respective contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernels.h"
#include <stdio.h>

#if defined(_OPENMP)
  #include <omp.h>
#endif

/* RMSNorm: y_i = (x_i / sqrt(mean(x^2) + eps)) * weight_i */
hl_status_t hl_kernel_rmsnorm(hl_context_t *ctx, const hl_tensor_t *x, const hl_tensor_t *weight, float eps, hl_tensor_t *out) {
    (void)ctx;
    if (!x || !x->data || !out || !out->data) return HL_ERROR_INVALID_ARGUMENT;
    if (x->desc.dtype != HL_DTYPE_F32 || out->desc.dtype != HL_DTYPE_F32) return HL_ERROR_UNSUPPORTED_DTYPE;

    size_t n = hl_tensor_numel(&x->desc);
    const float *x_data = (const float *)x->data;
    const float *w_data = weight ? (const float *)weight->data : NULL;
    float *out_data = (float *)out->data;

    float sum_sq = 0.0f;
    size_t i = 0;

#if defined(__AVX2__)
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x_data + i);
        vsum = _mm256_fmadd_ps(vx, vx, vsum);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, vsum);
    for (int k = 0; k < 8; ++k) sum_sq += tmp[k];
#elif defined(__ARM_NEON)
    float32x4_t vsum = vdupq_n_f32(0.0f);
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(x_data + i);
        vsum = vmlaq_f32(vsum, vx, vx);
    }
    sum_sq = vaddvq_f32(vsum);
#endif

    for (; i < n; ++i) {
        sum_sq += x_data[i] * x_data[i];
    }

    float mean = sum_sq / (float)n;
    float scale = 1.0f / sqrtf(mean + eps);

    i = 0;
#if defined(__AVX2__)
    __m256 vscale = _mm256_set1_ps(scale);
    for (; i + 7 < n; i += 8) {
        __m256 vx = _mm256_loadu_ps(x_data + i);
        __m256 vy = _mm256_mul_ps(vx, vscale);
        if (w_data) {
            __m256 vw = _mm256_loadu_ps(w_data + i);
            vy = _mm256_mul_ps(vy, vw);
        }
        _mm256_storeu_ps(out_data + i, vy);
    }
#elif defined(__ARM_NEON)
    float32x4_t vscale = vdupq_n_f32(scale);
    for (; i + 3 < n; i += 4) {
        float32x4_t vx = vld1q_f32(x_data + i);
        float32x4_t vy = vmulq_f32(vx, vscale);
        if (w_data) {
            float32x4_t vw = vld1q_f32(w_data + i);
            vy = vmulq_f32(vy, vw);
        }
        vst1q_f32(out_data + i, vy);
    }
#endif

    for (; i < n; ++i) {
        out_data[i] = x_data[i] * scale * (w_data ? w_data[i] : 1.0f);
    }

    return HL_SUCCESS;
}

/* RoPE: Rotary Position Embedding */
hl_status_t hl_kernel_rope(hl_context_t *ctx, hl_tensor_t *q, hl_tensor_t *k, int pos, int head_dim, float theta_base) {
    (void)ctx;
    if (!q || !q->data) return HL_ERROR_INVALID_ARGUMENT;
    if (q->desc.dtype != HL_DTYPE_F32) return HL_ERROR_UNSUPPORTED_DTYPE;

    size_t q_numel = hl_tensor_numel(&q->desc);
    size_t num_q_heads = q_numel / head_dim;
    float *q_data = (float *)q->data;

    for (size_t h = 0; h < num_q_heads; ++h) {
        float *head_ptr = q_data + h * head_dim;
        for (int i = 0; i < head_dim / 2; ++i) {
            float theta = (float)pos / powf(theta_base, (float)(2 * i) / (float)head_dim);
            float cos_th = cosf(theta);
            float sin_th = sinf(theta);

            float q0 = head_ptr[2 * i];
            float q1 = head_ptr[2 * i + 1];
            head_ptr[2 * i]     = q0 * cos_th - q1 * sin_th;
            head_ptr[2 * i + 1] = q0 * sin_th + q1 * cos_th;
        }
    }

    if (k && k->data) {
        size_t k_numel = hl_tensor_numel(&k->desc);
        size_t num_k_heads = k_numel / head_dim;
        float *k_data = (float *)k->data;

        for (size_t h = 0; h < num_k_heads; ++h) {
            float *head_ptr = k_data + h * head_dim;
            for (int i = 0; i < head_dim / 2; ++i) {
                float theta = (float)pos / powf(theta_base, (float)(2 * i) / (float)head_dim);
                float cos_th = cosf(theta);
                float sin_th = sinf(theta);

                float k0 = head_ptr[2 * i];
                float k1 = head_ptr[2 * i + 1];
                head_ptr[2 * i]     = k0 * cos_th - k1 * sin_th;
                head_ptr[2 * i + 1] = k0 * sin_th + k1 * cos_th;
            }
        }
    }

    return HL_SUCCESS;
}

/* Numerically Stable Softmax (in-place or out) */
hl_status_t hl_kernel_softmax(hl_context_t *ctx, const hl_tensor_t *x, hl_tensor_t *out) {
    (void)ctx;
    if (!x || !x->data || !out || !out->data) return HL_ERROR_INVALID_ARGUMENT;
    if (x->desc.dtype != HL_DTYPE_F32 || out->desc.dtype != HL_DTYPE_F32) return HL_ERROR_UNSUPPORTED_DTYPE;

    size_t n = hl_tensor_numel(&x->desc);
    const float *x_data = (const float *)x->data;
    float *out_data = (float *)out->data;

    float max_val = -INFINITY;
    for (size_t i = 0; i < n; ++i) {
        if (x_data[i] > max_val) max_val = x_data[i];
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float e = expf(x_data[i] - max_val);
        out_data[i] = e;
        sum_exp += e;
    }

    float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < n; ++i) {
        out_data[i] *= inv_sum;
    }

    return HL_SUCCESS;
}

/* Float32 Matrix-Vector Multiplication: y = W * x + bias */
hl_status_t hl_kernel_gemv(hl_context_t *ctx, const hl_tensor_t *W, const hl_tensor_t *x, const hl_tensor_t *bias, hl_tensor_t *y) {
    (void)ctx;
    if (!W || !W->data || !x || !x->data || !y || !y->data) return HL_ERROR_INVALID_ARGUMENT;
    if (W->desc.ndim != 2) return HL_ERROR_INVALID_ARGUMENT;

    int64_t M = W->desc.shape[0];
    int64_t K = W->desc.shape[1];

    const float *W_data = (const float *)W->data;
    const float *x_data = (const float *)x->data;
    const float *b_data = bias ? (const float *)bias->data : NULL;
    float *y_data = (float *)y->data;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
        const float *w_row = W_data + m * K;
        float dot = 0.0f;
        int64_t k = 0;

#if defined(__AVX2__)
        __m256 vdot = _mm256_setzero_ps();
        for (; k + 7 < K; k += 8) {
            __m256 vw = _mm256_loadu_ps(w_row + k);
            __m256 vx = _mm256_loadu_ps(x_data + k);
            vdot = _mm256_fmadd_ps(vw, vx, vdot);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, vdot);
        for (int j = 0; j < 8; ++j) dot += tmp[j];
#elif defined(__ARM_NEON)
        float32x4_t vdot = vdupq_n_f32(0.0f);
        for (; k + 3 < K; k += 4) {
            float32x4_t vw = vld1q_f32(w_row + k);
            float32x4_t vx = vld1q_f32(x_data + k);
            vdot = vmlaq_f32(vdot, vw, vx);
        }
        dot = vaddvq_f32(vdot);
#endif

        for (; k < K; ++k) {
            dot += w_row[k] * x_data[k];
        }

        if (b_data) dot += b_data[m];
        y_data[m] = dot;
    }

    return HL_SUCCESS;
}

/* Float32 GEMM: C = A * B (A: M x K, B: K x N, C: M x N) */
hl_status_t hl_kernel_gemm(hl_context_t *ctx, const hl_tensor_t *A, const hl_tensor_t *B, hl_tensor_t *C) {
    (void)ctx;
    if (!A || !B || !C) return HL_ERROR_INVALID_ARGUMENT;
    if (A->desc.ndim != 2 || B->desc.ndim != 2 || C->desc.ndim != 2) return HL_ERROR_INVALID_ARGUMENT;

    int64_t M = A->desc.shape[0];
    int64_t K = A->desc.shape[1];
    int64_t N = B->desc.shape[1];

    if (B->desc.shape[0] != K || C->desc.shape[0] != M || C->desc.shape[1] != N) {
        return HL_ERROR_SHAPE_MISMATCH;
    }

    const float *a_ptr = (const float *)A->data;
    const float *b_ptr = (const float *)B->data;
    float *c_ptr = (float *)C->data;

#if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += a_ptr[m * K + k] * b_ptr[k * N + n];
            }
            c_ptr[m * N + n] = sum;
        }
    }

    return HL_SUCCESS;
}

/* Grouped-Query Attention with continuous KV cache:
 * q: [num_heads * head_dim]
 * k_cache: [max_seq_len, num_kv_heads, head_dim]
 * v_cache: [max_seq_len, num_kv_heads, head_dim]
 * out: [num_heads * head_dim]
 */
hl_status_t hl_kernel_attention(hl_context_t *ctx,
                               const hl_tensor_t *q,
                               const hl_tensor_t *k_cache,
                               const hl_tensor_t *v_cache,
                               int seq_len,
                               int cur_pos,
                               int num_heads,
                               int num_kv_heads,
                               int head_dim,
                               hl_tensor_t *out) {
    (void)ctx;
    if (!q || !k_cache || !v_cache || !out) return HL_ERROR_INVALID_ARGUMENT;
    if (cur_pos >= seq_len) return HL_ERROR_INVALID_ARGUMENT;

    const float *q_data = (const float *)q->data;
    const float *k_data = (const float *)k_cache->data;
    const float *v_data = (const float *)v_cache->data;
    float *out_data = (float *)out->data;

    float scale = 1.0f / sqrtf((float)head_dim);
    int kv_group_size = num_heads / num_kv_heads;

    /* Buffer for attention scores up to current sequence position */
    float *scores = (float *)malloc((cur_pos + 1) * sizeof(float));
    if (!scores) return HL_ERROR_OUT_OF_MEMORY;

    for (int h = 0; h < num_heads; ++h) {
        int kv_h = h / kv_group_size;
        const float *q_h = q_data + h * head_dim;

        /* Step 1: Compute attention scores S[t] = (Q_h . K_{t, kv_h}) * scale */
        float max_score = -INFINITY;
        for (int t = 0; t <= cur_pos; ++t) {
            const float *k_vec = k_data + (t * num_kv_heads + kv_h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_h[d] * k_vec[d];
            }
            dot *= scale;
            scores[t] = dot;
            if (dot > max_score) max_score = dot;
        }

        /* Step 2: Softmax over scores */
        float sum_exp = 0.0f;
        for (int t = 0; t <= cur_pos; ++t) {
            float exp_val = expf(scores[t] - max_score);
            scores[t] = exp_val;
            sum_exp += exp_val;
        }
        float inv_sum = 1.0f / sum_exp;
        for (int t = 0; t <= cur_pos; ++t) {
            scores[t] *= inv_sum;
        }

        /* Step 3: Weighted sum over V: Out_h = sum(scores[t] * V_{t, kv_h}) */
        float *out_h = out_data + h * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            out_h[d] = 0.0f;
        }
        for (int t = 0; t <= cur_pos; ++t) {
            const float *v_vec = v_data + (t * num_kv_heads + kv_h) * head_dim;
            float sc = scores[t];
            for (int d = 0; d < head_dim; ++d) {
                out_h[d] += sc * v_vec[d];
            }
        }
    }

    free(scores);
    return HL_SUCCESS;
}

/* Quantize FP32 -> Q4_0 (blocks of 32 elements) */
hl_status_t hl_quantize_q4_0(const float *src, void *dst, size_t n_elements) {
    if (!src || !dst || (n_elements % HL_Q4_BLOCK_SIZE != 0)) {
        return HL_ERROR_INVALID_ARGUMENT;
    }

    size_t nb = n_elements / HL_Q4_BLOCK_SIZE;
    hl_block_q4_0_t *blocks = (hl_block_q4_0_t *)dst;

    for (size_t b = 0; b < nb; ++b) {
        const float *x = src + b * HL_Q4_BLOCK_SIZE;
        float amax = 0.0f;
        for (int i = 0; i < HL_Q4_BLOCK_SIZE; ++i) {
            float v = fabsf(x[i]);
            if (v > amax) amax = v;
        }

        float d = amax / -8.0f;
        float id = d ? 1.0f / d : 0.0f;
        blocks[b].scale_fp16 = hl_fp32_to_fp16(d);

        for (int i = 0; i < HL_Q4_BLOCK_SIZE / 2; ++i) {
            float x0 = x[i];
            float x1 = x[i + HL_Q4_BLOCK_SIZE / 2];

            int q0 = (int)roundf(x0 * id + 8.5f);
            int q1 = (int)roundf(x1 * id + 8.5f);

            if (q0 < 0) q0 = 0; else if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; else if (q1 > 15) q1 = 15;

            blocks[b].qs[i] = (uint8_t)(q0 | (q1 << 4));
        }
    }

    return HL_SUCCESS;
}

/* Dequantize Q4_0 -> FP32 */
hl_status_t hl_dequantize_q4_0(const void *src, float *dst, size_t n_elements) {
    if (!src || !dst || (n_elements % HL_Q4_BLOCK_SIZE != 0)) {
        return HL_ERROR_INVALID_ARGUMENT;
    }

    size_t nb = n_elements / HL_Q4_BLOCK_SIZE;
    const hl_block_q4_0_t *blocks = (const hl_block_q4_0_t *)src;

    for (size_t b = 0; b < nb; ++b) {
        float d = hl_fp16_to_fp32(blocks[b].scale_fp16);
        float *y = dst + b * HL_Q4_BLOCK_SIZE;

        for (int i = 0; i < HL_Q4_BLOCK_SIZE / 2; ++i) {
            uint8_t byte = blocks[b].qs[i];
            int q0 = byte & 0x0F;
            int q1 = (byte >> 4) & 0x0F;

            y[i] = (float)(q0 - 8) * d;
            y[i + HL_Q4_BLOCK_SIZE / 2] = (float)(q1 - 8) * d;
        }
    }

    return HL_SUCCESS;
}

/* Fast GEMV with Q4_0 Weights and FP32 Vector: y = W_q4 * x */
hl_status_t hl_gemv_q4_0(hl_context_t *ctx, const void *W_q4, const float *x, float *y, int M, int K) {
    (void)ctx;
    if (!W_q4 || !x || !y || (K % HL_Q4_BLOCK_SIZE != 0)) {
        return HL_ERROR_INVALID_ARGUMENT;
    }

    int blocks_per_row = K / HL_Q4_BLOCK_SIZE;
    const hl_block_q4_0_t *blocks = (const hl_block_q4_0_t *)W_q4;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < M; ++m) {
        const hl_block_q4_0_t *row_blocks = blocks + m * blocks_per_row;
        float row_sum = 0.0f;

        for (int b = 0; b < blocks_per_row; ++b) {
            float d = hl_fp16_to_fp32(row_blocks[b].scale_fp16);
            const float *xb = x + b * HL_Q4_BLOCK_SIZE;

            float block_sum = 0.0f;
            for (int i = 0; i < HL_Q4_BLOCK_SIZE / 2; ++i) {
                uint8_t byte = row_blocks[b].qs[i];
                int q0 = (byte & 0x0F) - 8;
                int q1 = ((byte >> 4) & 0x0F) - 8;

                block_sum += (float)q0 * xb[i];
                block_sum += (float)q1 * xb[i + HL_Q4_BLOCK_SIZE / 2];
            }
            row_sum += block_sum * d;
        }

        y[m] = row_sum;
    }

    return HL_SUCCESS;
}
