/*
 * HyperL Native Engine - Stable C ABI
 * Copyright 2026 Sabbir Hassan Imon and respective contributors.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Memory-aware, cross-platform inference runtime for small-to-mid models on consumer hardware.
 */

#ifndef HYPERL_ENGINE_H
#define HYPERL_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYPERL_STATIC)
  #define HL_API
#elif defined(_WIN32) || defined(__CYGWIN__)
  #if defined(HYPERL_BUILDING_DLL)
    #define HL_API __declspec(dllexport)
  #else
    #define HL_API __declspec(dllimport)
  #endif
#else
  #define HL_API __attribute__((visibility("default")))
#endif

#define HL_MAX_DIMS 4
#define HL_Q4_BLOCK_SIZE 32

/* Status and Error Codes */
typedef enum {
    HL_SUCCESS = 0,
    HL_ERROR_INVALID_ARGUMENT = 1,
    HL_ERROR_OUT_OF_MEMORY = 2,
    HL_ERROR_DEVICE_UNAVAILABLE = 3,
    HL_ERROR_UNSUPPORTED_DTYPE = 4,
    HL_ERROR_SHAPE_MISMATCH = 5,
    HL_ERROR_BACKEND_EXECUTION = 6,
    HL_ERROR_MODEL_PARSE = 7,
    HL_ERROR_IO = 8
} hl_status_t;

/* Tensor Data Types */
typedef enum {
    HL_DTYPE_F32 = 0,
    HL_DTYPE_F16 = 1,
    HL_DTYPE_BF16 = 2,
    HL_DTYPE_INT8 = 3,
    HL_DTYPE_INT32 = 4,
    HL_DTYPE_Q4_0 = 5,
    HL_DTYPE_Q4_K = 6,
    HL_DTYPE_Q8_0 = 7
} hl_dtype_t;

/* Compute Device Types */
typedef enum {
    HL_DEVICE_CPU = 0,
    HL_DEVICE_VULKAN = 1,
    HL_DEVICE_METAL = 2,
    HL_DEVICE_WEBGPU = 3,
    HL_DEVICE_CUDA = 4
} hl_device_type_t;

/* Tensor Descriptor */
typedef struct {
    hl_dtype_t dtype;
    uint32_t ndim;
    int64_t shape[HL_MAX_DIMS];
    int64_t strides[HL_MAX_DIMS];
} hl_tensor_desc_t;

/* Tensor Handle */
typedef struct hl_tensor {
    hl_tensor_desc_t desc;
    void *data;
    hl_device_type_t device;
    bool is_view;
    void *device_buffer_handle;
} hl_tensor_t;

/* Quantized Q4_0 Block Layout (32 nibbles = 16 bytes + 1 fp16 scale = 18 bytes) */
typedef struct {
    uint16_t scale_fp16;      /* 16-bit half precision delta */
    uint8_t qs[HL_Q4_BLOCK_SIZE / 2]; /* 16 bytes: each holding two 4-bit unsigned indices */
} hl_block_q4_0_t;

/* Quantized Q8_0 Block Layout (32 bytes + 1 fp16 scale = 34 bytes) */
typedef struct {
    uint16_t scale_fp16;      /* 16-bit half precision delta */
    int8_t qs[32];            /* 32 signed 8-bit values */
} hl_block_q8_0_t;

/* Opaque Context Handle */
typedef struct hl_context hl_context_t;

/* Engine Initialization & Context Lifecycle */
HL_API const char *hl_engine_version(void);
HL_API hl_status_t hl_context_create(hl_context_t **ctx, hl_device_type_t device, int num_threads);
HL_API hl_status_t hl_context_destroy(hl_context_t *ctx);
HL_API int hl_context_get_threads(const hl_context_t *ctx);
HL_API hl_status_t hl_context_set_threads(hl_context_t *ctx, int num_threads);

/* Memory & Tensor Management */
HL_API hl_status_t hl_tensor_alloc(hl_context_t *ctx, const hl_tensor_desc_t *desc, hl_tensor_t **out_tensor);
HL_API hl_status_t hl_tensor_wrap(const hl_tensor_desc_t *desc, void *data, hl_device_type_t device, hl_tensor_t **out_tensor);
HL_API hl_status_t hl_tensor_free(hl_context_t *ctx, hl_tensor_t *tensor);
HL_API size_t hl_dtype_size(hl_dtype_t dtype);
HL_API size_t hl_tensor_numel(const hl_tensor_desc_t *desc);
HL_API size_t hl_tensor_bytes(const hl_tensor_desc_t *desc);

/* Essential Compute Kernels */
HL_API hl_status_t hl_kernel_rmsnorm(hl_context_t *ctx, const hl_tensor_t *x, const hl_tensor_t *weight, float eps, hl_tensor_t *out);
HL_API hl_status_t hl_kernel_rope(hl_context_t *ctx, hl_tensor_t *q, hl_tensor_t *k, int pos, int head_dim, float theta_base);
HL_API hl_status_t hl_kernel_softmax(hl_context_t *ctx, const hl_tensor_t *x, hl_tensor_t *out);
HL_API hl_status_t hl_kernel_gemv(hl_context_t *ctx, const hl_tensor_t *W, const hl_tensor_t *x, const hl_tensor_t *bias, hl_tensor_t *y);
HL_API hl_status_t hl_kernel_gemm(hl_context_t *ctx, const hl_tensor_t *A, const hl_tensor_t *B, hl_tensor_t *C);
HL_API hl_status_t hl_kernel_attention(hl_context_t *ctx,
                                       const hl_tensor_t *q,
                                       const hl_tensor_t *k_cache,
                                       const hl_tensor_t *v_cache,
                                       int seq_len,
                                       int cur_pos,
                                       int num_heads,
                                       int num_kv_heads,
                                       int head_dim,
                                       hl_tensor_t *out);

/* Quantization Kernels */
HL_API hl_status_t hl_quantize_q4_0(const float *src, void *dst, size_t n_elements);
HL_API hl_status_t hl_dequantize_q4_0(const void *src, float *dst, size_t n_elements);
HL_API hl_status_t hl_gemv_q4_0(hl_context_t *ctx, const void *W_q4, const float *x, float *y, int M, int K);

#ifdef __cplusplus
}
#endif

#endif /* HYPERL_ENGINE_H */
