/*
 * HyperL Native Inference Engine Core
 * Copyright 2026 Sabbir Hassan Imon and respective contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/hyperl_engine.h"
#include "kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HYPERL_VERSION_STRING "0.2.0-native-edge"

struct hl_context {
    hl_device_type_t device;
    int num_threads;
    void *scratch_arena;
    size_t scratch_size;
    size_t scratch_used;
};

const char *hl_engine_version(void) {
    return HYPERL_VERSION_STRING;
}

size_t hl_dtype_size(hl_dtype_t dtype) {
    switch (dtype) {
        case HL_DTYPE_F32: return sizeof(float);
        case HL_DTYPE_F16: return 2;
        case HL_DTYPE_BF16: return 2;
        case HL_DTYPE_INT8: return 1;
        case HL_DTYPE_INT32: return 4;
        case HL_DTYPE_Q4_0: return sizeof(hl_block_q4_0_t); /* per block of 32 */
        case HL_DTYPE_Q8_0: return sizeof(hl_block_q8_0_t);
        default: return 0;
    }
}

size_t hl_tensor_numel(const hl_tensor_desc_t *desc) {
    if (!desc || desc->ndim == 0 || desc->ndim > HL_MAX_DIMS) return 0;
    size_t n = 1;
    for (uint32_t i = 0; i < desc->ndim; ++i) {
        if (desc->shape[i] <= 0) return 0;
        n *= (size_t)desc->shape[i];
    }
    return n;
}

size_t hl_tensor_bytes(const hl_tensor_desc_t *desc) {
    if (!desc) return 0;
    size_t n = hl_tensor_numel(desc);
    if (desc->dtype == HL_DTYPE_Q4_0) {
        return (n / HL_Q4_BLOCK_SIZE) * sizeof(hl_block_q4_0_t);
    } else if (desc->dtype == HL_DTYPE_Q8_0) {
        return (n / 32) * sizeof(hl_block_q8_0_t);
    }
    return n * hl_dtype_size(desc->dtype);
}

hl_status_t hl_context_create(hl_context_t **ctx, hl_device_type_t device, int num_threads) {
    if (!ctx) return HL_ERROR_INVALID_ARGUMENT;
    hl_context_t *c = (hl_context_t *)calloc(1, sizeof(hl_context_t));
    if (!c) return HL_ERROR_OUT_OF_MEMORY;

    c->device = device;
    c->num_threads = num_threads > 0 ? num_threads : 4;
    c->scratch_size = 64 * 1024 * 1024; /* 64 MB scratch arena */
    c->scratch_arena = malloc(c->scratch_size);
    c->scratch_used = 0;

    *ctx = c;
    return HL_SUCCESS;
}

hl_status_t hl_context_destroy(hl_context_t *ctx) {
    if (!ctx) return HL_SUCCESS;
    if (ctx->scratch_arena) {
        free(ctx->scratch_arena);
    }
    free(ctx);
    return HL_SUCCESS;
}

int hl_context_get_threads(const hl_context_t *ctx) {
    return ctx ? ctx->num_threads : 1;
}

hl_status_t hl_context_set_threads(hl_context_t *ctx, int num_threads) {
    if (!ctx || num_threads <= 0) return HL_ERROR_INVALID_ARGUMENT;
    ctx->num_threads = num_threads;
    return HL_SUCCESS;
}

hl_status_t hl_tensor_alloc(hl_context_t *ctx, const hl_tensor_desc_t *desc, hl_tensor_t **out_tensor) {
    if (!desc || !out_tensor) return HL_ERROR_INVALID_ARGUMENT;
    size_t bytes = hl_tensor_bytes(desc);
    if (bytes == 0) return HL_ERROR_INVALID_ARGUMENT;

    hl_tensor_t *t = (hl_tensor_t *)calloc(1, sizeof(hl_tensor_t));
    if (!t) return HL_ERROR_OUT_OF_MEMORY;

    t->desc = *desc;
    t->device = ctx ? ctx->device : HL_DEVICE_CPU;
    t->is_view = false;

    /* Posix memalign / aligned alloc for 64-byte SIMD alignment */
#if defined(_WIN32)
    t->data = _aligned_malloc(bytes, 64);
#else
    if (posix_memalign(&t->data, 64, bytes) != 0) {
        t->data = malloc(bytes);
    }
#endif
    if (!t->data) {
        free(t);
        return HL_ERROR_OUT_OF_MEMORY;
    }

    *out_tensor = t;
    return HL_SUCCESS;
}

hl_status_t hl_tensor_wrap(const hl_tensor_desc_t *desc, void *data, hl_device_type_t device, hl_tensor_t **out_tensor) {
    if (!desc || !data || !out_tensor) return HL_ERROR_INVALID_ARGUMENT;
    hl_tensor_t *t = (hl_tensor_t *)calloc(1, sizeof(hl_tensor_t));
    if (!t) return HL_ERROR_OUT_OF_MEMORY;

    t->desc = *desc;
    t->data = data;
    t->device = device;
    t->is_view = true;

    *out_tensor = t;
    return HL_SUCCESS;
}

hl_status_t hl_tensor_free(hl_context_t *ctx, hl_tensor_t *tensor) {
    (void)ctx;
    if (!tensor) return HL_SUCCESS;
    if (!tensor->is_view && tensor->data) {
#if defined(_WIN32)
        _aligned_free(tensor->data);
#else
        free(tensor->data);
#endif
    }
    free(tensor);
    return HL_SUCCESS;
}
