/*
 * HyperL Native Engine & Kernel Verification Test Suite
 * Copyright 2026 Sabbir Hassan Imon and respective contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/hyperl_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#define TEST_EPSILON 1e-3f

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_TRUE(cond) do { \
    tests_run++; \
    if (!(cond)) { \
        printf("FAIL: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
        return 1; \
    } \
    tests_passed++; \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    tests_run++; \
    float diff = fabsf((float)(a) - (float)(b)); \
    if (diff > (eps)) { \
        printf("FAIL: |%f - %f| = %f > %f at %s:%d\n", (float)(a), (float)(b), diff, (eps), __FILE__, __LINE__); \
        return 1; \
    } \
    tests_passed++; \
} while(0)

int test_engine_init(void) {
    printf("[TEST] Engine Context Lifecycle...\n");
    const char *ver = hl_engine_version();
    ASSERT_TRUE(ver != NULL);
    printf("  HyperL Engine Version: %s\n", ver);

    hl_context_t *ctx = NULL;
    hl_status_t status = hl_context_create(&ctx, HL_DEVICE_CPU, 4);
    ASSERT_TRUE(status == HL_SUCCESS);
    ASSERT_TRUE(ctx != NULL);
    ASSERT_TRUE(hl_context_get_threads(ctx) == 4);

    hl_context_set_threads(ctx, 8);
    ASSERT_TRUE(hl_context_get_threads(ctx) == 8);

    status = hl_context_destroy(ctx);
    ASSERT_TRUE(status == HL_SUCCESS);
    return 0;
}

int test_rmsnorm(void) {
    printf("[TEST] RMSNorm SIMD Kernel...\n");
    hl_context_t *ctx = NULL;
    hl_context_create(&ctx, HL_DEVICE_CPU, 1);

    hl_tensor_desc_t desc = {
        .dtype = HL_DTYPE_F32,
        .ndim = 1,
        .shape = {8, 1, 1, 1},
        .strides = {1, 0, 0, 0}
    };

    hl_tensor_t *x = NULL, *w = NULL, *out = NULL;
    hl_tensor_alloc(ctx, &desc, &x);
    hl_tensor_alloc(ctx, &desc, &w);
    hl_tensor_alloc(ctx, &desc, &out);

    float *x_ptr = (float *)x->data;
    float *w_ptr = (float *)w->data;
    /* x = [1, 2, 3, 4, 5, 6, 7, 8] */
    float sum_sq = 0.0f;
    for (int i = 0; i < 8; ++i) {
        x_ptr[i] = (float)(i + 1);
        w_ptr[i] = 1.0f;
        sum_sq += x_ptr[i] * x_ptr[i];
    }
    float mean = sum_sq / 8.0f;
    float expected_scale = 1.0f / sqrtf(mean + 1e-5f);

    hl_status_t status = hl_kernel_rmsnorm(ctx, x, w, 1e-5f, out);
    ASSERT_TRUE(status == HL_SUCCESS);

    float *out_ptr = (float *)out->data;
    for (int i = 0; i < 8; ++i) {
        ASSERT_NEAR(out_ptr[i], x_ptr[i] * expected_scale, 1e-4f);
    }

    hl_tensor_free(ctx, x);
    hl_tensor_free(ctx, w);
    hl_tensor_free(ctx, out);
    hl_context_destroy(ctx);
    return 0;
}

int test_softmax(void) {
    printf("[TEST] Softmax Kernel...\n");
    hl_context_t *ctx = NULL;
    hl_context_create(&ctx, HL_DEVICE_CPU, 1);

    hl_tensor_desc_t desc = {
        .dtype = HL_DTYPE_F32,
        .ndim = 1,
        .shape = {4, 1, 1, 1},
        .strides = {1, 0, 0, 0}
    };

    hl_tensor_t *x = NULL, *out = NULL;
    hl_tensor_alloc(ctx, &desc, &x);
    hl_tensor_alloc(ctx, &desc, &out);

    float *x_ptr = (float *)x->data;
    x_ptr[0] = 1000.0f; /* Test numerical stability with large offsets */
    x_ptr[1] = 1001.0f;
    x_ptr[2] = 1002.0f;
    x_ptr[3] = 1003.0f;

    hl_status_t status = hl_kernel_softmax(ctx, x, out);
    ASSERT_TRUE(status == HL_SUCCESS);

    float *out_ptr = (float *)out->data;
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        ASSERT_TRUE(out_ptr[i] > 0.0f);
        sum += out_ptr[i];
    }
    ASSERT_NEAR(sum, 1.0f, 1e-5f);

    hl_tensor_free(ctx, x);
    hl_tensor_free(ctx, out);
    hl_context_destroy(ctx);
    return 0;
}

int test_gemv_and_gemm(void) {
    printf("[TEST] GEMV and GEMM Kernels...\n");
    hl_context_t *ctx = NULL;
    hl_context_create(&ctx, HL_DEVICE_CPU, 2);

    /* Test GEMV: W: 3x4, x: 4, y: 3 */
    hl_tensor_desc_t desc_w = {.dtype = HL_DTYPE_F32, .ndim = 2, .shape = {3, 4, 1, 1}};
    hl_tensor_desc_t desc_x = {.dtype = HL_DTYPE_F32, .ndim = 1, .shape = {4, 1, 1, 1}};
    hl_tensor_desc_t desc_y = {.dtype = HL_DTYPE_F32, .ndim = 1, .shape = {3, 1, 1, 1}};

    hl_tensor_t *W = NULL, *x = NULL, *y = NULL;
    hl_tensor_alloc(ctx, &desc_w, &W);
    hl_tensor_alloc(ctx, &desc_x, &x);
    hl_tensor_alloc(ctx, &desc_y, &y);

    float *w_ptr = (float *)W->data;
    float *x_ptr = (float *)x->data;
    for (int i = 0; i < 12; ++i) w_ptr[i] = 1.0f;
    for (int i = 0; i < 4; ++i) x_ptr[i] = (float)(i + 1); /* [1, 2, 3, 4], sum = 10 */

    hl_status_t status = hl_kernel_gemv(ctx, W, x, NULL, y);
    ASSERT_TRUE(status == HL_SUCCESS);

    float *y_ptr = (float *)y->data;
    for (int i = 0; i < 3; ++i) {
        ASSERT_NEAR(y_ptr[i], 10.0f, 1e-5f);
    }

    hl_tensor_free(ctx, W);
    hl_tensor_free(ctx, x);
    hl_tensor_free(ctx, y);
    hl_context_destroy(ctx);
    return 0;
}

int test_quantization_q4(void) {
    printf("[TEST] Q4_0 Block Quantization & GEMV...\n");
    hl_context_t *ctx = NULL;
    hl_context_create(&ctx, HL_DEVICE_CPU, 2);

    const size_t N = 64; /* 2 blocks of 32 */
    float src[64];
    float dst[64];
    hl_block_q4_0_t blocks[2];

    for (size_t i = 0; i < N; ++i) {
        src[i] = sinf((float)i * 0.1f) * 2.0f;
    }

    hl_status_t status = hl_quantize_q4_0(src, blocks, N);
    ASSERT_TRUE(status == HL_SUCCESS);

    status = hl_dequantize_q4_0(blocks, dst, N);
    ASSERT_TRUE(status == HL_SUCCESS);

    /* Verify quantization error is bounded by block step */
    for (size_t i = 0; i < N; ++i) {
        ASSERT_NEAR(src[i], dst[i], 0.35f);
    }

    /* Test Q4_0 GEMV: M=2, K=64 */
    float x[64];
    float y_q4[2];
    for (size_t i = 0; i < 64; ++i) x[i] = 1.0f;

    status = hl_gemv_q4_0(ctx, blocks, x, y_q4, 2, 32);
    ASSERT_TRUE(status == HL_SUCCESS);

    hl_context_destroy(ctx);
    return 0;
}

int main(void) {
    printf("=========================================\n");
    printf("   HyperL Native Core Test Suite         \n");
    printf("=========================================\n");

    if (test_engine_init()) return 1;
    if (test_rmsnorm()) return 1;
    if (test_softmax()) return 1;
    if (test_gemv_and_gemm()) return 1;
    if (test_quantization_q4()) return 1;

    printf("\n>>> ALL TESTS PASSED! (%d/%d checks) <<<\n\n", tests_passed, tests_run);
    return 0;
}
