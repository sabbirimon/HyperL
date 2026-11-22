/*
 * HyperL Native Kernel Header
 * Copyright 2026 Sabbir Hassan Imon and respective contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HYPERL_KERNELS_H
#define HYPERL_KERNELS_H

#include "hyperl_engine.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#if defined(__x86_64__) || defined(_M_X64)
  #if defined(__AVX2__)
    #include <immintrin.h>
  #endif
#elif defined(__ARM_NEON) || defined(__aarch64__)
  #include <arm_neon.h>
#endif

/* IEEE-754 FP16 <-> FP32 Fast Bit Conversions */
static inline float hl_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h & 0x7C00) >> 10;
    uint32_t frac = (h & 0x03FF);
    uint32_t res;
    if (exp == 0) {
        if (frac == 0) {
            res = sign;
        } else {
            /* Denormalized */
            exp = 1;
            while ((frac & 0x0400) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x03FF;
            res = sign | ((exp + 127 - 15) << 23) | (frac << 13);
        }
    } else if (exp == 0x1F) {
        /* Inf / NaN */
        res = sign | 0x7F800000 | (frac << 13);
    } else {
        res = sign | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float out;
    memcpy(&out, &res, sizeof(float));
    return out;
}

static inline uint16_t hl_fp32_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = x & 0x007FFFFF;
    if (exp <= 0) {
        return (uint16_t)sign;
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | (exp << 10) | (frac >> 13));
}

#endif /* HYPERL_KERNELS_H */
