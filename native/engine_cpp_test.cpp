/*
 * HyperL C++20 API & SIMD Benchmark Test Suite
 * Copyright 2026 Sabbir Hassan Imon and respective contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "include/hyperl.hpp"
#include <iostream>
#include <cassert>
#include <chrono>

int main() {
    std::cout << "=========================================\n";
    std::cout << "   HyperL Modern C++20 API Test Suite    \n";
    std::cout << "=========================================\n";

    try {
        std::cout << "[TEST] Initializing C++ Context (4 threads)...\n";
        hyperl::Context ctx(HL_DEVICE_CPU, 4);
        std::cout << "  Engine Version: " << hyperl::Context::version() << "\n";
        assert(ctx.threads() == 4);

        std::cout << "[TEST] Testing C++20 RMSNorm...\n";
        std::array<int64_t, 1> shape_x = {8};
        hyperl::Tensor x(ctx, shape_x);
        for (int i = 0; i < 8; ++i) {
            x.data_f32()[i] = static_cast<float>(i + 1);
        }

        hyperl::Tensor norm_x = hyperl::ops::rmsnorm(ctx, x);
        std::cout << "  RMSNorm result: [";
        for (float v : norm_x.as_span()) {
            std::cout << v << " ";
        }
        std::cout << "]\n";
        assert(std::fabs(norm_x.data_f32()[0] - 0.1980295f) < 1e-4f);

        std::cout << "[TEST] Testing C++20 GEMV (Matrix-Vector Decode Bottleneck)...\n";
        std::array<int64_t, 2> shape_w = {4, 8};
        hyperl::Tensor W(ctx, shape_w);
        for (size_t i = 0; i < W.numel(); ++i) W.data_f32()[i] = 1.0f;

        hyperl::Tensor y = hyperl::ops::gemv(ctx, W, x);
        assert(y.shape()[0] == 4);
        std::cout << "  GEMV output vector length: " << y.shape()[0] << ", y[0]: " << y.data_f32()[0] << "\n";
        /* Sum of 1..8 is 36 */
        assert(std::fabs(y.data_f32()[0] - 36.0f) < 1e-4f);

        std::cout << "[TEST] Testing C++20 Softmax...\n";
        hyperl::Tensor prob = hyperl::ops::softmax(ctx, y);
        float p_sum = 0.0f;
        for (float p : prob.as_span()) p_sum += p;
        assert(std::fabs(p_sum - 1.0f) < 1e-4f);
        std::cout << "  Softmax sum of probabilities: " << p_sum << "\n";

        std::cout << "\n>>> ALL C++20 TESTS PASSED SUCCESSFULLY! <<<\n\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "C++20 Test Failed: " << ex.what() << "\n";
        return 1;
    }
}
