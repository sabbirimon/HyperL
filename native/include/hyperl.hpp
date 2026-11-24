/*
 * HyperL Modern C++20 High-Performance API
 * Copyright 2026 Sabbir Hassan Imon and respective contributors.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HYPERL_HPP
#define HYPERL_HPP

#include "hyperl_engine.h"
#include <span>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <array>
#include <cmath>

namespace hyperl {

class EngineException : public std::runtime_error {
public:
    explicit EngineException(const std::string& msg, hl_status_t status = HL_ERROR_BACKEND_EXECUTION)
        : std::runtime_error(msg + " (Status: " + std::to_string(status) + ")"), status_(status) {}
    hl_status_t status() const noexcept { return status_; }
private:
    hl_status_t status_;
};

/* Forward declarations */
class Context;
class Tensor;

/* Smart RAII Context Handle */
class Context {
public:
    explicit Context(hl_device_type_t device = HL_DEVICE_CPU, int num_threads = 4) {
        hl_status_t st = hl_context_create(&ctx_, device, num_threads);
        if (st != HL_SUCCESS || !ctx_) {
            throw EngineException("Failed to create HyperL Context", st);
        }
    }

    ~Context() {
        if (ctx_) {
            hl_context_destroy(ctx_);
            ctx_ = nullptr;
        }
    }

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    Context(Context&& other) noexcept : ctx_(other.ctx_) {
        other.ctx_ = nullptr;
    }

    Context& operator=(Context&& other) noexcept {
        if (this != &other) {
            if (ctx_) hl_context_destroy(ctx_);
            ctx_ = other.ctx_;
            other.ctx_ = nullptr;
        }
        return *this;
    }

    hl_context_t* handle() const noexcept { return ctx_; }
    int threads() const noexcept { return hl_context_get_threads(ctx_); }
    void set_threads(int n) { hl_context_set_threads(ctx_, n); }
    static std::string version() { return hl_engine_version(); }

private:
    hl_context_t* ctx_{nullptr};
};

/* High-Performance C++20 Tensor Wrapper */
class Tensor {
public:
    /* Construct owning tensor */
    Tensor(Context& ctx, std::span<const int64_t> shape, hl_dtype_t dtype = HL_DTYPE_F32)
        : ctx_(&ctx), owned_(true) {
        init_desc(shape, dtype);
        hl_status_t st = hl_tensor_alloc(ctx_->handle(), &desc_, &raw_);
        if (st != HL_SUCCESS || !raw_) {
            throw EngineException("Failed to allocate Tensor", st);
        }
    }

    /* Wrap non-owning external buffer (e.g. mmap weights) */
    Tensor(Context& ctx, void* data, std::span<const int64_t> shape, hl_dtype_t dtype = HL_DTYPE_F32)
        : ctx_(&ctx), owned_(false) {
        init_desc(shape, dtype);
        hl_status_t st = hl_tensor_wrap(&desc_, data, HL_DEVICE_CPU, &raw_);
        if (st != HL_SUCCESS || !raw_) {
            throw EngineException("Failed to wrap external buffer in Tensor", st);
        }
    }

    ~Tensor() {
        if (raw_ && ctx_) {
            hl_tensor_free(ctx_->handle(), raw_);
            raw_ = nullptr;
        }
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept
        : ctx_(other.ctx_), desc_(other.desc_), raw_(other.raw_), owned_(other.owned_) {
        other.raw_ = nullptr;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            if (raw_ && ctx_) hl_tensor_free(ctx_->handle(), raw_);
            ctx_ = other.ctx_;
            desc_ = other.desc_;
            raw_ = other.raw_;
            owned_ = other.owned_;
            other.raw_ = nullptr;
        }
        return *this;
    }

    /* Accessors */
    hl_tensor_t* handle() const noexcept { return raw_; }
    hl_dtype_t dtype() const noexcept { return desc_.dtype; }
    uint32_t ndim() const noexcept { return desc_.ndim; }
    std::span<const int64_t> shape() const noexcept { return {desc_.shape, desc_.ndim}; }
    size_t numel() const noexcept { return hl_tensor_numel(&desc_); }
    size_t bytes() const noexcept { return hl_tensor_bytes(&desc_); }

    float* data_f32() noexcept { return static_cast<float*>(raw_->data); }
    const float* data_f32() const noexcept { return static_cast<const float*>(raw_->data); }
    std::span<float> as_span() noexcept { return {data_f32(), numel()}; }
    std::span<const float> as_span() const noexcept { return {data_f32(), numel()}; }

private:
    void init_desc(std::span<const int64_t> shape, hl_dtype_t dtype) {
        if (shape.empty() || shape.size() > HL_MAX_DIMS) {
            throw EngineException("Invalid tensor rank: " + std::to_string(shape.size()));
        }
        desc_.dtype = dtype;
        desc_.ndim = static_cast<uint32_t>(shape.size());
        for (size_t i = 0; i < HL_MAX_DIMS; ++i) {
            desc_.shape[i] = (i < shape.size()) ? shape[i] : 1;
            desc_.strides[i] = 0;
        }
    }

    Context* ctx_{nullptr};
    hl_tensor_desc_t desc_{};
    hl_tensor_t* raw_{nullptr};
    bool owned_{true};
};

/* Type-Safe C++20 Functional Operations */
namespace ops {

inline Tensor rmsnorm(Context& ctx, const Tensor& x, const Tensor* weight = nullptr, float eps = 1e-5f) {
    Tensor out(ctx, x.shape(), x.dtype());
    hl_status_t st = hl_kernel_rmsnorm(ctx.handle(), x.handle(), weight ? weight->handle() : nullptr, eps, out.handle());
    if (st != HL_SUCCESS) throw EngineException("RMSNorm failed", st);
    return out;
}

inline Tensor softmax(Context& ctx, const Tensor& x) {
    Tensor out(ctx, x.shape(), x.dtype());
    hl_status_t st = hl_kernel_softmax(ctx.handle(), x.handle(), out.handle());
    if (st != HL_SUCCESS) throw EngineException("Softmax failed", st);
    return out;
}

inline Tensor gemv(Context& ctx, const Tensor& W, const Tensor& x, const Tensor* bias = nullptr) {
    if (W.ndim() != 2) throw EngineException("GEMV requires 2D weight matrix");
    int64_t M = W.shape()[0];
    std::array<int64_t, 1> out_shape = {M};
    Tensor y(ctx, out_shape, HL_DTYPE_F32);
    hl_status_t st = hl_kernel_gemv(ctx.handle(), W.handle(), x.handle(), bias ? bias->handle() : nullptr, y.handle());
    if (st != HL_SUCCESS) throw EngineException("GEMV failed", st);
    return y;
}

inline Tensor gemm(Context& ctx, const Tensor& A, const Tensor& B) {
    if (A.ndim() != 2 || B.ndim() != 2) throw EngineException("GEMM requires 2D matrices");
    int64_t M = A.shape()[0];
    int64_t N = B.shape()[1];
    std::array<int64_t, 2> out_shape = {M, N};
    Tensor C(ctx, out_shape, HL_DTYPE_F32);
    hl_status_t st = hl_kernel_gemm(ctx.handle(), A.handle(), B.handle(), C.handle());
    if (st != HL_SUCCESS) throw EngineException("GEMM failed", st);
    return C;
}

} // namespace ops

} // namespace hyperl

#endif /* HYPERL_HPP */
