// SPDX-License-Identifier: Apache-2.0
// Package hyperl provides high-performance Go bindings for the HyperL native inference engine.
package hyperl

/*
#cgo CFLAGS: -I../../native/include -I../../native -O3
#cgo LDFLAGS: -L../../build -lhyperl_engine_static -lm
#include "hyperl_engine.h"
#include <stdlib.h>

static hl_status_t hl_go_rmsnorm(hl_context_t *ctx, const float *x, const float *w, int n, float eps, float *out) {
    hl_tensor_desc_t desc = { .dtype = HL_DTYPE_F32, .ndim = 1, .shape = {n, 1, 1, 1} };
    hl_tensor_t tx = { .desc = desc, .data = (void*)x, .device = HL_DEVICE_CPU };
    hl_tensor_t tout = { .desc = desc, .data = (void*)out, .device = HL_DEVICE_CPU };
    hl_tensor_t tw = { .desc = desc, .data = (void*)w, .device = HL_DEVICE_CPU };
    return hl_kernel_rmsnorm(ctx, &tx, w ? &tw : NULL, eps, &tout);
}

static hl_status_t hl_go_gemv(hl_context_t *ctx, const float *w, const float *x, int m, int k, float *y) {
    hl_tensor_desc_t descW = { .dtype = HL_DTYPE_F32, .ndim = 2, .shape = {m, k, 1, 1} };
    hl_tensor_desc_t descX = { .dtype = HL_DTYPE_F32, .ndim = 1, .shape = {k, 1, 1, 1} };
    hl_tensor_desc_t descY = { .dtype = HL_DTYPE_F32, .ndim = 1, .shape = {m, 1, 1, 1} };
    hl_tensor_t tW = { .desc = descW, .data = (void*)w, .device = HL_DEVICE_CPU };
    hl_tensor_t tx = { .desc = descX, .data = (void*)x, .device = HL_DEVICE_CPU };
    hl_tensor_t ty = { .desc = descY, .data = (void*)y, .device = HL_DEVICE_CPU };
    return hl_kernel_gemv(ctx, &tW, &tx, NULL, &ty);
}
*/
import "C"
import (
	"errors"
	"fmt"
)

// Version returns the native engine version.
func Version() string {
	return C.GoString(C.hl_engine_version())
}

// Context wraps the native execution context.
type Context struct {
	raw *C.hl_context_t
}

// NewContext creates a new execution context.
func NewContext(numThreads int) (*Context, error) {
	var raw *C.hl_context_t
	st := C.hl_context_create(&raw, C.HL_DEVICE_CPU, C.int(numThreads))
	if st != C.HL_SUCCESS || raw == nil {
		return nil, fmt.Errorf("failed to create HyperL context: code %d", int(st))
	}
	return &Context{raw: raw}, nil
}

// Close destroys the execution context.
func (c *Context) Close() {
	if c.raw != nil {
		C.hl_context_destroy(c.raw)
		c.raw = nil
	}
}

// RMSNorm executes vectorized Root Mean Square Normalization.
func (c *Context) RMSNorm(x []float32, weight []float32, eps float32) ([]float32, error) {
	if len(x) == 0 {
		return nil, errors.New("input slice cannot be empty")
	}
	n := len(x)
	out := make([]float32, n)

	var wPtr *C.float
	if len(weight) == n {
		wPtr = (*C.float)(&weight[0])
	}

	st := C.hl_go_rmsnorm(c.raw, (*C.float)(&x[0]), wPtr, C.int(n), C.float(eps), (*C.float)(&out[0]))
	if st != C.HL_SUCCESS {
		return nil, fmt.Errorf("rmsnorm failed: code %d", int(st))
	}
	return out, nil
}

// GEMV executes matrix-vector multiplication y = W * x.
func (c *Context) GEMV(W []float32, x []float32, M, K int) ([]float32, error) {
	if len(W) != M*K || len(x) != K {
		return nil, errors.New("dimension mismatch in GEMV")
	}
	y := make([]float32, M)

	st := C.hl_go_gemv(c.raw, (*C.float)(&W[0]), (*C.float)(&x[0]), C.int(M), C.int(K), (*C.float)(&y[0]))
	if st != C.HL_SUCCESS {
		return nil, fmt.Errorf("gemv failed: code %d", int(st))
	}
	return y, nil
}
