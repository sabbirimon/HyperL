# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sabbir Hassan Imon and respective contributors.
"""HyperL Native Core Python Bindings via Stable C ABI."""

import ctypes as c
import os
import sys
from pathlib import Path
import numpy as np

# Find dynamic library
_LIB_NAMES = ["libhyperl_engine.dylib", "libhyperl_engine.so", "hyperl_engine.dll"]
_SEARCH_PATHS = [
    Path(__file__).parent.parent.parent / "build",
    Path(__file__).parent.parent.parent / "build" / "Release",
    Path(__file__).parent,
    Path("/usr/local/lib"),
    Path("/opt/homebrew/lib"),
]

_lib_path = None
for search_dir in _SEARCH_PATHS:
    for name in _LIB_NAMES:
        candidate = search_dir / name
        if candidate.is_file():
            _lib_path = str(candidate.resolve())
            break
    if _lib_path:
        break

if not _lib_path:
    # Fallback to system loader
    _lib_path = "libhyperl_engine.dylib" if sys.platform == "darwin" else ("hyperl_engine.dll" if sys.platform == "win32" else "libhyperl_engine.so")

try:
    _C = c.CDLL(_lib_path)
except Exception as exc:
    _C = None
    _LOAD_ERROR = str(exc)
else:
    _LOAD_ERROR = None


# Dtypes and enums
class DType:
    F32 = 0
    F16 = 1
    BF16 = 2
    INT8 = 3
    INT32 = 4
    Q4_0 = 5
    Q4_K = 6
    Q8_0 = 7


class Device:
    CPU = 0
    VULKAN = 1
    METAL = 2
    WEBGPU = 3
    CUDA = 4


class HL_TensorDesc(c.Structure):
    _fields_ = [
        ("dtype", c.c_int),
        ("ndim", c.c_uint32),
        ("shape", c.c_int64 * 4),
        ("strides", c.c_int64 * 4),
    ]


class HL_Tensor(c.Structure):
    _fields_ = [
        ("desc", HL_TensorDesc),
        ("data", c.c_void_p),
        ("device", c.c_int),
        ("is_view", c.c_bool),
        ("device_buffer_handle", c.c_void_p),
    ]


if _C:
    _C.hl_engine_version.restype = c.c_char_p
    _C.hl_engine_version.argtypes = []

    _C.hl_context_create.restype = c.c_int
    _C.hl_context_create.argtypes = [c.POINTER(c.c_void_p), c.c_int, c.c_int]

    _C.hl_context_destroy.restype = c.c_int
    _C.hl_context_destroy.argtypes = [c.c_void_p]

    _C.hl_tensor_alloc.restype = c.c_int
    _C.hl_tensor_alloc.argtypes = [c.c_void_p, c.POINTER(HL_TensorDesc), c.POINTER(c.POINTER(HL_Tensor))]

    _C.hl_tensor_free.restype = c.c_int
    _C.hl_tensor_free.argtypes = [c.c_void_p, c.POINTER(HL_Tensor)]

    _C.hl_kernel_rmsnorm.restype = c.c_int
    _C.hl_kernel_rmsnorm.argtypes = [c.c_void_p, c.POINTER(HL_Tensor), c.POINTER(HL_Tensor), c.c_float, c.POINTER(HL_Tensor)]

    _C.hl_kernel_rope.restype = c.c_int
    _C.hl_kernel_rope.argtypes = [c.c_void_p, c.POINTER(HL_Tensor), c.POINTER(HL_Tensor), c.c_int, c.c_int, c.c_float]

    _C.hl_kernel_softmax.restype = c.c_int
    _C.hl_kernel_softmax.argtypes = [c.c_void_p, c.POINTER(HL_Tensor), c.POINTER(HL_Tensor)]

    _C.hl_kernel_gemv.restype = c.c_int
    _C.hl_kernel_gemv.argtypes = [c.c_void_p, c.POINTER(HL_Tensor), c.POINTER(HL_Tensor), c.POINTER(HL_Tensor), c.POINTER(HL_Tensor)]

    _C.hl_kernel_gemm.restype = c.c_int
    _C.hl_kernel_gemm.argtypes = [c.c_void_p, c.POINTER(HL_Tensor), c.POINTER(HL_Tensor), c.POINTER(HL_Tensor)]

    _C.hl_quantize_q4_0.restype = c.c_int
    _C.hl_quantize_q4_0.argtypes = [c.POINTER(c.c_float), c.c_void_p, c.c_size_t]

    _C.hl_dequantize_q4_0.restype = c.c_int
    _C.hl_dequantize_q4_0.argtypes = [c.c_void_p, c.POINTER(c.c_float), c.c_size_t]


class Context:
    def __init__(self, device=Device.CPU, num_threads=4):
        if not _C:
            raise RuntimeError(f"HyperL native library not loaded: {_LOAD_ERROR}")
        self._ctx = c.c_void_p()
        err = _C.hl_context_create(c.byref(self._ctx), device, num_threads)
        if err != 0:
            raise RuntimeError(f"Failed to create HyperL context, error: {err}")

    def __del__(self):
        if hasattr(self, "_ctx") and self._ctx and _C:
            _C.hl_context_destroy(self._ctx)
            self._ctx = None

    @staticmethod
    def version():
        if not _C:
            return "unavailable"
        return _C.hl_engine_version().decode("utf-8")


class Tensor:
    def __init__(self, shape, dtype=DType.F32, context=None):
        self.context = context or Context()
        self.shape = tuple(shape)
        self.dtype = dtype

        desc = HL_TensorDesc()
        desc.dtype = dtype
        desc.ndim = len(shape)
        for i in range(4):
            desc.shape[i] = shape[i] if i < len(shape) else 1
            desc.strides[i] = 0

        self._ptr = c.POINTER(HL_Tensor)()
        err = _C.hl_tensor_alloc(self.context._ctx, c.byref(desc), c.byref(self._ptr))
        if err != 0:
            raise RuntimeError(f"Failed to allocate tensor: {err}")

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr and _C:
            _C.hl_tensor_free(self.context._ctx, self._ptr)
            self._ptr = None

    @classmethod
    def from_numpy(cls, arr, context=None):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        t = cls(arr.shape, dtype=DType.F32, context=context)
        data_addr = t._ptr.contents.data
        c.memmove(data_addr, arr.ctypes.data, arr.nbytes)
        return t

    def to_numpy(self):
        arr = np.empty(self.shape, dtype=np.float32)
        data_addr = self._ptr.contents.data
        c.memmove(arr.ctypes.data, data_addr, arr.nbytes)
        return arr


def rmsnorm(x: Tensor, weight: Tensor = None, eps: float = 1e-5) -> Tensor:
    out = Tensor(x.shape, dtype=x.dtype, context=x.context)
    w_ptr = weight._ptr if weight else None
    err = _C.hl_kernel_rmsnorm(x.context._ctx, x._ptr, w_ptr, eps, out._ptr)
    if err != 0:
        raise RuntimeError(f"rmsnorm failed: {err}")
    return out


def softmax(x: Tensor) -> Tensor:
    out = Tensor(x.shape, dtype=x.dtype, context=x.context)
    err = _C.hl_kernel_softmax(x.context._ctx, x._ptr, out._ptr)
    if err != 0:
        raise RuntimeError(f"softmax failed: {err}")
    return out


def gemv(W: Tensor, x: Tensor, bias: Tensor = None) -> Tensor:
    # W: [M, K], x: [K], out: [M]
    M, K = W.shape[0], W.shape[1]
    out = Tensor((M,), dtype=DType.F32, context=W.context)
    b_ptr = bias._ptr if bias else None
    err = _C.hl_kernel_gemv(W.context._ctx, W._ptr, x._ptr, b_ptr, out._ptr)
    if err != 0:
        raise RuntimeError(f"gemv failed: {err}")
    return out


def gemm(A: Tensor, B: Tensor) -> Tensor:
    # A: [M, K], B: [K, N], out: [M, N]
    M, K = A.shape[0], A.shape[1]
    K2, N = B.shape[0], B.shape[1]
    assert K == K2, f"Shape mismatch: {A.shape} vs {B.shape}"
    out = Tensor((M, N), dtype=DType.F32, context=A.context)
    err = _C.hl_kernel_gemm(A.context._ctx, A._ptr, B._ptr, out._ptr)
    if err != 0:
        raise RuntimeError(f"gemm failed: {err}")
    return out
