// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Sabbir Hassan Imon and respective contributors.

//! # HyperL: Memory-Aware Native Inference Runtime in Rust
//!
//! Provides safe, high-performance Rust bindings and native SIMD kernels for
//! small-to-mid LLMs on consumer hardware.

use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int, c_void};
use std::ops::{Deref, DerefMut};

pub const MAX_DIMS: usize = 4;
pub const Q4_BLOCK_SIZE: usize = 32;

#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    Int8 = 3,
    Int32 = 4,
    Q4_0 = 5,
    Q4_K = 6,
    Q8_0 = 7,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DeviceType {
    Cpu = 0,
    Vulkan = 1,
    Metal = 2,
    WebGpu = 3,
    Cuda = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TensorDesc {
    pub dtype: DType,
    pub ndim: u32,
    pub shape: [i64; MAX_DIMS],
    pub strides: [i64; MAX_DIMS],
}

#[repr(C)]
pub struct RawTensor {
    pub desc: TensorDesc,
    pub data: *mut c_void,
    pub device: DeviceType,
    pub is_view: bool,
    pub device_buffer_handle: *mut c_void,
}

pub enum RawContext {}

#[derive(Debug, Clone)]
pub enum Error {
    InvalidArgument,
    OutOfMemory,
    DeviceUnavailable,
    UnsupportedDtype,
    ShapeMismatch,
    BackendExecution,
    ModelParse,
    Io,
    Unknown(i32),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidArgument => write!(f, "Invalid argument"),
            Error::OutOfMemory => write!(f, "Out of memory"),
            Error::DeviceUnavailable => write!(f, "Device unavailable"),
            Error::UnsupportedDtype => write!(f, "Unsupported data type"),
            Error::ShapeMismatch => write!(f, "Shape mismatch"),
            Error::BackendExecution => write!(f, "Backend execution error"),
            Error::ModelParse => write!(f, "Model parse error"),
            Error::Io => write!(f, "IO error"),
            Error::Unknown(code) => write!(f, "Unknown error code: {}", code),
        }
    }
}

impl std::error::Error for Error {}

impl From<i32> for Error {
    fn from(code: i32) -> Self {
        match code {
            1 => Error::InvalidArgument,
            2 => Error::OutOfMemory,
            3 => Error::DeviceUnavailable,
            4 => Error::UnsupportedDtype,
            5 => Error::ShapeMismatch,
            6 => Error::BackendExecution,
            7 => Error::ModelParse,
            8 => Error::Io,
            n => Error::Unknown(n),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[allow(dead_code)]
extern "C" {
    fn hl_engine_version() -> *const c_char;
    fn hl_context_create(ctx: *mut *mut RawContext, device: DeviceType, num_threads: c_int) -> c_int;
    fn hl_context_destroy(ctx: *mut RawContext) -> c_int;
    fn hl_context_get_threads(ctx: *const RawContext) -> c_int;
    fn hl_context_set_threads(ctx: *mut RawContext, num_threads: c_int) -> c_int;

    fn hl_tensor_alloc(ctx: *mut RawContext, desc: *const TensorDesc, out: *mut *mut RawTensor) -> c_int;
    fn hl_tensor_free(ctx: *mut RawContext, tensor: *mut RawTensor) -> c_int;
    fn hl_tensor_numel(desc: *const TensorDesc) -> usize;

    fn hl_kernel_rmsnorm(ctx: *mut RawContext, x: *const RawTensor, weight: *const RawTensor, eps: c_float, out: *mut RawTensor) -> c_int;
    fn hl_kernel_rope(ctx: *mut RawContext, q: *mut RawTensor, k: *mut RawTensor, pos: c_int, head_dim: c_int, theta_base: c_float) -> c_int;
    fn hl_kernel_softmax(ctx: *mut RawContext, x: *const RawTensor, out: *mut RawTensor) -> c_int;
    fn hl_kernel_gemv(ctx: *mut RawContext, W: *const RawTensor, x: *const RawTensor, bias: *const RawTensor, y: *mut RawTensor) -> c_int;
    fn hl_kernel_gemm(ctx: *mut RawContext, A: *const RawTensor, B: *const RawTensor, C: *mut RawTensor) -> c_int;
}

/// Returns the native engine version string.
pub fn version() -> String {
    unsafe {
        let ptr = hl_engine_version();
        if ptr.is_null() {
            "unavailable".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

/// Thread-safe RAII execution context
pub struct Context {
    raw: *mut RawContext,
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    pub fn new(device: DeviceType, num_threads: i32) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        let ret = unsafe { hl_context_create(&mut raw, device, num_threads) };
        if ret != 0 || raw.is_null() {
            return Err(Error::from(ret));
        }
        Ok(Context { raw })
    }

    pub fn threads(&self) -> i32 {
        unsafe { hl_context_get_threads(self.raw) }
    }

    pub fn set_threads(&mut self, n: i32) -> Result<()> {
        let ret = unsafe { hl_context_set_threads(self.raw, n) };
        if ret != 0 { Err(Error::from(ret)) } else { Ok(()) }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { hl_context_destroy(self.raw); }
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Safe Owning or Viewing Tensor
pub struct Tensor<'a> {
    ctx: &'a Context,
    raw: *mut RawTensor,
    numel: usize,
}

unsafe impl<'a> Send for Tensor<'a> {}
unsafe impl<'a> Sync for Tensor<'a> {}

impl<'a> Tensor<'a> {
    pub fn new(ctx: &'a Context, shape: &[i64], dtype: DType) -> Result<Self> {
        let mut desc = TensorDesc {
            dtype,
            ndim: shape.len() as u32,
            shape: [1; MAX_DIMS],
            strides: [0; MAX_DIMS],
        };
        for (i, &dim) in shape.iter().enumerate().take(MAX_DIMS) {
            desc.shape[i] = dim;
        }

        let numel = unsafe { hl_tensor_numel(&desc) };
        let mut raw = std::ptr::null_mut();
        let ret = unsafe { hl_tensor_alloc(ctx.raw, &desc, &mut raw) };
        if ret != 0 || raw.is_null() {
            return Err(Error::from(ret));
        }

        Ok(Tensor { ctx, raw, numel })
    }

    pub fn from_slice(ctx: &'a Context, shape: &[i64], data: &[f32]) -> Result<Self> {
        let mut t = Self::new(ctx, shape, DType::F32)?;
        if t.numel != data.len() {
            return Err(Error::ShapeMismatch);
        }
        t.as_mut_slice().copy_from_slice(data);
        Ok(t)
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe {
            let data = (*self.raw).data as *const f32;
            std::slice::from_raw_parts(data, self.numel)
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe {
            let data = (*self.raw).data as *mut f32;
            std::slice::from_raw_parts_mut(data, self.numel)
        }
    }

    pub fn shape(&self) -> Vec<i64> {
        unsafe {
            let ndim = (*self.raw).desc.ndim as usize;
            (&(*self.raw).desc.shape)[..ndim].to_vec()
        }
    }
}

impl<'a> Deref for Tensor<'a> {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a> DerefMut for Tensor<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'a> Drop for Tensor<'a> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { hl_tensor_free(self.ctx.raw, self.raw); }
            self.raw = std::ptr::null_mut();
        }
    }
}

/// Compute Kernels
pub fn rmsnorm<'a>(ctx: &'a Context, x: &Tensor<'a>, weight: Option<&Tensor<'a>>, eps: f32) -> Result<Tensor<'a>> {
    let out = Tensor::new(ctx, &x.shape(), DType::F32)?;
    let w_ptr = weight.map_or(std::ptr::null(), |w| w.raw as *const RawTensor);
    let ret = unsafe {
        hl_kernel_rmsnorm(ctx.raw, x.raw, w_ptr, eps, out.raw)
    };
    if ret != 0 { Err(Error::from(ret)) } else { Ok(out) }
}

pub fn softmax<'a>(ctx: &'a Context, x: &Tensor<'a>) -> Result<Tensor<'a>> {
    let out = Tensor::new(ctx, &x.shape(), DType::F32)?;
    let ret = unsafe {
        hl_kernel_softmax(ctx.raw, x.raw, out.raw)
    };
    if ret != 0 { Err(Error::from(ret)) } else { Ok(out) }
}

pub fn gemv<'a>(ctx: &'a Context, w: &Tensor<'a>, x: &Tensor<'a>, bias: Option<&Tensor<'a>>) -> Result<Tensor<'a>> {
    let shape_w = w.shape();
    if shape_w.len() != 2 {
        return Err(Error::InvalidArgument);
    }
    let m = shape_w[0];
    let y = Tensor::new(ctx, &[m], DType::F32)?;
    let b_ptr = bias.map_or(std::ptr::null(), |b| b.raw as *const RawTensor);
    let ret = unsafe {
        hl_kernel_gemv(ctx.raw, w.raw, x.raw, b_ptr, y.raw)
    };
    if ret != 0 { Err(Error::from(ret)) } else { Ok(y) }
}
