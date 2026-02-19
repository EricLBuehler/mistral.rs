//! FFI bindings for custom GEMV CUDA kernels.
//!
//! These bindings allow Rust code to call the optimized CUDA kernels
//! for matrix-vector/matrix multiplication during decode-phase inference.
//! Supports batch sizes 1-8.

#![allow(dead_code)]
#![allow(improper_ctypes)]

use half::{bf16, f16};
use std::ffi::c_void;

extern "C" {
    /// Launch BF16 GEMV kernel
    /// Y = X @ A^T + bias (optional)
    /// A: [M, K], X: [B, K], bias: [M] (optional), Y: [B, M]
    /// batch_size: 1-8
    pub fn launch_gemv_bf16(
        a: *const bf16,
        x: *const bf16,
        bias: *const bf16,
        y: *mut bf16,
        m: i32,
        k: i32,
        batch_size: i32,
        has_bias: bool,
        stream: *mut c_void,
    );

    /// Launch F16 GEMV kernel
    pub fn launch_gemv_f16(
        a: *const f16,
        x: *const f16,
        bias: *const f16,
        y: *mut f16,
        m: i32,
        k: i32,
        batch_size: i32,
        has_bias: bool,
        stream: *mut c_void,
    );

    /// Launch F32 GEMV kernel
    pub fn launch_gemv_f32(
        a: *const f32,
        x: *const f32,
        bias: *const f32,
        y: *mut f32,
        m: i32,
        k: i32,
        batch_size: i32,
        has_bias: bool,
        stream: *mut c_void,
    );
}
