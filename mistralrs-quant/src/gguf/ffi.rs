//! FFI bindings for indexed MoE CUDA kernels.
//!
//! These bindings allow Rust code to call the optimized CUDA kernels
//! for indexed MoE forward pass with GGUF quantized weights.

#![allow(dead_code)]
#![allow(improper_ctypes)]

use std::ffi::c_void;

extern "C" {
    /// Launch Q8_1 quantization kernel
    /// Quantizes f32 input to Q8_1 format for use with quantized matmul kernels.
    pub fn launch_quantize_q8_1(
        x: *const f32,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_blocks_x: i32,
        num_rows: i32,
        stream: *mut c_void,
    );

    /// Launch indexed MoE forward kernel for Q2_K weights with Q8_1 input
    pub fn launch_indexed_moe_forward_q2k_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Launch indexed MoE forward kernel for Q3_K weights with Q8_1 input
    pub fn launch_indexed_moe_forward_q3k_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Launch indexed MoE forward kernel for Q4_K weights with Q8_1 input
    pub fn launch_indexed_moe_forward_q4k_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Launch indexed MoE forward kernel for Q5_K weights with Q8_1 input
    pub fn launch_indexed_moe_forward_q5k_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Launch indexed MoE forward kernel for Q6_K weights with Q8_1 input
    pub fn launch_indexed_moe_forward_q6k_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Launch indexed MoE forward kernel for Q8_0 weights with Q8_1 input
    pub fn launch_indexed_moe_forward_q8_0_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );
}
