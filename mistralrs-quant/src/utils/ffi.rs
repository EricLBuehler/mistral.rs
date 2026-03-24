use std::ffi::c_void;

use candle_core::cuda::cudarc::driver::sys::CUstream;

#[allow(dead_code)]
extern "C" {
    pub(crate) fn count_nonzero_bf16(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_f16(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_f32(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_f64(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_u8(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_u32(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_i16(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_i64(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_i32(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn nonzero_bf16(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_f16(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_f32(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_f64(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_u8(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_u32(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_i64(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_i16(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_i32(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );

    pub(crate) fn bitwise_and_u8(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_and_u32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_and_i64(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_and_i32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_or_u8(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_or_u32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_or_i64(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_or_i32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_xor_u8(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_xor_u32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_xor_i64(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_xor_i32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );

    pub(crate) fn leftshift_u8(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
    pub(crate) fn leftshift_u32(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
    pub(crate) fn leftshift_i64(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
    pub(crate) fn leftshift_i32(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);

    // Fused GPT-OSS SwiGLU kernel
    pub fn gptoss_swiglu_f16(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        N: u32,
        alpha: f32,
        limit: f32,
        stream: CUstream,
    );
    pub fn gptoss_swiglu_bf16(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        N: u32,
        alpha: f32,
        limit: f32,
        stream: CUstream,
    );
    pub fn gptoss_swiglu_f32(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        N: u32,
        alpha: f32,
        limit: f32,
        stream: CUstream,
    );

    // Fused GPT-OSS SwiGLU for interleaved gate/up data
    pub fn gptoss_swiglu_interleaved_f16(
        gate_up: *const c_void,
        output: *mut c_void,
        N: u32,
        intermediate_size: u32,
        alpha: f32,
        limit: f32,
        stream: CUstream,
    );
    pub fn gptoss_swiglu_interleaved_bf16(
        gate_up: *const c_void,
        output: *mut c_void,
        N: u32,
        intermediate_size: u32,
        alpha: f32,
        limit: f32,
        stream: CUstream,
    );
    pub fn gptoss_swiglu_interleaved_f32(
        gate_up: *const c_void,
        output: *mut c_void,
        N: u32,
        intermediate_size: u32,
        alpha: f32,
        limit: f32,
        stream: CUstream,
    );

    // Fused softmax with sinks for GPT-OSS attention
    pub fn softmax_with_sinks_f16(
        logits: *const c_void,
        sinks: *const c_void,
        mask: *const c_void,
        output: *mut c_void,
        batch_size: i32,
        num_heads: i32,
        q_len: i32,
        k_len: i32,
        scale: f32,
        stream: CUstream,
    );
    pub fn softmax_with_sinks_bf16(
        logits: *const c_void,
        sinks: *const c_void,
        mask: *const c_void,
        output: *mut c_void,
        batch_size: i32,
        num_heads: i32,
        q_len: i32,
        k_len: i32,
        scale: f32,
        stream: CUstream,
    );
    pub fn softmax_with_sinks_f32(
        logits: *const c_void,
        sinks: *const c_void,
        mask: *const c_void,
        output: *mut c_void,
        batch_size: i32,
        num_heads: i32,
        q_len: i32,
        k_len: i32,
        scale: f32,
        stream: CUstream,
    );

    // Fused GLU kernel: output = activation(a) * b
    // activation: 0=SiLU, 1=GELU, 2=ReLU
    pub fn fused_glu_f16(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        N: u32,
        activation: i32,
        stream: CUstream,
    );
    pub fn fused_glu_bf16(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        N: u32,
        activation: i32,
        stream: CUstream,
    );
    pub fn fused_glu_f32(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        N: u32,
        activation: i32,
        stream: CUstream,
    );
}
