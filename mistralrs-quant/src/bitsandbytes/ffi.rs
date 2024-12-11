use candle_core::cuda::cudarc::driver::sys::CUstream;
use half::{bf16, f16};

#[allow(dead_code)]
extern "C" {
    pub(crate) fn dequantize_blockwise_f32_int8(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f32,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_f32_fp4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f32,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_f32_nf4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f32,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );

    pub(crate) fn dequantize_blockwise_f16_int8(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_f16_fp4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_f16_nf4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut f16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );

    pub(crate) fn dequantize_blockwise_bf16_int8(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut bf16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_bf16_fp4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut bf16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );
    pub(crate) fn dequantize_blockwise_bf16_nf4(
        code: *const f32,
        a: *const u8,
        absmax: *const f32,
        out: *mut bf16,
        blocksize: i32,
        n: i32,
        stream: CUstream,
    );

    pub(crate) fn gemm_4bit_inference_naive_f16(
        m: i32,
        n: i32,
        k: i32,
        a: *const f16,
        b: *const u8,
        absmax: *const f32,
        code: *const f32,
        out: *mut f16,
        lda: i32,
        ldb: i32,
        ldc: i32,
        blocksize: i32,
        stream: CUstream,
    );
    pub(crate) fn gemm_4bit_inference_naive_bf16(
        m: i32,
        n: i32,
        k: i32,
        a: *const bf16,
        b: *const u8,
        absmax: *const f32,
        datatype: *const f32,
        out: *mut bf16,
        lda: i32,
        ldb: i32,
        ldc: i32,
        blocksize: i32,
        stream: CUstream,
    );
    pub(crate) fn gemm_4bit_inference_naive_f32(
        m: i32,
        n: i32,
        k: i32,
        a: *const f32,
        b: *const u8,
        absmax: *const f32,
        datatype: *const f32,
        out: *mut f32,
        lda: i32,
        ldb: i32,
        ldc: i32,
        blocksize: i32,
        stream: CUstream,
    );
}
