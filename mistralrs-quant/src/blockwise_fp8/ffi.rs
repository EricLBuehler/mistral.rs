use float8::F8E4M3;
use half::{bf16, f16};

#[allow(dead_code)]
extern "C" {
    pub(crate) fn launch_fp8_blockwise_dequantize_f32(
        weight: *const F8E4M3,
        d_scale: *const f32,
        d_out: *mut f32,
        rows: i32,
        cols: i32,
        block_size_rows: i32,
        block_size_cols: i32,
    );

    pub(crate) fn launch_fp8_blockwise_dequantize_bf16(
        weight: *const F8E4M3,
        d_scale: *const f32,
        d_out: *mut bf16,
        rows: i32,
        cols: i32,
        block_size_rows: i32,
        block_size_cols: i32,
    );

    pub(crate) fn launch_fp8_blockwise_dequantize_f16(
        weight: *const F8E4M3,
        d_scale: *const f32,
        d_out: *mut f16,
        rows: i32,
        cols: i32,
        block_size_rows: i32,
        block_size_cols: i32,
    );
}
