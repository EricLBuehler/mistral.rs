use candle_core::cuda::cudarc::driver::sys::CUstream;
use half::{bf16, f16};

extern "C" {
    pub fn launch_dynamic_lora_f16(
        input: *const f16,
        a: *const f16,
        b: *const f16,
        row_indices: *const u32,
        hidden: *mut f16,
        output: *mut f16,
        input_features: i32,
        output_features: i32,
        rank: i32,
        active_rows: i32,
        scale: f32,
        stream: CUstream,
    ) -> i32;

    pub fn launch_dynamic_lora_bf16(
        input: *const bf16,
        a: *const bf16,
        b: *const bf16,
        row_indices: *const u32,
        hidden: *mut bf16,
        output: *mut bf16,
        input_features: i32,
        output_features: i32,
        rank: i32,
        active_rows: i32,
        scale: f32,
        stream: CUstream,
    ) -> i32;
}
