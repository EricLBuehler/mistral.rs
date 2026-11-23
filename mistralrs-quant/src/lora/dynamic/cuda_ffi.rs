use candle_core::cuda::cudarc::driver::sys::CUstream;
use half::{bf16, f16};

pub const MAX_GROUPED_ADAPTERS: usize = 32;

#[repr(C)]
#[derive(Default)]
pub struct GroupedAdapters {
    pub a: [u64; MAX_GROUPED_ADAPTERS],
    pub b: [u64; MAX_GROUPED_ADAPTERS],
    pub rank: [i32; MAX_GROUPED_ADAPTERS],
    pub scale: [f32; MAX_GROUPED_ADAPTERS],
}

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

    pub fn launch_dynamic_lora_grouped_f16(
        input: *const f16,
        rows: *const u32,
        adapter_of_row: *const u32,
        adapters: *const GroupedAdapters,
        hidden: *mut f16,
        output: *mut f16,
        input_features: i32,
        output_features: i32,
        max_rank: i32,
        active_rows: i32,
        adapter_base: u32,
        stream: CUstream,
    ) -> i32;

    pub fn launch_dynamic_lora_grouped_bf16(
        input: *const bf16,
        rows: *const u32,
        adapter_of_row: *const u32,
        adapters: *const GroupedAdapters,
        hidden: *mut bf16,
        output: *mut bf16,
        input_features: i32,
        output_features: i32,
        max_rank: i32,
        active_rows: i32,
        adapter_base: u32,
        stream: CUstream,
    ) -> i32;
}
