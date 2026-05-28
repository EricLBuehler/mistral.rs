use candle_core::cuda::cudarc::driver::sys::CUstream;
use core::ffi::c_void;

extern "C" {
    pub fn launch_moe_align(
        topk_ids: *const i32,
        sorted_token_ids: *mut i32,
        expert_ids: *mut i32,
        num_tokens_post_pad: *mut i32,
        cumsum: *mut i32,
        num_experts: i32,
        block_size: i32,
        numel: i32,
        max_num_tokens_padded: i32,
        stream: CUstream,
    );

    pub fn launch_gelu_tanh_and_mul_bf16(
        out: *mut c_void,
        input: *const c_void,
        num_tokens: i32,
        d: i32,
        stream: CUstream,
    );

    pub fn launch_moe_sum_bf16(
        out: *mut c_void,
        input: *const c_void,
        num_tokens: i32,
        hidden: i32,
        topk: i32,
        stream: CUstream,
    );
}
