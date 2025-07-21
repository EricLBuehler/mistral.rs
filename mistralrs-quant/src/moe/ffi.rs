use core::ffi::{c_int, c_long, c_void};

unsafe extern "C" {
    pub(crate) fn topk_softmax(
        gating_output: *const c_void,
        topk_weight: *const c_void,
        topk_indices: *const c_void,
        token_expert_indices: *const c_void,

        num_experts: c_int,
        num_tokens: c_long,
        topk: c_int,
    );

    pub(crate) fn moe_sum(
        input: *const c_void,
        output: *const c_void,
        hidden_size: c_int,
        num_token: c_long,
        topk: c_int,
        dtype: u32,
    );

    #[allow(dead_code)]
    pub(crate) fn moe_align_block_size(
        topk_ids: *const c_void,
        num_experts: c_long,
        block_size: c_long,
        numel: c_long,
        sorted_token_ids: *const c_void,
        experts_ids: *const c_void,
        num_tokens_post_pad: *const c_void,
        dtype: u32,
    );

    pub(crate) fn moe_wna16_gemm(
        input: *const c_void,
        output: *const c_void,
        b_qweight: *const c_void,
        b_scales: *const c_void,
        b_qzeros: *const c_void,
        topk_weights: *const c_void,
        sorted_token_ids: *const c_void,
        expert_ids: *const c_void,
        num_tokens_post_pad: *const c_void,

        top_k: c_long,
        BLOCK_SIZE_M: c_long,
        BLOCK_SIZE_N: c_long,
        BLOCK_SIZE_K: c_long,
        bit: c_long,

        num_experts: c_int,
        size_m: c_int,
        size_n: c_int,
        size_k: c_int,
        group_size: c_int,
        EM: c_long,
        has_zp: bool,
        mul_topk_weight: bool,

        dtype: u32,
    );
}