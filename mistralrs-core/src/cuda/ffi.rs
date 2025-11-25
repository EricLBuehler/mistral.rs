use std::ffi::c_void;

#[allow(dead_code)]
extern "C" {
    pub(crate) fn asort_asc_f32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_f16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_bf16(
        x: *const c_void,
        dst: *const c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_f64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_u8(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_u32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_asc_i64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_f32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_f16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_bf16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_f64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_u8(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_u32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub(crate) fn asort_desc_i64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );

    // Indexed matrix multiplication for MoE
    pub(crate) fn indexed_matmul_f32(
        input: *const c_void,
        expert_weights: *const c_void,
        routing_weights: *const c_void,
        expert_indices: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_dim: i32,
        out_dim: i32,
        num_selected_experts: i32,
        num_experts: i32,
        stream: i64,
    );
    pub(crate) fn indexed_matmul_f16(
        input: *const c_void,
        expert_weights: *const c_void,
        routing_weights: *const c_void,
        expert_indices: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_dim: i32,
        out_dim: i32,
        num_selected_experts: i32,
        num_experts: i32,
        stream: i64,
    );
    pub(crate) fn indexed_matmul_bf16(
        input: *const c_void,
        expert_weights: *const c_void,
        routing_weights: *const c_void,
        expert_indices: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_dim: i32,
        out_dim: i32,
        num_selected_experts: i32,
        num_experts: i32,
        stream: i64,
    );

    // Fused MoE forward pass
    pub(crate) fn fused_moe_forward_f32(
        input: *const c_void,
        gate_weights: *const c_void,
        up_weights: *const c_void,
        down_weights: *const c_void,
        routing_weights: *const c_void,
        expert_indices: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_dim: i32,
        intermediate_dim: i32,
        num_selected_experts: i32,
        num_experts: i32,
        activation_type: i32,
        stream: i64,
    );
    pub(crate) fn fused_moe_forward_f16(
        input: *const c_void,
        gate_weights: *const c_void,
        up_weights: *const c_void,
        down_weights: *const c_void,
        routing_weights: *const c_void,
        expert_indices: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_dim: i32,
        intermediate_dim: i32,
        num_selected_experts: i32,
        num_experts: i32,
        activation_type: i32,
        stream: i64,
    );
    pub(crate) fn fused_moe_forward_bf16(
        input: *const c_void,
        gate_weights: *const c_void,
        up_weights: *const c_void,
        down_weights: *const c_void,
        routing_weights: *const c_void,
        expert_indices: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_dim: i32,
        intermediate_dim: i32,
        num_selected_experts: i32,
        num_experts: i32,
        activation_type: i32,
        stream: i64,
    );

    // Optimized fused MoE forward pass
    pub(crate) fn fused_moe_forward_optimized_f32(
        input: *const c_void,
        gate_weights: *const c_void,
        up_weights: *const c_void,
        down_weights: *const c_void,
        routing_weights: *const c_void,
        expert_indices: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_dim: i32,
        intermediate_dim: i32,
        num_selected_experts: i32,
        num_experts: i32,
        activation_type: i32,
        stream: i64,
    );

    // Chunked fused MoE forward pass for large batches
    pub(crate) fn fused_moe_forward_chunked_f32(
        input: *const c_void,
        gate_weights: *const c_void,
        up_weights: *const c_void,
        down_weights: *const c_void,
        routing_weights: *const c_void,
        expert_indices: *const c_void,
        output: *mut c_void,
        num_tokens: i32,
        hidden_dim: i32,
        intermediate_dim: i32,
        num_selected_experts: i32,
        num_experts: i32,
        activation_type: i32,
        chunk_size: i32,
        stream: i64,
    );
}
