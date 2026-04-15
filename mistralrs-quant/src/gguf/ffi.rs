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

    /// Launch Q8_1 quantization kernel with BF16 input (fuses bf16→f32 + quantize)
    pub fn launch_quantize_q8_1_bf16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );

    /// Launch Q8_1 quantization kernel with F16 input (fuses f16→f32 + quantize)
    pub fn launch_quantize_q8_1_f16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
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

    /// Launch indexed MoE forward kernel for Q4_0 weights with Q8_1 input
    pub fn launch_indexed_moe_forward_q4_0_q8_1(
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

    /// Launch indexed MoE forward kernel for Q4_1 weights with Q8_1 input
    pub fn launch_indexed_moe_forward_q4_1_q8_1(
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

    /// Launch indexed MoE forward kernel for Q5_0 weights with Q8_1 input
    pub fn launch_indexed_moe_forward_q5_0_q8_1(
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

    /// Launch indexed MoE forward kernel for Q5_1 weights with Q8_1 input
    pub fn launch_indexed_moe_forward_q5_1_q8_1(
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

    /// Launch indexed MoE forward kernel for Q8_1 weights with Q8_1 input
    pub fn launch_indexed_moe_forward_q8_1_q8_1(
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

    // ============== Grouped MoE dispatch and GEMM ==============

    /// Build expert dispatch tables on GPU: expert_bounds + sorted_token_ids
    pub fn launch_moe_dispatch(
        topk_ids: *const i32,
        expert_bounds: *mut i32,
        sorted_token_ids: *mut i32,
        total_assignments: i32,
        num_experts: i32,
        stream: *mut c_void,
    );

    /// Grouped MoE GEMM for Q8_0 weights
    pub fn launch_moe_grouped_gemm_q8_0(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Grouped MoE GEMM for Q4_0 weights
    pub fn launch_moe_grouped_gemm_q4_0(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Grouped MoE GEMM for Q4_1 weights
    pub fn launch_moe_grouped_gemm_q4_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Grouped MoE GEMM for Q5_0 weights
    pub fn launch_moe_grouped_gemm_q5_0(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Grouped MoE GEMM for Q5_1 weights
    pub fn launch_moe_grouped_gemm_q5_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Grouped MoE GEMM for Q8_1 weights
    pub fn launch_moe_grouped_gemm_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Grouped MoE GEMM for Q2_K weights
    pub fn launch_moe_grouped_gemm_q2k(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Grouped MoE GEMM for Q3_K weights
    pub fn launch_moe_grouped_gemm_q3k(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Grouped MoE GEMM for Q4_K weights
    pub fn launch_moe_grouped_gemm_q4k(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Grouped MoE GEMM for Q5_K weights
    pub fn launch_moe_grouped_gemm_q5k(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    /// Grouped MoE GEMM for Q6_K weights
    pub fn launch_moe_grouped_gemm_q6k(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        expert_bounds: *const i32,
        sorted_token_ids: *const i32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        k_padded: i32,
        num_experts: i32,
        topk: i32,
        input_dim1: i32,
        stream: *mut c_void,
    );

    // ============== Fused MoE decode kernels ==============

    // Fused gate+up+activation+multiply launchers
    // All share the same signature: (gate_weights, up_weights, inputs_q8_1,
    //   indices, outputs, n, k, batch, topk, k_padded, act_type, stream)

    pub fn launch_moe_gemv_fused_gate_up_q8_0_q8_1(
        gate_weights: *const c_void,
        up_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_fused_gate_up_q4_0_q8_1(
        gate_weights: *const c_void,
        up_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_fused_gate_up_q4_1_q8_1(
        gate_weights: *const c_void,
        up_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_fused_gate_up_q5_0_q8_1(
        gate_weights: *const c_void,
        up_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_fused_gate_up_q5_1_q8_1(
        gate_weights: *const c_void,
        up_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_fused_gate_up_q8_1_q8_1(
        gate_weights: *const c_void,
        up_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_fused_gate_up_q2k_q8_1(
        gate_weights: *const c_void,
        up_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_fused_gate_up_q3k_q8_1(
        gate_weights: *const c_void,
        up_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_fused_gate_up_q4k_q8_1(
        gate_weights: *const c_void,
        up_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_fused_gate_up_q5k_q8_1(
        gate_weights: *const c_void,
        up_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_fused_gate_up_q6k_q8_1(
        gate_weights: *const c_void,
        up_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        act_type: i32,
        stream: *mut c_void,
    );

    // Fused down+aggregate launchers
    // All share the same signature: (weights, inputs_q8_1, indices,
    //   topk_weights, outputs, n, k, batch, topk, k_padded, stream)

    pub fn launch_moe_gemv_down_aggregate_q8_0_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_down_aggregate_q4_0_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_down_aggregate_q4_1_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_down_aggregate_q5_0_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_down_aggregate_q5_1_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_down_aggregate_q8_1_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_down_aggregate_q2k_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_down_aggregate_q3k_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_down_aggregate_q4k_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_down_aggregate_q5k_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut c_void,
    );
    pub fn launch_moe_gemv_down_aggregate_q6k_q8_1(
        all_weights: *const c_void,
        all_inputs: *const c_void,
        indices: *const u32,
        topk_weights: *const f32,
        all_outputs: *mut f32,
        n: i32,
        k: i32,
        batch: i32,
        topk: i32,
        k_padded: i32,
        stream: *mut c_void,
    );

    // Launchers for the dense GGUF mmvq kernels used by `fast_mmvq`.

    pub fn launch_mmvq_gguf_q4_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_bf16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    pub fn launch_mmvq_gguf_q4_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_f32_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    pub fn launch_mmvq_gguf_q4_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_1_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_1_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q8_0_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q2_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q3_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q4_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q5_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmvq_gguf_q6_k_f16_plain(
        vx: *const c_void,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        stride_col_y: i32,
        stride_col_dst: i32,
        b_size: i32,
        stream: *mut c_void,
    );

    /// BF16 -> Q8_1 quantize
    pub fn launch_mmvq_gguf_quantize_q8_1_bf16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );

    /// F16 -> Q8_1 quantize
    pub fn launch_mmvq_gguf_quantize_q8_1_f16(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );

    /// F32 -> Q8_1 quantize
    pub fn launch_mmvq_gguf_quantize_q8_1_f32(
        x: *const c_void,
        vy: *mut c_void,
        kx: i32,
        kx_padded: i32,
        num_rows: i32,
        stream: *mut c_void,
    );

    // ---- MMQ (prompt) kernels ----

    // MMQ quantize launchers (f32 -> block_q8_1_mmq)
    pub fn launch_mmq_quantize_q8_1_D4(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );
    pub fn launch_mmq_quantize_q8_1_DS4(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );
    pub fn launch_mmq_quantize_q8_1_D2S6(
        x: *const c_void,
        ids: *const i32,
        vy: *mut c_void,
        type_x: i32,
        ne00: i64,
        s01: i64,
        s02: i64,
        s03: i64,
        ne0: i64,
        ne1: i64,
        ne2: i64,
        ne3: i64,
        stream: *mut c_void,
    );

    // MMQ matmul launchers (one per quant type)
    pub fn launch_mmq_gguf_q4_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q4_1(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_1(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q8_0(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q2_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q3_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q4_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q5_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
    pub fn launch_mmq_gguf_q6_k(
        tmp_fixup: *mut c_void,
        x: *const c_void,
        y: *const c_void,
        dst: *mut c_void,
        ncols_x: i64,
        nrows_x: i64,
        ncols_y: i64,
        stride_row_x: i64,
        stride_col_dst: i64,
        cc: i32,
        nsm: i32,
        smpbo: i64,
        warp_size: i32,
        stream: *mut c_void,
    );
}
