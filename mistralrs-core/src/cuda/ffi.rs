use std::ffi::c_void;

#[allow(dead_code)]
extern "C" {
    pub(crate) fn apply_sparse_penalties_f32(
        x: *const c_void,
        dst: *mut c_void,
        token_ids: *const u32,
        counts: *const f32,
        n: i32,
        n_tokens: i32,
        frequency_penalty: f32,
        presence_penalty: f32,
        repetition_penalty: f32,
        stream: i64,
    );
    pub(crate) fn apply_sparse_logits_bias_f32(
        x: *const c_void,
        dst: *mut c_void,
        token_ids: *const u32,
        biases: *const f32,
        n: i32,
        n_tokens: i32,
        stream: i64,
    );
    pub(crate) fn rms_norm_residual_f32(
        x: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        scale: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        eps: f32,
        stream: i64,
    );
    pub(crate) fn rms_norm_residual_f16(
        x: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        scale: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        eps: f32,
        stream: i64,
    );
    pub(crate) fn rms_norm_residual_bf16(
        x: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        scale: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        eps: f32,
        stream: i64,
    );
    pub(crate) fn rms_norm_residual_then_rms_norm_f32(
        x: *const c_void,
        residual: *const c_void,
        residual_weight: *const c_void,
        scale: *const c_void,
        norm_weight: *const c_void,
        residual_dst: *mut c_void,
        norm_dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        residual_eps: f32,
        norm_eps: f32,
        stream: i64,
    );
    pub(crate) fn rms_norm_residual_then_rms_norm_f16(
        x: *const c_void,
        residual: *const c_void,
        residual_weight: *const c_void,
        scale: *const c_void,
        norm_weight: *const c_void,
        residual_dst: *mut c_void,
        norm_dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        residual_eps: f32,
        norm_eps: f32,
        stream: i64,
    );
    pub(crate) fn rms_norm_residual_then_rms_norm_bf16(
        x: *const c_void,
        residual: *const c_void,
        residual_weight: *const c_void,
        scale: *const c_void,
        norm_weight: *const c_void,
        residual_dst: *mut c_void,
        norm_dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        residual_eps: f32,
        norm_eps: f32,
        stream: i64,
    );
    pub(crate) fn rms_norm_strided_4d_f32(
        x: *const c_void,
        weight: *const c_void,
        dst: *mut c_void,
        stride_b: i64,
        stride_h: i64,
        stride_s: i64,
        stride_d: i64,
        batch: i32,
        heads: i32,
        seq_len: i32,
        head_dim: i32,
        eps: f32,
        stream: i64,
    );
    pub(crate) fn rms_norm_strided_4d_f16(
        x: *const c_void,
        weight: *const c_void,
        dst: *mut c_void,
        stride_b: i64,
        stride_h: i64,
        stride_s: i64,
        stride_d: i64,
        batch: i32,
        heads: i32,
        seq_len: i32,
        head_dim: i32,
        eps: f32,
        stream: i64,
    );
    pub(crate) fn rms_norm_strided_4d_bf16(
        x: *const c_void,
        weight: *const c_void,
        dst: *mut c_void,
        stride_b: i64,
        stride_h: i64,
        stride_s: i64,
        stride_d: i64,
        batch: i32,
        heads: i32,
        seq_len: i32,
        head_dim: i32,
        eps: f32,
        stream: i64,
    );
    pub(crate) fn qk_rms_norm_rope(
        q: *const c_void,
        k: *const c_void,
        q_weight: *const c_void,
        k_weight: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        q_stride_b: i64,
        q_stride_h: i64,
        q_stride_s: i64,
        q_stride_d: i64,
        k_stride_b: i64,
        k_stride_h: i64,
        k_stride_s: i64,
        k_stride_d: i64,
        batch: i32,
        q_heads: i32,
        k_heads: i32,
        seq_len: i32,
        head_dim: i32,
        rot_dim: i32,
        cos_batch_stride: i32,
        q_eps: f32,
        k_eps: f32,
        is_neox: i32,
        dtype: i32,
        stream: i64,
    );

    pub(crate) fn qk_rms_norm_rope_positions(
        q: *const c_void,
        k: *const c_void,
        q_weight: *const c_void,
        k_weight: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        q_stride_b: i64,
        q_stride_h: i64,
        q_stride_s: i64,
        q_stride_d: i64,
        k_stride_b: i64,
        k_stride_h: i64,
        k_stride_s: i64,
        k_stride_d: i64,
        batch: i32,
        q_heads: i32,
        k_heads: i32,
        seq_len: i32,
        head_dim: i32,
        rot_dim: i32,
        q_eps: f32,
        k_eps: f32,
        is_neox: i32,
        dtype: i32,
        stream: i64,
    );

    pub(crate) fn qkv_rms_norm_rope_positions(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        q_weight: *const c_void,
        k_weight: *const c_void,
        v_weight: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        v_out: *mut c_void,
        q_stride_b: i64,
        q_stride_h: i64,
        q_stride_s: i64,
        q_stride_d: i64,
        k_stride_b: i64,
        k_stride_h: i64,
        k_stride_s: i64,
        k_stride_d: i64,
        v_stride_b: i64,
        v_stride_h: i64,
        v_stride_s: i64,
        v_stride_d: i64,
        batch: i32,
        q_heads: i32,
        k_heads: i32,
        seq_len: i32,
        head_dim: i32,
        rot_dim: i32,
        q_eps: f32,
        k_eps: f32,
        v_eps: f32,
        is_neox: i32,
        dtype: i32,
        stream: i64,
    );

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

    // for unquntized models (decoding)
    pub fn moe_gemm(
        input: *const c_void,   // input [size_m, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input)
        stream: i64,
    );

    // for unquntized models (prefill)
    pub fn moe_gemm_wmma(
        input: *const c_void,         // device pointer [size_m, size_k]
        weights: *const c_void,       // device pointer [num_experts, size_n, size_k]
        sorted_token_ids: *const i32, // device pointer [size_m]
        expert_ids: *const i32,       // host array [size_m] (expert id per sorted token)
        topk_weights: *const f32,
        output: *mut c_void, // device pointer [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input/output)
        stream: i64,
    );

    // MoE GEMV for decode phase (optimized for small batch sizes M <= 8)
    pub fn moe_gemv(
        input: *const c_void,   // input [size_m or size_m / topk, size_k]
        weights: *const c_void, // weights [num_experts, size_n, size_k]
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        topk_weights: *const f32, // device ptr or nullptr
        output: *mut c_void,      // output [size_m, size_n]
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        dtype: i32, // 0=float16, 1=bf16 (for input)
        stream: i64,
    );

    // Optimized parallel topk for small k (MoE routing)
    // Single kernel call writes to both values and indices buffers
    pub(crate) fn topk_f32(
        input: *const c_void,
        values_out: *mut c_void,  // [nrows, k]
        indices_out: *mut c_void, // [nrows, k] as u32
        nrows: i32,
        ncols: i32,
        k: i32,
        stream: i64,
    );
    pub(crate) fn topk_bf16(
        input: *const c_void,
        values_out: *mut c_void,  // [nrows, k]
        indices_out: *mut c_void, // [nrows, k] as u32
        nrows: i32,
        ncols: i32,
        k: i32,
        stream: i64,
    );
    pub(crate) fn topk_f16(
        input: *const c_void,
        values_out: *mut c_void,  // [nrows, k]
        indices_out: *mut c_void, // [nrows, k] as u32
        nrows: i32,
        ncols: i32,
        k: i32,
        stream: i64,
    );

    pub(crate) fn moe_router_topk_f32(
        logits: *const c_void,
        weights: *mut c_void,
        ids: *mut c_void,
        selection_bias: *const c_void,
        expert_scale: *const c_void,
        nrows: i32,
        n_experts: i32,
        top_k: i32,
        score_mode: i32,
        weight_mode: i32,
        renormalize: bool,
        clamp_logits: bool,
        clamp_min: f32,
        clamp_max: f32,
        norm_min: f32,
        output_scale: f32,
        stream: i64,
    );
    pub(crate) fn moe_router_topk_bf16(
        logits: *const c_void,
        weights: *mut c_void,
        ids: *mut c_void,
        selection_bias: *const c_void,
        expert_scale: *const c_void,
        nrows: i32,
        n_experts: i32,
        top_k: i32,
        score_mode: i32,
        weight_mode: i32,
        renormalize: bool,
        clamp_logits: bool,
        clamp_min: f32,
        clamp_max: f32,
        norm_min: f32,
        output_scale: f32,
        stream: i64,
    );
    pub(crate) fn moe_router_topk_f16(
        logits: *const c_void,
        weights: *mut c_void,
        ids: *mut c_void,
        selection_bias: *const c_void,
        expert_scale: *const c_void,
        nrows: i32,
        n_experts: i32,
        top_k: i32,
        score_mode: i32,
        weight_mode: i32,
        renormalize: bool,
        clamp_logits: bool,
        clamp_min: f32,
        clamp_max: f32,
        norm_min: f32,
        output_scale: f32,
        stream: i64,
    );

    pub(crate) fn topk_large_f32(
        input: *const f32,
        block_values: *mut f32,
        block_indices: *mut u32,
        block_maxes: *mut f32,
        block_sums: *mut f32,
        values_out: *mut f32,
        indices_out: *mut u32,
        softmax_info_out: *mut f32,
        ncols: i32,
        k: i32,
        chunk_size: i32,
        nblocks: i32,
        inv_temperature: f32,
        stream: i64,
    );
    pub(crate) fn topk_large_f32_packed(
        input: *const f32,
        block_values: *mut f32,
        block_indices: *mut u32,
        block_maxes: *mut f32,
        block_sums: *mut f32,
        packed_out: *mut f32,
        ncols: i32,
        k: i32,
        chunk_size: i32,
        nblocks: i32,
        inv_temperature: f32,
        stream: i64,
    );
    pub(crate) fn top1_large_f32_packed(
        input: *const f32,
        block_values: *mut f32,
        block_indices: *mut u32,
        packed_out: *mut f32,
        ncols: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    );

    // Mamba SSM selective scan kernel
    pub(crate) fn selective_scan_cuda(
        x: *const f32,       // (batch, seq_len, n_heads * head_dim)
        dt: *const f32,      // (batch, seq_len, n_heads)
        a: *const f32,       // (n_heads,) - negative exp of A_log
        b: *const f32,       // (batch, seq_len, n_heads * d_state)
        c: *const f32,       // (batch, seq_len, n_heads * d_state)
        d: *const f32,       // (n_heads,)
        dt_bias: *const f32, // (n_heads,)
        state: *mut f32,     // (batch, n_heads, head_dim, d_state)
        y: *mut f32,         // (batch, seq_len, n_heads * head_dim)
        batch_size: i32,
        n_heads: i32,
        head_dim: i32,
        d_state: i32,
        seq_len: i32,
        dt_min: f32,
        dt_max: f32,
        stream: i64,
    );

    // GDN (Gated Delta Net) kernels for qwen3_next
    pub(crate) fn gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        stream: i64,
    );
    // Chunked GDN recurrence for prefill (processes tokens in BT=64 chunks)
    pub(crate) fn chunked_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        stream: i64,
    );
    pub(crate) fn causal_conv1d_update(
        x: *const c_void,
        weight: *const c_void,
        conv_state: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        kernel_size: i32,
        dtype: i32,
        stream: i64,
    );
    pub(crate) fn causal_conv1d_full(
        x: *const c_void,
        weight: *const c_void,
        conv_state_out: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        seq_len: i32,
        kernel_size: i32,
        dtype: i32,
        stream: i64,
    );
    pub(crate) fn fused_gdn_gating(
        b: *const c_void,
        a: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        beta_out: *mut c_void,
        g_out: *mut c_void,
        total_elements: i32,
        num_heads: i32,
        dtype: i32,
        stream: i64,
    );
}
