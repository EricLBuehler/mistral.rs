use std::ffi::c_void;

#[allow(dead_code)]
extern "C" {

    pub(crate) fn gdn_mma_prefill_init() -> i32;
    pub(crate) fn gdn_mma_prefill(params: *const c_void) -> i32;
    pub(crate) fn gdn_mma_fused_prefill(params: *const c_void) -> i32;

    pub(crate) fn cuda_graph_copy_bytes(
        src: *const c_void,
        dst: *mut c_void,
        n: i64,
        stream: i64,
    ) -> i32;
    pub(crate) fn cuda_graph_copy_2d_bytes(
        src: *const c_void,
        dst: *mut c_void,
        width: i64,
        height: i64,
        src_pitch: i64,
        dst_pitch: i64,
        stream: i64,
    ) -> i32;
    pub(crate) fn indexed_row_copy_bf16(
        src: *const c_void,
        dst: *mut c_void,
        dst_rows: *const u32,
        rows: i32,
        row_elements: i64,
        stream: i64,
    ) -> i32;
    pub(crate) fn indexed_row_copy_f16(
        src: *const c_void,
        dst: *mut c_void,
        dst_rows: *const u32,
        rows: i32,
        row_elements: i64,
        stream: i64,
    ) -> i32;
    pub(crate) fn indexed_row_copy_f32(
        src: *const c_void,
        dst: *mut c_void,
        dst_rows: *const u32,
        rows: i32,
        row_elements: i64,
        stream: i64,
    ) -> i32;
    pub(crate) fn pack_completion_input_u32(
        host: *const c_void,
        staged_rows: *const *const c_void,
        output: *mut c_void,
        rows: i32,
        host_width: i32,
        staged_width: i32,
        stream: i64,
    ) -> i32;
    pub(crate) fn pad_decode_input_u32(
        input: *const c_void,
        output: *mut c_void,
        input_rows: i32,
        output_rows: i32,
        width: i32,
        stream: i64,
    ) -> i32;
    pub(crate) fn dynamic_conv_bf16(
        hidden: *const c_void,
        dynamic: *const c_void,
        base: *const c_void,
        output: *mut c_void,
        batch: i32,
        sequence_length: i32,
        hidden_size: i32,
        group_size: i32,
        kernel_size: i32,
        hidden_stride_b: i64,
        hidden_stride_s: i64,
        hidden_stride_h: i64,
        dynamic_stride_b: i64,
        dynamic_stride_s: i64,
        dynamic_stride_k: i64,
        dynamic_stride_g: i64,
        base_stride_k: i64,
        base_stride_h: i64,
        stream: i64,
    ) -> i32;
    pub(crate) fn dynamic_conv_f16(
        hidden: *const c_void,
        dynamic: *const c_void,
        base: *const c_void,
        output: *mut c_void,
        batch: i32,
        sequence_length: i32,
        hidden_size: i32,
        group_size: i32,
        kernel_size: i32,
        hidden_stride_b: i64,
        hidden_stride_s: i64,
        hidden_stride_h: i64,
        dynamic_stride_b: i64,
        dynamic_stride_s: i64,
        dynamic_stride_k: i64,
        dynamic_stride_g: i64,
        base_stride_k: i64,
        base_stride_h: i64,
        stream: i64,
    ) -> i32;
    pub(crate) fn dynamic_conv_f32(
        hidden: *const c_void,
        dynamic: *const c_void,
        base: *const c_void,
        output: *mut c_void,
        batch: i32,
        sequence_length: i32,
        hidden_size: i32,
        group_size: i32,
        kernel_size: i32,
        hidden_stride_b: i64,
        hidden_stride_s: i64,
        hidden_stride_h: i64,
        dynamic_stride_b: i64,
        dynamic_stride_s: i64,
        dynamic_stride_k: i64,
        dynamic_stride_g: i64,
        base_stride_k: i64,
        base_stride_h: i64,
        stream: i64,
    ) -> i32;
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
    pub(crate) fn apply_causal_mask_f32(
        scores: *mut c_void,
        batch_heads: i32,
        q_len: i32,
        kv_len: i32,
        q_offset: i32,
        prefix_len: i32,
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
    pub(crate) fn add_rms_norm_f32(
        x: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        residual_dst: *mut c_void,
        norm_dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        eps: f32,
        stream: i64,
    );
    pub(crate) fn add_rms_norm_f16(
        x: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        residual_dst: *mut c_void,
        norm_dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        eps: f32,
        stream: i64,
    );
    pub(crate) fn add_rms_norm_bf16(
        x: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        residual_dst: *mut c_void,
        norm_dst: *mut c_void,
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
    pub(crate) fn qk_rms_norm_rope(params: *const super::abi::QkRmsNormRopeParams) -> i32;

    pub(crate) fn qk_rms_norm_rope_positions(
        params: *const super::abi::QkRmsNormRopePositionsParams,
    ) -> i32;

    pub(crate) fn rope_sincos_positions(
        positions: *const c_void,
        inv_freq: *const c_void,
        cos_out: *mut c_void,
        sin_out: *mut c_void,
        rows: i32,
        width: i32,
        dtype: i32,
        stream: i64,
    );

    pub(crate) fn qkv_rms_norm_rope_positions(
        params: *const super::abi::QkvRmsNormRopePositionsParams,
    ) -> i32;

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
    pub(crate) fn topk_large_f32_packed_batched(
        input: *const f32,
        inv_temperatures: *const f32,
        block_values: *mut f32,
        block_indices: *mut u32,
        block_maxes: *mut f32,
        block_sums: *mut f32,
        packed_out: *mut f32,
        nrows: i32,
        ncols: i32,
        k: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    );
    pub(crate) fn topk_large_bf16_packed_batched(
        input: *const c_void,
        inv_temperatures: *const f32,
        block_values: *mut f32,
        block_indices: *mut u32,
        block_maxes: *mut f32,
        block_sums: *mut f32,
        packed_out: *mut f32,
        nrows: i32,
        ncols: i32,
        k: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    );
    pub(crate) fn topk_large_f16_packed_batched(
        input: *const c_void,
        inv_temperatures: *const f32,
        block_values: *mut f32,
        block_indices: *mut u32,
        block_maxes: *mut f32,
        block_sums: *mut f32,
        packed_out: *mut f32,
        nrows: i32,
        ncols: i32,
        k: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    );
    pub(crate) fn topk_large_ranked_f32_packed_batched(
        input: *const f32,
        block_values: *mut f32,
        block_indices: *mut u32,
        packed_out: *mut f32,
        radix_workspace: *mut c_void,
        nrows: i32,
        ncols: i32,
        k: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    ) -> i32;
    pub(crate) fn topk_large_ranked_state_words_per_row() -> usize;
    pub(crate) fn topk_large_ranked_bf16_packed_batched(
        input: *const c_void,
        block_values: *mut f32,
        block_indices: *mut u32,
        packed_out: *mut f32,
        radix_workspace: *mut c_void,
        nrows: i32,
        ncols: i32,
        k: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    ) -> i32;
    pub(crate) fn topk_large_ranked_f16_packed_batched(
        input: *const c_void,
        block_values: *mut f32,
        block_indices: *mut u32,
        packed_out: *mut f32,
        radix_workspace: *mut c_void,
        nrows: i32,
        ncols: i32,
        k: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    ) -> i32;
    pub(crate) fn sample_ranked_topk(
        packed: *const f32,
        params: *const f32,
        tokens: *mut u32,
        nrows: i32,
        packed_k: i32,
        stream: i64,
    );
    pub(crate) fn dflash_greedy_select(
        packed_topk: *const f32,
        projected_hidden: *const c_void,
        predecessor_codebook: *const c_void,
        successor_codebook: *const c_void,
        anchors: *const u32,
        selected: *mut u32,
        batch: i32,
        positions: i32,
        rank: i32,
        vocab: i32,
        k: i32,
        packed_width: i32,
        hidden_dtype: i32,
        predecessor_dtype: i32,
        successor_dtype: i32,
        stream: i64,
    );
    pub(crate) fn dflash_sample_select(params: *const super::abi::DFlashSampleSelectParams) -> i32;
    pub(crate) fn sparse_rejection_categorical_f32(
        target_logits: *const f32,
        draft_tokens: *const u32,
        q_token_ids: *const u32,
        q_probs: *const f32,
        inverse_temperatures: *const f32,
        target_top_k: *const u32,
        top_p: *const f32,
        min_p: *const f32,
        accept_uniforms: *const f32,
        sample_uniforms: *const f32,
        outcomes: *mut u32,
        batch: i32,
        drafts: i32,
        vocab: i32,
        q_width: i32,
        stream: i64,
    );
    pub(crate) fn sparse_rejection_topk_f32(
        packed_target: *const f32,
        draft_tokens: *const u32,
        q_token_ids: *const u32,
        q_probs: *const f32,
        inverse_temperatures: *const f32,
        target_top_k: *const u32,
        top_p: *const f32,
        min_p: *const f32,
        accept_uniforms: *const f32,
        sample_uniforms: *const f32,
        outcomes: *mut u32,
        batch: i32,
        drafts: i32,
        vocab: i32,
        q_width: i32,
        packed_k: i32,
        stream: i64,
    );
    pub(crate) fn top1_large_f32_packed(
        input: *const f32,
        block_values: *mut f32,
        block_indices: *mut u32,
        packed_out: *mut f32,
        token_ids_out: *mut u32,
        ncols: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    );
    pub(crate) fn top1_large_bf16_packed(
        input: *const c_void,
        block_values: *mut f32,
        block_indices: *mut u32,
        packed_out: *mut f32,
        token_ids_out: *mut u32,
        ncols: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    );
    pub(crate) fn top1_large_f16_packed(
        input: *const c_void,
        block_values: *mut f32,
        block_indices: *mut u32,
        packed_out: *mut f32,
        token_ids_out: *mut u32,
        ncols: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    );
    pub(crate) fn top1_large_f32_packed_batched(
        input: *const f32,
        block_values: *mut f32,
        block_indices: *mut u32,
        packed_out: *mut f32,
        token_ids_out: *mut u32,
        nrows: i32,
        ncols: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    );
    pub(crate) fn top1_large_bf16_packed_batched(
        input: *const c_void,
        block_values: *mut f32,
        block_indices: *mut u32,
        packed_out: *mut f32,
        token_ids_out: *mut u32,
        nrows: i32,
        ncols: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    );
    pub(crate) fn top1_large_f16_packed_batched(
        input: *const c_void,
        block_values: *mut f32,
        block_indices: *mut u32,
        packed_out: *mut f32,
        token_ids_out: *mut u32,
        nrows: i32,
        ncols: i32,
        chunk_size: i32,
        nblocks: i32,
        stream: i64,
    );
    pub(crate) fn categorical_large_f32_packed_batched(
        input: *const f32,
        inv_temperatures: *const f32,
        uniforms: *const f32,
        block_values: *mut f32,
        block_sums: *mut f32,
        packed_out: *mut f32,
        nrows: i32,
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
    pub(crate) fn gdn_packed_to_padded(
        source: *const c_void,
        output: *mut c_void,
        cu_seqlens: *const u32,
        batch_size: i32,
        padded_len: i32,
        width: i32,
        source_token_stride: i64,
        padding_value: f32,
        dtype: i32,
        stream: i64,
    );
    pub(crate) fn gdn_padded_to_packed(
        source: *const c_void,
        output: *mut c_void,
        cu_seqlens: *const u32,
        batch_size: i32,
        padded_len: i32,
        width: i32,
        source_batch_stride: i64,
        source_token_stride: i64,
        source_feature_stride: i64,
        feature_inner_width: i32,
        dtype: i32,
        stream: i64,
    );
    pub(crate) fn gdn_extract_ragged_conv_state(
        padded_input: *const c_void,
        initial_state: *const c_void,
        output: *mut c_void,
        cu_seqlens: *const u32,
        batch_size: i32,
        padded_len: i32,
        channels: i32,
        state_width: i32,
        input_batch_stride: i64,
        input_token_stride: i64,
        dtype: i32,
        stream: i64,
    );
    pub(crate) fn gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );
    pub(crate) fn warp_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );
    pub(crate) fn vmajor_warp_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );
    pub(crate) fn vmajor_grouped_warp_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        values_per_warp: i32,
        state_dtype: i32,
        stream: i64,
    ) -> i32;
    #[cfg(has_flashinfer_gdn_sm90_kernel)]
    pub(crate) fn mistralrs_flashinfer_gdn_sm90_workspace_size(
        batch_size: i32,
        seq_len: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        sm_count: i32,
        has_slots: i32,
    ) -> u64;
    #[cfg(has_flashinfer_gdn_sm90_kernel)]
    pub(crate) fn mistralrs_flashinfer_gdn_sm90_launch(params: *const c_void) -> i32;
    // Chunked GDN recurrence for prefill (processes tokens in BT=64 chunks)
    pub(crate) fn chunked_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    );
    pub(crate) fn vmajor_chunked_gated_delta_rule_recurrence(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta: *const f32,
        state: *mut c_void,
        output: *mut f32,
        bh: i32,
        seq_len: i32,
        k_dim: i32,
        v_dim: i32,
        slot_indices: *const i32,
        num_heads: i32,
        state_dtype: i32,
        stream: i64,
    ) -> i32;
    pub(crate) fn causal_conv1d_update(
        x: *const c_void,
        weight: *const c_void,
        conv_state: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        kernel_size: i32,
        x_stride_b: i64,
        x_stride_s: i64,
        x_stride_c: i64,
        slot_indices: *const i32,
        dtype: i32,
        stream: i64,
    );
    pub(crate) fn causal_conv1d_full(
        x: *const c_void,
        weight: *const c_void,
        conv_state_in: *const c_void,
        conv_state_out: *mut c_void,
        output: *mut c_void,
        batch_size: i32,
        conv_dim: i32,
        seq_len: i32,
        kernel_size: i32,
        x_stride_b: i64,
        x_stride_s: i64,
        x_stride_c: i64,
        slot_indices: *const i32,
        dtype: i32,
        stream: i64,
    );
    pub(crate) fn gdn_rmsnorm_gated(params: *const super::abi::GdnRmsNormGatedParams) -> i32;
    pub(crate) fn gdn_rmsnorm_gated_quantized_bf16(
        params: *const super::abi::GdnRmsNormGatedQuantizedParams,
    ) -> i32;
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
    pub(crate) fn gdn_prepare_recurrence(
        params: *const super::abi::GdnPrepareRecurrenceParams,
    ) -> i32;

    pub(crate) fn gdn_decode_recurrence(
        params: *const super::abi::GdnDecodeRecurrenceParams,
    ) -> i32;
    pub(crate) fn gdn_speculative_state_commit(
        params: *const super::abi::GdnSpeculativeStateCommitParams,
    ) -> i32;
    pub(crate) fn gdn_speculative_conv_checkpoints(
        params: *const super::abi::GdnSpeculativeConvCheckpointsParams,
    ) -> i32;
    pub(crate) fn gdn_speculative_recurrence_checkpoints(
        params: *const super::abi::GdnSpeculativeRecurrenceCheckpointsParams,
    ) -> i32;
    pub(crate) fn gdn_deferred_recurrence_rmsnorm_gate_value_major_128(
        params: *const super::abi::GdnDeferredRecurrenceParams,
    ) -> i32;
    pub(crate) fn gdn_flush_deferred_state_value_major_128(
        state_pool: *mut f32,
        active_slots: *const u32,
        deferred_key: *const f32,
        deferred_delta: *const f32,
        deferred_decay: *const f32,
        deferred_cursor: *mut u32,
        batch_size: i32,
        capacity: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        tiled_v_heads: i32,
        stream: i64,
    );
    pub(crate) fn gdn_speculative_transition_commit_batched(
        pointer_table: *const u64,
        keep_rows: *const u32,
        active_slots: *const u32,
        layer_count: i32,
        batch_size: i32,
        seq_len: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        conv_dim: i32,
        conv_width: i32,
        tiled_v_heads: i32,
        value_major: i32,
        activation_dtype: i32,
        state_dtype: i32,
        stream: i64,
    );
    pub(crate) fn gdn_speculative_transition_stage_batched(
        pointer_table: *const u64,
        keep_rows: *const u32,
        destination_slots: *const u32,
        layer_count: i32,
        batch_size: i32,
        seq_len: i32,
        max_rows: i32,
        destination_capacity: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        conv_dim: i32,
        activation_dtype: i32,
        stream: i64,
    );
    pub(crate) fn gdn_pending_transition_publish_batched(
        pointer_table: *const u64,
        keep_rows: *const u32,
        destination_slots: *const u32,
        layer_count: i32,
        batch_size: i32,
        max_rows: i32,
        destination_capacity: i32,
        stream: i64,
    );
    pub(crate) fn gdn_pending_transition_apply_batched(
        pointer_table: *const u64,
        active_slots: *const u32,
        layer_count: i32,
        batch_size: i32,
        max_rows: i32,
        capacity: i32,
        num_k_heads: i32,
        num_v_heads: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        conv_dim: i32,
        conv_width: i32,
        tiled_v_heads: i32,
        activation_dtype: i32,
        state_dtype: i32,
        stream: i64,
    );
}
