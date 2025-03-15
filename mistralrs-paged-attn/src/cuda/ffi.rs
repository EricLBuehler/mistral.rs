use core::ffi::{c_int, c_long, c_void};

use candle_core::cuda::cudarc::driver::sys::CUstream;

extern "C" {
    pub fn reshape_and_cache(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        slot_mapping: *const c_long,

        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        x: c_int,
        key_stride: c_int,
        value_stride: c_int,
        stream: CUstream,

        dtype: u32,
    );

    pub fn paged_attention_v1(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,

        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: CUstream,

        dtype: u32,
    );

    pub fn paged_attention_v2(
        out: *const c_void,
        exp_sums: *const f32,
        max_logits: *const f32,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        alibi_slopes: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        softcapping: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,

        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,
        stream: CUstream,

        dtype: u32,
    );

    pub fn concat_and_cache_mla(
        kv_c: *const c_void,         // [num_tokens, kv_lora_rank]
        k_pe: *const c_void,         // [num_tokens, pe_dim]
        kv_cache: *mut c_void,       // [num_blocks, block_size, (kv_lora_rank + pe_dim)]
        slot_mapping: *const c_long, // [num_tokens] or [num_actual_tokens]

        num_tokens: c_int,
        kv_lora_rank: c_int,
        pe_dim: c_int,
        block_size: c_int,

        kv_c_stride: c_int,
        k_pe_stride: c_int,
        block_stride: c_int,
        entry_stride: c_int,
        stream: CUstream,

        dtype: u32, // 0 => f16; 1 => bf16; 2 => f32
    );
}
