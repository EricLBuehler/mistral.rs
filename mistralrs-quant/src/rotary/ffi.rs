// The kernels take `int64_t`; `c_long` is only 32 bits on Windows.
use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn rotary_embedding(
        query: *const c_void,
        key: *const c_void,
        cos_cache: *const c_void,
        sin_cache: *const c_void,

        is_neox: c_int,

        head_size: c_int,
        num_tokens: i64,
        rot_dim: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        query_stride: i64,
        key_stride: i64,

        dtype: u32,
        stream: i64,
    );

    pub(crate) fn rotary_embedding_positions(
        query: *const c_void,
        key: *const c_void,
        cos_cache: *const c_void,
        sin_cache: *const c_void,
        positions: *const c_void,

        is_neox: c_int,

        head_size: c_int,
        num_tokens: i64,
        rot_dim: c_int,
        seq_len: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        query_stride: i64,
        key_stride: i64,

        dtype: u32,
        stream: i64,
    );
}
