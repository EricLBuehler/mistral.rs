use core::ffi::{c_int, c_long, c_void};

extern "C" {
    pub(crate) fn rotary_embedding(
        query: *const c_void,
        key: *const c_void,
        cos_cache: *const c_void,
        sin_cache: *const c_void,

        is_neox: c_int,

        head_size: c_int,
        num_tokens: c_long,
        rot_dim: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        query_stride: c_long,
        key_stride: c_long,

        dtype: u32,
    );
}
