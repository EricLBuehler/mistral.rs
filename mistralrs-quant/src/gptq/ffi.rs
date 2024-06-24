use half::f16;
use std::ffi::c_void;

#[allow(dead_code)]
extern "C" {
    /// - `c` is of shape `(a.shape()[0], b_q_weight.shape()[1])`
    /// - `temp_dq` is of shape `(b_q_weight.shape()[0]*32/bit, b_q_weight.shape()[1])`
    /// - `temp_dq` may be preallocated, the init values are unimportant.
    pub(crate) fn gemm_half_q_half_cuda(
        gemm_handle: *const c_void,
        a: *const f16,
        b_q_weight: *const u32,
        b_gptq_qzeros: *const u32,
        b_gptq_scales: *const f16,
        b_g_idx: *const i32,
        c: *mut f16,
        temp_dq: *mut f16,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        groups: i32,
        use_exllama: bool,
        bit: i32,
    ) -> u32;
}
