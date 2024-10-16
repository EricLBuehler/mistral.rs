use std::os::raw::c_void;

use half::f16;

#[allow(dead_code)]
extern "C" {
    pub(crate) fn reconstruct_exllama(
        b_q_weight: *const u32,
        b_gptq_qzeros: *const u32,
        b_gptq_scales: *const f16,
        b_q_perm: *const i32,
        out: *mut f16,
        size_k: i32,
        size_n: i32,
        groups: i32,
        bit: i32,
    );

    pub(crate) fn reconstruct_gptq(
        b_q_weight: *const u32,
        b_gptq_qzeros: *const u32,
        b_gptq_scales: *const f16,
        b_q_perm: *const i32,
        out: *mut f16,
        size_k: i32,
        size_n: i32,
        groups: i32,
        bit: i32,
    );

    pub(crate) fn gemm_half_q_half_cuda_part(
        a: *const f16,
        b_q_weight: *const u32,
        b_gptq_qzeros: *const u32,
        b_gptq_scales: *const f16,
        b_q_perm: *const i32,
        out: *mut f16,
        m: i32,
        n: i32,
        k: i32,
        m_count: i32,
        groups: i32,
        bit: i32,
    );

    pub(crate) fn gemm_half_q_half_alt(
        a: *const f16,
        b_q_weight: *const u32,
        b_gptq_qzeros: *const u32,
        b_gptq_scales: *const f16,
        b_g_idx: *const i32,
        out: *mut f16,
        m: i32,
        n: i32,
        k: i32,
        bit: i32,
    );

    pub(crate) fn marlin_4bit_f16(
        inputs: *const c_void,
        weight: *const i32,
        scales: *const c_void,
        out: *const c_void,
        m: i32,
        k: i32,
        n: i32,
        workspace: *const c_void, //tensor with at least `n / 128 * max_par` entries that are all zero
        groupsize: i32,
    );

    pub(crate) fn marlin_4bit_bf16(
        inputs: *const c_void,
        weight: *const i32,
        scales: *const c_void,
        out: *const c_void,
        m: i32,
        k: i32,
        n: i32,
        workspace: *const c_void, //tensor with at least `n / 128 * max_par` entries that are all zero
        groupsize: i32,
    );

    pub(crate) fn gptq_marlin_repack(
        weight: *const c_void,
        perm: *const c_void,
        result: *const c_void,
        k: i32,
        n: i32,
        bits: i32,
    );
}
