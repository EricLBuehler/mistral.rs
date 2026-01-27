use std::os::raw::c_void;

pub(crate) const HAVE_MARLIN_KERNELS: bool = cfg!(has_marlin_kernels);

#[allow(dead_code)]
extern "C" {
    pub(crate) fn marlin_gptq_4bit_f16(
        inputs: *const c_void,
        weight: *const i32,
        scales: *const c_void,
        zeros: *const c_void,
        out: *const c_void,
        m: i32,
        k: i32,
        n: i32,
        workspace: *const c_void, //tensor with at least `n / 128 * max_par` entries that are all zero
        groupsize: i32,
        stream: i64,
    );

    pub(crate) fn marlin_gptq_4bit_bf16(
        inputs: *const c_void,
        weight: *const i32,
        scales: *const c_void,
        zeros: *const c_void,
        out: *const c_void,
        m: i32,
        k: i32,
        n: i32,
        workspace: *const c_void, //tensor with at least `n / 128 * max_par` entries that are all zero
        groupsize: i32,
        stream: i64,
    );

    pub(crate) fn marlin_awq_4bit_f16(
        inputs: *const c_void,
        weight: *const i32,
        scales: *const c_void,
        zeros: *const c_void,
        out: *const c_void,
        m: i32,
        k: i32,
        n: i32,
        workspace: *const c_void,
        groupsize: i32,
        stream: i64,
    );

    pub(crate) fn marlin_awq_4bit_bf16(
        inputs: *const c_void,
        weight: *const i32,
        scales: *const c_void,
        zeros: *const c_void,
        out: *const c_void,
        m: i32,
        k: i32,
        n: i32,
        workspace: *const c_void,
        groupsize: i32,
        stream: i64,
    );

    pub(crate) fn gptq_marlin_repack(
        weight: *const c_void,
        perm: *const c_void,
        result: *const c_void,
        k: i32,
        n: i32,
        bits: i32,
        stream: i64,
    );

    pub(crate) fn awq_marlin_repack(
        weight: *const c_void,
        perm: *const c_void,
        result: *const c_void,
        k: i32,
        n: i32,
        bits: i32,
        stream: i64,
    );
}
