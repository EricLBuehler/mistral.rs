//! FFI bindings for the PTQ1_0 CUDA kernels.

#![allow(dead_code)]

use std::ffi::c_void;

extern "C" {
    /// PTQ1_0 matmul: optional gather and signs, FWHT into `scratch` (f32), then the ternary dot per row
    pub fn launch_ptq1_0_matmul_f32(
        x: *const c_void,
        w: *const c_void,
        signs: *const c_void,
        gather: *const c_void,
        scratch: *mut c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        b_size: i32,
        do_fwht: i32,
        stream: *mut c_void,
    );

    pub fn launch_ptq1_0_matmul_f16(
        x: *const c_void,
        w: *const c_void,
        signs: *const c_void,
        gather: *const c_void,
        scratch: *mut c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        b_size: i32,
        do_fwht: i32,
        stream: *mut c_void,
    );

    pub fn launch_ptq1_0_matmul_bf16(
        x: *const c_void,
        w: *const c_void,
        signs: *const c_void,
        gather: *const c_void,
        scratch: *mut c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        b_size: i32,
        do_fwht: i32,
        stream: *mut c_void,
    );

    /// PTQ1_0 embedding rows: decode, FWHT, then signs (the inverse fold)
    pub fn launch_ptq1_0_embedding_f32(
        ids: *const c_void,
        w: *const c_void,
        signs: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        n_ids: i32,
        stream: *mut c_void,
    );

    pub fn launch_ptq1_0_embedding_f16(
        ids: *const c_void,
        w: *const c_void,
        signs: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        n_ids: i32,
        stream: *mut c_void,
    );

    pub fn launch_ptq1_0_embedding_bf16(
        ids: *const c_void,
        w: *const c_void,
        signs: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        n_ids: i32,
        stream: *mut c_void,
    );

    /// PTQ1_0 tensor-core GEMM for prefill: bf16 activations, weights decoded to bf16 tiles (sm_80+)
    #[cfg(has_ptq1_0_wmma_kernels)]
    pub fn launch_ptq1_0_gemm_f32(
        x: *const c_void,
        w: *const c_void,
        signs: *const c_void,
        gather: *const c_void,
        scratch: *mut c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        b_size: i32,
        do_fwht: i32,
        stream: *mut c_void,
    );

    #[cfg(has_ptq1_0_wmma_kernels)]
    pub fn launch_ptq1_0_gemm_f16(
        x: *const c_void,
        w: *const c_void,
        signs: *const c_void,
        gather: *const c_void,
        scratch: *mut c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        b_size: i32,
        do_fwht: i32,
        stream: *mut c_void,
    );

    #[cfg(has_ptq1_0_wmma_kernels)]
    pub fn launch_ptq1_0_gemm_bf16(
        x: *const c_void,
        w: *const c_void,
        signs: *const c_void,
        gather: *const c_void,
        scratch: *mut c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        b_size: i32,
        do_fwht: i32,
        stream: *mut c_void,
    );
}
