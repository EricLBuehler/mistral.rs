use half::f16;
use std::ffi::c_void;

// Opaque pointer type for QMatrix
type QMatrixPtr = *mut c_void;

#[allow(dead_code)]
extern "C" {
    pub fn exl2_make_q_matrix(
        device: i32,
        height: i32, // q_perm.size(0);
        width: i32,  // q_weight.size(1);
        groups: i32, // q_scale.size(0);
        q_weight: *const u32,
        q_perm: *const u16,
        q_invperm: *const u16,
        q_scale: *const u32,
        q_scale_max: *const f16,
        q_groups: *const u16,
        q_group_map: *const u16,
    ) -> QMatrixPtr;

    pub fn exl2_destroy_q_matrix(q_matrix: QMatrixPtr);

    pub fn exl2_reconstruct_q_matrix(q_matrix: QMatrixPtr, out: *mut f16);

    pub fn exl2_gemm_cuda(a: *const f16, b: *const c_void, c: *mut f16, m: i32, n: i32, k: i32);
}
