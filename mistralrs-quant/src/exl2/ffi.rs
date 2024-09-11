use half::f16;

#[allow(dead_code)]
extern "C" {
    pub fn exl2_make_q_matrix(
        q_weight: *const u32,
        q_perm: *const u16,
        q_invperm: *const u16,
        q_scale: *const u32,
        q_scale_max: *const f16,
        q_groups: *const u16,
        q_group_map: *const u16,
    ) -> *mut std::ffi::c_void;

    pub fn exl2_gemm(
        a: *const f16,
        b: *const std::ffi::c_void,
        c: *mut f16,
        m: i32,
        n: i32,
        k: i32,
    );
}