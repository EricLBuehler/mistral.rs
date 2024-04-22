use core::ffi::{c_int, c_void};

extern "C" {
    pub(crate) fn run_ln(
        x: *const c_void,
        residual: *const c_void,
        gamma: *const c_void,
        beta: *const c_void,
        dst_add: *const c_void,
        dst: *const c_void,
        mu: *const c_void,
        rsigma: *const c_void,

        epsilon: f32,

        hidden_size_rounded: u32,
        rows: u32,
        cols: u32,
        multi_processor_count: i32,

        wtype: u32,
        itype: u32,
        rtype: u32,
        otype: u32,
        ctype: u32,

        is_rms_norm: c_int,
    );
}
