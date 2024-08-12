use std::ffi::c_void;

#[allow(dead_code)]
extern "C" {
    pub(crate) fn bitwise_or_u8(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_or_i32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );

    pub(crate) fn leftshift_u8(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
    pub(crate) fn leftshift_i32(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
}
