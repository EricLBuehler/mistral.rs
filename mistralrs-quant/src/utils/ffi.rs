use std::ffi::c_void;

use candle_core::cuda::cudarc::driver::sys::CUstream;

#[allow(dead_code)]
extern "C" {
    pub(crate) fn count_nonzero_bf16(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_f16(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_f32(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_f64(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_u8(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_u32(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_i16(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_i64(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn count_nonzero_i32(d_in: *const c_void, N: u32, stream: CUstream) -> u32;
    pub(crate) fn nonzero_bf16(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_f16(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_f32(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_f64(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_u8(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_u32(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_i64(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_i16(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );
    pub(crate) fn nonzero_i32(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
        stream: CUstream,
    );

    pub(crate) fn bitwise_and_u8(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_and_u32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_and_i64(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_and_i32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_or_u8(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_or_u32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_or_i64(
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
    pub(crate) fn bitwise_xor_u8(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_xor_u32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_xor_i64(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );
    pub(crate) fn bitwise_xor_i32(
        d_in1: *const c_void,
        d_in2: *const c_void,
        d_out: *mut c_void,
        N: u32,
    );

    pub(crate) fn leftshift_u8(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
    pub(crate) fn leftshift_u32(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
    pub(crate) fn leftshift_i64(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
    pub(crate) fn leftshift_i32(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
}
