extern "C" {
    pub(crate) fn launch_pack_1bit_kernel(
        d_input: *const u8,
        d_output: *mut u8,
        num_input_elements: usize,
        input_width: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_pack_2bit_kernel(
        d_input: *const u8,
        d_output: *mut u8,
        num_input_elements: usize,
        input_width: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_pack_3bit_kernel(
        d_input: *const u32,
        d_output: *mut i32,
        num_input_elements: usize,
        input_width: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_pack_4bit_kernel(
        d_input: *const u8,
        d_output: *mut u8,
        num_input_elements: usize,
        input_width: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );

    pub(crate) fn launch_pack_8bit_kernel(
        d_input: *const u8,
        d_output: *mut u8,
        num_elements: usize,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    );
}
