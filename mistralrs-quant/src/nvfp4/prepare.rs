use candle_core::{cuda::cudarc::driver::sys::CUstream, DType, Result};
use std::ffi::c_void;

const MIN_SOURCE_ELEMENTS: usize = 1 << 20;
const MIN_SOURCE_REUSE: usize = 4;

#[repr(C)]
pub(crate) struct RoutedActivation {
    pub input: *const c_void,
    pub global: *const f32,
    pub sorted_ids: *const i32,
    pub expert_ids: *const i32,
    pub num_padded: *const i32,
    pub packed: *mut u8,
    pub scales: *mut u8,
    pub aligned: *mut c_void,
    pub rows: i32,
    pub routes: i32,
    pub k: i32,
    pub input_stride: i32,
    pub alignment: i32,
    pub a4: i32,
}

#[repr(C)]
pub(crate) struct SourceActivation {
    pub packed: *mut u8,
    pub scales: *mut u8,
}

pub(crate) fn supports_source_quantization(rows: usize, k: usize, reuse: usize) -> bool {
    reuse >= MIN_SOURCE_REUSE && rows * k >= MIN_SOURCE_ELEMENTS
}

unsafe extern "C" {
    fn mistralrs_nvfp4_prepare_f16(args: *const RoutedActivation, stream: CUstream) -> i32;
    fn mistralrs_nvfp4_prepare_bf16(args: *const RoutedActivation, stream: CUstream) -> i32;
    fn mistralrs_nvfp4_prepare_f32(args: *const RoutedActivation, stream: CUstream) -> i32;
    fn mistralrs_nvfp4_prepare_source_f16(
        args: *const RoutedActivation,
        source: *const SourceActivation,
        stream: CUstream,
    ) -> i32;

    fn mistralrs_nvfp4_prepare_source_bf16(
        args: *const RoutedActivation,
        source: *const SourceActivation,
        stream: CUstream,
    ) -> i32;

    fn mistralrs_nvfp4_prepare_source_f32(
        args: *const RoutedActivation,
        source: *const SourceActivation,
        stream: CUstream,
    ) -> i32;
}

pub(crate) unsafe fn launch(args: &RoutedActivation, dtype: DType, stream: CUstream) -> Result<()> {
    let status = match dtype {
        DType::F16 => mistralrs_nvfp4_prepare_f16(args, stream),
        DType::BF16 => mistralrs_nvfp4_prepare_bf16(args, stream),
        DType::F32 => mistralrs_nvfp4_prepare_f32(args, stream),
        _ => unreachable!(),
    };
    if status != 0 {
        candle_core::bail!("NVFP4 routed activation CUDA launch failed with error {status}");
    }
    Ok(())
}

pub(crate) unsafe fn launch_source(
    args: &RoutedActivation,
    source: &SourceActivation,
    dtype: DType,
    stream: CUstream,
) -> Result<()> {
    let status = match dtype {
        DType::F16 => mistralrs_nvfp4_prepare_source_f16(args, source, stream),
        DType::BF16 => mistralrs_nvfp4_prepare_source_bf16(args, source, stream),
        DType::F32 => mistralrs_nvfp4_prepare_source_f32(args, source, stream),
        _ => unreachable!(),
    };
    if status != 0 {
        candle_core::bail!("NVFP4 source activation CUDA launch failed with error {status}");
    }
    Ok(())
}
