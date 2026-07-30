use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Result, Storage, Tensor};

fn launch(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    n: usize,
    stream: i64,
) -> Result<()> {
    let status = unsafe { crate::cuda::ffi::cuda_graph_copy_bytes(src, dst, n as i64, stream) };
    if status != 0 {
        candle_core::bail!("cuda_graph_copy_bytes failed with status {status}");
    }
    Ok(())
}

macro_rules! copy_dtype {
    ($storage:expr, $layout:expr, $out_storage:expr, $out_layout:expr, $ty:ty, $n:expr, $stream:expr) => {{
        let src_slice = $storage.as_cuda_slice::<$ty>()?;
        let dst_slice = $out_storage.as_cuda_slice::<$ty>()?;
        let src_view = src_slice.slice($layout.start_offset()..);
        let dst_view = dst_slice.slice($out_layout.start_offset()..);
        let (src_ptr, _src_guard) = src_view.device_ptr(src_slice.stream());
        let (dst_ptr, _dst_guard) = dst_view.device_ptr(dst_slice.stream());
        launch(
            src_ptr as *const core::ffi::c_void,
            dst_ptr as *mut core::ffi::c_void,
            $n * std::mem::size_of::<$ty>(),
            $stream,
        )
    }};
}

pub fn copy_tensor(src: &Tensor, dst: &Tensor) -> Result<()> {
    if src.shape() != dst.shape()
        || src.dtype() != dst.dtype()
        || src.device().location() != dst.device().location()
    {
        candle_core::bail!("CUDA graph copy expected matching tensors");
    }
    if !src.device().is_cuda() {
        candle_core::bail!("CUDA graph copy expected CUDA tensors");
    }

    let (src_storage, src_layout) = src.storage_and_layout();
    let (dst_storage, dst_layout) = dst.storage_and_layout();
    if !src_layout.is_contiguous() || !dst_layout.is_contiguous() {
        candle_core::bail!("CUDA graph copy expected contiguous tensors");
    }
    let Storage::Cuda(src_storage) = &*src_storage else {
        candle_core::bail!("CUDA graph copy expected CUDA source storage");
    };
    let Storage::Cuda(dst_storage) = &*dst_storage else {
        candle_core::bail!("CUDA graph copy expected CUDA destination storage");
    };

    let stream = src.device().as_cuda_device()?.cuda_stream().cu_stream() as i64;
    let n = src.elem_count();
    match src.dtype() {
        DType::U8 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            u8,
            n,
            stream
        )?,
        DType::U32 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            u32,
            n,
            stream
        )?,
        DType::I16 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            i16,
            n,
            stream
        )?,
        DType::I32 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            i32,
            n,
            stream
        )?,
        DType::I64 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            i64,
            n,
            stream
        )?,
        DType::BF16 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            half::bf16,
            n,
            stream
        )?,
        DType::F16 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            half::f16,
            n,
            stream
        )?,
        DType::F32 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            f32,
            n,
            stream
        )?,
        DType::F64 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            f64,
            n,
            stream
        )?,
        dtype => candle_core::bail!("CUDA graph copy unsupported dtype {dtype:?}"),
    }
    Ok(())
}
