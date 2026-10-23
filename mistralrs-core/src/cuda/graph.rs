use candle_core::cuda_backend::cudarc::driver::DevicePtr;
use candle_core::{DType, Layout, Result, Storage, Tensor};

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

fn launch_2d(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    width: usize,
    height: usize,
    src_pitch: usize,
    dst_pitch: usize,
    stream: i64,
) -> Result<()> {
    let status = unsafe {
        crate::cuda::ffi::cuda_graph_copy_2d_bytes(
            src,
            dst,
            width as i64,
            height as i64,
            src_pitch as i64,
            dst_pitch as i64,
            stream,
        )
    };
    if status != 0 {
        candle_core::bail!("cuda_graph_copy_2d_bytes failed with status {status}");
    }
    Ok(())
}

fn dense_row_geometry(layout: &Layout) -> Option<(usize, usize, usize)> {
    let dims = layout.dims();
    let strides = layout.stride();
    let (&width, outer_dims) = dims.split_last()?;
    let (&last_stride, outer_strides) = strides.split_last()?;
    if outer_dims.is_empty() || last_stride != 1 || width == 0 {
        return None;
    }
    let &pitch = outer_strides.last()?;
    if pitch < width {
        return None;
    }
    let mut expected = pitch;
    for (&dim, &stride) in outer_dims.iter().zip(outer_strides).rev() {
        if dim > 1 && stride != expected {
            return None;
        }
        expected = expected.checked_mul(dim)?;
    }
    Some((width, layout.shape().elem_count() / width, pitch))
}

fn launch_layout(
    src: *const core::ffi::c_void,
    dst: *mut core::ffi::c_void,
    src_layout: &Layout,
    dst_layout: &Layout,
    elem_size: usize,
    stream: i64,
) -> Result<()> {
    if !dst_layout.is_contiguous() {
        candle_core::bail!("CUDA graph copy expected a contiguous destination tensor");
    }
    let n = src_layout.shape().elem_count();
    if src_layout.is_contiguous() {
        return launch(src, dst, n * elem_size, stream);
    }
    let Some((width, height, src_pitch)) = dense_row_geometry(src_layout) else {
        candle_core::bail!("CUDA graph copy source layout is not dense by row");
    };
    launch_2d(
        src,
        dst,
        width * elem_size,
        height,
        src_pitch * elem_size,
        width * elem_size,
        stream,
    )
}

macro_rules! copy_dtype {
    ($storage:expr, $layout:expr, $out_storage:expr, $out_layout:expr, $ty:ty, $stream:expr) => {{
        let src_slice = $storage.as_cuda_slice::<$ty>()?;
        let dst_slice = $out_storage.as_cuda_slice::<$ty>()?;
        let src_view = src_slice.slice($layout.start_offset()..);
        let dst_view = dst_slice.slice($out_layout.start_offset()..);
        let (src_ptr, _src_guard) = src_view.device_ptr(src_slice.stream());
        let (dst_ptr, _dst_guard) = dst_view.device_ptr(dst_slice.stream());
        launch_layout(
            src_ptr as *const core::ffi::c_void,
            dst_ptr as *mut core::ffi::c_void,
            &$layout,
            &$out_layout,
            std::mem::size_of::<$ty>(),
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
    let Storage::Cuda(src_storage) = &*src_storage else {
        candle_core::bail!("CUDA graph copy expected CUDA source storage");
    };
    let Storage::Cuda(dst_storage) = &*dst_storage else {
        candle_core::bail!("CUDA graph copy expected CUDA destination storage");
    };

    let stream = src.device().as_cuda_device()?.cuda_stream().cu_stream() as i64;
    match src.dtype() {
        DType::U8 => copy_dtype!(src_storage, src_layout, dst_storage, dst_layout, u8, stream)?,
        DType::U32 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            u32,
            stream
        )?,
        DType::I16 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            i16,
            stream
        )?,
        DType::I32 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            i32,
            stream
        )?,
        DType::I64 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            i64,
            stream
        )?,
        DType::BF16 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            half::bf16,
            stream
        )?,
        DType::F16 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            half::f16,
            stream
        )?,
        DType::F32 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            f32,
            stream
        )?,
        DType::F64 => copy_dtype!(
            src_storage,
            src_layout,
            dst_storage,
            dst_layout,
            f64,
            stream
        )?,
        dtype => candle_core::bail!("CUDA graph copy unsupported dtype {dtype:?}"),
    }
    Ok(())
}
