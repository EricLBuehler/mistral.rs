use candle_core::{cuda::cudarc::driver::sys::CUstream, DType, Result, Storage, Tensor};

use crate::utils::slice_ptr_on_stream;

use super::SYMMETRIC_GROUP;

unsafe extern "C" {
    fn mistralrs_pack_embedding_rows(
        source: *const std::ffi::c_void,
        data: *mut u8,
        scales: *mut std::ffi::c_void,
        groups: i32,
        bits: i32,
        dtype: i32,
        stream: CUstream,
    ) -> i32;
}

pub(super) fn quantize(source: &Tensor, data: &Tensor, scales: &Tensor, bits: usize) -> Result<()> {
    let (rows, columns) = source.dims2()?;
    if !matches!(bits, 4 | 8)
        || !columns.is_multiple_of(SYMMETRIC_GROUP)
        || data.dims() != [rows, columns * bits / 8]
        || scales.dims() != [rows, columns / SYMMETRIC_GROUP]
    {
        candle_core::bail!("Invalid symmetric embedding packing shape or bit width");
    }
    let dev = source.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let (source_storage, source_layout) = source.storage_and_layout();
    let (data_storage, data_layout) = data.storage_and_layout();
    let (scales_storage, scales_layout) = scales.storage_and_layout();
    let (Storage::Cuda(source_storage), Storage::Cuda(data_storage), Storage::Cuda(scales_storage)) =
        (&*source_storage, &*data_storage, &*scales_storage)
    else {
        candle_core::bail!("Embedding packing requires CUDA tensors");
    };
    let (data_ptr, _data_guard) = slice_ptr_on_stream(
        data_storage.as_cuda_slice::<u8>()?,
        data_layout.start_offset(),
        &stream,
    );
    let (scales_ptr, _scales_guard) = slice_ptr_on_stream(
        scales_storage.as_cuda_slice::<half::f16>()?,
        scales_layout.start_offset(),
        &stream,
    );
    let (dtype, source_ptr) = match source.dtype() {
        DType::F16 => (
            0,
            slice_ptr_on_stream(
                source_storage.as_cuda_slice::<half::f16>()?,
                source_layout.start_offset(),
                &stream,
            ),
        ),
        DType::BF16 => (
            1,
            slice_ptr_on_stream(
                source_storage.as_cuda_slice::<half::bf16>()?,
                source_layout.start_offset(),
                &stream,
            ),
        ),
        DType::F32 => (
            2,
            slice_ptr_on_stream(
                source_storage.as_cuda_slice::<f32>()?,
                source_layout.start_offset(),
                &stream,
            ),
        ),
        dtype => candle_core::bail!("Embedding packing does not support {dtype:?} inputs"),
    };
    let status = unsafe {
        mistralrs_pack_embedding_rows(
            source_ptr.0 as *const _,
            data_ptr as *mut u8,
            scales_ptr as *mut _,
            i32::try_from(scales.elem_count())?,
            bits as i32,
            dtype,
            stream.cu_stream(),
        )
    };
    if status != 0 {
        candle_core::bail!("Embedding packing failed with CUDA status {status}");
    }
    Ok(())
}
