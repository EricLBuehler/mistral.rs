use candle_core::{CpuStorage, CudaStorage, DType, InplaceOp3, Layout, Result, Tensor};

struct IndexedRowCopy {
    rows: i32,
    row_elements: i64,
}

impl InplaceOp3 for IndexedRowCopy {
    fn name(&self) -> &'static str {
        "indexed-row-copy"
    }

    fn cpu_fwd(
        &self,
        _dst: &mut CpuStorage,
        _dst_layout: &Layout,
        _src: &CpuStorage,
        _src_layout: &Layout,
        _rows: &CpuStorage,
        _rows_layout: &Layout,
    ) -> Result<()> {
        candle_core::bail!("indexed row copy requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        dst: &mut CudaStorage,
        dst_layout: &Layout,
        src: &CudaStorage,
        src_layout: &Layout,
        rows: &CudaStorage,
        rows_layout: &Layout,
    ) -> Result<()> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        let dev = dst.device();
        let stream = dev.cuda_stream();
        let rows_slice = rows.as_cuda_slice::<u32>()?;
        let rows_view = rows_slice.slice(rows_layout.start_offset()..);
        let (rows_ptr, rows_guard) = rows_view.device_ptr(&stream);

        macro_rules! launch {
            ($ty:ty, $ffi:ident) => {{
                let src_slice = src.as_cuda_slice::<$ty>()?;
                let dst_slice = dst.as_cuda_slice_mut::<$ty>()?;
                let src_view = src_slice.slice(src_layout.start_offset()..);
                let mut dst_view = dst_slice.slice_mut(dst_layout.start_offset()..);
                let (src_ptr, src_guard) = src_view.device_ptr(&stream);
                let (dst_ptr, dst_guard) = dst_view.device_ptr_mut(&stream);
                let status = unsafe {
                    crate::cuda::ffi::$ffi(
                        src_ptr as *const core::ffi::c_void,
                        dst_ptr as *mut core::ffi::c_void,
                        rows_ptr as *const u32,
                        self.rows,
                        self.row_elements,
                        stream.cu_stream() as i64,
                    )
                };
                drop(dst_guard);
                drop(src_guard);
                if status != 0 {
                    candle_core::bail!(concat!(stringify!($ffi), " failed with status {}"), status);
                }
            }};
        }

        match dst.dtype() {
            DType::BF16 => launch!(half::bf16, indexed_row_copy_bf16),
            DType::F16 => launch!(half::f16, indexed_row_copy_f16),
            DType::F32 => launch!(f32, indexed_row_copy_f32),
            dtype => candle_core::bail!("indexed row copy does not support {dtype:?}"),
        }
        drop(rows_guard);
        Ok(())
    }
}

pub(crate) fn copy_rows(src: &Tensor, dst: &Tensor, dst_rows: &Tensor) -> Result<()> {
    if !src.device().is_cuda()
        || !src.device().same_device(dst.device())
        || !src.device().same_device(dst_rows.device())
    {
        candle_core::bail!("indexed row copy requires CUDA tensors on one device");
    }
    if src.dtype() != dst.dtype() || dst_rows.dtype() != DType::U32 {
        candle_core::bail!(
            "indexed row copy dtype mismatch: src={:?}, dst={:?}, rows={:?}",
            src.dtype(),
            dst.dtype(),
            dst_rows.dtype()
        );
    }
    if !src.is_contiguous() || !dst.is_contiguous() || !dst_rows.is_contiguous() {
        candle_core::bail!("indexed row copy requires contiguous tensors");
    }
    let src_dims = src.dims();
    let dst_dims = dst.dims();
    let Some((&src_rows, src_row_dims)) = src_dims.split_first() else {
        candle_core::bail!("indexed row copy source must have at least one dimension");
    };
    let Some((_, dst_row_dims)) = dst_dims.split_first() else {
        candle_core::bail!("indexed row copy destination must have at least one dimension");
    };
    if src_rows != dst_rows.dim(0)? || src_row_dims != dst_row_dims {
        candle_core::bail!(
            "indexed row copy shape mismatch: src={src_dims:?}, dst={dst_dims:?}, rows={:?}",
            dst_rows.dims()
        );
    }
    if src_rows == 0 {
        return Ok(());
    }
    let row_elements = src_row_dims.iter().try_fold(1usize, |elements, dim| {
        elements
            .checked_mul(*dim)
            .ok_or_else(|| candle_core::Error::msg("indexed row copy size overflow"))
    })?;
    if row_elements == 0 {
        return Ok(());
    }
    let rows = i32::try_from(src_rows)
        .map_err(|_| candle_core::Error::msg("indexed row copy row count exceeds i32"))?;
    let row_elements = i64::try_from(row_elements)
        .map_err(|_| candle_core::Error::msg("indexed row copy row size exceeds i64"))?;

    dst.inplace_op3(src, dst_rows, &IndexedRowCopy { rows, row_elements })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    #[ignore = "requires CUDA"]
    fn copies_selected_rows_in_place() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let expected = vec![
            vec![4., 5., 6.],
            vec![0., 0., 0.],
            vec![1., 2., 3.],
            vec![0., 0., 0.],
        ];
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let src = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (2, 3), &device)?
                .to_dtype(dtype)?;
            let dst = Tensor::zeros((4, 3), dtype, &device)?;
            let rows = Tensor::from_vec(vec![2u32, 0], (2,), &device)?;
            copy_rows(&src, &dst, &rows)?;
            device.synchronize()?;
            assert_eq!(dst.to_dtype(DType::F32)?.to_vec2::<f32>()?, expected);
        }
        Ok(())
    }
}
