use candle_core::{
    backend::BackendStorage, CpuStorage, CudaStorage, CustomOp3, DType, Layout, Result, Shape,
    Tensor,
};

pub(crate) const MAX_DYNAMIC_CONV_KERNEL_SIZE: usize = 8;

#[derive(Clone, Copy)]
struct DynamicConvOp {
    batch: i32,
    sequence_length: i32,
    hidden_size: i32,
    group_size: i32,
    kernel_size: i32,
}

impl DynamicConvOp {
    fn validate(
        &self,
        hidden: &CudaStorage,
        hidden_layout: &Layout,
        dynamic: &CudaStorage,
        dynamic_layout: &Layout,
        base: &CudaStorage,
        base_layout: &Layout,
    ) -> Result<()> {
        if hidden.dtype() != dynamic.dtype() || hidden.dtype() != base.dtype() {
            candle_core::bail!(
                "dynamic convolution dtype mismatch: hidden={:?}, dynamic={:?}, base={:?}",
                hidden.dtype(),
                dynamic.dtype(),
                base.dtype()
            );
        }
        let batch = usize::try_from(self.batch).map_err(candle_core::Error::wrap)?;
        let sequence_length =
            usize::try_from(self.sequence_length).map_err(candle_core::Error::wrap)?;
        let hidden_size = usize::try_from(self.hidden_size).map_err(candle_core::Error::wrap)?;
        let group_size = usize::try_from(self.group_size).map_err(candle_core::Error::wrap)?;
        let kernel_size = usize::try_from(self.kernel_size).map_err(candle_core::Error::wrap)?;
        if group_size == 0 || hidden_size % group_size != 0 {
            candle_core::bail!("dynamic convolution group size must divide hidden size");
        }
        let groups = hidden_size / group_size;
        if hidden_layout.shape().dims3()? != (batch, sequence_length, hidden_size)
            || dynamic_layout.shape().dims4()? != (batch, sequence_length, kernel_size, groups)
            || base_layout.shape().dims2()? != (kernel_size, hidden_size)
        {
            candle_core::bail!(
                "dynamic convolution shape mismatch: hidden={:?}, dynamic={:?}, base={:?}",
                hidden_layout.shape(),
                dynamic_layout.shape(),
                base_layout.shape()
            );
        }
        Ok(())
    }
}

impl CustomOp3 for DynamicConvOp {
    fn name(&self) -> &'static str {
        "dynamic-conv"
    }

    fn cpu_fwd(
        &self,
        _hidden: &CpuStorage,
        _hidden_layout: &Layout,
        _dynamic: &CpuStorage,
        _dynamic_layout: &Layout,
        _base: &CpuStorage,
        _base_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("fused dynamic convolution requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        hidden: &CudaStorage,
        hidden_layout: &Layout,
        dynamic: &CudaStorage,
        dynamic_layout: &Layout,
        base: &CudaStorage,
        base_layout: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        self.validate(
            hidden,
            hidden_layout,
            dynamic,
            dynamic_layout,
            base,
            base_layout,
        )?;
        let device = hidden.device();
        let stream = device.cuda_stream();
        let elements = hidden_layout.shape().elem_count();
        let hidden_stride = hidden_layout.stride();
        let dynamic_stride = dynamic_layout.stride();
        let base_stride = base_layout.stride();

        macro_rules! launch {
            ($ty:ty, $ffi:ident) => {{
                let hidden_slice = hidden.as_cuda_slice::<$ty>()?;
                let dynamic_slice = dynamic.as_cuda_slice::<$ty>()?;
                let base_slice = base.as_cuda_slice::<$ty>()?;
                let hidden_view = hidden_slice.slice(hidden_layout.start_offset()..);
                let dynamic_view = dynamic_slice.slice(dynamic_layout.start_offset()..);
                let base_view = base_slice.slice(base_layout.start_offset()..);
                let mut output = unsafe { device.alloc::<$ty>(elements)? };
                let (hidden_ptr, hidden_guard) = hidden_view.device_ptr(&stream);
                let (dynamic_ptr, dynamic_guard) = dynamic_view.device_ptr(&stream);
                let (base_ptr, base_guard) = base_view.device_ptr(&stream);
                let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
                let status = unsafe {
                    crate::cuda::ffi::$ffi(
                        hidden_ptr as *const core::ffi::c_void,
                        dynamic_ptr as *const core::ffi::c_void,
                        base_ptr as *const core::ffi::c_void,
                        output_ptr as *mut core::ffi::c_void,
                        self.batch,
                        self.sequence_length,
                        self.hidden_size,
                        self.group_size,
                        self.kernel_size,
                        hidden_stride[0] as i64,
                        hidden_stride[1] as i64,
                        hidden_stride[2] as i64,
                        dynamic_stride[0] as i64,
                        dynamic_stride[1] as i64,
                        dynamic_stride[2] as i64,
                        dynamic_stride[3] as i64,
                        base_stride[0] as i64,
                        base_stride[1] as i64,
                        stream.cu_stream() as i64,
                    )
                };
                drop(output_guard);
                drop(base_guard);
                drop(dynamic_guard);
                drop(hidden_guard);
                if status != 0 {
                    candle_core::bail!(concat!(stringify!($ffi), " failed with status {}"), status);
                }
                CudaStorage::wrap_cuda_slice(output, device.clone())
            }};
        }

        let output = match hidden.dtype() {
            DType::BF16 => launch!(half::bf16, dynamic_conv_bf16),
            DType::F16 => launch!(half::f16, dynamic_conv_f16),
            DType::F32 => launch!(f32, dynamic_conv_f32),
            dtype => candle_core::bail!("dynamic convolution does not support {dtype:?}"),
        };
        Ok((output, hidden_layout.shape().clone()))
    }
}

pub(crate) fn dynamic_conv(
    hidden: &Tensor,
    dynamic: &Tensor,
    base: &Tensor,
    kernel_size: usize,
    group_size: usize,
) -> Result<Tensor> {
    let (batch, sequence_length, hidden_size) = hidden.dims3()?;
    let op = DynamicConvOp {
        batch: i32::try_from(batch).map_err(candle_core::Error::wrap)?,
        sequence_length: i32::try_from(sequence_length).map_err(candle_core::Error::wrap)?,
        hidden_size: i32::try_from(hidden_size).map_err(candle_core::Error::wrap)?,
        group_size: i32::try_from(group_size).map_err(candle_core::Error::wrap)?,
        kernel_size: i32::try_from(kernel_size).map_err(candle_core::Error::wrap)?,
    };
    let dynamic = dynamic.to_dtype(hidden.dtype())?;
    let base = base.to_dtype(hidden.dtype())?;
    hidden.apply_op3_no_bwd(&dynamic, &base, &op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};

    #[test]
    #[ignore = "requires CUDA"]
    fn matches_reference() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let hidden = Tensor::arange(0f32, 128f32, &device)?
            .reshape((2, 4, 16))?
            .narrow(2, 0, 8)?;
        let dynamic = Tensor::arange(0f32, 128f32, &device)?
            .reshape((2, 4, 2, 2, 4))?
            .affine(0.01, 0.0)?;
        let dynamic = dynamic.i((.., .., 0))?;
        let base = Tensor::arange(0f32, 32f32, &device)?
            .reshape((2, 2, 8))?
            .affine(0.02, 0.0)?
            .i(1)?;
        assert!(!hidden.is_contiguous());
        assert!(!dynamic.is_contiguous());
        let output = dynamic_conv(&hidden, &dynamic, &base, 2, 2)?;
        let hidden_rows = hidden.to_vec3::<f32>()?;
        let dynamic_rows = dynamic.reshape((2, 4, 8))?.to_vec3::<f32>()?;
        let base_rows = base.to_vec2::<f32>()?;
        let mut expected = vec![vec![vec![0f32; 8]; 4]; 2];
        for batch in 0..2 {
            for position in 0..4 {
                for channel in 0..8 {
                    for offset in 0..2 {
                        if offset <= position {
                            expected[batch][position][channel] += hidden_rows[batch]
                                [position - offset][channel]
                                * (base_rows[offset][channel]
                                    + dynamic_rows[batch][position][offset * 4 + channel / 2]);
                        }
                    }
                }
            }
        }
        let actual = output.to_vec3::<f32>()?;
        for (actual_batch, expected_batch) in actual.iter().zip(expected) {
            for (actual_row, expected_row) in actual_batch.iter().zip(expected_batch) {
                for (actual, expected) in actual_row.iter().zip(expected_row) {
                    assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
                }
            }
        }
        Ok(())
    }
}
