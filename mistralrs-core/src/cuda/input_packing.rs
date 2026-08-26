use candle_core::{
    backend::BackendStorage, CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape,
    Storage, Tensor,
};

struct CompletionInputPackOp {
    staged_rows: Vec<Tensor>,
    batch: usize,
    host_width: usize,
    staged_width: usize,
}

impl CustomOp1 for CompletionInputPackOp {
    fn name(&self) -> &'static str {
        "completion-input-pack"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("completion input packing requires CUDA storage")
    }

    fn cuda_fwd(&self, host: &CudaStorage, host_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        if host.dtype() != DType::U32
            || !host_layout.is_contiguous()
            || host_layout.shape().dims2()? != (self.batch, self.host_width)
            || self.staged_rows.len() != self.batch
        {
            candle_core::bail!("invalid CUDA completion input packing layout")
        }
        let row_width = self
            .host_width
            .checked_add(self.staged_width)
            .ok_or_else(|| candle_core::Error::msg("completion input width overflow"))?;
        let output_elements = self
            .batch
            .checked_mul(row_width)
            .ok_or_else(|| candle_core::Error::msg("completion input size overflow"))?;

        let device = host.device();
        let stream = device.cuda_stream();
        let host_values = host.as_cuda_slice::<u32>()?;
        let (host_ptr, host_guard) = host_values.device_ptr(&stream);
        let host_ptr = host_ptr
            .checked_add((host_layout.start_offset() * size_of::<u32>()) as u64)
            .ok_or_else(|| candle_core::Error::msg("completion host pointer overflow"))?;

        let staged_storage = self
            .staged_rows
            .iter()
            .map(Tensor::storage_and_layout)
            .collect::<Vec<_>>();
        let mut staged_ptrs = Vec::with_capacity(self.batch);
        let mut staged_guards = Vec::with_capacity(self.batch);
        for (storage, layout) in &staged_storage {
            let Storage::Cuda(storage) = &**storage else {
                candle_core::bail!("completion staged row is not on CUDA")
            };
            if storage.dtype() != DType::U32
                || !layout.is_contiguous()
                || layout.shape().dims1()? != self.staged_width
            {
                candle_core::bail!("invalid CUDA completion staged row layout")
            }
            let values = storage.as_cuda_slice::<u32>()?;
            let (ptr, guard) = values.device_ptr(&stream);
            let ptr = ptr
                .checked_add((layout.start_offset() * size_of::<u32>()) as u64)
                .ok_or_else(|| candle_core::Error::msg("completion staged pointer overflow"))?;
            staged_ptrs.push(ptr as *const core::ffi::c_void);
            staged_guards.push(guard);
        }

        let mut output = unsafe { device.alloc::<u32>(output_elements)? };
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let batch = i32::try_from(self.batch).map_err(candle_core::Error::wrap)?;
        let host_width = i32::try_from(self.host_width).map_err(candle_core::Error::wrap)?;
        let staged_width = i32::try_from(self.staged_width).map_err(candle_core::Error::wrap)?;
        let status = unsafe {
            super::ffi::pack_completion_input_u32(
                host_ptr as *const core::ffi::c_void,
                staged_ptrs.as_ptr(),
                output_ptr as *mut core::ffi::c_void,
                batch,
                host_width,
                staged_width,
                stream.cu_stream() as i64,
            )
        };
        drop(output_guard);
        drop(staged_guards);
        drop(host_guard);
        if status != 0 {
            candle_core::bail!("pack_completion_input_u32 failed with status {status}")
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            (self.batch, row_width).into(),
        ))
    }
}

struct DecodeInputPadOp {
    input_rows: usize,
    output_rows: usize,
    width: usize,
}

impl CustomOp1 for DecodeInputPadOp {
    fn name(&self) -> &'static str {
        "decode-input-pad"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("decode input padding requires CUDA storage")
    }

    fn cuda_fwd(&self, input: &CudaStorage, input_layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        if input.dtype() != DType::U32
            || !input_layout.is_contiguous()
            || input_layout.shape().dims2()? != (self.input_rows, self.width)
            || self.input_rows == 0
            || self.output_rows < self.input_rows
        {
            candle_core::bail!("invalid CUDA decode input padding layout")
        }
        let output_elements = self
            .output_rows
            .checked_mul(self.width)
            .ok_or_else(|| candle_core::Error::msg("decode input padding size overflow"))?;
        let device = input.device();
        let stream = device.cuda_stream();
        let input_values = input.as_cuda_slice::<u32>()?;
        let (input_ptr, input_guard) = input_values.device_ptr(&stream);
        let input_ptr = input_ptr
            .checked_add((input_layout.start_offset() * size_of::<u32>()) as u64)
            .ok_or_else(|| candle_core::Error::msg("decode input pointer overflow"))?;
        let mut output = unsafe { device.alloc::<u32>(output_elements)? };
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
        let status = unsafe {
            super::ffi::pad_decode_input_u32(
                input_ptr as *const core::ffi::c_void,
                output_ptr as *mut core::ffi::c_void,
                i32::try_from(self.input_rows).map_err(candle_core::Error::wrap)?,
                i32::try_from(self.output_rows).map_err(candle_core::Error::wrap)?,
                i32::try_from(self.width).map_err(candle_core::Error::wrap)?,
                stream.cu_stream() as i64,
            )
        };
        drop(output_guard);
        drop(input_guard);
        if status != 0 {
            candle_core::bail!("pad_decode_input_u32 failed with status {status}")
        }
        Ok((
            CudaStorage::wrap_cuda_slice(output, device.clone()),
            (self.output_rows, self.width).into(),
        ))
    }
}

pub(crate) fn pack_completion_input(host: &Tensor, staged_rows: &[Tensor]) -> Result<Tensor> {
    let (batch, host_width) = host.dims2()?;
    let Some(first) = staged_rows.first() else {
        return Ok(host.clone());
    };
    if !host.device().is_cuda() || host.dtype() != DType::U32 || !host.is_contiguous() {
        candle_core::bail!("completion input packing requires contiguous CUDA U32 host rows")
    }
    if staged_rows.len() != batch {
        candle_core::bail!(
            "completion input has {batch} host rows but {} staged rows",
            staged_rows.len()
        )
    }
    let staged_width = first.dim(0)?;
    if staged_width == 0 {
        return Ok(host.clone());
    }
    for row in staged_rows {
        if row.dtype() != DType::U32
            || row.rank() != 1
            || row.dim(0)? != staged_width
            || !row.is_contiguous()
            || !host.device().same_device(row.device())
        {
            candle_core::bail!("completion input staged rows must be contiguous CUDA U32 rows")
        }
    }
    host.apply_op1_no_bwd(&CompletionInputPackOp {
        staged_rows: staged_rows.to_vec(),
        batch,
        host_width,
        staged_width,
    })
}

pub(crate) fn pad_decode_input(input: &Tensor, output_rows: usize) -> Result<Tensor> {
    let (input_rows, width) = input.dims2()?;
    if output_rows == input_rows {
        return Ok(input.clone());
    }
    if !input.device().is_cuda() || input.dtype() != DType::U32 || !input.is_contiguous() {
        candle_core::bail!("decode input padding requires a contiguous CUDA U32 tensor")
    }
    input.apply_op1_no_bwd(&DecodeInputPadOp {
        input_rows,
        output_rows,
        width,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};

    #[test]
    #[ignore = "requires CUDA"]
    fn packs_current_and_staged_tokens_in_one_launch() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for batch in [1usize, 64, 65] {
            let host_values = (0..(batch + 2) * 2)
                .map(|value| u32::try_from(value).unwrap())
                .collect::<Vec<_>>();
            let staged_values = (0..(batch + 2) * 3)
                .map(|value| u32::try_from(value + 10_000).unwrap())
                .collect::<Vec<_>>();
            let host =
                Tensor::from_vec(host_values, (batch + 2, 2), &device)?.narrow(0, 1, batch)?;
            let staged = Tensor::from_vec(staged_values, (batch + 2, 3), &device)?;
            let staged_rows = (1..batch + 1)
                .map(|row| staged.i(row))
                .collect::<Result<Vec<_>>>()?;
            let packed = pack_completion_input(&host, &staged_rows)?;
            let expected = Tensor::cat(&[host.clone(), staged.narrow(0, 1, batch)?], 1)?;
            device.synchronize()?;
            assert_eq!(packed.to_vec2::<u32>()?, expected.to_vec2::<u32>()?);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn pads_decode_input_by_aliasing_row_zero_values() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let input = Tensor::new(&[[8u32, 9], [1, 2], [3, 4]], &device)?.narrow(0, 1, 2)?;
        let padded = pad_decode_input(&input, 4)?;
        device.synchronize()?;
        assert_eq!(
            padded.to_vec2::<u32>()?,
            vec![vec![1, 2], vec![3, 4], vec![1, 2], vec![1, 2]]
        );
        Ok(())
    }
}
