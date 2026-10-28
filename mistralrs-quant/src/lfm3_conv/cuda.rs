use std::ffi::c_void;

use candle_core::{backend::BackendStorage, cuda::cudarc::driver::sys::CUstream, CudaStorage};

use super::*;
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

extern "C" {
    fn lfm3_conv_f32(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        state: *const c_void,
        output: *mut c_void,
        next_state: *mut c_void,
        params: *const u64,
        stream: CUstream,
    ) -> i32;
    fn lfm3_conv_f16(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        state: *const c_void,
        output: *mut c_void,
        next_state: *mut c_void,
        params: *const u64,
        stream: CUstream,
    ) -> i32;
    fn lfm3_conv_bf16(
        x: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        state: *const c_void,
        output: *mut c_void,
        next_state: *mut c_void,
        params: *const u64,
        stream: CUstream,
    ) -> i32;
}

fn cuda_storage(storage: &Storage) -> Result<&CudaStorage> {
    match storage {
        Storage::Cuda(storage) => Ok(storage),
        _ => candle_core::bail!("LFM3 convolution expects CUDA storage"),
    }
}

impl ConvInputs<'_> {
    pub(super) fn cuda(&self) -> Result<(Tensor, Tensor)> {
        let (projected, projected_layout) = self.projected.storage_and_layout();
        let (weight, weight_layout) = self.weight.storage_and_layout();
        let (state, state_layout) = self.state.storage_and_layout();
        let bias = self.bias.map(Tensor::storage_and_layout);
        let projected = cuda_storage(&projected)?;
        let weight = cuda_storage(&weight)?;
        let state = cuda_storage(&state)?;
        let (bias, bias_offset) = match &bias {
            Some((storage, layout)) => (cuda_storage(storage)?, layout.start_offset()),
            None => (weight, weight_layout.start_offset()),
        };
        let device = projected.device();
        let stream = device.cuda_stream();

        macro_rules! run {
            ($ty:ty, $kernel:ident) => {{
                let mut output = unsafe { device.alloc::<$ty>(self.output_len()) }?;
                let mut next_state = unsafe { device.alloc::<$ty>(self.state_len()) }?;
                let (x_ptr, _x_guard) = slice_ptr_on_stream(
                    projected.as_cuda_slice::<$ty>()?,
                    projected_layout.start_offset(),
                    &stream,
                );
                let (weight_ptr, _weight_guard) = slice_ptr_on_stream(
                    weight.as_cuda_slice::<$ty>()?,
                    weight_layout.start_offset(),
                    &stream,
                );
                let (bias_ptr, _bias_guard) =
                    slice_ptr_on_stream(bias.as_cuda_slice::<$ty>()?, bias_offset, &stream);
                let (state_ptr, _state_guard) = slice_ptr_on_stream(
                    state.as_cuda_slice::<$ty>()?,
                    state_layout.start_offset(),
                    &stream,
                );
                let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
                let (next_state_ptr, next_state_guard) =
                    slice_ptr_mut_on_stream(&mut next_state, 0, &stream);
                let status = unsafe {
                    $kernel(
                        x_ptr as *const c_void,
                        weight_ptr as *const c_void,
                        bias_ptr as *const c_void,
                        state_ptr as *const c_void,
                        output_ptr as *mut c_void,
                        next_state_ptr as *mut c_void,
                        self.params.as_ptr(),
                        stream.cu_stream(),
                    )
                };
                if status != 0 {
                    candle_core::bail!("LFM3 CUDA convolution launch failed with status {status}");
                }
                drop(output_guard);
                drop(next_state_guard);
                self.tensors(
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(next_state, device.clone())),
                )
            }};
        }
        Ok(match self.projected.dtype() {
            DType::F32 => run!(f32, lfm3_conv_f32),
            DType::F16 => run!(half::f16, lfm3_conv_f16),
            DType::BF16 => run!(half::bf16, lfm3_conv_bf16),
            dtype => candle_core::bail!("Unsupported LFM3 convolution dtype {dtype:?}"),
        })
    }
}
