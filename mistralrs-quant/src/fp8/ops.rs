use candle_core::{DType, Result, Tensor};
use float8::F8E4M3;

struct Fp8ScalarQuantize;

impl Fp8ScalarQuantize {
    // Returns (q_output, scale_output)
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, weight: &Tensor) -> Result<(Tensor, Tensor)> {
        use candle_core::{
            backend::BackendStorage,
            cuda::{
                cudarc::driver::{DevicePtr, DevicePtrMut},
                WrapErr,
            },
            from_storage_no_op, CudaStorage, Shape, Storage,
        };
        use half::{bf16, f16};

        use crate::fp8::ffi;

        if !ffi::HAVE_FP8_QUANT_KERNELS {
            candle_core::bail!("Do not have FP8 quant kernels.");
        }

        let (weight_s, weight_l) = weight.storage_and_layout();
        let weight_s = match &*weight_s {
            Storage::Cuda(s) => s,
            _ => unreachable!(),
        };

        if weight_l.start_offset() != 0 || !weight_l.is_contiguous() {
            candle_core::bail!("Expected weight to have start offset 0, continuous");
        }

        let dev = weight_s.device();

        let elem_count = weight_l.shape().elem_count();

        let (q_output, scale_output) = match weight_s.dtype() {
            DType::F32 => {
                let weight = weight_s
                    .as_cuda_slice::<f32>()?
                    .slice(weight_l.start_offset()..);

                let mut q_output = weight_s.device().alloc_zeros::<F8E4M3>(elem_count).w()?;
                let mut scale_output = weight_s.device().alloc_zeros::<f32>(1).w()?;
                unsafe {
                    ffi::quantize_scalar_fp8_f32(
                        (*weight.device_ptr()) as *const _,
                        (*q_output.device_ptr_mut()) as *mut _,
                        (*scale_output.device_ptr_mut()) as *mut _,
                        elem_count as u32,
                        *dev.cu_stream(),
                    )
                };
                (
                    CudaStorage::wrap_cuda_slice(q_output, weight_s.device().clone()),
                    CudaStorage::wrap_cuda_slice(scale_output, weight_s.device().clone()),
                )
            }
            DType::F16 => {
                let weight = weight_s
                    .as_cuda_slice::<f16>()?
                    .slice(weight_l.start_offset()..);

                let mut q_output = weight_s.device().alloc_zeros::<F8E4M3>(elem_count).w()?;
                let mut scale_output = weight_s.device().alloc_zeros::<f32>(1).w()?;
                unsafe {
                    ffi::quantize_scalar_fp8_f16(
                        (*weight.device_ptr()) as *const _,
                        (*q_output.device_ptr_mut()) as *mut _,
                        (*scale_output.device_ptr_mut()) as *mut _,
                        elem_count as u32,
                        *dev.cu_stream(),
                    )
                };
                (
                    CudaStorage::wrap_cuda_slice(q_output, weight_s.device().clone()),
                    CudaStorage::wrap_cuda_slice(scale_output, weight_s.device().clone()),
                )
            }
            DType::BF16 => {
                let weight = weight_s
                    .as_cuda_slice::<bf16>()?
                    .slice(weight_l.start_offset()..);

                let mut q_output = unsafe { weight_s.device().alloc::<F8E4M3>(elem_count).w()? };
                let mut scale_output = unsafe { weight_s.device().alloc::<f32>(1).w()? };
                unsafe {
                    ffi::quantize_scalar_fp8_bf16(
                        (*weight.device_ptr()) as *const _,
                        (*q_output.device_ptr_mut()) as *mut _,
                        (*scale_output.device_ptr_mut()) as *mut _,
                        elem_count as u32,
                        *dev.cu_stream(),
                    )
                };
                (
                    CudaStorage::wrap_cuda_slice(q_output, weight_s.device().clone()),
                    CudaStorage::wrap_cuda_slice(scale_output, weight_s.device().clone()),
                )
            }
            other => candle_core::bail!("unexpected out type of fp8 quant {other:?}"),
        };

        let q_output = from_storage_no_op(Storage::Cuda(q_output), weight_l.shape().clone(), false);
        let scale_output =
            from_storage_no_op(Storage::Cuda(scale_output), Shape::from_dims(&[]), false);

        Ok((q_output, scale_output))
    }
}

/// FP8 tensorwide quantization.
/// - Returns (q_weight=f8e4m3, scale=f32)
pub fn quantize_scalar_fp8(weight: &Tensor) -> Result<(Tensor, Tensor)> {
    #[cfg(not(feature = "cuda"))]
    {
        candle_core::bail!("Expected CUDA");
    }
    #[cfg(feature = "cuda")]
    {
        let (qw, scale) = Fp8ScalarQuantize.cuda_fwd(weight)?;
        dbg!(&qw, &scale, scale.shape());

        // let qw = weight.to_dtype(DType::F8E4M3)?;
        // let scale = Tensor::ones(&[], DType::F32, weight.device())?;
        // dbg!(&qw, &scale, scale.shape());
        Ok((qw, scale))
    }
}
