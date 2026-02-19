use candle_core::{CpuStorage, CustomOp1, DType, Result, Tensor};
use float8::F8E4M3;

#[allow(dead_code)]
struct Fp8ToDtype {
    target_dtype: DType,
}

impl CustomOp1 for Fp8ToDtype {
    fn name(&self) -> &'static str {
        "fp8-to-dtype"
    }

    fn cpu_fwd(
        &self,
        input_s: &candle_core::CpuStorage,
        input_l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let CpuStorage::F8E4M3(input) = input_s else {
            candle_core::bail!("Expected F8E4M3 input!");
        };
        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }

        let output = match self.target_dtype {
            DType::F32 => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    output.push(val.to_f32());
                }
                CpuStorage::F32(output)
            }
            DType::F16 => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    output.push(half::f16::from_f32(val.to_f32()));
                }
                CpuStorage::F16(output)
            }
            DType::BF16 => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    output.push(half::bf16::from_f32(val.to_f32()));
                }
                CpuStorage::BF16(output)
            }
            other => candle_core::bail!("Unsupported target dtype for FP8 conversion: {other:?}"),
        };

        Ok((output, input_l.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input_s: &candle_core::CudaStorage,
        input_l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::{backend::BackendStorage, CudaStorage};
        use half::{bf16, f16};

        use crate::utils::slice_ptr;

        if !super::ffi::HAVE_SCALAR_FP8_KERNELS {
            candle_core::bail!("Do not have scalar FP8 kernels.");
        }

        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }

        let dev = input_s.device();
        let num_elements = input_l.shape().elem_count();

        let (input, _input_guard) =
            slice_ptr(input_s.as_cuda_slice::<F8E4M3>()?, input_l.start_offset());

        let res = match self.target_dtype {
            DType::F32 => {
                let output = dev.alloc_zeros::<f32>(num_elements)?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    super::ffi::launch_fp8_to_f32_kernel(
                        input as *const _,
                        output_ptr as *mut _,
                        num_elements,
                        dev.cuda_stream().cu_stream(),
                    );
                }
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, dev.clone())
            }
            DType::F16 => {
                let output = dev.alloc_zeros::<f16>(num_elements)?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    super::ffi::launch_fp8_to_f16_kernel(
                        input as *const _,
                        output_ptr as *mut _,
                        num_elements,
                        dev.cuda_stream().cu_stream(),
                    );
                }
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, dev.clone())
            }
            DType::BF16 => {
                let output = dev.alloc_zeros::<bf16>(num_elements)?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    super::ffi::launch_fp8_to_bf16_kernel(
                        input as *const _,
                        output_ptr as *mut _,
                        num_elements,
                        dev.cuda_stream().cu_stream(),
                    );
                }
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, dev.clone())
            }
            other => candle_core::bail!("Unsupported target dtype for FP8 conversion: {other:?}"),
        };

        Ok((res, input_l.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        input_s: &candle_core::MetalStorage,
        input_l: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;

        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }

        let device = input_s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("fp8-to-dtype");

        let num_elements = input_l.shape().elem_count();
        let out_shape = input_l.shape().clone();

        let output = device.new_buffer(num_elements, self.target_dtype, "fp8-to-dtype-output")?;

        crate::metal_kernels::call_fp8_to_dtype(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            self.target_dtype,
            input_s.buffer(),
            &output,
            num_elements,
        )
        .map_err(candle_core::Error::wrap)?;

        let newstorage =
            candle_core::MetalStorage::new(output, device.clone(), num_elements, self.target_dtype);
        Ok((newstorage, out_shape))
    }
}

struct DtypeToFp8 {
    source_dtype: DType,
}

impl CustomOp1 for DtypeToFp8 {
    fn name(&self) -> &'static str {
        "dtype-to-fp8"
    }

    fn cpu_fwd(
        &self,
        input_s: &candle_core::CpuStorage,
        input_l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }

        let output = match (self.source_dtype, input_s) {
            (DType::F32, CpuStorage::F32(input)) => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    let clamped = val.clamp(-448.0, 448.0);
                    output.push(F8E4M3::from_f32(clamped));
                }
                CpuStorage::F8E4M3(output)
            }
            (DType::F16, CpuStorage::F16(input)) => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    let f32_val = val.to_f32();
                    let clamped = f32_val.clamp(-448.0, 448.0);
                    output.push(F8E4M3::from_f32(clamped));
                }
                CpuStorage::F8E4M3(output)
            }
            (DType::BF16, CpuStorage::BF16(input)) => {
                let mut output = Vec::with_capacity(input.len());
                for &val in input {
                    let f32_val = val.to_f32();
                    let clamped = f32_val.clamp(-448.0, 448.0);
                    output.push(F8E4M3::from_f32(clamped));
                }
                CpuStorage::F8E4M3(output)
            }
            _ => candle_core::bail!("Mismatched source dtype and storage type"),
        };

        Ok((output, input_l.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input_s: &candle_core::CudaStorage,
        input_l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::{backend::BackendStorage, CudaStorage};
        use half::{bf16, f16};

        use crate::utils::slice_ptr;

        if !super::ffi::HAVE_SCALAR_FP8_KERNELS {
            candle_core::bail!("Do not have scalar FP8 kernels.");
        }

        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }

        let dev = input_s.device();
        let num_elements = input_l.shape().elem_count();

        let output = dev.alloc_zeros::<F8E4M3>(num_elements)?;
        let (output_ptr, output_guard) = slice_ptr(&output, 0);

        match self.source_dtype {
            DType::F32 => {
                let (input, _input_guard) =
                    slice_ptr(input_s.as_cuda_slice::<f32>()?, input_l.start_offset());
                unsafe {
                    super::ffi::launch_f32_to_fp8_kernel(
                        input as *const _,
                        output_ptr as *mut _,
                        num_elements,
                        dev.cuda_stream().cu_stream(),
                    );
                }
            }
            DType::F16 => {
                let (input, _input_guard) =
                    slice_ptr(input_s.as_cuda_slice::<f16>()?, input_l.start_offset());
                unsafe {
                    super::ffi::launch_f16_to_fp8_kernel(
                        input as *const _,
                        output_ptr as *mut _,
                        num_elements,
                        dev.cuda_stream().cu_stream(),
                    );
                }
            }
            DType::BF16 => {
                let (input, _input_guard) =
                    slice_ptr(input_s.as_cuda_slice::<bf16>()?, input_l.start_offset());
                unsafe {
                    super::ffi::launch_bf16_to_fp8_kernel(
                        input as *const _,
                        output_ptr as *mut _,
                        num_elements,
                        dev.cuda_stream().cu_stream(),
                    );
                }
            }
            other => candle_core::bail!("Unsupported source dtype for FP8 conversion: {other:?}"),
        }

        drop(output_guard);
        let res = CudaStorage::wrap_cuda_slice(output, dev.clone());
        Ok((res, input_l.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        input_s: &candle_core::MetalStorage,
        input_l: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;

        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }

        let device = input_s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("dtype-to-fp8");

        let num_elements = input_l.shape().elem_count();
        let out_shape = input_l.shape().clone();

        let output = device.new_buffer(num_elements, DType::F8E4M3, "dtype-to-fp8-output")?;

        crate::metal_kernels::call_dtype_to_fp8(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            self.source_dtype,
            input_s.buffer(),
            &output,
            num_elements,
        )
        .map_err(candle_core::Error::wrap)?;

        let newstorage =
            candle_core::MetalStorage::new(output, device.clone(), num_elements, DType::F8E4M3);
        Ok((newstorage, out_shape))
    }
}

/// Convert an FP8 tensor to another dtype.
#[allow(dead_code)]
pub(crate) fn fp8_to_dtype(input: &Tensor, target_dtype: DType) -> Result<Tensor> {
    if input.dtype() != DType::F8E4M3 {
        candle_core::bail!("Input tensor must be F8E4M3, got {:?}", input.dtype());
    }
    input.apply_op1_no_bwd(&Fp8ToDtype { target_dtype })
}

/// Convert a tensor to FP8.
pub(crate) fn dtype_to_fp8(input: &Tensor) -> Result<Tensor> {
    let source_dtype = input.dtype();
    if !matches!(source_dtype, DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!(
            "Input tensor must be F32, F16, or BF16, got {:?}",
            source_dtype
        );
    }
    input.apply_op1_no_bwd(&DtypeToFp8 { source_dtype })
}
