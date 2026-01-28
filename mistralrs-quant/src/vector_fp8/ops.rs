use candle_core::{CpuStorage, CustomOp2, DType, Result, Tensor, WithDType};
use float8::F8E4M3;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::VECTOR_SIZE;

struct Fp8VectorDequantize {
    out_ty: DType,
}

impl Fp8VectorDequantize {
    fn dispatch_dequant_vector<T: WithDType>(
        &self,
        weight: &[F8E4M3],
        scale: &[f32],
        _weight_l: &candle_core::Layout,
        scale_l: &candle_core::Layout,
    ) -> candle_core::Result<Vec<T>> {
        let num_elements = weight.len();
        let num_vectors = num_elements.div_ceil(VECTOR_SIZE);

        if scale.len() != num_vectors {
            candle_core::bail!(
                "Scale length {} doesn't match expected number of vectors {}",
                scale.len(),
                num_vectors
            );
        }

        let res = vec![T::zero(); num_elements];

        (0..num_vectors).into_par_iter().for_each(|vector_idx| {
            let res_ptr = res.as_ptr() as *mut T;
            let vector_scale = scale[vector_idx * scale_l.stride()[0]];
            let vector_start = vector_idx * VECTOR_SIZE;
            let vector_end = vector_start + VECTOR_SIZE.min(num_elements - vector_start);

            for (idx, &weight_val) in weight[vector_start..vector_end].iter().enumerate() {
                let global_idx = vector_start + idx;
                // SAFETY: We know each thread will only update independent values!
                unsafe {
                    *res_ptr.wrapping_add(global_idx) =
                        T::from_f64((weight_val.to_f32() * vector_scale) as f64);
                }
            }
        });

        Ok(res)
    }
}

impl CustomOp2 for Fp8VectorDequantize {
    fn name(&self) -> &'static str {
        "fp8-vector-dequantize"
    }

    fn cpu_fwd(
        &self,
        scale_s: &candle_core::CpuStorage,
        scale_l: &candle_core::Layout,
        weight_s: &candle_core::CpuStorage,
        weight_l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        let candle_core::CpuStorage::F8E4M3(weight) = weight_s else {
            candle_core::bail!("Expected F8E4M3 weight!");
        };
        let candle_core::CpuStorage::F32(scale) = scale_s else {
            candle_core::bail!("Expected F32 scale!");
        };
        if weight_l.start_offset() != 0 || !weight_l.is_contiguous() {
            candle_core::bail!("Expected weight to have start offset 0, continuous");
        }
        if scale_l.start_offset() != 0 || !scale_l.is_contiguous() {
            candle_core::bail!("Expected scales to have start offset 0, continuous");
        }

        match self.out_ty {
            DType::F32 => Ok((
                CpuStorage::F32(self.dispatch_dequant_vector(weight, scale, weight_l, scale_l)?),
                weight_l.shape().clone(),
            )),
            DType::BF16 => Ok((
                CpuStorage::BF16(self.dispatch_dequant_vector(weight, scale, weight_l, scale_l)?),
                weight_l.shape().clone(),
            )),
            DType::F16 => Ok((
                CpuStorage::F16(self.dispatch_dequant_vector(weight, scale, weight_l, scale_l)?),
                weight_l.shape().clone(),
            )),
            other => candle_core::bail!("unexpected out type of fp8 vector dequant {other:?}"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        scale_s: &candle_core::CudaStorage,
        scale_l: &candle_core::Layout,
        weight_s: &candle_core::CudaStorage,
        weight_l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::{backend::BackendStorage, CudaStorage};
        use half::{bf16, f16};

        use crate::{utils::slice_ptr, vector_fp8::ffi};

        if !ffi::HAVE_VECTOR_DEQUANT_KERNELS {
            candle_core::bail!("Do not have vector FP8 dequant kernels.");
        }

        if weight_l.start_offset() != 0 || !weight_l.is_contiguous() {
            candle_core::bail!("Expected weight to have start offset 0, continuous");
        }
        if scale_l.start_offset() != 0 || !scale_l.is_contiguous() {
            candle_core::bail!("Expected scales to have start offset 0, continuous");
        }

        let dev = weight_s.device();
        let num_elements = weight_l.shape().elem_count();

        let (weight, _weight_guard) =
            slice_ptr(weight_s.as_cuda_slice::<F8E4M3>()?, weight_l.start_offset());
        let (scale, _scale_guard) =
            slice_ptr(scale_s.as_cuda_slice::<f32>()?, scale_l.start_offset());

        let res = match self.out_ty {
            DType::F32 => {
                let output = dev.alloc_zeros::<f32>(num_elements)?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    ffi::launch_dequant_fp8_vector_kernel_f32(
                        weight as *const _,
                        scale as *const _,
                        output_ptr as *mut _,
                        num_elements,
                        dev.cuda_stream().cu_stream(),
                    )
                };
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, dev.clone())
            }
            DType::F16 => {
                let output = dev.alloc_zeros::<f16>(num_elements)?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    ffi::launch_dequant_fp8_vector_kernel_f16(
                        weight as *const _,
                        scale as *const _,
                        output_ptr as *mut _,
                        num_elements,
                        dev.cuda_stream().cu_stream(),
                    )
                };
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, dev.clone())
            }
            DType::BF16 => {
                let output = dev.alloc_zeros::<bf16>(num_elements)?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    ffi::launch_dequant_fp8_vector_kernel_bf16(
                        weight as *const _,
                        scale as *const _,
                        output_ptr as *mut _,
                        num_elements,
                        dev.cuda_stream().cu_stream(),
                    )
                };
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, dev.clone())
            }
            other => candle_core::bail!("unexpected out type of fp8 vector dequant {other:?}"),
        };

        Ok((res, weight_l.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        scale_s: &candle_core::MetalStorage,
        scale_l: &candle_core::Layout,
        weight_s: &candle_core::MetalStorage,
        weight_l: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;

        if weight_l.start_offset() != 0 || !weight_l.is_contiguous() {
            candle_core::bail!("Expected weight to have start offset 0, continuous");
        }
        if scale_l.start_offset() != 0 || !scale_l.is_contiguous() {
            candle_core::bail!("Expected scales to have start offset 0, continuous");
        }

        let device = weight_s.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("fp8-vector-dequant");

        let num_elements = weight_l.shape().elem_count();
        let out_shape = weight_l.shape().clone();

        let output = device.new_buffer(num_elements, self.out_ty, "fp8-vector-dequant-output")?;

        crate::metal_kernels::call_fp8_vector_dequant(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            self.out_ty,
            weight_s.buffer(),
            scale_s.buffer(),
            &output,
            num_elements,
        )
        .map_err(candle_core::Error::wrap)?;

        let newstorage =
            candle_core::MetalStorage::new(output, device.clone(), num_elements, self.out_ty);
        Ok((newstorage, out_shape))
    }
}

/// FP8 vector dequantize.
/// - Expects weight to be fp8
/// - Expects inv_scales to be f32
/// - weight * inv_scale = dequantized
pub fn fp8_vector_dequantize(
    weight: &Tensor,
    inv_scales: &Tensor,
    out_ty: DType,
) -> Result<Tensor> {
    inv_scales.apply_op2_no_bwd(weight, &Fp8VectorDequantize { out_ty })
}

/// CPU implementation of vector quantization
fn cpu_vector_quantize<T: WithDType>(
    input: &[T],
    num_elements: usize,
) -> candle_core::Result<(Vec<F8E4M3>, Vec<f32>)> {
    let num_vectors = num_elements.div_ceil(VECTOR_SIZE);

    let weight = vec![F8E4M3::from_f32(0.0); num_elements];
    let scale = vec![0f32; num_vectors];

    (0..num_vectors).into_par_iter().for_each(|vector_idx| {
        let weight_ptr = weight.as_ptr() as *mut F8E4M3;
        let scale_ptr = scale.as_ptr() as *mut f32;

        let vector_start = vector_idx * VECTOR_SIZE;
        let vector_end = vector_start + VECTOR_SIZE.min(num_elements - vector_start);

        // Find max absolute value in vector
        let mut max_abs = 0f32;
        for &input_val in &input[vector_start..vector_end] {
            let val = input_val.to_f64() as f32;
            let abs_val = val.abs();
            if abs_val > max_abs {
                max_abs = abs_val;
            }
        }

        // Calculate scale
        let vector_scale = if max_abs > 0.0 {
            max_abs / 448.0
        } else {
            1e-12
        };

        // SAFETY: We know each thread will only update independent values!
        unsafe {
            *scale_ptr.wrapping_add(vector_idx) = vector_scale;
        }

        // Quantize values
        for (idx, &input_val) in input[vector_start..vector_end].iter().enumerate() {
            let global_idx = vector_start + idx;
            let val = input_val.to_f64() as f32;
            let scaled_val = (val / vector_scale).clamp(-448.0, 448.0);

            // SAFETY: We know each thread will only update independent values!
            unsafe {
                *weight_ptr.wrapping_add(global_idx) = F8E4M3::from_f32(scaled_val);
            }
        }
    });

    Ok((weight, scale))
}

/// FP8 vector quantize for CPU
fn cpu_fp8_vector_quantize(input: &Tensor) -> Result<(Tensor, Tensor)> {
    let num_elements = input.shape().elem_count();
    let num_vectors = num_elements.div_ceil(VECTOR_SIZE);

    let (weight_data, scale_data) = match input.dtype() {
        DType::F32 => {
            let data = input.to_vec1::<f32>()?;
            cpu_vector_quantize(&data, num_elements)?
        }
        DType::F16 => {
            let data = input.to_vec1::<half::f16>()?;
            cpu_vector_quantize(&data, num_elements)?
        }
        DType::BF16 => {
            let data = input.to_vec1::<half::bf16>()?;
            cpu_vector_quantize(&data, num_elements)?
        }
        other => candle_core::bail!("unexpected input type for fp8 vector quant: {other:?}"),
    };

    // Create tensors from the raw data
    let weight = Tensor::from_vec(weight_data, input.shape(), input.device())?;
    let scale = Tensor::from_vec(scale_data, num_vectors, input.device())?;

    Ok((weight, scale))
}

/// FP8 vector quantize.
/// - Expects input to be f32, f16, or bf16
/// - Returns a tuple of (quantized_weight, scales)
/// - quantized_weight is fp8
/// - scales is f32
/// - Each scale corresponds to a vector of 128 elements
pub fn fp8_vector_quantize(input: &Tensor) -> Result<(Tensor, Tensor)> {
    // Check that tensor size is divisible by 128
    let num_elements = input.shape().elem_count();
    if !num_elements.is_multiple_of(VECTOR_SIZE) {
        candle_core::bail!(
            "Tensor size {} must be divisible by {} for vector FP8 quantization",
            num_elements,
            VECTOR_SIZE
        );
    }

    // Check if we should use CPU implementation
    if matches!(input.device(), candle_core::Device::Cpu) {
        return cpu_fp8_vector_quantize(input);
    }

    #[cfg(feature = "cuda")]
    {
        use candle_core::{CudaStorage, Device, Storage};
        use half::{bf16, f16};

        use crate::{utils::slice_ptr, vector_fp8::ffi};

        if matches!(input.device(), Device::Cuda(_)) {
            if !ffi::HAVE_VECTOR_QUANT_KERNELS {
                candle_core::bail!("Do not have vector FP8 quant kernels.");
            }

            let input_l = input.layout();
            if input_l.start_offset() != 0 || !input_l.is_contiguous() {
                candle_core::bail!("Expected input to have start offset 0, continuous");
            }

            let dev = match input.device() {
                Device::Cuda(dev) => dev,
                _ => unreachable!(),
            };

            let num_vectors = num_elements.div_ceil(VECTOR_SIZE);

            // Allocate output buffers
            let weight_output = dev.alloc_zeros::<F8E4M3>(num_elements)?;
            let scale_output = dev.alloc_zeros::<f32>(num_vectors)?;

            let (weight_ptr, _weight_guard) = slice_ptr(&weight_output, 0);
            let (scale_ptr, _scale_guard) = slice_ptr(&scale_output, 0);

            match input.dtype() {
                DType::F32 => {
                    let input_storage = input.storage_and_layout().0;
                    let input_s = match &*input_storage {
                        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f32>()?,
                        _ => candle_core::bail!("Expected CUDA storage"),
                    };
                    let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
                    unsafe {
                        ffi::launch_quant_fp8_vector_kernel_f32(
                            input_ptr as *const _,
                            weight_ptr as *mut _,
                            scale_ptr as *mut _,
                            num_elements,
                            dev.cuda_stream().cu_stream(),
                        )
                    };
                }
                DType::F16 => {
                    let input_storage = input.storage_and_layout().0;
                    let input_s = match &*input_storage {
                        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f16>()?,
                        _ => candle_core::bail!("Expected CUDA storage"),
                    };
                    let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
                    unsafe {
                        ffi::launch_quant_fp8_vector_kernel_f16(
                            input_ptr as *const _,
                            weight_ptr as *mut _,
                            scale_ptr as *mut _,
                            num_elements,
                            dev.cuda_stream().cu_stream(),
                        )
                    };
                }
                DType::BF16 => {
                    let input_storage = input.storage_and_layout().0;
                    let input_s = match &*input_storage {
                        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<bf16>()?,
                        _ => candle_core::bail!("Expected CUDA storage"),
                    };
                    let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
                    unsafe {
                        ffi::launch_quant_fp8_vector_kernel_bf16(
                            input_ptr as *const _,
                            weight_ptr as *mut _,
                            scale_ptr as *mut _,
                            num_elements,
                            dev.cuda_stream().cu_stream(),
                        )
                    };
                }
                other => {
                    candle_core::bail!("unexpected input type for fp8 vector quant: {other:?}")
                }
            }

            // Drop guards before moving the buffers
            drop(_weight_guard);
            drop(_scale_guard);

            // Create weight tensor by wrapping the CUDA storage
            let weight_storage = CudaStorage::wrap_cuda_slice(weight_output, dev.clone());
            let weight = Tensor::from((Storage::Cuda(weight_storage), input.shape().clone()));

            // Create scale tensor
            let scale_storage = CudaStorage::wrap_cuda_slice(scale_output, dev.clone());
            let scale = Tensor::from((
                Storage::Cuda(scale_storage),
                candle_core::Shape::from_dims(&[num_vectors]),
            ));

            Ok((weight, scale))
        } else {
            candle_core::bail!("Expected CUDA device.");
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        candle_core::bail!("FP8 vector quantization on non-CPU devices requires CUDA feature");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Result, Tensor};

    #[test]
    fn test_fp8_vector_dequant() -> Result<()> {
        let dev = &Device::Cpu;
        let num_elements = 256; // 2 vectors of 128 elements
        let weight = Tensor::ones(num_elements, DType::F8E4M3, dev)?;
        let scales = Tensor::new(&[2.0f32, 3.0f32], dev)?; // 2 scales for 2 vectors

        let dequant = fp8_vector_dequantize(&weight, &scales, DType::F32)?;
        let res = dequant.to_vec1::<f32>()?;

        // First 128 elements should be 2.0, next 128 should be 3.0
        for &val in &res[0..128] {
            assert_eq!(val, 2.0);
        }
        for &val in &res[128..256] {
            assert_eq!(val, 3.0);
        }

        Ok(())
    }

    #[test]
    fn test_fp8_vector_quant_cpu() -> Result<()> {
        let dev = &Device::Cpu;

        // Create test input with 256 elements (2 vectors)
        let input = Tensor::randn(0f32, 2f32, 256, dev)?;

        // Quantize
        let (quantized, scales) = fp8_vector_quantize(&input)?;

        // Verify shapes
        assert_eq!(quantized.shape(), input.shape());
        assert_eq!(scales.dims1()?, 2); // 256/128 = 2 vectors

        // Dequantize
        let dequantized = fp8_vector_dequantize(&quantized, &scales, input.dtype())?;

        // Check that shapes match
        assert_eq!(dequantized.shape(), input.shape());

        // The values won't be exactly the same due to quantization loss,
        // but they should be reasonably close
        let input_vec = input.to_vec1::<f32>()?;
        let dequant_vec = dequantized.to_vec1::<f32>()?;

        let mut max_error = 0f32;
        for (val_in, val_out) in input_vec.iter().zip(dequant_vec.iter()) {
            let error = (val_in - val_out).abs();
            max_error = max_error.max(error);
        }

        // FP8 E4M3 has limited precision, so we expect some error
        assert!(max_error < 0.27, "Max error {max_error} is too large");

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_vector_quant_dequant_roundtrip() -> Result<()> {
        let dev = &Device::new_cuda(0)?;

        // Create test input with 256 elements (2 vectors)
        let input = Tensor::randn(0f32, 2f32, 256, dev)?;

        // Quantize
        let (quantized, scales) = fp8_vector_quantize(&input)?;

        // Verify shapes
        assert_eq!(quantized.shape(), input.shape());
        assert_eq!(scales.dims1()?, 2); // 256/128 = 2 vectors

        // Dequantize
        let dequantized = fp8_vector_dequantize(&quantized, &scales, input.dtype())?;

        // Check that shapes match
        assert_eq!(dequantized.shape(), input.shape());

        // The values won't be exactly the same due to quantization loss,
        // but they should be reasonably close
        let input_vec = input.to_vec1::<f32>()?;
        let dequant_vec = dequantized.to_vec1::<f32>()?;

        let mut max_error = 0f32;
        for (val_in, val_out) in input_vec.iter().zip(dequant_vec.iter()) {
            let error = (val_in - val_out).abs();
            max_error = max_error.max(error);
        }

        // FP8 E4M3 has limited precision, so we expect some error
        assert!(max_error < 0.24, "Max error {} is too large", max_error);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_vector_cpu_cuda_equivalence() -> Result<()> {
        let cpu_dev = &Device::Cpu;
        let cuda_dev = &Device::new_cuda(0)?;

        // Create the same input data on both devices
        let input_data: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) / 10.0).collect();
        let cpu_input = Tensor::from_vec(input_data.clone(), 256, cpu_dev)?;
        let cuda_input = Tensor::from_vec(input_data, 256, cuda_dev)?;

        // Quantize on CPU
        let (cpu_quantized, cpu_scales) = fp8_vector_quantize(&cpu_input)?;

        // Quantize on CUDA
        let (cuda_quantized, cuda_scales) = fp8_vector_quantize(&cuda_input)?;

        // Move CUDA results to CPU for comparison
        let cuda_quantized_cpu = cuda_quantized.to_device(cpu_dev)?;
        let cuda_scales_cpu = cuda_scales.to_device(cpu_dev)?;

        // Compare quantized weights
        let cpu_quant_vec = cpu_quantized.to_vec1::<F8E4M3>()?;
        let cuda_quant_vec = cuda_quantized_cpu.to_vec1::<F8E4M3>()?;

        assert_eq!(cpu_quant_vec.len(), cuda_quant_vec.len());

        let mut num_differences = 0;
        for (i, (cpu_val, cuda_val)) in cpu_quant_vec.iter().zip(cuda_quant_vec.iter()).enumerate()
        {
            if cpu_val.to_f32() != cuda_val.to_f32() {
                // Allow small differences due to floating point precision
                let diff = (cpu_val.to_f32() - cuda_val.to_f32()).abs();
                if diff > 1e-6 {
                    num_differences += 1;
                    if num_differences < 10 {
                        println!(
                            "Difference at index {}: CPU={}, CUDA={}, diff={}",
                            i,
                            cpu_val.to_f32(),
                            cuda_val.to_f32(),
                            diff
                        );
                    }
                }
            }
        }

        // FP8 quantization should be deterministic, so we expect very few differences
        assert!(
            num_differences < 5,
            "Too many differences between CPU and CUDA quantization: {}",
            num_differences
        );

        // Compare scales
        let cpu_scales_vec = cpu_scales.to_vec1::<f32>()?;
        let cuda_scales_vec = cuda_scales_cpu.to_vec1::<f32>()?;

        assert_eq!(cpu_scales_vec.len(), cuda_scales_vec.len());

        for (i, (cpu_scale, cuda_scale)) in cpu_scales_vec
            .iter()
            .zip(cuda_scales_vec.iter())
            .enumerate()
        {
            let scale_diff = (cpu_scale - cuda_scale).abs();
            assert!(
                scale_diff < 1e-6,
                "Scale difference at index {}: CPU={}, CUDA={}, diff={}",
                i,
                cpu_scale,
                cuda_scale,
                scale_diff
            );
        }

        // Also test that dequantization gives the same results
        let cpu_dequant = fp8_vector_dequantize(&cpu_quantized, &cpu_scales, DType::F32)?;
        let cuda_dequant =
            fp8_vector_dequantize(&cuda_quantized_cpu, &cuda_scales_cpu, DType::F32)?;

        let cpu_dequant_vec = cpu_dequant.to_vec1::<f32>()?;
        let cuda_dequant_vec = cuda_dequant.to_vec1::<f32>()?;

        let mut max_dequant_diff = 0f32;
        for (cpu_val, cuda_val) in cpu_dequant_vec.iter().zip(cuda_dequant_vec.iter()) {
            let diff = (cpu_val - cuda_val).abs();
            max_dequant_diff = max_dequant_diff.max(diff);
        }

        assert!(
            max_dequant_diff < 1e-5,
            "Max dequantization difference too large: {}",
            max_dequant_diff
        );

        Ok(())
    }
}
