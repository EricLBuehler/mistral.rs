#[cfg(feature = "cuda")]
use candle_core::from_storage_no_op;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, DType, Result, Tensor, WithDType};
use float8::F8E4M3;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

struct Fp8BlockwiseDequantize {
    weight_block_size: Vec<usize>,
    out_ty: DType,
}

impl Fp8BlockwiseDequantize {
    fn dispatch_dequant_blockwise<T: WithDType>(
        &self,
        weight: &[F8E4M3],
        scale: &[f32],
        weight_l: &candle_core::Layout,
        scale_l: &candle_core::Layout,
    ) -> candle_core::Result<Vec<T>> {
        let grid_y = weight_l.dim(0)?.div_ceil(self.weight_block_size[0]);
        let grid_x = weight_l.dim(1)?.div_ceil(self.weight_block_size[1]);

        let res = vec![T::zero(); weight.len()];

        (0..grid_y).into_par_iter().for_each(|y| {
            (0..grid_x).into_par_iter().for_each(|x| {
                let res_ptr = res.as_ptr() as *mut T;

                let scale = scale[y * scale_l.stride()[0] + x];

                let start_y = y * self.weight_block_size[0];
                let end_y = start_y + self.weight_block_size[0];

                let start_x = x * self.weight_block_size[1];
                let end_x = start_x + self.weight_block_size[1];

                for weight_y in start_y..end_y {
                    if weight_y >= weight_l.dims()[0] {
                        break;
                    }

                    let row_offset = weight_y * weight_l.stride()[0];
                    for weight_x in start_x..end_x {
                        if weight_x >= weight_l.dims()[1] {
                            break;
                        }

                        let weight_pos = row_offset + weight_x;

                        // SAFETY: We know each thread will only update indepedant values!
                        unsafe {
                            *res_ptr.wrapping_add(weight_pos) =
                                T::from_f64((weight[weight_pos].to_f32() * scale) as f64);
                        }
                    }
                }
            });
        });

        Ok(res)
    }
}

impl CustomOp2 for Fp8BlockwiseDequantize {
    fn name(&self) -> &'static str {
        "fp8-blockwise-dequantize"
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
            candle_core::bail!("Expected F8E4M3 weight!");
        };
        if weight_l.start_offset() != 0 || !weight_l.is_contiguous() {
            candle_core::bail!("Expected weight to have start offset 0, continuous");
        }
        if scale_l.start_offset() != 0 || !scale_l.is_contiguous() {
            candle_core::bail!("Expected scales to have start offset 0, continuous");
        }
        if weight_l.dims().len() != 2 {
            candle_core::bail!("Expected weight to be rank 2");
        }
        if scale_l.dims().len() != 2 || self.weight_block_size.len() != 2 {
            candle_core::bail!("Expected scale to be rank 2");
        }

        match self.out_ty {
            DType::F32 => Ok((
                CpuStorage::F32(self.dispatch_dequant_blockwise(weight, scale, weight_l, scale_l)?),
                weight_l.shape().clone(),
            )),
            DType::BF16 => Ok((
                CpuStorage::BF16(
                    self.dispatch_dequant_blockwise(weight, scale, weight_l, scale_l)?,
                ),
                weight_l.shape().clone(),
            )),
            DType::F16 => Ok((
                CpuStorage::F16(self.dispatch_dequant_blockwise(weight, scale, weight_l, scale_l)?),
                weight_l.shape().clone(),
            )),
            other => candle_core::bail!("unexpected out type of fp8 blockwise dequant {other:?}"),
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

        use crate::{blockwise_fp8::ffi, utils::slice_ptr};

        if !ffi::HAVE_BLOCKWISE_DEQUANT_KERNELS {
            candle_core::bail!("Do not have blockwise FP8 dequant kernels.");
        }

        if weight_l.start_offset() != 0 || !weight_l.is_contiguous() {
            candle_core::bail!("Expected weight to have start offset 0, continuous");
        }
        if scale_l.start_offset() != 0 || !scale_l.is_contiguous() {
            candle_core::bail!("Expected scales to have start offset 0, continuous");
        }
        if weight_l.dims().len() != 2 {
            candle_core::bail!("Expected weight to be rank 2");
        }
        if scale_l.dims().len() != 2 || self.weight_block_size.len() != 2 {
            candle_core::bail!("Expected scale to be rank 2");
        }

        let dev = weight_s.device();

        let (weight, _weight_guard) =
            slice_ptr(weight_s.as_cuda_slice::<F8E4M3>()?, weight_l.start_offset());
        let (scale, _scale_guard) =
            slice_ptr(scale_s.as_cuda_slice::<f32>()?, scale_l.start_offset());

        let weight_height = weight_l.dim(0)? as i32;
        let weight_block_size_x = self.weight_block_size[0] as i32;
        let weight_width = weight_l.dim(1)? as i32;
        let weight_block_size_y = self.weight_block_size[1] as i32;
        let scale_stride = scale_l.stride()[0] as i32;
        let weight_row_stride = weight_l.stride()[0] as i32;

        let res = match self.out_ty {
            DType::F32 => {
                let output = weight_s
                    .device()
                    .alloc_zeros::<f32>(weight_l.shape().elem_count())?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    ffi::launch_dequant_fp8_blockwise_kernel_f32(
                        weight as *const _,
                        scale as *const _,
                        output_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, weight_s.device().clone())
            }
            DType::F16 => {
                let output = weight_s
                    .device()
                    .alloc_zeros::<f16>(weight_l.shape().elem_count())?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    ffi::launch_dequant_fp8_blockwise_kernel_f16(
                        weight as *const _,
                        scale as *const _,
                        output_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, weight_s.device().clone())
            }
            DType::BF16 => {
                let output = weight_s
                    .device()
                    .alloc_zeros::<bf16>(weight_l.shape().elem_count())?;
                let (output_ptr, output_guard) = slice_ptr(&output, 0);
                unsafe {
                    ffi::launch_dequant_fp8_blockwise_kernel_bf16(
                        weight as *const _,
                        scale as *const _,
                        output_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
                drop(output_guard);
                CudaStorage::wrap_cuda_slice(output, weight_s.device().clone())
            }
            other => candle_core::bail!("unexpected out type of fp8 blockwise dequant {other:?}"),
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

        if weight_l.start_offset() != 0
            || !weight_l.is_contiguous()
            || weight_s.dtype() != DType::F8E4M3
        {
            candle_core::bail!("Expected f8e4m3 weight to have start offset 0, continuous");
        }
        if scale_l.start_offset() != 0 || !scale_l.is_contiguous() || scale_s.dtype() != DType::F32
        {
            candle_core::bail!("Expected f32 scales to have start offset 0, continuous");
        }
        if weight_l.dims().len() != 2 {
            candle_core::bail!("Expected weight to be rank 2");
        }
        if scale_l.dims().len() != 2 || self.weight_block_size.len() != 2 {
            candle_core::bail!("Expected scale to be rank 2");
        }

        let command_buffer = weight_s.device().command_buffer()?;
        command_buffer.set_label("dequant-blockwise-fp8");

        let device = weight_s.device();

        let out_shape = weight_l.shape().clone();

        let output = device.new_buffer(
            out_shape.elem_count(),
            weight_s.dtype(),
            "dequant-blockwise-fp8",
        )?;

        let weight_height = weight_l.dim(0)? as u32;
        let weight_block_size_x = self.weight_block_size[0] as u32;
        let weight_width = weight_l.dim(1)? as u32;
        let weight_block_size_y = self.weight_block_size[1] as u32;
        let scale_stride = scale_l.stride()[0] as u32;
        let weight_row_stride = weight_l.stride()[0] as u32;

        crate::metal_kernels::call_dequant_blockwise_fp8(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            self.out_ty,
            weight_s.buffer(),
            scale_s.buffer(),
            &output,
            weight_height,
            weight_width,
            weight_row_stride,
            scale_stride,
            weight_block_size_y,
            weight_block_size_x,
        )
        .map_err(candle_core::Error::wrap)?;

        let newstorage = candle_core::MetalStorage::new(
            output,
            device.clone(),
            out_shape.elem_count(),
            self.out_ty,
        );
        Ok((newstorage, out_shape))
    }
}

/// FP8 blockwise dequantize.
/// - Expects weight to be fp8
/// - Expects inv_scales to be f32
/// - weight * inv_scale = dequantized
pub fn fp8_blockwise_dequantize(
    weight: &Tensor,
    inv_scales: &Tensor,
    weight_block_size: Vec<usize>,
    out_ty: DType,
) -> Result<Tensor> {
    inv_scales.apply_op2_no_bwd(
        weight,
        &Fp8BlockwiseDequantize {
            weight_block_size,
            out_ty,
        },
    )
}

#[allow(dead_code)]
struct Fp8BlockwiseQuantize {
    weight_block_size: Vec<usize>,
}

impl Fp8BlockwiseQuantize {
    #[allow(dead_code)]
    fn dispatch_quant_blockwise<T: WithDType>(
        &self,
        input: &[T],
        input_l: &candle_core::Layout,
    ) -> candle_core::Result<(Vec<F8E4M3>, Vec<f32>)> {
        let grid_y = input_l.dim(0)?.div_ceil(self.weight_block_size[0]);
        let grid_x = input_l.dim(1)?.div_ceil(self.weight_block_size[1]);

        let weight = vec![F8E4M3::from_f32(0.0); input.len()];
        let scale = vec![0f32; grid_y * grid_x];

        (0..grid_y).into_par_iter().for_each(|y| {
            (0..grid_x).into_par_iter().for_each(|x| {
                let weight_ptr = weight.as_ptr() as *mut F8E4M3;
                let scale_ptr = scale.as_ptr() as *mut f32;

                let start_y = y * self.weight_block_size[0];
                let end_y = start_y + self.weight_block_size[0];

                let start_x = x * self.weight_block_size[1];
                let end_x = start_x + self.weight_block_size[1];

                // Find max absolute value in block
                let mut max_abs = 0f32;
                for weight_y in start_y..end_y {
                    if weight_y >= input_l.dims()[0] {
                        break;
                    }

                    let row_offset = weight_y * input_l.stride()[0];
                    for weight_x in start_x..end_x {
                        if weight_x >= input_l.dims()[1] {
                            break;
                        }

                        let pos = row_offset + weight_x;
                        let val = input[pos].to_f64() as f32;
                        let abs_val = val.abs();
                        if abs_val > max_abs {
                            max_abs = abs_val;
                        }
                    }
                }

                // Calculate scale
                let block_scale = if max_abs > 0.0 {
                    max_abs / 448.0
                } else {
                    1e-12
                };

                // SAFETY: We know each thread will only update independent values!
                unsafe {
                    *scale_ptr.wrapping_add(y * grid_x + x) = block_scale;
                }

                // Quantize values
                for weight_y in start_y..end_y {
                    if weight_y >= input_l.dims()[0] {
                        break;
                    }

                    let row_offset = weight_y * input_l.stride()[0];
                    for weight_x in start_x..end_x {
                        if weight_x >= input_l.dims()[1] {
                            break;
                        }

                        let pos = row_offset + weight_x;
                        let val = input[pos].to_f64() as f32;
                        let scaled_val = (val / block_scale).clamp(-448.0, 448.0);

                        // SAFETY: We know each thread will only update independent values!
                        unsafe {
                            *weight_ptr.wrapping_add(pos) = F8E4M3::from_f32(scaled_val);
                        }
                    }
                }
            });
        });

        Ok((weight, scale))
    }
}

impl CustomOp1 for Fp8BlockwiseQuantize {
    fn name(&self) -> &'static str {
        "fp8-blockwise-quantize"
    }

    fn cpu_fwd(
        &self,
        input_s: &candle_core::CpuStorage,
        input_l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }
        if input_l.dims().len() != 2 {
            candle_core::bail!("Expected input to be rank 2");
        }
        if self.weight_block_size.len() != 2 {
            candle_core::bail!("Expected weight_block_size to have length 2");
        }

        let grid_y = input_l.dim(0)?.div_ceil(self.weight_block_size[0]);
        let grid_x = input_l.dim(1)?.div_ceil(self.weight_block_size[1]);

        let (weight, scale) = match input_s {
            CpuStorage::F32(input) => self.dispatch_quant_blockwise(input, input_l)?,
            CpuStorage::F16(input) => self.dispatch_quant_blockwise(input, input_l)?,
            CpuStorage::BF16(input) => self.dispatch_quant_blockwise(input, input_l)?,
            other => candle_core::bail!("unexpected input type for fp8 blockwise quant: {other:?}"),
        };

        // Return both weight and scale tensors packed into a single storage
        // We'll need to unpack them after the op
        let mut packed = Vec::with_capacity(weight.len() + scale.len());
        packed.extend_from_slice(&weight);

        // Convert scale to F8E4M3 for storage (will convert back when unpacking)
        for &s in &scale {
            packed.push(F8E4M3::from_f32(s));
        }

        Ok((
            CpuStorage::F8E4M3(packed),
            candle_core::Shape::from_dims(&[
                input_l.dims()[0] + grid_y,
                input_l.dims()[1].max(grid_x),
            ]),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        input_s: &candle_core::CudaStorage,
        input_l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::{backend::BackendStorage, CudaStorage};
        use half::{bf16, f16};

        use crate::{blockwise_fp8::ffi, utils::slice_ptr};

        if !ffi::HAVE_BLOCKWISE_QUANT_KERNELS {
            candle_core::bail!("Do not have blockwise FP8 quant kernels.");
        }

        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }
        if input_l.dims().len() != 2 {
            candle_core::bail!("Expected input to be rank 2");
        }
        if self.weight_block_size.len() != 2 {
            candle_core::bail!("Expected weight_block_size to have length 2");
        }

        let dev = input_s.device();

        let weight_height = input_l.dim(0)? as i32;
        let weight_block_size_y = self.weight_block_size[0] as i32;
        let weight_width = input_l.dim(1)? as i32;
        let weight_block_size_x = self.weight_block_size[1] as i32;
        let weight_row_stride = input_l.stride()[0] as i32;

        let grid_y = input_l.dim(0)?.div_ceil(self.weight_block_size[0]);
        let grid_x = input_l.dim(1)?.div_ceil(self.weight_block_size[1]);
        let scale_stride = grid_x as i32;

        // Allocate output buffers
        let weight_output = dev.alloc_zeros::<F8E4M3>(input_l.shape().elem_count())?;
        let scale_output = dev.alloc_zeros::<f32>(grid_y * grid_x)?;

        let (weight_ptr, weight_guard) = slice_ptr(&weight_output, 0);
        let (scale_ptr, scale_guard) = slice_ptr(&scale_output, 0);

        match input_s.dtype() {
            DType::F32 => {
                let (input, _input_guard) =
                    slice_ptr(input_s.as_cuda_slice::<f32>()?, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_f32(
                        input as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }
            DType::F16 => {
                let (input, _input_guard) =
                    slice_ptr(input_s.as_cuda_slice::<f16>()?, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_f16(
                        input as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }
            DType::BF16 => {
                let (input, _input_guard) =
                    slice_ptr(input_s.as_cuda_slice::<bf16>()?, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_bf16(
                        input as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }
            other => candle_core::bail!("unexpected input type for fp8 blockwise quant: {other:?}"),
        }

        drop(weight_guard);
        drop(scale_guard);

        // Return just the weight tensor - we'll handle scale separately
        let res = CudaStorage::wrap_cuda_slice(weight_output, input_s.device().clone());
        Ok((res, input_l.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        _input_s: &candle_core::MetalStorage,
        _input_l: &candle_core::Layout,
    ) -> Result<(candle_core::MetalStorage, candle_core::Shape)> {
        candle_core::bail!("FP8 blockwise quantization not yet implemented for Metal");
    }
}

/// FP8 blockwise quantize.
/// - Expects input to be f32, f16, or bf16
/// - Returns a tuple of (quantized_weight, scales)
/// - quantized_weight is fp8
/// - scales is f32
pub fn fp8_blockwise_quantize(
    #[allow(unused_variables)] input: &Tensor,
    #[allow(unused_variables)] weight_block_size: Vec<usize>,
) -> Result<(Tensor, Tensor)> {
    // Since CustomOp1 only returns a single tensor, we need a different approach
    // Let's implement this using the CUDA kernels directly
    #[cfg(feature = "cuda")]
    {
        use candle_core::{CudaStorage, Device, Storage};
        use half::{bf16, f16};

        use crate::{blockwise_fp8::ffi, utils::slice_ptr};

        if !matches!(input.device(), Device::Cuda(_)) {
            candle_core::bail!("FP8 blockwise quantization only supported on CUDA for now");
        }

        if !ffi::HAVE_BLOCKWISE_QUANT_KERNELS {
            candle_core::bail!("Do not have blockwise FP8 quant kernels.");
        }

        let input_l = input.layout();
        if input_l.start_offset() != 0 || !input_l.is_contiguous() {
            candle_core::bail!("Expected input to have start offset 0, continuous");
        }
        if input.dims().len() != 2 {
            candle_core::bail!("Expected input to be rank 2");
        }
        if weight_block_size.len() != 2 {
            candle_core::bail!("Expected weight_block_size to have length 2");
        }

        let dev = match input.device() {
            Device::Cuda(dev) => dev,
            _ => unreachable!(),
        };

        let weight_height = input.dim(0)? as i32;
        let weight_block_size_y = weight_block_size[0] as i32;
        let weight_width = input.dim(1)? as i32;
        let weight_block_size_x = weight_block_size[1] as i32;
        let weight_row_stride = input_l.stride()[0] as i32;

        let grid_y = input.dim(0)?.div_ceil(weight_block_size[0]);
        let grid_x = input.dim(1)?.div_ceil(weight_block_size[1]);
        let scale_stride = grid_x as i32;

        // Allocate output buffers
        let weight_output = dev.alloc_zeros::<F8E4M3>(input.shape().elem_count())?;
        let scale_output = dev.alloc_zeros::<f32>(grid_y * grid_x)?;

        let (weight_ptr, _weight_guard) = slice_ptr(&weight_output, 0);
        let (scale_ptr, _scale_guard) = slice_ptr(&scale_output, 0);

        match input.dtype() {
            DType::F32 => {
                let input_storage = input.storage_and_layout().0;
                let input_s = match &*input_storage {
                    Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f32>()?,
                    _ => candle_core::bail!("Expected CUDA storage"),
                };
                let (input_ptr, _input_guard) = slice_ptr(&input_s, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_f32(
                        input_ptr as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
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
                let (input_ptr, _input_guard) = slice_ptr(&input_s, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_f16(
                        input_ptr as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
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
                let (input_ptr, _input_guard) = slice_ptr(&input_s, input_l.start_offset());
                unsafe {
                    ffi::launch_quant_fp8_blockwise_kernel_bf16(
                        input_ptr as *const _,
                        weight_ptr as *mut _,
                        scale_ptr as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }
            other => candle_core::bail!("unexpected input type for fp8 blockwise quant: {other:?}"),
        }

        // Drop guards before moving the buffers
        drop(_weight_guard);
        drop(_scale_guard);

        // Create weight tensor by wrapping the CUDA storage
        let weight_storage = CudaStorage::wrap_cuda_slice(weight_output, dev.clone());
        let weight =
            from_storage_no_op(Storage::Cuda(weight_storage), input.shape().clone(), false);

        // Create scale tensor
        let scale_storage = CudaStorage::wrap_cuda_slice(scale_output, dev.clone());
        let scale = from_storage_no_op(
            Storage::Cuda(scale_storage),
            candle_core::Shape::from_dims(&[grid_y, grid_x]),
            false,
        );

        Ok((weight, scale))
    }

    #[cfg(not(feature = "cuda"))]
    {
        candle_core::bail!("FP8 blockwise quantization requires CUDA feature");
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::{Linear, Module};
    use half::bf16;
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

    use crate::{blockwise_fp8::ops, safetensors::MmapedSafetensors};

    #[test]
    fn test_fp8_blockwise_dequant() -> Result<()> {
        let dev = &Device::Cpu;
        let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
        let weight_block_size = vec![2, 2];
        let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

        let dequant =
            ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::F32)?;

        let res = dequant.to_vec2::<f32>()?;
        assert_eq!(
            res,
            vec![
                vec![0., 0., 1., 1., 2.],
                vec![0., 0., 1., 1., 2.],
                vec![3., 3., 4., 4., 5.],
                vec![3., 3., 4., 4., 5.],
                vec![6., 6., 7., 7., 8.],
            ]
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_blockwise_dequant_cuda() -> Result<()> {
        let truth = {
            let dev = &Device::Cpu;
            let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
            let weight_block_size = vec![2, 2];
            let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

            let dequant =
                ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::F32)?;

            dequant.to_vec2::<f32>()?
        };
        let test = {
            let dev = &Device::new_cuda(0)?;
            // Create FP8 weight by first creating on CPU then moving to CUDA
            let weight_cpu = Tensor::ones((5, 5), DType::F8E4M3, &Device::Cpu)?;
            let weight = weight_cpu.to_device(dev)?;
            let weight_block_size = vec![2, 2];
            let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

            let dequant =
                ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::F32)?;

            dequant.to_vec2::<f32>()?
        };

        assert_eq!(test, truth);
        assert_eq!(
            test,
            vec![
                vec![0., 0., 1., 1., 2.],
                vec![0., 0., 1., 1., 2.],
                vec![3., 3., 4., 4., 5.],
                vec![3., 3., 4., 4., 5.],
                vec![6., 6., 7., 7., 8.],
            ]
        );

        Ok(())
    }

    #[test]
    fn test_fp8_blockwise_dequant_bf16() -> Result<()> {
        let dev = &Device::Cpu;
        let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
        let weight_block_size = vec![2, 2];
        let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

        let dequant =
            ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::BF16)?;

        let res = dequant.to_vec2::<bf16>()?;
        assert_eq!(
            res,
            vec![
                vec![
                    bf16::from_f32(0.),
                    bf16::from_f32(0.),
                    bf16::from_f32(1.),
                    bf16::from_f32(1.),
                    bf16::from_f32(2.)
                ],
                vec![
                    bf16::from_f32(0.),
                    bf16::from_f32(0.),
                    bf16::from_f32(1.),
                    bf16::from_f32(1.),
                    bf16::from_f32(2.)
                ],
                vec![
                    bf16::from_f32(3.),
                    bf16::from_f32(3.),
                    bf16::from_f32(4.),
                    bf16::from_f32(4.),
                    bf16::from_f32(5.)
                ],
                vec![
                    bf16::from_f32(3.),
                    bf16::from_f32(3.),
                    bf16::from_f32(4.),
                    bf16::from_f32(4.),
                    bf16::from_f32(5.)
                ],
                vec![
                    bf16::from_f32(6.),
                    bf16::from_f32(6.),
                    bf16::from_f32(7.),
                    bf16::from_f32(7.),
                    bf16::from_f32(8.)
                ],
            ]
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_blockwise_dequant_cuda_bf16() -> Result<()> {
        let truth = {
            let dev = &Device::Cpu;
            let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
            let weight_block_size = vec![2, 2];
            let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

            let dequant = ops::fp8_blockwise_dequantize(
                &weight,
                &inv_scales,
                weight_block_size,
                DType::BF16,
            )?;

            dequant.to_vec2::<bf16>()?
        };
        let test = {
            let dev = &Device::new_cuda(0)?;
            // Create FP8 weight by first creating on CPU then moving to CUDA
            let weight_cpu = Tensor::ones((5, 5), DType::F8E4M3, &Device::Cpu)?;
            let weight = weight_cpu.to_device(dev)?;
            let weight_block_size = vec![2, 2];
            let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

            let dequant = ops::fp8_blockwise_dequantize(
                &weight,
                &inv_scales,
                weight_block_size,
                DType::BF16,
            )?;

            dequant.to_vec2::<bf16>()?
        };

        assert_eq!(test, truth);
        assert_eq!(
            test,
            vec![
                vec![
                    bf16::from_f32(0.),
                    bf16::from_f32(0.),
                    bf16::from_f32(1.),
                    bf16::from_f32(1.),
                    bf16::from_f32(2.)
                ],
                vec![
                    bf16::from_f32(0.),
                    bf16::from_f32(0.),
                    bf16::from_f32(1.),
                    bf16::from_f32(1.),
                    bf16::from_f32(2.)
                ],
                vec![
                    bf16::from_f32(3.),
                    bf16::from_f32(3.),
                    bf16::from_f32(4.),
                    bf16::from_f32(4.),
                    bf16::from_f32(5.)
                ],
                vec![
                    bf16::from_f32(3.),
                    bf16::from_f32(3.),
                    bf16::from_f32(4.),
                    bf16::from_f32(4.),
                    bf16::from_f32(5.)
                ],
                vec![
                    bf16::from_f32(6.),
                    bf16::from_f32(6.),
                    bf16::from_f32(7.),
                    bf16::from_f32(7.),
                    bf16::from_f32(8.)
                ],
            ]
        );

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_blockwise_quant_dequant_roundtrip() -> Result<()> {
        let dev = &Device::new_cuda(0)?;

        // Create test input
        let input = Tensor::randn(0f32, 2f32, (8, 8), dev)?;
        let weight_block_size = vec![4, 4];

        // Quantize
        let (quantized, scales) = ops::fp8_blockwise_quantize(&input, weight_block_size.clone())?;

        // Verify shapes
        assert_eq!(quantized.shape(), input.shape());
        assert_eq!(scales.dims2()?, (2, 2)); // 8/4 = 2 blocks in each dimension

        // Dequantize
        let dequantized =
            ops::fp8_blockwise_dequantize(&quantized, &scales, weight_block_size, input.dtype())?;

        // Check that shapes match
        assert_eq!(dequantized.shape(), input.shape());

        // The values won't be exactly the same due to quantization loss,
        // but they should be reasonably close
        let input_vec = input.to_vec2::<f32>()?;
        let dequant_vec = dequantized.to_vec2::<f32>()?;

        let mut max_error = 0f32;
        for (row_in, row_out) in input_vec.iter().zip(dequant_vec.iter()) {
            for (val_in, val_out) in row_in.iter().zip(row_out.iter()) {
                let error = (val_in - val_out).abs();
                max_error = max_error.max(error);
            }
        }

        // FP8 E4M3 has limited precision, so we expect some error
        // but it should be reasonable
        assert!(max_error < 0.16, "Max error {} is too large", max_error);

        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_blockwise_fp8_gemm() -> Result<()> {
        let dev = Device::cuda_if_available(0)?;

        let api = ApiBuilder::new().with_progress(true).build().unwrap();
        let api = api.repo(Repo::with_revision(
            "EricB/mistralrs_tests".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let filename = api.get("test_fp8.safetensors").unwrap();
        let vb = unsafe { MmapedSafetensors::new(filename)? };

        let weight = vb.load("weight", &dev, None)?;
        assert_eq!((7168, 2048), weight.dims2()?);
        assert_eq!(DType::F8E4M3, weight.dtype());

        let scale = vb.load("scale", &dev, None)?;
        assert_eq!((56, 16), scale.dims2()?);
        assert_eq!(DType::F32, scale.dtype());

        let weight_block_size = vec![128, 128];

        // in dim is 2048.
        let xs = Tensor::randn(0f32, 1f32, (32, 2048), &dev)?.to_dtype(DType::BF16)?;

        let truth = {
            let weight_dq =
                ops::fp8_blockwise_dequantize(&weight, &scale, weight_block_size, DType::BF16)?;

            let lin_dq = Linear::new(weight_dq, None);
            lin_dq.forward(&xs)?
        };

        // TODO: will be adding real blockwise fp8 gemm shortly ;)
        assert_eq!((32, 7168), truth.dims2()?);

        Ok(())
    }
}
