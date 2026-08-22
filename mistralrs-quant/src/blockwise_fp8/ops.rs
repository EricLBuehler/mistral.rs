use candle_core::{CpuStorage, CustomOp1, CustomOp2, DType, Result, Tensor, WithDType};
use float8::F8E4M3;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
use super::ffi;
#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
use std::{
    borrow::Cow,
    collections::HashMap,
    ffi::CStr,
    sync::{Arc, Mutex, OnceLock},
};

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
const FP8_ALIGNMENT_BYTES: usize = 16;
#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
const CUDA_STREAM_PER_THREAD_HANDLE: usize = 2;
#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
pub(super) const FP8_BLOCK_SIZE: usize = 128;
#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
const CUTLASS_FP8_N_ALIGNMENT: usize = 16;
#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
const CUTLASS_OUTPUT_F16: i32 = 0;
#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
const CUTLASS_OUTPUT_BF16: i32 = 1;

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

        let res = vec![T::zero(); weight_l.shape().elem_count()];
        let output_width = weight_l.dim(1)?;

        (0..grid_y).into_par_iter().for_each(|y| {
            (0..grid_x).into_par_iter().for_each(|x| {
                let res_ptr = res.as_ptr() as *mut T;

                let scale = scale[scale_l.start_offset() + y * scale_l.stride()[0] + x];

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

                        let weight_pos = weight_l.start_offset() + row_offset + weight_x;
                        let output_pos = weight_y * output_width + weight_x;

                        // SAFETY: We know each thread will only update indepedant values!
                        unsafe {
                            *res_ptr.wrapping_add(output_pos) =
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
        if !weight_l.is_contiguous() {
            candle_core::bail!("Expected weight to be continuous");
        }
        if !scale_l.is_contiguous() {
            candle_core::bail!("Expected scales to be continuous");
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

        if !weight_l.is_contiguous() {
            candle_core::bail!("Expected weight to be continuous");
        }
        if !scale_l.is_contiguous() {
            candle_core::bail!("Expected scales to be continuous");
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
        let weight_block_size_y = self.weight_block_size[0] as i32;
        let weight_width = weight_l.dim(1)? as i32;
        let weight_block_size_x = self.weight_block_size[1] as i32;
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

        let encoder = weight_s.device().command_encoder()?;
        encoder.set_label("dequant-blockwise-fp8");

        let device = weight_s.device();

        let out_shape = weight_l.shape().clone();

        let output = device.new_buffer(
            out_shape.elem_count(),
            weight_s.dtype(),
            "dequant-blockwise-fp8",
        )?;

        let weight_height = weight_l.dim(0)? as u32;
        let weight_block_size_y = self.weight_block_size[0] as u32;
        let weight_width = weight_l.dim(1)? as u32;
        let weight_block_size_x = self.weight_block_size[1] as u32;
        let scale_stride = scale_l.stride()[0] as u32;
        let weight_row_stride = weight_l.stride()[0] as u32;

        crate::metal_kernels::call_dequant_blockwise_fp8(
            device.device(),
            &encoder,
            crate::metal_kernels::Kernels::global(),
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
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
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
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
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
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
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
        let weight = Tensor::from((Storage::Cuda(weight_storage), input.shape().clone()));

        // Create scale tensor
        let scale_storage = CudaStorage::wrap_cuda_slice(scale_output, dev.clone());
        let scale = Tensor::from((
            Storage::Cuda(scale_storage),
            candle_core::Shape::from_dims(&[grid_y, grid_x]),
        ));

        Ok((weight, scale))
    }

    #[cfg(not(feature = "cuda"))]
    {
        candle_core::bail!("FP8 blockwise quantization requires CUDA feature");
    }
}

/// FP8 blockwise matmul.
/// Computes output = input @ weight.T where weight is FP8 blockwise quantized.
/// - input: [M, K] in fp16/bf16
/// - weight: [N, K] in FP8 with blockwise scales
/// - scales: [N/block_y, K/block_x] in f32
/// - output: [M, N] in fp16/bf16
#[cfg(feature = "cuda")]
pub fn fp8_blockwise_matmul(
    input: &Tensor,
    weight: &Tensor,
    scales: &Tensor,
    weight_block_size: &[usize],
) -> Result<Tensor> {
    use candle_core::{CudaStorage, Device, Storage};
    use half::{bf16, f16};

    use crate::{blockwise_fp8::ffi, utils::slice_ptr};

    if !ffi::HAVE_BLOCKWISE_GEMM_KERNELS {
        candle_core::bail!("Do not have blockwise FP8 GEMM kernels.");
    }

    if !matches!(input.device(), Device::Cuda(_)) {
        candle_core::bail!("FP8 blockwise matmul only supported on CUDA");
    }

    let input = input.contiguous()?;
    let input = if input.layout().start_offset() * input.dtype().size_in_bytes() % 16 == 0 {
        input
    } else {
        input.copy()?
    };
    let weight = weight.contiguous()?;
    let scales = scales.contiguous()?;

    if input.dims().len() != 2 {
        candle_core::bail!("Expected input to be rank 2, got {:?}", input.dims());
    }
    if weight.dims().len() != 2 {
        candle_core::bail!("Expected weight to be rank 2, got {:?}", weight.dims());
    }
    if weight.dtype() != DType::F8E4M3 {
        candle_core::bail!("Expected FP8 weight, got {:?}", weight.dtype());
    }

    let m = input.dim(0)? as i32;
    let k = input.dim(1)? as i32;
    let n = weight.dim(0)? as i32;

    if weight.dim(1)? as i32 != k {
        candle_core::bail!(
            "Weight K dimension {} doesn't match input K dimension {}",
            weight.dim(1)?,
            k
        );
    }

    let dev = match input.device() {
        Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };

    let block_size_y = weight_block_size[0] as i32;
    let block_size_x = weight_block_size[1] as i32;
    let scale_row_stride = scales.dim(1)? as i32;

    let input_l = input.layout();
    let weight_l = weight.layout();
    let scales_l = scales.layout();

    let input_storage = input.storage_and_layout().0;
    let weight_storage = weight.storage_and_layout().0;
    let scales_storage = scales.storage_and_layout().0;

    let weight_s = match &*weight_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<F8E4M3>()?,
        _ => candle_core::bail!("Expected CUDA storage for weight"),
    };
    let scales_s = match &*scales_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("Expected CUDA storage for scales"),
    };

    let (weight_ptr, _weight_guard) = slice_ptr(weight_s, weight_l.start_offset());
    let (scales_ptr, _scales_guard) = slice_ptr(scales_s, scales_l.start_offset());

    match input.dtype() {
        DType::F16 => {
            let output = dev.alloc_zeros::<f16>((m * n) as usize)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };

            {
                let (output_ptr, _output_guard) = slice_ptr(&output, 0);
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());

                unsafe {
                    ffi::launch_fp8_matmul_f16(
                        input_ptr as *const _,
                        weight_ptr as *const _,
                        scales_ptr as *const _,
                        output_ptr as *mut _,
                        m,
                        n,
                        k,
                        scale_row_stride,
                        block_size_y,
                        block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }

            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                candle_core::Shape::from_dims(&[m as usize, n as usize]),
            )))
        }
        DType::BF16 => {
            let output = dev.alloc_zeros::<bf16>((m * n) as usize)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<bf16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };

            {
                let (output_ptr, _output_guard) = slice_ptr(&output, 0);
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());

                unsafe {
                    ffi::launch_fp8_matmul_bf16(
                        input_ptr as *const _,
                        weight_ptr as *const _,
                        scales_ptr as *const _,
                        output_ptr as *mut _,
                        m,
                        n,
                        k,
                        scale_row_stride,
                        block_size_y,
                        block_size_x,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }

            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                candle_core::Shape::from_dims(&[m as usize, n as usize]),
            )))
        }
        other => candle_core::bail!("Unsupported input dtype for FP8 matmul: {:?}", other),
    }
}

/// FP8 indexed MoE GEMM for gather_forward.
/// Computes indexed matmul for MoE where each token selects specific experts.
/// - input: [num_tokens, 1, K] or [num_tokens, topk, K] in fp16/bf16
/// - weights: [num_experts, N, K] in FP8 with blockwise scales
/// - scales: [num_experts, N/block_y, K/block_x] in f32
/// - indices: [num_tokens, topk] in i32
/// - output: [num_tokens, topk, N] in fp16/bf16
#[cfg(feature = "cuda")]
pub fn fp8_indexed_moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    scales: &Tensor,
    indices: &Tensor,
    weight_block_size: &[usize],
) -> Result<Tensor> {
    use candle_core::{CudaStorage, Device, Storage};
    use half::{bf16, f16};

    use crate::{blockwise_fp8::ffi, utils::slice_ptr};

    if !ffi::HAVE_BLOCKWISE_GEMM_KERNELS {
        candle_core::bail!("Do not have blockwise FP8 GEMM kernels.");
    }

    if !matches!(input.device(), Device::Cuda(_)) {
        candle_core::bail!("FP8 indexed MoE GEMM only supported on CUDA");
    }

    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let scales = scales.contiguous()?;
    let indices = indices.contiguous()?;

    // Determine input shape
    // Input can be [num_tokens, 1, K] or [num_tokens, topk, K]
    let (num_tokens, input_has_topk_dim, k) = if input.dims().len() == 3 {
        let dims = input.dims3()?;
        (dims.0, dims.1 > 1, dims.2)
    } else if input.dims().len() == 2 {
        let dims = input.dims2()?;
        (dims.0, false, dims.1)
    } else {
        candle_core::bail!("Expected input to be rank 2 or 3, got {:?}", input.dims());
    };

    // Get topk from indices
    let (indices_tokens, topk) = indices.dims2()?;
    if indices_tokens != num_tokens {
        candle_core::bail!(
            "Indices num_tokens {} doesn't match input num_tokens {}",
            indices_tokens,
            num_tokens
        );
    }

    // Weights shape: [num_experts, N, K]
    if weights.dims().len() != 3 {
        candle_core::bail!("Expected weights to be rank 3, got {:?}", weights.dims());
    }
    let (num_experts, n, weight_k) = weights.dims3()?;
    if weight_k != k {
        candle_core::bail!(
            "Weights K dimension {} doesn't match input K dimension {}",
            weight_k,
            k
        );
    }

    if weights.dtype() != DType::F8E4M3 {
        candle_core::bail!("Expected FP8 weights, got {:?}", weights.dtype());
    }

    let dev = match input.device() {
        Device::Cuda(dev) => dev,
        _ => unreachable!(),
    };

    let block_size_y = weight_block_size[0] as i32;
    let block_size_x = weight_block_size[1] as i32;

    // Scales shape should be [num_experts, N/block_y, K/block_x]
    let scale_row_stride = scales.dim(2)? as i32; // K/block_x

    let input_l = input.layout();
    let weights_l = weights.layout();
    let scales_l = scales.layout();
    let indices_l = indices.layout();

    let input_storage = input.storage_and_layout().0;
    let weights_storage = weights.storage_and_layout().0;
    let scales_storage = scales.storage_and_layout().0;
    let indices_storage = indices.storage_and_layout().0;

    let weights_s = match &*weights_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<F8E4M3>()?,
        _ => candle_core::bail!("Expected CUDA storage for weights"),
    };
    let scales_s = match &*scales_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("Expected CUDA storage for scales"),
    };
    let indices_s = match &*indices_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<u32>()?,
        _ => candle_core::bail!("Expected CUDA storage for indices"),
    };

    let (weights_ptr, _weights_guard) = slice_ptr(weights_s, weights_l.start_offset());
    let (scales_ptr, _scales_guard) = slice_ptr(scales_s, scales_l.start_offset());
    let (indices_ptr, _indices_guard) = slice_ptr(indices_s, indices_l.start_offset());

    match input.dtype() {
        DType::F16 => {
            let output = dev.alloc_zeros::<f16>(num_tokens * topk * n)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };

            {
                let (output_ptr, _output_guard) = slice_ptr(&output, 0);
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());

                unsafe {
                    ffi::launch_fp8_indexed_moe_gemm_f16(
                        input_ptr as *const _,
                        weights_ptr as *const _,
                        scales_ptr as *const _,
                        indices_ptr as *const _,
                        output_ptr as *mut _,
                        num_tokens as i32,
                        topk as i32,
                        num_experts as i32,
                        n as i32,
                        k as i32,
                        scale_row_stride,
                        block_size_y,
                        block_size_x,
                        input_has_topk_dim,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }

            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                candle_core::Shape::from_dims(&[num_tokens, topk, n]),
            )))
        }
        DType::BF16 => {
            let output = dev.alloc_zeros::<bf16>(num_tokens * topk * n)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<bf16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };

            {
                let (output_ptr, _output_guard) = slice_ptr(&output, 0);
                let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());

                unsafe {
                    ffi::launch_fp8_indexed_moe_gemm_bf16(
                        input_ptr as *const _,
                        weights_ptr as *const _,
                        scales_ptr as *const _,
                        indices_ptr as *const _,
                        output_ptr as *mut _,
                        num_tokens as i32,
                        topk as i32,
                        num_experts as i32,
                        n as i32,
                        k as i32,
                        scale_row_stride,
                        block_size_y,
                        block_size_x,
                        input_has_topk_dim,
                        dev.cuda_stream().cu_stream(),
                    )
                };
            }

            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                candle_core::Shape::from_dims(&[num_tokens, topk, n]),
            )))
        }
        other => candle_core::bail!(
            "Unsupported input dtype for FP8 indexed MoE GEMM: {:?}",
            other
        ),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn cutlass_fp8_blockwise_supported(
    weight: &Tensor,
    weight_scales: &Tensor,
    weight_block_size: &[usize],
) -> bool {
    #[cfg(not(has_cutlass_fp8_sm90_kernels))]
    {
        let _ = (weight, weight_scales, weight_block_size);
        false
    }

    #[cfg(has_cutlass_fp8_sm90_kernels)]
    {
        use candle_core::Device;

        if !ffi::HAVE_CUTLASS_FP8_SM90_KERNELS
            || weight_block_size != [FP8_BLOCK_SIZE, FP8_BLOCK_SIZE]
            || weight.dtype() != DType::F8E4M3
            || weight_scales.dtype() != DType::F32
            || !weight.is_contiguous()
            || !weight_scales.is_contiguous()
            || !weight.device().same_device(weight_scales.device())
            || !fp8_tensor_aligned(weight)
            || !fp8_tensor_aligned(weight_scales)
        {
            return false;
        }
        let [n, k] = weight.dims() else {
            return false;
        };
        if *n == 0 || *k == 0 || n % CUTLASS_FP8_N_ALIGNMENT != 0 || k % FP8_BLOCK_SIZE != 0 {
            return false;
        }
        if weight_scales.dims() != [n.div_ceil(FP8_BLOCK_SIZE), k / FP8_BLOCK_SIZE] {
            return false;
        }
        let Device::Cuda(dev) = weight.device() else {
            return false;
        };
        is_sm90(dev)
    }
}

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
pub(super) fn fp8_tensor_aligned(tensor: &Tensor) -> bool {
    tensor
        .layout()
        .start_offset()
        .checked_mul(tensor.dtype().size_in_bytes())
        .is_some_and(|offset| offset % FP8_ALIGNMENT_BYTES == 0)
}

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
static FP8_SM90_DEVICES: OnceLock<Mutex<HashMap<candle_core::cuda::DeviceId, bool>>> =
    OnceLock::new();

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
pub(super) fn is_sm90(dev: &candle_core::CudaDevice) -> bool {
    use candle_core::cuda::cudarc::driver::{result, sys};

    let devices = FP8_SM90_DEVICES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(supported) = devices.lock().unwrap().get(&dev.id()).copied() {
        return supported;
    }
    let device = dev.cuda_stream().context().cu_device();
    let major = unsafe {
        result::device::get_attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    };
    let minor = unsafe {
        result::device::get_attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    };
    let supported = matches!((major, minor), (Ok(9), Ok(0)));
    devices.lock().unwrap().insert(dev.id(), supported);
    supported
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
fn check_cutlass_status(operation: &str, status: i32) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe {
        let message = ffi::mistralrs_cutlass_fp8_error_string(status);
        if message.is_null() {
            Cow::Borrowed("unknown error")
        } else {
            CStr::from_ptr(message).to_string_lossy()
        }
    };
    let domain = if status < 0 { "CUDA" } else { "CUTLASS" };
    candle_core::bail!("{operation} failed: {message} ({domain} status {status})")
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
static PREPARED_CUTLASS_FP8_DEVICES: OnceLock<Mutex<HashMap<candle_core::cuda::DeviceId, i32>>> =
    OnceLock::new();

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
pub(super) fn prepare_cutlass_fp8(dev: &candle_core::CudaDevice) -> Result<i32> {
    use candle_core::cuda::cudarc::driver::{result, sys};

    dev.cuda_stream()
        .context()
        .bind_to_thread()
        .map_err(|error| {
            candle_core::Error::msg(format!("CUDA context binding failed: {error}"))
        })?;
    if !is_sm90(dev) {
        candle_core::bail!("CUTLASS FP8 provider requires an SM90 device")
    }
    let prepared = PREPARED_CUTLASS_FP8_DEVICES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut prepared = prepared.lock().unwrap();
    if let Some(sm_count) = prepared.get(&dev.id()) {
        return Ok(*sm_count);
    }
    let sm_count = unsafe {
        result::device::get_attribute(
            dev.cuda_stream().context().cu_device(),
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }
    .map_err(|error| {
        candle_core::Error::msg(format!("CUDA multiprocessor query failed: {error}"))
    })?;
    let status = unsafe { ffi::mistralrs_cutlass_fp8_blockwise_prepare() };
    check_cutlass_status("CUTLASS FP8 kernel preparation", status)?;
    prepared.insert(dev.id(), sm_count);
    Ok(sm_count)
}

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
#[derive(Eq, Hash, PartialEq)]
struct Fp8WorkspaceKey {
    device: candle_core::cuda::DeviceId,
    stream: usize,
    thread: Option<std::thread::ThreadId>,
    capacity: usize,
}

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
fn fp8_workspace_thread(stream: usize) -> Option<std::thread::ThreadId> {
    (stream == CUDA_STREAM_PER_THREAD_HANDLE).then(|| std::thread::current().id())
}

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
pub(super) struct Fp8Workspace {
    pub(super) slice: candle_core::cuda::cudarc::driver::CudaSlice<u8>,
}

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
type Fp8WorkspaceMap = Mutex<HashMap<Fp8WorkspaceKey, Arc<Mutex<Fp8Workspace>>>>;

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
static FP8_WORKSPACES: OnceLock<Fp8WorkspaceMap> = OnceLock::new();

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct CutlassWorkspaceRequirementsKey {
    m: i32,
    n: i32,
    k: i32,
    output_dtype: i32,
    sm_count: i32,
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
static CUTLASS_FP8_WORKSPACE_REQUIREMENTS: OnceLock<
    Mutex<HashMap<CutlassWorkspaceRequirementsKey, usize>>,
> = OnceLock::new();

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
fn cutlass_workspace_size(key: CutlassWorkspaceRequirementsKey) -> Result<usize> {
    let requirements =
        CUTLASS_FP8_WORKSPACE_REQUIREMENTS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(bytes) = requirements.lock().unwrap().get(&key).copied() {
        return Ok(bytes);
    }
    let mut bytes = 0usize;
    let status = unsafe {
        ffi::mistralrs_cutlass_fp8_blockwise_workspace_size(
            key.m,
            key.n,
            key.k,
            key.output_dtype,
            key.sm_count,
            &mut bytes,
        )
    };
    check_cutlass_status("CUTLASS FP8 workspace query", status)?;
    requirements.lock().unwrap().insert(key, bytes);
    Ok(bytes)
}

#[cfg(all(
    feature = "cuda",
    any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
))]
pub(super) fn fp8_workspace(
    dev: &candle_core::CudaDevice,
    bytes: usize,
    provider: &str,
) -> Result<Option<Arc<Mutex<Fp8Workspace>>>> {
    if bytes == 0 {
        return Ok(None);
    }
    let capacity = bytes.checked_next_power_of_two().ok_or_else(|| {
        candle_core::Error::msg(format!("{provider} FP8 workspace size overflow"))
    })?;
    let stream = dev.cuda_stream();
    let stream_handle = stream.cu_stream() as usize;
    let key = Fp8WorkspaceKey {
        device: dev.id(),
        stream: stream_handle,
        thread: fp8_workspace_thread(stream_handle),
        capacity,
    };
    let workspaces = FP8_WORKSPACES.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(workspace) = workspaces.lock().unwrap().get(&key).cloned() {
        return Ok(Some(workspace));
    }
    let capture_status = stream.capture_status().map_err(|error| {
        candle_core::Error::msg(format!("CUDA stream capture status query failed: {error}"))
    })?;
    if capture_status
        != candle_core::cuda::cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
    {
        candle_core::bail!(
            "{provider} FP8 workspace for this shape must be warmed before CUDA graph capture"
        )
    }
    let slice = unsafe { dev.alloc::<u8>(capacity)? };
    let workspace = Arc::new(Mutex::new(Fp8Workspace { slice }));
    let workspace = workspaces
        .lock()
        .unwrap()
        .entry(key)
        .or_insert_with(|| Arc::clone(&workspace))
        .clone();
    Ok(Some(workspace))
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
pub(crate) fn fp8_quantize_activation_cutlass(input: &Tensor) -> Result<(Tensor, Tensor)> {
    use candle_core::{CudaStorage, Device, Shape, Storage};
    use half::{bf16, f16};

    use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

    let Device::Cuda(dev) = input.device() else {
        candle_core::bail!("CUTLASS FP8 activation quantization requires CUDA")
    };
    let input = input.contiguous()?;
    let input = if fp8_tensor_aligned(&input) {
        input
    } else {
        input.copy()?
    };
    let (rows, cols) = input.dims2()?;
    if rows == 0 || cols == 0 || cols % FP8_BLOCK_SIZE != 0 {
        candle_core::bail!(
            "CUTLASS FP8 activation shape ({rows}, {cols}) requires nonzero dimensions and K divisible by 128"
        )
    }
    let rows_i32 = i32::try_from(rows)
        .map_err(|_| candle_core::Error::msg("FP8 activation row count exceeds i32"))?;
    let cols_i32 = i32::try_from(cols)
        .map_err(|_| candle_core::Error::msg("FP8 activation column count exceeds i32"))?;
    let scale_count = rows
        .checked_mul(cols / FP8_BLOCK_SIZE)
        .ok_or_else(|| candle_core::Error::msg("FP8 activation scale count overflow"))?;
    let _ = i32::try_from(scale_count)
        .map_err(|_| candle_core::Error::msg("FP8 activation scale count exceeds i32"))?;
    let _ = prepare_cutlass_fp8(dev)?;

    let stream = dev.cuda_stream();
    let mut quantized = unsafe { dev.alloc::<F8E4M3>(rows * cols)? };
    let mut scales = unsafe { dev.alloc::<f32>(scale_count)? };
    let (quantized_ptr, quantized_guard) = slice_ptr_mut_on_stream(&mut quantized, 0, &stream);
    let (scales_ptr, scales_guard) = slice_ptr_mut_on_stream(&mut scales, 0, &stream);
    let (input_storage, input_layout) = input.storage_and_layout();
    let status = match input.dtype() {
        DType::F16 => {
            let Storage::Cuda(input_storage) = &*input_storage else {
                unreachable!()
            };
            let input = input_storage.as_cuda_slice::<f16>()?;
            let (input_ptr, input_guard) =
                slice_ptr_on_stream(input, input_layout.start_offset(), &stream);
            let status = unsafe {
                ffi::mistralrs_fp8_quantize_activation_f16(
                    input_ptr as *const f16,
                    quantized_ptr as *mut F8E4M3,
                    scales_ptr as *mut f32,
                    rows_i32,
                    cols_i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            drop(input_guard);
            status
        }
        DType::BF16 => {
            let Storage::Cuda(input_storage) = &*input_storage else {
                unreachable!()
            };
            let input = input_storage.as_cuda_slice::<bf16>()?;
            let (input_ptr, input_guard) =
                slice_ptr_on_stream(input, input_layout.start_offset(), &stream);
            let status = unsafe {
                ffi::mistralrs_fp8_quantize_activation_bf16(
                    input_ptr as *const bf16,
                    quantized_ptr as *mut F8E4M3,
                    scales_ptr as *mut f32,
                    rows_i32,
                    cols_i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                )
            };
            drop(input_guard);
            status
        }
        dtype => candle_core::bail!(
            "CUTLASS FP8 activation quantization requires F16 or BF16, got {dtype:?}"
        ),
    };
    check_cutlass_status("FP8 activation quantization", status)?;
    drop((quantized_guard, scales_guard));

    let quantized = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(quantized, dev.clone())),
        Shape::from_dims(&[rows, cols]),
    ));
    let scales = Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(scales, dev.clone())),
        Shape::from_dims(&[rows, cols / FP8_BLOCK_SIZE]),
    ));
    Ok((quantized, scales))
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
struct CutlassGemm<'a> {
    activation: &'a Tensor,
    activation_scales: &'a Tensor,
    weight: &'a Tensor,
    weight_scales: &'a Tensor,
    output_dtype: DType,
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
fn launch_cutlass_gemm(context: CutlassGemm<'_>, output_ptr: u64) -> Result<()> {
    use candle_core::{Device, Storage};

    use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

    let Device::Cuda(dev) = context.activation.device() else {
        unreachable!()
    };
    let stream = dev.cuda_stream();
    let (m, k) = context.activation.dims2()?;
    let (n, _) = context.weight.dims2()?;
    let m = i32::try_from(m).map_err(|_| candle_core::Error::msg("CUTLASS FP8 M exceeds i32"))?;
    let n = i32::try_from(n).map_err(|_| candle_core::Error::msg("CUTLASS FP8 N exceeds i32"))?;
    let k = i32::try_from(k).map_err(|_| candle_core::Error::msg("CUTLASS FP8 K exceeds i32"))?;
    let output_dtype = match context.output_dtype {
        DType::F16 => CUTLASS_OUTPUT_F16,
        DType::BF16 => CUTLASS_OUTPUT_BF16,
        dtype => candle_core::bail!("unsupported CUTLASS FP8 output dtype {dtype:?}"),
    };
    let sm_count = prepare_cutlass_fp8(dev)?;

    let workspace_bytes = cutlass_workspace_size(CutlassWorkspaceRequirementsKey {
        m,
        n,
        k,
        output_dtype,
        sm_count,
    })?;
    let workspace = fp8_workspace(dev, workspace_bytes, "CUTLASS")?;
    let mut workspace_lock = workspace
        .as_ref()
        .map(|workspace| workspace.lock().unwrap());
    let (workspace_ptr, workspace_guard) = match workspace_lock.as_mut() {
        Some(workspace) => {
            let (pointer, guard) = slice_ptr_mut_on_stream(&mut workspace.slice, 0, &stream);
            (pointer, Some(guard))
        }
        None => (0, None),
    };

    let (activation_storage, activation_layout) = context.activation.storage_and_layout();
    let Storage::Cuda(activation_storage) = &*activation_storage else {
        unreachable!()
    };
    let (activation_ptr, activation_guard) = slice_ptr_on_stream(
        activation_storage.as_cuda_slice::<F8E4M3>()?,
        activation_layout.start_offset(),
        &stream,
    );
    let (weight_storage, weight_layout) = context.weight.storage_and_layout();
    let Storage::Cuda(weight_storage) = &*weight_storage else {
        unreachable!()
    };
    let (weight_ptr, weight_guard) = slice_ptr_on_stream(
        weight_storage.as_cuda_slice::<F8E4M3>()?,
        weight_layout.start_offset(),
        &stream,
    );
    let (activation_scales_storage, activation_scales_layout) =
        context.activation_scales.storage_and_layout();
    let Storage::Cuda(activation_scales_storage) = &*activation_scales_storage else {
        unreachable!()
    };
    let (activation_scales_ptr, activation_scales_guard) = slice_ptr_on_stream(
        activation_scales_storage.as_cuda_slice::<f32>()?,
        activation_scales_layout.start_offset(),
        &stream,
    );
    let (weight_scales_storage, weight_scales_layout) = context.weight_scales.storage_and_layout();
    let Storage::Cuda(weight_scales_storage) = &*weight_scales_storage else {
        unreachable!()
    };
    let (weight_scales_ptr, weight_scales_guard) = slice_ptr_on_stream(
        weight_scales_storage.as_cuda_slice::<f32>()?,
        weight_scales_layout.start_offset(),
        &stream,
    );

    let status = unsafe {
        ffi::mistralrs_cutlass_fp8_blockwise_gemm(
            activation_ptr as *const core::ffi::c_void,
            weight_ptr as *const core::ffi::c_void,
            activation_scales_ptr as *const f32,
            weight_scales_ptr as *const f32,
            output_ptr as *mut core::ffi::c_void,
            m,
            n,
            k,
            output_dtype,
            workspace_ptr as *mut core::ffi::c_void,
            workspace_bytes,
            sm_count,
            stream.cu_stream() as *mut core::ffi::c_void,
        )
    };
    check_cutlass_status("CUTLASS blockwise FP8 GEMM", status)?;
    drop((
        activation_guard,
        weight_guard,
        activation_scales_guard,
        weight_scales_guard,
        workspace_guard,
    ));
    drop(workspace_lock);
    drop(workspace);
    Ok(())
}

#[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
pub(crate) fn fp8_blockwise_matmul_cutlass(
    activation: &Tensor,
    activation_scales: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    output_dtype: DType,
) -> Result<Tensor> {
    use candle_core::{CudaStorage, Device, Shape, Storage};
    use half::{bf16, f16};

    use crate::utils::slice_ptr_mut_on_stream;

    if activation.dtype() != DType::F8E4M3 || activation_scales.dtype() != DType::F32 {
        candle_core::bail!("CUTLASS FP8 activation values/scales must be F8E4M3/F32")
    }
    if !cutlass_fp8_blockwise_supported(weight, weight_scales, &[FP8_BLOCK_SIZE, FP8_BLOCK_SIZE]) {
        candle_core::bail!("CUTLASS blockwise FP8 GEMM does not support this weight layout")
    }
    if !activation.is_contiguous()
        || !activation_scales.is_contiguous()
        || !activation.device().same_device(activation_scales.device())
        || !activation.device().same_device(weight.device())
        || !fp8_tensor_aligned(activation)
        || !fp8_tensor_aligned(activation_scales)
    {
        candle_core::bail!("CUTLASS FP8 operands must be contiguous and on the same device")
    }
    let (m, k) = activation.dims2()?;
    let (n, weight_k) = weight.dims2()?;
    if m == 0 || weight_k != k || activation_scales.dims() != [m, k / FP8_BLOCK_SIZE] {
        candle_core::bail!("CUTLASS FP8 activation, scale, and weight shapes are incompatible")
    }
    let Device::Cuda(dev) = activation.device() else {
        unreachable!()
    };
    let stream = dev.cuda_stream();
    let context = CutlassGemm {
        activation,
        activation_scales,
        weight,
        weight_scales,
        output_dtype,
    };
    let shape = Shape::from_dims(&[m, n]);
    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| candle_core::Error::msg("CUTLASS FP8 output shape overflows usize"))?;
    match output_dtype {
        DType::F16 => {
            let mut output = unsafe { dev.alloc::<f16>(output_len)? };
            let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            launch_cutlass_gemm(context, output_ptr)?;
            drop(output_guard);
            Ok(Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                shape,
            )))
        }
        DType::BF16 => {
            let mut output = unsafe { dev.alloc::<bf16>(output_len)? };
            let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            launch_cutlass_gemm(context, output_ptr)?;
            drop(output_guard);
            Ok(Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                shape,
            )))
        }
        dtype => candle_core::bail!("unsupported CUTLASS FP8 output dtype {dtype:?}"),
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};
    use candle_nn::{Linear, Module};
    use half::bf16;

    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
    use crate::blockwise_fp8::deepgemm;
    use crate::blockwise_fp8::ops;

    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
    #[test]
    fn deepgemm_ffi_plan_layout() {
        use crate::blockwise_fp8::ffi::{DeepGemmPlan, DeepGemmPrepared};

        assert_eq!(std::mem::offset_of!(DeepGemmPlan, workspace_bytes), 56);
        assert_eq!(std::mem::offset_of!(DeepGemmPlan, cache_key), 64);
        assert_eq!(std::mem::size_of::<DeepGemmPlan>(), 72);
        assert_eq!(std::mem::offset_of!(DeepGemmPrepared, function), 72);
        assert_eq!(std::mem::size_of::<DeepGemmPrepared>(), 80);
    }

    #[cfg(all(
        feature = "cuda",
        any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
    ))]
    #[test]
    fn fp8_workspace_keys_per_thread_default_streams() {
        let current = ops::fp8_workspace_thread(ops::CUDA_STREAM_PER_THREAD_HANDLE).unwrap();
        let other = std::thread::spawn(|| {
            ops::fp8_workspace_thread(ops::CUDA_STREAM_PER_THREAD_HANDLE).unwrap()
        })
        .join()
        .unwrap();
        assert_ne!(current, other);
        assert_eq!(ops::fp8_workspace_thread(0x1000), None);
    }

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

    #[test]
    fn test_fp8_blockwise_dequant_rectangular_blocks() -> Result<()> {
        let dev = &Device::Cpu;
        let weight = Tensor::ones((5, 7), DType::F8E4M3, dev)?;
        let inv_scales = Tensor::arange(0f32, 9f32, dev)?.reshape((3, 3))?;
        let dequant = ops::fp8_blockwise_dequantize(&weight, &inv_scales, vec![2, 3], DType::F32)?;

        assert_eq!(
            dequant.to_vec2::<f32>()?,
            vec![
                vec![0., 0., 0., 1., 1., 1., 2.],
                vec![0., 0., 0., 1., 1., 1., 2.],
                vec![3., 3., 3., 4., 4., 4., 5.],
                vec![3., 3., 3., 4., 4., 4., 5.],
                vec![6., 6., 6., 7., 7., 7., 8.],
            ]
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fp8_blockwise_dequant_cuda() -> Result<()> {
        let truth = {
            let dev = &Device::Cpu;
            let weight = Tensor::ones((5, 7), DType::F8E4M3, dev)?;
            let weight_block_size = vec![2, 3];
            let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

            let dequant =
                ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::F32)?;

            dequant.to_vec2::<f32>()?
        };
        let test = {
            let dev = &Device::new_cuda(0)?;
            // Create FP8 weight by first creating on CPU then moving to CUDA
            let weight_cpu = Tensor::ones((5, 7), DType::F8E4M3, &Device::Cpu)?;
            let weight = weight_cpu.to_device(dev)?;
            let weight_block_size = vec![2, 3];
            let inv_scales = Tensor::arange(0f32, (3 * 3) as f32, dev)?.reshape((3, 3))?;

            let dequant =
                ops::fp8_blockwise_dequantize(&weight, &inv_scales, weight_block_size, DType::F32)?;

            dequant.to_vec2::<f32>()?
        };

        assert_eq!(test, truth);
        assert_eq!(
            test,
            vec![
                vec![0., 0., 0., 1., 1., 1., 2.],
                vec![0., 0., 0., 1., 1., 1., 2.],
                vec![3., 3., 3., 4., 4., 4., 5.],
                vec![3., 3., 3., 4., 4., 4., 5.],
                vec![6., 6., 6., 7., 7., 7., 8.],
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

    #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
    #[test]
    fn test_cutlass_blockwise_fp8_gemm() -> Result<()> {
        const BLOCK_SIZE: usize = 128;
        const FP8_E4M3_MAX: f32 = 448.0;
        const SCALE_ABS_TOLERANCE: f32 = 1.0e-8;
        const SCALE_REL_TOLERANCE: f32 = 1.0e-5;
        const K: usize = 256;
        const N: usize = 256;

        let dev = Device::new_cuda(0)?;
        let weight_values = (0..N * K)
            .map(|index| ((index * 7 + index / K * 13) % 23) as f32 * 0.03 - 0.33)
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(weight_values, (N, K), &dev)?.to_dtype(DType::BF16)?;
        let (weight_q, weight_scales) = ops::fp8_blockwise_quantize(&weight, vec![128, 128])?;
        assert!(ops::cutlass_fp8_blockwise_supported(
            &weight_q,
            &weight_scales,
            &[128, 128]
        ));
        let reference_weight = weight.to_dtype(DType::F32)?.t()?;

        for rows in [1usize, 8] {
            for dtype in [DType::F16, DType::BF16] {
                let k_blocks = K / BLOCK_SIZE;
                let input_values = (0..rows * K)
                    .map(|index| {
                        let row = index / K;
                        let k_block = index % K / BLOCK_SIZE;
                        (row * k_blocks + k_block + 1) as f32 / 16.0
                    })
                    .collect::<Vec<_>>();
                let input = Tensor::from_vec(input_values, (rows, K), &dev)?.to_dtype(dtype)?;
                let reference = input.to_dtype(DType::F32)?.matmul(&reference_weight)?;
                let (input_q, input_scales) = ops::fp8_quantize_activation_cutlass(&input)?;
                let input_scale_values = input_scales.flatten_all()?.to_vec1::<f32>()?;
                let expected_scales = (0..k_blocks)
                    .flat_map(|k_block| {
                        (0..rows).map(move |row| {
                            (row * k_blocks + k_block + 1) as f32 / 16.0 / FP8_E4M3_MAX
                        })
                    })
                    .collect::<Vec<_>>();
                for (index, (actual, expected)) in input_scale_values
                    .iter()
                    .zip(expected_scales.iter())
                    .enumerate()
                {
                    let tolerance = SCALE_ABS_TOLERANCE + SCALE_REL_TOLERANCE * expected.abs();
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "rows={rows}, dtype={dtype:?}, scale index {index}: expected {expected}, got {actual}"
                    );
                }
                let output = ops::fp8_blockwise_matmul_cutlass(
                    &input_q,
                    &input_scales,
                    &weight_q,
                    &weight_scales,
                    dtype,
                )?
                .to_dtype(DType::F32)?;

                let reference = reference.flatten_all()?.to_vec1::<f32>()?;
                let output = output.flatten_all()?.to_vec1::<f32>()?;
                let max_reference = reference.iter().copied().map(f32::abs).fold(0f32, f32::max);
                let max_error = reference
                    .iter()
                    .zip(output.iter())
                    .map(|(reference, output)| (reference - output).abs())
                    .fold(0f32, f32::max);
                assert!(
                    max_error <= 0.12 + 0.08 * max_reference,
                    "rows={rows}, dtype={dtype:?}, max error {max_error}, max reference {max_reference}"
                );
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
    #[test]
    #[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
    fn test_deepgemm_blockwise_fp8_and_cuda_graph() -> Result<()> {
        use candle_core::cuda::cudarc::driver::sys;

        const BLOCK_SIZE: usize = 128;
        const K: usize = 256;
        const N: usize = 256;

        fn assert_close(rows: usize, reference: &Tensor, output: &Tensor) -> Result<()> {
            let reference = reference.flatten_all()?.to_vec1::<f32>()?;
            let output = output.flatten_all()?.to_vec1::<f32>()?;
            let max_reference = reference.iter().copied().map(f32::abs).fold(0.0, f32::max);
            let max_error = reference
                .iter()
                .zip(&output)
                .map(|(reference, output)| (reference - output).abs())
                .fold(0.0, f32::max);
            assert!(
                max_error <= 0.02 + 0.02 * max_reference,
                "rows={rows}: max error {max_error}, max reference {max_reference}"
            );
            Ok(())
        }

        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda_dev) = &dev else {
            unreachable!()
        };
        let stream = cuda_dev.cuda_stream();
        unsafe { stream.context().disable_event_tracking() };
        let weight_values = (0..N * K)
            .map(|index| {
                let row = index / K;
                let column = index % K;
                let block = row / BLOCK_SIZE * 2 + column / BLOCK_SIZE;
                let amplitude = [0.04, 0.18, 0.75, 1.6][block];
                let value = ((row * 17 + column * 29) % 31) as f32 - 15.0;
                value * amplitude
            })
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(weight_values, (N, K), &dev)?.to_dtype(DType::BF16)?;
        let (weight, weight_scales) =
            ops::fp8_blockwise_quantize(&weight, vec![BLOCK_SIZE, BLOCK_SIZE])?;
        let weight_scale_values = weight_scales.flatten_all()?.to_vec1::<f32>()?;
        let min_scale = weight_scale_values
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min);
        let max_scale = weight_scale_values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        assert!(max_scale > min_scale * 8.0);
        assert!(deepgemm::supported(
            &weight,
            &weight_scales,
            &[BLOCK_SIZE, BLOCK_SIZE]
        ));
        let live_event = stream
            .record_event(None)
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        stream
            .wait(&live_event)
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        dev.synchronize()?;
        let prepared = deepgemm::prepare(&weight, &weight_scales, &[BLOCK_SIZE, BLOCK_SIZE])?;
        stream
            .wait(&live_event)
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        dev.synchronize()?;
        for rows in [1usize, 8, 16] {
            let input_values = (0..rows * K)
                .map(|index| {
                    let row = index / K;
                    let column = index % K;
                    let block = column / BLOCK_SIZE;
                    let amplitude = (row + 1) as f32 * (block + 1) as f32 * 0.025;
                    let value = ((row * 11 + column * 7) % 29) as f32 - 14.0;
                    value * amplitude
                })
                .collect::<Vec<_>>();
            let input = Tensor::from_vec(input_values, (rows, K), &dev)?.to_dtype(DType::BF16)?;
            let (activation, activation_scales) = ops::fp8_quantize_activation_cutlass(&input)?;
            dev.synchronize()?;
            let reference = ops::fp8_blockwise_matmul_cutlass(
                &activation,
                &activation_scales,
                &weight,
                &weight_scales,
                DType::BF16,
            )?;
            dev.synchronize()?;
            let reference = reference.to_dtype(DType::F32)?;
            dev.synchronize()?;
            let output = deepgemm::matmul(&prepared, &input, &weight, &weight_scales)?;
            dev.synchronize()?;
            let output = output.to_dtype(DType::F32)?;
            dev.synchronize()?;
            assert_close(rows, &reference, &output)?;
            dev.synchronize()?;

            if rows != 8 {
                continue;
            }

            let graph_output = Tensor::zeros((rows, N), DType::BF16, &dev)?;
            let restore_event_tracking = stream.context().is_event_tracking();
            if restore_event_tracking {
                unsafe { stream.context().disable_event_tracking() };
            }
            if let Err(error) =
                stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            {
                if restore_event_tracking {
                    unsafe { stream.context().enable_event_tracking() };
                }
                return Err(candle_core::Error::msg(error.to_string()));
            }
            let captured =
                deepgemm::matmul(&prepared, &input, &weight, &weight_scales).and_then(|captured| {
                    use crate::utils::slice_ptr_on_stream;
                    use candle_core::Storage;

                    let status = {
                        let (src_storage, src_layout) = captured.storage_and_layout();
                        let Storage::Cuda(src_storage) = &*src_storage else {
                            unreachable!()
                        };
                        let (dst_storage, dst_layout) = graph_output.storage_and_layout();
                        let Storage::Cuda(dst_storage) = &*dst_storage else {
                            unreachable!()
                        };
                        let (src_ptr, src_guard) = slice_ptr_on_stream(
                            src_storage.as_cuda_slice::<bf16>()?,
                            src_layout.start_offset(),
                            &stream,
                        );
                        let (dst_ptr, dst_guard) = slice_ptr_on_stream(
                            dst_storage.as_cuda_slice::<bf16>()?,
                            dst_layout.start_offset(),
                            &stream,
                        );
                        let status = unsafe {
                            sys::cuMemcpyDtoDAsync_v2(
                                dst_ptr,
                                src_ptr,
                                rows * N * std::mem::size_of::<bf16>(),
                                stream.cu_stream(),
                            )
                        };
                        drop((src_guard, dst_guard));
                        status
                    };
                    drop(captured);
                    if status != sys::cudaError_enum::CUDA_SUCCESS {
                        candle_core::bail!("CUDA graph output copy failed: {status:?}")
                    }
                    Ok(())
                });
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            if restore_event_tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            captured?;
            let graph = graph
                .map_err(|error| candle_core::Error::msg(error.to_string()))?
                .ok_or_else(|| candle_core::Error::msg("CUDA graph capture returned no graph"))?;
            graph
                .launch()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            graph
                .launch()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            stream
                .synchronize()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            assert_close(rows, &reference, &graph_output.to_dtype(DType::F32)?)?;
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
    #[test]
    #[ignore = "requires an SM90 GPU and runtime nvcc or a prepared cubin cache"]
    fn test_deepgemm_shared_prepared_state_across_ptds_threads() -> Result<()> {
        const BLOCK_SIZE: usize = 128;
        const ITERATIONS: usize = 32;
        const K: usize = 256;
        const N: usize = 256;
        const ROWS: usize = 8;

        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda_dev) = &dev else {
            unreachable!()
        };
        unsafe { cuda_dev.cuda_stream().context().disable_event_tracking() };
        let weight = Tensor::ones((N, K), DType::F8E4M3, &Device::Cpu)?.to_device(&dev)?;
        let weight_scales = Tensor::ones((N / BLOCK_SIZE, K / BLOCK_SIZE), DType::F32, &dev)?
            .affine(1.0 / K as f64, 0.0)?;
        let prepared = deepgemm::prepare(&weight, &weight_scales, &[BLOCK_SIZE, BLOCK_SIZE])?;
        dev.synchronize()?;

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        std::thread::scope(|scope| -> Result<()> {
            let spawn = |value: f32| {
                let dev = dev.clone();
                let weight = weight.clone();
                let weight_scales = weight_scales.clone();
                let prepared = prepared.clone();
                let barrier = barrier.clone();
                scope.spawn(move || -> Result<()> {
                    let values = vec![bf16::from_f32(value); ROWS * K];
                    let input = Tensor::from_vec(values, (ROWS, K), &dev)?;
                    barrier.wait();
                    let mut output = None;
                    for _ in 0..ITERATIONS {
                        output = Some(deepgemm::matmul(
                            &prepared,
                            &input,
                            &weight,
                            &weight_scales,
                        )?);
                        std::thread::yield_now();
                    }
                    dev.synchronize()?;
                    let output = output.unwrap().to_dtype(DType::F32)?;
                    let max_error = output
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .into_iter()
                        .map(|output| (output - value).abs())
                        .fold(0.0, f32::max);
                    assert!(max_error <= 0.02, "value={value}: max error {max_error}");
                    Ok(())
                })
            };
            let first = spawn(0.5);
            let second = spawn(2.0);
            first.join().expect("first PTDS worker panicked")?;
            second.join().expect("second PTDS worker panicked")?;
            Ok(())
        })
    }

    #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
    #[derive(Clone, Copy)]
    struct BlockwiseFp8BenchShape {
        name: &'static str,
        n: usize,
        k: usize,
    }

    #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
    fn blockwise_fp8_bench_iterations(variable: &str, default: usize) -> Result<usize> {
        match std::env::var(variable) {
            Ok(value) => value
                .parse()
                .ok()
                .filter(|value| *value != 0)
                .ok_or_else(|| {
                    candle_core::Error::msg(format!("{variable} must be a positive integer"))
                }),
            Err(std::env::VarError::NotPresent) => Ok(default),
            Err(error) => Err(candle_core::Error::msg(format!(
                "failed to read {variable}: {error}"
            ))),
        }
    }

    #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
    fn measure_blockwise_fp8_cuda_us<T>(
        dev: &Device,
        warmup: usize,
        iterations: usize,
        mut launch: impl FnMut() -> Result<T>,
    ) -> Result<f64> {
        use candle_core::cuda::cudarc::driver::sys;

        for _ in 0..warmup {
            drop(launch()?);
        }
        dev.synchronize()?;

        let stream = dev.as_cuda_device()?.cuda_stream();
        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| {
                candle_core::Error::msg(format!("CUDA start event failed: {error}"))
            })?;
        for _ in 0..iterations {
            drop(launch()?);
        }
        let end = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| candle_core::Error::msg(format!("CUDA end event failed: {error}")))?;
        end.synchronize().map_err(|error| {
            candle_core::Error::msg(format!("CUDA event synchronization failed: {error}"))
        })?;
        let elapsed_ms = start
            .elapsed_ms(&end)
            .map_err(|error| candle_core::Error::msg(format!("CUDA timing failed: {error}")))?;
        Ok(f64::from(elapsed_ms) * 1_000.0 / iterations as f64)
    }

    #[cfg(all(
        feature = "cuda",
        any(has_cutlass_fp8_sm90_kernels, has_deepgemm_fp8_sm90_provider)
    ))]
    fn measure_blockwise_fp8_cuda_graph_us<T>(
        dev: &Device,
        warmup: usize,
        iterations: usize,
        samples: usize,
        capture: impl FnOnce() -> Result<T>,
    ) -> Result<Vec<f64>> {
        use candle_core::cuda::cudarc::driver::sys;

        dev.synchronize()?;
        let Device::Cuda(cuda_dev) = dev else {
            candle_core::bail!("CUDA graph timing requires a CUDA device")
        };
        let stream = cuda_dev.cuda_stream();
        let restore_event_tracking = stream.context().is_event_tracking();
        if restore_event_tracking {
            unsafe { stream.context().disable_event_tracking() };
        }
        if let Err(error) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            if restore_event_tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            return Err(candle_core::Error::msg(error.to_string()));
        }
        let captured = capture();
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if restore_event_tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let captured = captured?;
        let graph = graph
            .map_err(|error| candle_core::Error::msg(error.to_string()))?
            .ok_or_else(|| candle_core::Error::msg("CUDA graph capture returned no graph"))?;
        for _ in 0..warmup {
            graph
                .launch()
                .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        }
        stream
            .synchronize()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;

        let mut timings = Vec::with_capacity(samples);
        for _ in 0..samples {
            let start = stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| {
                    candle_core::Error::msg(format!("CUDA start event failed: {error}"))
                })?;
            for _ in 0..iterations {
                graph
                    .launch()
                    .map_err(|error| candle_core::Error::msg(error.to_string()))?;
            }
            let end = stream
                .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(|error| {
                    candle_core::Error::msg(format!("CUDA end event failed: {error}"))
                })?;
            end.synchronize().map_err(|error| {
                candle_core::Error::msg(format!("CUDA event synchronization failed: {error}"))
            })?;
            let elapsed_ms = start
                .elapsed_ms(&end)
                .map_err(|error| candle_core::Error::msg(format!("CUDA timing failed: {error}")))?;
            timings.push(f64::from(elapsed_ms) * 1_000.0 / iterations as f64);
        }
        drop((captured, graph));
        Ok(timings)
    }

    #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
    fn validate_blockwise_fp8_bench_case(
        activation_scales: &Tensor,
        output: &Tensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<f32> {
        const FP8_E4M3_MAX: f32 = 448.0;
        const SCALE_TOLERANCE: f32 = 1.0e-7;
        const OUTPUT_TOLERANCE: f32 = 0.02;

        let expected_scale = 1.0 / FP8_E4M3_MAX;
        let scale_error = activation_scales
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|scale| (scale - expected_scale).abs())
            .fold(0.0, f32::max);
        assert!(
            scale_error <= SCALE_TOLERANCE,
            "M={m}, N={n}, K={k}: activation scale error {scale_error}"
        );

        assert_eq!(output.dims2()?, (m, n));
        let output_error = output
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(|value| (value - 1.0).abs())
            .fold(0.0, f32::max);
        assert!(
            output_error <= OUTPUT_TOLERANCE,
            "M={m}, N={n}, K={k}: output error {output_error}"
        );
        Ok(output_error)
    }

    #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
    #[test]
    #[ignore = "requires an SM90 GPU and reports latency to stdout"]
    fn bench_cutlass_blockwise_fp8_production_shapes_sm90() -> Result<()> {
        const BLOCK_SIZE: usize = 128;
        const DEFAULT_WARMUP: usize = 10;
        const DEFAULT_ITERATIONS: usize = 100;
        const M_VALUES: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64];
        const SHAPES: [BlockwiseFp8BenchShape; 5] = [
            BlockwiseFp8BenchShape {
                name: "gdn_qkvz",
                n: 16_384,
                k: 5_120,
            },
            BlockwiseFp8BenchShape {
                name: "attention_qkv",
                n: 14_336,
                k: 5_120,
            },
            BlockwiseFp8BenchShape {
                name: "output_projection",
                n: 5_120,
                k: 6_144,
            },
            BlockwiseFp8BenchShape {
                name: "mlp_gate_up",
                n: 34_816,
                k: 5_120,
            },
            BlockwiseFp8BenchShape {
                name: "mlp_down",
                n: 5_120,
                k: 17_408,
            },
        ];

        let warmup =
            blockwise_fp8_bench_iterations("MISTRALRS_BLOCKWISE_FP8_BENCH_WARMUP", DEFAULT_WARMUP)?;
        let iterations = blockwise_fp8_bench_iterations(
            "MISTRALRS_BLOCKWISE_FP8_BENCH_ITERATIONS",
            DEFAULT_ITERATIONS,
        )?;
        let dev = Device::new_cuda(0)?;
        println!("shape,m,n,k,quantize_gpu_us,gemm_gpu_us,max_abs_error,warmup,iterations");

        for shape in SHAPES {
            let weight =
                Tensor::ones((shape.n, shape.k), DType::F8E4M3, &Device::Cpu)?.to_device(&dev)?;
            let weight_scales = Tensor::ones(
                (shape.n / BLOCK_SIZE, shape.k / BLOCK_SIZE),
                DType::F32,
                &dev,
            )?
            .affine(1.0 / shape.k as f64, 0.0)?;
            assert!(ops::cutlass_fp8_blockwise_supported(
                &weight,
                &weight_scales,
                &[BLOCK_SIZE, BLOCK_SIZE]
            ));

            for m in M_VALUES {
                let input = Tensor::ones((m, shape.k), DType::BF16, &dev)?;
                let quantize_us = measure_blockwise_fp8_cuda_us(&dev, warmup, iterations, || {
                    ops::fp8_quantize_activation_cutlass(&input)
                })?;
                let (activation, activation_scales) = ops::fp8_quantize_activation_cutlass(&input)?;
                let gemm_us = measure_blockwise_fp8_cuda_us(&dev, warmup, iterations, || {
                    ops::fp8_blockwise_matmul_cutlass(
                        &activation,
                        &activation_scales,
                        &weight,
                        &weight_scales,
                        DType::BF16,
                    )
                })?;
                let output = ops::fp8_blockwise_matmul_cutlass(
                    &activation,
                    &activation_scales,
                    &weight,
                    &weight_scales,
                    DType::BF16,
                )?;
                let max_error = validate_blockwise_fp8_bench_case(
                    &activation_scales,
                    &output,
                    m,
                    shape.n,
                    shape.k,
                )?;
                println!(
                    "{},{},{},{},{quantize_us:.3},{gemm_us:.3},{max_error:.6},{warmup},{iterations}",
                    shape.name, m, shape.n, shape.k
                );
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
    #[test]
    #[ignore = "requires an SM90 GPU and reports latency to stdout"]
    fn bench_deepgemm_blockwise_fp8_production_shapes_sm90() -> Result<()> {
        const BLOCK_SIZE: usize = 128;
        const DEFAULT_WARMUP: usize = 10;
        const DEFAULT_ITERATIONS: usize = 100;
        const OUTPUT_TOLERANCE: f32 = 0.02;
        const M_VALUES: [usize; 6] = [1, 8, 16, 32, 64, 128];
        const SHAPES: [BlockwiseFp8BenchShape; 5] = [
            BlockwiseFp8BenchShape {
                name: "gdn_qkvz",
                n: 16_384,
                k: 5_120,
            },
            BlockwiseFp8BenchShape {
                name: "attention_qkv",
                n: 14_336,
                k: 5_120,
            },
            BlockwiseFp8BenchShape {
                name: "output_projection",
                n: 5_120,
                k: 6_144,
            },
            BlockwiseFp8BenchShape {
                name: "mlp_gate_up",
                n: 34_816,
                k: 5_120,
            },
            BlockwiseFp8BenchShape {
                name: "mlp_down",
                n: 5_120,
                k: 17_408,
            },
        ];

        let warmup =
            blockwise_fp8_bench_iterations("MISTRALRS_BLOCKWISE_FP8_BENCH_WARMUP", DEFAULT_WARMUP)?;
        let iterations = blockwise_fp8_bench_iterations(
            "MISTRALRS_BLOCKWISE_FP8_BENCH_ITERATIONS",
            DEFAULT_ITERATIONS,
        )?;
        let dev = Device::new_cuda(0)?;
        println!("shape,m,n,k,fused_gpu_us,max_abs_error,warmup,iterations");

        for shape in SHAPES {
            let weight =
                Tensor::ones((shape.n, shape.k), DType::F8E4M3, &Device::Cpu)?.to_device(&dev)?;
            let weight_scales = Tensor::ones(
                (shape.n / BLOCK_SIZE, shape.k / BLOCK_SIZE),
                DType::F32,
                &dev,
            )?
            .affine(1.0 / shape.k as f64, 0.0)?;
            let prepared = deepgemm::prepare(&weight, &weight_scales, &[BLOCK_SIZE, BLOCK_SIZE])?;

            for m in M_VALUES {
                let input = Tensor::ones((m, shape.k), DType::BF16, &dev)?;
                let fused_us = measure_blockwise_fp8_cuda_us(&dev, warmup, iterations, || {
                    deepgemm::matmul(&prepared, &input, &weight, &weight_scales)
                })?;
                let output = deepgemm::matmul(&prepared, &input, &weight, &weight_scales)?;
                let max_error = output
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(|value| (value - 1.0).abs())
                    .fold(0.0, f32::max);
                assert!(
                    max_error <= OUTPUT_TOLERANCE,
                    "M={m}, N={}, K={}: output error {max_error}",
                    shape.n,
                    shape.k
                );
                println!(
                    "{},{},{},{},{fused_us:.3},{max_error:.6},{warmup},{iterations}",
                    shape.name, m, shape.n, shape.k
                );
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
    fn median_cuda_us(mut samples: Vec<f64>) -> f64 {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    #[cfg(all(feature = "cuda", has_deepgemm_fp8_sm90_provider))]
    #[test]
    #[ignore = "requires an SM90 GPU and reports latency to stdout"]
    fn bench_cutlass_vs_deepgemm_fused_production_shapes_sm90() -> Result<()> {
        const BLOCK_SIZE: usize = 128;
        const DEFAULT_WARMUP: usize = 10;
        const DEFAULT_ITERATIONS: usize = 100;
        const SAMPLES: usize = 7;
        const M_VALUES: [usize; 6] = [1, 8, 16, 32, 64, 128];
        const SHAPES: [BlockwiseFp8BenchShape; 5] = [
            BlockwiseFp8BenchShape {
                name: "gdn_qkvz",
                n: 16_384,
                k: 5_120,
            },
            BlockwiseFp8BenchShape {
                name: "attention_qkv",
                n: 14_336,
                k: 5_120,
            },
            BlockwiseFp8BenchShape {
                name: "output_projection",
                n: 5_120,
                k: 6_144,
            },
            BlockwiseFp8BenchShape {
                name: "mlp_gate_up",
                n: 34_816,
                k: 5_120,
            },
            BlockwiseFp8BenchShape {
                name: "mlp_down",
                n: 5_120,
                k: 17_408,
            },
        ];

        let warmup =
            blockwise_fp8_bench_iterations("MISTRALRS_BLOCKWISE_FP8_BENCH_WARMUP", DEFAULT_WARMUP)?;
        let iterations = blockwise_fp8_bench_iterations(
            "MISTRALRS_BLOCKWISE_FP8_BENCH_ITERATIONS",
            DEFAULT_ITERATIONS,
        )?;
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda_dev) = &dev else {
            unreachable!()
        };
        unsafe { cuda_dev.cuda_stream().context().disable_event_tracking() };
        println!(
            "shape,m,n,k,cutlass_quant_median_us,cutlass_gemm_median_us,cutlass_fused_median_us,deepgemm_fused_median_us,deepgemm_speedup,cutlass_graph_median_us,deepgemm_graph_median_us,deepgemm_graph_speedup,samples,warmup,iterations"
        );

        for shape in SHAPES {
            let weight =
                Tensor::ones((shape.n, shape.k), DType::F8E4M3, &Device::Cpu)?.to_device(&dev)?;
            let weight_scales = Tensor::ones(
                (shape.n / BLOCK_SIZE, shape.k / BLOCK_SIZE),
                DType::F32,
                &dev,
            )?
            .affine(1.0 / shape.k as f64, 0.0)?;
            let prepared = deepgemm::prepare(&weight, &weight_scales, &[BLOCK_SIZE, BLOCK_SIZE])?;

            for m in M_VALUES {
                let input = Tensor::ones((m, shape.k), DType::BF16, &dev)?;
                let (activation, activation_scales) = ops::fp8_quantize_activation_cutlass(&input)?;
                let mut cutlass_quant = Vec::with_capacity(SAMPLES);
                let mut cutlass_gemm = Vec::with_capacity(SAMPLES);
                let mut cutlass_fused = Vec::with_capacity(SAMPLES);
                let mut deepgemm_fused = Vec::with_capacity(SAMPLES);
                for sample in 0..SAMPLES {
                    if sample % 2 == 0 {
                        cutlass_fused.push(measure_blockwise_fp8_cuda_us(
                            &dev,
                            warmup,
                            iterations,
                            || {
                                let (activation, scales) =
                                    ops::fp8_quantize_activation_cutlass(&input)?;
                                ops::fp8_blockwise_matmul_cutlass(
                                    &activation,
                                    &scales,
                                    &weight,
                                    &weight_scales,
                                    DType::BF16,
                                )
                            },
                        )?);
                        deepgemm_fused.push(measure_blockwise_fp8_cuda_us(
                            &dev,
                            warmup,
                            iterations,
                            || deepgemm::matmul(&prepared, &input, &weight, &weight_scales),
                        )?);
                    } else {
                        deepgemm_fused.push(measure_blockwise_fp8_cuda_us(
                            &dev,
                            warmup,
                            iterations,
                            || deepgemm::matmul(&prepared, &input, &weight, &weight_scales),
                        )?);
                        cutlass_fused.push(measure_blockwise_fp8_cuda_us(
                            &dev,
                            warmup,
                            iterations,
                            || {
                                let (activation, scales) =
                                    ops::fp8_quantize_activation_cutlass(&input)?;
                                ops::fp8_blockwise_matmul_cutlass(
                                    &activation,
                                    &scales,
                                    &weight,
                                    &weight_scales,
                                    DType::BF16,
                                )
                            },
                        )?);
                    }
                    cutlass_quant.push(measure_blockwise_fp8_cuda_us(
                        &dev,
                        warmup,
                        iterations,
                        || ops::fp8_quantize_activation_cutlass(&input),
                    )?);
                    cutlass_gemm.push(measure_blockwise_fp8_cuda_us(
                        &dev,
                        warmup,
                        iterations,
                        || {
                            ops::fp8_blockwise_matmul_cutlass(
                                &activation,
                                &activation_scales,
                                &weight,
                                &weight_scales,
                                DType::BF16,
                            )
                        },
                    )?);
                }
                let cutlass_quant = median_cuda_us(cutlass_quant);
                let cutlass_gemm = median_cuda_us(cutlass_gemm);
                let cutlass_fused = median_cuda_us(cutlass_fused);
                let deepgemm_fused = median_cuda_us(deepgemm_fused);
                let cutlass_graph = median_cuda_us(measure_blockwise_fp8_cuda_graph_us(
                    &dev,
                    warmup,
                    iterations,
                    SAMPLES,
                    || {
                        let (activation, scales) = ops::fp8_quantize_activation_cutlass(&input)?;
                        let output = ops::fp8_blockwise_matmul_cutlass(
                            &activation,
                            &scales,
                            &weight,
                            &weight_scales,
                            DType::BF16,
                        )?;
                        Ok((activation, scales, output))
                    },
                )?);
                let deepgemm_graph = median_cuda_us(measure_blockwise_fp8_cuda_graph_us(
                    &dev,
                    warmup,
                    iterations,
                    SAMPLES,
                    || deepgemm::matmul(&prepared, &input, &weight, &weight_scales),
                )?);
                println!(
                    "{},{},{},{},{cutlass_quant:.3},{cutlass_gemm:.3},{cutlass_fused:.3},{deepgemm_fused:.3},{:.3},{cutlass_graph:.3},{deepgemm_graph:.3},{:.3},{SAMPLES},{warmup},{iterations}",
                    shape.name,
                    m,
                    shape.n,
                    shape.k,
                    cutlass_fused / deepgemm_fused,
                    cutlass_graph / deepgemm_graph
                );
            }
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", has_cutlass_fp8_sm90_kernels))]
    #[test]
    fn test_cutlass_blockwise_fp8_cuda_graph() -> Result<()> {
        use candle_core::cuda::cudarc::driver::sys;

        const K: usize = 128;
        const N: usize = 128;
        const ROWS: usize = 8;

        let dev = Device::new_cuda(0)?;
        let weight = Tensor::randn(0f32, 0.25, (N, K), &dev)?.to_dtype(DType::BF16)?;
        let input = Tensor::randn(0f32, 0.25, (ROWS, K), &dev)?.to_dtype(DType::BF16)?;
        let (weight_q, weight_scales) = ops::fp8_blockwise_quantize(&weight, vec![128, 128])?;

        let (warmup_q, warmup_scales) = ops::fp8_quantize_activation_cutlass(&input)?;
        let warmup_output = ops::fp8_blockwise_matmul_cutlass(
            &warmup_q,
            &warmup_scales,
            &weight_q,
            &weight_scales,
            DType::BF16,
        )?;
        drop((warmup_q, warmup_scales, warmup_output));
        dev.synchronize()?;

        let Device::Cuda(cuda_dev) = &dev else {
            unreachable!()
        };
        let stream = cuda_dev.cuda_stream();
        let restore_event_tracking = stream.context().is_event_tracking();
        if restore_event_tracking {
            unsafe { stream.context().disable_event_tracking() };
        }
        if let Err(error) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            if restore_event_tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            return Err(candle_core::Error::msg(error.to_string()));
        }

        let captured = ops::fp8_quantize_activation_cutlass(&input).and_then(
            |(activation, activation_scales)| {
                let output = ops::fp8_blockwise_matmul_cutlass(
                    &activation,
                    &activation_scales,
                    &weight_q,
                    &weight_scales,
                    DType::BF16,
                )?;
                drop((activation, activation_scales, output));
                Ok(())
            },
        );
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if restore_event_tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        captured?;
        let graph = graph
            .map_err(|error| candle_core::Error::msg(error.to_string()))?
            .ok_or_else(|| candle_core::Error::msg("CUDA graph capture returned no graph"))?;
        graph
            .launch()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        graph
            .launch()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        stream
            .synchronize()
            .map_err(|error| candle_core::Error::msg(error.to_string()))?;
        Ok(())
    }
}
