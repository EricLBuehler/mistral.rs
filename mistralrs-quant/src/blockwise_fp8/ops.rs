use candle_core::{CpuStorage, CustomOp2, DType, Result, Tensor, WithDType};
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
        use candle_core::{
            backend::BackendStorage,
            cuda::{cudarc::driver::DevicePtr, WrapErr},
            CudaStorage,
        };
        use half::{bf16, f16};

        use crate::blockwise_fp8::ffi;

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

        let weight = weight_s
            .as_cuda_slice::<F8E4M3>()?
            .slice(weight_l.start_offset()..);
        let scale = scale_s
            .as_cuda_slice::<f32>()?
            .slice(scale_l.start_offset()..);

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
                    .alloc_zeros::<f32>(weight_l.shape().elem_count())
                    .w()?;
                unsafe {
                    ffi::launch_dequant_fp8_blockwise_kernel_f32(
                        (*weight.device_ptr()) as *const _,
                        (*scale.device_ptr()) as *const _,
                        (*output.device_ptr()) as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                    )
                };
                CudaStorage::wrap_cuda_slice(output, weight_s.device().clone())
            }
            DType::F16 => {
                let output = weight_s
                    .device()
                    .alloc_zeros::<f16>(weight_l.shape().elem_count())
                    .w()?;
                unsafe {
                    ffi::launch_dequant_fp8_blockwise_kernel_f16(
                        (*weight.device_ptr()) as *const _,
                        (*scale.device_ptr()) as *const _,
                        (*output.device_ptr()) as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                    )
                };
                CudaStorage::wrap_cuda_slice(output, weight_s.device().clone())
            }
            DType::BF16 => {
                let output = weight_s
                    .device()
                    .alloc_zeros::<bf16>(weight_l.shape().elem_count())
                    .w()?;
                unsafe {
                    ffi::launch_dequant_fp8_blockwise_kernel_bf16(
                        (*weight.device_ptr()) as *const _,
                        (*scale.device_ptr()) as *const _,
                        (*output.device_ptr()) as *mut _,
                        weight_height,
                        weight_width,
                        weight_row_stride,
                        scale_stride,
                        weight_block_size_y,
                        weight_block_size_x,
                    )
                };
                CudaStorage::wrap_cuda_slice(output, weight_s.device().clone())
            }
            other => candle_core::bail!("unexpected out type of fp8 blockwise dequant {other:?}"),
        };

        Ok((res, weight_l.shape().clone()))
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

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};
    use half::bf16;

    use crate::blockwise_fp8::ops;

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
            let weight = Tensor::ones((5, 5), DType::F8E4M3, dev)?;
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
}
