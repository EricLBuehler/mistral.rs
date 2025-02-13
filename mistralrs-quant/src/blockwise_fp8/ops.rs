#[cfg(feature = "cuda")]
use candle_core::cuda::{
    cudarc::driver::{CudaSlice, DeviceRepr, ValidAsZeroBits},
    CudaDevice,
};

use candle_core::{CpuStorage, CustomOp2, DType, Result, Tensor, WithDType};

use float8::F8E4M3;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "cuda")]
use crate::blockwise_fp8::ffi;

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

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn dispatch_cuda_kernel<T: WithDType + DeviceRepr + ValidAsZeroBits>(
        &self,
        weight: &CudaSlice<F8E4M3>,
        d_scale: &CudaSlice<f32>,
        rows: i32,
        cols: i32,
        block_size_rows: i32,
        block_size_cols: i32,
        dev: &CudaDevice,
        kernel: unsafe extern "C" fn(*const F8E4M3, *const f32, *mut T, i32, i32, i32, i32),
    ) -> Result<CudaSlice<T>> {
        use candle_core::cuda::{cudarc::driver::DevicePtr, WrapErr};

        let out = unsafe { dev.alloc::<T>(rows as usize * cols as usize).w()? };
        unsafe {
            kernel(
                (*weight.device_ptr()) as *const _,
                (*d_scale.device_ptr()) as *const _,
                (*out.device_ptr()) as *mut _,
                rows,
                cols,
                block_size_rows,
                block_size_cols,
            )
        };

        Ok(out)
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
        use candle_core::backend::BackendStorage;
        use half::{bf16, f16};

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

        let weight_slice = weight_s.as_cuda_slice::<F8E4M3>()?;
        let scale_slice = scale_s.as_cuda_slice::<f32>()?;

        let (rows, cols) = weight_l.shape().dims2()?;
        let (block_size_rows, block_size_cols) = scale_l.shape().dims2()?;

        let dev = weight_s.device().clone();
        let out = match self.out_ty {
            DType::F32 => candle_core::CudaStorage::wrap_cuda_slice(
                self.dispatch_cuda_kernel::<f32>(
                    weight_slice,
                    scale_slice,
                    rows as i32,
                    cols as i32,
                    block_size_rows as i32,
                    block_size_cols as i32,
                    &dev,
                    ffi::launch_fp8_blockwise_dequantize_f32,
                )?,
                dev,
            ),
            DType::BF16 => candle_core::CudaStorage::wrap_cuda_slice(
                self.dispatch_cuda_kernel::<bf16>(
                    weight_slice,
                    scale_slice,
                    rows as i32,
                    cols as i32,
                    block_size_rows as i32,
                    block_size_cols as i32,
                    &dev,
                    ffi::launch_fp8_blockwise_dequantize_bf16,
                )?,
                dev,
            ),
            DType::F16 => candle_core::CudaStorage::wrap_cuda_slice(
                self.dispatch_cuda_kernel::<f16>(
                    weight_slice,
                    scale_slice,
                    rows as i32,
                    cols as i32,
                    block_size_rows as i32,
                    block_size_cols as i32,
                    &dev,
                    ffi::launch_fp8_blockwise_dequantize_f16,
                )?,
                dev,
            ),
            _ => candle_core::bail!("Invalid out type, expected either f32/bf16/f16"),
        };

        Ok((out, weight_l.shape().clone()))
    }
}

/// FP8 blockwise dequantize.
/// - Expects weight to be fp8
/// - Expects inv_scales to be f32
/// - weight * inv_scale = dequantized
/// - Only works on the CPU
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
}
