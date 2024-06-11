
use std::ffi::c_void;
use candle_core::
use candle_core::{
    backend::BackendStorage, cuda::cudarc::driver::DevicePtr, CpuStorage, CustomOp1, Error, Layout,
    Result, Shape, Tensor, WithDType,
};
use half::{bf16, f16};

use crate::cuda::ffi;

struct NonZero {}
impl NonZero {
    // Sequential version

    fn nonzero<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Vec<u32> {
        let n = layout.dims().len();
        let mut result = Vec::new();
        let mut indices = vec![0u32; n];
        for (i, v) in vs.iter().enumerate() {
            if !v.is_zero() {
                //result.push(i as u32);
                let mut idx = i;
                for (dim_index, dim) in layout.dims().iter().enumerate().rev() {
                    let d = idx % dim;
                    indices[dim_index] = d as u32;
                    idx /= dim;
                }
                result.extend_from_slice(&indices);
            }
        }
        result
    }
    // Parallel version
    /*
    use rayon::prelude::*;
    fn nonzero<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Vec<u32> {
        let n = layout.dims().len();
        let result = vs
            .par_iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if !v.is_zero() {
                    let mut indices = vec![0u32; n];
                    let mut idx = i;
                    for (dim_index, dim) in layout.dims().iter().enumerate().rev() {
                        let d = idx % dim;
                        indices[dim_index] = d as u32;
                        idx /= dim;
                    }
                    Some(indices)
                } else {
                    None
                }
            })
            .flatten()
            .collect();
        result
    }
    */
}

fn count_nonzero(dtype: candle_core::DType, d_in: *const c_void, n: u32) -> u32 {
    unsafe {
        match dtype {
            candle_core::DType::U8 => ffi::count_nonzero_u8(d_in, n),
            candle_core::DType::U32 => ffi::count_nonzero_u32(d_in, n),
            candle_core::DType::I64 => ffi::count_nonzero_i64(d_in, n),
            candle_core::DType::BF16 => ffi::count_nonzero_bf16(d_in, n),
            candle_core::DType::F16 => ffi::count_nonzero_f16(d_in, n),
            candle_core::DType::F32 => ffi::count_nonzero_f32(d_in, n),
            candle_core::DType::F64 => ffi::count_nonzero_f64(d_in, n),
        }
    }
}

fn nonzero(
    dtype: candle_core::DType,
    d_in: *const c_void,
    n: u32,
    num_nonzero: u32,
    dims: *const c_void,
    num_dims: u32,
    d_out: *mut c_void,
) {
    unsafe {
        match dtype {
            candle_core::DType::U8 => ffi::nonzero_u8(d_in, n, num_nonzero, dims, num_dims, d_out),
            candle_core::DType::U32 => {
                ffi::nonzero_u32(d_in, n, num_nonzero, dims, num_dims, d_out)
            }
            candle_core::DType::I64 => {
                ffi::nonzero_i64(d_in, n, num_nonzero, dims, num_dims, d_out)
            }
            candle_core::DType::BF16 => {
                ffi::nonzero_bf16(d_in, n, num_nonzero, dims, num_dims, d_out)
            }
            candle_core::DType::F16 => {
                ffi::nonzero_f16(d_in, n, num_nonzero, dims, num_dims, d_out)
            }
            candle_core::DType::F32 => {
                ffi::nonzero_f32(d_in, n, num_nonzero, dims, num_dims, d_out)
            }
            candle_core::DType::F64 => {
                ffi::nonzero_f64(d_in, n, num_nonzero, dims, num_dims, d_out)
            }
        }
    }
}

impl CustomOp1 for NonZero {
    fn name(&self) -> &'static str {
        "nonzero"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            return Err(Error::RequiresContiguous { op: "nonzero" });
        }
        let result = match storage {
            candle_core::CpuStorage::U8(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::U32(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::I64(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::BF16(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::F16(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::F32(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::F64(vs) => self.nonzero(vs, layout),
        };
        let index_len = layout.dims().len();
        let result_len = result.len() / index_len;
        let result = CpuStorage::U32(result);
        let shape = Shape::from_dims(&[result_len, index_len]);
        Ok((result, shape))
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        if !layout.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "nonzero" });
        }
        let dev = storage.device().clone();
        let d_in = match storage.dtype() {
            candle_core::DType::U8 => *storage.as_cuda_slice::<u8>()?.device_ptr(),
            candle_core::DType::U32 => *storage.as_cuda_slice::<u32>()?.device_ptr(),
            candle_core::DType::I64 => *storage.as_cuda_slice::<i64>()?.device_ptr(),
            candle_core::DType::BF16 => *storage.as_cuda_slice::<bf16>()?.device_ptr(),
            candle_core::DType::F16 => *storage.as_cuda_slice::<f16>()?.device_ptr(),
            candle_core::DType::F32 => *storage.as_cuda_slice::<f32>()?.device_ptr(),
            candle_core::DType::F64 => *storage.as_cuda_slice::<f64>()?.device_ptr(),
        } as *const c_void;
        let n = layout.shape().elem_count();
        let num_nonzero = count_nonzero(storage.dtype(), d_in, n as u32);
        let d_out = unsafe { dev.alloc::<u32>(num_nonzero as usize * layout.dims().len()) }
            .map_err(|_| Error::Msg("Failed to allocate memory for nonzero result".to_string()))?;
        let d_out_ptr = *d_out.device_ptr() as *mut c_void;
        let dims = layout
            .dims()
            .iter()
            .map(|&x| x as u32)
            .collect::<Vec<u32>>();
        let d_dims = dev
            .htod_copy(dims)
            .map_err(|_| Error::Msg("Failed to copy dims to device".to_string()))?;
        let d_dims_ptr = *d_dims.device_ptr() as *const c_void;
        nonzero(
            storage.dtype(),
            d_in,
            n as u32,
            num_nonzero,
            d_dims_ptr,
            layout.dims().len() as u32,
            d_out_ptr,
        );
        let shape = Shape::from_dims(&[num_nonzero as usize, layout.dims().len()]);
        let dst = candle_core::CudaStorage::wrap_cuda_slice(d_out, dev);
        Ok((dst, shape))
    }
}

pub trait NonZeroOp {
    fn nonzero(&self) -> Result<Tensor>;
}

impl NonZeroOp for Tensor {
    fn nonzero(&self) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "nonzero" });
        }
        self.apply_op1_no_bwd(&NonZero {})
    }
}

#[test]
fn test_nonzero_cuda() {
    use crate::NonZeroOp;
    use candle_core::Tensor;
    let device = candle_core::Device::new_cuda(0).unwrap();
    let a = Tensor::from_vec(
        vec![1f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
        &[2, 4],
        &device,
    )
    .unwrap();
    let b = a.nonzero().unwrap();
    println!("b: {}", b);
}

#[test]
fn test_nonzero_cpu() {
    use crate::ops::NonZeroOp;
    use candle_core::Tensor;
    let device = candle_core::Device::Cpu;
    let a = Tensor::from_vec(
        vec![1f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
        &[2, 4],
        &device,
    )
    .unwrap();
    let b = a.nonzero().unwrap();
    println!("b: {}", b);
}
