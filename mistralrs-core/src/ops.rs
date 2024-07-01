use candle_core::{
    backend::BackendStorage, CpuStorage, CustomOp1, CustomOp2, DType, Error, Layout, Result, Shape,
    Tensor, WithDType, D,
};

use std::{
    fmt::Display,
    ops::{BitAnd, BitOr, BitXor},
};

#[cfg(feature = "cuda")]
use crate::cuda::ffi;
#[cfg(feature = "cuda")]
use candle_core::cuda::{cudarc::driver::DevicePtr, CudaStorage, WrapErr};
#[cfg(feature = "cuda")]
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use std::ffi::c_void;
pub enum BitWiseOpEnum {
    And,
    Or,
    Xor,
}

impl Display for BitWiseOpEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BitWiseOpEnum::And => write!(f, "And"),
            BitWiseOpEnum::Or => write!(f, "Or"),
            BitWiseOpEnum::Xor => write!(f, "Xor"),
        }
    }
}

struct BitWise {
    pub op: BitWiseOpEnum,
}

impl BitWise {
    pub fn new(op: BitWiseOpEnum) -> Self {
        Self { op }
    }

    fn bitwise<T: WithDType + BitAnd<Output = T> + BitOr<Output = T> + BitXor<Output = T>>(
        &self,
        vs1: &[T],
        vs2: &[T],
    ) -> Vec<T> {
        let n = vs1.len();
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let v1 = vs1[i];
            let v2 = vs2[i];
            let r = match self.op {
                BitWiseOpEnum::And => v1 & v2,
                BitWiseOpEnum::Or => v1 | v2,
                BitWiseOpEnum::Xor => v1 ^ v2,
            };
            result.push(r);
        }
        result
    }
}

impl CustomOp2 for BitWise {
    fn name(&self) -> &'static str {
        "bitwise"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if l1 != l2 {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: "bitwise",
            });
        }
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: "bitwise",
            });
        }
        match s1 {
            CpuStorage::U8(vs1) => {
                let vs2 = s2.as_slice::<u8>().unwrap();
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::U8(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::U32(vs1) => {
                let vs2 = s2.as_slice::<u32>().unwrap();
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::U32(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I64(vs1) => {
                let vs2 = s2.as_slice::<i64>().unwrap();
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::I64(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::BF16(_) => Err(Error::UnsupportedDTypeForOp(DType::BF16, "bitwise")),
            CpuStorage::F16(_) => Err(Error::UnsupportedDTypeForOp(DType::F16, "bitwise")),
            CpuStorage::F32(_) => Err(Error::UnsupportedDTypeForOp(DType::F32, "bitwise")),
            CpuStorage::F64(_) => Err(Error::UnsupportedDTypeForOp(DType::F64, "bitwise")),
        }
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        if l1 != l2 {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: "bitwise",
            });
        }
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: "bitwise",
            });
        }
        let dev = s1.device().clone();
        let (d_in1_ptr, d_in2_ptr, elem_count) = match s1.dtype() {
            DType::U8 => {
                let d_in1_ptr = *s1.as_cuda_slice::<u8>()?.device_ptr() as *const c_void;
                let d_in2_ptr = *s2.as_cuda_slice::<u8>()?.device_ptr() as *const c_void;
                let elem_count = l1.shape().elem_count();
                (d_in1_ptr, d_in2_ptr, elem_count)
            }
            DType::U32 => {
                let d_in1_ptr = *s1.as_cuda_slice::<u32>()?.device_ptr() as *const c_void;
                let d_in2_ptr = *s2.as_cuda_slice::<u32>()?.device_ptr() as *const c_void;
                let elem_count = l1.shape().elem_count();
                (d_in1_ptr, d_in2_ptr, elem_count)
            }
            DType::I64 => {
                let d_in1_ptr = *s1.as_cuda_slice::<i64>()?.device_ptr() as *const c_void;
                let d_in2_ptr = *s2.as_cuda_slice::<i64>()?.device_ptr() as *const c_void;
                let elem_count = l1.shape().elem_count();
                (d_in1_ptr, d_in2_ptr, elem_count)
            }
            DType::BF16 => {
                return Err(Error::UnsupportedDTypeForOp(DType::BF16, "bitwise"));
            }
            DType::F16 => {
                return Err(Error::UnsupportedDTypeForOp(DType::F16, "bitwise"));
            }
            DType::F32 => {
                return Err(Error::UnsupportedDTypeForOp(DType::F32, "bitwise"));
            }
            DType::F64 => {
                return Err(Error::UnsupportedDTypeForOp(DType::F64, "bitwise"));
            }
        };
        let dst = match s1.dtype() {
            DType::U8 => {
                let d_out = unsafe { dev.alloc::<u8>(elem_count) }.w()?;
                let d_out_ptr = *d_out.device_ptr() as *mut c_void;
                unsafe {
                    match self.op {
                        BitWiseOpEnum::And => ffi::bitwise_and_u8(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseOpEnum::Or => ffi::bitwise_or_u8(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseOpEnum::Xor => ffi::bitwise_xor_u8(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                    }
                };
                CudaStorage::wrap_cuda_slice(d_out, dev)
            }
            DType::U32 => {
                let d_out = unsafe { dev.alloc::<u32>(elem_count) }.w()?;
                let d_out_ptr = *d_out.device_ptr() as *mut c_void;
                unsafe {
                    match self.op {
                        BitWiseOpEnum::And => ffi::bitwise_and_u32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseOpEnum::Or => ffi::bitwise_or_u32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseOpEnum::Xor => ffi::bitwise_xor_u32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                    }
                };
                CudaStorage::wrap_cuda_slice(d_out, dev)
            }
            DType::I64 => {
                let d_out = unsafe { dev.alloc::<i64>(elem_count) }.w()?;
                let d_out_ptr = *d_out.device_ptr() as *mut c_void;
                unsafe {
                    match self.op {
                        BitWiseOpEnum::And => ffi::bitwise_and_i64(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseOpEnum::Or => ffi::bitwise_or_i64(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseOpEnum::Xor => ffi::bitwise_xor_i64(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                    }
                };
                CudaStorage::wrap_cuda_slice(d_out, dev)
            }
            _ => unreachable!(),
        };
        Ok((dst, l1.shape().clone()))
    }
}

#[allow(dead_code)]
pub trait BitWiseOp {
    fn bitwise_and(&self, rhs: &Tensor) -> Result<Tensor>;
    fn bitwise_or(&self, rhs: &Tensor) -> Result<Tensor>;
    fn bitwise_xor(&self, rhs: &Tensor) -> Result<Tensor>;
}

impl BitWiseOp for Tensor {
    #[cfg(feature = "metal")]
    fn bitwise_and(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_device = rhs.device();
        self.to_device(&candle_core::Device::Cpu)?
            .apply_op2_no_bwd(
                &rhs.to_device(&candle_core::Device::Cpu)?,
                &BitWise::new(BitWiseOpEnum::And),
            )?
            .to_device(original_device)
    }
    #[cfg(not(feature = "metal"))]
    fn bitwise_and(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op2_no_bwd(rhs, &BitWise::new(BitWiseOpEnum::And))
    }

    #[cfg(feature = "metal")]
    fn bitwise_or(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_device = rhs.device();
        self.to_device(&candle_core::Device::Cpu)?
            .apply_op2_no_bwd(
                &rhs.to_device(&candle_core::Device::Cpu)?,
                &BitWise::new(BitWiseOpEnum::Or),
            )?
            .to_device(original_device)
    }
    #[cfg(not(feature = "metal"))]
    fn bitwise_or(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op2_no_bwd(rhs, &BitWise::new(BitWiseOpEnum::Or))
    }

    #[cfg(feature = "metal")]
    fn bitwise_xor(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_device = rhs.device();
        self.to_device(&candle_core::Device::Cpu)?
            .apply_op2_no_bwd(
                &rhs.to_device(&candle_core::Device::Cpu)?,
                &BitWise::new(BitWiseOpEnum::Xor),
            )?
            .to_device(original_device)
    }
    #[cfg(not(feature = "metal"))]
    fn bitwise_xor(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op2_no_bwd(rhs, &BitWise::new(BitWiseOpEnum::Xor))
    }
}

struct NonZero {}
impl NonZero {
    // Sequential version

    fn nonzero<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Vec<u32> {
        let n = layout.dims().len();
        let mut result = Vec::new();
        let mut indices = vec![0u32; n];
        for (i, v) in vs.iter().enumerate() {
            if !v.is_zero() {
                let mut idx = i;
                for (dim_index, dim) in layout.dims().iter().enumerate().rev() {
                    let d = idx % dim;
                    indices[dim_index] = u32::try_from(d).unwrap();
                    idx /= dim;
                }
                result.extend_from_slice(&indices);
            }
        }
        result
    }
}

#[cfg(feature = "cuda")]
fn count_nonzero_cuda(dtype: candle_core::DType, d_in: *const c_void, n: u32) -> u32 {
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

#[cfg(feature = "cuda")]
fn nonzero_cuda(
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
    #[cfg(feature = "cuda")]
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
        let num_nonzero = count_nonzero_cuda(storage.dtype(), d_in, u32::try_from(n)?);
        let d_out = unsafe { dev.alloc::<u32>(num_nonzero as usize * layout.dims().len()) }
            .map_err(|_| Error::Msg("Failed to allocate memory for nonzero result".to_string()))?;
        let d_out_ptr = *d_out.device_ptr() as *mut c_void;
        let dims = layout
            .dims()
            .iter()
            .map(|&x| u32::try_from(x).unwrap())
            .collect::<Vec<u32>>();
        let d_dims = dev
            .htod_copy(dims)
            .map_err(|_| Error::Msg("Failed to copy dims to device".to_string()))?;
        let d_dims_ptr = *d_dims.device_ptr() as *const c_void;
        nonzero_cuda(
            storage.dtype(),
            d_in,
            u32::try_from(n)?,
            num_nonzero,
            d_dims_ptr,
            u32::try_from(layout.dims().len())?,
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
    #[cfg(feature = "metal")]
    fn nonzero(&self) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "nonzero" });
        }
        let original_device = self.device();
        self.to_device(&candle_core::Device::Cpu)?
            .apply_op1_no_bwd(&NonZero {})?
            .to_device(original_device)
    }
    #[cfg(not(feature = "metal"))]
    fn nonzero(&self) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "nonzero" });
        }
        self.apply_op1_no_bwd(&NonZero {})
    }
}

#[allow(dead_code)]
pub struct TopKOutput {
    pub values: Tensor,
    pub indices: Tensor,
}

pub trait TopKLastDimOp {
    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    fn topk(&self, topk: usize) -> Result<TopKOutput>;
}

impl TopKLastDimOp for Tensor {
    fn topk(&self, topk: usize) -> Result<TopKOutput> {
        // Sorted descending
        let sorted_indices = self.arg_sort_last_dim(false)?;
        let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
        Ok(TopKOutput {
            values: self.gather(&topk_indices, D::Minus1)?,
            indices: topk_indices,
        })
    }
}

mod tests {
    #[test]
    fn test_topk() {
        use crate::ops::{TopKLastDimOp, TopKOutput};
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        //  [[1, 3, 5],
        //   [2, 4, 6]]
        let x = Tensor::arange(1f32, 7f32, &device)
            .unwrap()
            .reshape((3, 2))
            .unwrap()
            .t()
            .unwrap()
            .contiguous()
            .unwrap();
        let TopKOutput { values, indices } = x.topk(2).unwrap();
        assert_eq!(
            x.to_vec2::<f32>().unwrap(),
            vec![vec![1f32, 3f32, 5f32], vec![2f32, 4f32, 6f32]]
        );
        assert_eq!(
            values.to_vec2::<f32>().unwrap(),
            vec![vec![5f32, 3f32], vec![6f32, 4f32]]
        );
        assert_eq!(
            indices.to_vec2::<u32>().unwrap(),
            vec![vec![2u32, 1u32], vec![2u32, 1u32]]
        );
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
        let b = a.nonzero().unwrap().to_vec2::<u32>().unwrap();
        assert_eq!(b, [[0, 0], [0, 2], [1, 0], [1, 2]]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_nonzero_cuda() {
        use crate::ops::NonZeroOp;
        use candle_core::Tensor;
        let device = candle_core::Device::new_cuda(0).unwrap();
        let a = Tensor::from_vec(
            vec![1f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0],
            &[2, 4],
            &device,
        )
        .unwrap();
        let b = a.nonzero().unwrap().to_vec2::<u32>().unwrap();
        assert_eq!(b, [[0, 0], [0, 2], [1, 0], [1, 2]]);
    }

    #[test]
    fn test_bitwise_and_cpu() {
        use crate::ops::BitWiseOp;
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        let a =
            Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
        let b =
            Tensor::from_vec(vec![-1i64, 2, 3, -1, 1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
        let c = a.bitwise_and(&b).unwrap().to_vec2::<i64>().unwrap();
        assert_eq!(c, [[1, 2], [3, -1], [1, -1], [-1, 4], [5, 7]]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_bitwise_and_cuda() {
        use crate::ops::BitWiseOp;
        use candle_core::Tensor;
        let device = candle_core::Device::new_cuda(0).unwrap();
        let a =
            Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
        let b =
            Tensor::from_vec(vec![-1i64, 2, 3, -1, 1, -1, -1, 4, 0, 7], (5, 2), &device).unwrap();
        let c = a.bitwise_and(&b).unwrap().to_vec2::<i64>().unwrap();
        assert_eq!(c, [[1, 2], [3, -1], [1, -1], [-1, 4], [0, 7]]);
    }

    #[test]
    fn test_bitwise_or_cpu() {
        use crate::ops::BitWiseOp;
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        let a =
            Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![-1i64, 0, 0, 0, 0, 0, 0, 0, 0, 8], (5, 2), &device).unwrap();
        let c = a.bitwise_or(&b).unwrap().to_vec2::<i64>().unwrap();
        assert_eq!(c, [[-1, 2], [3, -1], [-1, -1], [-1, 4], [5, 15]]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_bitwise_or_cuda() {
        use crate::ops::BitWiseOp;
        use candle_core::Tensor;
        let device = candle_core::Device::new_cuda(0).unwrap();
        let a =
            Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![-1i64, 0, 0, 0, 0, 0, 0, 0, 0, 8], (5, 2), &device).unwrap();
        let c = a.bitwise_or(&b).unwrap().to_vec2::<i64>().unwrap();
        assert_eq!(c, [[-1, 2], [3, -1], [-1, -1], [-1, 4], [5, 15]]);
    }

    #[test]
    fn test_bitwise_xor_cpu() {
        use crate::ops::BitWiseOp;
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        let a =
            Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![-1i64, 0, 0, 0, 0, 0, 0, 0, 0, 8], (5, 2), &device).unwrap();
        let c = a.bitwise_xor(&b).unwrap().to_vec2::<i64>().unwrap();
        assert_eq!(c, [[-2, 2], [3, -1], [-1, -1], [-1, 4], [5, 15]]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_bitwise_xor_cuda() {
        use crate::ops::BitWiseOp;
        use candle_core::Tensor;
        let device = candle_core::Device::new_cuda(0).unwrap();
        let a =
            Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![-1i64, 0, 0, 0, 0, 0, 0, 0, 0, 8], (5, 2), &device).unwrap();
        let c = a.bitwise_xor(&b).unwrap().to_vec2::<i64>().unwrap();
        assert_eq!(c, [[-2, 2], [3, -1], [-1, -1], [-1, 4], [5, 15]]);
    }

    #[test]
    fn test_nonzero_and() {
        use crate::ops::{BitWiseOp, NonZeroOp};
        use candle_core::{Device, Tensor};

        let input1 = Tensor::from_vec(
            vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7],
            (10,),
            &Device::Cpu,
        )
        .unwrap();
        let input2 = Tensor::from_vec(
            vec![-1i64, 2, 3, -1, 1, -1, -1, 4, 5, 7],
            (10,),
            &Device::Cpu,
        )
        .unwrap();
        let input = Tensor::stack(&[input1, input2], 0).unwrap();

        let lt = input.lt(0.0).unwrap();
        let gt = input.gt(-10.0).unwrap();
        let res = lt
            .bitwise_and(&gt)
            .unwrap()
            .nonzero()
            .unwrap()
            .to_vec2::<u32>()
            .unwrap();

        assert_eq!(
            res,
            [
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [1, 0],
                [1, 3],
                [1, 5],
                [1, 6]
            ]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn nonzero_and_cuda() {
        use crate::ops::{BitWiseOp, NonZeroOp};
        use candle_core::{Device, Tensor};

        let device = Device::new_cuda(0).unwrap();
        let input1 =
            Tensor::from_vec(vec![1i64, 2, 3, -1, -1, -1, -1, 4, 5, 7], (10,), &device).unwrap();
        let input2 =
            Tensor::from_vec(vec![-1i64, 2, 3, -1, 1, -1, -1, 4, 5, 7], (10,), &device).unwrap();
        let input = Tensor::stack(&[input1, input2], 0).unwrap();

        let lt = input.lt(0.0).unwrap();
        let gt = input.gt(-10.0).unwrap();
        let res = lt
            .bitwise_and(&gt)
            .unwrap()
            .nonzero()
            .unwrap()
            .to_vec2::<u32>()
            .unwrap();

        assert_eq!(
            res,
            [
                [0, 3],
                [0, 4],
                [0, 5],
                [0, 6],
                [1, 0],
                [1, 3],
                [1, 5],
                [1, 6]
            ]
        );
    }
}
