use candle_core::{
    backend::BackendStorage, CpuStorage, CustomOp1, CustomOp2, DType, Error, Layout, Result, Shape,
    Tensor, WithDType,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use std::ops::{BitOr, Shl};

#[cfg(feature = "cuda")]
use crate::utils::ffi;
#[cfg(feature = "cuda")]
use candle_core::cuda::{cudarc::driver::DevicePtr, CudaStorage, WrapErr};
#[cfg(feature = "cuda")]
use std::ffi::c_void;

struct BitWiseOr;

impl BitWiseOr {
    fn bitwise<T: WithDType + BitOr<Output = T>>(&self, vs1: &[T], vs2: &[T]) -> Vec<T> {
        vs1.into_par_iter()
            .zip_eq(vs2)
            .map(|(v1, v2)| *v1 | *v2)
            .collect()
    }
}

impl CustomOp2 for BitWiseOr {
    fn name(&self) -> &'static str {
        "bitwise-or"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        if l1.shape() != l2.shape() || l1.stride() != l2.stride() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: "bitwise-or",
            });
        }
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: "bitwise-or",
            });
        }
        match s1 {
            CpuStorage::U8(vs1) => {
                let vs2 = &s2.as_slice::<u8>().unwrap();
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::U8(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I16(_) => Err(Error::UnsupportedDTypeForOp(DType::I16, "bitwise-or")),
            CpuStorage::U32(_) => Err(Error::UnsupportedDTypeForOp(DType::U32, "bitwise-or")),
            CpuStorage::I64(_) => Err(Error::UnsupportedDTypeForOp(DType::I64, "bitwise-or")),
            CpuStorage::I32(vs1) => {
                let vs2 = &s2.as_slice::<i32>().unwrap();
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::I32(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::BF16(_) => Err(Error::UnsupportedDTypeForOp(DType::BF16, "bitwise-or")),
            CpuStorage::F16(_) => Err(Error::UnsupportedDTypeForOp(DType::F16, "bitwise-or")),
            CpuStorage::F32(_) => Err(Error::UnsupportedDTypeForOp(DType::F32, "bitwise-or")),
            CpuStorage::F64(_) => Err(Error::UnsupportedDTypeForOp(DType::F64, "bitwise-or")),
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
        if l1.shape() != l2.shape() || l1.stride() != l2.stride() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: "bitwise-or",
            });
        }
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: "bitwise-or",
            });
        }
        let dev = s1.device().clone();
        let (d_in1_ptr, d_in2_ptr, elem_count) = match s1.dtype() {
            DType::U8 => {
                let d_in1_ptr = *s1
                    .as_cuda_slice::<u8>()?
                    .slice(l1.start_offset()..)
                    .device_ptr() as *const c_void;
                let d_in2_ptr = *s2
                    .as_cuda_slice::<u8>()?
                    .slice(l2.start_offset()..)
                    .device_ptr() as *const c_void;
                let elem_count = l1.shape().elem_count();
                (d_in1_ptr, d_in2_ptr, elem_count)
            }
            DType::I16 => {
                return Err(Error::UnsupportedDTypeForOp(DType::I16, "bitwise-or"));
            }
            DType::U32 => {
                return Err(Error::UnsupportedDTypeForOp(DType::U32, "bitwise-or"));
            }
            DType::I64 => {
                return Err(Error::UnsupportedDTypeForOp(DType::I64, "bitwise-or"));
            }
            DType::I32 => {
                let d_in1_ptr = *s1
                    .as_cuda_slice::<i32>()?
                    .slice(l1.start_offset()..)
                    .device_ptr() as *const c_void;
                let d_in2_ptr = *s2
                    .as_cuda_slice::<i32>()?
                    .slice(l2.start_offset()..)
                    .device_ptr() as *const c_void;
                let elem_count = l1.shape().elem_count();
                (d_in1_ptr, d_in2_ptr, elem_count)
            }
            DType::BF16 => {
                return Err(Error::UnsupportedDTypeForOp(DType::BF16, "bitwise-or"));
            }
            DType::F16 => {
                return Err(Error::UnsupportedDTypeForOp(DType::F16, "bitwise-or"));
            }
            DType::F32 => {
                return Err(Error::UnsupportedDTypeForOp(DType::F32, "bitwise-or"));
            }
            DType::F64 => {
                return Err(Error::UnsupportedDTypeForOp(DType::F64, "bitwise-or"));
            }
        };
        let dst = match s1.dtype() {
            DType::U8 => {
                let d_out = unsafe { dev.alloc::<u8>(elem_count) }.w()?;
                let d_out_ptr = *d_out.device_ptr() as *mut c_void;
                unsafe {
                    ffi::mq_bitwise_or_u8(
                        d_in1_ptr,
                        d_in2_ptr,
                        d_out_ptr,
                        u32::try_from(elem_count)?,
                    )
                };
                CudaStorage::wrap_cuda_slice(d_out, dev)
            }
            DType::I32 => {
                let d_out = unsafe { dev.alloc::<i32>(elem_count) }.w()?;
                let d_out_ptr = *d_out.device_ptr() as *mut c_void;
                unsafe {
                    ffi::mq_bitwise_or_i32(
                        d_in1_ptr,
                        d_in2_ptr,
                        d_out_ptr,
                        u32::try_from(elem_count)?,
                    )
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
    fn bitwise_or(&self, rhs: &Tensor) -> Result<Tensor>;
}

impl BitWiseOp for Tensor {
    #[cfg(feature = "metal")]
    fn bitwise_or(&self, rhs: &Tensor) -> Result<Tensor> {
        let original_device = rhs.device();
        self.to_device(&candle_core::Device::Cpu)?
            .apply_op2_no_bwd(&rhs.to_device(&candle_core::Device::Cpu)?, &BitWiseOr)?
            .to_device(original_device)
    }
    #[cfg(not(feature = "metal"))]
    fn bitwise_or(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op2_no_bwd(rhs, &BitWiseOr)
    }
}
struct Leftshift(usize);

impl Leftshift {
    fn leftshift<T: WithDType + Shl<Output = T>>(&self, vs: &[T]) -> Vec<T> {
        let offset = T::from_f64(self.0 as f64);
        vs.into_par_iter().map(|v| *v << offset).collect()
    }
}

impl CustomOp1 for Leftshift {
    fn name(&self) -> &'static str {
        "left"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout) -> Result<(CpuStorage, Shape)> {
        match s1 {
            CpuStorage::U8(vs1) => {
                let result = self.leftshift(vs1);
                let result = CpuStorage::U8(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I16(_) => Err(Error::UnsupportedDTypeForOp(DType::I16, "leftshifr")),
            CpuStorage::U32(_) => Err(Error::UnsupportedDTypeForOp(DType::U32, "leftshifr")),
            CpuStorage::I64(_) => Err(Error::UnsupportedDTypeForOp(DType::I64, "leftshifr")),
            CpuStorage::I32(vs1) => {
                let result = self.leftshift(vs1);
                let result = CpuStorage::I32(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::BF16(_) => Err(Error::UnsupportedDTypeForOp(DType::BF16, "leftshifr")),
            CpuStorage::F16(_) => Err(Error::UnsupportedDTypeForOp(DType::F16, "leftshifr")),
            CpuStorage::F32(_) => Err(Error::UnsupportedDTypeForOp(DType::F32, "leftshifr")),
            CpuStorage::F64(_) => Err(Error::UnsupportedDTypeForOp(DType::F64, "leftshifr")),
        }
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s1: &CudaStorage, l1: &Layout) -> Result<(CudaStorage, Shape)> {
        let dev = s1.device().clone();
        let (d_in1_ptr, elem_count) = match s1.dtype() {
            DType::U8 => {
                let d_in1_ptr = *s1
                    .as_cuda_slice::<u8>()?
                    .slice(l1.start_offset()..)
                    .device_ptr() as *const c_void;
                let elem_count = l1.shape().elem_count();
                (d_in1_ptr, elem_count)
            }
            DType::I16 => {
                return Err(Error::UnsupportedDTypeForOp(DType::I16, "leftshift"));
            }
            DType::U32 => {
                return Err(Error::UnsupportedDTypeForOp(DType::U32, "leftshift"));
            }
            DType::I64 => {
                return Err(Error::UnsupportedDTypeForOp(DType::I64, "leftshift"));
            }
            DType::I32 => {
                let d_in1_ptr = *s1
                    .as_cuda_slice::<i32>()?
                    .slice(l1.start_offset()..)
                    .device_ptr() as *const c_void;
                let elem_count = l1.shape().elem_count();
                (d_in1_ptr, elem_count)
            }
            DType::BF16 => {
                return Err(Error::UnsupportedDTypeForOp(DType::BF16, "leftshift"));
            }
            DType::F16 => {
                return Err(Error::UnsupportedDTypeForOp(DType::F16, "leftshift"));
            }
            DType::F32 => {
                return Err(Error::UnsupportedDTypeForOp(DType::F32, "leftshift"));
            }
            DType::F64 => {
                return Err(Error::UnsupportedDTypeForOp(DType::F64, "leftshift"));
            }
        };
        let dst = match s1.dtype() {
            DType::U8 => {
                let d_out = unsafe { dev.alloc::<u8>(elem_count) }.w()?;
                let d_out_ptr = *d_out.device_ptr() as *mut c_void;
                unsafe {
                    ffi::mq_leftshift_u8(
                        d_in1_ptr,
                        d_out_ptr,
                        u32::try_from(elem_count)?,
                        self.0 as i32,
                    )
                };
                CudaStorage::wrap_cuda_slice(d_out, dev)
            }
            DType::I32 => {
                let d_out = unsafe { dev.alloc::<i32>(elem_count) }.w()?;
                let d_out_ptr = *d_out.device_ptr() as *mut c_void;
                unsafe {
                    ffi::mq_leftshift_i32(
                        d_in1_ptr,
                        d_out_ptr,
                        u32::try_from(elem_count)?,
                        self.0 as i32,
                    )
                };
                CudaStorage::wrap_cuda_slice(d_out, dev)
            }
            _ => unreachable!(),
        };
        Ok((dst, l1.shape().clone()))
    }
}

#[allow(dead_code)]
pub trait LeftshiftOp {
    fn leftshift(&self, n: usize) -> Result<Tensor>;
}

impl LeftshiftOp for Tensor {
    #[cfg(feature = "metal")]
    fn leftshift(&self, n: usize) -> Result<Tensor> {
        let original_device = self.device();
        self.to_device(&candle_core::Device::Cpu)?
            .apply_op1_no_bwd(&Leftshift(n))?
            .to_device(original_device)
    }
    #[cfg(not(feature = "metal"))]
    fn leftshift(&self, n: usize) -> Result<Tensor> {
        self.apply_op1_no_bwd(&Leftshift(n))
    }
}

mod tests {
    #[test]
    fn test_bitwise_or_cpu() {
        use crate::utils::ops::BitWiseOp;
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        let a =
            Tensor::from_vec(vec![1i32, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![-1i32, 0, 0, 0, 0, 0, 0, 0, 0, 8], (5, 2), &device).unwrap();
        let c = a.bitwise_or(&b).unwrap().to_vec2::<i32>().unwrap();
        assert_eq!(c, [[-1, 2], [3, -1], [-1, -1], [-1, 4], [5, 15]]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_bitwise_or_cuda() {
        use crate::utils::ops::BitWiseOp;
        use candle_core::Tensor;
        let device = candle_core::Device::new_cuda(0).unwrap();
        let a =
            Tensor::from_vec(vec![1i32, 2, 3, -1, -1, -1, -1, 4, 5, 7], (5, 2), &device).unwrap();
        let b = Tensor::from_vec(vec![-1i32, 0, 0, 0, 0, 0, 0, 0, 0, 8], (5, 2), &device).unwrap();
        let c = a.bitwise_or(&b).unwrap().to_vec2::<i32>().unwrap();
        assert_eq!(c, [[-1, 2], [3, -1], [-1, -1], [-1, 4], [5, 15]]);
    }

    #[test]
    fn test_leftshift_cpu() {
        use crate::utils::ops::LeftshiftOp;
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        let a = Tensor::from_vec(vec![1i32, 2, 3, 4, 5, 6], (3, 2), &device).unwrap();
        let c = a.leftshift(2).unwrap().to_vec2::<i32>().unwrap();
        assert_eq!(c, [[4, 8], [12, 16], [20, 24]]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_leftshift_cuda() {
        use crate::utils::ops::LeftshiftOp;
        use candle_core::Tensor;
        let device = candle_core::Device::new_cuda(0).unwrap();
        let a = Tensor::from_vec(vec![1i32, 2, 3, 4, 5, 6], (3, 2), &device).unwrap();
        let c = a.leftshift(2).unwrap().to_vec2::<i32>().unwrap();
        assert_eq!(c, [[4, 8], [12, 16], [20, 24]]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_bitwise_or_and_leftshift_cuda() {
        use crate::utils::{ops::BitWiseOp, LeftshiftOp};
        use candle_core::Tensor;
        let device = candle_core::Device::new_cuda(0).unwrap();
        let a = Tensor::from_vec(vec![0b00001111u8], (1,), &device).unwrap();
        let b = Tensor::from_vec(vec![0b00001111u8], (1,), &device).unwrap();
        let c = a
            .leftshift(4)
            .unwrap()
            .bitwise_or(&b)
            .unwrap()
            .to_vec1::<u8>()
            .unwrap();
        let av = a.to_vec1::<u8>().unwrap();
        let bv = b.to_vec1::<u8>().unwrap();
        assert_eq!(av, [0b00001111]);
        assert_eq!(bv, [0b00001111]);
        assert_eq!(c, [0b11111111]);
    }
}
