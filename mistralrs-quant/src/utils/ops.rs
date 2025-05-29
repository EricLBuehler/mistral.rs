use candle_core::{
    backend::BackendStorage, shape::Dim, CpuStorage, CustomOp1, CustomOp2, DType, Error, Layout,
    Result, Shape, Tensor, WithDType,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use std::{
    fmt::Display,
    ops::{BitAnd, BitOr, BitXor, Not, Shl},
};

#[cfg(feature = "cuda")]
use crate::utils::ffi;
#[cfg(feature = "cuda")]
use candle_core::cuda::{cudarc::driver::DevicePtr, CudaStorage, WrapErr};
#[cfg(feature = "cuda")]
use std::ffi::c_void;

#[cfg(feature = "metal")]
use crate::metal_kernels::SortScratchCache; // re‑export for clarity
#[cfg(feature = "metal")]
use std::sync::OnceLock;

#[cfg(feature = "metal")]
static SORT_SCRATCH_CACHE: OnceLock<SortScratchCache> = OnceLock::new();

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
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.leftshift(vs1);
                let result = CpuStorage::U8(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I16(vs1) => {
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.leftshift(vs1);
                let result = CpuStorage::I16(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::U32(vs1) => {
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.leftshift(vs1);
                let result = CpuStorage::U32(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I64(vs1) => {
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.leftshift(vs1);
                let result = CpuStorage::I64(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I32(vs1) => {
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.leftshift(vs1);
                let result = CpuStorage::I32(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::BF16(_) => Err(Error::UnsupportedDTypeForOp(DType::BF16, "leftshifr")),
            CpuStorage::F16(_) => Err(Error::UnsupportedDTypeForOp(DType::F16, "leftshifr")),
            CpuStorage::F32(_) => Err(Error::UnsupportedDTypeForOp(DType::F32, "leftshifr")),
            CpuStorage::F64(_) => Err(Error::UnsupportedDTypeForOp(DType::F64, "leftshifr")),
            CpuStorage::F8E4M3(_) => Err(Error::UnsupportedDTypeForOp(DType::F8E4M3, "leftshifr")),
        }
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s1: &CudaStorage, l1: &Layout) -> Result<(CudaStorage, Shape)> {
        if !l1.is_contiguous() {
            candle_core::bail!("Input tensor s1 must be contiguous");
        }
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
            DType::F8E4M3 => {
                return Err(Error::UnsupportedDTypeForOp(DType::F8E4M3, "leftshift"));
            }
        };
        let dst = match s1.dtype() {
            DType::U8 => {
                let d_out = unsafe { dev.alloc::<u8>(elem_count) }.w()?;
                let d_out_ptr = *d_out.device_ptr() as *mut c_void;
                unsafe {
                    ffi::leftshift_u8(
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
                    ffi::leftshift_i32(
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
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        if !l1.is_contiguous() {
            candle_core::bail!("Input tensor s1 must be contiguous");
        }

        let command_buffer = s1.device().command_buffer()?;
        command_buffer.set_label("bitwise-leftshift");

        let device = s1.device();

        let out_shape = l1.shape().clone();

        let output = device.new_buffer(out_shape.elem_count(), s1.dtype(), "bitwise-leftshift")?;

        crate::metal_kernels::call_bitwise_leftshift(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            s1.dtype(),
            s1.buffer(),
            l1.start_offset(),
            self.0 as u32,
            out_shape.elem_count(),
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        let newstorage = candle_core::MetalStorage::new(
            output,
            device.clone(),
            out_shape.elem_count(),
            s1.dtype(),
        );
        Ok((newstorage, out_shape))
    }
}

#[allow(dead_code)]
pub trait LeftshiftOp {
    fn leftshift(&self, n: usize) -> Result<Tensor>;
}

impl LeftshiftOp for Tensor {
    fn leftshift(&self, n: usize) -> Result<Tensor> {
        self.apply_op1_no_bwd(&Leftshift(n))
    }
}

pub enum BitWiseBinaryOpEnum {
    And,
    Or,
    Xor,
}

impl Display for BitWiseBinaryOpEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BitWiseBinaryOpEnum::And => write!(f, "And"),
            BitWiseBinaryOpEnum::Or => write!(f, "Or"),
            BitWiseBinaryOpEnum::Xor => write!(f, "Xor"),
        }
    }
}

pub enum BitWiseUnaryOpEnum {
    Not,
}

impl Display for BitWiseUnaryOpEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BitWiseUnaryOpEnum::Not => write!(f, "Not"),
        }
    }
}

struct BitWise {
    pub op: BitWiseBinaryOpEnum,
}

impl BitWise {
    pub fn new(op: BitWiseBinaryOpEnum) -> Self {
        Self { op }
    }

    fn bitwise<T: WithDType + BitAnd<Output = T> + BitOr<Output = T> + BitXor<Output = T>>(
        &self,
        vs1: &[T],
        vs2: &[T],
    ) -> Vec<T> {
        vs1.into_par_iter()
            .zip_eq(vs2)
            .map(|(v1, v2)| match self.op {
                BitWiseBinaryOpEnum::And => *v1 & *v2,
                BitWiseBinaryOpEnum::Or => *v1 | *v2,
                BitWiseBinaryOpEnum::Xor => *v1 ^ *v2,
            })
            .collect()
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
        if l1.shape() != l2.shape() || l1.stride() != l2.stride() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: "bitwise-op",
            });
        }
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: "bitwise-op",
            });
        }
        if !l1.is_contiguous() {
            candle_core::bail!("Input tensor s1 must be contiguous");
        }
        if !l2.is_contiguous() {
            candle_core::bail!("Input tensor s2 must be contiguous");
        }

        match s1 {
            CpuStorage::U8(vs1) => {
                let vs2 = s2.as_slice::<u8>().unwrap();
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let vs2 = match l2.contiguous_offsets() {
                    Some((a, b)) => &vs2[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::U8(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::U32(vs1) => {
                let vs2 = s2.as_slice::<u32>().unwrap();
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let vs2 = match l2.contiguous_offsets() {
                    Some((a, b)) => &vs2[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::U32(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I64(vs1) => {
                let vs2 = s2.as_slice::<i64>().unwrap();
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let vs2 = match l2.contiguous_offsets() {
                    Some((a, b)) => &vs2[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::I64(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I16(vs1) => {
                let vs2 = s2.as_slice::<i16>().unwrap();
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let vs2 = match l2.contiguous_offsets() {
                    Some((a, b)) => &vs2[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::I16(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I32(vs1) => {
                let vs2 = s2.as_slice::<i32>().unwrap();
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let vs2 = match l2.contiguous_offsets() {
                    Some((a, b)) => &vs2[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.bitwise(vs1, vs2);
                let result = CpuStorage::I32(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::BF16(_) => Err(Error::UnsupportedDTypeForOp(DType::BF16, "bitwise")),
            CpuStorage::F16(_) => Err(Error::UnsupportedDTypeForOp(DType::F16, "bitwise")),
            CpuStorage::F32(_) => Err(Error::UnsupportedDTypeForOp(DType::F32, "bitwise")),
            CpuStorage::F64(_) => Err(Error::UnsupportedDTypeForOp(DType::F64, "bitwise")),
            CpuStorage::F8E4M3(_) => Err(Error::UnsupportedDTypeForOp(DType::F8E4M3, "bitwise")),
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
                op: "bitwise-op",
            });
        }
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: "bitwise-op",
            });
        }
        if !l1.is_contiguous() {
            candle_core::bail!("Input tensor s1 must be contiguous");
        }
        if !l2.is_contiguous() {
            candle_core::bail!("Input tensor s2 must be contiguous");
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
            DType::U32 => {
                let d_in1_ptr = *s1
                    .as_cuda_slice::<u32>()?
                    .slice(l1.start_offset()..)
                    .device_ptr() as *const c_void;
                let d_in2_ptr = *s2
                    .as_cuda_slice::<u32>()?
                    .slice(l2.start_offset()..)
                    .device_ptr() as *const c_void;
                let elem_count = l1.shape().elem_count();
                (d_in1_ptr, d_in2_ptr, elem_count)
            }
            DType::I64 => {
                let d_in1_ptr = *s1
                    .as_cuda_slice::<i64>()?
                    .slice(l1.start_offset()..)
                    .device_ptr() as *const c_void;
                let d_in2_ptr = *s2
                    .as_cuda_slice::<i64>()?
                    .slice(l2.start_offset()..)
                    .device_ptr() as *const c_void;
                let elem_count = l1.shape().elem_count();
                (d_in1_ptr, d_in2_ptr, elem_count)
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
            DType::I16 => {
                let d_in1_ptr = *s1
                    .as_cuda_slice::<i16>()?
                    .slice(l1.start_offset()..)
                    .device_ptr() as *const c_void;
                let d_in2_ptr = *s2
                    .as_cuda_slice::<i16>()?
                    .slice(l2.start_offset()..)
                    .device_ptr() as *const c_void;
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
            DType::F8E4M3 => {
                return Err(Error::UnsupportedDTypeForOp(DType::F8E4M3, "bitwise"));
            }
        };
        let dst = match s1.dtype() {
            DType::U8 => {
                let d_out = unsafe { dev.alloc::<u8>(elem_count) }.w()?;
                let d_out_ptr = *d_out.device_ptr() as *mut c_void;
                unsafe {
                    match self.op {
                        BitWiseBinaryOpEnum::And => ffi::bitwise_and_u8(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Or => ffi::bitwise_or_u8(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Xor => ffi::bitwise_xor_u8(
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
                        BitWiseBinaryOpEnum::And => ffi::bitwise_and_u32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Or => ffi::bitwise_or_u32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Xor => ffi::bitwise_xor_u32(
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
                        BitWiseBinaryOpEnum::And => ffi::bitwise_and_i64(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Or => ffi::bitwise_or_i64(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Xor => ffi::bitwise_xor_i64(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                    }
                };
                CudaStorage::wrap_cuda_slice(d_out, dev)
            }
            DType::I32 => {
                let d_out = unsafe { dev.alloc::<i32>(elem_count) }.w()?;
                let d_out_ptr = *d_out.device_ptr() as *mut c_void;
                unsafe {
                    match self.op {
                        BitWiseBinaryOpEnum::And => ffi::bitwise_and_i32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Or => ffi::bitwise_or_i32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Xor => ffi::bitwise_xor_i32(
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

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        if l1.shape() != l2.shape() || l1.stride() != l2.stride() {
            return Err(Error::ShapeMismatchBinaryOp {
                lhs: l1.shape().clone(),
                rhs: l2.shape().clone(),
                op: "bitwise-op",
            });
        }
        if s1.dtype() != s2.dtype() {
            return Err(Error::DTypeMismatchBinaryOp {
                lhs: s1.dtype(),
                rhs: s2.dtype(),
                op: "bitwise-op",
            });
        }
        if !l1.is_contiguous() {
            candle_core::bail!("Input tensor s1 must be contiguous");
        }
        if !l2.is_contiguous() {
            candle_core::bail!("Input tensor s2 must be contiguous");
        }

        let command_buffer = s1.device().command_buffer()?;
        command_buffer.set_label("bitwise-op");

        let device = s1.device();

        let out_shape = l1.shape().clone();

        let output = device.new_buffer(out_shape.elem_count(), s1.dtype(), "bitwise-op")?;

        match self.op {
            BitWiseBinaryOpEnum::Or => crate::metal_kernels::call_bitwise_or(
                device.device(),
                &command_buffer,
                &crate::metal_kernels::Kernels::new(),
                s1.dtype(),
                s1.buffer(),
                s2.buffer(),
                l1.start_offset() * s1.dtype().size_in_bytes(),
                l2.start_offset() * s2.dtype().size_in_bytes(),
                out_shape.elem_count(),
                &output,
            )
            .map_err(candle_core::Error::wrap)?,
            BitWiseBinaryOpEnum::And => crate::metal_kernels::call_bitwise_and(
                device.device(),
                &command_buffer,
                &crate::metal_kernels::Kernels::new(),
                s1.dtype(),
                s1.buffer(),
                s2.buffer(),
                l1.start_offset() * s1.dtype().size_in_bytes(),
                l2.start_offset() * s2.dtype().size_in_bytes(),
                out_shape.elem_count(),
                &output,
            )
            .map_err(candle_core::Error::wrap)?,
            BitWiseBinaryOpEnum::Xor => crate::metal_kernels::call_bitwise_xor(
                device.device(),
                &command_buffer,
                &crate::metal_kernels::Kernels::new(),
                s1.dtype(),
                s1.buffer(),
                s2.buffer(),
                l1.start_offset() * s1.dtype().size_in_bytes(),
                l2.start_offset() * s2.dtype().size_in_bytes(),
                out_shape.elem_count(),
                &output,
            )
            .map_err(candle_core::Error::wrap)?,
        }

        let newstorage = candle_core::MetalStorage::new(
            output,
            device.clone(),
            out_shape.elem_count(),
            s1.dtype(),
        );
        Ok((newstorage, out_shape))
    }
}

struct BitWiseUnary {
    pub op: BitWiseUnaryOpEnum,
}

impl BitWiseUnary {
    pub fn new(op: BitWiseUnaryOpEnum) -> Self {
        Self { op }
    }

    fn bitwise<T: WithDType + Not<Output = T>>(&self, vs1: &[T]) -> Vec<T> {
        vs1.into_par_iter()
            .map(|v1| match self.op {
                BitWiseUnaryOpEnum::Not => !*v1,
            })
            .collect()
    }
}

impl CustomOp1 for BitWiseUnary {
    fn name(&self) -> &'static str {
        "bitwise-unary"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout) -> Result<(CpuStorage, Shape)> {
        if !l1.is_contiguous() {
            candle_core::bail!("Input tensor s1 must be contiguous");
        }

        match s1 {
            CpuStorage::U8(vs1) => {
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.bitwise(vs1);
                let result = CpuStorage::U8(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::U32(vs1) => {
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.bitwise(vs1);
                let result = CpuStorage::U32(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I64(vs1) => {
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.bitwise(vs1);
                let result = CpuStorage::I64(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I16(vs1) => {
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.bitwise(vs1);
                let result = CpuStorage::I16(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::I32(vs1) => {
                let vs1 = match l1.contiguous_offsets() {
                    Some((a, b)) => &vs1[a..b],
                    None => Err(Error::RequiresContiguous { op: "index-add" }.bt())?,
                };
                let result = self.bitwise(vs1);
                let result = CpuStorage::I32(result);
                Ok((result, l1.shape().clone()))
            }
            CpuStorage::BF16(_) => Err(Error::UnsupportedDTypeForOp(DType::BF16, "bitwise")),
            CpuStorage::F16(_) => Err(Error::UnsupportedDTypeForOp(DType::F16, "bitwise")),
            CpuStorage::F32(_) => Err(Error::UnsupportedDTypeForOp(DType::F32, "bitwise")),
            CpuStorage::F64(_) => Err(Error::UnsupportedDTypeForOp(DType::F64, "bitwise")),
            CpuStorage::F8E4M3(_) => Err(Error::UnsupportedDTypeForOp(DType::F8E4M3, "bitwise")),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, _s1: &CudaStorage, _l1: &Layout) -> Result<(CudaStorage, Shape)> {
        todo!()
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        if !l1.is_contiguous() {
            candle_core::bail!("Input tensor s1 must be contiguous");
        }

        let command_buffer = s1.device().command_buffer()?;
        command_buffer.set_label("bitwise-unary-op");

        let device = s1.device();

        let out_shape = l1.shape().clone();

        let output = device.new_buffer(out_shape.elem_count(), s1.dtype(), "bitwise-op")?;

        match self.op {
            BitWiseUnaryOpEnum::Not => crate::metal_kernels::call_bitwise_not(
                device.device(),
                &command_buffer,
                &crate::metal_kernels::Kernels::new(),
                s1.dtype(),
                s1.buffer(),
                l1.start_offset() * s1.dtype().size_in_bytes(),
                out_shape.elem_count(),
                &output,
            )
            .map_err(candle_core::Error::wrap)?,
        }

        let newstorage = candle_core::MetalStorage::new(
            output,
            device.clone(),
            out_shape.elem_count(),
            s1.dtype(),
        );
        Ok((newstorage, out_shape))
    }
}

#[allow(dead_code)]
pub trait BitWiseOp {
    fn bitwise_and(&self, rhs: &Tensor) -> Result<Tensor>;
    fn bitwise_or(&self, rhs: &Tensor) -> Result<Tensor>;
    fn bitwise_xor(&self, rhs: &Tensor) -> Result<Tensor>;
    fn bitwise_not(&self) -> Result<Tensor>;
}

impl BitWiseOp for Tensor {
    fn bitwise_and(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op2_no_bwd(rhs, &BitWise::new(BitWiseBinaryOpEnum::And))
    }

    fn bitwise_or(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op2_no_bwd(rhs, &BitWise::new(BitWiseBinaryOpEnum::Or))
    }

    fn bitwise_xor(&self, rhs: &Tensor) -> Result<Tensor> {
        self.apply_op2_no_bwd(rhs, &BitWise::new(BitWiseBinaryOpEnum::Xor))
    }

    fn bitwise_not(&self) -> Result<Tensor> {
        self.apply_op1_no_bwd(&BitWiseUnary::new(BitWiseUnaryOpEnum::Not))
    }
}

// ────────────────────────────── ArgSort / Sort ────────────────────────────────

#[allow(unused)]
/// Configuration for an **argsort** (returns indices) operation.
struct ArgSort {
    axis: usize,
}

#[allow(unused)]
/// Configuration for a **sort** (returns re‑ordered values) operation.
struct Sort {
    axis: usize,
}

impl CustomOp1 for ArgSort {
    fn name(&self) -> &'static str {
        "argsort"
    }

    // -------- CPU ------------------------------------------------------------
    fn cpu_fwd(&self, _s1: &CpuStorage, _l1: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("ArgSort is not implemented for the CPU backend");
    }

    // -------- CUDA -----------------------------------------------------------
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, _s1: &CudaStorage, _l1: &Layout) -> Result<(CudaStorage, Shape)> {
        candle_core::bail!("ArgSort is not implemented for the CUDA backend");
    }

    // -------- Metal ----------------------------------------------------------
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        // Require contiguous input (same as other metal ops in this file)
        if !l1.is_contiguous() {
            candle_core::bail!("Input tensor s1 must be contiguous");
        }

        // Create a command‑buffer and label it for easy debugging in Xcode’s GPU frame‑capture
        let command_buffer = s1.device().command_buffer()?;
        command_buffer.set_label("argsort");

        let device = s1.device();
        let out_shape = l1.shape().clone();
        let elem_count = out_shape.elem_count();

        // Output buffer holds the sorted indices → always `U32`
        let output = device.new_buffer(elem_count, candle_core::DType::U32, "argsort")?;

        // ------------------------------------------------------------------
        // Obtain a scratch‑buffer set from the global LRU cache (cap=4)
        // ------------------------------------------------------------------
        let cache = SORT_SCRATCH_CACHE.get_or_init(|| SortScratchCache::new(4));

        let dims = l1.dims();
        let size_sorted_axis = dims[self.axis];
        let n_rows = l1.shape().elem_count() / size_sorted_axis;

        // Replicate the kernel’s internal block sizing to derive `n_blocks`
        let tn = 4usize;
        let mut bn = match size_sorted_axis.div_ceil(tn) {
            v if v > 256 => 512,
            v if v > 128 => 256,
            v if v > 64 => 128,
            v if v > 32 => 64,
            _ => 32,
        };
        if bn == 512 && s1.dtype().size_in_bytes() > 4 {
            bn = 256;
        }
        let n_per_block = bn * tn;
        let n_blocks = size_sorted_axis.div_ceil(n_per_block);

        // Borrow the buffers for this launch
        let scratch = cache.checkout(device, n_rows, size_sorted_axis, s1.dtype(), n_blocks);

        // ------------------------------------------------------------------
        // Build the unified SortArgs payload
        // ------------------------------------------------------------------
        let sort_args = crate::metal_kernels::SortArgs {
            axis: self.axis,
            shape: l1.dims(),
            strides: l1.stride(),
            out_shape: l1.dims(), // same as input for argsort
            out_strides: l1.stride(),
            in_contiguous: l1.is_contiguous(),
            in_ty: s1.dtype(),
            out_ty: candle_core::DType::U32,
            src: s1.buffer(),
            src_offset: l1.start_offset(), // element offset
            dst: &output,
            bn,
            tn,
            n_blocks,
        };

        // Launch the Metal kernel via the new API
        crate::metal_kernels::call_argsort(
            device.device(), // &metal::Device
            &command_buffer, // impl EncoderProvider
            &crate::metal_kernels::Kernels::new(),
            &sort_args,
            &scratch,
        )
        .map_err(candle_core::Error::wrap)?;

        // Wrap and return as a new MetalStorage
        let newstorage = candle_core::MetalStorage::new(
            output,
            device.clone(),
            elem_count,
            candle_core::DType::U32,
        );
        Ok((newstorage, out_shape))
    }
}

impl CustomOp1 for Sort {
    fn name(&self) -> &'static str {
        "sort"
    }

    // -------- CPU ------------------------------------------------------------
    fn cpu_fwd(&self, _s1: &CpuStorage, _l1: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("Sort is not implemented for the CPU backend");
    }

    // -------- CUDA -----------------------------------------------------------
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, _s1: &CudaStorage, _l1: &Layout) -> Result<(CudaStorage, Shape)> {
        candle_core::bail!("Sort is not implemented for the CUDA backend");
    }

    // -------- Metal ----------------------------------------------------------
    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        // Require contiguous input (same as other metal ops in this file)
        if !l1.is_contiguous() {
            candle_core::bail!("Input tensor s1 must be contiguous");
        }

        // Create a command‑buffer and label it for easy debugging in Xcode’s GPU frame‑capture
        let command_buffer = s1.device().command_buffer()?;
        command_buffer.set_label("sort");

        let device = s1.device();
        let out_shape = l1.shape().clone();
        let elem_count = out_shape.elem_count();

        // Output buffer keeps the same dtype as the input (these are the reordered values)
        let output = device.new_buffer(elem_count, s1.dtype(), "sort")?;

        // ------------------------------------------------------------------
        // Obtain a scratch‑buffer set from the global LRU cache (cap=4)
        // ------------------------------------------------------------------
        let cache = SORT_SCRATCH_CACHE.get_or_init(|| SortScratchCache::new(4));

        let dims = l1.dims();
        let size_sorted_axis = dims[self.axis];
        let n_rows = l1.shape().elem_count() / size_sorted_axis;

        // Replicate the kernel’s internal block sizing to derive `n_blocks`
        let tn = 4usize;
        let mut bn = match size_sorted_axis.div_ceil(tn) {
            v if v > 256 => 512,
            v if v > 128 => 256,
            v if v > 64 => 128,
            v if v > 32 => 64,
            _ => 32,
        };
        if bn == 512 && s1.dtype().size_in_bytes() > 4 {
            bn = 256;
        }
        let n_per_block = bn * tn;
        let n_blocks = size_sorted_axis.div_ceil(n_per_block);

        // Borrow the buffers for this launch
        let scratch = cache.checkout(device, n_rows, size_sorted_axis, s1.dtype(), n_blocks);

        // ------------------------------------------------------------------
        // Build the unified SortArgs payload
        // ------------------------------------------------------------------
        let sort_args = crate::metal_kernels::SortArgs {
            axis: self.axis,
            shape: l1.dims(),
            strides: l1.stride(),
            out_shape: l1.dims(), // same shape for value sort
            out_strides: l1.stride(),
            in_contiguous: l1.is_contiguous(),
            in_ty: s1.dtype(),
            out_ty: s1.dtype(),
            src: s1.buffer(),
            src_offset: l1.start_offset(), // element offset
            dst: &output,
            bn,
            tn,
            n_blocks,
        };

        // Launch the Metal kernel via the new API
        crate::metal_kernels::call_sort(
            device.device(), // &metal::Device
            &command_buffer, // impl EncoderProvider
            &crate::metal_kernels::Kernels::new(),
            &sort_args,
            &scratch,
        )
        .map_err(candle_core::Error::wrap)?;

        // Wrap and return as a new MetalStorage
        let newstorage =
            candle_core::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, out_shape))
    }
}

/// Extension trait adding `argsort` / `sort` convenience calls on `Tensor`.
pub trait SortOp {
    /// Returns the indices that would (ascending) sort the tensor along `axis`.
    fn fast_argsort_asc<D: Dim>(&self, axis: D) -> Result<Tensor>;
    /// Returns the tensor's values (ascending) sorted along `axis`.
    fn fast_sort_asc<D: Dim>(&self, axis: D) -> Result<Tensor>;
}

impl SortOp for Tensor {
    fn fast_argsort_asc<D: Dim>(&self, axis: D) -> Result<Tensor> {
        if self.device().is_cpu() || self.device().is_cuda() {
            return self.arg_sort_last_dim(true);
        }
        self.apply_op1_no_bwd(&ArgSort {
            axis: axis.to_index(self.shape(), "argsort")?,
        })
    }

    fn fast_sort_asc<D: Dim>(&self, axis: D) -> Result<Tensor> {
        if self.device().is_cpu() || self.device().is_cuda() {
            return Ok(self.sort_last_dim(true)?.0);
        }
        self.apply_op1_no_bwd(&Sort {
            axis: axis.to_index(self.shape(), "sort")?,
        })
    }
}

struct NonZero;

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
fn count_nonzero_cuda(
    dtype: candle_core::DType,
    d_in: *const c_void,
    n: u32,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
) -> u32 {
    unsafe {
        match dtype {
            candle_core::DType::U8 => ffi::count_nonzero_u8(d_in, n, stream),
            candle_core::DType::U32 => ffi::count_nonzero_u32(d_in, n, stream),
            candle_core::DType::I64 => ffi::count_nonzero_i64(d_in, n, stream),
            candle_core::DType::I16 => ffi::count_nonzero_i16(d_in, n, stream),
            candle_core::DType::I32 => ffi::count_nonzero_i32(d_in, n, stream),
            candle_core::DType::BF16 => ffi::count_nonzero_bf16(d_in, n, stream),
            candle_core::DType::F16 => ffi::count_nonzero_f16(d_in, n, stream),
            candle_core::DType::F32 => ffi::count_nonzero_f32(d_in, n, stream),
            candle_core::DType::F64 => ffi::count_nonzero_f64(d_in, n, stream),
            candle_core::DType::F8E4M3 => todo!(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
fn nonzero_cuda(
    dtype: candle_core::DType,
    d_in: *const c_void,
    n: u32,
    num_nonzero: u32,
    dims: *const c_void,
    num_dims: u32,
    d_out: *mut c_void,
    stream: candle_core::cuda::cudarc::driver::sys::CUstream,
) {
    unsafe {
        match dtype {
            candle_core::DType::U8 => {
                ffi::nonzero_u8(d_in, n, num_nonzero, dims, num_dims, d_out, stream)
            }
            candle_core::DType::U32 => {
                ffi::nonzero_u32(d_in, n, num_nonzero, dims, num_dims, d_out, stream)
            }
            candle_core::DType::I64 => {
                ffi::nonzero_i64(d_in, n, num_nonzero, dims, num_dims, d_out, stream)
            }
            candle_core::DType::I32 => {
                ffi::nonzero_i64(d_in, n, num_nonzero, dims, num_dims, d_out, stream)
            }
            candle_core::DType::I16 => {
                ffi::nonzero_i16(d_in, n, num_nonzero, dims, num_dims, d_out, stream)
            }
            candle_core::DType::BF16 => {
                ffi::nonzero_bf16(d_in, n, num_nonzero, dims, num_dims, d_out, stream)
            }
            candle_core::DType::F16 => {
                ffi::nonzero_f16(d_in, n, num_nonzero, dims, num_dims, d_out, stream)
            }
            candle_core::DType::F32 => {
                ffi::nonzero_f32(d_in, n, num_nonzero, dims, num_dims, d_out, stream)
            }
            candle_core::DType::F64 => {
                ffi::nonzero_f64(d_in, n, num_nonzero, dims, num_dims, d_out, stream)
            }
            candle_core::DType::F8E4M3 => todo!(),
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
            candle_core::CpuStorage::I16(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::I32(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::I64(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::BF16(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::F16(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::F32(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::F64(vs) => self.nonzero(vs, layout),
            candle_core::CpuStorage::F8E4M3(_vs) => todo!(),
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
            candle_core::DType::I32 => *storage.as_cuda_slice::<i32>()?.device_ptr(),
            candle_core::DType::I16 => *storage.as_cuda_slice::<i16>()?.device_ptr(),
            candle_core::DType::I64 => *storage.as_cuda_slice::<i64>()?.device_ptr(),
            candle_core::DType::BF16 => *storage.as_cuda_slice::<half::bf16>()?.device_ptr(),
            candle_core::DType::F16 => *storage.as_cuda_slice::<half::f16>()?.device_ptr(),
            candle_core::DType::F32 => *storage.as_cuda_slice::<f32>()?.device_ptr(),
            candle_core::DType::F64 => *storage.as_cuda_slice::<f64>()?.device_ptr(),
            candle_core::DType::F8E4M3 => todo!(),
        } as *const c_void;
        let n = layout.shape().elem_count();

        let num_nonzero =
            count_nonzero_cuda(storage.dtype(), d_in, u32::try_from(n)?, *dev.cu_stream());
        let d_out = unsafe { dev.alloc::<u32>(num_nonzero as usize * layout.dims().len()) }
            .map_err(|_| Error::Msg("Failed to allocate memory for nonzero result".to_string()))?;
        if num_nonzero != 0 {
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
                *dev.cu_stream(),
            );
        }
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
            .apply_op1_no_bwd(&NonZero)?
            .to_device(original_device)
    }

    #[cfg(not(feature = "metal"))]
    fn nonzero(&self) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle_core::Error::RequiresContiguous { op: "nonzero" });
        }
        self.apply_op1_no_bwd(&NonZero)
    }
}

struct CumSum {
    inclusive: bool,
    reverse: bool,
    axis: usize,
}

impl CustomOp1 for CumSum {
    fn name(&self) -> &'static str {
        "cumsum"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout) -> Result<(CpuStorage, Shape)> {
        use std::ops::Add;
        if !l1.is_contiguous() {
            candle_core::bail!("Input tensor s1 must be contiguous");
        }
        let dims = l1.dims();
        let axis = self.axis;
        let axis_len = dims[axis];
        let (start, end) = l1
            .contiguous_offsets()
            .ok_or(Error::RequiresContiguous { op: "cumsum" })?;

        // helper to execute scan for a slice of T
        macro_rules! scan_block {
            ($vt:ident, $ty:ty, $add:ident, $init:expr) => {{
                let vs: &[$ty] = $vt;
                let input = &vs[start..end];
                let count = input.len() / axis_len;
                let mut result = Vec::<$ty>::with_capacity(input.len());
                if !self.reverse {
                    if self.inclusive {
                        for block in 0..count {
                            let base = block * axis_len;
                            let mut sum = input[base];
                            result.push(sum);
                            for j in 1..axis_len {
                                sum = sum.$add(input[base + j]);
                                result.push(sum);
                            }
                        }
                    } else {
                        let init: $ty = $init;
                        for block in 0..count {
                            let base = block * axis_len;
                            let mut sum = init;
                            for j in 0..axis_len {
                                result.push(sum);
                                sum = sum.$add(input[base + j]);
                            }
                        }
                    }
                } else {
                    if self.inclusive {
                        for block in 0..count {
                            let base = block * axis_len;
                            let mut temp = Vec::<$ty>::with_capacity(axis_len);
                            let mut sum = input[base + axis_len - 1];
                            temp.push(sum);
                            for k in 1..axis_len {
                                let idx = axis_len - 1 - k;
                                sum = sum.$add(input[base + idx]);
                                temp.push(sum);
                            }
                            temp.reverse();
                            result.extend(temp);
                        }
                    } else {
                        let init: $ty = $init;
                        for block in 0..count {
                            let base = block * axis_len;
                            let mut temp = Vec::<$ty>::with_capacity(axis_len);
                            let mut sum = init;
                            for k in 0..axis_len {
                                let idx = axis_len - 1 - k;
                                temp.push(sum);
                                sum = sum.$add(input[base + idx]);
                            }
                            temp.reverse();
                            result.extend(temp);
                        }
                    }
                }
                result
            }};
        }
        match s1 {
            CpuStorage::U8(vs) => {
                let result = scan_block!(vs, u8, wrapping_add, 0u8);
                Ok((CpuStorage::U8(result), l1.shape().clone()))
            }
            CpuStorage::I16(vs) => {
                let result = scan_block!(vs, i16, add, 0i16);
                Ok((CpuStorage::I16(result), l1.shape().clone()))
            }
            CpuStorage::U32(vs) => {
                let result = scan_block!(vs, u32, wrapping_add, 0u32);
                Ok((CpuStorage::U32(result), l1.shape().clone()))
            }
            CpuStorage::I32(vs) => {
                let result = scan_block!(vs, i32, add, 0i32);
                Ok((CpuStorage::I32(result), l1.shape().clone()))
            }
            CpuStorage::I64(vs) => {
                let result = scan_block!(vs, i64, add, 0i64);
                Ok((CpuStorage::I64(result), l1.shape().clone()))
            }
            CpuStorage::F32(vs) => {
                let result = scan_block!(vs, f32, add, 0.0f32);
                Ok((CpuStorage::F32(result), l1.shape().clone()))
            }
            CpuStorage::F64(vs) => {
                let result = scan_block!(vs, f64, add, 0.0f64);
                Ok((CpuStorage::F64(result), l1.shape().clone()))
            }
            _ => Err(Error::UnsupportedDTypeForOp(DType::F32, "cumsum")),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, _s1: &CudaStorage, _l1: &Layout) -> Result<(CudaStorage, Shape)> {
        todo!()
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use crate::metal_kernels::ScanType;

        let command_buffer = s1.device().command_buffer()?;
        command_buffer.set_label("cumsum");

        let device = s1.device();

        let out_shape = l1.shape().clone();

        let output = device.new_buffer(out_shape.elem_count(), s1.dtype(), "cumsum")?;

        crate::metal_kernels::call_scan(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            s1.dtype(),
            ScanType::Sum,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            self.axis,
            l1.dims(),
            l1.stride(),
            self.reverse,
            self.inclusive,
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        let newstorage = candle_core::MetalStorage::new(
            output,
            device.clone(),
            out_shape.elem_count(),
            s1.dtype(),
        );
        Ok((newstorage, out_shape))
    }
}

#[allow(dead_code)]
pub trait CumSumOp {
    /// inclusive = false, reverse = false
    fn fast_cumsum<D: Dim>(&self, axis: D) -> Result<Tensor>;

    fn fast_cumsum_config<D: Dim>(&self, axis: D, inclusive: bool, reverse: bool)
        -> Result<Tensor>;
}

impl CumSumOp for Tensor {
    fn fast_cumsum<D: Dim>(&self, axis: D) -> Result<Tensor> {
        self.fast_cumsum_config(axis, false, false)
    }

    fn fast_cumsum_config<D: Dim>(
        &self,
        axis: D,
        inclusive: bool,
        reverse: bool,
    ) -> Result<Tensor> {
        self.apply_op1_no_bwd(&CumSum {
            inclusive,
            reverse,
            axis: axis.to_index(self.shape(), "cumsum")?,
        })
    }
}

mod tests {
    #[test]
    fn test_cumsum_exclusive_forward_cpu() {
        use crate::utils::ops::CumSumOp;
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
        let b = a.fast_cumsum(0).unwrap().to_vec1::<i64>().unwrap();
        assert_eq!(b, [0, 1, 3, 6]);
    }

    #[test]
    fn test_cumsum_inclusive_forward_cpu() {
        use crate::utils::ops::CumSumOp;
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
        let b = a
            .fast_cumsum_config(0, true, false)
            .unwrap()
            .to_vec1::<i64>()
            .unwrap();
        assert_eq!(b, [1, 3, 6, 10]);
    }

    #[test]
    fn test_cumsum_exclusive_reverse_cpu() {
        use crate::utils::ops::CumSumOp;
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
        let b = a
            .fast_cumsum_config(0, false, true)
            .unwrap()
            .to_vec1::<i64>()
            .unwrap();
        assert_eq!(b, [9, 7, 4, 0]);
    }

    #[test]
    fn test_cumsum_inclusive_reverse_cpu() {
        use crate::utils::ops::CumSumOp;
        use candle_core::Tensor;
        let device = candle_core::Device::Cpu;
        let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
        let b = a
            .fast_cumsum_config(0, true, true)
            .unwrap()
            .to_vec1::<i64>()
            .unwrap();
        assert_eq!(b, [10, 9, 7, 4]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_cumsum_exclusive_forward_metal() {
        use crate::utils::ops::CumSumOp;
        use candle_core::Tensor;
        let device = candle_core::Device::new_metal(0).unwrap();
        let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
        let b = a.fast_cumsum(0).unwrap().to_vec1::<i64>().unwrap();
        assert_eq!(b, [0, 1, 3, 6]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_cumsum_inclusive_forward_metal() {
        use crate::utils::ops::CumSumOp;
        use candle_core::Tensor;
        let device = candle_core::Device::new_metal(0).unwrap();
        let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
        let b = a
            .fast_cumsum_config(0, true, false)
            .unwrap()
            .to_vec1::<i64>()
            .unwrap();
        assert_eq!(b, [1, 3, 6, 10]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_cumsum_exclusive_reverse_metal() {
        use crate::utils::ops::CumSumOp;
        use candle_core::Tensor;
        let device = candle_core::Device::new_metal(0).unwrap();
        let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
        let b = a
            .fast_cumsum_config(0, false, true)
            .unwrap()
            .to_vec1::<i64>()
            .unwrap();
        assert_eq!(b, [9, 7, 4, 0]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_cumsum_inclusive_reverse_metal() {
        use crate::utils::ops::CumSumOp;
        use candle_core::Tensor;
        let device = candle_core::Device::new_metal(0).unwrap();
        let a = Tensor::from_vec(vec![1i64, 2, 3, 4], &[4], &device).unwrap();
        let b = a
            .fast_cumsum_config(0, true, true)
            .unwrap()
            .to_vec1::<i64>()
            .unwrap();
        assert_eq!(b, [10, 9, 7, 4]);
    }

    #[test]
    fn test_nonzero_cpu() {
        use crate::utils::ops::NonZeroOp;
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
        use crate::utils::ops::NonZeroOp;
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
        use crate::utils::ops::BitWiseOp;
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
        use crate::utils::ops::BitWiseOp;
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
        use crate::utils::ops::BitWiseOp;
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
        use crate::utils::ops::BitWiseOp;
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
        use crate::utils::ops::BitWiseOp;
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
        use crate::utils::ops::BitWiseOp;
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
        use crate::utils::ops::{BitWiseOp, NonZeroOp};
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
        use crate::utils::ops::{BitWiseOp, NonZeroOp};
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

    #[test]
    fn test_bitpack_8bit_cpu() {
        use crate::HqqBits;
        use candle_core::{Device, Tensor};
        let bits = HqqBits::Eight;
        let device = Device::Cpu;
        let wq = Tensor::from_vec(vec![257_i32, 258, 259, 260, 511, 512], (3, 2), &device).unwrap();
        let c = bits.bitpack_type()(wq.clone())
            .unwrap()
            .to_vec2::<u8>()
            .unwrap();
        assert_eq!(c, [[1, 2], [3, 4], [255, 0]]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_bitpack_8bit_cuda() {
        use crate::HqqBits;
        use candle_core::DType;
        use candle_core::{Device, Tensor};
        let bits = HqqBits::Eight;
        let device = Device::new_cuda(0).unwrap();
        let wq = Tensor::from_vec(vec![257_i32, 258, 259, 260, 511, 512], (3, 2), &device).unwrap();
        let c = bits.bitpack_type()(wq.clone())
            .unwrap()
            .to_dtype(DType::U8)
            .unwrap()
            .to_vec2::<u8>()
            .unwrap();
        assert_eq!(c, [[1, 2], [3, 4], [255, 0]]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_bitpack_8bit_metal() {
        use crate::HqqBits;
        use candle_core::{Device, Tensor};
        let bits = HqqBits::Eight;
        let device = Device::new_metal(0).unwrap();
        let wq = Tensor::from_vec(vec![257_i32, 258, 259, 260, 511, 512], (3, 2), &device).unwrap();
        let c = bits.bitpack_type()(wq.clone())
            .unwrap()
            .to_vec2::<u8>()
            .unwrap();
        assert_eq!(c, [[1, 2], [3, 4], [255, 0]]);
    }

    #[test]
    fn test_bitpack_4bit() {
        use crate::HqqBits;
        use candle_core::{Device, Tensor};
        let bits = HqqBits::Four;
        let device = Device::Cpu;
        let wq = Tensor::from_vec(vec![1_u8, 2, 3, 4, 5, 6], (3, 2), &device).unwrap();
        let c = bits.bitpack_type()(wq.clone())
            .unwrap()
            .to_vec2::<u8>()
            .unwrap();
        assert_eq!(c, [[19, 36]]);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_bitpack_4bit_cuda() {
        use crate::HqqBits;
        use candle_core::{Device, Tensor};
        let bits = HqqBits::Four;
        let device = Device::new_cuda(0).unwrap();
        let wq = Tensor::from_vec(vec![1_u8, 2, 3, 4, 5, 6], (3, 2), &device).unwrap();
        let c = bits.bitpack_type()(wq.clone())
            .unwrap()
            .to_vec2::<u8>()
            .unwrap();
        assert_eq!(c, [[19, 36]]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_bitpack_4bit_metal() {
        use crate::HqqBits;
        use candle_core::{Device, Tensor};
        let bits = HqqBits::Four;
        let device = Device::new_metal(0).unwrap();
        let wq = Tensor::from_vec(vec![1_u8, 2, 3, 4, 5, 6], (3, 2), &device).unwrap();
        let c = bits.bitpack_type()(wq.clone())
            .unwrap()
            .to_vec2::<u8>()
            .unwrap();
        assert_eq!(c, [[19, 36]]);
    }
    // ─────────────────────────────── Sort / ArgSort ────────────────────────────────
    #[cfg(feature = "metal")]
    #[test]
    fn test_sort_and_argsort_vector_metal() {
        use crate::utils::ops::SortOp;
        use candle_core::Tensor;

        let device = candle_core::Device::new_metal(0).unwrap();
        let a = Tensor::from_vec(vec![3i32, 1, 4, 2], &[4], &device).unwrap();

        // sort (ascending)
        let sorted = a.fast_sort_asc(0).unwrap().to_vec1::<i32>().unwrap();
        assert_eq!(sorted, [1, 2, 3, 4]);

        // argsort (ascending indices)
        let idx = a.fast_argsort_asc(0).unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(idx, [1, 3, 0, 2]);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_sort_and_argsort_matrix_axis1_metal() {
        use crate::utils::ops::SortOp;
        use candle_core::Tensor;

        let device = candle_core::Device::new_metal(0).unwrap();
        // 2 × 3 matrix:
        // [[3, 1, 2],
        //  [0, 4, 5]]
        let a = Tensor::from_vec(vec![3i32, 1, 2, 0, 4, 5], &[2, 3], &device).unwrap();

        // Sort along axis=1 (second dimension)
        let sorted = a.fast_sort_asc(1).unwrap().to_vec2::<i32>().unwrap();
        assert_eq!(sorted, [[1, 2, 3], [0, 4, 5]]);

        // ArgSort indices along axis=1
        let idx = a.fast_argsort_asc(1).unwrap().to_vec2::<u32>().unwrap();
        assert_eq!(idx, [[1, 2, 0], [0, 1, 2]]);
    }

    // ─────────────────────────────── 2 048-element vector ────────────────────────────────
    #[cfg(feature = "metal")]
    #[test]
    fn test_sort_and_argsort_vector_2048_metal() {
        use crate::utils::ops::SortOp;
        use candle_core::Tensor;

        const N: usize = 4096;

        let device = candle_core::Device::new_metal(0).expect("Metal device");

        // Create a descending vector [4095, 4094, …, 0]
        let vals: Vec<i32> = (0..N as i32).rev().collect();
        let a = Tensor::from_vec(vals.clone(), &[N], &device).unwrap();

        // ---- sort (ascending) ---------------------------------------------------------
        let sorted = a.fast_sort_asc(0).unwrap().to_vec1::<i32>().unwrap();
        let expected: Vec<i32> = (0..N as i32).collect();
        assert_eq!(sorted, expected);

        // ---- argsort (indices that would sort) ---------------------------------------
        let idx = a.fast_argsort_asc(0).unwrap().to_vec1::<u32>().unwrap();
        // Because the input is reversed, the correct indices are likewise reversed
        for (i, &v) in idx.iter().enumerate() {
            assert_eq!(v as usize, N - 1 - i);
        }
    }
}
