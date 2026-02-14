use candle_core::{
    backend::BackendStorage, shape::Dim, CpuStorage, CustomOp1, CustomOp2, DType, Error, Layout,
    Result, Shape, Tensor, WithDType,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

use std::{
    fmt::Display,
    ops::{BitAnd, BitOr, BitXor, Not, Shl},
};

#[cfg(feature = "cuda")]
use crate::utils::{ffi, slice_ptr};
#[cfg(feature = "cuda")]
use candle_core::cuda::{cudarc::driver::DevicePtr, CudaStorage};
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
            _ => Err(Error::UnsupportedDTypeForOp(s1.dtype(), "leftshift")),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, s1: &CudaStorage, l1: &Layout) -> Result<(CudaStorage, Shape)> {
        if !l1.is_contiguous() {
            candle_core::bail!("Input tensor s1 must be contiguous");
        }
        let dev = s1.device().clone();
        let (d_in1_ptr, _d_guard, elem_count) = match s1.dtype() {
            DType::U8 => {
                let (d_in1, d_in1_guard) = slice_ptr(s1.as_cuda_slice::<u8>()?, l1.start_offset());
                let elem_count = l1.shape().elem_count();
                (d_in1 as *const c_void, d_in1_guard, elem_count)
            }
            DType::I32 => {
                let (d_in1, d_in1_guard) = slice_ptr(s1.as_cuda_slice::<i32>()?, l1.start_offset());
                let elem_count = l1.shape().elem_count();
                (d_in1 as *const c_void, d_in1_guard, elem_count)
            }
            other => {
                return Err(Error::UnsupportedDTypeForOp(other, "leftshift"));
            }
        };
        let dst = match s1.dtype() {
            DType::U8 => {
                let d_out = unsafe { dev.alloc::<u8>(elem_count) }?;
                let (d_out_ptr, d_out_guard) = d_out.device_ptr(d_out.stream());
                unsafe {
                    ffi::leftshift_u8(
                        d_in1_ptr,
                        d_out_ptr as *mut std::ffi::c_void,
                        u32::try_from(elem_count)?,
                        self.0 as i32,
                    )
                };
                drop(d_out_guard);
                CudaStorage::wrap_cuda_slice(d_out, dev)
            }
            DType::I32 => {
                let d_out = unsafe { dev.alloc::<i32>(elem_count) }?;
                let (d_out_ptr, d_out_guard) = d_out.device_ptr(d_out.stream());
                unsafe {
                    ffi::leftshift_i32(
                        d_in1_ptr,
                        d_out_ptr as *mut std::ffi::c_void,
                        u32::try_from(elem_count)?,
                        self.0 as i32,
                    )
                };
                drop(d_out_guard);
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

        let encoder = s1.device().command_encoder()?;
        encoder.set_label("bitwise-leftshift");

        let device = s1.device();

        let out_shape = l1.shape().clone();

        let output = device.new_buffer(out_shape.elem_count(), s1.dtype(), "bitwise-leftshift")?;

        crate::metal_kernels::call_bitwise_leftshift(
            device.device(),
            &encoder,
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
            _ => Err(Error::UnsupportedDTypeForOp(s1.dtype(), "bitwise")),
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
        let (d_in1_ptr, d_in2_ptr, _d_in1_guard, _d_in2_guard, elem_count) = match s1.dtype() {
            DType::U8 => {
                let (d_in1, d_in1_guard) = slice_ptr(s1.as_cuda_slice::<u8>()?, l1.start_offset());
                let (d_in2, d_in2_guard) = slice_ptr(s2.as_cuda_slice::<u8>()?, l2.start_offset());
                let elem_count = l1.shape().elem_count();
                (
                    d_in1 as *const std::ffi::c_void,
                    d_in2 as *const std::ffi::c_void,
                    d_in1_guard,
                    d_in2_guard,
                    elem_count,
                )
            }
            DType::U32 => {
                let (d_in1, d_in1_guard) = slice_ptr(s1.as_cuda_slice::<u32>()?, l1.start_offset());
                let (d_in2, d_in2_guard) = slice_ptr(s2.as_cuda_slice::<u32>()?, l2.start_offset());
                let elem_count = l1.shape().elem_count();
                (
                    d_in1 as *const std::ffi::c_void,
                    d_in2 as *const std::ffi::c_void,
                    d_in1_guard,
                    d_in2_guard,
                    elem_count,
                )
            }
            DType::I64 => {
                let (d_in1, d_in1_guard) = slice_ptr(s1.as_cuda_slice::<i64>()?, l1.start_offset());
                let (d_in2, d_in2_guard) = slice_ptr(s2.as_cuda_slice::<i64>()?, l2.start_offset());
                let elem_count = l1.shape().elem_count();
                (
                    d_in1 as *const std::ffi::c_void,
                    d_in2 as *const std::ffi::c_void,
                    d_in1_guard,
                    d_in2_guard,
                    elem_count,
                )
            }
            DType::I32 => {
                let (d_in1, d_in1_guard) = slice_ptr(s1.as_cuda_slice::<i32>()?, l1.start_offset());
                let (d_in2, d_in2_guard) = slice_ptr(s2.as_cuda_slice::<i32>()?, l2.start_offset());
                let elem_count = l1.shape().elem_count();
                (
                    d_in1 as *const std::ffi::c_void,
                    d_in2 as *const std::ffi::c_void,
                    d_in1_guard,
                    d_in2_guard,
                    elem_count,
                )
            }
            DType::I16 => {
                let (d_in1, d_in1_guard) = slice_ptr(s1.as_cuda_slice::<i16>()?, l1.start_offset());
                let (d_in2, d_in2_guard) = slice_ptr(s2.as_cuda_slice::<i16>()?, l2.start_offset());
                let elem_count = l1.shape().elem_count();
                (
                    d_in1 as *const std::ffi::c_void,
                    d_in2 as *const std::ffi::c_void,
                    d_in1_guard,
                    d_in2_guard,
                    elem_count,
                )
            }
            other => {
                return Err(Error::UnsupportedDTypeForOp(other, "bitwise"));
            }
        };
        let dst = match s1.dtype() {
            DType::U8 => {
                let d_out = unsafe { dev.alloc::<u8>(elem_count) }?;
                let (d_out_ptr, d_out_guard) = d_out.device_ptr(d_out.stream());
                unsafe {
                    match self.op {
                        BitWiseBinaryOpEnum::And => ffi::bitwise_and_u8(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Or => ffi::bitwise_or_u8(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Xor => ffi::bitwise_xor_u8(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                    }
                };
                drop(d_out_guard);
                CudaStorage::wrap_cuda_slice(d_out, dev)
            }
            DType::U32 => {
                let d_out = unsafe { dev.alloc::<u32>(elem_count) }?;
                let (d_out_ptr, d_out_guard) = d_out.device_ptr(d_out.stream());
                unsafe {
                    match self.op {
                        BitWiseBinaryOpEnum::And => ffi::bitwise_and_u32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Or => ffi::bitwise_or_u32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Xor => ffi::bitwise_xor_u32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                    }
                };
                drop(d_out_guard);
                CudaStorage::wrap_cuda_slice(d_out, dev)
            }
            DType::I64 => {
                let d_out = unsafe { dev.alloc::<i64>(elem_count) }?;
                let (d_out_ptr, d_out_guard) = d_out.device_ptr(d_out.stream());
                unsafe {
                    match self.op {
                        BitWiseBinaryOpEnum::And => ffi::bitwise_and_i64(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Or => ffi::bitwise_or_i64(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Xor => ffi::bitwise_xor_i64(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                    }
                };
                drop(d_out_guard);
                CudaStorage::wrap_cuda_slice(d_out, dev)
            }
            DType::I32 => {
                let d_out = unsafe { dev.alloc::<i64>(elem_count) }?;
                let (d_out_ptr, d_out_guard) = d_out.device_ptr(d_out.stream());
                unsafe {
                    match self.op {
                        BitWiseBinaryOpEnum::And => ffi::bitwise_and_i32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Or => ffi::bitwise_or_i32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                        BitWiseBinaryOpEnum::Xor => ffi::bitwise_xor_i32(
                            d_in1_ptr,
                            d_in2_ptr,
                            d_out_ptr as *mut c_void,
                            u32::try_from(elem_count)?,
                        ),
                    }
                };
                drop(d_out_guard);
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

        let encoder = s1.device().command_encoder()?;
        encoder.set_label("bitwise-op");

        let device = s1.device();

        let out_shape = l1.shape().clone();

        let output = device.new_buffer(out_shape.elem_count(), s1.dtype(), "bitwise-op")?;

        match self.op {
            BitWiseBinaryOpEnum::Or => crate::metal_kernels::call_bitwise_or(
                device.device(),
                &encoder,
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
                &encoder,
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
                &encoder,
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
            _ => Err(Error::UnsupportedDTypeForOp(s1.dtype(), "bitwise")),
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

        let encoder = s1.device().command_encoder()?;
        encoder.set_label("bitwise-unary-op");

        let device = s1.device();

        let out_shape = l1.shape().clone();

        let output = device.new_buffer(out_shape.elem_count(), s1.dtype(), "bitwise-op")?;

        match self.op {
            BitWiseUnaryOpEnum::Not => crate::metal_kernels::call_bitwise_not(
                device.device(),
                &encoder,
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

        // Create a command encoder and label it for easy debugging in Xcode’s GPU frame‑capture
        let encoder = s1.device().command_encoder()?;
        encoder.set_label("argsort");

        let device = s1.device();
        let out_shape = l1.shape().clone();
        let elem_count = out_shape.elem_count();

        // Output buffer holds the sorted indices -> always `U32`
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
            device.device(),
            &encoder, // impl EncoderProvider
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

        // Create a command encoder and label it for easy debugging in Xcode’s GPU frame‑capture
        let encoder = s1.device().command_encoder()?;
        encoder.set_label("sort");

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
            device.device(),
            &encoder, // impl EncoderProvider
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

#[cfg(all(feature = "cuda", not(feature = "cuda-13000")))]
mod cuda_ops_cccl2 {
    use super::*;

    pub(super) fn count_nonzero_cuda(
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
                _ => unreachable!(),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn nonzero_cuda(
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
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(all(feature = "cuda", feature = "cuda-13000"))]
mod cuda_ops_cccl3 {
    use super::*;

    pub(super) fn count_nonzero_cuda(
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
                _ => unreachable!(),
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn nonzero_cuda(
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
                _ => unreachable!(),
            }
        }
    }
}

#[cfg(all(feature = "cuda", not(feature = "cuda-13000")))]
use cuda_ops_cccl2::{count_nonzero_cuda, nonzero_cuda};
#[cfg(all(feature = "cuda", feature = "cuda-13000"))]
use cuda_ops_cccl3::{count_nonzero_cuda, nonzero_cuda};

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
            _ => unreachable!(),
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
        let (d_in, _d_in_guard) = match storage.dtype() {
            candle_core::DType::U8 => {
                let slice = storage.as_cuda_slice::<u8>()?;
                let (d_in, d_in_guard) = slice_ptr(slice, 0);
                (d_in as *const std::ffi::c_void, d_in_guard)
            }
            candle_core::DType::U32 => {
                let slice = storage.as_cuda_slice::<u32>()?;
                let (d_in, d_in_guard) = slice_ptr(slice, 0);
                (d_in as *const std::ffi::c_void, d_in_guard)
            }
            candle_core::DType::I32 => {
                let slice = storage.as_cuda_slice::<i32>()?;
                let (d_in, d_in_guard) = slice_ptr(slice, 0);
                (d_in as *const std::ffi::c_void, d_in_guard)
            }
            candle_core::DType::I16 => {
                let slice = storage.as_cuda_slice::<i16>()?;
                let (d_in, d_in_guard) = slice_ptr(slice, 0);
                (d_in as *const std::ffi::c_void, d_in_guard)
            }
            candle_core::DType::I64 => {
                let slice = storage.as_cuda_slice::<i64>()?;
                let (d_in, d_in_guard) = slice_ptr(slice, 0);
                (d_in as *const std::ffi::c_void, d_in_guard)
            }
            candle_core::DType::BF16 => {
                let slice = storage.as_cuda_slice::<half::bf16>()?;
                let (d_in, d_in_guard) = slice_ptr(slice, 0);
                (d_in as *const std::ffi::c_void, d_in_guard)
            }
            candle_core::DType::F16 => {
                let slice = storage.as_cuda_slice::<half::f16>()?;
                let (d_in, d_in_guard) = slice_ptr(slice, 0);
                (d_in as *const std::ffi::c_void, d_in_guard)
            }
            candle_core::DType::F32 => {
                let slice = storage.as_cuda_slice::<f32>()?;
                let (d_in, d_in_guard) = slice_ptr(slice, 0);
                (d_in as *const std::ffi::c_void, d_in_guard)
            }
            candle_core::DType::F64 => {
                let slice = storage.as_cuda_slice::<f64>()?;
                let (d_in, d_in_guard) = slice_ptr(slice, 0);
                (d_in as *const std::ffi::c_void, d_in_guard)
            }
            _ => unreachable!(),
        };
        let n = layout.shape().elem_count();

        let num_nonzero = count_nonzero_cuda(
            storage.dtype(),
            d_in,
            u32::try_from(n)?,
            dev.cuda_stream().cu_stream(),
        );
        let d_out = unsafe { dev.alloc::<u32>(num_nonzero as usize * layout.dims().len()) }
            .map_err(|_| Error::Msg("Failed to allocate memory for nonzero result".to_string()))?;
        if num_nonzero != 0 {
            let (d_out, _d_out_guard) = d_out.device_ptr(d_out.stream());
            let dims = layout
                .dims()
                .iter()
                .map(|&x| u32::try_from(x).unwrap())
                .collect::<Vec<u32>>();
            let mut d_dims = unsafe { dev.alloc::<u32>(dims.len()) }?;
            dev.memcpy_htod(&dims, &mut d_dims)?;
            let (d_dims_ptr, _d_dims_guard) = d_dims.device_ptr(d_dims.stream());
            nonzero_cuda(
                storage.dtype(),
                d_in,
                u32::try_from(n)?,
                num_nonzero,
                d_dims_ptr as *const c_void,
                u32::try_from(layout.dims().len())?,
                d_out as *mut c_void,
                dev.cuda_stream().cu_stream(),
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

        let encoder = s1.device().command_encoder()?;
        encoder.set_label("cumsum");

        let device = s1.device();

        let out_shape = l1.shape().clone();

        let output = device.new_buffer(out_shape.elem_count(), s1.dtype(), "cumsum")?;

        crate::metal_kernels::call_scan(
            device.device(),
            &encoder,
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

/// Fused GPT-OSS SwiGLU activation
/// Formula: output = (clamp(up, -limit, limit) + 1) * gate_clamped * sigmoid(gate_clamped * alpha)
/// where gate_clamped = min(gate, limit)
#[cfg(feature = "cuda")]
pub fn gptoss_swiglu_fused(gate: &Tensor, up: &Tensor, alpha: f32, limit: f32) -> Result<Tensor> {
    use half::{bf16, f16};

    let gate = gate.contiguous()?;
    let up = up.contiguous()?;

    if gate.shape() != up.shape() {
        candle_core::bail!(
            "gptoss_swiglu: gate and up must have same shape, got {:?} vs {:?}",
            gate.shape(),
            up.shape()
        );
    }

    let device = match gate.device() {
        candle_core::Device::Cuda(dev) => dev,
        _ => candle_core::bail!("gptoss_swiglu requires CUDA device"),
    };

    let n_elements = gate.elem_count();
    let dtype = gate.dtype();

    let gate_storage = gate.storage_and_layout().0;
    let up_storage = up.storage_and_layout().0;

    let gate_cuda = match &*gate_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("Expected CUDA storage for gate"),
    };
    let up_cuda = match &*up_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("Expected CUDA storage for up"),
    };

    let stream = device.cuda_stream().cu_stream();

    match dtype {
        DType::F16 => {
            let output = device.alloc_zeros::<f16>(n_elements)?;
            let gate_slice = gate_cuda.as_cuda_slice::<f16>()?;
            let up_slice = up_cuda.as_cuda_slice::<f16>()?;

            let (gate_ptr, _g_guard) = slice_ptr(gate_slice, 0);
            let (up_ptr, _u_guard) = slice_ptr(up_slice, 0);
            let (out_ptr, _o_guard) = slice_ptr(&output, 0);

            unsafe {
                ffi::gptoss_swiglu_f16(
                    gate_ptr as *const c_void,
                    up_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    n_elements as u32,
                    alpha,
                    limit,
                    stream,
                );
            }

            drop(_o_guard);
            let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
            Ok(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                gate.shape().clone(),
            )))
        }
        DType::BF16 => {
            let output = device.alloc_zeros::<bf16>(n_elements)?;
            let gate_slice = gate_cuda.as_cuda_slice::<bf16>()?;
            let up_slice = up_cuda.as_cuda_slice::<bf16>()?;

            let (gate_ptr, _g_guard) = slice_ptr(gate_slice, 0);
            let (up_ptr, _u_guard) = slice_ptr(up_slice, 0);
            let (out_ptr, _o_guard) = slice_ptr(&output, 0);

            unsafe {
                ffi::gptoss_swiglu_bf16(
                    gate_ptr as *const c_void,
                    up_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    n_elements as u32,
                    alpha,
                    limit,
                    stream,
                );
            }

            drop(_o_guard);
            let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
            Ok(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                gate.shape().clone(),
            )))
        }
        DType::F32 => {
            let output = device.alloc_zeros::<f32>(n_elements)?;
            let gate_slice = gate_cuda.as_cuda_slice::<f32>()?;
            let up_slice = up_cuda.as_cuda_slice::<f32>()?;

            let (gate_ptr, _g_guard) = slice_ptr(gate_slice, 0);
            let (up_ptr, _u_guard) = slice_ptr(up_slice, 0);
            let (out_ptr, _o_guard) = slice_ptr(&output, 0);

            unsafe {
                ffi::gptoss_swiglu_f32(
                    gate_ptr as *const c_void,
                    up_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    n_elements as u32,
                    alpha,
                    limit,
                    stream,
                );
            }

            drop(_o_guard);
            let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
            Ok(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                gate.shape().clone(),
            )))
        }
        _ => candle_core::bail!("gptoss_swiglu: unsupported dtype {:?}", dtype),
    }
}

/// Fused GPT-OSS SwiGLU for interleaved gate/up data.
///
/// This handles interleaved gate/up format directly, avoiding 2 tensor copies
/// from narrow().squeeze().contiguous().
///
/// Args:
///   gate_up: [N, intermediate_size, 2] - interleaved gate/up data
///   alpha: SwiGLU alpha parameter
///   limit: SwiGLU limit parameter
///
/// Returns: [N, intermediate_size] - activated output
#[cfg(feature = "cuda")]
pub fn gptoss_swiglu_interleaved(
    gate_up: &Tensor,
    intermediate_size: usize,
    alpha: f32,
    limit: f32,
) -> Result<Tensor> {
    use half::{bf16, f16};
    use std::ffi::c_void;

    let gate_up = gate_up.contiguous()?;

    let dims = gate_up.dims();
    if dims.len() != 3 || dims[2] != 2 {
        candle_core::bail!(
            "gptoss_swiglu_interleaved: expected gate_up shape [N, intermediate_size, 2], got {:?}",
            dims
        );
    }

    let n = dims[0]; // num_tokens * topk
    let device = match gate_up.device() {
        candle_core::Device::Cuda(dev) => dev,
        _ => candle_core::bail!("gptoss_swiglu_interleaved requires CUDA device"),
    };

    let dtype = gate_up.dtype();
    let n_output_elements = n * intermediate_size;

    let gate_up_storage = gate_up.storage_and_layout().0;
    let gate_up_cuda = match &*gate_up_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("Expected CUDA storage for gate_up"),
    };

    let stream = device.cuda_stream().cu_stream();

    match dtype {
        DType::F16 => {
            let output = device.alloc_zeros::<f16>(n_output_elements)?;
            let gate_up_slice = gate_up_cuda.as_cuda_slice::<f16>()?;

            let (gate_up_ptr, _gu_guard) = slice_ptr(gate_up_slice, 0);
            let (out_ptr, _o_guard) = slice_ptr(&output, 0);

            unsafe {
                ffi::gptoss_swiglu_interleaved_f16(
                    gate_up_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    n as u32,
                    intermediate_size as u32,
                    alpha,
                    limit,
                    stream,
                );
            }

            drop(_o_guard);
            let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
            Ok(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                Shape::from(vec![n, intermediate_size]),
            )))
        }
        DType::BF16 => {
            let output = device.alloc_zeros::<bf16>(n_output_elements)?;
            let gate_up_slice = gate_up_cuda.as_cuda_slice::<bf16>()?;

            let (gate_up_ptr, _gu_guard) = slice_ptr(gate_up_slice, 0);
            let (out_ptr, _o_guard) = slice_ptr(&output, 0);

            unsafe {
                ffi::gptoss_swiglu_interleaved_bf16(
                    gate_up_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    n as u32,
                    intermediate_size as u32,
                    alpha,
                    limit,
                    stream,
                );
            }

            drop(_o_guard);
            let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
            Ok(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                Shape::from(vec![n, intermediate_size]),
            )))
        }
        DType::F32 => {
            let output = device.alloc_zeros::<f32>(n_output_elements)?;
            let gate_up_slice = gate_up_cuda.as_cuda_slice::<f32>()?;

            let (gate_up_ptr, _gu_guard) = slice_ptr(gate_up_slice, 0);
            let (out_ptr, _o_guard) = slice_ptr(&output, 0);

            unsafe {
                ffi::gptoss_swiglu_interleaved_f32(
                    gate_up_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    n as u32,
                    intermediate_size as u32,
                    alpha,
                    limit,
                    stream,
                );
            }

            drop(_o_guard);
            let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
            Ok(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                Shape::from(vec![n, intermediate_size]),
            )))
        }
        _ => candle_core::bail!("gptoss_swiglu_interleaved: unsupported dtype {:?}", dtype),
    }
}

/// Fused softmax with sinks for GPT-OSS attention.
///
/// This computes softmax over attention logits while including a per-head "sink" value
/// in the normalization, then drops the sink from the output.
///
/// Args:
///   logits: [batch, heads, q_len, k_len] - attention scores (q @ k.T * scale)
///   sinks: [heads] - per-head sink values
///   mask: Optional [batch, 1, q_len, k_len] - attention mask (0 = attend, -inf = mask)
///
/// Returns: [batch, heads, q_len, k_len] - softmax probabilities (sink dropped from normalization)
struct SoftmaxWithSinks {
    sinks: Tensor,
    num_heads: usize,
    q_len: usize,
    k_len: usize,
}

impl CustomOp1 for SoftmaxWithSinks {
    fn name(&self) -> &'static str {
        "softmax-with-sinks"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        use half::{bf16, f16};

        let out_shape = layout.shape().clone();
        let total_rows = out_shape.elem_count() / self.k_len;
        let k_len = self.k_len;
        let num_heads = self.num_heads;
        let q_len = self.q_len;
        let offset = layout.start_offset();

        let sinks_data = self.sinks.storage_and_layout();
        let sinks_cpu = match &*sinks_data.0 {
            candle_core::Storage::Cpu(s) => s,
            _ => candle_core::bail!("softmax_with_sinks cpu_fwd: sinks must be on CPU"),
        };
        let sinks_offset = sinks_data.1.start_offset();

        match storage.dtype() {
            DType::F32 => {
                let logits = storage.as_slice::<f32>()?;
                let sinks_vals = sinks_cpu.as_slice::<f32>()?;

                let mut result = vec![0f32; total_rows * k_len];
                result
                    .par_chunks_mut(k_len)
                    .enumerate()
                    .for_each(|(row, out_row)| {
                        let h = (row / q_len) % num_heads;
                        let sink_val = sinks_vals[sinks_offset + h];
                        let row_start = offset + row * k_len;

                        let mut max_val = sink_val;
                        for k in 0..k_len {
                            let v = logits[row_start + k];
                            if v > max_val {
                                max_val = v;
                            }
                        }

                        let mut sum = (sink_val - max_val).exp();
                        for k in 0..k_len {
                            let e = (logits[row_start + k] - max_val).exp();
                            out_row[k] = e;
                            sum += e;
                        }

                        let inv_sum = 1.0 / sum;
                        for item in out_row.iter_mut().take(k_len) {
                            *item *= inv_sum;
                        }
                    });

                Ok((CpuStorage::F32(result), out_shape))
            }
            DType::F16 => {
                let logits = storage.as_slice::<f16>()?;
                let sinks_vals = sinks_cpu.as_slice::<f16>()?;

                let mut result = vec![f16::ZERO; total_rows * k_len];
                result
                    .par_chunks_mut(k_len)
                    .enumerate()
                    .for_each(|(row, out_row)| {
                        let h = (row / q_len) % num_heads;
                        let sink_val = sinks_vals[sinks_offset + h].to_f32();
                        let row_start = offset + row * k_len;

                        let mut max_val = sink_val;
                        for k in 0..k_len {
                            let v = logits[row_start + k].to_f32();
                            if v > max_val {
                                max_val = v;
                            }
                        }

                        let mut sum = (sink_val - max_val).exp();
                        for k in 0..k_len {
                            let e = (logits[row_start + k].to_f32() - max_val).exp();
                            out_row[k] = f16::from_f32(e);
                            sum += e;
                        }

                        let inv_sum = 1.0f32 / sum;
                        for item in out_row.iter_mut().take(k_len) {
                            *item = f16::from_f32(item.to_f32() * inv_sum);
                        }
                    });

                Ok((CpuStorage::F16(result), out_shape))
            }
            DType::BF16 => {
                let logits = storage.as_slice::<bf16>()?;
                let sinks_vals = sinks_cpu.as_slice::<bf16>()?;

                let mut result = vec![bf16::ZERO; total_rows * k_len];
                result
                    .par_chunks_mut(k_len)
                    .enumerate()
                    .for_each(|(row, out_row)| {
                        let h = (row / q_len) % num_heads;
                        let sink_val = sinks_vals[sinks_offset + h].to_f32();
                        let row_start = offset + row * k_len;

                        let mut max_val = sink_val;
                        for k in 0..k_len {
                            let v = logits[row_start + k].to_f32();
                            if v > max_val {
                                max_val = v;
                            }
                        }

                        let mut sum = (sink_val - max_val).exp();
                        for k in 0..k_len {
                            let e = (logits[row_start + k].to_f32() - max_val).exp();
                            out_row[k] = bf16::from_f32(e);
                            sum += e;
                        }

                        let inv_sum = 1.0f32 / sum;
                        for item in out_row.iter_mut().take(k_len) {
                            *item = bf16::from_f32(item.to_f32() * inv_sum);
                        }
                    });

                Ok((CpuStorage::BF16(result), out_shape))
            }
            other => candle_core::bail!("softmax_with_sinks: unsupported dtype {:?}", other),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use half::{bf16, f16};

        let device = storage.device();
        let dtype = storage.dtype();
        let n_elements = layout.shape().elem_count();
        let out_shape = layout.shape().clone();
        let stream = device.cuda_stream().cu_stream();
        let logits_offset = layout.start_offset();

        let batch_size = out_shape.dims()[0];

        let sinks_data = self.sinks.storage_and_layout();
        let sinks_cuda = match &*sinks_data.0 {
            candle_core::Storage::Cuda(s) => s,
            _ => candle_core::bail!("softmax_with_sinks cuda_fwd: sinks must be on CUDA"),
        };
        let sinks_offset = sinks_data.1.start_offset();

        match dtype {
            DType::F16 => {
                let output = device.alloc_zeros::<f16>(n_elements)?;
                let logits_slice = storage.as_cuda_slice::<f16>()?;
                let sinks_slice = sinks_cuda.as_cuda_slice::<f16>()?;

                let (logits_ptr, _l_guard) = slice_ptr(logits_slice, logits_offset);
                let (sinks_ptr, _s_guard) = slice_ptr(sinks_slice, sinks_offset);
                let (out_ptr, _o_guard) = slice_ptr(&output, 0);

                unsafe {
                    ffi::softmax_with_sinks_f16(
                        logits_ptr as *const c_void,
                        sinks_ptr as *const c_void,
                        std::ptr::null(), // mask pre-applied
                        out_ptr as *mut c_void,
                        batch_size as i32,
                        self.num_heads as i32,
                        self.q_len as i32,
                        self.k_len as i32,
                        1.0,
                        stream,
                    );
                }

                drop(_o_guard);
                let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((out_storage, out_shape))
            }
            DType::BF16 => {
                let output = device.alloc_zeros::<bf16>(n_elements)?;
                let logits_slice = storage.as_cuda_slice::<bf16>()?;
                let sinks_slice = sinks_cuda.as_cuda_slice::<bf16>()?;

                let (logits_ptr, _l_guard) = slice_ptr(logits_slice, logits_offset);
                let (sinks_ptr, _s_guard) = slice_ptr(sinks_slice, sinks_offset);
                let (out_ptr, _o_guard) = slice_ptr(&output, 0);

                unsafe {
                    ffi::softmax_with_sinks_bf16(
                        logits_ptr as *const c_void,
                        sinks_ptr as *const c_void,
                        std::ptr::null(),
                        out_ptr as *mut c_void,
                        batch_size as i32,
                        self.num_heads as i32,
                        self.q_len as i32,
                        self.k_len as i32,
                        1.0,
                        stream,
                    );
                }

                drop(_o_guard);
                let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((out_storage, out_shape))
            }
            DType::F32 => {
                let output = device.alloc_zeros::<f32>(n_elements)?;
                let logits_slice = storage.as_cuda_slice::<f32>()?;
                let sinks_slice = sinks_cuda.as_cuda_slice::<f32>()?;

                let (logits_ptr, _l_guard) = slice_ptr(logits_slice, logits_offset);
                let (sinks_ptr, _s_guard) = slice_ptr(sinks_slice, sinks_offset);
                let (out_ptr, _o_guard) = slice_ptr(&output, 0);

                unsafe {
                    ffi::softmax_with_sinks_f32(
                        logits_ptr as *const c_void,
                        sinks_ptr as *const c_void,
                        std::ptr::null(),
                        out_ptr as *mut c_void,
                        batch_size as i32,
                        self.num_heads as i32,
                        self.q_len as i32,
                        self.k_len as i32,
                        1.0,
                        stream,
                    );
                }

                drop(_o_guard);
                let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((out_storage, out_shape))
            }
            _ => candle_core::bail!("softmax_with_sinks: unsupported dtype {:?}", dtype),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let dtype = storage.dtype();
        let n_elements = layout.shape().elem_count();
        let out_shape = layout.shape().clone();
        let total_rows = n_elements / self.k_len;

        let device = storage.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("softmax-with-sinks");

        let output = device.new_buffer(n_elements, dtype, "softmax-with-sinks-output")?;

        let sinks_data = self.sinks.storage_and_layout();
        let sinks_metal = match &*sinks_data.0 {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("softmax_with_sinks metal_fwd: sinks must be on Metal"),
        };
        let sinks_offset = sinks_data.1.start_offset() * self.sinks.dtype().size_in_bytes();

        crate::metal_kernels::call_softmax_with_sinks(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            dtype,
            storage.buffer(),
            layout.start_offset() * dtype.size_in_bytes(),
            sinks_metal.buffer(),
            sinks_offset,
            &output,
            self.num_heads as u32,
            self.q_len as u32,
            self.k_len as u32,
            total_rows,
        )
        .map_err(candle_core::Error::wrap)?;

        let newstorage = candle_core::MetalStorage::new(output, device.clone(), n_elements, dtype);
        Ok((newstorage, out_shape))
    }
}

pub fn softmax_with_sinks(
    logits: &Tensor,
    sinks: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let logits = if let Some(mask) = mask {
        logits.broadcast_add(mask)?
    } else {
        logits.clone()
    };
    let logits = logits.contiguous()?;
    let sinks = sinks.contiguous()?;

    let dims = logits.dims();
    if dims.len() != 4 {
        candle_core::bail!(
            "softmax_with_sinks: expected logits to have 4 dims [b, h, q, k], got {:?}",
            dims
        );
    }

    let num_heads = dims[1];
    let q_len = dims[2];
    let k_len = dims[3];

    if sinks.dims() != [num_heads] {
        candle_core::bail!(
            "softmax_with_sinks: expected sinks shape [{}], got {:?}",
            num_heads,
            sinks.dims()
        );
    }

    logits.apply_op1_no_bwd(&SoftmaxWithSinks {
        sinks: sinks.clone(),
        num_heads,
        q_len,
        k_len,
    })
}

// ============================================================================
// Fused flash attention with sinks (Metal)
// ============================================================================

#[allow(dead_code)]
struct FlashAttnSinksMetal {
    key: Tensor,
    value: Tensor,
    sinks: Tensor, // [num_heads], always f32
    softmax_scale: f32,
    window_size: usize,
}

impl CustomOp1 for FlashAttnSinksMetal {
    fn name(&self) -> &'static str {
        "flash-attn-sinks-metal"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!(
            "flash_attn_sinks_metal: no CPU support, use softmax_with_sinks fallback"
        )
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q_storage: &candle_core::MetalStorage,
        q_layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let dtype = q_storage.dtype();
        let out_shape = q_layout.shape().clone();
        let (batch_size, num_heads, q_len, head_dim) = q_layout.shape().dims4()?;

        // Extract K storage
        let (k_s, k_l) = self.key.storage_and_layout();
        let k_metal = match &*k_s {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("flash_attn_sinks_metal: key must be a Metal tensor"),
        };
        let (_, num_kv_heads, k_len, _) = k_l.shape().dims4()?;

        // Extract V storage
        let (v_s, v_l) = self.value.storage_and_layout();
        let v_metal = match &*v_s {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("flash_attn_sinks_metal: value must be a Metal tensor"),
        };

        // Extract sinks storage
        let (s_s, s_l) = self.sinks.storage_and_layout();
        let sinks_metal = match &*s_s {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("flash_attn_sinks_metal: sinks must be a Metal tensor"),
        };
        let sinks_offset = s_l.start_offset() * self.sinks.dtype().size_in_bytes();

        let device = q_storage.device();
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(elem_count, dtype, "flash-attn-sinks-output")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("flash-attn-sinks");

        let kernels = crate::metal_kernels::Kernels::new();

        let q_offset = q_layout.start_offset() * dtype.size_in_bytes();
        let k_offset = k_l.start_offset() * dtype.size_in_bytes();
        let v_offset = v_l.start_offset() * dtype.size_in_bytes();

        if q_len == 1 {
            // Decode path: use sdpa_vector_with_sinks
            let gqa_factor = (num_heads / num_kv_heads) as i32;
            let b = batch_size * num_heads;

            // k_stride and v_stride: stride between consecutive KV positions in the head dimension
            // For contiguous [B, Hkv, S, D] layout: stride between KV heads = S * D
            let k_stride = k_l.stride()[1]; // stride for kv_head dim (= k_len * head_dim)
            let v_stride = v_l.stride()[1];

            let two_pass_threshold = 1024;
            if k_len >= two_pass_threshold {
                // Two-pass for long contexts
                let blocks: usize = 32;
                let intermediate = device.new_buffer(
                    b * blocks * head_dim,
                    DType::F32,
                    "sdpa-sinks-intermediate",
                )?;
                let sums = device.new_buffer(b * blocks, DType::F32, "sdpa-sinks-sums")?;
                let maxs = device.new_buffer(b * blocks, DType::F32, "sdpa-sinks-maxs")?;

                crate::metal_kernels::call_sdpa_vector_with_sinks_2pass(
                    device.device(),
                    &encoder,
                    &kernels,
                    dtype,
                    q_storage.buffer(),
                    q_offset,
                    k_metal.buffer(),
                    k_offset,
                    v_metal.buffer(),
                    v_offset,
                    sinks_metal.buffer(),
                    sinks_offset,
                    &output,
                    &intermediate,
                    &sums,
                    &maxs,
                    head_dim,
                    gqa_factor,
                    k_len as i32,
                    k_stride,
                    v_stride,
                    self.softmax_scale,
                    b,
                )
                .map_err(candle_core::Error::wrap)?;
            } else {
                // Single-pass
                crate::metal_kernels::call_sdpa_vector_with_sinks(
                    device.device(),
                    &encoder,
                    &kernels,
                    dtype,
                    q_storage.buffer(),
                    q_offset,
                    k_metal.buffer(),
                    k_offset,
                    v_metal.buffer(),
                    v_offset,
                    sinks_metal.buffer(),
                    sinks_offset,
                    &output,
                    head_dim,
                    gqa_factor,
                    k_len as i32,
                    k_stride,
                    v_stride,
                    self.softmax_scale,
                    b,
                )
                .map_err(candle_core::Error::wrap)?;
            }
        } else {
            // Prefill path: use flash_attn_sinks_kernel
            crate::metal_kernels::call_flash_attn_sinks_prefill(
                device.device(),
                &encoder,
                &kernels,
                dtype,
                q_storage.buffer(),
                q_offset,
                k_metal.buffer(),
                k_offset,
                v_metal.buffer(),
                v_offset,
                sinks_metal.buffer(),
                sinks_offset,
                &output,
                self.softmax_scale,
                batch_size,
                q_len,
                k_len,
                num_heads,
                num_kv_heads,
                head_dim,
                self.window_size,
            )
            .map_err(candle_core::Error::wrap)?;
        }

        let newstorage = candle_core::MetalStorage::new(output, device.clone(), elem_count, dtype);
        Ok((newstorage, out_shape))
    }
}

/// Fused flash attention with per-head sinks for Metal devices.
///
/// Uses fused Metal kernels that compute Q·K^T -> softmax_with_sinks -> ×V
/// without materializing the N×N attention matrix. Per-head sinks contribute
/// to the softmax denominator without an associated value contribution.
///
/// Causal masking is applied for prefill (q_len > 1). For decode (q_len == 1),
/// all K/V positions are attended to.
///
/// # Arguments
///
/// * `q` - Query tensor `[batch_size, num_heads, q_len, head_dim]`
/// * `k` - Key tensor `[batch_size, num_kv_heads, k_len, head_dim]`
/// * `v` - Value tensor `[batch_size, num_kv_heads, k_len, head_dim]`
/// * `sinks` - Per-head sink values `[num_heads]` (will be cast to f32)
/// * `softmax_scale` - Scaling factor (typically `1 / sqrt(head_dim)`)
/// * `window_size` - Sliding window size (0 = full attention)
///
/// Returns `[batch_size, num_heads, q_len, head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_sinks_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    let sinks = match sinks {
        Some(s) => s.to_dtype(DType::F32)?.contiguous()?,
        None => {
            // No sinks: create zeros (no effect on softmax)
            let num_heads = q.dim(1)?;
            Tensor::zeros(num_heads, DType::F32, q.device())?
        }
    };

    let op = FlashAttnSinksMetal {
        key: k.clone(),
        value: v.clone(),
        sinks,
        softmax_scale,
        window_size,
    };
    q.apply_op1_no_bwd(&op)
}

#[allow(dead_code)]
struct FlashAttnSinksVarlenMetal {
    key: Tensor,          // [total_kv, num_kv_heads, D]
    value: Tensor,        // [total_kv, num_kv_heads, D]
    sinks: Tensor,        // [num_heads], always f32
    cu_seqlens_q: Tensor, // [B+1] u32
    cu_seqlens_k: Tensor, // [B+1] u32
    softmax_scale: f32,
    window_size: usize,
}

impl CustomOp1 for FlashAttnSinksVarlenMetal {
    fn name(&self) -> &'static str {
        "flash-attn-sinks-varlen-metal"
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("flash_attn_sinks_varlen_metal: no CPU support")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q_storage: &candle_core::MetalStorage,
        q_layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let dtype = q_storage.dtype();
        let out_shape = q_layout.shape().clone();
        let (batch_size, num_heads, max_q_len, head_dim) = q_layout.shape().dims4()?;

        // Extract K storage [total_kv, num_kv_heads, D]
        let (k_s, k_l) = self.key.storage_and_layout();
        let k_metal = match &*k_s {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("flash_attn_sinks_varlen_metal: key must be a Metal tensor"),
        };
        let (_, num_kv_heads, _) = k_l.shape().dims3()?;

        // Extract V storage
        let (v_s, v_l) = self.value.storage_and_layout();
        let v_metal = match &*v_s {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("flash_attn_sinks_varlen_metal: value must be a Metal tensor"),
        };

        // Extract sinks storage
        let (s_s, s_l) = self.sinks.storage_and_layout();
        let sinks_metal = match &*s_s {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!("flash_attn_sinks_varlen_metal: sinks must be a Metal tensor"),
        };
        let sinks_offset = s_l.start_offset() * self.sinks.dtype().size_in_bytes();

        // Extract cu_seqlens_q storage
        let (csq_s, csq_l) = self.cu_seqlens_q.storage_and_layout();
        let csq_metal = match &*csq_s {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!(
                "flash_attn_sinks_varlen_metal: cu_seqlens_q must be a Metal tensor"
            ),
        };
        let csq_offset = csq_l.start_offset() * DType::U32.size_in_bytes();

        // Extract cu_seqlens_k storage
        let (csk_s, csk_l) = self.cu_seqlens_k.storage_and_layout();
        let csk_metal = match &*csk_s {
            candle_core::Storage::Metal(s) => s,
            _ => candle_core::bail!(
                "flash_attn_sinks_varlen_metal: cu_seqlens_k must be a Metal tensor"
            ),
        };
        let csk_offset = csk_l.start_offset() * DType::U32.size_in_bytes();

        let device = q_storage.device();
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(elem_count, dtype, "flash-attn-sinks-varlen-output")?;

        let encoder = device.command_encoder()?;
        encoder.set_label("flash-attn-sinks-varlen");

        let kernels = crate::metal_kernels::Kernels::new();

        let q_offset = q_layout.start_offset() * dtype.size_in_bytes();
        let k_offset = k_l.start_offset() * dtype.size_in_bytes();
        let v_offset = v_l.start_offset() * dtype.size_in_bytes();

        crate::metal_kernels::call_flash_attn_sinks_varlen_prefill(
            device.device(),
            &encoder,
            &kernels,
            dtype,
            q_storage.buffer(),
            q_offset,
            k_metal.buffer(),
            k_offset,
            v_metal.buffer(),
            v_offset,
            sinks_metal.buffer(),
            sinks_offset,
            &output,
            csq_metal.buffer(),
            csq_offset,
            csk_metal.buffer(),
            csk_offset,
            self.softmax_scale,
            batch_size,
            max_q_len,
            num_heads,
            num_kv_heads,
            head_dim,
            self.window_size,
        )
        .map_err(candle_core::Error::wrap)?;

        let newstorage = candle_core::MetalStorage::new(output, device.clone(), elem_count, dtype);
        Ok((newstorage, out_shape))
    }
}

/// Fused varlen flash attention with per-head sinks for Metal devices.
///
/// Handles variable-length sequences within a batch. Q is padded,
/// K/V are packed (concatenated across sequences).
///
/// # Arguments
///
/// * `q` - Query tensor `[batch_size, num_heads, max_q_len, head_dim]` (padded)
/// * `k` - Key tensor `[total_kv, num_kv_heads, head_dim]` (packed)
/// * `v` - Value tensor `[total_kv, num_kv_heads, head_dim]` (packed)
/// * `sinks` - Per-head sink values `[num_heads]` (will be cast to f32)
/// * `cu_seqlens_q` - Cumulative Q sequence lengths `[batch_size + 1]` (u32)
/// * `cu_seqlens_k` - Cumulative KV sequence lengths `[batch_size + 1]` (u32)
/// * `softmax_scale` - Scaling factor (typically `1 / sqrt(head_dim)`)
/// * `window_size` - Sliding window size (0 = full attention)
///
/// Returns `[batch_size, num_heads, max_q_len, head_dim]` (padding rows are zero)
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_sinks_varlen_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    let sinks = match sinks {
        Some(s) => s.to_dtype(DType::F32)?.contiguous()?,
        None => {
            let num_heads = q.dim(1)?;
            Tensor::zeros(num_heads, DType::F32, q.device())?
        }
    };

    let op = FlashAttnSinksVarlenMetal {
        key: k.clone(),
        value: v.clone(),
        sinks,
        cu_seqlens_q: cu_seqlens_q.clone(),
        cu_seqlens_k: cu_seqlens_k.clone(),
        softmax_scale,
        window_size,
    };
    q.apply_op1_no_bwd(&op)
}

/// Activation enum for fused GLU kernel.
/// Must match the GluActivation enum in CUDA (ops.cu) and Metal (fused_glu.metal) kernels.
#[derive(Clone, Copy, Debug)]
#[repr(i32)]
pub enum GluActivationType {
    Silu = 0,
    Gelu = 1,
    Relu = 2,
    GeluErf = 3,
}

// CPU activation functions for fused GLU
fn cpu_silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn cpu_gelu(x: f32) -> f32 {
    #[allow(clippy::excessive_precision)]
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    const COEFF: f32 = 0.044715;
    let x3 = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    0.5 * x * (1.0 + inner.tanh())
}

fn cpu_relu(x: f32) -> f32 {
    x.max(0.0)
}

fn cpu_gelu_erf(x: f32) -> f32 {
    // gelu_erf: x * (1 + erf(x / sqrt(2))) / 2
    x * (1.0 + candle_core::cpu::erf::erf_f32(x * std::f32::consts::FRAC_1_SQRT_2)) / 2.0
}

fn apply_cpu_activation(x: f32, activation: GluActivationType) -> f32 {
    match activation {
        GluActivationType::Silu => cpu_silu(x),
        GluActivationType::Gelu => cpu_gelu(x),
        GluActivationType::Relu => cpu_relu(x),
        GluActivationType::GeluErf => cpu_gelu_erf(x),
    }
}

struct FusedGlu(GluActivationType);

impl CustomOp2 for FusedGlu {
    fn name(&self) -> &'static str {
        "fused-glu"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use half::{bf16, f16};

        let activation = self.0;
        let out_shape = l1.shape().clone();
        let len = out_shape.elem_count();

        let result_storage = match s1.dtype() {
            DType::F32 => {
                let a_slice = s1.as_slice::<f32>()?;
                let b_slice = s2.as_slice::<f32>()?;
                let a_offset = l1.start_offset();
                let b_offset = l2.start_offset();

                let result: Vec<f32> = (0..len)
                    .into_par_iter()
                    .map(|i| {
                        let a_val = a_slice[a_offset + i];
                        let b_val = b_slice[b_offset + i];
                        apply_cpu_activation(a_val, activation) * b_val
                    })
                    .collect();
                CpuStorage::F32(result)
            }
            DType::F16 => {
                let a_slice = s1.as_slice::<f16>()?;
                let b_slice = s2.as_slice::<f16>()?;
                let a_offset = l1.start_offset();
                let b_offset = l2.start_offset();

                let result: Vec<f16> = (0..len)
                    .into_par_iter()
                    .map(|i| {
                        let a_val = a_slice[a_offset + i].to_f32();
                        // Cast activation back to f16 before multiplying, matching candle's
                        // two-step behavior: unary op in f32 -> cast to f16 -> binary mul
                        let activated = f16::from_f32(apply_cpu_activation(a_val, activation));
                        f16::from_f32(activated.to_f32() * b_slice[b_offset + i].to_f32())
                    })
                    .collect();
                CpuStorage::F16(result)
            }
            DType::BF16 => {
                let a_slice = s1.as_slice::<bf16>()?;
                let b_slice = s2.as_slice::<bf16>()?;
                let a_offset = l1.start_offset();
                let b_offset = l2.start_offset();

                let result: Vec<bf16> = (0..len)
                    .into_par_iter()
                    .map(|i| {
                        let a_val = a_slice[a_offset + i].to_f32();
                        // Cast activation back to bf16 before multiplying, matching candle's
                        // two-step behavior: unary op in f32 -> cast to bf16 -> binary mul
                        let activated = bf16::from_f32(apply_cpu_activation(a_val, activation));
                        bf16::from_f32(activated.to_f32() * b_slice[b_offset + i].to_f32())
                    })
                    .collect();
                CpuStorage::BF16(result)
            }
            other => candle_core::bail!("fused_glu: unsupported dtype {:?}", other),
        };

        Ok((result_storage, out_shape))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use half::{bf16, f16};

        let activation = self.0;
        let device = s1.device();
        let n_elements = l1.shape().elem_count();
        let dtype = s1.dtype();
        let out_shape = l1.shape().clone();
        let stream = device.cuda_stream().cu_stream();
        let a_offset = l1.start_offset();
        let b_offset = l2.start_offset();

        match dtype {
            DType::F16 => {
                let output = device.alloc_zeros::<f16>(n_elements)?;
                let a_slice = s1.as_cuda_slice::<f16>()?;
                let b_slice = s2.as_cuda_slice::<f16>()?;

                let (a_ptr, _a_guard) = slice_ptr(a_slice, a_offset);
                let (b_ptr, _b_guard) = slice_ptr(b_slice, b_offset);
                let (out_ptr, _o_guard) = slice_ptr(&output, 0);

                unsafe {
                    ffi::fused_glu_f16(
                        a_ptr as *const c_void,
                        b_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        n_elements as u32,
                        activation as i32,
                        stream,
                    );
                }

                drop(_o_guard);
                let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((out_storage, out_shape))
            }
            DType::BF16 => {
                let output = device.alloc_zeros::<bf16>(n_elements)?;
                let a_slice = s1.as_cuda_slice::<bf16>()?;
                let b_slice = s2.as_cuda_slice::<bf16>()?;

                let (a_ptr, _a_guard) = slice_ptr(a_slice, a_offset);
                let (b_ptr, _b_guard) = slice_ptr(b_slice, b_offset);
                let (out_ptr, _o_guard) = slice_ptr(&output, 0);

                unsafe {
                    ffi::fused_glu_bf16(
                        a_ptr as *const c_void,
                        b_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        n_elements as u32,
                        activation as i32,
                        stream,
                    );
                }

                drop(_o_guard);
                let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((out_storage, out_shape))
            }
            DType::F32 => {
                let output = device.alloc_zeros::<f32>(n_elements)?;
                let a_slice = s1.as_cuda_slice::<f32>()?;
                let b_slice = s2.as_cuda_slice::<f32>()?;

                let (a_ptr, _a_guard) = slice_ptr(a_slice, a_offset);
                let (b_ptr, _b_guard) = slice_ptr(b_slice, b_offset);
                let (out_ptr, _o_guard) = slice_ptr(&output, 0);

                unsafe {
                    ffi::fused_glu_f32(
                        a_ptr as *const c_void,
                        b_ptr as *const c_void,
                        out_ptr as *mut c_void,
                        n_elements as u32,
                        activation as i32,
                        stream,
                    );
                }

                drop(_o_guard);
                let out_storage = CudaStorage::wrap_cuda_slice(output, device.clone());
                Ok((out_storage, out_shape))
            }
            _ => candle_core::bail!("fused_glu: unsupported dtype {:?}", dtype),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        let activation = self.0;
        let n_elements = l1.shape().elem_count();
        let dtype = s1.dtype();
        let out_shape = l1.shape().clone();

        let device = s1.device();
        let encoder = device.command_encoder()?;
        encoder.set_label("fused-glu");

        let output = device.new_buffer(n_elements, dtype, "fused-glu-output")?;

        crate::metal_kernels::call_fused_glu(
            device.device(),
            &encoder,
            &crate::metal_kernels::Kernels::new(),
            dtype,
            s1.buffer(),
            s2.buffer(),
            l1.start_offset() * dtype.size_in_bytes(),
            l2.start_offset() * dtype.size_in_bytes(),
            n_elements,
            activation as i32,
            &output,
        )
        .map_err(candle_core::Error::wrap)?;

        let newstorage = candle_core::MetalStorage::new(output, device.clone(), n_elements, dtype);
        Ok((newstorage, out_shape))
    }
}

/// Fused GLU activation: output = activation(a) * b
///
/// This fuses the activation function application and element-wise multiplication
/// into a single pass, reducing memory bandwidth and eliminating
/// intermediate tensor allocation.
pub fn fused_glu(a: &Tensor, b: &Tensor, activation: GluActivationType) -> Result<Tensor> {
    let a = a.contiguous()?;
    let b = b.contiguous()?;

    if a.shape() != b.shape() {
        candle_core::bail!(
            "fused_glu: a and b must have same shape, got {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
    }

    a.apply_op2_no_bwd(&b, &FusedGlu(activation))
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
        use candle_core::{Device, Tensor};
        let bits = HqqBits::Eight;
        let device = Device::new_cuda(0).unwrap();
        // Use U8 tensor directly to avoid candle's to_dtype which may not have
        // PTX compiled for newer GPU architectures (e.g., SM 120)
        let wq = Tensor::from_vec(vec![1_u8, 2, 3, 4, 255, 0], (3, 2), &device).unwrap();
        let c = bits.bitpack_type()(wq.clone())
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

    #[cfg(feature = "metal")]
    #[test]
    fn test_fused_glu_metal_silu_f32() {
        use super::{fused_glu, GluActivationType};
        use candle_core::Tensor;

        let cpu = candle_core::Device::Cpu;
        let metal = candle_core::Device::new_metal(0).unwrap();

        let a_data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
        let b_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.7 - 90.0) / 50.0).collect();

        let a_cpu = Tensor::from_vec(a_data.clone(), &[4, 64], &cpu).unwrap();
        let b_cpu = Tensor::from_vec(b_data.clone(), &[4, 64], &cpu).unwrap();
        let a_metal = Tensor::from_vec(a_data, &[4, 64], &metal).unwrap();
        let b_metal = Tensor::from_vec(b_data, &[4, 64], &metal).unwrap();

        let cpu_result = fused_glu(&a_cpu, &b_cpu, GluActivationType::Silu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let metal_result = fused_glu(&a_metal, &b_metal, GluActivationType::Silu)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        for (row_cpu, row_metal) in cpu_result.iter().zip(metal_result.iter()) {
            for (c, m) in row_cpu.iter().zip(row_metal.iter()) {
                let diff = (c - m).abs();
                assert!(
                    diff < 1e-4,
                    "SiLU F32 mismatch: cpu={c}, metal={m}, diff={diff}"
                );
            }
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_fused_glu_metal_silu_f16() {
        use super::{fused_glu, GluActivationType};
        use candle_core::{DType, Tensor};

        let cpu = candle_core::Device::Cpu;
        let metal = candle_core::Device::new_metal(0).unwrap();

        let a_data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
        let b_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.7 - 90.0) / 50.0).collect();

        let a_cpu = Tensor::from_vec(a_data.clone(), &[256], &cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let b_cpu = Tensor::from_vec(b_data.clone(), &[256], &cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let a_metal = Tensor::from_vec(a_data, &[256], &metal)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let b_metal = Tensor::from_vec(b_data, &[256], &metal)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let cpu_result = fused_glu(&a_cpu, &b_cpu, GluActivationType::Silu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let metal_result = fused_glu(&a_metal, &b_metal, GluActivationType::Silu)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (i, (c, m)) in cpu_result.iter().zip(metal_result.iter()).enumerate() {
            let diff = (c - m).abs();
            assert!(
                diff < 1e-2,
                "SiLU F16 mismatch at {i}: cpu={c}, metal={m}, diff={diff}"
            );
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_fused_glu_metal_all_activations() {
        use super::{fused_glu, GluActivationType};
        use candle_core::Tensor;

        let cpu = candle_core::Device::Cpu;
        let metal = candle_core::Device::new_metal(0).unwrap();

        let a_data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 32.0).collect();
        let b_data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.5 - 32.0) / 20.0).collect();

        for act in [
            GluActivationType::Silu,
            GluActivationType::Gelu,
            GluActivationType::Relu,
            GluActivationType::GeluErf,
        ] {
            let a_cpu = Tensor::from_vec(a_data.clone(), &[128], &cpu).unwrap();
            let b_cpu = Tensor::from_vec(b_data.clone(), &[128], &cpu).unwrap();
            let a_metal = Tensor::from_vec(a_data.clone(), &[128], &metal).unwrap();
            let b_metal = Tensor::from_vec(b_data.clone(), &[128], &metal).unwrap();

            let cpu_result = fused_glu(&a_cpu, &b_cpu, act)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let metal_result = fused_glu(&a_metal, &b_metal, act)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            for (i, (c, m)) in cpu_result.iter().zip(metal_result.iter()).enumerate() {
                let diff = (c - m).abs();
                assert!(
                    diff < 1e-4,
                    "{act:?} F32 mismatch at {i}: cpu={c}, metal={m}, diff={diff}"
                );
            }
        }
    }

    /// Test that fused_glu matches candle's fallback path (a.gelu() * b) for BF16.
    /// This was the exact scenario that caused model failure (Gemma 3 4B, BF16, GeluPytorchTanh).
    #[cfg(feature = "metal")]
    #[test]
    fn test_fused_glu_matches_candle_fallback_bf16() {
        use super::{fused_glu, GluActivationType};
        use candle_core::{DType, Tensor};

        let metal = candle_core::Device::new_metal(0).unwrap();

        // Use realistic-sized data matching model dimensions
        let n = 10240;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 - 5120.0) / 2560.0).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 - 1500.0) / 1000.0).collect();

        let a_metal = Tensor::from_vec(a_data.clone(), &[1, 2, n / 2], &metal)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let b_metal = Tensor::from_vec(b_data.clone(), &[1, 2, n / 2], &metal)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        // Fused path
        let fused = fused_glu(&a_metal, &b_metal, GluActivationType::Gelu).unwrap();

        // Candle's fallback: a.gelu() * b (the tanh-approx GELU)
        let fallback = (a_metal.gelu().unwrap() * &b_metal).unwrap();

        let fused_f32 = fused
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let fallback_f32 = fallback
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let mut max_diff: f32 = 0.0;
        let mut num_mismatches = 0;
        for (_i, (f, fb)) in fused_f32.iter().zip(fallback_f32.iter()).enumerate() {
            let diff = (f - fb).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 0.0 {
                num_mismatches += 1;
            }
        }
        eprintln!(
            "BF16 Gelu fused vs fallback: max_diff={max_diff}, mismatches={num_mismatches}/{}",
            fused_f32.len()
        );
        // Allow up to 1 BF16 ULP difference (0.015625 at values around 1-2)
        // This is acceptable since Metal compiler may keep intermediate precision
        assert!(
            max_diff <= 0.015625,
            "BF16 Gelu fused vs candle fallback max_diff {max_diff} exceeds 1 BF16 ULP"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fused_glu_cuda_silu_f32() {
        use super::{fused_glu, GluActivationType};
        use candle_core::Tensor;

        let cpu = candle_core::Device::Cpu;
        let cuda = candle_core::Device::new_cuda(0).unwrap();

        let a_data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
        let b_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.7 - 90.0) / 50.0).collect();

        let a_cpu = Tensor::from_vec(a_data.clone(), &[4, 64], &cpu).unwrap();
        let b_cpu = Tensor::from_vec(b_data.clone(), &[4, 64], &cpu).unwrap();
        let a_cuda = Tensor::from_vec(a_data, &[4, 64], &cuda).unwrap();
        let b_cuda = Tensor::from_vec(b_data, &[4, 64], &cuda).unwrap();

        let cpu_result = fused_glu(&a_cpu, &b_cpu, GluActivationType::Silu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let cuda_result = fused_glu(&a_cuda, &b_cuda, GluActivationType::Silu)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();

        for (row_cpu, row_cuda) in cpu_result.iter().zip(cuda_result.iter()) {
            for (c, g) in row_cpu.iter().zip(row_cuda.iter()) {
                let diff = (c - g).abs();
                assert!(
                    diff < 1e-4,
                    "SiLU F32 mismatch: cpu={c}, cuda={g}, diff={diff}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fused_glu_cuda_silu_f16() {
        use super::{fused_glu, GluActivationType};
        use candle_core::{DType, Tensor};

        let cpu = candle_core::Device::Cpu;
        let cuda = candle_core::Device::new_cuda(0).unwrap();

        let a_data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 64.0).collect();
        let b_data: Vec<f32> = (0..256).map(|i| (i as f32 * 0.7 - 90.0) / 50.0).collect();

        let a_cpu = Tensor::from_vec(a_data.clone(), &[256], &cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let b_cpu = Tensor::from_vec(b_data.clone(), &[256], &cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let a_cuda = Tensor::from_vec(a_data, &[256], &cuda)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let b_cuda = Tensor::from_vec(b_data, &[256], &cuda)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let cpu_result = fused_glu(&a_cpu, &b_cpu, GluActivationType::Silu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let cuda_result = fused_glu(&a_cuda, &b_cuda, GluActivationType::Silu)
            .unwrap()
            .to_device(&cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        for (i, (c, g)) in cpu_result.iter().zip(cuda_result.iter()).enumerate() {
            let diff = (c - g).abs();
            assert!(
                diff < 1e-2,
                "SiLU F16 mismatch at {i}: cpu={c}, cuda={g}, diff={diff}"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fused_glu_cuda_all_activations() {
        use super::{fused_glu, GluActivationType};
        use candle_core::Tensor;

        let cpu = candle_core::Device::Cpu;
        let cuda = candle_core::Device::new_cuda(0).unwrap();

        let a_data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 32.0).collect();
        let b_data: Vec<f32> = (0..128).map(|i| (i as f32 * 0.5 - 32.0) / 20.0).collect();

        for act in [
            GluActivationType::Silu,
            GluActivationType::Gelu,
            GluActivationType::Relu,
            GluActivationType::GeluErf,
        ] {
            let a_cpu = Tensor::from_vec(a_data.clone(), &[128], &cpu).unwrap();
            let b_cpu = Tensor::from_vec(b_data.clone(), &[128], &cpu).unwrap();
            let a_cuda = Tensor::from_vec(a_data.clone(), &[128], &cuda).unwrap();
            let b_cuda = Tensor::from_vec(b_data.clone(), &[128], &cuda).unwrap();

            let cpu_result = fused_glu(&a_cpu, &b_cpu, act)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let cuda_result = fused_glu(&a_cuda, &b_cuda, act)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            for (i, (c, g)) in cpu_result.iter().zip(cuda_result.iter()).enumerate() {
                let diff = (c - g).abs();
                assert!(
                    diff < 1e-4,
                    "{act:?} F32 mismatch at {i}: cpu={c}, cuda={g}, diff={diff}"
                );
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_fused_glu_matches_candle_fallback_bf16_cuda() {
        use super::{fused_glu, GluActivationType};
        use candle_core::{DType, Tensor};

        let cuda = candle_core::Device::new_cuda(0).unwrap();

        let n = 10240;
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32 - 5120.0) / 2560.0).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 - 1500.0) / 1000.0).collect();

        let a_cuda = Tensor::from_vec(a_data.clone(), &[1, 2, n / 2], &cuda)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let b_cuda = Tensor::from_vec(b_data.clone(), &[1, 2, n / 2], &cuda)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

        // Fused path
        let fused = fused_glu(&a_cuda, &b_cuda, GluActivationType::Gelu).unwrap();

        // Candle's fallback: a.gelu() * b (the tanh-approx GELU)
        let fallback = (a_cuda.gelu().unwrap() * &b_cuda).unwrap();

        let fused_f32 = fused
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let fallback_f32 = fallback
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let mut max_diff: f32 = 0.0;
        let mut num_mismatches = 0;
        for (_i, (f, fb)) in fused_f32.iter().zip(fallback_f32.iter()).enumerate() {
            let diff = (f - fb).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 0.0 {
                num_mismatches += 1;
            }
        }
        eprintln!(
            "CUDA BF16 Gelu fused vs fallback: max_diff={max_diff}, mismatches={num_mismatches}/{}",
            fused_f32.len()
        );
        // Allow up to 1 BF16 ULP difference (0.015625 at values around 1-2)
        assert!(
            max_diff <= 0.015625,
            "CUDA BF16 Gelu fused vs candle fallback max_diff {max_diff} exceeds 1 BF16 ULP"
        );
    }
}
