use std::{fmt::Debug, ops::Deref, sync::Arc};

use candle_core::{
    backend::BackendStorage,
    cuda::cudarc::{self, nccl::Id},
    cuda_backend::WrapErr,
    CpuStorage, CustomOp1, DType, Device, Layout, Result, Shape,
};

#[derive(Debug)]
pub struct Comm(cudarc::nccl::Comm);

impl Comm {
    pub fn from_device(dev: &Device, rank: usize, world_size: usize) -> Result<Self> {
        let id = Id::new().expect("Failed to create `Id`.");
        let Device::Cuda(device) = dev else {
            candle_core::bail!("Expected CUDA device.")
        };
        Ok(Self(
            cudarc::nccl::Comm::from_rank(device.cuda_device(), rank, world_size, id)
                .expect("Failed to create `Comm`"),
        ))
    }
}

/// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
unsafe impl Sync for Comm {}
unsafe impl Send for Comm {}

impl Deref for Comm {
    type Target = cudarc::nccl::Comm;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug)]
pub struct AllReduce {
    comm: Arc<Comm>,
}

impl AllReduce {
    pub fn new(comm: &Arc<Comm>) -> Self {
        Self { comm: comm.clone() }
    }
}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("AllReduce is never used on cpu")
    }

    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use cudarc::{driver::DeviceSlice, nccl::ReduceOp};
        use half::{bf16, f16};

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let dst = match s.dtype() {
            DType::BF16 => {
                let s = s.as_cuda_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .0
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let s = s.as_cuda_slice::<f16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                self.comm
                    .0
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F32 => {
                let s = s.as_cuda_slice::<f32>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<f32>(elem_count) }.w()?;
                self.comm
                    .0
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}
