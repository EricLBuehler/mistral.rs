use std::{
    rc::Rc,
    sync::{LazyLock, RwLock},
};

use candle_core::{
    backend::BackendStorage,
    cuda::{cudarc::driver::DeviceSlice, WrapErr},
    DType, Device, Tensor,
};
use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape};
use cudarc::nccl::ReduceOp;
use cudarc::nccl::{Comm, Id};
use half::{bf16, f16};

thread_local! {
    static COMM: LazyLock<RwLock<Option<Rc<Comm>>>> = LazyLock::new(|| RwLock::new(None));
}

struct AllReduce {
    comm: Rc<Comm>,
    op: ReduceOp,
}

unsafe impl Sync for AllReduce {}
unsafe impl Send for AllReduce {}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "all-reduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("all-reduce is not supported on cpu")
    }

    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let dst = match s.dtype() {
            DType::F32 => {
                let s = s.as_cuda_slice::<f32>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<f32>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &self.op)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::BF16 => {
                let s = s.as_cuda_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle_core::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &self.op)
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
                    .all_reduce(s, &mut dst, &self.op)
                    .map_err(candle_core::Error::debug)?;
                candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}

pub fn init_comm(id: Id, rank: usize, world_size: usize, dev: &Device) -> Result<()> {
    let dev = dev.as_cuda_device()?;
    let comm = Comm::from_rank(dev.cuda_device(), rank, world_size, id)
        .map_err(candle_core::Error::debug)?;
    COMM.with(|x| {
        *x.write().unwrap() = Some(Rc::new(comm));
    });
    Ok(())
}

/// CUDA all-reduce operation:
/// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html#allreduce
pub fn all_reduce(x: &Tensor, op: ReduceOp) -> Result<Tensor> {
    let comm = COMM.with(|x| x.read().unwrap().as_ref().unwrap().clone());
    x.apply_op1_no_bwd(&AllReduce { comm, op })
}
