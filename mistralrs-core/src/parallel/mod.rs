use std::sync::Arc;

use candle_core::{CpuStorage, CustomOp1, Layout, Result, Shape, Tensor};
use candle_nn::{var_builder::ShardedVarBuilder as VarBuilder, Embedding, Linear, Module};
use cudarc::nccl::safe::{Comm, ReduceOp};

pub struct TensorParallelColumnLinear {
    linear: Linear,
}

impl TensorParallelColumnLinear {
    fn new(linear: Linear) -> Self {
        Self { linear }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

/// Kind of suspicous to do this: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
unsafe impl Sync for TensorParallelColumnLinear {}
/// Kind of suspicous to do this: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
unsafe impl Send for TensorParallelColumnLinear {}

pub struct TensorParallelRowLinear {
    linear: Linear,
    comm: Arc<Comm>,
}

/// Kind of suspicous to do this: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
unsafe impl Sync for TensorParallelRowLinear {}
/// Kind of suspicous to do this: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
unsafe impl Send for TensorParallelRowLinear {}

struct AllReduce {
    comm: Arc<Comm>,
}

/// Kind of suspicous to do this: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
unsafe impl Sync for AllReduce {}
/// Kind of suspicous to do this: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
unsafe impl Send for AllReduce {}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        todo!("implement allreduce for cpu is not necessary for single node");
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::{backend::BackendStorage, cuda_backend::WrapErr};
        use half::f16;
        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let s = s.as_cuda_slice::<half::f16>()?;
        let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
        self.comm.all_reduce(s, &mut dst, &ReduceOp::Sum).unwrap();
        let dst = candle_core::CudaStorage::wrap_cuda_slice(dst, dev);
        Ok((dst, l.shape().clone()))
    }
}

fn all_reduce_sum(x: &Tensor, comm: &Arc<Comm>) -> Result<Tensor> {
    x.apply_op1(AllReduce { comm: comm.clone() })
}

impl TensorParallelRowLinear {
    fn new(linear: Linear, comm: Arc<Comm>) -> Self {
        Self { linear, comm }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear.forward(x)?;
        all_reduce_sum(&x, &self.comm)
    }
}

pub fn shard(dim: usize, rank: usize, world_size: usize) -> candle_nn::var_builder::Shard {
    candle_nn::var_builder::Shard {
        dim,
        rank,
        world_size,
    }
}

impl TensorParallelColumnLinear {
    pub fn load(vb: VarBuilder, comm: Arc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(0, rank, size))?;
        Ok(Self::new(Linear::new(weight, None)))
    }

    pub fn load_multi(vb: VarBuilder, prefixes: &[&str], comm: Arc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weights: Vec<_> = prefixes
            .iter()
            .map(|p| vb.pp(p).get_with_hints((), "weight", shard(0, rank, size)))
            .collect::<Result<Vec<_>>>()?;
        let weight = Tensor::cat(&weights, 0)?;
        Ok(Self::new(Linear::new(weight, None)))
    }
}

impl TensorParallelRowLinear {
    pub fn load(vb: VarBuilder, comm: Arc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(1, rank, size))?;
        Ok(Self::new(Linear::new(weight, None), comm))
    }
}

pub fn linear_no_bias(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), "weight")?;
    Ok(Linear::new(weight, None))
}

pub fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}
