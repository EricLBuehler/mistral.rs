
struct TensorParallelColumnLinear {
    linear: Linear,
}

impl TensorParallelColumnLinear {
    fn new(linear: Linear) -> Self {
        Self { linear }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }
}

struct TensorParallelRowLinear {
    linear: Linear,
    comm: Rc<Comm>,
}

struct AllReduce {
    comm: Rc<Comm>,
}

/// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
/// But for this example purposes, this will work
unsafe impl Sync for AllReduce {}
/// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
/// But for this example purposes, this will work
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
        s: &candle::CudaStorage,
        l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::cuda_backend::WrapErr;
        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let s = s.as_cuda_slice::<f16>()?;
        // let s = match l.contiguous_offsets() {
        //     None => Err(Error::Wrapped("input has to be contiguous".into()))?,
        //     Some((o1, o2)) => s.slice(o1..o2),
        // };
        let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
        self.comm.all_reduce(s, &mut dst, &ReduceOp::Sum).unwrap();
        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev);
        Ok((dst, l.shape().clone()))
    }
}

fn all_reduce_sum(x: &Tensor, comm: &Rc<Comm>) -> Result<Tensor> {
    x.apply_op1(AllReduce { comm: comm.clone() })
}

impl TensorParallelRowLinear {
    fn new(linear: Linear, comm: Rc<Comm>) -> Self {
        Self { linear, comm }
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear.forward(x)?;
        all_reduce_sum(&x, &self.comm)
    }
}

fn shard(dim: usize, rank: usize, world_size: usize) -> candle_nn::var_builder::Shard {
    candle_nn::var_builder::Shard {
        dim,
        rank,
        world_size,
    }
}

impl TensorParallelColumnLinear {
    fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(0, rank, size))?;
        Ok(Self::new(Linear::new(weight, None)))
    }

    fn load_multi(vb: VarBuilder, prefixes: &[&str], comm: Rc<Comm>) -> Result<Self> {
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
    fn load(vb: VarBuilder, comm: Rc<Comm>) -> Result<Self> {
        let rank = comm.rank();
        let size = comm.world_size();
        let weight = vb.get_with_hints((), "weight", shard(1, rank, size))?;
        Ok(Self::new(Linear::new(weight, None), comm))
    }
}