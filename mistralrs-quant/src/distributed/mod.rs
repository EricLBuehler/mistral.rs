use std::{fmt::Debug, sync::Barrier};

use candle_core::Result;
pub use ops::{Comm, Id, SumAllReduce};
pub mod layers;
pub mod socket;

pub trait BarrierLike: Debug + Send + Sync {
    fn wait(&self) -> Result<()>;
}

impl BarrierLike for Barrier {
    fn wait(&self) -> Result<()> {
        Barrier::wait(self);
        Ok(())
    }
}

pub fn get_global_tp_size_from_devices() -> Result<usize> {
    #[cfg(feature = "cuda")]
    {
        use candle_core::cuda::WrapErr;
        candle_core::cuda::cudarc::driver::result::device::get_count()
            .w()
            .map(|x| x as usize)
    }
    #[cfg(not(feature = "cuda"))]
    Ok(1)
}

pub fn use_nccl() -> bool {
    (std::env::var("MISTRALRS_NO_NCCL").is_err()
        || std::env::var("MISTRALRS_NO_NCCL").is_ok_and(|x| x != "1"))
        && (cfg!(feature = "nccl") && cfg!(feature = "cuda"))
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
mod ops {
    use std::{fmt::Debug, ops::Deref, sync::Arc};

    use candle_core::{
        backend::BackendStorage, cuda::cudarc, cuda_backend::WrapErr, CpuStorage, CustomOp1, DType,
        Device, Layout, Result, Shape, Tensor,
    };

    #[derive(Debug, Clone, Copy)]
    pub struct Id(cudarc::nccl::Id);

    impl Id {
        pub fn new() -> Self {
            let id = cudarc::nccl::Id::new().expect("Failed to create `Id`.");
            Self(id)
        }

        pub fn uninit(internal: [::core::ffi::c_char; 128usize]) -> Self {
            Self(cudarc::nccl::Id::uninit(internal))
        }

        pub fn internal(&self) -> &[::core::ffi::c_char; 128usize] {
            self.0.internal()
        }
    }

    #[derive(Debug)]
    pub struct Comm {
        comm: cudarc::nccl::Comm,
    }

    impl Comm {
        pub fn from_device(id: Id, dev: &Device, rank: usize, world_size: usize) -> Result<Self> {
            let device = dev.as_cuda_device()?.cuda_device();
            Ok(Self {
                comm: cudarc::nccl::Comm::from_rank(device, rank, world_size, id.0)
                    .map_err(|e| e.0)
                    .expect("Failed to create `Comm`, error code"),
            })
        }
    }

    /// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
    unsafe impl Sync for Comm {}
    unsafe impl Send for Comm {}

    impl Deref for Comm {
        type Target = cudarc::nccl::Comm;

        fn deref(&self) -> &Self::Target {
            &self.comm
        }
    }

    #[derive(Clone, Debug)]
    pub struct SumAllReduce {
        comm: Arc<Comm>,
    }

    impl SumAllReduce {
        pub fn new(comm: &Arc<Comm>) -> Self {
            Self { comm: comm.clone() }
        }

        pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
            // use candle_core::cuda::cudarc::driver::result;
            // unsafe { result::ctx::set_current(*self.comm.comm.device().cu_primary_ctx()) }.unwrap();
            // self.comm.barrier.wait()?;
            xs.apply_op1_no_bwd(self)
        }
    }

    impl CustomOp1 for SumAllReduce {
        fn name(&self) -> &'static str {
            "SumAllReduce"
        }

        fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("SumAllReduce is never used on cpu")
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
                        .comm
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
                        .comm
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
                        .comm
                        .all_reduce(s, &mut dst, &ReduceOp::Sum)
                        .map_err(candle_core::Error::debug)?;
                    candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
                }
                dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
            };
            Ok((dst, l.shape().clone()))
        }
    }
}

#[cfg(not(all(feature = "cuda", feature = "nccl")))]
mod ops {
    use std::sync::Arc;

    use candle_core::{Device, Result, Tensor};

    #[derive(Debug, Clone, Copy)]
    pub struct Id;

    impl Default for Id {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Id {
        pub fn new() -> Self {
            Self
        }

        pub fn uninit(_internal: [::core::ffi::c_char; 128usize]) -> Self {
            Self
        }

        pub fn internal(&self) -> &[::core::ffi::c_char; 128usize] {
            &[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ]
        }
    }

    #[derive(Debug)]
    pub struct Comm;

    impl Comm {
        pub fn from_device(
            _id: Id,
            _dev: &Device,
            _rank: usize,
            _world_size: usize,
        ) -> Result<Self> {
            Ok(Self)
        }

        pub fn rank(&self) -> usize {
            0
        }

        pub fn world_size(&self) -> usize {
            1
        }
    }

    #[derive(Clone, Debug)]
    pub struct SumAllReduce;

    impl SumAllReduce {
        pub fn new(_comm: &Arc<Comm>) -> Self {
            Self
        }

        pub fn apply(&self, xs: &Tensor) -> Result<Tensor> {
            Ok(xs.clone())
        }
    }
}
