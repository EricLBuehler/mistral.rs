use std::{fmt::Debug, fs::File, sync::Barrier};

use candle_core::Result;
pub mod layers;
pub mod socket;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct RingConfig {
    master_ip: Option<String>,
    pub master_port: u16,
    pub port: u16,
    pub right_port: u16,
    right_ip: Option<String>,
    pub rank: usize,
    pub world_size: usize,
}

impl RingConfig {
    /// Loads the ring backend config from a path at `RING_CONFIG`
    pub fn load() -> Self {
        let config_json = std::env::var("RING_CONFIG").expect("RING_CONFIG must be set");
        let config: RingConfig = serde_json::from_reader(
            &File::open(config_json).expect("Could not access Ring config JSON"),
        )
        .expect("Invalid JSON config");

        if config.master_ip.is_none() && !config.is_master_rank() {
            panic!("Invalid Ring config. Non-master ranks (rank != 0) must specify master_ip.");
        }
        config
    }

    pub fn is_master_rank(&self) -> bool {
        self.rank == 0
    }

    pub fn master_ip(&self) -> String {
        self.master_ip.clone().unwrap_or("0.0.0.0".to_string())
    }

    pub fn right_ip(&self) -> String {
        self.right_ip.clone().unwrap_or("0.0.0.0".to_string())
    }
}

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
    #[cfg(all(feature = "cuda", feature = "ring"))]
    {
        use candle_core::cuda::WrapErr;
        candle_core::cuda::cudarc::driver::result::device::get_count()
            .w()
            .map(|x| x as usize)
    }
    #[cfg(all(not(feature = "cuda"), feature = "ring"))]
    {
        let config = RingConfig::load();
        Ok(config.world_size)
    }

    #[cfg(all(feature = "cuda", feature = "nccl"))]
    {
        // In case we have manual set of TP size
        if let Ok(x) = std::env::var("MISTRALRS_MN_LOCAL_WORLD_SIZE") {
            use std::str::FromStr;
            Ok(usize::from_str(&x).expect("Not a number for MISTRALRS_MN_LOCAL_WORLD_SIZE!"))
        } else {
            use candle_core::cuda::WrapErr;
            candle_core::cuda::cudarc::driver::result::device::get_count()
                .w()
                .map(|x| x as usize)
        }
    }

    #[cfg(all(not(feature = "ring"), not(feature = "nccl")))]
    Ok(1)
}

pub fn use_nccl() -> bool {
    (std::env::var("MISTRALRS_NO_NCCL").is_err()
        || std::env::var("MISTRALRS_NO_NCCL").is_ok_and(|x| x != "1"))
        && (cfg!(feature = "nccl") && cfg!(feature = "cuda"))
}

// Unified Comm enum
#[derive(Debug)]
pub enum Comm {
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    Nccl(nccl::NcclComm),
    #[cfg(feature = "ring")]
    Ring(ring::RingComm),
    Dummy(dummy::DummyComm),
}

impl Comm {
    pub fn from_device(
        id: Id,
        dev: &candle_core::Device,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if use_nccl() {
            return Ok(Self::Nccl(nccl::NcclComm::from_device(
                id, dev, rank, world_size,
            )?));
        }

        #[cfg(feature = "ring")]
        {
            return Ok(Self::Ring(ring::RingComm::from_device(
                id, dev, rank, world_size,
            )?));
        }

        #[allow(unreachable_code)]
        Ok(Self::Dummy(dummy::DummyComm::from_device(
            id, dev, rank, world_size,
        )?))
    }

    pub fn rank(&self) -> usize {
        match self {
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            Self::Nccl(comm) => comm.rank(),
            #[cfg(feature = "ring")]
            Self::Ring(comm) => comm.rank(),
            Self::Dummy(comm) => comm.rank(),
        }
    }

    pub fn world_size(&self) -> usize {
        match self {
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            Self::Nccl(comm) => comm.world_size(),
            #[cfg(feature = "ring")]
            Self::Ring(comm) => comm.world_size(),
            Self::Dummy(comm) => comm.world_size(),
        }
    }
}

// Unified Id enum
#[derive(Debug, Clone, Copy)]
pub enum Id {
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    Nccl(cudarc::nccl::Id),
    Dummy,
}

impl Id {
    pub fn new() -> Self {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if use_nccl() {
            let id = cudarc::nccl::Id::new().expect("Failed to create `Id`.");
            return Self::Nccl(id);
        }
        Self::Dummy
    }

    pub fn uninit(_internal: [::core::ffi::c_char; 128usize]) -> Self {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if use_nccl() {
            return Self::Nccl(cudarc::nccl::Id::uninit(_internal));
        }
        Self::Dummy
    }

    pub fn internal(&self) -> &[::core::ffi::c_char; 128usize] {
        match self {
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            Self::Nccl(id) => id.internal(),
            Self::Dummy => {
                static ZEROED_ID: [::core::ffi::c_char; 128] = [0; 128];
                &ZEROED_ID
            }
        }
    }
}

impl Default for Id {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(feature = "cuda", feature = "nccl"))]
use candle_core::cuda::cudarc;

// NCCL backend implementation
#[cfg(all(feature = "cuda", feature = "nccl"))]
mod nccl {
    use candle_core::{cuda::cudarc, Device, Result};

    #[derive(Debug)]
    pub struct NcclComm {
        comm: cudarc::nccl::Comm,
    }

    impl NcclComm {
        pub fn from_device(
            id: super::Id,
            dev: &Device,
            rank: usize,
            world_size: usize,
        ) -> Result<Self> {
            if !super::use_nccl() {
                candle_core::bail!("NCCL is disabled but NCCL Comm was requested");
            }
            if !world_size.is_power_of_two() {
                candle_core::bail!(
                    "NCCL backend requires world_size to be a power of 2, got {}",
                    world_size
                );
            }
            let stream = dev.as_cuda_device()?.cuda_stream();
            let device_ordinal = stream.context().ordinal();
            if rank != device_ordinal {
                candle_core::bail!(
                    "NCCL rank {} must match device ordinal, but device ordinal is {}. \
                     Ensure GPUs are visible in the correct order (check CUDA_VISIBLE_DEVICES).",
                    rank,
                    device_ordinal
                );
            }
            let nccl_id = match id {
                super::Id::Nccl(id) => id,
                _ => candle_core::bail!("Expected NCCL Id variant for NCCL Comm initialization"),
            };
            tracing::info!(
                "Initializing NCCL communicator: rank={}, world_size={}, device={}",
                rank,
                world_size,
                device_ordinal
            );
            let comm = cudarc::nccl::Comm::from_rank(stream, rank, world_size, nccl_id)
                .map_err(|e| candle_core::Error::debug(e.0))?;
            Ok(Self { comm })
        }

        pub fn rank(&self) -> usize {
            self.comm.rank()
        }

        pub fn world_size(&self) -> usize {
            self.comm.world_size()
        }

        pub fn inner(&self) -> &cudarc::nccl::Comm {
            &self.comm
        }
    }

    /// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
    unsafe impl Sync for NcclComm {}
    unsafe impl Send for NcclComm {}
}

// Ring backend implementation
#[cfg(feature = "ring")]
mod ring {
    use super::RingConfig;
    use candle_core::{Device, Result};

    #[derive(Debug)]
    pub struct RingComm {
        config: RingConfig,
    }

    impl RingComm {
        pub fn from_device(
            _id: super::Id,
            _dev: &Device,
            _rank: usize,
            _world_size: usize,
        ) -> Result<Self> {
            let config = RingConfig::load();
            // Validate ring configuration
            if config.world_size < 2 {
                candle_core::bail!(
                    "Ring backend requires world_size >= 2, got {}",
                    config.world_size
                );
            }
            if config.rank >= config.world_size {
                candle_core::bail!(
                    "Ring backend invalid config: rank {} >= world_size {}",
                    config.rank,
                    config.world_size
                );
            }
            if !config.world_size.is_power_of_two() {
                candle_core::bail!(
                    "Ring backend requires world_size to be a power of 2, got {}",
                    config.world_size
                );
            }
            Ok(Self { config })
        }

        pub fn rank(&self) -> usize {
            self.config.rank
        }

        pub fn world_size(&self) -> usize {
            self.config.world_size
        }

        pub fn config(&self) -> &RingConfig {
            &self.config
        }
    }
}

// Dummy backend implementation
mod dummy {
    use candle_core::{Device, Result};

    #[derive(Debug)]
    pub struct DummyComm;

    impl DummyComm {
        pub fn from_device(
            _id: super::Id,
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
}

// Unified operations that work with the Comm enum
#[derive(Clone, Debug)]
pub struct SumAllReduce {
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    nccl: Option<nccl_ops::SumAllReduce>,
    #[cfg(feature = "ring")]
    ring: Option<ring_ops::SumAllReduce>,
    dummy: Option<dummy_ops::SumAllReduce>,
}

impl SumAllReduce {
    pub fn new(comm: &std::sync::Arc<Comm>) -> Self {
        match &**comm {
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            Comm::Nccl(_) => Self {
                #[cfg(all(feature = "cuda", feature = "nccl"))]
                nccl: Some(nccl_ops::SumAllReduce::new(comm)),
                #[cfg(feature = "ring")]
                ring: None,
                dummy: None,
            },
            #[cfg(feature = "ring")]
            Comm::Ring(_) => Self {
                #[cfg(all(feature = "cuda", feature = "nccl"))]
                nccl: None,
                #[cfg(feature = "ring")]
                ring: Some(ring_ops::SumAllReduce::new(comm)),
                dummy: None,
            },
            Comm::Dummy(_) => Self {
                #[cfg(all(feature = "cuda", feature = "nccl"))]
                nccl: None,
                #[cfg(feature = "ring")]
                ring: None,
                dummy: Some(dummy_ops::SumAllReduce::new(comm)),
            },
        }
    }

    pub fn sum_all_reduce(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(ref nccl) = self.nccl {
            return nccl.sum_all_reduce(xs);
        }
        #[cfg(feature = "ring")]
        if let Some(ref ring) = self.ring {
            return ring.sum_all_reduce(xs);
        }
        if let Some(ref dummy) = self.dummy {
            return dummy.sum_all_reduce(xs);
        }
        candle_core::bail!("No valid SumAllReduce implementation available")
    }
}

#[derive(Clone, Debug)]
pub struct AllGather {
    #[cfg(all(feature = "cuda", feature = "nccl"))]
    nccl: Option<nccl_ops::AllGather>,
    #[cfg(feature = "ring")]
    ring: Option<ring_ops::AllGather>,
    dummy: Option<dummy_ops::AllGather>,
}

impl AllGather {
    pub fn new(comm: &std::sync::Arc<Comm>, dim: usize) -> Self {
        match &**comm {
            #[cfg(all(feature = "cuda", feature = "nccl"))]
            Comm::Nccl(_) => Self {
                #[cfg(all(feature = "cuda", feature = "nccl"))]
                nccl: Some(nccl_ops::AllGather::new(comm, dim)),
                #[cfg(feature = "ring")]
                ring: None,
                dummy: None,
            },
            #[cfg(feature = "ring")]
            Comm::Ring(_) => Self {
                #[cfg(all(feature = "cuda", feature = "nccl"))]
                nccl: None,
                #[cfg(feature = "ring")]
                ring: Some(ring_ops::AllGather::new(comm, dim)),
                dummy: None,
            },
            Comm::Dummy(_) => Self {
                #[cfg(all(feature = "cuda", feature = "nccl"))]
                nccl: None,
                #[cfg(feature = "ring")]
                ring: None,
                dummy: Some(dummy_ops::AllGather::new(comm, dim)),
            },
        }
    }

    pub fn all_gather(&self, xs: &candle_core::Tensor) -> Result<candle_core::Tensor> {
        #[cfg(all(feature = "cuda", feature = "nccl"))]
        if let Some(ref nccl) = self.nccl {
            return nccl.all_gather(xs);
        }
        #[cfg(feature = "ring")]
        if let Some(ref ring) = self.ring {
            return ring.all_gather(xs);
        }
        if let Some(ref dummy) = self.dummy {
            return dummy.all_gather(xs);
        }
        candle_core::bail!("No valid AllGather implementation available")
    }
}

// Implementation modules for each backend
#[cfg(all(feature = "cuda", feature = "nccl"))]
mod nccl_ops {
    use std::{fmt::Debug, sync::Arc};

    use candle_core::{
        backend::BackendStorage, cuda::cudarc, CpuStorage, CustomOp1, DType, Layout, Result, Shape,
        Tensor,
    };

    #[derive(Clone, Debug)]
    pub struct SumAllReduce {
        comm: Arc<super::Comm>,
    }

    impl SumAllReduce {
        pub fn new(comm: &Arc<super::Comm>) -> Self {
            Self { comm: comm.clone() }
        }
    }

    impl SumAllReduce {
        pub fn sum_all_reduce(&self, xs: &Tensor) -> Result<Tensor> {
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
            use cudarc::nccl::ReduceOp;
            use half::{bf16, f16};

            let elem_count = l.shape().elem_count();
            let dev = s.device().clone();

            match self.comm.as_ref() {
                super::Comm::Nccl(nccl_comm) => {
                    let dst = match s.dtype() {
                        DType::BF16 => {
                            let s = s.as_cuda_slice::<bf16>()?;
                            let s = match l.contiguous_offsets() {
                                Some((0, l)) if l == s.len() => s,
                                Some(_) | None => candle_core::bail!("input has to be contiguous"),
                            };
                            if elem_count == 0 {
                                candle_core::bail!("NCCL all_reduce: elem_count must be > 0");
                            }
                            let device_ordinal = dev.cuda_stream().context().ordinal();
                            if device_ordinal != nccl_comm.rank() {
                                candle_core::bail!(
                                    "NCCL device mismatch: tensor on device {} but NCCL rank is {}. \
                                     Ensure each rank uses the correct GPU.",
                                    device_ordinal,
                                    nccl_comm.rank()
                                );
                            }
                            tracing::debug!(
                                "NCCL all_reduce (BF16): rank={}, device={}, elem_count={}",
                                nccl_comm.rank(),
                                device_ordinal,
                                elem_count
                            );
                            let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }?;
                            nccl_comm
                                .inner()
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
                            if elem_count == 0 {
                                candle_core::bail!("NCCL all_reduce: elem_count must be > 0");
                            }
                            let device_ordinal = dev.cuda_stream().context().ordinal();
                            if device_ordinal != nccl_comm.rank() {
                                candle_core::bail!(
                                    "NCCL device mismatch: tensor on device {} but NCCL rank is {}. \
                                     Ensure each rank uses the correct GPU.",
                                    device_ordinal,
                                    nccl_comm.rank()
                                );
                            }
                            tracing::debug!(
                                "NCCL all_reduce (F16): rank={}, device={}, elem_count={}",
                                nccl_comm.rank(),
                                device_ordinal,
                                elem_count
                            );
                            let mut dst = unsafe { dev.alloc::<f16>(elem_count) }?;
                            nccl_comm
                                .inner()
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
                            if elem_count == 0 {
                                candle_core::bail!("NCCL all_reduce: elem_count must be > 0");
                            }
                            let device_ordinal = dev.cuda_stream().context().ordinal();
                            if device_ordinal != nccl_comm.rank() {
                                candle_core::bail!(
                                    "NCCL device mismatch: tensor on device {} but NCCL rank is {}. \
                                     Ensure each rank uses the correct GPU.",
                                    device_ordinal,
                                    nccl_comm.rank()
                                );
                            }
                            tracing::debug!(
                                "NCCL all_reduce (F32): rank={}, device={}, elem_count={}",
                                nccl_comm.rank(),
                                device_ordinal,
                                elem_count
                            );
                            let mut dst = unsafe { dev.alloc::<f32>(elem_count) }?;
                            nccl_comm
                                .inner()
                                .all_reduce(s, &mut dst, &ReduceOp::Sum)
                                .map_err(candle_core::Error::debug)?;
                            candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
                        }
                        dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
                    };
                    Ok((dst, l.shape().clone()))
                }
                _ => candle_core::bail!("SumAllReduce requires NCCL backend"),
            }
        }
    }

    #[derive(Clone, Debug)]
    pub struct AllGather {
        comm: Arc<super::Comm>,
        dim: usize,
    }

    impl AllGather {
        pub fn new(comm: &Arc<super::Comm>, dim: usize) -> Self {
            Self {
                comm: comm.clone(),
                dim,
            }
        }
    }

    impl AllGather {
        pub fn all_gather(&self, xs: &Tensor) -> Result<Tensor> {
            xs.apply_op1_no_bwd(self)
        }
    }

    impl CustomOp1 for AllGather {
        fn name(&self) -> &'static str {
            "AllGather"
        }

        fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("AllGather is never used on cpu")
        }

        fn cuda_fwd(
            &self,
            s: &candle_core::CudaStorage,
            l: &Layout,
        ) -> Result<(candle_core::CudaStorage, Shape)> {
            use half::{bf16, f16};

            let mut out_shape = l.shape().dims().to_vec();
            out_shape[self.dim] = out_shape[self.dim] * self.comm.world_size();
            let out_shape = Shape::from(out_shape);

            let elem_count = out_shape.elem_count();
            let dev = s.device().clone();

            match self.comm.as_ref() {
                super::Comm::Nccl(nccl_comm) => {
                    let dst = match s.dtype() {
                        DType::BF16 => {
                            let s = s.as_cuda_slice::<bf16>()?;
                            let s = match l.contiguous_offsets() {
                                Some((0, l)) if l == s.len() => s,
                                Some(_) | None => candle_core::bail!("input has to be contiguous"),
                            };
                            if elem_count == 0 {
                                candle_core::bail!("NCCL all_gather: elem_count must be > 0");
                            }
                            let device_ordinal = dev.cuda_stream().context().ordinal();
                            if device_ordinal != nccl_comm.rank() {
                                candle_core::bail!(
                                    "NCCL device mismatch: tensor on device {} but NCCL rank is {}. \
                                     Ensure each rank uses the correct GPU.",
                                    device_ordinal,
                                    nccl_comm.rank()
                                );
                            }
                            tracing::debug!(
                                "NCCL all_gather (BF16): rank={}, device={}, elem_count={}",
                                nccl_comm.rank(),
                                device_ordinal,
                                elem_count
                            );
                            let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }?;
                            nccl_comm
                                .inner()
                                .all_gather(s, &mut dst)
                                .map_err(candle_core::Error::debug)?;
                            candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
                        }
                        DType::F16 => {
                            let s = s.as_cuda_slice::<f16>()?;
                            let s = match l.contiguous_offsets() {
                                Some((0, l)) if l == s.len() => s,
                                Some(_) | None => candle_core::bail!("input has to be contiguous"),
                            };
                            if elem_count == 0 {
                                candle_core::bail!("NCCL all_gather: elem_count must be > 0");
                            }
                            let device_ordinal = dev.cuda_stream().context().ordinal();
                            if device_ordinal != nccl_comm.rank() {
                                candle_core::bail!(
                                    "NCCL device mismatch: tensor on device {} but NCCL rank is {}. \
                                     Ensure each rank uses the correct GPU.",
                                    device_ordinal,
                                    nccl_comm.rank()
                                );
                            }
                            tracing::debug!(
                                "NCCL all_gather (F16): rank={}, device={}, elem_count={}",
                                nccl_comm.rank(),
                                device_ordinal,
                                elem_count
                            );
                            let mut dst = unsafe { dev.alloc::<f16>(elem_count) }?;
                            nccl_comm
                                .inner()
                                .all_gather(s, &mut dst)
                                .map_err(candle_core::Error::debug)?;
                            candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
                        }
                        DType::F32 => {
                            let s = s.as_cuda_slice::<f32>()?;
                            let s = match l.contiguous_offsets() {
                                Some((0, l)) if l == s.len() => s,
                                Some(_) | None => candle_core::bail!("input has to be contiguous"),
                            };
                            if elem_count == 0 {
                                candle_core::bail!("NCCL all_gather: elem_count must be > 0");
                            }
                            let device_ordinal = dev.cuda_stream().context().ordinal();
                            if device_ordinal != nccl_comm.rank() {
                                candle_core::bail!(
                                    "NCCL device mismatch: tensor on device {} but NCCL rank is {}. \
                                     Ensure each rank uses the correct GPU.",
                                    device_ordinal,
                                    nccl_comm.rank()
                                );
                            }
                            tracing::debug!(
                                "NCCL all_gather (F32): rank={}, device={}, elem_count={}",
                                nccl_comm.rank(),
                                device_ordinal,
                                elem_count
                            );
                            let mut dst = unsafe { dev.alloc::<f32>(elem_count) }?;
                            nccl_comm
                                .inner()
                                .all_gather(s, &mut dst)
                                .map_err(candle_core::Error::debug)?;
                            candle_core::CudaStorage::wrap_cuda_slice(dst, dev)
                        }
                        dtype => candle_core::bail!("unsupported dtype {dtype:?}"),
                    };
                    Ok((dst, out_shape))
                }
                _ => candle_core::bail!("AllGather requires NCCL backend"),
            }
        }
    }
}

// Ring operations
#[cfg(feature = "ring")]
mod ring_ops {
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex, OnceLock},
        time::{Duration, Instant},
    };

    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};

    // Friendly aliases to tame type complexity.
    type SharedTcpStream = Arc<Mutex<TcpStream>>;
    type LeftRight = (SharedTcpStream, SharedTcpStream);

    use candle_core::{
        backend::BackendStorage, CpuStorage, Device, Result, Storage, Tensor, WithDType,
    };

    use super::RingConfig;

    // Lazily–initialized pair of TCP streams shared by every ring‑based collective op
    static LEFT_RIGHT_STREAMS: OnceLock<LeftRight> = OnceLock::new();

    fn get_ring_streams(config: &RingConfig) -> LeftRight {
        LEFT_RIGHT_STREAMS
            .get_or_init(|| {
                let cur_port = config.port;

                let right_ip = config.right_ip();
                let right_port = config.right_port;

                let left_listener =
                    TcpListener::bind(format!("0.0.0.0:{cur_port}")).expect("bind left");

                let start = Instant::now();
                // Connect to the right neighbor using the provided IP
                let right = loop {
                    match TcpStream::connect(format!("{}:{}", right_ip, right_port)) {
                        Ok(s) => break s,
                        Err(_) if start.elapsed() > Duration::from_secs(10) => {
                            panic!("Failed to connect to right node due to 10-second timeout");
                        }
                        Err(_) => continue,
                    }
                };

                // Accept connection from the left neighbour
                let (left, _) = left_listener.accept().expect("accept left neighbour");

                left.set_nodelay(true).unwrap();
                left.set_nonblocking(false).unwrap();
                right.set_nodelay(true).unwrap();
                right.set_nonblocking(false).unwrap();

                (Arc::new(Mutex::new(left)), Arc::new(Mutex::new(right)))
            })
            .clone()
    }

    #[derive(Clone, Debug)]
    pub struct SumAllReduce {
        left: SharedTcpStream,
        right: SharedTcpStream,
        buffers: Arc<Mutex<HashMap<usize, Vec<u8>>>>,
    }

    impl SumAllReduce {
        pub fn new(comm: &Arc<super::Comm>) -> Self {
            match &**comm {
                super::Comm::Ring(ring_comm) => {
                    let (left, right) = get_ring_streams(ring_comm.config());
                    Self {
                        left,
                        right,
                        buffers: Arc::new(Mutex::new(HashMap::new())),
                    }
                }
                _ => panic!("SumAllReduce requires Ring backend"),
            }
        }

        fn run<T: WithDType + Copy>(
            &self,
            x: &[T],
            dims: &[usize],
            device: &Device,
        ) -> Result<Tensor> {
            let nbytes = x.len() * std::mem::size_of_val(x);

            // --- ping‑pong to overlap latency ---------------------------------------
            // Clone the Arc references
            let right = self.right.clone();
            let left = self.left.clone();

            // View the local slice as bytes that can be written on the wire.
            let data_bytes = unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, nbytes) };

            // Re‑use (or allocate) a receive buffer of identical size.
            let mut buffers_guard = self.buffers.lock().map_err(|e| {
                candle_core::Error::msg(format!("Failed to lock buffers mutex: {:?}", e))
            })?;
            let recv_buf = buffers_guard
                .entry(nbytes)
                .or_insert_with(|| vec![0u8; nbytes]);

            // Lock both sockets once to avoid per-call mutex overhead.
            let mut right_guard = right.lock().map_err(|e| {
                candle_core::Error::msg(format!("Failed to lock right stream mutex: {:?}", e))
            })?;
            let mut left_guard = left.lock().map_err(|e| {
                candle_core::Error::msg(format!("Failed to lock left stream mutex: {:?}", e))
            })?;

            // For the typical tensor size we see (~ 6 KiB) a single
            // write/read pair is faster than chunking because the extra
            // system‑call and loop overhead dominates.  Only fall back to the
            // chunked "ping‑pong" pipeline for larger transfers.
            if nbytes <= 8 * 1024 {
                // --- fast path: one shot ------------------------------------
                right_guard
                    .write_all(data_bytes)
                    .map_err(|e| candle_core::Error::msg(format!("write error: {:?}", e)))?;

                left_guard
                    .read_exact(recv_buf)
                    .map_err(|e| candle_core::Error::msg(format!("read error: {:?}", e)))?;
            } else {
                // --- slow path: chunked ping‑pong ---------------------------
                const CHUNK_SIZE: usize = 64 * 1024; // 64 KiB
                let mut offset = 0;

                while offset < nbytes {
                    let len = std::cmp::min(CHUNK_SIZE, nbytes - offset);

                    // send this chunk to the right neighbour
                    right_guard
                        .write_all(&data_bytes[offset..offset + len])
                        .map_err(|e| candle_core::Error::msg(format!("write error: {:?}", e)))?;

                    // receive the matching chunk from the left neighbour
                    left_guard
                        .read_exact(&mut recv_buf[offset..offset + len])
                        .map_err(|e| candle_core::Error::msg(format!("read error: {:?}", e)))?;

                    offset += len;
                }
            }

            drop(left_guard);
            drop(right_guard);

            // -------------------------------------------------------------------------
            // Interpret the received bytes as a slice of T and add element‑wise into x
            let received: &[T] =
                unsafe { std::slice::from_raw_parts(recv_buf.as_ptr() as *const T, x.len()) };

            Tensor::from_slice(received, dims, device)
        }

        pub fn sum_all_reduce(&self, xs: &Tensor) -> Result<Tensor> {
            let storage = xs.storage_and_layout().0;
            let cpu_storage = match &*storage {
                Storage::Cpu(storage) => storage,
                Storage::Cuda(storage) => &storage.to_cpu_storage()?,
                Storage::Metal(storage) => &storage.to_cpu_storage()?,
            };

            let delta = match cpu_storage {
                CpuStorage::BF16(x) => self.run(x.as_slice(), xs.dims(), xs.device())?,
                CpuStorage::F32(x) => self.run(x.as_slice(), xs.dims(), xs.device())?,
                CpuStorage::F16(x) => self.run(x.as_slice(), xs.dims(), xs.device())?,
                _ => candle_core::bail!("Unsupported dtype for ring backend"),
            };

            xs + delta
        }
    }

    #[derive(Clone, Debug)]
    pub struct AllGather {
        left: SharedTcpStream,
        right: SharedTcpStream,
        buffers: Arc<Mutex<HashMap<usize, Vec<u8>>>>,
        dim: usize,
        world_size: usize,
        rank: usize,
    }

    impl AllGather {
        pub fn new(comm: &Arc<super::Comm>, dim: usize) -> Self {
            match &**comm {
                super::Comm::Ring(ring_comm) => {
                    let (left, right) = get_ring_streams(ring_comm.config());
                    Self {
                        left,
                        right,
                        buffers: Arc::new(Mutex::new(HashMap::new())),
                        dim,
                        world_size: ring_comm.world_size(),
                        rank: ring_comm.rank(),
                    }
                }
                _ => panic!("AllGather requires Ring backend"),
            }
        }

        fn run<T: WithDType + Copy + Default>(
            &self,
            x: &[T],
            dims: &[usize],
            device: &Device,
        ) -> Result<Tensor> {
            // Validate gather dimension
            if self.dim >= dims.len() {
                candle_core::bail!(
                    "AllGather: invalid dimension {} for tensor of rank {}",
                    self.dim,
                    dims.len()
                );
            }
            let elem_cnt = x.len();
            let nbytes = elem_cnt * std::mem::size_of_val(x);

            // Prepare output buffer that will hold slices from every rank.
            let mut out: Vec<T> = vec![T::default(); elem_cnt * self.world_size];

            // Copy this rank's slice into its final slot.
            let start = self.rank * elem_cnt;
            out[start..start + elem_cnt].copy_from_slice(x);

            let right = self.right.clone();
            let left = self.left.clone();
            let mut send_piece: &[T] = x;

            for step in 0..(self.world_size - 1) {
                // ---------- send to the right ----------
                let bytes =
                    unsafe { std::slice::from_raw_parts(send_piece.as_ptr() as *const u8, nbytes) };
                {
                    let mut rg = right.lock().map_err(|e| {
                        candle_core::Error::msg(format!(
                            "Failed to lock right stream mutex: {:?}",
                            e
                        ))
                    })?;
                    rg.write_all(bytes)
                        .map_err(|e| candle_core::Error::msg(format!("write error: {:?}", e)))?;
                }

                // ---------- receive from the left ----------
                let mut bg = self.buffers.lock().map_err(|e| {
                    candle_core::Error::msg(format!("Failed to lock buffers mutex: {:?}", e))
                })?;
                let buf = bg.entry(nbytes).or_insert_with(|| vec![0u8; nbytes]);
                {
                    let mut lg = left.lock().map_err(|e| {
                        candle_core::Error::msg(format!(
                            "Failed to lock left stream mutex: {:?}",
                            e
                        ))
                    })?;
                    lg.read_exact(buf)
                        .map_err(|e| candle_core::Error::msg(format!("read error: {:?}", e)))?;
                }
                let recv_piece: &[T] =
                    unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const T, elem_cnt) };

                // Determine which global rank the received slice came from.
                let src_rank = (self.rank + self.world_size - step - 1) % self.world_size;
                let dst = src_rank * elem_cnt;
                out[dst..dst + elem_cnt].copy_from_slice(recv_piece);

                // Forward that slice in the next iteration.
                send_piece = recv_piece;
            }

            let mut out_dims = dims.to_vec();
            out_dims[self.dim] *= self.world_size;
            Tensor::from_slice(&out, out_dims, device)
        }

        pub fn all_gather(&self, xs: &Tensor) -> Result<Tensor> {
            let storage = xs.storage_and_layout().0;
            let cpu_storage = match &*storage {
                Storage::Cpu(s) => s,
                Storage::Cuda(s) => &s.to_cpu_storage()?,
                Storage::Metal(s) => &s.to_cpu_storage()?,
            };

            match cpu_storage {
                CpuStorage::BF16(x) => self.run(x.as_slice(), xs.dims(), xs.device()),
                CpuStorage::F32(x) => self.run(x.as_slice(), xs.dims(), xs.device()),
                CpuStorage::F16(x) => self.run(x.as_slice(), xs.dims(), xs.device()),
                _ => candle_core::bail!("Unsupported dtype for ring backend"),
            }
        }
    }
}

// Dummy operations
mod dummy_ops {
    use candle_core::{Result, Tensor};
    use std::sync::Arc;

    #[derive(Clone, Debug)]
    pub struct SumAllReduce;

    impl SumAllReduce {
        pub fn new(_comm: &Arc<super::Comm>) -> Self {
            Self
        }

        pub fn sum_all_reduce(&self, xs: &Tensor) -> Result<Tensor> {
            Ok(xs.clone())
        }
    }

    #[derive(Clone, Debug)]
    pub struct AllGather;

    impl AllGather {
        pub fn new(_comm: &Arc<super::Comm>, _dim: usize) -> Self {
            Self
        }

        pub fn all_gather(&self, xs: &Tensor) -> Result<Tensor> {
            Ok(xs.clone())
        }
    }
}
