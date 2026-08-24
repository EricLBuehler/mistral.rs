use candle_core::{Device, Result};
use sysinfo::System;
#[cfg(feature = "metal")]
use tracing::warn;

#[cfg(feature = "metal")]
const SIZE_IN_MB: usize = 1024 * 1024;

#[derive(Debug, Clone, Copy)]
pub enum DeviceMemory {
    Discrete { total: usize, free: usize },
    Unified { budget: usize, allocated: usize },
}

impl DeviceMemory {
    pub fn total(&self) -> usize {
        match *self {
            Self::Discrete { total, .. } => total,
            Self::Unified { budget, .. } => budget,
        }
    }

    pub fn available(&self) -> usize {
        match *self {
            Self::Discrete { free, .. } => free,
            Self::Unified { budget, allocated } => budget.saturating_sub(allocated),
        }
    }

    pub fn is_unified(&self) -> bool {
        matches!(self, Self::Unified { .. })
    }
}

pub struct MemoryUsage;

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMemoryPoolUsage {
    pub reserved: usize,
    pub used: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMemoryPoolSnapshot {
    pub current: CudaMemoryPoolUsage,
    pub reserved_high: usize,
    pub used_high: usize,
    pub release_threshold: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaGraphMemoryUsage {
    pub reserved: usize,
    pub used: usize,
    pub reserved_high: usize,
    pub used_high: usize,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaAllocatorSnapshot {
    pub total: usize,
    pub available: usize,
    pub async_pool: Option<CudaMemoryPoolSnapshot>,
    pub graph_pool: Option<CudaGraphMemoryUsage>,
}

#[cfg(feature = "cuda")]
impl CudaMemoryPoolUsage {
    pub fn cached(self) -> usize {
        self.reserved.saturating_sub(self.used)
    }
}

impl MemoryUsage {
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn query(&self, device: &Device) -> Result<DeviceMemory> {
        match device {
            Device::Cpu => {
                let sys = System::new_all();
                Ok(DeviceMemory::Discrete {
                    total: usize::try_from(sys.total_memory())?,
                    free: usize::try_from(sys.available_memory())?,
                })
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(dev) => {
                if super::normal::is_integrated_gpu(device) {
                    let sys = System::new_all();
                    let total_bytes = usize::try_from(sys.total_memory())?;
                    let avail_bytes = usize::try_from(sys.available_memory())?;
                    let fraction = igpu_memory_fraction();
                    let budget = (total_bytes as f64 * fraction) as usize;
                    let free = (avail_bytes as f64 * fraction) as usize;
                    Ok(DeviceMemory::Unified {
                        budget,
                        allocated: budget.saturating_sub(free),
                    })
                } else {
                    use candle_core::cuda::cudarc::driver::result;
                    use candle_core::cuda_backend::WrapErr;

                    dev.cuda_stream().context().bind_to_thread().w()?;
                    let (free, total) = result::mem_get_info().w()?;
                    Ok(DeviceMemory::Discrete { total, free })
                }
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                candle_core::bail!("Cannot query memory for CUDA device")
            }
            #[cfg(feature = "metal")]
            Device::Metal(dev) => {
                let sysctl_floor = metal_sysctl_floor_bytes()?;
                let device_max = dev.device().recommended_max_working_set_size();
                let budget = sysctl_floor.max(device_max);
                let allocated = dev.current_allocated_size();

                // recommendedMaxWorkingSetSize is dynamic and can underreport on small/pressured Apple Silicon.
                // Dividing by 2 here is a heuristic to indicate that we are now below an expected value.
                // See: https://github.com/EricLBuehler/mistral.rs/issues/2127
                if device_max < sysctl_floor / 2 {
                    warn!(
                        "Metal recommendedMaxWorkingSetSize ({} MB) is much smaller than the system-RAM floor ({} MB); currentAllocatedSize = {} MB. Using the floor.",
                        device_max / SIZE_IN_MB,
                        sysctl_floor / SIZE_IN_MB,
                        allocated / SIZE_IN_MB,
                    );
                }

                Ok(DeviceMemory::Unified { budget, allocated })
            }
            #[cfg(not(feature = "metal"))]
            Device::Metal(_) => {
                candle_core::bail!("Cannot query memory for Metal device")
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn query_cuda_memory_pool(&self, device: &Device) -> Result<Option<CudaMemoryPoolUsage>> {
        use candle_core::cuda_backend::cudarc::driver::sys;
        use candle_core::cuda_backend::WrapErr;

        let Device::Cuda(device) = device else {
            return Ok(None);
        };
        let stream = device.cuda_stream();
        stream.context().bind_to_thread().w()?;
        if !stream.context().has_async_alloc() {
            return Ok(None);
        }

        let mut pool = std::ptr::null_mut();
        cuda_result(
            unsafe { sys::cuDeviceGetMemPool(&mut pool, stream.context().cu_device()) },
            "CUDA memory pool lookup",
        )?;
        let reserved = cuda_memory_pool_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
        )?;
        let used = cuda_memory_pool_attribute(
            pool,
            sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
        )?;
        Ok(Some(CudaMemoryPoolUsage {
            reserved: usize::try_from(reserved)?,
            used: usize::try_from(used)?,
        }))
    }

    #[cfg(feature = "cuda")]
    pub fn query_cuda_allocator(&self, device: &Device) -> Result<Option<CudaAllocatorSnapshot>> {
        use candle_core::cuda::cudarc::driver::result;
        use candle_core::cuda_backend::cudarc::driver::sys;
        use candle_core::cuda_backend::WrapErr;

        let Device::Cuda(device) = device else {
            return Ok(None);
        };
        let stream = device.cuda_stream();
        stream.context().bind_to_thread().w()?;
        let (available, total) = result::mem_get_info().w()?;
        let async_pool = if stream.context().has_async_alloc() {
            let mut pool = std::ptr::null_mut();
            cuda_result(
                unsafe { sys::cuDeviceGetMemPool(&mut pool, stream.context().cu_device()) },
                "CUDA memory pool lookup",
            )?;
            let reserved = cuda_memory_pool_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            )?;
            let used = cuda_memory_pool_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            )?;
            let reserved_high = cuda_memory_pool_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            )?;
            let used_high = cuda_memory_pool_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
            )?;
            let release_threshold = cuda_memory_pool_attribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            )?;
            Some(CudaMemoryPoolSnapshot {
                current: CudaMemoryPoolUsage {
                    reserved: usize::try_from(reserved)?,
                    used: usize::try_from(used)?,
                },
                reserved_high: usize::try_from(reserved_high)?,
                used_high: usize::try_from(used_high)?,
                release_threshold: usize::try_from(release_threshold)?,
            })
        } else {
            None
        };
        let graph_pool = query_cuda_graph_memory(stream.context().cu_device())?;
        Ok(Some(CudaAllocatorSnapshot {
            total,
            available,
            async_pool,
            graph_pool,
        }))
    }

    #[cfg(feature = "cuda")]
    pub fn trim_cuda_memory_pool(&self, device: &Device, min_bytes: usize) -> Result<bool> {
        use candle_core::cuda_backend::cudarc::driver::sys;
        use candle_core::cuda_backend::WrapErr;

        let Device::Cuda(device) = device else {
            return Ok(false);
        };
        let stream = device.cuda_stream();
        stream.context().bind_to_thread().w()?;
        if !stream.context().has_async_alloc() {
            return Ok(false);
        }

        let mut pool = std::ptr::null_mut();
        cuda_result(
            unsafe { sys::cuDeviceGetMemPool(&mut pool, stream.context().cu_device()) },
            "CUDA memory pool lookup",
        )?;
        cuda_result(
            unsafe { sys::cuMemPoolTrimTo(pool, min_bytes) },
            "CUDA memory pool trim",
        )?;
        Ok(true)
    }

    #[cfg(feature = "cuda")]
    pub fn synchronize_cuda_context(&self, device: &Device) -> Result<bool> {
        let Device::Cuda(device) = device else {
            return Ok(false);
        };
        device
            .cuda_stream()
            .context()
            .synchronize()
            .map_err(candle_core::Error::wrap)?;
        Ok(true)
    }
}

#[cfg(feature = "cuda")]
fn cuda_memory_pool_attribute(
    pool: candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool,
    attribute: candle_core::cuda_backend::cudarc::driver::sys::CUmemPool_attribute,
) -> Result<u64> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    let mut value = 0u64;
    cuda_result(
        unsafe {
            sys::cuMemPoolGetAttribute(
                pool,
                attribute,
                (&mut value as *mut u64).cast::<std::ffi::c_void>(),
            )
        },
        "CUDA memory pool attribute lookup",
    )?;
    Ok(value)
}

#[cfg(feature = "cuda")]
fn query_cuda_graph_memory(
    device: candle_core::cuda_backend::cudarc::driver::sys::CUdevice,
) -> Result<Option<CudaGraphMemoryUsage>> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    let Some(used) = cuda_graph_memory_attribute(
        device,
        sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
    )?
    else {
        return Ok(None);
    };
    let Some(reserved) = cuda_graph_memory_attribute(
        device,
        sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
    )?
    else {
        return Ok(None);
    };
    let Some(used_high) = cuda_graph_memory_attribute(
        device,
        sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_HIGH,
    )?
    else {
        return Ok(None);
    };
    let Some(reserved_high) = cuda_graph_memory_attribute(
        device,
        sys::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH,
    )?
    else {
        return Ok(None);
    };
    Ok(Some(CudaGraphMemoryUsage {
        reserved,
        used,
        reserved_high,
        used_high,
    }))
}

#[cfg(feature = "cuda")]
fn cuda_graph_memory_attribute(
    device: candle_core::cuda_backend::cudarc::driver::sys::CUdevice,
    attribute: candle_core::cuda_backend::cudarc::driver::sys::CUgraphMem_attribute,
) -> Result<Option<usize>> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    let mut value = 0usize;
    let result = unsafe {
        sys::cuDeviceGetGraphMemAttribute(
            device,
            attribute,
            (&mut value as *mut usize).cast::<std::ffi::c_void>(),
        )
    };
    match result {
        sys::CUresult::CUDA_SUCCESS => Ok(Some(value)),
        sys::CUresult::CUDA_ERROR_NOT_SUPPORTED | sys::CUresult::CUDA_ERROR_INVALID_VALUE => {
            Ok(None)
        }
        _ => Err(candle_core::Error::msg(format!(
            "CUDA graph memory attribute lookup failed: {result:?}"
        ))),
    }
}

#[cfg(feature = "cuda")]
fn cuda_result(
    result: candle_core::cuda_backend::cudarc::driver::sys::CUresult,
    context: &'static str,
) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::sys;

    if result == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(candle_core::Error::msg(format!(
            "{context} failed: {result:?}"
        )))
    }
}

#[cfg(feature = "cuda")]
fn igpu_memory_fraction() -> f64 {
    std::env::var("MISTRALRS_IGPU_MEMORY_FRACTION")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&f| (0.0..=1.0).contains(&f))
        .unwrap_or(0.75)
}

#[cfg(feature = "metal")]
fn metal_sysctl_floor_bytes() -> Result<usize> {
    let sys = System::new_all();
    let system_ram_mb = usize::try_from(sys.total_memory())? / SIZE_IN_MB;

    let sysctl_mb = std::process::Command::new("sysctl")
        .arg("-n")
        .arg("iogpu.wired_limit_mb")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<usize>().ok());

    let default_cap_mb = match system_ram_mb {
        x if x <= 36 * 1024 => (system_ram_mb * 2) / 3,
        x if x > 36 * 1024 => (system_ram_mb * 3) / 4,
        x => {
            return Err(candle_core::Error::Msg(format!(
                "Invalid system ram mb value {x}."
            )))
        }
    };

    let floor_mb = match sysctl_mb {
        Some(0) | None => default_cap_mb,
        Some(x) => x,
    };
    Ok(floor_mb * SIZE_IN_MB)
}
