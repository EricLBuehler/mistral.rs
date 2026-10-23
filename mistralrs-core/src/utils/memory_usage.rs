use candle_core::{Device, Result};
#[cfg(not(target_os = "macos"))]
use sysinfo::System;

#[cfg(not(target_os = "macos"))]
fn system_memory() -> Result<(usize, usize)> {
    let mut sys = System::new();
    sys.refresh_memory();
    Ok((
        usize::try_from(sys.total_memory())?,
        usize::try_from(sys.available_memory())?,
    ))
}

#[cfg(target_os = "macos")]
fn system_memory() -> Result<(usize, usize)> {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }

    let mut total = 0u64;
    let mut total_size = std::mem::size_of_val(&total);
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut total as *mut u64).cast(),
            &mut total_size,
            std::ptr::null_mut(),
            0,
        )
    };
    if result != 0 {
        return Err(candle_core::Error::wrap(std::io::Error::last_os_error()));
    }
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        candle_core::bail!("macOS memory page size query failed");
    }
    let mut stat: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    #[allow(deprecated)]
    let host = unsafe { libc::mach_host_self() };
    let result = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&mut stat as *mut libc::vm_statistics64).cast(),
            &mut count,
        )
    };
    #[allow(deprecated)]
    let release_result = unsafe { mach_port_deallocate(libc::mach_task_self_, host) };
    if result != libc::KERN_SUCCESS {
        candle_core::bail!("macOS memory statistics query failed: {result}");
    }
    if release_result != libc::KERN_SUCCESS {
        candle_core::bail!("macOS memory statistics port release failed: {release_result}");
    }
    let total = usize::try_from(total)?;
    Ok((
        total,
        macos_available_memory(&stat, usize::try_from(page_size)?, total)?,
    ))
}

#[cfg(target_os = "macos")]
fn macos_available_memory(
    stat: &libc::vm_statistics64,
    page_size: usize,
    total: usize,
) -> Result<usize> {
    // Fully reclaimable pages: https://github.com/apple-oss-distributions/xnu/blob/main/doc/vm/memorystatus.md
    // Mach free and pageable external counts both include speculative pages.
    let pages = u64::from(stat.free_count)
        .saturating_sub(u64::from(stat.speculative_count))
        .saturating_add(u64::from(stat.external_page_count))
        .saturating_add(u64::from(stat.purgeable_count));
    let bytes = pages
        .saturating_mul(u64::try_from(page_size)?)
        .min(u64::try_from(total)?);
    Ok(usize::try_from(bytes)?)
}

#[derive(Debug, Clone, Copy)]
pub enum DeviceMemory {
    Discrete {
        total: usize,
        free: usize,
    },
    Unified {
        total: usize,
        available: usize,
        allocated: usize,
        working_set_limit: usize,
    },
}

impl DeviceMemory {
    pub fn total(&self) -> usize {
        match *self {
            Self::Discrete { total, .. } => total,
            Self::Unified { total, .. } => total,
        }
    }

    pub fn available(&self) -> usize {
        match *self {
            Self::Discrete { free, .. } => free,
            Self::Unified {
                available,
                allocated,
                working_set_limit,
                ..
            } => available.min(working_set_limit.saturating_sub(allocated)),
        }
    }

    pub fn allocated(&self) -> usize {
        match *self {
            Self::Discrete { total, free } => total.saturating_sub(free),
            Self::Unified { allocated, .. } => allocated,
        }
    }

    pub fn physical_available(&self) -> usize {
        match *self {
            Self::Discrete { free, .. } => free,
            Self::Unified { available, .. } => available,
        }
    }

    pub fn working_set_limit(&self) -> usize {
        match *self {
            Self::Discrete { total, .. } => total,
            Self::Unified {
                working_set_limit, ..
            } => working_set_limit,
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
                let (total, free) = system_memory()?;
                Ok(DeviceMemory::Discrete { total, free })
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(dev) => {
                if super::normal::is_integrated_gpu(device) {
                    let (total_bytes, avail_bytes) = system_memory()?;
                    Ok(DeviceMemory::Unified {
                        total: total_bytes,
                        available: avail_bytes,
                        allocated: total_bytes.saturating_sub(avail_bytes),
                        working_set_limit: total_bytes,
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
                let (total, available) = system_memory()?;
                let recommended = dev.device().recommended_max_working_set_size();
                Ok(DeviceMemory::Unified {
                    total,
                    available,
                    allocated: dev.current_allocated_size(),
                    working_set_limit: if recommended == 0 {
                        total
                    } else {
                        recommended.min(total)
                    },
                })
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

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::{macos_available_memory, system_memory};

    const PAGE_SIZE: usize = 16 * 1024;
    const TOTAL_PAGES: u32 = 1_000;
    const INITIAL_FREE_PAGES: u32 = 600;
    const TRANSFER_PAGES: u32 = 100;
    const COMPRESSED_PAGES: u32 = 25;

    fn free_memory() -> libc::vm_statistics64 {
        let mut stat: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        stat.free_count = INITIAL_FREE_PAGES;
        stat.internal_page_count = TOTAL_PAGES - INITIAL_FREE_PAGES;
        stat.active_count = stat.internal_page_count;
        stat
    }

    fn available_pages(stat: &libc::vm_statistics64) -> usize {
        macos_available_memory(stat, PAGE_SIZE, TOTAL_PAGES as usize * PAGE_SIZE).unwrap()
            / PAGE_SIZE
    }

    #[test]
    fn macos_system_memory_query_succeeds() {
        let (total, available) = system_memory().unwrap();
        assert!(total > 0);
        assert!(available <= total);
    }

    #[test]
    fn file_faults_and_queue_promotion_preserve_available_memory() {
        let mut stat = free_memory();
        let initial = available_pages(&stat);
        stat.free_count -= TRANSFER_PAGES;
        stat.external_page_count += TRANSFER_PAGES;
        stat.inactive_count += TRANSFER_PAGES;
        assert_eq!(available_pages(&stat), initial);

        stat.inactive_count -= TRANSFER_PAGES;
        stat.active_count += TRANSFER_PAGES;
        assert_eq!(available_pages(&stat), initial);
    }

    #[test]
    fn speculative_file_pages_are_counted_once_before_and_after_activation() {
        let mut stat = free_memory();
        let initial = available_pages(&stat);
        stat.speculative_count += TRANSFER_PAGES;
        stat.external_page_count += TRANSFER_PAGES;
        assert_eq!(available_pages(&stat), initial);

        stat.free_count -= TRANSFER_PAGES;
        stat.speculative_count -= TRANSFER_PAGES;
        stat.active_count += TRANSFER_PAGES;
        assert_eq!(available_pages(&stat), initial);
    }

    #[test]
    fn anonymous_pages_require_explicit_volatile_purgeability() {
        let mut stat = free_memory();
        stat.free_count -= TRANSFER_PAGES;
        stat.internal_page_count += TRANSFER_PAGES;
        stat.active_count += TRANSFER_PAGES;
        let expected = (INITIAL_FREE_PAGES - TRANSFER_PAGES) as usize;
        assert_eq!(available_pages(&stat), expected);

        stat.active_count -= TRANSFER_PAGES;
        stat.inactive_count += TRANSFER_PAGES;
        assert_eq!(available_pages(&stat), expected);

        stat.purgeable_count += TRANSFER_PAGES;
        assert_eq!(available_pages(&stat), INITIAL_FREE_PAGES as usize);
    }

    #[test]
    fn compression_counts_only_the_pages_it_releases() {
        let mut stat = free_memory();
        stat.free_count -= TRANSFER_PAGES;
        stat.internal_page_count += TRANSFER_PAGES;
        stat.active_count += TRANSFER_PAGES;
        let before = available_pages(&stat);

        stat.internal_page_count -= TRANSFER_PAGES;
        stat.active_count -= TRANSFER_PAGES;
        stat.compressor_page_count += COMPRESSED_PAGES;
        stat.free_count += TRANSFER_PAGES - COMPRESSED_PAGES;
        assert_eq!(
            available_pages(&stat),
            before + (TRANSFER_PAGES - COMPRESSED_PAGES) as usize
        );
    }

    #[test]
    fn wiring_file_pages_removes_them_from_available_memory() {
        let mut stat = free_memory();
        stat.free_count -= TRANSFER_PAGES;
        stat.external_page_count += TRANSFER_PAGES;
        stat.active_count += TRANSFER_PAGES;
        let before = available_pages(&stat);

        stat.external_page_count -= TRANSFER_PAGES;
        stat.active_count -= TRANSFER_PAGES;
        stat.wire_count += TRANSFER_PAGES;
        assert_eq!(available_pages(&stat), before - TRANSFER_PAGES as usize);
    }
}
