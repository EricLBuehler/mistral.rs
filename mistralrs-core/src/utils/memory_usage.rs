use candle_core::{Device, Result};
use sysinfo::System;

pub struct MemoryUsage;

impl MemoryUsage {
    /// Amount of available memory in bytes.
    pub fn get_memory_available(&self, device: &Device) -> Result<usize> {
        match device {
            Device::Cpu => {
                let mut sys = System::new_all();
                sys.refresh_cpu();
                Ok(usize::try_from(sys.available_memory())?)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(dev) => {
                use candle_core::cuda::cudarc;
                use candle_core::cuda_backend::WrapErr;
                use candle_core::{backend::BackendDevice, DeviceLocation};

                let DeviceLocation::Cuda { gpu_id } = dev.location() else {
                    candle_core::bail!("device and location do match")
                };

                let original_ctx = dev.cu_primary_ctx();

                let avail_mem = {
                    #[allow(clippy::cast_possible_truncation)]
                    let cu_device = cudarc::driver::result::device::get(gpu_id as i32).w()?;

                    // primary context initialization, can fail with OOM
                    let cu_primary_ctx =
                        unsafe { cudarc::driver::result::primary_ctx::retain(cu_device) }.w()?;

                    unsafe { cudarc::driver::result::ctx::set_current(cu_primary_ctx) }.unwrap();

                    let res = cudarc::driver::result::mem_get_info().w()?.0;

                    unsafe { cudarc::driver::result::primary_ctx::release(cu_device) }.unwrap();

                    res
                };

                unsafe { cudarc::driver::result::ctx::set_current(*original_ctx) }.unwrap();

                Ok(avail_mem)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                candle_core::bail!("Cannot get memory available for CUDA device")
            }
            #[cfg(feature = "metal")]
            Device::Metal(dev) => {
                let max = dev.recommended_max_working_set_size();
                let alloc = dev.current_allocated_size();
                let avail = max.saturating_sub(alloc);

                #[allow(clippy::cast_possible_truncation)]
                Ok(avail as usize)
            }
            #[cfg(not(feature = "metal"))]
            Device::Metal(_) => {
                candle_core::bail!("Cannot get memory available for Metal device")
            }
        }
    }

    /// Amount of total memory in bytes.
    pub fn get_total_memory(&self, device: &Device) -> Result<usize> {
        match device {
            Device::Cpu => {
                let mut sys = System::new_all();
                sys.refresh_cpu();
                Ok(usize::try_from(sys.total_memory())?)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(dev) => {
                use candle_core::cuda::cudarc;
                use candle_core::cuda_backend::WrapErr;
                use candle_core::{backend::BackendDevice, DeviceLocation};

                let DeviceLocation::Cuda { gpu_id } = dev.location() else {
                    candle_core::bail!("device and location do match")
                };

                let original_ctx = dev.cu_primary_ctx();

                let total_mem = {
                    #[allow(clippy::cast_possible_truncation)]
                    let cu_device = cudarc::driver::result::device::get(gpu_id as i32).w()?;

                    // primary context initialization, can fail with OOM
                    let cu_primary_ctx =
                        unsafe { cudarc::driver::result::primary_ctx::retain(cu_device) }.w()?;

                    unsafe { cudarc::driver::result::ctx::set_current(cu_primary_ctx) }.unwrap();

                    let res = cudarc::driver::result::mem_get_info().w()?.1;

                    unsafe { cudarc::driver::result::primary_ctx::release(cu_device) }.unwrap();

                    res
                };

                unsafe { cudarc::driver::result::ctx::set_current(*original_ctx) }.unwrap();

                Ok(total_mem)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                candle_core::bail!("Cannot get total memory for CUDA device")
            }
            #[cfg(feature = "metal")]
            #[allow(clippy::cast_possible_truncation)]
            Device::Metal(dev) => Ok(dev.recommended_max_working_set_size() as usize),
            #[cfg(not(feature = "metal"))]
            Device::Metal(_) => {
                candle_core::bail!("Cannot get memory available for Metal device")
            }
        }
    }
}
