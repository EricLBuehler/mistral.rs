use candle_core::{Device, Result};
use sysinfo::System;

const KB_TO_BYTES: usize = 1024;

pub struct MemoryUsage;

impl MemoryUsage {
    pub fn get_memory_available(&self, device: &Device) -> Result<usize> {
        match device {
            Device::Cpu => {
                let mut sys = System::new_all();
                sys.refresh_cpu();
                Ok(usize::try_from(sys.free_memory())? * KB_TO_BYTES)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                use candle_core::cuda_backend::WrapErr;
                Ok(candle_core::cuda::cudarc::driver::result::mem_get_info()
                    .w()?
                    .0)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                candle_core::bail!("Cannot get memory available for CUDA device")
            }
            Device::Metal(_) => {
                candle_core::bail!("Cannot get memory available for Metal device")
            }
        }
    }

    pub fn get_total_memory(&self, device: &Device) -> Result<usize> {
        match device {
            Device::Cpu => {
                let mut sys = System::new_all();
                sys.refresh_cpu();
                Ok(usize::try_from(sys.total_memory())? * KB_TO_BYTES)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                use candle_core::cuda_backend::WrapErr;
                Ok(candle_core::cuda::cudarc::driver::result::mem_get_info()
                    .w()?
                    .1)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                candle_core::bail!("Cannot get total memory for CUDA device")
            }
            Device::Metal(_) => {
                candle_core::bail!("Cannot get total memory for Metal device")
            }
        }
    }
}
