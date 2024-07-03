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
                Ok(sys.free_memory() as usize * KB_TO_BYTES)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => candle_core::cuda::cudarc::driver::result::mem_get_info().0,
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                candle_core::bail!("Cannot get memory available for CUDA device")
            }
            Device::Metal(_) => {
                candle_core::bail!("Cannot get memory available for Metal device")
            }
        }
    }
}
