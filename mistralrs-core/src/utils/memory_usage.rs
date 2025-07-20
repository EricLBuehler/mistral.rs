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
                use candle_core::cuda::cudarc::driver::result;
                use candle_core::cuda_backend::WrapErr;

                dev.cuda_stream().context().bind_to_thread().w()?;

                let (free, _total) = result::mem_get_info().w()?;

                Ok(free)
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
                use candle_core::cuda::cudarc::driver::result;
                use candle_core::cuda_backend::WrapErr;

                dev.cuda_stream().context().bind_to_thread().w()?;

                let (_free, total) = result::mem_get_info().w()?;

                Ok(total)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                candle_core::bail!("Cannot get total memory for CUDA device")
            }
            #[cfg(feature = "metal")]
            #[allow(clippy::cast_possible_truncation)]
            Device::Metal(dev) => {
                const SIZE_IN_MB: usize = 1024 * 1024;

                // Get system RAM in MB
                let system_ram_mb = {
                    let mut sys = System::new_all();
                    sys.refresh_cpu();
                    usize::try_from(sys.total_memory())? / SIZE_IN_MB
                };

                // Check for Metal GPU wired limit
                let metal_cap_mb = std::process::Command::new("sysctl")
                    .arg("-n")
                    .arg("iogpu.wired_limit_mb")
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .and_then(|s| s.trim().parse::<usize>().ok());

                // Apply default cap based on system RAM if not set or 0
                let default_cap = match system_ram_mb {
                    x if x <= 36 * 1024 => (system_ram_mb * 2) / 3,
                    x if x > 36 * 1024 => (system_ram_mb * 3) / 4,
                    x => {
                        return Err(candle_core::Error::Msg(format!(
                            "Invalid system ram mb value {x}."
                        )))
                    }
                };

                let metal_cap_mb = match metal_cap_mb {
                    Some(x) if x == 0 => default_cap,
                    Some(x) => x,
                    None => default_cap,
                };

                let device_max = dev.recommended_max_working_set_size() as usize;
                let metal_cap_bytes = metal_cap_mb * SIZE_IN_MB;

                Ok(device_max.min(metal_cap_bytes))
            }
            #[cfg(not(feature = "metal"))]
            Device::Metal(_) => {
                candle_core::bail!("Cannot get memory available for Metal device")
            }
        }
    }
}
