use candle_core::{Device, Result};
use sysinfo::System;

pub struct MemoryUsage;

/// Returns the memory fraction to use for integrated CUDA GPUs.
/// Defaults to 0.75, configurable via MISTRALRS_IGPU_MEMORY_FRACTION.
#[cfg(feature = "cuda")]
fn igpu_memory_fraction() -> f64 {
    std::env::var("MISTRALRS_IGPU_MEMORY_FRACTION")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .and_then(|f| {
            if (0.0..=1.0).contains(&f) {
                Some(f)
            } else {
                None
            }
        })
        .unwrap_or(0.75)
}

impl MemoryUsage {
    /// Amount of available memory in bytes.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn get_memory_available(&self, device: &Device) -> Result<usize> {
        match device {
            Device::Cpu => {
                let mut sys = System::new_all();
                sys.refresh_cpu_all();
                Ok(usize::try_from(sys.available_memory())?)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(dev) => {
                if super::normal::is_integrated_gpu(device) {
                    // For integrated GPUs with unified memory, use system memory
                    // scaled by a configurable fraction (default 75%)
                    let mut sys = System::new_all();
                    sys.refresh_cpu_all();
                    let avail = usize::try_from(sys.available_memory())?;
                    let fraction = igpu_memory_fraction();
                    Ok((avail as f64 * fraction) as usize)
                } else {
                    use candle_core::cuda::cudarc::driver::result;
                    use candle_core::cuda_backend::WrapErr;

                    dev.cuda_stream().context().bind_to_thread().w()?;
                    let (free, _total) = result::mem_get_info().w()?;
                    Ok(free)
                }
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                candle_core::bail!("Cannot get memory available for CUDA device")
            }
            #[cfg(feature = "metal")]
            Device::Metal(dev) => {
                let max = dev.device().recommended_max_working_set_size();
                let alloc = dev.current_allocated_size();
                let avail = max.saturating_sub(alloc);

                #[allow(clippy::cast_possible_truncation)]
                Ok(avail)
            }
            #[cfg(not(feature = "metal"))]
            Device::Metal(_) => {
                candle_core::bail!("Cannot get memory available for Metal device")
            }
        }
    }

    /// Amount of total memory in bytes.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn get_total_memory(&self, device: &Device) -> Result<usize> {
        match device {
            Device::Cpu => {
                let mut sys = System::new_all();
                sys.refresh_cpu_all();
                Ok(usize::try_from(sys.total_memory())?)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(dev) => {
                if super::normal::is_integrated_gpu(device) {
                    // For integrated GPUs with unified memory, use system total memory
                    // scaled by a configurable fraction (default 75%)
                    let mut sys = System::new_all();
                    sys.refresh_cpu_all();
                    let total = usize::try_from(sys.total_memory())?;
                    let fraction = igpu_memory_fraction();
                    Ok((total as f64 * fraction) as usize)
                } else {
                    use candle_core::cuda::cudarc::driver::result;
                    use candle_core::cuda_backend::WrapErr;

                    dev.cuda_stream().context().bind_to_thread().w()?;
                    let (_free, total) = result::mem_get_info().w()?;
                    Ok(total)
                }
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
                    sys.refresh_cpu_all();
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
                    Some(0) => default_cap,
                    Some(x) => x,
                    None => default_cap,
                };

                let device_max = dev.recommended_max_working_set_size();
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
