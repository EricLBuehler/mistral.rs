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
}

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
