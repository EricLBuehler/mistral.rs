use std::path::PathBuf;

use hf_hub::Cache;
use serde::{Deserialize, Serialize};
use sysinfo::{Disks, System};

#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
use crate::MemoryUsage;
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
use candle_core::Device;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: Option<String>,
    pub logical_cores: usize,
    pub physical_cores: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub kind: String,
    pub ordinal: Option<usize>,
    pub name: Option<String>,
    pub total_memory_bytes: Option<u64>,
    pub available_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildInfo {
    pub cuda: bool,
    pub metal: bool,
    pub cudnn: bool,
    pub flash_attn: bool,
    pub flash_attn_v3: bool,
    pub accelerate: bool,
    pub mkl: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: Option<String>,
    pub kernel: Option<String>,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub devices: Vec<DeviceInfo>,
    pub build: BuildInfo,
    pub hf_cache_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DoctorStatus {
    Ok,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorCheck {
    pub name: String,
    pub status: DoctorStatus,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoctorReport {
    pub system: SystemInfo,
    pub checks: Vec<DoctorCheck>,
}

fn build_info() -> BuildInfo {
    BuildInfo {
        cuda: cfg!(all(feature = "cuda", target_family = "unix")),
        metal: cfg!(feature = "metal"),
        cudnn: cfg!(feature = "cudnn"),
        flash_attn: cfg!(feature = "flash-attn"),
        flash_attn_v3: cfg!(feature = "flash-attn-v3"),
        accelerate: cfg!(feature = "accelerate"),
        mkl: cfg!(feature = "mkl"),
    }
}

fn collect_devices(sys: &System) -> Vec<DeviceInfo> {
    let mut devices = Vec::new();

    // CPU device
    let cpu_brand = sys.cpus().first().map(|c| c.brand().to_string());
    devices.push(DeviceInfo {
        kind: "cpu".to_string(),
        ordinal: None,
        name: cpu_brand,
        total_memory_bytes: Some(sys.total_memory() as u64),
        available_memory_bytes: Some(sys.available_memory() as u64),
    });

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    {
        let mut ord = 0;
        loop {
            match Device::new_cuda(ord) {
                Ok(dev) => {
                    let total = MemoryUsage.get_total_memory(&dev).ok().map(|v| v as u64);
                    let avail = MemoryUsage.get_memory_available(&dev).ok().map(|v| v as u64);
                    devices.push(DeviceInfo {
                        kind: "cuda".to_string(),
                        ordinal: Some(ord),
                        name: None,
                        total_memory_bytes: total,
                        available_memory_bytes: avail,
                    });
                    ord += 1;
                }
                Err(_) => break,
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        let total = candle_metal_kernels::metal::Device::all().len();
        for ord in 0..total {
            if let Ok(dev) = Device::new_metal(ord) {
                let total = MemoryUsage.get_total_memory(&dev).ok().map(|v| v as u64);
                let avail = MemoryUsage.get_memory_available(&dev).ok().map(|v| v as u64);
                devices.push(DeviceInfo {
                    kind: "metal".to_string(),
                    ordinal: Some(ord),
                    name: None,
                    total_memory_bytes: total,
                    available_memory_bytes: avail,
                });
            }
        }
    }

    devices
}

pub fn collect_system_info() -> SystemInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    let cpu = CpuInfo {
        brand: sys.cpus().first().map(|c| c.brand().to_string()),
        logical_cores: sys.cpus().len(),
        physical_cores: System::physical_core_count(),
    };

    let memory = MemoryInfo {
        total_bytes: sys.total_memory() as u64,
        available_bytes: sys.available_memory() as u64,
    };

    let hf_cache = Cache::from_env();
    let hf_cache_path = hf_cache.path().to_string_lossy().to_string();

    SystemInfo {
        os: System::name(),
        kernel: System::kernel_version(),
        cpu,
        memory,
        devices: collect_devices(&sys),
        build: build_info(),
        hf_cache_path: Some(hf_cache_path),
    }
}

fn disk_usage_for(path: &PathBuf) -> Option<(u64, u64)> {
    let disks = Disks::new_with_refreshed_list();
    let mut best: Option<(usize, u64, u64)> = None;
    for disk in disks.list() {
        let mount = disk.mount_point();
        if path.starts_with(mount) {
            let len = mount.as_os_str().len();
            let avail = disk.available_space();
            let total = disk.total_space();
            if best.map(|b| len > b.0).unwrap_or(true) {
                best = Some((len, avail, total));
            }
        }
    }
    best.map(|(_, avail, total)| (avail, total))
}

pub fn run_doctor() -> DoctorReport {
    let system = collect_system_info();
    let mut checks = Vec::new();

    let hf_cache_path = system
        .hf_cache_path
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| Cache::from_env().path().clone());

    if std::fs::create_dir_all(&hf_cache_path).is_err() {
        checks.push(DoctorCheck {
            name: "hf_cache_writable".to_string(),
            status: DoctorStatus::Error,
            message: format!(
                "Cannot create or access Hugging Face cache dir at {}",
                hf_cache_path.display()
            ),
            suggestion: Some("Set HF_HOME or fix permissions.".to_string()),
        });
    } else {
        checks.push(DoctorCheck {
            name: "hf_cache_writable".to_string(),
            status: DoctorStatus::Ok,
            message: format!(
                "Hugging Face cache dir is writable: {}",
                hf_cache_path.display()
            ),
            suggestion: None,
        });
    }

    let token_present = Cache::from_env().token().is_some()
        || std::env::var("HF_TOKEN").is_ok()
        || std::env::var("HUGGINGFACE_HUB_TOKEN").is_ok();
    checks.push(DoctorCheck {
        name: "hf_token".to_string(),
        status: if token_present {
            DoctorStatus::Ok
        } else {
            DoctorStatus::Warn
        },
        message: if token_present {
            "Hugging Face token detected.".to_string()
        } else {
            "No Hugging Face token detected.".to_string()
        },
        suggestion: if token_present {
            None
        } else {
            Some("Run `huggingface-cli login` or set HF_TOKEN.".to_string())
        },
    });

    if let Some((avail, total)) = disk_usage_for(&hf_cache_path) {
        let min_free = 10_u64 * 1024 * 1024 * 1024;
        let status = if avail < min_free {
            DoctorStatus::Warn
        } else {
            DoctorStatus::Ok
        };
        checks.push(DoctorCheck {
            name: "disk_space".to_string(),
            status,
            message: format!(
                "Disk free: {:.1} GB / {:.1} GB on the volume containing the HF cache at {}.",
                avail as f64 / 1e9,
                total as f64 / 1e9,
                hf_cache_path.display()
            ),
            suggestion: if avail < min_free {
                Some("Free up disk space or move HF cache.".to_string())
            } else {
                None
            },
        });
    }

    let total_ram = system.memory.total_bytes;
    if total_ram < 8_u64 * 1024 * 1024 * 1024 {
        checks.push(DoctorCheck {
            name: "system_memory".to_string(),
            status: DoctorStatus::Warn,
            message: format!(
                "System RAM is {:.1} GB; larger models may not fit.",
                total_ram as f64 / 1e9
            ),
            suggestion: Some("Use smaller models or stronger quantization.".to_string()),
        });
    } else {
        checks.push(DoctorCheck {
            name: "system_memory".to_string(),
            status: DoctorStatus::Ok,
            message: format!("System RAM is {:.1} GB.", total_ram as f64 / 1e9),
            suggestion: None,
        });
    }

    let has_cuda = system
        .devices
        .iter()
        .any(|d| d.kind == "cuda");
    let has_metal = system
        .devices
        .iter()
        .any(|d| d.kind == "metal");

    if system.build.cuda && !has_cuda {
        checks.push(DoctorCheck {
            name: "cuda_devices".to_string(),
            status: DoctorStatus::Warn,
            message: "CUDA support is enabled but no CUDA devices were found.".to_string(),
            suggestion: Some("Check NVIDIA driver installation.".to_string()),
        });
    }
    if system.build.metal && !has_metal {
        checks.push(DoctorCheck {
            name: "metal_devices".to_string(),
            status: DoctorStatus::Warn,
            message: "Metal support is enabled but no Metal devices were found.".to_string(),
            suggestion: Some("Ensure Metal is available on this machine.".to_string()),
        });
    }

    for dev in system.devices.iter().filter(|d| d.kind != "cpu") {
        if let Some(avail) = dev.available_memory_bytes {
            if avail < 4_u64 * 1024 * 1024 * 1024 {
                let label = match dev.ordinal {
                    Some(ord) => format!("{}[{}]", dev.kind, ord),
                    None => dev.kind.clone(),
                };
                checks.push(DoctorCheck {
                    name: format!("{}_memory", label),
                    status: DoctorStatus::Warn,
                    message: format!(
                        "{} has only {:.1} GB free.",
                        label,
                        avail as f64 / 1e9
                    ),
                    suggestion: Some(
                        "Use a smaller model or a stronger quantization level.".to_string(),
                    ),
                });
            }
        }
    }

    DoctorReport { system, checks }
}
