use std::path::{Path, PathBuf};
use std::time::Instant;

use hf_hub::{api::sync::ApiBuilder, Cache};
use serde::{Deserialize, Serialize};
use sysinfo::{Disks, System};

#[cfg(any(feature = "cuda", feature = "metal"))]
use crate::MemoryUsage;
#[cfg(any(feature = "cuda", feature = "metal"))]
use candle_core::Device;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: Option<String>,
    pub logical_cores: usize,
    pub physical_cores: Option<usize>,
    pub avx: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub fma: bool,
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
    /// CUDA compute capability (major, minor) - None for non-CUDA devices
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_capability: Option<(u32, u32)>,
    /// Whether this GPU supports Flash Attention v2 (compute capability >= 8.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_attn_compatible: Option<bool>,
    /// Whether this GPU supports Flash Attention v3 (compute capability == 9.0, Hopper only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flash_attn_v3_compatible: Option<bool>,
    /// Whether this device uses unified memory (GPU and CPU share the same physical RAM)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unified_memory: Option<bool>,
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
    pub git_revision: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfConnectivityInfo {
    /// Whether HuggingFace is reachable
    pub reachable: bool,
    /// Latency in milliseconds (if reachable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
    /// Whether the token is valid for gated models
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_valid_for_gated: Option<bool>,
    /// Error message if not reachable
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
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
        cuda: cfg!(feature = "cuda"),
        metal: cfg!(feature = "metal"),
        cudnn: cfg!(feature = "cudnn"),
        flash_attn: cfg!(feature = "flash-attn"),
        flash_attn_v3: cfg!(feature = "flash-attn-v3"),
        accelerate: cfg!(feature = "accelerate"),
        mkl: cfg!(feature = "mkl"),
        git_revision: crate::MISTRALRS_GIT_REVISION.to_string(),
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
        total_memory_bytes: Some(sys.total_memory()),
        available_memory_bytes: Some(sys.available_memory()),
        compute_capability: None,
        flash_attn_compatible: None,
        flash_attn_v3_compatible: None,
        unified_memory: None,
    });

    #[cfg(feature = "cuda")]
    {
        let mut ord = 0;
        loop {
            match Device::new_cuda(ord) {
                Ok(dev) => {
                    let total = MemoryUsage.get_total_memory(&dev).ok().map(|v| v as u64);
                    let avail = MemoryUsage
                        .get_memory_available(&dev)
                        .ok()
                        .map(|v| v as u64);

                    // Get compute capability
                    let compute_cap = get_cuda_compute_capability(ord);
                    let flash_attn_v2_ok = compute_cap.map(|(major, _minor)| {
                        // Flash Attention v2 requires compute capability >= 8.0 (Ampere+)
                        major >= 8
                    });
                    let flash_attn_v3_ok = compute_cap.map(|(major, minor)| {
                        // Flash Attention v3 requires compute capability == 9.0 (Hopper only)
                        major == 9 && minor == 0
                    });

                    devices.push(DeviceInfo {
                        kind: "cuda".to_string(),
                        ordinal: Some(ord),
                        name: None,
                        total_memory_bytes: total,
                        available_memory_bytes: avail,
                        compute_capability: compute_cap,
                        flash_attn_compatible: flash_attn_v2_ok,
                        flash_attn_v3_compatible: flash_attn_v3_ok,
                        unified_memory: Some(crate::utils::normal::is_integrated_gpu(&dev)),
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
                let avail = MemoryUsage
                    .get_memory_available(&dev)
                    .ok()
                    .map(|v| v as u64);
                devices.push(DeviceInfo {
                    kind: "metal".to_string(),
                    ordinal: Some(ord),
                    name: None,
                    total_memory_bytes: total,
                    available_memory_bytes: avail,
                    compute_capability: None,
                    flash_attn_compatible: Some(true), // Metal always supports flash attention
                    flash_attn_v3_compatible: None,    // Flash Attn v3 is CUDA Hopper only
                    unified_memory: Some(true),        // Apple Silicon always uses unified memory
                });
            }
        }
    }

    devices
}

/// Get CUDA compute capability for a device ordinal
#[cfg(feature = "cuda")]
fn get_cuda_compute_capability(ordinal: usize) -> Option<(u32, u32)> {
    // Use nvidia-smi to query compute capability
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=compute_cap",
            "--format=csv,noheader",
            &format!("-i={ordinal}"),
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let cap = stdout.trim();

    // Parse "8.9" format
    let parts: Vec<&str> = cap.split('.').collect();
    if parts.len() == 2 {
        let major = parts[0].parse().ok()?;
        let minor = parts[1].parse().ok()?;
        Some((major, minor))
    } else {
        None
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
fn get_cuda_compute_capability(_ordinal: usize) -> Option<(u32, u32)> {
    None
}

/// Detect CPU extensions (AVX, AVX2, AVX-512, FMA)
fn detect_cpu_extensions() -> (bool, bool, bool, bool) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let avx = std::arch::is_x86_feature_detected!("avx");
        let avx2 = std::arch::is_x86_feature_detected!("avx2");
        let avx512 = std::arch::is_x86_feature_detected!("avx512f");
        let fma = std::arch::is_x86_feature_detected!("fma");
        (avx, avx2, avx512, fma)
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        (false, false, false, false)
    }
}

pub fn collect_system_info() -> SystemInfo {
    let mut sys = System::new_all();
    sys.refresh_all();

    let (avx, avx2, avx512, fma) = detect_cpu_extensions();

    let cpu = CpuInfo {
        brand: sys.cpus().first().map(|c| c.brand().to_string()),
        logical_cores: sys.cpus().len(),
        physical_cores: System::physical_core_count(),
        avx,
        avx2,
        avx512,
        fma,
    };

    let memory = MemoryInfo {
        total_bytes: sys.total_memory(),
        available_bytes: sys.available_memory(),
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

/// Check HuggingFace connectivity and token validity by accessing a gated model
#[allow(clippy::cast_possible_truncation)]
pub fn check_hf_gated_access() -> HfConnectivityInfo {
    let start = Instant::now();

    // Try to access a gated model (google/gemma-3-4b-it)
    let api_result = ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .and_then(|api| api.model("google/gemma-3-4b-it".to_string()).info());

    let latency_ms = start.elapsed().as_millis() as u64;

    match api_result {
        Ok(_) => HfConnectivityInfo {
            reachable: true,
            latency_ms: Some(latency_ms),
            token_valid_for_gated: Some(true),
            error: None,
        },
        Err(e) => {
            let error_str = e.to_string();
            // Check if it's an auth error vs network error
            let is_auth_error = error_str.contains("401")
                || error_str.contains("403")
                || error_str.contains("unauthorized")
                || error_str.contains("Unauthorized")
                || error_str.contains("Access denied")
                || error_str.contains("gated");

            if is_auth_error {
                // Network works, but token is invalid/missing
                HfConnectivityInfo {
                    reachable: true,
                    latency_ms: Some(latency_ms),
                    token_valid_for_gated: Some(false),
                    error: Some("Token invalid or missing for gated models".to_string()),
                }
            } else {
                // Network/other error
                HfConnectivityInfo {
                    reachable: false,
                    latency_ms: None,
                    token_valid_for_gated: None,
                    error: Some(error_str),
                }
            }
        }
    }
}

fn disk_usage_for(path: &Path) -> Option<(u64, u64)> {
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

    // CPU extensions check (ARM-aware)
    {
        let is_arm = cfg!(any(target_arch = "aarch64", target_arch = "arm"));

        if is_arm {
            // ARM CPUs use NEON, not AVX - no warning needed
            checks.push(DoctorCheck {
                name: "cpu_extensions".to_string(),
                status: DoctorStatus::Ok,
                message: "CPU: ARM architecture (uses NEON)".to_string(),
                suggestion: None,
            });
        } else {
            // x86/x86_64 - check for AVX extensions
            let mut extensions = Vec::new();
            if system.cpu.avx {
                extensions.push("AVX");
            }
            if system.cpu.avx2 {
                extensions.push("AVX2");
            }
            if system.cpu.fma {
                extensions.push("FMA");
            }
            if system.cpu.avx512 {
                extensions.push("AVX-512");
            }

            let has_avx2 = system.cpu.avx2;
            let ext_str = if extensions.is_empty() {
                "none detected".to_string()
            } else {
                extensions.join(", ")
            };

            checks.push(DoctorCheck {
                name: "cpu_extensions".to_string(),
                status: if has_avx2 {
                    DoctorStatus::Ok
                } else {
                    DoctorStatus::Warn
                },
                message: format!("CPU extensions: {ext_str}"),
                suggestion: if !has_avx2 {
                    Some("AVX2 is recommended for optimal GGML performance on x86.".to_string())
                } else {
                    None
                },
            });
        }
    }

    // Binary vs hardware mismatch check
    {
        let has_cuda_device = system.devices.iter().any(|d| d.kind == "cuda");
        let has_metal_device = system.devices.iter().any(|d| d.kind == "metal");

        if has_cuda_device && !system.build.cuda {
            checks.push(DoctorCheck {
                name: "binary_hardware_match".to_string(),
                status: DoctorStatus::Error,
                message: "NVIDIA GPU detected but binary compiled without CUDA support."
                    .to_string(),
                suggestion: Some("Reinstall with CUDA: cargo install --features cuda".to_string()),
            });
        } else if has_metal_device && !system.build.metal {
            checks.push(DoctorCheck {
                name: "binary_hardware_match".to_string(),
                status: DoctorStatus::Error,
                message: "Apple GPU detected but binary compiled without Metal support."
                    .to_string(),
                suggestion: Some(
                    "Reinstall with Metal: cargo install --features metal".to_string(),
                ),
            });
        } else {
            checks.push(DoctorCheck {
                name: "binary_hardware_match".to_string(),
                status: DoctorStatus::Ok,
                message: "Binary features match detected hardware.".to_string(),
                suggestion: None,
            });
        }
    }

    // Unified memory detection
    for dev in system
        .devices
        .iter()
        .filter(|d| d.unified_memory == Some(true))
    {
        let kind = &dev.kind;
        let ord = dev.ordinal.map(|o| format!(" {o}")).unwrap_or_default();
        checks.push(DoctorCheck {
            name: format!("{}_{}_unified_memory", kind, dev.ordinal.unwrap_or(0)),
            status: DoctorStatus::Ok,
            message: format!(
                "{}{}: unified memory detected. GPU and CPU share the same physical RAM.",
                kind.to_uppercase(),
                ord,
            ),
            suggestion: None,
        });
    }

    // CUDA compute capability + Flash Attention v2/v3 check
    #[cfg(feature = "cuda")]
    {
        for dev in system.devices.iter().filter(|d| d.kind == "cuda") {
            if let (Some(ord), Some((major, minor))) = (dev.ordinal, dev.compute_capability) {
                let fa_v2_ok = dev.flash_attn_compatible.unwrap_or(false);
                let fa_v3_ok = dev.flash_attn_v3_compatible.unwrap_or(false);

                // Build status strings with emojis
                let fa_v2_str = if fa_v2_ok { "✅" } else { "❌" };
                let fa_v3_str = if fa_v3_ok {
                    "✅"
                } else {
                    "❌ (requires Hopper/Compute 9.0)"
                };

                checks.push(DoctorCheck {
                    name: format!("cuda_{}_compute", ord),
                    status: DoctorStatus::Ok,
                    message: format!(
                        "GPU {}: compute {}.{} - Flash Attn v2 {}, v3 {}",
                        ord, major, minor, fa_v2_str, fa_v3_str
                    ),
                    suggestion: None,
                });

                // Warn if hardware supports flash attn v2 but binary doesn't have it
                if fa_v2_ok && !system.build.flash_attn {
                    checks.push(DoctorCheck {
                        name: format!("cuda_{}_flash_attn_v2_missing", ord),
                        status: DoctorStatus::Warn,
                        message: format!(
                            "GPU {} supports Flash Attention v2 but binary compiled without it.",
                            ord
                        ),
                        suggestion: Some(
                            "Reinstall with: cargo install --features flash-attn".to_string(),
                        ),
                    });
                }

                // Warn if hardware supports flash attn v3 but binary doesn't have it
                if fa_v3_ok && !system.build.flash_attn_v3 {
                    checks.push(DoctorCheck {
                        name: format!("cuda_{}_flash_attn_v3_missing", ord),
                        status: DoctorStatus::Warn,
                        message: format!(
                            "GPU {} supports Flash Attention v3 but binary compiled without it.",
                            ord
                        ),
                        suggestion: Some(
                            "Reinstall with: cargo install --features flash-attn-v3".to_string(),
                        ),
                    });
                }
            }
        }
    }

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

    // HuggingFace connectivity + gated model access check
    {
        let hf_info = check_hf_gated_access();
        if hf_info.reachable {
            if hf_info.token_valid_for_gated == Some(true) {
                checks.push(DoctorCheck {
                    name: "hf_connectivity".to_string(),
                    status: DoctorStatus::Ok,
                    message: format!(
                        "Hugging Face: connected ({}ms), token valid for allowed gated models.",
                        hf_info.latency_ms.unwrap_or(0)
                    ),
                    suggestion: None,
                });
            } else {
                checks.push(DoctorCheck {
                    name: "hf_connectivity".to_string(),
                    status: DoctorStatus::Warn,
                    message: format!(
                        "Hugging Face: connected ({}ms), but token invalid/missing.",
                        hf_info.latency_ms.unwrap_or(0)
                    ),
                    suggestion: Some(
                        "Run `huggingface-cli login` or set HF_TOKEN to access gated models."
                            .to_string(),
                    ),
                });
            }
        } else {
            checks.push(DoctorCheck {
                name: "hf_connectivity".to_string(),
                status: DoctorStatus::Error,
                message: format!(
                    "Hugging Face: unreachable - {}",
                    hf_info.error.unwrap_or_else(|| "unknown error".to_string())
                ),
                suggestion: Some(
                    "Check your internet connection and firewall settings.".to_string(),
                ),
            });
        }
    }

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
            #[allow(clippy::cast_precision_loss)]
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

    let has_cuda = system.devices.iter().any(|d| d.kind == "cuda");

    if system.build.cuda && !has_cuda {
        checks.push(DoctorCheck {
            name: "cuda_devices".to_string(),
            status: DoctorStatus::Warn,
            message: "CUDA support is enabled but no CUDA devices were found.".to_string(),
            suggestion: Some("Check NVIDIA driver installation.".to_string()),
        });
    }

    DoctorReport { system, checks }
}
