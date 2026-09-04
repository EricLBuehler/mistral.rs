//! cuTile CUDA kernels, launch wrappers, and JIT warmup.

pub mod context;
mod fp8_gemm;
mod fp8_w8a16;
mod fp8_w8a8;
mod fused_moe;
mod fused_moe_fp8;
mod gdn_prefill;
mod routed_lora;
mod split_k;
mod tune;
mod warmup;

pub use fp8_gemm::{
    cutile_fp8_gemm, fp8_gemm_supported, register_fp8_gemm_shape, FP8_GEMM_BLOCK_ROWS,
};
pub use fp8_w8a16::{cutile_fp8_w8a16, fp8_w8a16_supported, register_fp8_w8a16_shape};
pub use fp8_w8a8::{
    cutile_fp8_w8a8, fp8_w8a8_supported, register_fp8_w8a8_shape, CutileFp8W8A8Args, Fp8W8A8Scheme,
};
pub use fused_moe::{cutile_grouped_gemm, register_moe_shape};
pub use fused_moe_fp8::{
    cutile_fused_moe_fp8, fp8_moe_supported, register_moe_fp8_shape, CutileFp8MoeWeights,
    FP8_MOE_GROUP,
};
pub use gdn_prefill::{
    cutile_gdn_prefill, gdn_prefill_supported, GdnPrefillArgs, GDN_PREFILL_CHUNK,
    GDN_PREFILL_HEAD_DIM,
};
pub use routed_lora::{
    cached_cutile_routed_lora_config, cutile_routed_lora_candidate_configs,
    selected_cutile_routed_lora_config, set_cutile_routed_lora_tuned_config,
    try_cutile_routed_lora, try_cutile_routed_lora_no_sort, CutileRoutedLoraConfig,
    CutileRoutedLoraDeviceKey, CutileRoutedLoraLaunch, CutileRoutedLoraOptimizationHint,
    CutileRoutedLoraShapeKey, CutileRoutedLoraStatus, CutileRoutedLoraTuningKey,
    CutileRoutedLoraUnsupported, CUTILE_ROUTED_LORA_MAX_RANK,
};
pub use tune::{TuneMode, TUNE_CACHE_ENV, TUNE_MODE_ENV};
pub use warmup::warmup_moe_kernels;

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return (*message).to_string();
    }
    "non-string panic".to_string()
}

pub(super) fn catch_cutile_panic<T>(
    operation: &str,
    f: impl FnOnce() -> candle_core::Result<T>,
) -> candle_core::Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            candle_core::bail!("cuTile {operation} panicked: {}", panic_message(payload))
        }
    }
}

pub fn device_compute_capability(dev: &candle_core::CudaDevice) -> (i32, i32) {
    use candle_core::cuda::cudarc::driver::{result, sys};
    let cu_device = dev.cuda_stream().context().cu_device();
    let major = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(0);
    let minor = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .unwrap_or(0);
    (major, minor)
}

pub fn device_compute_major(dev: &candle_core::CudaDevice) -> i32 {
    device_compute_capability(dev).0
}

pub fn device_multiprocessor_count(dev: &candle_core::CudaDevice) -> usize {
    use candle_core::cuda::cudarc::driver::{result, sys};
    let cu_device = dev.cuda_stream().context().cu_device();
    let count = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }
    .unwrap_or(1);
    usize::try_from(count).unwrap_or(1).max(1)
}

pub fn device_supported(dev: &candle_core::CudaDevice) -> bool {
    let (major, minor) = device_compute_capability(dev);
    let Some(cuda_code) = build_cuda_version_code() else {
        return false;
    };

    device_supported_for(cuda_code, major, minor)
}

fn device_supported_for(cuda_code: u32, major: i32, minor: i32) -> bool {
    (major == 8 && cuda_code >= 1302)
        || (major == 9 && minor == 0 && cuda_code >= 1303)
        || (major >= 10 && cuda_code >= 1302)
}

fn build_cuda_version_code() -> Option<u32> {
    option_env!("MISTRALRS_BUILD_CUDA_VERSION_CODE")?
        .parse()
        .ok()
}

const MIN_TILEIRAS_MAJOR: u32 = 13;
const MIN_TILEIRAS_MINOR: u32 = 1;

fn parse_tileiras_version(output: &str) -> Option<(u32, u32)> {
    for part in output.split_whitespace() {
        let Some(version) = part.strip_prefix('V') else {
            continue;
        };
        let mut segments = version.split('.');
        let major = segments.next()?.parse().ok()?;
        let minor = segments.next()?.parse().ok()?;
        return Some((major, minor));
    }
    None
}

fn tileiras_version_supported(output: &str) -> bool {
    let Some((major, minor)) = parse_tileiras_version(output) else {
        return false;
    };
    major > MIN_TILEIRAS_MAJOR || (major == MIN_TILEIRAS_MAJOR && minor >= MIN_TILEIRAS_MINOR)
}

#[derive(Debug)]
struct TileirasCapabilities {
    targets: Vec<i32>,
}

fn parse_tileiras_targets(output: &str) -> Vec<i32> {
    let mut targets = output
        .split_whitespace()
        .filter_map(|part| part.strip_prefix("=sm_"))
        .filter(|part| part.chars().all(|c| c.is_ascii_digit()))
        .filter_map(|part| part.parse().ok())
        .collect::<Vec<_>>();
    targets.sort_unstable();
    targets.dedup();
    targets
}

fn tileiras_output(bin: &std::path::Path, arg: &str) -> Option<String> {
    let output = std::process::Command::new(bin).arg(arg).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    Some(text)
}

fn tileiras_capabilities() -> Option<&'static TileirasCapabilities> {
    static CAPABILITIES: std::sync::OnceLock<Option<TileirasCapabilities>> =
        std::sync::OnceLock::new();
    CAPABILITIES
        .get_or_init(|| {
            let bin = cutile::cutile_compiler::cuda_tile_runtime_utils::tileiras_binary();
            let version = tileiras_output(&bin, "--version")?;
            if !tileiras_version_supported(&version) {
                return None;
            }
            let targets = parse_tileiras_targets(&tileiras_output(&bin, "--help")?);
            if targets.is_empty() {
                return None;
            }
            Some(TileirasCapabilities { targets })
        })
        .as_ref()
}

/// Whether `tileiras` can JIT this Tile IR for the active GPU.
pub fn jit_available(dev: &candle_core::CudaDevice) -> bool {
    let (major, minor) = device_compute_capability(dev);
    let target = major * 10 + minor;
    tileiras_capabilities().is_some_and(|capabilities| capabilities.targets.contains(&target))
}

#[cfg(test)]
mod tests {
    use super::{device_supported_for, parse_tileiras_targets, tileiras_version_supported};

    #[test]
    fn cuda_architecture_gate_matches_tileiras_support() {
        assert!(!device_supported_for(1301, 8, 0));
        assert!(device_supported_for(1302, 8, 0));
        assert!(!device_supported_for(1302, 9, 0));
        assert!(device_supported_for(1303, 9, 0));
        assert!(!device_supported_for(1301, 10, 0));
        assert!(device_supported_for(1302, 10, 0));
    }

    #[test]
    fn tileiras_version_gate_accepts_compatible_versions() {
        assert!(!tileiras_version_supported(
            "Cuda compilation tools, release 13.0, V13.0.88"
        ));
        assert!(tileiras_version_supported(
            "Cuda compilation tools, release 13.1, V13.1.80"
        ));
        assert!(tileiras_version_supported(
            "Cuda compilation tools, release 13.2, V13.2.11"
        ));
        assert!(tileiras_version_supported(
            "Cuda compilation tools, release 14.0, V14.0.1"
        ));
    }

    #[test]
    fn tileiras_target_parser_reads_only_gpu_values() {
        let help = "--gpu-name=<value>\n  =sm_80 - SM 80\n  =sm_121 - SM 121\n  sm_90\n  =sm_bad";
        assert_eq!(parse_tileiras_targets(help), vec![80, 121]);
    }

    #[test]
    fn dependency_panics_become_errors() {
        let error = super::catch_cutile_panic("test JIT", || -> candle_core::Result<()> {
            panic!("tileiras rejected the target")
        })
        .unwrap_err();
        assert!(error.to_string().contains("test JIT panicked"));
        assert!(error.to_string().contains("tileiras rejected the target"));
    }
}

/// Launch config for the grouped MoE GEMMs, computed once from the token count and reused for both
/// GEMMs (`bm` is the `moe_align` block size). The knobs past `group_m` default to "let the
/// compiler decide" and exist for the autotuner to sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct MoeTileConfig {
    pub bm: i32,
    pub bn: i32,
    pub bk: i32,
    pub group_m: i32,
    /// K-range splits per output tile; partials reduce in a second pass when above 1.
    #[serde(default = "default_split_k")]
    pub split_k: i32,
    /// Software-pipelining latency hint on the operand loads; 0 leaves it to the compiler.
    #[serde(default)]
    pub latency: i32,
    /// Worker warps per CTA; 0 leaves it to the compiler.
    #[serde(default)]
    pub warps: i32,
    /// Target CTAs per SM; 0 leaves it to the compiler.
    #[serde(default)]
    pub occupancy: i32,
    /// CTAs per cluster; 0 leaves it to the compiler.
    #[serde(default)]
    pub cluster: i32,
}

fn default_split_k() -> i32 {
    1
}

impl MoeTileConfig {
    pub const fn tiles(bm: i32, bn: i32, bk: i32, group_m: i32) -> Self {
        Self {
            bm,
            bn,
            bk,
            group_m,
            split_k: 1,
            latency: 0,
            warps: 0,
            occupancy: 0,
            cluster: 0,
        }
    }

    pub fn to_config(&self) -> cutile::tune::Config {
        tune::config([
            ("bm", i64::from(self.bm)),
            ("bn", i64::from(self.bn)),
            ("bk", i64::from(self.bk)),
            ("group_m", i64::from(self.group_m)),
            ("split_k", i64::from(self.split_k)),
            ("latency", i64::from(self.latency)),
            ("warps", i64::from(self.warps)),
            ("occupancy", i64::from(self.occupancy)),
            ("cluster", i64::from(self.cluster)),
        ])
    }

    pub fn from_config(config: &cutile::tune::Config) -> Option<Self> {
        let int = |key: &str| config.int(key).and_then(|v| i32::try_from(v).ok());
        Some(Self {
            bm: int("bm")?,
            bn: int("bn")?,
            bk: int("bk")?,
            group_m: int("group_m")?,
            split_k: int("split_k")?,
            latency: int("latency")?,
            warps: int("warps")?,
            occupancy: int("occupancy")?,
            cluster: int("cluster")?,
        })
    }

    pub fn compile_options(&self) -> cutile::tile_kernel::CompileOptions {
        let mut options = cutile::tile_kernel::CompileOptions::new();
        if self.warps > 0 {
            options = options.num_worker_warps_per_cta(self.warps);
        }
        if self.occupancy > 0 {
            options = options.occupancy(self.occupancy);
        }
        if self.cluster > 0 {
            options = options.num_cta_in_cga(self.cluster);
        }
        options
    }
}

impl std::fmt::Display for MoeTileConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}x{} g{}", self.bm, self.bn, self.bk, self.group_m)?;
        for (label, value, default) in [
            ("s", self.split_k, 1),
            ("l", self.latency, 0),
            ("w", self.warps, 0),
            ("o", self.occupancy, 0),
            ("c", self.cluster, 0),
        ] {
            if value != default {
                write!(f, " {label}{value}")?;
            }
        }
        Ok(())
    }
}

/// What distinguishes one MoE expert shape from another for warmup and tuning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MoeShapeKey {
    pub hidden: usize,
    pub inter: usize,
    pub num_experts: usize,
    pub top_k: usize,
}

impl std::fmt::Display for MoeShapeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "e{}_h{}_i{}_k{}",
            self.num_experts, self.hidden, self.inter, self.top_k
        )
    }
}

pub const fn get_default_config(m: usize, num_experts: usize) -> MoeTileConfig {
    let bm = if m <= 32 {
        16
    } else if m <= 96 {
        32
    } else if m <= 512 {
        64
    } else {
        128
    };
    let bn = if m <= 64 { 64 } else { 128 };
    let bk = if m <= 64 { 128 } else { 64 };
    let num_experts = if num_experts == 0 { 1 } else { num_experts };
    let group_m = if m / num_experts > 128 { 16 } else { 1 };
    MoeTileConfig::tiles(bm, bn, bk, group_m)
}
