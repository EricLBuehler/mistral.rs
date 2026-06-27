//! cuTile CUDA kernels, launch wrappers, and JIT warmup.

pub mod context;
mod fused_moe;
mod warmup;

pub use fused_moe::{cutile_grouped_gemm, register_moe_shape};
pub use warmup::warmup_moe_kernels;

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

pub fn device_supported(dev: &candle_core::CudaDevice) -> bool {
    let (major, minor) = device_compute_capability(dev);
    let Some(cuda_code) = build_cuda_version_code() else {
        return false;
    };

    (major == 8 && cuda_code >= 1302)
        || (major == 9 && minor == 0 && cuda_code >= 1303)
        || (major >= 10 && cuda_code >= 1301)
}

fn build_cuda_version_code() -> Option<u32> {
    option_env!("MISTRALRS_BUILD_CUDA_VERSION_CODE")?
        .parse()
        .ok()
}

const MIN_TILEIRAS_MAJOR: u32 = 13;
const MIN_TILEIRAS_MINOR: u32 = 2;

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

/// Whether the external `tileiras` JIT assembler is reachable at runtime and accepts this Tile IR.
/// Probed once; resolution matches cutile-compiler: `CUTILE_TILEIRAS_PATH` env, else PATH lookup.
pub fn jit_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let bin = std::env::var_os("CUTILE_TILEIRAS_PATH")
            .filter(|v| !v.is_empty())
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("tileiras"));
        let Ok(output) = std::process::Command::new(bin).arg("--version").output() else {
            return false;
        };
        if !output.status.success() {
            return false;
        }
        let mut version = String::from_utf8_lossy(&output.stdout).into_owned();
        version.push_str(&String::from_utf8_lossy(&output.stderr));
        tileiras_version_supported(&version)
    })
}

#[cfg(test)]
mod tests {
    use super::tileiras_version_supported;

    #[test]
    fn tileiras_version_gate_accepts_compatible_versions() {
        assert!(!tileiras_version_supported(
            "Cuda compilation tools, release 13.1, V13.1.80"
        ));
        assert!(tileiras_version_supported(
            "Cuda compilation tools, release 13.2, V13.2.11"
        ));
        assert!(tileiras_version_supported(
            "Cuda compilation tools, release 14.0, V14.0.1"
        ));
    }
}

/// Launch tile config for the grouped GEMM, computed once from the token count and reused for both GEMMs (`bm` is the `moe_align` block size).
#[derive(Clone, Copy)]
pub struct MoeTileConfig {
    pub bm: i32,
    pub bn: i32,
    pub bk: i32,
    pub group_m: i32,
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
    MoeTileConfig {
        bm,
        bn,
        bk,
        group_m,
    }
}
