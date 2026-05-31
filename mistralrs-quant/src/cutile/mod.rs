//! cuTile CUDA kernels, launch wrappers, and JIT warmup.

pub mod context;
mod fused_moe;
mod warmup;

pub use fused_moe::{cutile_grouped_gemm, register_moe_shape};
pub use warmup::warmup_moe_kernels;

pub fn device_compute_major(dev: &candle_core::CudaDevice) -> i32 {
    use candle_core::cuda::cudarc::driver::{result, sys};
    let cu_device = dev.cuda_stream().context().cu_device();
    unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(0)
}

/// Whether cuTile's JIT supports this device: Ampere (sm_8x) or Blackwell+ (sm_100/sm_120), not Hopper (sm_90).
pub fn device_supported(dev: &candle_core::CudaDevice) -> bool {
    let major = device_compute_major(dev);
    major == 8 || major >= 10
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
