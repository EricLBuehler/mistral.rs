//! cuTile MoE backend: a faithful port of vLLM's fused MoE (unquantized BF16). The kernel, its
//! launch, and its warmup live in [`fused_moe`]; [`warmup`] drives JIT warmup across kernels;
//! [`context`] bridges candle's CUDA stream into cuTile.

pub mod context;
mod fused_moe;
mod warmup;

pub use fused_moe::{cutile_grouped_gemm, register_moe_shape};
pub use warmup::warmup_moe_kernels;

/// Launch tile config from vLLM `get_default_config` (bf16 branch; E used only for GROUP_SIZE_M).
/// Computed once from the original token count and reused for both GEMMs; `bm` is the `moe_align`
/// block size.
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
