/// The higher-level manager of the blocks allocated. Operations performed by the block engine do
/// not directly change memory.
mod block_engine;
mod block_engine_sequence;
/// This is the lower-level manager of the cache. It manages swapping and copying the blocks and
/// actually allocates the KV cache for the CPU and GPU. It is used by the LLMEngine to execute
/// operations issued by the scheduler.
mod cache_engine;
mod config;
mod layers;
mod scheduler;
pub const _PAD_SLOT_ID: i64 = -1;

pub use block_engine::{BlockEngine, BlockTables, LogicalTokenBlock};
pub use block_engine_sequence::BlockEngineSequence;
pub use cache_engine::{CacheConfig, CacheEngine};
use candle_core::{DType, Device};
pub use config::{ModelConfigLike, ModelConfigMetadata};
pub use layers::PagedAttention;
pub use scheduler::{
    PagedAttentionScheduler, PagedAttentionSchedulerConfig, PagedAttentionSchedulerOutput,
};

use crate::MemoryUsage;
use tracing::info;

/// All memory counts in MB. Default for block size is 32.
#[derive(Clone, Copy)]
pub struct PagedAttentionConfig {
    pub(crate) block_size: Option<usize>,
    pub(crate) mem_cpu: usize,
    pub(crate) mem_gpu: MemoryGpuConfig,
}

impl PagedAttentionConfig {
    pub fn new(
        _block_size: Option<usize>,
        _mem_cpu: usize,
        _mem_gpu: MemoryGpuConfig,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("PagedAttention is only supported for CUDA, compile with feature `cuda`.")
    }
}

pub enum AttentionImplementation {
    Eager,
    PagedAttention,
}

#[derive(Clone, Copy)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
pub enum MemoryGpuConfig {
    Amount(usize),
    Utilization(f32),
    ContextSize(usize),
}

// See `pagedattention.cu` CALL_V1_LAUNCHER_BLOCK_SIZE
const SUPPORTED_BLOCK_SIZE: &[usize] = &[8, 16, 32];

const SIZE_IN_MB: usize = 1024 * 1024;

macro_rules! mb_to_blocks {
    ($mb_size:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $mb_size
            / $dtype_size
            / $block_size
            / $config.num_kv_heads()
            / ($config.hidden_size() / $config.num_attn_heads())
            / $config.num_layers()
            / 2
    };
}

macro_rules! ctxt_to_blocks {
    ($context_len:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $context_len
            * $dtype_size
            * $config.num_kv_heads()
            * ($config.hidden_size() / $config.num_attn_heads())
            * $config.num_layers()
            * 2
    };
}

/// Memory values are in MBs or a percentage in [0,1]. Specify block size or the default is 32.
pub fn calculate_cache_config(
    mem_gpu: MemoryGpuConfig,
    mem_cpu: usize,
    block_size: Option<usize>,
    dtype: DType,
    config: &dyn ModelConfigLike,
    device: &Device,
) -> anyhow::Result<CacheConfig> {
    let block_size = block_size.unwrap_or(32);
    if !SUPPORTED_BLOCK_SIZE.contains(&block_size) {
        anyhow::bail!("Block size must be in {SUPPORTED_BLOCK_SIZE:?}, got {block_size}");
    }
    let dtype_size = dtype.size_in_bytes();

    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    let mem_gpu = match mem_gpu {
        MemoryGpuConfig::Amount(v) => v,
        MemoryGpuConfig::Utilization(f) => {
            let free = MemoryUsage.get_memory_available(device)? as f32 / SIZE_IN_MB as f32;
            let total = MemoryUsage.get_total_memory(device)? as f32 / SIZE_IN_MB as f32;
            let used = total - free;
            (total * f - used) as usize
        }
        MemoryGpuConfig::ContextSize(toks) => {
            ctxt_to_blocks!(toks, dtype_size, block_size, config) / SIZE_IN_MB
        }
    };
    info!("Allocating {mem_gpu} MB for PagedAttention KV cache");

    let num_gpu_blocks = mb_to_blocks!(mem_gpu * SIZE_IN_MB, dtype_size, block_size, config);
    let num_cpu_blocks = mb_to_blocks!(mem_cpu * SIZE_IN_MB, dtype_size, block_size, config);
    if num_gpu_blocks == 0 {
        anyhow::bail!("Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention.");
    }
    info!("Using PagedAttention with block size {block_size} and {num_gpu_blocks} GPU blocks: available context length is {} tokens", num_gpu_blocks*block_size);
    Ok(CacheConfig {
        block_size,
        num_gpu_blocks,
        num_cpu_blocks,
    })
}
