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
use candle_core::DType;
pub use config::{ModelConfigLike, ModelConfigMetadata};
pub use layers::PagedAttention;
pub use scheduler::{
    PagedAttentionScheduler, PagedAttentionSchedulerConfig, PagedAttentionSchedulerOutput,
};

/// All memory counts in MB. Default for block size is 16.
#[derive(Clone, Copy)]
pub struct PagedAttentionConfig {
    pub(crate) block_size: Option<usize>,
    pub(crate) mem_cpu: usize,
    pub(crate) mem_gpu: usize,
}

impl PagedAttentionConfig {
    pub fn new(
        _block_size: Option<usize>,
        _mem_cpu: usize,
        _mem_gpu: usize,
    ) -> anyhow::Result<Self> {
        anyhow::bail!("PagedAttention is only supported for CUDA, compile with feature `cuda`.")
    }
}

pub enum AttentionImplementation {
    Eager,
    PagedAttention,
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

/// Memory values are in MBs. Specify block size or the default is 16.
pub fn calculate_cache_config(
    mem_gpu: usize,
    mem_cpu: usize,
    block_size: Option<usize>,
    dtype: DType,
    config: &dyn ModelConfigLike,
) -> anyhow::Result<CacheConfig> {
    let block_size = block_size.unwrap_or(16);
    if !SUPPORTED_BLOCK_SIZE.contains(&block_size) {
        anyhow::bail!("Block size must be in {SUPPORTED_BLOCK_SIZE:?}, got {block_size}");
    }
    let dtype_size = dtype.size_in_bytes();

    let num_gpu_blocks = mb_to_blocks!(mem_gpu * SIZE_IN_MB, dtype_size, block_size, config);
    let num_cpu_blocks = mb_to_blocks!(mem_cpu * SIZE_IN_MB, dtype_size, block_size, config);
    Ok(CacheConfig {
        block_size,
        num_gpu_blocks,
        num_cpu_blocks,
    })
}
