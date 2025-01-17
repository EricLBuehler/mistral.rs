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
    MbAmount(usize),
    Utilization(f32),
    ContextSize(usize),
}

/// Memory values are in MBs or a percentage in [0,1]. Specify block size or the default is 32.
#[allow(clippy::too_many_arguments)]
pub fn calculate_cache_config(
    _mem_gpu: MemoryGpuConfig,
    _mem_cpu: usize,
    _block_size: Option<usize>,
    _dtype: DType,
    _config: &dyn ModelConfigLike,
    _device: &Device,
    _layer_devices: &[Option<Device>],
    _silent: bool,
) -> anyhow::Result<CacheConfig> {
    anyhow::bail!("Cannot calculate cache config when not using PagedAttention.")
}
