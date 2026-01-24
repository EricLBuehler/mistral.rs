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
/// Prefix caching for KV cache reuse across requests with shared prefixes.
mod prefix_cacher;
mod scheduler;
pub const _PAD_SLOT_ID: i64 = -1;

pub use block_engine::{BlockEngine, BlockRef, BlockTables, LogicalTokenBlock};
pub use block_engine_sequence::BlockEngineSequence;
pub use cache_engine::{CacheConfig, CacheEngine, PagedCacheType};
use candle_core::{DType, Device};
pub use config::{KvCacheLayout, ModelConfigLike, ModelConfigMetadata};
pub use layers::PagedAttention;
pub use scheduler::{
    PagedAttentionScheduler, PagedAttentionSchedulerConfig, PagedAttentionSchedulerOutput,
};

use crate::MemoryUsage;
use tracing::{info, warn};

pub const DEFAULT_PAGED_ATTENTION_BLOCK_SIZE: usize = 32;

/// All memory counts in MB. Default for block size is 32.
#[derive(Clone, Copy)]
pub struct PagedAttentionConfig {
    pub(crate) block_size: Option<usize>,
    pub(crate) mem_gpu: MemoryGpuConfig,
    pub(crate) cache_type: PagedCacheType,
}

impl PagedAttentionConfig {
    pub fn new(
        block_size: Option<usize>,
        mem_gpu: MemoryGpuConfig,
        cache_type: PagedCacheType,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            block_size,
            mem_gpu,
            cache_type,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
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

// See `pagedattention.cu` CALL_V1_LAUNCHER_BLOCK_SIZE
const SUPPORTED_BLOCK_SIZE: &[usize] = &[8, 16, 32];

const SIZE_IN_MB: usize = 1024 * 1024;

macro_rules! mb_to_blocks {
    ($mb_size:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $mb_size
            / $dtype_size
            / $block_size
            / $config.num_layers()
            / $config.kv_cache_elements_per_token()
    };
}

macro_rules! ctxt_to_blocks {
    ($context_len:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $context_len * $dtype_size * $config.num_layers() * $config.kv_cache_elements_per_token()
    };
}

/// Memory values are in MBs or a percentage in [0,1]. Specify block size or the default is 32.
#[allow(clippy::too_many_arguments)]
pub fn calculate_cache_config(
    mem_gpu: MemoryGpuConfig,
    block_size: Option<usize>,
    dtype: DType,
    cache_type: PagedCacheType,
    config: &dyn ModelConfigLike,
    device: &Device,
    layer_devices: &[Option<Device>],
    silent: bool,
) -> anyhow::Result<CacheConfig> {
    let block_size = block_size.unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE);
    if !SUPPORTED_BLOCK_SIZE.contains(&block_size) {
        anyhow::bail!("Block size must be in {SUPPORTED_BLOCK_SIZE:?}, got {block_size}");
    }
    let dtype = cache_type.to_dtype(dtype);
    let dtype_size = dtype.size_in_bytes();

    let mut min_mem_gpu = usize::MAX;
    for dev in layer_devices {
        let device = dev.as_ref().unwrap_or(device);

        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let mem_gpu = match mem_gpu {
            MemoryGpuConfig::MbAmount(v) => v,
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
        min_mem_gpu = min_mem_gpu.min(mem_gpu);
    }

    // // Cap at kv cache for max seq len
    // let mem_for_toks =
    //     ctxt_to_blocks!(config.max_seq_len(), dtype_size, block_size, config) / SIZE_IN_MB;
    // let mem_gpu = min_mem_gpu.min(mem_for_toks);

    // Cap Metal GPU memory to the wired (non‑paged) allocation limit reported by the kernel (`iogpu.wired_limit_mb`).
    // Users can raise this limit with `sudo sysctl -w iogpu.wired_limit_mb=<desired_mb>`.
    let mem_gpu = if matches!(device, Device::Metal(_)) {
        let metal_cap_mb = MemoryUsage.get_total_memory(device)? / SIZE_IN_MB;

        info!("Metal GPU wired limit is {metal_cap_mb} MB.");

        if min_mem_gpu > metal_cap_mb {
            if !silent {
                warn!(
                    "Capping Metal GPU memory allocation from {} MB to {} MB (limited by iogpu.wired_limit_mb). \
To raise this cap run: `sudo sysctl -w iogpu.wired_limit_mb=<desired_mb>`.",
                    min_mem_gpu,
                    metal_cap_mb
                );
            }
            metal_cap_mb
        } else {
            min_mem_gpu
        }
    } else {
        min_mem_gpu
    };

    let num_gpu_blocks = mb_to_blocks!(mem_gpu * SIZE_IN_MB, dtype_size, block_size, config);
    if num_gpu_blocks == 0 {
        anyhow::bail!("Num GPU blocks is 0. This means there is not enough memory. Either reduce the memory amount/utilization/context size or disable PagedAttention.");
    }

    if !silent {
        info!("Allocating {mem_gpu} MB for PagedAttention KV cache per GPU");
        info!("PagedAttention KV cache type is {dtype:?}");
        info!("Using PagedAttention with block size {block_size} and {num_gpu_blocks} GPU blocks: available context length is {} tokens", num_gpu_blocks*block_size);
    }
    Ok(CacheConfig {
        block_size,
        num_gpu_blocks,
        cache_type,
    })
}
