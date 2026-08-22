/// This is the lower-level manager of the cache. It manages swapping and copying the blocks and
/// actually allocates the KV cache for the CPU and GPU. It is used by the LLMEngine to execute
/// operations issued by the scheduler.
pub(crate) mod attention_backend;
/// Content-addressable block hashing for prefix caching (vLLM v1 approach).
pub mod block_hash;
/// Flat block pool with LRU free list for KV cache block management (vLLM v1 approach).
pub mod block_pool;
mod cache_engine;
mod config;
/// Encoder output cache for multimodal models (vision/audio encoder outputs).
pub mod encoder_cache;
/// KV Cache Manager: high-level block allocation, prefix cache lookups, per-request tracking.
pub mod kv_cache_manager;
mod layers;
pub(crate) mod mm_prefix;
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub(crate) mod plan;
mod scales;
mod scheduler;
#[cfg(any(
    test,
    all(feature = "cuda", feature = "flash-attn", target_family = "unix")
))]
pub(crate) mod windowed_pool;
pub const _PAD_SLOT_ID: i64 = -1;

pub use attention_backend::AttentionBackendKind;
pub use cache_engine::{CacheConfig, CacheEngine, PagedCacheType};
use candle_core::{DType, Device};
pub use config::{
    HybridPagedKvCacheConfig, KvCacheLayout, KvCacheTopology, ModelConfigLike, ModelConfigMetadata,
};
pub use kv_cache_manager::KVCacheManager;
pub use layers::PagedAttention;
pub use scales::{load_fp8_attention_scales, Fp8AttentionScales};
pub use scheduler::{
    PagedAttentionScheduler, PagedAttentionSchedulerConfig, PagedAttentionSchedulerOutput,
};

use crate::MemoryUsage;
use tracing::info;

pub const DEFAULT_PAGED_ATTENTION_BLOCK_SIZE: usize = 32;
const GPU_RESERVE_FRACTION: f64 = 0.02;
const GPU_MIN_RESERVE_BYTES: usize = 512 * 1024 * 1024;

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub(crate) fn device_memory_cap(available_bytes: usize, device: &Device) -> usize {
    if device.is_cpu() {
        available_bytes
    } else {
        let fractional_reserve = (available_bytes as f64 * GPU_RESERVE_FRACTION) as usize;
        let reserve = fractional_reserve
            .max(GPU_MIN_RESERVE_BYTES)
            .min(available_bytes);
        available_bytes.saturating_sub(reserve)
    }
}

pub(crate) fn block_aligned_sliding_window_start(
    full_len: usize,
    query_len: usize,
    window: usize,
    block_size: usize,
) -> usize {
    let retained_len = window
        .saturating_sub(1)
        .saturating_add(query_len)
        .min(full_len);
    ((full_len - retained_len) / block_size) * block_size
}

#[cfg(test)]
mod tests {
    use super::{
        block_aligned_sliding_window_start, fit_post_load_cache_budget, MemoryGpuConfig,
        PagedAttentionConfig, PagedCacheType,
    };

    #[test]
    fn sliding_window_retains_prior_window_and_whole_query() {
        assert_eq!(block_aligned_sliding_window_start(100, 1, 4, 32), 96);
        assert_eq!(block_aligned_sliding_window_start(100, 10, 4, 32), 64);
        assert_eq!(block_aligned_sliding_window_start(8, 8, 4, 32), 0);

        for full_len in 1..130 {
            for query_len in 1..=full_len {
                for window in [1, 2, 4, 31, 32, 33, 128] {
                    let start = block_aligned_sliding_window_start(full_len, query_len, window, 32);
                    let required_start =
                        full_len.saturating_sub(window.saturating_sub(1).saturating_add(query_len));
                    assert!(start <= required_start);
                    assert!(required_start - start < 32);
                }
            }
        }
    }

    #[test]
    fn base_device_reservation_rejects_additive_overflow() -> anyhow::Result<()> {
        let config = PagedAttentionConfig::new(
            Some(32),
            MemoryGpuConfig::MbAmount(1),
            PagedCacheType::Auto,
        )?
        .with_base_device_memory_reservation(usize::MAX)?;

        let error = config
            .with_base_device_memory_reservation(1)
            .expect_err("reservation addition should overflow");
        assert!(error
            .to_string()
            .contains("paged attention device memory reservation overflow"));
        Ok(())
    }

    #[test]
    fn activation_reservation_is_idempotent_and_device_specific() -> anyhow::Result<()> {
        let mut config = PagedAttentionConfig::new(
            Some(32),
            MemoryGpuConfig::Utilization(0.85),
            PagedCacheType::Auto,
        )?
        .with_base_device_memory_reservation(4 * 1024 * 1024 * 1024)?;
        config.reserve_activation_memory(512 * 1024 * 1024, 256 * 1024 * 1024);
        config.reserve_activation_memory(256 * 1024 * 1024, 128 * 1024 * 1024);

        let reservations = config.memory_reservations()?;
        assert_eq!(
            reservations.primary_device_bytes,
            4 * 1024 * 1024 * 1024 + 512 * 1024 * 1024
        );
        assert_eq!(reservations.secondary_device_bytes, 256 * 1024 * 1024);
        Ok(())
    }

    #[test]
    fn post_load_cache_budget_preserves_memory_mode_semantics() -> anyhow::Result<()> {
        assert_eq!(
            fit_post_load_cache_budget(MemoryGpuConfig::Utilization(0.85), 40_000, 35_000)?,
            35_000
        );
        assert_eq!(
            fit_post_load_cache_budget(
                MemoryGpuConfig::BestEffortMbAmount {
                    target_mb: 40_000,
                    min_mb: Some(30_000),
                },
                40_000,
                35_000,
            )?,
            35_000
        );
        assert!(fit_post_load_cache_budget(
            MemoryGpuConfig::BestEffortMbAmount {
                target_mb: 40_000,
                min_mb: Some(36_000),
            },
            40_000,
            35_000,
        )
        .is_err());
        assert!(
            fit_post_load_cache_budget(MemoryGpuConfig::MbAmount(40_000), 40_000, 35_000,).is_err()
        );
        assert!(
            fit_post_load_cache_budget(MemoryGpuConfig::ContextSize(131_072), 40_000, 35_000,)
                .is_err()
        );
        Ok(())
    }
}

/// All memory counts in MB. Default for block size is 32.
#[derive(Clone, Copy, Debug)]
pub struct PagedAttentionConfig {
    pub(crate) block_size: Option<usize>,
    pub(crate) mem_gpu: MemoryGpuConfig,
    pub(crate) cache_type: PagedCacheType,
    pub(crate) serving_capacity: Option<usize>,
    pub(crate) base_device_memory_reservation_bytes: usize,
    pub(crate) primary_activation_memory_reservation_bytes: usize,
    pub(crate) mapped_activation_memory_reservation_bytes: usize,
    pub(crate) recurrent_checkpoint_lanes: usize,
    pub(crate) resolve_memory_utilization_after_load: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CacheMemoryReservations {
    pub primary_device_bytes: usize,
    pub secondary_device_bytes: usize,
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
            serving_capacity: None,
            base_device_memory_reservation_bytes: 0,
            primary_activation_memory_reservation_bytes: 0,
            mapped_activation_memory_reservation_bytes: 0,
            recurrent_checkpoint_lanes: 1,
            resolve_memory_utilization_after_load: true,
        })
    }

    pub fn with_serving_capacity(mut self, serving_capacity: usize) -> anyhow::Result<Self> {
        if serving_capacity == 0 {
            anyhow::bail!("paged attention serving capacity must be nonzero")
        }
        self.serving_capacity = Some(serving_capacity);
        Ok(self)
    }

    /// Reserves primary-device memory for components loaded after the paged cache is sized.
    pub fn with_base_device_memory_reservation(mut self, bytes: usize) -> anyhow::Result<Self> {
        self.base_device_memory_reservation_bytes = self
            .base_device_memory_reservation_bytes
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("paged attention device memory reservation overflow"))?;
        Ok(self)
    }

    pub fn with_recurrent_checkpoint_lanes(mut self, lanes: usize) -> anyhow::Result<Self> {
        if lanes == 0 {
            anyhow::bail!("recurrent checkpoint lane count must be nonzero")
        }
        self.recurrent_checkpoint_lanes = lanes;
        Ok(self)
    }

    pub(crate) fn reserve_activation_memory(
        &mut self,
        primary_device_bytes: usize,
        mapped_device_bytes: usize,
    ) {
        self.primary_activation_memory_reservation_bytes = self
            .primary_activation_memory_reservation_bytes
            .max(primary_device_bytes);
        self.mapped_activation_memory_reservation_bytes = self
            .mapped_activation_memory_reservation_bytes
            .max(mapped_device_bytes);
    }

    pub(crate) fn memory_reservations(&self) -> anyhow::Result<CacheMemoryReservations> {
        let primary_device_bytes = self
            .base_device_memory_reservation_bytes
            .checked_add(self.primary_activation_memory_reservation_bytes)
            .ok_or_else(|| anyhow::anyhow!("paged attention device memory reservation overflow"))?;
        Ok(CacheMemoryReservations {
            primary_device_bytes,
            secondary_device_bytes: self.mapped_activation_memory_reservation_bytes,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionImplementation {
    Eager,
    PagedAttention,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
pub enum MemoryGpuConfig {
    MbAmount(usize),
    BestEffortMbAmount {
        target_mb: usize,
        min_mb: Option<usize>,
    },
    Utilization(f32),
    ContextSize(usize),
}

// See `pagedattention.cu` CALL_V1_LAUNCHER_BLOCK_SIZE
const SUPPORTED_BLOCK_SIZE: &[usize] = &[8, 16, 32];

// Weight-loading transients freed into the stream-ordered pool are fragmented and cannot back the
// large contiguous KV tensors; return them to the driver before sizing and allocating the cache.
#[cfg(feature = "cuda")]
fn trim_cuda_mempool(device: &Device) {
    use candle_core::cuda_backend::cudarc::driver::sys;
    let Device::Cuda(cuda) = device else { return };
    let stream = cuda.cuda_stream();
    if !stream.context().has_async_alloc() {
        return;
    }
    let dev = stream.context().cu_device();
    let mut pool = std::ptr::null_mut();
    if unsafe { sys::cuDeviceGetMemPool(&mut pool, dev) } != sys::CUresult::CUDA_SUCCESS {
        return;
    }
    unsafe { sys::cuMemPoolTrimTo(pool, 0) };
}

const SIZE_IN_MB: usize = 1024 * 1024;

macro_rules! mb_to_blocks {
    ($mb_size:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $mb_size / $dtype_size / $block_size / $config.total_kv_cache_elements_per_token()
    };
}

macro_rules! ctxt_to_blocks {
    ($context_len:expr, $dtype_size:expr, $block_size:expr, $config:expr) => {
        $context_len * $dtype_size * $config.total_kv_cache_elements_per_token()
    };
}

fn fit_post_load_cache_budget(
    config: MemoryGpuConfig,
    requested_mb: usize,
    available_mb: usize,
) -> anyhow::Result<usize> {
    match config {
        MemoryGpuConfig::Utilization(_) => Ok(requested_mb.min(available_mb)),
        MemoryGpuConfig::BestEffortMbAmount { min_mb, .. } => {
            let fitted_mb = requested_mb.min(available_mb);
            if let Some(minimum) = min_mb {
                if fitted_mb < minimum {
                    anyhow::bail!(
                        "PagedAttention KV cache has {available_mb} MB available, below the required best-effort minimum of {minimum} MB."
                    );
                }
            }
            Ok(fitted_mb)
        }
        MemoryGpuConfig::MbAmount(_) | MemoryGpuConfig::ContextSize(_) => {
            if requested_mb > available_mb {
                anyhow::bail!(
                    "PagedAttention KV cache requires {requested_mb} MB but only {available_mb} MB is safely available after model loading."
                );
            }
            Ok(requested_mb)
        }
    }
}

/// Memory values are in MBs or a percentage in [0,1]. Specify block size or the default is 32.
///
/// `model_weight_size_in_bytes`: total model weight footprint. When provided, the per-device
/// share (divided by number of devices for tensor parallelism) is subtracted from the KV cache
/// memory budget. Pass `Some(total_model_size_in_bytes)` when calling **before** model loading
/// (e.g. during device mapping) so the KV cache estimate reflects memory that will actually
/// remain after the weights are loaded. Post-loading callers should pass `None` since
/// `get_memory_available()` already reflects the loaded model.
///
/// `max_num_tokens`: on Metal (unified memory), caps the KV cache to this many tokens.
/// Unlike CUDA with dedicated VRAM where unused memory is wasted, Metal's wired buffers
/// compete with the OS and CPU for the same physical RAM. On CUDA this is ignored.
/// If `None` on Metal, falls back to `config.max_seq_len()`.
#[allow(clippy::too_many_arguments)]
pub fn calculate_cache_config(
    mem_gpu: MemoryGpuConfig,
    memory_reservations: CacheMemoryReservations,
    block_size: Option<usize>,
    dtype: DType,
    cache_type: PagedCacheType,
    config: &dyn ModelConfigLike,
    device: &Device,
    layer_devices: &[Option<Device>],
    silent: bool,
    model_weight_size_in_bytes: Option<usize>,
    max_num_tokens: Option<usize>,
) -> anyhow::Result<CacheConfig> {
    let block_size = block_size.unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE);
    if !SUPPORTED_BLOCK_SIZE.contains(&block_size) {
        anyhow::bail!("Block size must be in {SUPPORTED_BLOCK_SIZE:?}, got {block_size}");
    }
    cache_type
        .validate(dtype, config, device, layer_devices)
        .map_err(anyhow::Error::msg)?;
    let model_dtype = dtype;
    let dtype = cache_type.to_dtype(dtype);
    let dtype_size = dtype.size_in_bytes();

    let mut cache_devices = Vec::new();
    for layer_device in layer_devices {
        let candidate = layer_device.as_ref().unwrap_or(device);
        if cache_devices
            .iter()
            .all(|existing: &&Device| existing.location() != candidate.location())
        {
            cache_devices.push(candidate);
        }
    }
    if cache_devices.is_empty() {
        cache_devices.push(device);
    }

    // Tensor-parallel devices hold an approximately equal share of the model weights.
    let num_devices = cache_devices.len();
    let model_weight_per_device_mb =
        model_weight_size_in_bytes.unwrap_or(0) / num_devices / SIZE_IN_MB;

    let mut min_mem_gpu = usize::MAX;
    let mut affine_reserved_mb = 0usize;
    let primary_device_memory_reservation_mb = memory_reservations
        .primary_device_bytes
        .div_ceil(SIZE_IN_MB);
    let secondary_device_memory_reservation_mb = memory_reservations
        .secondary_device_bytes
        .div_ceil(SIZE_IN_MB);
    let primary_device = device.location();
    for device in cache_devices {
        let reserved_bytes = if device.location() == primary_device {
            memory_reservations.primary_device_bytes
        } else {
            memory_reservations.secondary_device_bytes
        };
        // Weight loading enqueues stream-ordered frees without draining; sync so the memory
        // reading and the cache allocation right after this see the real free VRAM.
        if device.is_cuda() {
            device.synchronize()?;
            #[cfg(feature = "cuda")]
            trim_cuda_mempool(device);
        }
        let post_load_memory = if model_weight_size_in_bytes.is_none() && device.is_cuda() {
            Some(MemoryUsage.query(device)?)
        } else {
            None
        };
        let affine_bytes = if post_load_memory.is_some() {
            mistralrs_quant::gguf_affine_budget_bytes(device, model_dtype)
        } else {
            0
        };
        affine_reserved_mb = affine_reserved_mb.max(affine_bytes.div_ceil(SIZE_IN_MB));
        let future_reserved_bytes = reserved_bytes
            .checked_add(affine_bytes)
            .ok_or_else(|| anyhow::anyhow!("paged attention device memory reservation overflow"))?;
        let future_reserved_mb = future_reserved_bytes.div_ceil(SIZE_IN_MB);

        #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let mut mem_gpu_mb = match mem_gpu {
            MemoryGpuConfig::MbAmount(v) => v,
            MemoryGpuConfig::BestEffortMbAmount { target_mb, min_mb } => {
                if let Some(min_mb) = min_mb {
                    if target_mb < min_mb {
                        anyhow::bail!(
                            "Best-effort PagedAttention KV cache target {target_mb} MB is below the required minimum {min_mb} MB."
                        );
                    }
                }
                target_mb
            }
            MemoryGpuConfig::Utilization(f) => {
                let memory = match post_load_memory {
                    Some(memory) => memory,
                    None => MemoryUsage.query(device)?,
                };
                let total = memory.total() as f32 / SIZE_IN_MB as f32;
                if model_weight_size_in_bytes.is_some() {
                    // Pre-loading: compute budget from total memory and known model size.
                    (total * f - model_weight_per_device_mb as f32).max(0.0) as usize
                } else {
                    let used = (memory.total() - memory.available()) as f32 / SIZE_IN_MB as f32;
                    (total * f - used).max(0.0) as usize
                }
                .saturating_sub(future_reserved_mb)
            }
            MemoryGpuConfig::ContextSize(toks) => {
                // ContextSize is demand-driven (bytes needed for N tokens), not a memory budget, so model weight does not apply here.
                ctxt_to_blocks!(toks, dtype_size, block_size, config).div_ceil(SIZE_IN_MB)
            }
        };
        if let Some(memory) = post_load_memory {
            let available_mb = device_memory_cap(memory.available(), device)
                .saturating_sub(future_reserved_bytes)
                / SIZE_IN_MB;
            mem_gpu_mb = fit_post_load_cache_budget(mem_gpu, mem_gpu_mb, available_mb)?;
        }
        min_mem_gpu = min_mem_gpu.min(mem_gpu_mb);
    }
    if affine_reserved_mb > 0 && !silent {
        info!("Reserving {affine_reserved_mb} MB per GPU for packed GGUF affine weights.");
    }
    if primary_device_memory_reservation_mb > 0 && !silent {
        info!(
            "Reserving {primary_device_memory_reservation_mb} MB on the primary device for runtime components and activations."
        );
    }
    if secondary_device_memory_reservation_mb > 0 && !silent && num_devices > 1 {
        info!(
            "Reserving {secondary_device_memory_reservation_mb} MB on each mapped device for activations."
        );
    }

    // On Metal (unified memory), cap KV cache to what the model can actually use.
    // Unlike CUDA with dedicated VRAM where unused memory is wasted, Metal's wired
    // buffers compete with the OS and CPU for the same physical RAM.
    // On CUDA, all available memory is used for maximum request concurrency (vLLM approach).
    #[allow(unused_mut, unused_variables)]
    let mut mem_gpu = min_mem_gpu;
    if device.is_metal() {
        let max_tokens = max_num_tokens.unwrap_or(config.max_seq_len());
        let mem_for_tokens =
            ctxt_to_blocks!(max_tokens, dtype_size, block_size, config) / SIZE_IN_MB;
        if mem_for_tokens < mem_gpu {
            if !silent {
                info!(
                    "Metal: capping KV cache from {} MB to {} MB ({} tokens).",
                    mem_gpu, mem_for_tokens, max_tokens
                );
            }
            mem_gpu = mem_for_tokens;
        }
    }

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
        kv_cache_group_ids: config.kv_cache_group_ids(),
    })
}
