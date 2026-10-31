/// This is the lower-level manager of the cache. It manages swapping and copying the blocks and
/// actually allocates the KV cache for the CPU and GPU. It is used by the LLMEngine to execute
/// operations issued by the scheduler.
pub(crate) mod attention_backend;
/// Content-addressable block hashing for prefix caching (vLLM v1 approach).
pub mod block_hash;
/// Flat block pool with LRU free list for KV cache block management (vLLM v1 approach).
pub mod block_pool;
pub(crate) mod block_table_rows;
mod cache_allocation;
mod cache_engine;
mod config;
/// Encoder output cache for multimodal models (vision/audio encoder outputs).
pub mod encoder_cache;
/// KV Cache Manager: high-level block allocation, prefix cache lookups, per-request tracking.
pub mod kv_cache_manager;
mod layers;
pub(crate) mod mask_cache;
#[cfg(all(test, feature = "metal"))]
mod metal_hnd_tests;
pub(crate) mod mm_prefix;
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
pub use cache_allocation::allocate_cache;
#[cfg(feature = "metal")]
pub(crate) use cache_engine::configured_metal_cache_layout;
pub use cache_engine::{CacheConfig, CacheEngine, PagedCacheType};
use candle_core::{DType, Device};
pub(crate) use config::with_attention_features;
#[cfg(feature = "metal")]
pub(crate) use config::AttentionFeaturesConfig;
pub use config::{
    HybridPagedKvCacheConfig, KvCacheLayout, KvCacheTopology, ModelConfigLike, ModelConfigMetadata,
    PrefixPrefillAttentionFeatures,
};
pub use kv_cache_manager::KVCacheManager;
pub use layers::PagedAttention;
pub use scales::{load_fp8_attention_scales, Fp8AttentionScales};
pub use scheduler::{
    PagedAttentionScheduler, PagedAttentionSchedulerConfig, PagedAttentionSchedulerOutput,
};

use crate::memory_plan::{
    allocation_budget_with_reservations, resolve_device_budget, resolve_forward_budget,
    ExecutionMemoryLimits, RuntimeMemoryPlan,
};
pub use crate::MemoryGoal;
use crate::MemoryUsage;
use tracing::info;

pub const DEFAULT_PAGED_ATTENTION_BLOCK_SIZE: usize = 32;
const MIN_CACHE_BLOCKS: usize = 2;
#[cfg(feature = "metal")]
const GPU_RESERVE_FRACTION: f64 = 0.02;
#[cfg(feature = "metal")]
const GPU_MIN_RESERVE_BYTES: usize = 512 * 1024 * 1024;

#[cfg(feature = "metal")]
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
    use candle_core::Device;

    use super::{
        block_aligned_sliding_window_start, cache_prefix_capacity, kv_layer_buffer_bytes_per_block,
        max_cache_blocks, requested_cache_blocks, HybridPagedKvCacheConfig, KvCacheLayout,
        MemoryGoal, ModelConfigMetadata, PagedAttentionConfig, PagedCacheType, PrefixCacheCapacity,
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
            MemoryGoal::KvBytes(1024 * 1024),
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
    fn recurrent_prefix_capacity_defaults_to_zero_and_can_be_configured() -> anyhow::Result<()> {
        let config = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::KvBytes(1024 * 1024),
            PagedCacheType::Auto,
        )?;
        assert_eq!(config.recurrent_checkpoint_lanes, 1);
        assert!(!config.recurrent_checkpoint_lanes_auto);
        assert_eq!(config.recurrent_prefix_capacity, 0);
        let fixed = config
            .clone()
            .with_recurrent_prefix_capacity(PrefixCacheCapacity::Fixed(7));
        assert_eq!(fixed.recurrent_prefix_capacity, 7);
        assert!(!fixed.recurrent_prefix_capacity_auto);
        let auto = config.with_recurrent_prefix_capacity(PrefixCacheCapacity::Auto);
        assert_eq!(
            auto.recurrent_prefix_capacity,
            crate::DEFAULT_PREFIX_CACHE_N
        );
        assert!(auto.recurrent_prefix_capacity_auto);
        assert_eq!(PrefixCacheCapacity::from(None), PrefixCacheCapacity::Auto);
        assert!(PrefixCacheCapacity::from(Some(0)).is_disabled());
        Ok(())
    }

    #[test]
    fn activation_reservation_is_idempotent_and_device_specific() -> anyhow::Result<()> {
        let mut config = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::Utilization(0.85),
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
    fn reclaimable_weights_limit_physical_cache_budget_without_double_counting_headroom(
    ) -> anyhow::Result<()> {
        use crate::utils::memory_usage::DeviceMemory;
        const GIB: usize = 1 << 30;
        let snapshot = DeviceMemory::Unified {
            total: 128 * GIB,
            available: 44 * GIB,
            allocated: 84 * GIB,
            working_set_limit: 128 * GIB,
        };
        let mut config = PagedAttentionConfig::new(None, MemoryGoal::Auto, PagedCacheType::Auto)?;
        let policy = config.allocation_budget(&Device::Cpu, snapshot)?;
        config.reserve_reclaimable_weight_cache(&Device::Cpu, 30 * GIB)?;
        assert_eq!(
            config.allocation_budget(&Device::Cpu, snapshot)?,
            policy.min(14 * GIB)
        );
        let discrete = DeviceMemory::Discrete {
            total: 128 * GIB,
            free: 44 * GIB,
        };
        assert_eq!(
            config.allocation_budget(&Device::Cpu, discrete)?,
            crate::memory_plan::device_memory_budget(MemoryGoal::Auto, discrete)?,
        );
        Ok(())
    }

    #[test]
    fn explicit_cache_capacity_includes_reserved_block() -> anyhow::Result<()> {
        assert_eq!(
            requested_cache_blocks(MemoryGoal::KvTokens(33), 32, 1024)?,
            Some(3)
        );
        assert_eq!(
            requested_cache_blocks(MemoryGoal::KvTokens(32), 32, 1024)?,
            Some(2)
        );
        assert_eq!(
            requested_cache_blocks(MemoryGoal::KvBytes(2500), 32, 1024)?,
            Some(2)
        );
        assert!(requested_cache_blocks(MemoryGoal::KvBytes(1024), 32, 1024).is_err());
        assert_eq!(requested_cache_blocks(MemoryGoal::Auto, 32, 1024)?, None);
        Ok(())
    }

    #[test]
    fn token_budget_fits_requested_tokens_and_reserved_block() -> anyhow::Result<()> {
        for head_dim in [8, 128] {
            let model = ModelConfigMetadata {
                max_seq_len: 4096,
                num_layers: 16,
                hidden_size: 1024,
                num_kv_heads: 8,
                num_attn_heads: 8,
                sliding_window: None,
                k_head_dim: head_dim,
                v_head_dim: head_dim,
                kv_cache_layout: KvCacheLayout::Standard,
            };
            for tokens in [1, 31, 32, 33, 4096] {
                let mut config = PagedAttentionConfig::new(
                    Some(32),
                    MemoryGoal::KvTokens(tokens),
                    PagedCacheType::Auto,
                )?;
                let cache = super::calculate_cache_config(
                    &mut config,
                    super::CacheConfigRequest {
                        memory_reservations: super::CacheMemoryReservations::default(),
                        dtype: candle_core::DType::BF16,
                        model: &model,
                        device: &candle_core::Device::Cpu,
                        layer_devices: &[],
                        capacity_activation_estimate: &|_| Ok(Default::default()),
                        reserve_decode_graphs: false,
                        silent: true,
                    },
                )?;
                assert!(
                    (cache.num_gpu_blocks - 1) * cache.block_size >= tokens,
                    "token capacity {tokens} does not fit with head dimension {head_dim}"
                );
                assert_eq!(cache.num_gpu_blocks, tokens.div_ceil(cache.block_size) + 1);
            }
        }
        Ok(())
    }

    #[test]
    fn rounded_buffers_reduce_byte_capacity_without_reducing_token_goals() -> anyhow::Result<()> {
        const ALIGNMENT: usize = 16;
        let allocation = |blocks: usize| {
            Ok([7usize, 13]
                .into_iter()
                .map(|bytes| (blocks * bytes).max(1).div_ceil(ALIGNMENT) * ALIGNMENT)
                .sum::<usize>())
        };
        assert_eq!(max_cache_blocks(4, 80, allocation)?, 3);
        assert_eq!(max_cache_blocks(4, 79, allocation)?, 2);
        assert_eq!(max_cache_blocks(4, 31, allocation)?, 0);
        let token_blocks = requested_cache_blocks(MemoryGoal::KvTokens(96), 32, 20)?.unwrap();
        assert_eq!(token_blocks, 4);
        assert!(allocation(token_blocks)? > 80);
        assert_eq!(max_cache_blocks(usize::MAX, 1, |_| Ok(1))?, usize::MAX);
        assert!(max_cache_blocks(1, 1, |_| anyhow::bail!("allocation failed")).is_err());
        Ok(())
    }

    #[test]
    fn cache_capacity_leaves_room_for_its_growing_prefix_workspace() -> anyhow::Result<()> {
        const BUDGET: usize = 1024;
        const BYTES_PER_BLOCK: usize = 20;
        const PREFIX_BYTES_PER_BLOCK: usize = 5;
        let required = |blocks: usize| {
            Ok(blocks * BYTES_PER_BLOCK + blocks.saturating_sub(1) * PREFIX_BYTES_PER_BLOCK)
        };
        let blocks = max_cache_blocks(BUDGET / BYTES_PER_BLOCK, BUDGET, required)?;
        assert_eq!(blocks, 41);
        assert!(required(blocks)? <= BUDGET);
        assert!(required(blocks + 1)? > BUDGET);
        assert!(required(BUDGET / BYTES_PER_BLOCK)? > BUDGET);
        Ok(())
    }

    #[test]
    fn minimum_cache_search_does_not_send_zero_capacity_to_estimator() -> anyhow::Result<()> {
        const BLOCK_SIZE: usize = 32;
        const BYTES_PER_BLOCK: usize = 20;
        const MAX_CONTEXT: usize = 4096;
        let required = |blocks: usize| {
            let tokens = cache_prefix_capacity(blocks, BLOCK_SIZE, MAX_CONTEXT);
            anyhow::ensure!(tokens > 0, "prefix estimator requires nonzero capacity");
            Ok(blocks * BYTES_PER_BLOCK + tokens)
        };
        let minimum_budget = 2 * BYTES_PER_BLOCK + BLOCK_SIZE;
        assert_eq!(max_cache_blocks(2, minimum_budget, required)?, 2);
        assert_eq!(max_cache_blocks(2, minimum_budget - 1, required)?, 1);
        Ok(())
    }

    #[test]
    fn allocation_geometry_keeps_distinct_k_v_and_empty_hybrid_buffers() -> anyhow::Result<()> {
        let mut model = ModelConfigMetadata {
            max_seq_len: 4096,
            num_layers: 2,
            hidden_size: 1024,
            num_kv_heads: 2,
            num_attn_heads: 8,
            sliding_window: None,
            k_head_dim: 64,
            v_head_dim: 128,
            kv_cache_layout: KvCacheLayout::Standard,
        };
        assert_eq!(
            kv_layer_buffer_bytes_per_block(&model, 0, 32, candle_core::DType::BF16)?,
            [8192, 16384]
        );
        model.kv_cache_layout = KvCacheLayout::Mla {
            kv_lora_rank: 256,
            kpe_head_dim: 64,
        };
        assert_eq!(
            kv_layer_buffer_bytes_per_block(&model, 0, 32, candle_core::DType::F8E4M3)?,
            [8192, 2048]
        );
        let hybrid = HybridPagedKvCacheConfig::new(model, vec![true, false]);
        assert_eq!(
            kv_layer_buffer_bytes_per_block(&hybrid, 1, 32, candle_core::DType::BF16)?,
            [0, 0]
        );
        Ok(())
    }
}

/// Shared cache capacity and runtime memory reservations.
/// Prefix cache capacity: `Auto` retains up to `DEFAULT_PREFIX_CACHE_N` sequences within the memory budget,
/// `Fixed(n)` retains exactly `n`, and `Fixed(0)` disables prefix caching.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefixCacheCapacity {
    #[default]
    Auto,
    Fixed(usize),
}

impl PrefixCacheCapacity {
    pub const DISABLED: Self = Self::Fixed(0);

    /// Sequence count requested before any memory-budget reduction.
    pub fn requested(self) -> usize {
        match self {
            Self::Auto => crate::DEFAULT_PREFIX_CACHE_N,
            Self::Fixed(n) => n,
        }
    }

    pub fn is_disabled(self) -> bool {
        self == Self::DISABLED
    }
}

impl From<Option<usize>> for PrefixCacheCapacity {
    fn from(count: Option<usize>) -> Self {
        count.map_or(Self::Auto, Self::Fixed)
    }
}

#[derive(Clone, Debug)]
pub struct PagedAttentionConfig {
    pub(crate) block_size: Option<usize>,
    pub(crate) memory_goal: MemoryGoal,
    pub(crate) cache_type: PagedCacheType,
    pub(crate) serving_capacity: Option<usize>,
    pub(crate) base_device_memory_reservation_bytes: usize,
    pub(crate) primary_activation_memory_reservation_bytes: usize,
    pub(crate) mapped_activation_memory_reservation_bytes: usize,
    pub(crate) recurrent_checkpoint_lanes: usize,
    pub(crate) recurrent_checkpoint_lanes_auto: bool,
    pub(crate) recurrent_prefix_capacity: usize,
    pub(crate) recurrent_prefix_capacity_auto: bool,
    pub(crate) memory_budget: Option<std::sync::Arc<crate::resource_plan::ModelMemoryBudget>>,
    reclaimable_weight_cache_bytes: std::collections::HashMap<candle_core::DeviceLocation, usize>,
    pub(crate) execution_limits: ExecutionMemoryLimits,
    pub(crate) planned_query_tokens: usize,
    pub(crate) logits_bytes_per_query_token: usize,
    pub(crate) primary_non_attention_memory_reservation_bytes: usize,
    pub(crate) mapped_non_attention_memory_reservation_bytes: usize,
    pub(crate) encoder_activation_memory_reservation_bytes: usize,
    pub(crate) custom_prefill_mask: bool,
    pub(crate) metal_prefill_mask_is_structured: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CacheMemoryReservations {
    pub primary_device_bytes: usize,
    pub secondary_device_bytes: usize,
    pub primary_staging_bytes: usize,
    pub secondary_staging_bytes: usize,
}

impl PagedAttentionConfig {
    pub fn memory_goal(&self) -> MemoryGoal {
        self.memory_goal
    }

    pub fn new(
        block_size: Option<usize>,
        memory_goal: MemoryGoal,
        cache_type: PagedCacheType,
    ) -> anyhow::Result<Self> {
        memory_goal.validate()?;
        if block_size.is_some_and(|size| !SUPPORTED_BLOCK_SIZE.contains(&size)) {
            anyhow::bail!("Block size must be in {SUPPORTED_BLOCK_SIZE:?}, got {block_size:?}");
        }
        Ok(Self {
            block_size,
            memory_goal,
            cache_type,
            serving_capacity: None,
            base_device_memory_reservation_bytes: 0,
            primary_activation_memory_reservation_bytes: 0,
            mapped_activation_memory_reservation_bytes: 0,
            recurrent_checkpoint_lanes: 1,
            recurrent_checkpoint_lanes_auto: false,
            recurrent_prefix_capacity: 0,
            recurrent_prefix_capacity_auto: false,
            memory_budget: None,
            reclaimable_weight_cache_bytes: std::collections::HashMap::new(),
            execution_limits: ExecutionMemoryLimits::default(),
            planned_query_tokens: 0,
            logits_bytes_per_query_token: 0,
            primary_non_attention_memory_reservation_bytes: 0,
            mapped_non_attention_memory_reservation_bytes: 0,
            encoder_activation_memory_reservation_bytes: 0,
            custom_prefill_mask: false,
            metal_prefill_mask_is_structured: false,
        })
    }

    pub(crate) fn allocation_memory_goal(&self) -> MemoryGoal {
        self.memory_budget
            .as_ref()
            .map_or(self.memory_goal, |budget| budget.memory_goal())
    }

    pub fn initialize_memory_budget(&self, devices: &[Device]) -> anyhow::Result<()> {
        if let Some(budget) = &self.memory_budget {
            budget.initialize(devices)?;
        }
        Ok(())
    }

    pub fn allocation_budget(
        &self,
        device: &Device,
        snapshot: crate::utils::memory_usage::DeviceMemory,
    ) -> anyhow::Result<usize> {
        let policy_budget = match &self.memory_budget {
            Some(budget) => budget.allocation_budget(device.location(), snapshot),
            None => crate::memory_plan::device_memory_budget(self.memory_goal, snapshot),
        }?;
        if snapshot.is_unified() {
            let cached = self
                .reclaimable_weight_cache_bytes
                .get(&device.location())
                .copied()
                .unwrap_or_default();
            // Clean weight pages count as available RAM but must survive persistent KV allocations.
            Ok(policy_budget.min(snapshot.physical_available().saturating_sub(cached)))
        } else {
            Ok(policy_budget)
        }
    }

    pub(crate) fn execution_budget(
        &self,
        device: &Device,
        snapshot: crate::utils::memory_usage::DeviceMemory,
    ) -> anyhow::Result<usize> {
        let budget = match &self.memory_budget {
            Some(budget) => budget.execution_budget(device.location(), snapshot)?,
            None => resolve_forward_budget(self.memory_goal, snapshot)?.allocatable_bytes,
        };
        let cached = if snapshot.is_unified() {
            self.reclaimable_weight_cache_bytes
                .get(&device.location())
                .copied()
                .unwrap_or_default()
        } else {
            0
        };
        Ok(budget.min(snapshot.physical_available().saturating_sub(cached)))
    }

    pub(crate) fn allocation_budget_with_reservations(
        &self,
        device: &Device,
        snapshot: crate::utils::memory_usage::DeviceMemory,
        persistent: usize,
        transient: usize,
    ) -> anyhow::Result<usize> {
        allocation_budget_with_reservations(
            self.allocation_budget(device, snapshot)?,
            self.execution_budget(device, snapshot)?,
            persistent,
            transient,
        )
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn reserve_reclaimable_weight_cache(
        &mut self,
        device: &Device,
        bytes: usize,
    ) -> anyhow::Result<()> {
        let cached = self
            .reclaimable_weight_cache_bytes
            .entry(device.location())
            .or_default();
        *cached = cached
            .checked_add(bytes)
            .ok_or_else(|| anyhow::anyhow!("reclaimable weight cache reservation overflow"))?;
        Ok(())
    }

    pub fn with_memory_goal(mut self, goal: MemoryGoal) -> anyhow::Result<Self> {
        goal.validate()?;
        anyhow::ensure!(
            self.memory_budget.is_none() || goal == self.memory_goal,
            "Set memory goals before multi-model budget planning; re-plan the group to change a goal"
        );
        self.memory_goal = goal;
        Ok(self)
    }

    pub fn with_execution_limits(mut self, limits: ExecutionMemoryLimits) -> anyhow::Result<Self> {
        limits.validate()?;
        self.execution_limits = limits;
        self.serving_capacity = Some(limits.max_num_seqs);
        Ok(self)
    }

    pub(crate) fn runtime_memory_plan(&self) -> RuntimeMemoryPlan {
        RuntimeMemoryPlan {
            memory_goal: self.allocation_memory_goal(),
            limits: self.execution_limits,
            recurrent_prefix_capacity: self.recurrent_prefix_capacity,
            primary_activation_bytes: self.primary_activation_memory_reservation_bytes,
            mapped_activation_bytes: self.mapped_activation_memory_reservation_bytes,
            primary_non_attention_bytes: self.primary_non_attention_memory_reservation_bytes,
            mapped_non_attention_bytes: self.mapped_non_attention_memory_reservation_bytes,
            encoder_activation_bytes: self.encoder_activation_memory_reservation_bytes,
            planned_query_tokens: self.planned_query_tokens,
            logits_bytes_per_query_token: self.logits_bytes_per_query_token,
            custom_prefill_mask: self.custom_prefill_mask,
            metal_prefill_mask_is_structured: self.metal_prefill_mask_is_structured,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn runtime_memory_reservation(
        &self,
        primary: &Device,
        devices: &[Device],
        prefix: Option<crate::resource_plan::RecurrentPrefixReservation>,
    ) -> Option<std::sync::Arc<crate::resource_plan::RuntimeMemoryReservation>> {
        let Some(budget) = self.memory_budget.as_ref() else {
            return prefix.map(crate::resource_plan::RuntimeMemoryReservation::standalone);
        };
        let mut bytes = devices
            .iter()
            .map(|device| {
                (
                    device.location(),
                    self.mapped_activation_memory_reservation_bytes,
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        bytes.insert(
            primary.location(),
            self.primary_activation_memory_reservation_bytes,
        );
        Some(budget.reserve_runtime(bytes, prefix))
    }

    pub fn with_serving_capacity(mut self, serving_capacity: usize) -> anyhow::Result<Self> {
        if serving_capacity == 0 {
            anyhow::bail!("paged attention serving capacity must be nonzero")
        }
        self.serving_capacity = Some(serving_capacity);
        self.execution_limits.max_num_seqs = serving_capacity;
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
        self.recurrent_checkpoint_lanes_auto = false;
        Ok(self)
    }

    pub fn with_recurrent_prefix_capacity(mut self, capacity: PrefixCacheCapacity) -> Self {
        self.recurrent_prefix_capacity = capacity.requested();
        self.recurrent_prefix_capacity_auto = capacity == PrefixCacheCapacity::Auto;
        self
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
            ..Default::default()
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionImplementation {
    Eager,
    PagedAttention,
}

// See `pagedattention.cu` CALL_V1_LAUNCHER_BLOCK_SIZE
const SUPPORTED_BLOCK_SIZE: &[usize] = &[8, 16, 32];

// Weight-loading transients freed into the stream-ordered pool are fragmented and cannot back the
// large contiguous KV tensors; return them to the driver before sizing and allocating the cache.
#[cfg(feature = "cuda")]
fn trim_cuda_mempool(device: &Device) -> candle_core::Result<()> {
    MemoryUsage.trim_cuda_memory_pool(device, 0)?;
    Ok(())
}

const SIZE_IN_MB: usize = 1024 * 1024;

pub(crate) fn requested_cache_blocks(
    goal: MemoryGoal,
    block_size: usize,
    total_bytes_per_block: usize,
) -> anyhow::Result<Option<usize>> {
    goal.validate()?;
    let blocks = match goal {
        MemoryGoal::KvBytes(bytes) => Some(bytes / total_bytes_per_block),
        MemoryGoal::KvTokens(tokens) => Some(
            tokens
                .div_ceil(block_size)
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("KV token capacity overflow"))?,
        ),
        _ => None,
    };
    if blocks.is_some_and(|blocks| blocks < 2) {
        anyhow::bail!(
            "KV capacity must fit at least two cache blocks, including the reserved block"
        );
    }
    Ok(blocks)
}

fn max_cache_blocks(
    upper: usize,
    budget: usize,
    allocation_bytes: impl Fn(usize) -> anyhow::Result<usize>,
) -> anyhow::Result<usize> {
    let mut low = 0;
    let mut high = upper;
    while low < high {
        let blocks = low + (high - low).div_ceil(2);
        if allocation_bytes(blocks)? <= budget {
            low = blocks;
        } else {
            high = blocks - 1;
        }
    }
    Ok(low)
}

fn cache_prefix_capacity(blocks: usize, block_size: usize, max_context: usize) -> usize {
    // The search can probe the reserved-only block before ruling out unusable caches.
    blocks
        .saturating_sub(1)
        .saturating_mul(block_size)
        .max(1)
        .min(max_context)
}

#[cfg(any(feature = "metal", test))]
fn kv_layer_buffer_bytes_per_block(
    config: &dyn ModelConfigLike,
    layer: usize,
    block_size: usize,
    dtype: DType,
) -> anyhow::Result<[usize; 2]> {
    if !config.layer_has_paged_kv_cache(layer) {
        return Ok([0, 0]);
    }
    let (heads, dimensions) = match config.kv_cache_layout_for_layer(layer) {
        KvCacheLayout::Mla {
            kv_lora_rank,
            kpe_head_dim,
        } => (1, [kv_lora_rank, kpe_head_dim]),
        _ => (
            config.num_kv_heads_for_layer(layer),
            [
                config.k_head_dim_for_layer(layer),
                config.v_head_dim_for_layer(layer),
            ],
        ),
    };
    let bytes = |dimension| {
        heads
            .checked_mul(dimension)
            .and_then(|elements| elements.checked_mul(block_size))
            .and_then(|elements| elements.checked_mul(dtype.size_in_bytes()))
            .ok_or_else(|| anyhow::anyhow!("KV buffer size overflow"))
    };
    Ok([bytes(dimensions[0])?, bytes(dimensions[1])?])
}

struct CacheDeviceAllocation<'a> {
    device: &'a Device,
    bytes_per_block: usize,
    #[cfg(feature = "metal")]
    buffer_bytes_per_block: Vec<usize>,
    #[cfg(feature = "metal")]
    minimum_buffer_bytes: usize,
}

impl CacheDeviceAllocation<'_> {
    fn allocation_bytes(&self, blocks: usize) -> anyhow::Result<usize> {
        #[cfg(feature = "metal")]
        if let Device::Metal(device) = self.device {
            return self
                .buffer_bytes_per_block
                .iter()
                .try_fold(0usize, |total, bytes| {
                    let logical = bytes
                        .checked_mul(blocks)
                        .ok_or_else(|| anyhow::anyhow!("KV allocation size overflow"))?
                        .max(self.minimum_buffer_bytes);
                    let allocated =
                        cache_engine::metal_kv_buffer_allocation_bytes(device, logical)?;
                    total
                        .checked_add(allocated)
                        .ok_or_else(|| anyhow::anyhow!("KV allocation size overflow"))
                });
        }
        self.bytes_per_block
            .checked_mul(blocks)
            .ok_or_else(|| anyhow::anyhow!("KV allocation size overflow"))
    }
}

fn cache_device_allocations<'a>(
    config: &dyn ModelConfigLike,
    cache_dtype: DType,
    block_size: usize,
    device: &'a Device,
    layer_devices: &'a [Option<Device>],
) -> anyhow::Result<Vec<CacheDeviceAllocation<'a>>> {
    let mut devices: Vec<CacheDeviceAllocation<'_>> = Vec::new();
    for layer in 0..config.num_layers() {
        let elements = config.layer_kv_cache_elements_per_token(layer);
        let bytes = elements
            .unwrap_or(0)
            .checked_mul(cache_dtype.size_in_bytes())
            .and_then(|bytes| bytes.checked_mul(block_size))
            .ok_or_else(|| anyhow::anyhow!("KV block size overflow"))?;
        let layer_device = layer_devices
            .get(layer)
            .and_then(Option::as_ref)
            .unwrap_or(device);
        let existing = devices
            .iter()
            .position(|allocation| allocation.device.same_device(layer_device));
        let index = if let Some(index) = existing {
            devices[index].bytes_per_block = devices[index]
                .bytes_per_block
                .checked_add(bytes)
                .ok_or_else(|| anyhow::anyhow!("KV block size overflow"))?;
            index
        } else {
            devices.push(CacheDeviceAllocation {
                device: layer_device,
                bytes_per_block: bytes,
                #[cfg(feature = "metal")]
                buffer_bytes_per_block: Vec::new(),
                #[cfg(feature = "metal")]
                minimum_buffer_bytes: cache_dtype.size_in_bytes(),
            });
            devices.len() - 1
        };
        #[cfg(feature = "metal")]
        if layer_device.is_metal() {
            devices[index]
                .buffer_bytes_per_block
                .extend(kv_layer_buffer_bytes_per_block(
                    config,
                    layer,
                    block_size,
                    cache_dtype,
                )?);
        }
        #[cfg(not(feature = "metal"))]
        let _ = index;
    }
    if !devices
        .iter()
        .any(|allocation| allocation.device.same_device(device))
    {
        devices.push(CacheDeviceAllocation {
            device,
            bytes_per_block: 0,
            #[cfg(feature = "metal")]
            buffer_bytes_per_block: Vec::new(),
            #[cfg(feature = "metal")]
            minimum_buffer_bytes: cache_dtype.size_in_bytes(),
        });
    }
    Ok(devices)
}

#[derive(Clone, Copy)]
pub struct CacheConfigRequest<'a> {
    pub memory_reservations: CacheMemoryReservations,
    pub dtype: DType,
    pub model: &'a dyn ModelConfigLike,
    pub device: &'a Device,
    pub layer_devices: &'a [Option<Device>],
    pub capacity_activation_estimate:
        &'a dyn Fn(usize) -> anyhow::Result<crate::pipeline::ActivationMemoryEstimate>,
    pub reserve_decode_graphs: bool,
    pub silent: bool,
}

pub(crate) fn automatic_runtime_allowance(
    enabled: bool,
    activation_bytes: usize,
    available_above_workload: usize,
) -> usize {
    if enabled {
        activation_bytes.min(available_above_workload)
    } else {
        0
    }
}

fn workload_cache_blocks(
    limits: ExecutionMemoryLimits,
    max_context: usize,
    block_size: usize,
) -> anyhow::Result<usize> {
    let max_tokens = max_context
        .checked_mul(limits.max_num_seqs)
        .ok_or_else(|| anyhow::anyhow!("workload KV token capacity overflow"))?;
    let tokens = limits
        .max_num_batched_tokens
        .max(limits.max_num_seqs)
        .min(max_tokens);
    let sequences = limits.max_num_seqs.min(tokens);
    let context_blocks = max_context
        .div_ceil(block_size)
        .checked_mul(sequences)
        .ok_or_else(|| anyhow::anyhow!("workload KV block capacity overflow"))?;
    // Each sequence can leave a partially occupied block.
    sequences
        .checked_add(tokens.saturating_sub(sequences) / block_size)
        .map(|blocks| blocks.min(context_blocks))
        .and_then(|blocks| blocks.checked_add(1))
        .map(|blocks| blocks.max(MIN_CACHE_BLOCKS))
        .ok_or_else(|| anyhow::anyhow!("workload KV block capacity overflow"))
}

fn total_cache_allocation_bytes(
    devices: &[CacheDeviceAllocation<'_>],
    blocks: usize,
) -> anyhow::Result<usize> {
    devices.iter().try_fold(0usize, |total, allocation| {
        total
            .checked_add(allocation.allocation_bytes(blocks)?)
            .ok_or_else(|| anyhow::anyhow!("KV allocation size overflow"))
    })
}

fn requested_device_cache_blocks(
    goal: MemoryGoal,
    block_size: usize,
    devices: &[CacheDeviceAllocation<'_>],
    total_bytes_per_block: usize,
) -> anyhow::Result<Option<usize>> {
    let blocks = requested_cache_blocks(goal, block_size, total_bytes_per_block)?;
    match (goal, blocks) {
        (MemoryGoal::KvBytes(bytes), Some(upper)) => {
            Ok(Some(max_cache_blocks(upper, bytes, |blocks| {
                total_cache_allocation_bytes(devices, blocks)
            })?))
        }
        (_, blocks) => Ok(blocks),
    }
}

pub(crate) fn optional_weight_reservation(
    paged_config: &PagedAttentionConfig,
    request: CacheConfigRequest<'_>,
    device: &Device,
) -> anyhow::Result<(usize, usize)> {
    let block_size = paged_config
        .block_size
        .unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE);
    let devices = cache_device_allocations(
        request.model,
        paged_config.cache_type.to_dtype(request.dtype),
        block_size,
        request.device,
        request.layer_devices,
    )?;
    let bytes_per_block = devices.iter().try_fold(0usize, |total, allocation| {
        total
            .checked_add(allocation.bytes_per_block)
            .ok_or_else(|| anyhow::anyhow!("KV block size overflow"))
    })?;
    let blocks = match requested_device_cache_blocks(
        paged_config.memory_goal,
        block_size,
        &devices,
        bytes_per_block,
    )? {
        Some(blocks) => blocks,
        None => workload_cache_blocks(
            paged_config.execution_limits,
            request.model.max_seq_len(),
            block_size,
        )?,
    };
    let cache_bytes = devices
        .iter()
        .find(|allocation| allocation.device.same_device(device))
        .map_or(Ok(0), |allocation| allocation.allocation_bytes(blocks))?;
    let capacity = cache_prefix_capacity(blocks, block_size, request.model.max_seq_len());
    let activation = (request.capacity_activation_estimate)(capacity)?;
    let primary = device.same_device(request.device);
    let (runtime, baseline_activation, activation, staging) = if primary {
        (
            request.memory_reservations.primary_device_bytes,
            paged_config.primary_activation_memory_reservation_bytes,
            activation.non_mapped_bytes.max(activation.mapped_bytes),
            request.memory_reservations.primary_staging_bytes,
        )
    } else {
        (
            request.memory_reservations.secondary_device_bytes,
            paged_config.mapped_activation_memory_reservation_bytes,
            activation.mapped_bytes,
            request.memory_reservations.secondary_staging_bytes,
        )
    };
    let pending_weights = if device.is_cuda() {
        mistralrs_quant::gguf_affine_budget_bytes(device, request.dtype)
    } else {
        0
    };
    let persistent = [
        runtime
            .saturating_sub(baseline_activation)
            .saturating_sub(staging),
        cache_bytes,
        pending_weights,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| total.checked_add(bytes))
    .ok_or_else(|| anyhow::anyhow!("optional weight runtime reservation overflow"))?;
    let transient = baseline_activation
        .max(activation)
        .checked_add(staging)
        .ok_or_else(|| anyhow::anyhow!("optional weight staging reservation overflow"))?;
    Ok((persistent, transient))
}

#[cfg(feature = "cuda")]
pub(crate) fn optional_weight_budget(
    paged_config: &PagedAttentionConfig,
    request: CacheConfigRequest<'_>,
    device: &Device,
) -> anyhow::Result<crate::memory_plan::WeightAllocationBudget> {
    device.synchronize()?;
    if device.is_cuda() {
        MemoryUsage.synchronize_cuda_context(device)?;
        trim_cuda_mempool(device)?;
    }
    let (persistent, transient) = optional_weight_reservation(paged_config, request, device)?;
    let memory = MemoryUsage.query(device)?;
    Ok(crate::memory_plan::WeightAllocationBudget {
        persistent_bytes: paged_config
            .allocation_budget_with_reservations(device, memory, persistent, transient)?,
        execution_bytes: paged_config.execution_budget(device, memory)?,
    })
}

pub fn calculate_cache_config(
    paged_config: &mut PagedAttentionConfig,
    request: CacheConfigRequest<'_>,
) -> anyhow::Result<CacheConfig> {
    let CacheConfigRequest {
        memory_reservations,
        dtype,
        model: config,
        device,
        layer_devices,
        capacity_activation_estimate,
        reserve_decode_graphs,
        silent,
    } = request;
    let goal = paged_config.memory_goal;
    goal.validate()?;
    let cache_type = paged_config.cache_type;
    let block_size = paged_config
        .block_size
        .unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE);
    if !SUPPORTED_BLOCK_SIZE.contains(&block_size) {
        anyhow::bail!("Block size must be in {SUPPORTED_BLOCK_SIZE:?}, got {block_size}");
    }
    cache_type
        .validate(dtype, config, device, layer_devices)
        .map_err(anyhow::Error::msg)?;
    let cache_dtype = cache_type.to_dtype(dtype);
    let devices = cache_device_allocations(config, cache_dtype, block_size, device, layer_devices)?;
    let total_bytes_per_block = devices
        .iter()
        .try_fold(0usize, |sum, allocation| {
            sum.checked_add(allocation.bytes_per_block)
        })
        .ok_or_else(|| anyhow::anyhow!("KV block size overflow"))?;
    if total_bytes_per_block == 0 {
        anyhow::bail!("The model has no paged KV cache storage");
    }
    let requested_blocks =
        requested_device_cache_blocks(goal, block_size, &devices, total_bytes_per_block)?;
    let workload_blocks = match requested_blocks {
        Some(blocks) => blocks,
        None => workload_cache_blocks(
            paged_config.execution_limits,
            config.max_seq_len(),
            block_size,
        )?,
    };
    let activation_for_blocks = |blocks: usize| {
        let tokens = cache_prefix_capacity(blocks, block_size, config.max_seq_len());
        capacity_activation_estimate(tokens)
    };
    let baseline_primary = paged_config.primary_activation_memory_reservation_bytes;
    let baseline_mapped = paged_config.mapped_activation_memory_reservation_bytes;
    let extra_activation = |blocks: usize, primary: bool| -> anyhow::Result<usize> {
        let estimate = activation_for_blocks(blocks)?;
        Ok(if primary {
            estimate
                .non_mapped_bytes
                .max(estimate.mapped_bytes)
                .saturating_sub(baseline_primary)
        } else {
            estimate.mapped_bytes.saturating_sub(baseline_mapped)
        })
    };
    let mut available_blocks = usize::MAX;
    let mut budgets = Vec::with_capacity(devices.len());
    for allocation in &devices {
        let cache_device = allocation.device;
        // Pool cleanup makes the snapshot and following persistent allocation agree.
        cache_device.synchronize()?;
        #[cfg(feature = "cuda")]
        if cache_device.is_cuda() {
            MemoryUsage.synchronize_cuda_context(cache_device)?;
            trim_cuda_mempool(cache_device)?;
        }
        let memory = MemoryUsage.query(cache_device)?;
        let budget = resolve_device_budget(paged_config.allocation_memory_goal(), memory)?;
        let primary = cache_device.same_device(device);
        let reserved = if primary {
            memory_reservations.primary_device_bytes
        } else {
            memory_reservations.secondary_device_bytes
        };
        let affine = if cache_device.is_cuda() {
            mistralrs_quant::gguf_affine_budget_bytes(cache_device, dtype)
        } else {
            0
        };
        let allocation_budget = paged_config.allocation_budget(cache_device, memory)?;
        let execution_budget = paged_config.execution_budget(cache_device, memory)?;
        let baseline_activation = if primary {
            baseline_primary
        } else {
            baseline_mapped
        };
        let staging = if primary {
            memory_reservations.primary_staging_bytes
        } else {
            memory_reservations.secondary_staging_bytes
        };
        let baseline_transient = baseline_activation
            .checked_add(staging)
            .ok_or_else(|| anyhow::anyhow!("cache staging reservation overflow"))?;
        let persistent = reserved
            .saturating_sub(baseline_transient)
            .checked_add(affine)
            .ok_or_else(|| anyhow::anyhow!("persistent cache reservation overflow"))?;
        let available = allocation_budget_with_reservations(
            allocation_budget,
            execution_budget,
            persistent,
            baseline_transient,
        )?;
        let workspace_credit = execution_budget
            .saturating_sub(persistent)
            .saturating_sub(baseline_transient)
            .saturating_sub(available);
        let runtime_allowance = automatic_runtime_allowance(
            reserve_decode_graphs && cache_device.is_cuda() && requested_blocks.is_none(),
            baseline_activation,
            available.saturating_sub(
                allocation
                    .allocation_bytes(workload_blocks)?
                    .checked_add(
                        extra_activation(workload_blocks, primary)?
                            .saturating_sub(workspace_credit),
                    )
                    .ok_or_else(|| {
                        anyhow::anyhow!("workload KV and activation reservation overflow")
                    })?,
            ),
        );
        let available = available - runtime_allowance;
        let upper = available
            .checked_div(allocation.bytes_per_block)
            .unwrap_or(usize::MAX);
        let blocks = max_cache_blocks(upper, available, |blocks| {
            allocation
                .allocation_bytes(blocks)?
                .checked_add(extra_activation(blocks, primary)?.saturating_sub(workspace_credit))
                .ok_or_else(|| anyhow::anyhow!("KV and activation reservation overflow"))
        })?;
        if requested_blocks.is_some_and(|requested| requested > blocks) {
            let requested = requested_blocks.unwrap();
            anyhow::bail!(
                "Memory goal ({goal}) needs {} MiB of KV cache and {} MiB of additional prefix workspace on {:?}, but only {} MiB remains after model and baseline runtime reservations ({} MiB allocated, {} MiB physically available, {} MiB policy budget, {} MiB runtime reservations, {} MiB pending weights)",
                allocation.allocation_bytes(requested)?.div_ceil(SIZE_IN_MB),
                extra_activation(requested, primary)?.div_ceil(SIZE_IN_MB),
                cache_device.location(), available / SIZE_IN_MB,
                memory.allocated() / SIZE_IN_MB,
                memory.physical_available() / SIZE_IN_MB,
                allocation_budget / SIZE_IN_MB,
                reserved.div_ceil(SIZE_IN_MB),
                affine.div_ceil(SIZE_IN_MB),
            );
        }
        available_blocks = available_blocks.min(blocks);
        budgets.push((
            budget,
            reserved.saturating_add(affine),
            available,
            memory.physical_available(),
            allocation_budget,
            workspace_credit,
            runtime_allowance,
        ));
    }
    let num_gpu_blocks = requested_blocks.unwrap_or(available_blocks);
    if num_gpu_blocks < MIN_CACHE_BLOCKS {
        anyhow::bail!("Insufficient memory for a usable KV cache after weights and runtime reservations. Reduce concurrency or prefill work, change device placement, or disable paged attention.");
    }
    let cache_bytes = num_gpu_blocks
        .checked_mul(total_bytes_per_block)
        .ok_or_else(|| anyhow::anyhow!("KV allocation size overflow"))?;
    let tokens = num_gpu_blocks
        .saturating_sub(1)
        .checked_mul(block_size)
        .ok_or_else(|| anyhow::anyhow!("KV token capacity overflow"))?;
    let final_activation = activation_for_blocks(num_gpu_blocks)?;
    if !silent {
        for (
            allocation,
            (
                budget,
                reserved,
                available,
                physical_available,
                allocation_budget,
                workspace_credit,
                runtime_allowance,
            ),
        ) in devices.iter().zip(budgets)
        {
            let extra = extra_activation(num_gpu_blocks, allocation.device.same_device(device))?;
            info!(
                "Memory plan {:?}: physical {} MiB, engine limit {} MiB, existing occupancy {} MiB, physically available {} MiB, live allocation budget {} MiB, model allocation budget {} MiB, runtime reservation {} MiB, runtime initialization allowance {} MiB, KV budget {} MiB ({goal})",
                allocation.device.location(), budget.physical_bytes / SIZE_IN_MB,
                budget.engine_limit_bytes / SIZE_IN_MB, budget.occupied_bytes / SIZE_IN_MB,
                physical_available / SIZE_IN_MB, budget.allocatable_bytes / SIZE_IN_MB,
                allocation_budget / SIZE_IN_MB,
                reserved.saturating_add(extra).div_ceil(SIZE_IN_MB), runtime_allowance / SIZE_IN_MB,
                available.saturating_sub(extra.saturating_sub(workspace_credit)) / SIZE_IN_MB,
            );
        }
        let allocated_bytes = total_cache_allocation_bytes(&devices, num_gpu_blocks)?;
        info!("Allocating {} MiB logical shared {cache_dtype:?} KV cache (estimated allocation {} MiB): {tokens} usable tokens, {num_gpu_blocks} blocks of {block_size} tokens", cache_bytes.div_ceil(SIZE_IN_MB), allocated_bytes.div_ceil(SIZE_IN_MB));
    }
    crate::pipeline::apply_model_activation_memory(paged_config, final_activation);
    Ok(CacheConfig {
        block_size,
        num_gpu_blocks,
        cache_type,
        kv_cache_group_ids: config.kv_cache_group_ids(),
    })
}
