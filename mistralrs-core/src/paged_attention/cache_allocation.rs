use anyhow::{bail, Context, Result};
use candle_core::Device;

use super::{
    cache_device_allocations, cache_prefix_capacity, calculate_cache_config, CacheConfig,
    CacheConfigRequest, CacheEngine, MemoryGoal, PagedAttentionConfig, SIZE_IN_MB,
};
use crate::memory_plan::resolve_device_budget;
use crate::utils::memory_usage::DeviceMemory;
use crate::MemoryUsage;

const MIN_CACHE_BLOCKS: usize = 2;

#[derive(Clone, Copy)]
struct AllocationSnapshot {
    memory: DeviceMemory,
    budget: usize,
}

fn allocation_snapshot(
    config: &PagedAttentionConfig,
    device: &Device,
) -> Result<AllocationSnapshot> {
    device.synchronize()?;
    #[cfg(feature = "cuda")]
    if device.is_cuda() {
        MemoryUsage.synchronize_cuda_context(device)?;
        super::trim_cuda_mempool(device)?;
    }
    let memory = MemoryUsage.query(device)?;
    Ok(AllocationSnapshot {
        memory,
        budget: config.allocation_budget(device, memory)?,
    })
}

fn allocation_deficit(
    before: AllocationSnapshot,
    after: AllocationSnapshot,
    required: usize,
    engine_limit: usize,
) -> usize {
    let allocated = after
        .memory
        .allocated()
        .saturating_sub(before.memory.allocated());
    // Remaining budgets saturate at zero; retain any amount already beyond the engine limit or quota.
    after
        .memory
        .allocated()
        .saturating_add(required)
        .saturating_sub(engine_limit)
        .max(
            allocated
                .saturating_add(required)
                .saturating_sub(before.budget),
        )
        .max(required.saturating_sub(after.budget))
        .max(required.saturating_sub(after.memory.available()))
}

fn adjusted_cache_blocks(
    goal: MemoryGoal,
    blocks: usize,
    deficits: &[(usize, usize)],
) -> Result<usize> {
    let mut reduction = 0;
    for &(deficit, bytes_per_block) in deficits {
        if deficit == 0 {
            continue;
        }
        if goal.is_capacity() {
            bail!(
                "Memory goal ({goal}) leaves {} MiB too little runtime headroom after allocating the KV cache",
                deficit.div_ceil(SIZE_IN_MB)
            );
        }
        if bytes_per_block == 0 {
            bail!(
                "Runtime reservations exceed available memory by {} MiB on a device without paged KV storage to reduce",
                deficit.div_ceil(SIZE_IN_MB)
            );
        }
        reduction = reduction.max(deficit.div_ceil(bytes_per_block));
    }
    let adjusted = blocks.saturating_sub(reduction);
    if adjusted < MIN_CACHE_BLOCKS {
        bail!(
            "Insufficient memory for a usable KV cache and runtime reservations after allocation"
        );
    }
    Ok(adjusted)
}

fn config_for_capacity(
    baseline: &PagedAttentionConfig,
    request: CacheConfigRequest<'_>,
    cache: &CacheConfig,
) -> Result<PagedAttentionConfig> {
    let mut config = baseline.clone();
    let tokens = cache_prefix_capacity(
        cache.num_gpu_blocks,
        cache.block_size,
        request.model.max_seq_len(),
    );
    crate::pipeline::apply_model_activation_memory(
        &mut config,
        (request.capacity_activation_estimate)(tokens)?,
    );
    Ok(config)
}

pub fn allocate_cache(
    paged_config: &mut PagedAttentionConfig,
    request: CacheConfigRequest<'_>,
) -> Result<(CacheConfig, CacheEngine)> {
    let baseline = paged_config.clone();
    let mut final_config = baseline.clone();
    let mut cache_config = calculate_cache_config(&mut final_config, request)?;
    let allocations = cache_device_allocations(
        request.model,
        cache_config.cache_type.to_dtype(request.dtype),
        cache_config.block_size,
        request.device,
        request.layer_devices,
    )?;
    let layer_devices = (0..request.model.num_layers())
        .map(|layer| request.layer_devices.get(layer).cloned().unwrap_or(None))
        .collect::<Vec<_>>();

    loop {
        let before = allocations
            .iter()
            .map(|allocation| allocation_snapshot(&baseline, allocation.device))
            .collect::<Result<Vec<_>>>()?;
        let cache_engine = CacheEngine::new(
            request.model,
            &cache_config,
            request.dtype,
            request.device,
            layer_devices.clone(),
        )?;
        let mut deficits = Vec::with_capacity(allocations.len());
        let mut headroom = Vec::with_capacity(allocations.len());
        for (allocation, before) in allocations.iter().zip(before) {
            let primary = allocation.device.same_device(request.device);
            let (reserved, activation, baseline_activation) = if primary {
                (
                    request.memory_reservations.primary_device_bytes,
                    final_config.primary_activation_memory_reservation_bytes,
                    baseline.primary_activation_memory_reservation_bytes,
                )
            } else {
                (
                    request.memory_reservations.secondary_device_bytes,
                    final_config.mapped_activation_memory_reservation_bytes,
                    baseline.mapped_activation_memory_reservation_bytes,
                )
            };
            let affine = if allocation.device.is_cuda() {
                mistralrs_quant::gguf_affine_budget_bytes(allocation.device, request.dtype)
            } else {
                0
            };
            let required = reserved
                .checked_add(activation.saturating_sub(baseline_activation))
                .and_then(|bytes| bytes.checked_add(affine))
                .context("KV runtime reservation size overflow")?;
            let after = allocation_snapshot(&baseline, allocation.device)?;
            let budget = resolve_device_budget(baseline.allocation_memory_goal(), after.memory)?;
            let deficit = allocation_deficit(before, after, required, budget.engine_limit_bytes);
            deficits.push((deficit, allocation.bytes_per_block));
            headroom.push((allocation.device.location(), after.budget, required));
        }
        let next_blocks = adjusted_cache_blocks(
            baseline.memory_goal(),
            cache_config.num_gpu_blocks,
            &deficits,
        )?;
        if next_blocks == cache_config.num_gpu_blocks {
            if !request.silent {
                for (device, budget, required) in headroom {
                    tracing::info!(
                        ?device,
                        blocks = next_blocks,
                        usable_tokens = next_blocks.saturating_sub(1) * cache_config.block_size,
                        runtime_budget_mib = budget / SIZE_IN_MB,
                        runtime_reservation_mib = required.div_ceil(SIZE_IN_MB),
                        "Validated KV cache allocation"
                    );
                }
            }
            *paged_config = final_config;
            return Ok((cache_config, cache_engine));
        }
        if !request.silent {
            tracing::info!(
                previous_blocks = cache_config.num_gpu_blocks,
                blocks = next_blocks,
                missing_runtime_mib = deficits
                    .iter()
                    .map(|(deficit, _)| deficit.div_ceil(SIZE_IN_MB))
                    .max()
                    .unwrap_or_default(),
                "Reducing automatic KV cache to preserve measured runtime headroom"
            );
        }
        drop(cache_engine);
        cache_config.num_gpu_blocks = next_blocks;
        final_config = config_for_capacity(&baseline, request, &cache_config)?;
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        CacheMemoryReservations, KvCacheLayout, ModelConfigMetadata, PagedCacheType,
    };
    use super::*;
    use candle_core::DType;

    fn model_config() -> ModelConfigMetadata {
        ModelConfigMetadata {
            max_seq_len: 4096,
            num_layers: 2,
            hidden_size: 512,
            num_kv_heads: 4,
            num_attn_heads: 4,
            sliding_window: None,
            k_head_dim: 128,
            v_head_dim: 128,
            kv_cache_layout: KvCacheLayout::Standard,
        }
    }

    fn snapshot(total: usize, occupied: usize, budget: usize) -> AllocationSnapshot {
        AllocationSnapshot {
            memory: DeviceMemory::Discrete {
                total,
                free: total - occupied,
            },
            budget,
        }
    }

    #[test]
    fn deficit_includes_occupancy_beyond_saturated_engine_budget() {
        let before = snapshot(1000, 600, 300);
        let after = snapshot(1000, 950, 0);
        assert_eq!(allocation_deficit(before, after, 40, 900), 90);
    }

    #[test]
    fn deficit_includes_consumption_beyond_model_quota() {
        let before = snapshot(1000, 400, 200);
        let after = snapshot(1000, 650, 0);
        assert_eq!(allocation_deficit(before, after, 40, 900), 90);
    }

    #[test]
    fn deficit_preserves_unified_physical_headroom() {
        let before = snapshot(1000, 400, 400);
        let after = AllocationSnapshot {
            memory: DeviceMemory::Unified {
                total: 1000,
                available: 20,
                allocated: 650,
                working_set_limit: 900,
            },
            budget: 150,
        };
        assert_eq!(allocation_deficit(before, after, 40, 900), 20);
    }

    #[test]
    fn allocation_that_leaves_exactly_the_reservation_fits() {
        let before = snapshot(1000, 400, 300);
        let after = snapshot(1000, 660, 40);
        assert_eq!(allocation_deficit(before, after, 40, 900), 0);
    }

    #[test]
    fn limiting_device_selects_shared_block_reduction() -> Result<()> {
        assert_eq!(
            adjusted_cache_blocks(MemoryGoal::Auto, 100, &[(21, 10), (41, 10)])?,
            95
        );
        assert_eq!(
            adjusted_cache_blocks(MemoryGoal::Utilization(0.9), 100, &[(21, 10), (41, 100)])?,
            97
        );
        assert_eq!(
            adjusted_cache_blocks(MemoryGoal::Auto, 100, &[(0, 0), (0, 10)])?,
            100
        );
        assert!(adjusted_cache_blocks(MemoryGoal::Auto, 3, &[(11, 10)]).is_err());
        assert!(adjusted_cache_blocks(MemoryGoal::Auto, 3, &[(1, 0)]).is_err());
        Ok(())
    }

    #[test]
    fn explicit_capacity_is_preserved_or_rejected() -> Result<()> {
        for goal in [MemoryGoal::KvBytes(1000), MemoryGoal::KvTokens(1000)] {
            assert_eq!(adjusted_cache_blocks(goal, 100, &[(0, 10)])?, 100);
            assert!(adjusted_cache_blocks(goal, 100, &[(1, 10)]).is_err());
        }
        Ok(())
    }

    #[test]
    fn resizing_recomputes_activation_from_the_original_baseline() -> Result<()> {
        const BASELINE: usize = 16;
        let mut baseline =
            PagedAttentionConfig::new(Some(32), MemoryGoal::Auto, PagedCacheType::Auto)?;
        baseline.reserve_activation_memory(BASELINE, BASELINE);
        let model = model_config();
        let request = CacheConfigRequest {
            memory_reservations: CacheMemoryReservations::default(),
            dtype: DType::BF16,
            model: &model,
            device: &Device::Cpu,
            layer_devices: &[],
            capacity_activation_estimate: &|tokens| {
                Ok(crate::pipeline::ActivationMemoryEstimate {
                    mapped_bytes: tokens,
                    non_mapped_bytes: tokens,
                    ..Default::default()
                })
            },
            silent: true,
        };
        let mut cache = CacheConfig {
            block_size: 32,
            num_gpu_blocks: 10,
            cache_type: PagedCacheType::Auto,
            kv_cache_group_ids: vec![0],
        };
        let initial = config_for_capacity(&baseline, request, &cache)?;
        cache.num_gpu_blocks = 5;
        let resized = config_for_capacity(&baseline, request, &cache)?;
        assert_eq!(initial.runtime_memory_plan().primary_activation_bytes, 288);
        assert_eq!(resized.runtime_memory_plan().primary_activation_bytes, 128);
        assert_eq!(
            baseline.runtime_memory_plan().primary_activation_bytes,
            BASELINE
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires exclusive CUDA allocator accounting"]
    fn cuda_cache_allocation_preserves_capacity_and_runtime_reservation() -> Result<()> {
        const TOKENS: usize = 65;
        const RESERVATION: usize = 4 * SIZE_IN_MB;
        let device = Device::new_cuda(0)?;
        let model = model_config();
        let mut paged = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::KvTokens(TOKENS),
            PagedCacheType::Auto,
        )?;
        let (config, engine) = allocate_cache(
            &mut paged,
            CacheConfigRequest {
                memory_reservations: CacheMemoryReservations::default(),
                dtype: DType::BF16,
                model: &model,
                device: &device,
                layer_devices: &[],
                capacity_activation_estimate: &|_| {
                    Ok(crate::pipeline::ActivationMemoryEstimate {
                        mapped_bytes: RESERVATION,
                        non_mapped_bytes: RESERVATION,
                        ..Default::default()
                    })
                },
                silent: true,
            },
        )?;
        assert_eq!(
            config.num_gpu_blocks,
            TOKENS.div_ceil(config.block_size) + 1
        );
        assert_eq!(engine.get_kv_cache().len(), model.num_layers);
        assert_eq!(
            paged.runtime_memory_plan().primary_activation_bytes,
            RESERVATION
        );
        assert!(allocation_snapshot(&paged, &device)?.budget >= RESERVATION);
        Ok(())
    }
}
