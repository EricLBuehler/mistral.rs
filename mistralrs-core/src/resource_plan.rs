use anyhow::Context;
use num_traits::ToPrimitive;

use crate::{MemoryGpuConfig, PagedAttentionConfig};

#[derive(Clone, Copy, Debug)]
pub enum PagedKvPolicy {
    FairContext,
}

#[derive(Clone, Copy, Debug)]
pub struct PagedKvModelRequest {
    pub paged_attn: Option<PagedAttentionConfig>,
    pub max_num_seqs: usize,
}

#[derive(Clone, Debug)]
pub struct PagedKvPlan {
    pub paged_attn: Vec<Option<PagedAttentionConfig>>,
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeResourcePlanOptions {
    pub paged_kv_policy: PagedKvPolicy,
}

impl Default for RuntimeResourcePlanOptions {
    fn default() -> Self {
        Self {
            paged_kv_policy: PagedKvPolicy::FairContext,
        }
    }
}

pub fn plan_paged_kv(
    models: &[PagedKvModelRequest],
    options: RuntimeResourcePlanOptions,
) -> anyhow::Result<PagedKvPlan> {
    match options.paged_kv_policy {
        PagedKvPolicy::FairContext => plan_fair_context_paged_kv(models),
    }
}

fn plan_fair_context_paged_kv(models: &[PagedKvModelRequest]) -> anyhow::Result<PagedKvPlan> {
    let active_weight = models
        .iter()
        .filter(|model| model.paged_attn.is_some())
        .map(|model| model.max_num_seqs.max(1))
        .sum::<usize>()
        .max(1);

    let paged_attn = models
        .iter()
        .map(|model| {
            model
                .paged_attn
                .map(|config| split_paged_config(config, model.max_num_seqs.max(1), active_weight))
                .transpose()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(PagedKvPlan { paged_attn })
}

fn split_paged_config(
    config: PagedAttentionConfig,
    model_weight: usize,
    active_weight: usize,
) -> anyhow::Result<PagedAttentionConfig> {
    let share = |value: usize| {
        value
            .saturating_mul(model_weight)
            .div_ceil(active_weight)
            .max(1)
    };
    let model_weight_f32 = model_weight
        .to_f32()
        .context("model weight cannot be represented as f32")?;
    let active_weight_f32 = active_weight
        .to_f32()
        .context("active weight cannot be represented as f32")?;
    let share_f32 = |value: f32| value * model_weight_f32 / active_weight_f32;

    let mem_gpu = match config.mem_gpu {
        MemoryGpuConfig::MbAmount(mb) => MemoryGpuConfig::BestEffortMbAmount {
            target_mb: share(mb),
            min_mb: None,
        },
        MemoryGpuConfig::BestEffortMbAmount { target_mb, min_mb } => {
            MemoryGpuConfig::BestEffortMbAmount {
                target_mb: share(target_mb),
                min_mb: min_mb.map(share),
            }
        }
        MemoryGpuConfig::Utilization(f) => MemoryGpuConfig::Utilization(share_f32(f)),
        MemoryGpuConfig::ContextSize(tokens) => MemoryGpuConfig::ContextSize(share(tokens)),
    };

    let mut split = PagedAttentionConfig::new(config.block_size, mem_gpu, config.cache_type)?;
    split.serving_capacity = config.serving_capacity;
    split.base_device_memory_reservation_bytes = config.base_device_memory_reservation_bytes;
    split.primary_activation_memory_reservation_bytes =
        config.primary_activation_memory_reservation_bytes;
    split.mapped_activation_memory_reservation_bytes =
        config.mapped_activation_memory_reservation_bytes;
    split.recurrent_checkpoint_lanes = config.recurrent_checkpoint_lanes;
    split.resolve_memory_utilization_after_load =
        config.resolve_memory_utilization_after_load && model_weight == active_weight;
    Ok(split)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PagedCacheType;

    #[test]
    fn split_preserves_base_device_memory_reservation() -> anyhow::Result<()> {
        let mut config = PagedAttentionConfig::new(
            Some(32),
            MemoryGpuConfig::Utilization(0.9),
            PagedCacheType::Auto,
        )?
        .with_serving_capacity(16)?
        .with_base_device_memory_reservation(4 * 1024 * 1024 * 1024)?
        .with_recurrent_checkpoint_lanes(8)?;
        config.reserve_activation_memory(512 * 1024 * 1024, 256 * 1024 * 1024);

        let split = split_paged_config(config, 1, 2)?;
        assert_eq!(split.serving_capacity, Some(16));
        assert_eq!(
            split.base_device_memory_reservation_bytes,
            4 * 1024 * 1024 * 1024
        );
        assert_eq!(
            split.primary_activation_memory_reservation_bytes,
            512 * 1024 * 1024
        );
        assert_eq!(
            split.mapped_activation_memory_reservation_bytes,
            256 * 1024 * 1024
        );
        assert_eq!(split.recurrent_checkpoint_lanes, 8);
        assert!(!split.resolve_memory_utilization_after_load);
        assert!(matches!(
            split.mem_gpu,
            MemoryGpuConfig::Utilization(value) if value == 0.45
        ));
        Ok(())
    }

    #[test]
    fn single_model_plan_keeps_runtime_utilization_resolution() -> anyhow::Result<()> {
        let config = PagedAttentionConfig::new(
            Some(32),
            MemoryGpuConfig::Utilization(0.9),
            PagedCacheType::Auto,
        )?;

        let split = split_paged_config(config, 16, 16)?;
        assert!(split.resolve_memory_utilization_after_load);
        assert!(matches!(
            split.mem_gpu,
            MemoryGpuConfig::Utilization(value) if value == 0.9
        ));
        Ok(())
    }
}
