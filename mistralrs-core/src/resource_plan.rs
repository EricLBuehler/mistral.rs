#[path = "memory_budget.rs"]
mod memory_budget;
pub(crate) use memory_budget::ModelMemoryBudget;
#[cfg(any(feature = "cuda", test))]
pub use memory_budget::RuntimeMemoryReservation;

use std::sync::Arc;

use crate::{MemoryGoal, PagedAttentionConfig};
use memory_budget::MemoryBudgetGroup;

#[derive(Clone, Copy, Debug)]
pub enum PagedKvPolicy {
    FairContext,
}

#[derive(Clone, Debug)]
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
        .map(|model| model.max_num_seqs.max(1))
        .try_fold(0usize, |sum, weight| sum.checked_add(weight))
        .ok_or_else(|| anyhow::anyhow!("multi-model serving capacity overflow"))?
        .max(1);
    let group = Arc::new(MemoryBudgetGroup::new(shared_memory_goal(models)?));
    #[cfg(any(feature = "cuda", test))]
    if models.iter().any(|model| model.paged_attn.is_none()) {
        // Nonpaged members do not publish a runtime reserve that retained graphs can protect.
        group.disable_runtime_capture();
    }

    let paged_attn = models
        .iter()
        .map(|model| {
            model
                .paged_attn
                .clone()
                .map(|config| {
                    split_paged_config(
                        config,
                        model.max_num_seqs.max(1),
                        active_weight,
                        group.clone(),
                    )
                })
                .transpose()
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    Ok(PagedKvPlan { paged_attn })
}

fn shared_memory_goal(models: &[PagedKvModelRequest]) -> anyhow::Result<MemoryGoal> {
    let mut utilization = None;
    for config in models.iter().filter_map(|model| model.paged_attn.as_ref()) {
        config.memory_goal.validate()?;
        if let MemoryGoal::Utilization(value) = config.memory_goal {
            anyhow::ensure!(
                utilization.is_none_or(|previous| previous == value),
                "multi-model loading requires one shared GPU memory utilization target"
            );
            utilization = Some(value);
        }
    }
    Ok(utilization.map_or(MemoryGoal::Auto, MemoryGoal::Utilization))
}

fn split_paged_config(
    config: PagedAttentionConfig,
    model_weight: usize,
    active_weight: usize,
    group: Arc<MemoryBudgetGroup>,
) -> anyhow::Result<PagedAttentionConfig> {
    config.memory_goal.validate()?;
    let mut split = config;
    #[allow(clippy::cast_precision_loss)]
    let share = model_weight as f64 / active_weight as f64;
    split.memory_budget = Some(Arc::new(ModelMemoryBudget::new(group, share)));
    Ok(split)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryGoal, PagedCacheType};

    #[test]
    fn mixed_nonpaged_groups_keep_runtime_capture_disabled() -> anyhow::Result<()> {
        let config = PagedAttentionConfig::new(Some(32), MemoryGoal::Auto, PagedCacheType::Auto)?;
        let plan = plan_paged_kv(
            &[
                PagedKvModelRequest {
                    paged_attn: Some(config),
                    max_num_seqs: 1,
                },
                PagedKvModelRequest {
                    paged_attn: None,
                    max_num_seqs: 1,
                },
            ],
            RuntimeResourcePlanOptions::default(),
        )?;
        let runtime = plan.paged_attn[0]
            .as_ref()
            .unwrap()
            .memory_budget
            .as_ref()
            .unwrap()
            .reserve_runtime(std::collections::HashMap::new());
        assert!(!runtime.capture_ready());
        Ok(())
    }

    #[test]
    fn split_preserves_base_device_memory_reservation() -> anyhow::Result<()> {
        let mut config = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::Utilization(0.9),
            PagedCacheType::Auto,
        )?
        .with_serving_capacity(16)?
        .with_base_device_memory_reservation(4 * 1024 * 1024 * 1024)?
        .with_recurrent_checkpoint_lanes(8)?
        .with_recurrent_prefix_capacity(16);
        config.recurrent_checkpoint_lanes_auto = true;
        config.reserve_activation_memory(512 * 1024 * 1024, 256 * 1024 * 1024);

        let split = split_paged_config(config, 1, 2, Arc::default())?;
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
        assert!(split.recurrent_checkpoint_lanes_auto);
        assert_eq!(split.recurrent_prefix_capacity, 16);
        assert!(split.memory_budget.is_some());
        assert!(matches!(
            split.memory_goal,
            MemoryGoal::Utilization(value) if value == 0.9
        ));
        Ok(())
    }

    #[test]
    fn single_model_plan_keeps_runtime_utilization_resolution() -> anyhow::Result<()> {
        let config = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::Utilization(0.9),
            PagedCacheType::Auto,
        )?;

        let split = split_paged_config(config, 16, 16, Arc::default())?;
        assert!(split.memory_budget.is_some());
        assert!(matches!(
            split.memory_goal,
            MemoryGoal::Utilization(value) if value == 0.9
        ));
        Ok(())
    }

    #[test]
    fn explicit_capacity_is_not_silently_shared() -> anyhow::Result<()> {
        let config =
            PagedAttentionConfig::new(None, MemoryGoal::KvBytes(1024), PagedCacheType::Auto)?;
        let split = split_paged_config(config, 1, 2, Arc::default())?;
        assert_eq!(split.memory_goal, MemoryGoal::KvBytes(1024));
        assert!(split.memory_budget.is_some());
        Ok(())
    }
    #[test]
    fn conflicting_utilization_targets_are_rejected() -> anyhow::Result<()> {
        let request = |goal| -> anyhow::Result<_> {
            Ok(PagedKvModelRequest {
                paged_attn: Some(PagedAttentionConfig::new(None, goal, PagedCacheType::Auto)?),
                max_num_seqs: 1,
            })
        };
        let models = [
            request(MemoryGoal::Utilization(0.5))?,
            request(MemoryGoal::Utilization(0.9))?,
        ];
        assert!(plan_paged_kv(&models, Default::default()).is_err());
        Ok(())
    }

    #[test]
    fn changing_a_planned_goal_requires_replanning_the_group() -> anyhow::Result<()> {
        let goal = MemoryGoal::Utilization(0.5);
        let plan = plan_paged_kv(
            &[PagedKvModelRequest {
                paged_attn: Some(PagedAttentionConfig::new(None, goal, PagedCacheType::Auto)?),
                max_num_seqs: 1,
            }],
            Default::default(),
        )?;
        let config = plan.paged_attn[0].as_ref().unwrap();
        assert_eq!(config.clone().with_memory_goal(goal)?.memory_goal(), goal);
        for changed in [
            MemoryGoal::Auto,
            MemoryGoal::Utilization(0.9),
            MemoryGoal::KvTokens(1024),
        ] {
            assert!(config.clone().with_memory_goal(changed).is_err());
        }
        Ok(())
    }

    #[test]
    fn explicit_utilization_is_shared_with_auto_and_capacity_models() -> anyhow::Result<()> {
        let models = [
            MemoryGoal::Auto,
            MemoryGoal::Utilization(0.5),
            MemoryGoal::KvTokens(1024),
        ]
        .into_iter()
        .map(|goal| {
            Ok(PagedKvModelRequest {
                paged_attn: Some(PagedAttentionConfig::new(None, goal, PagedCacheType::Auto)?),
                max_num_seqs: 1,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
        let plan = plan_paged_kv(&models, Default::default())?;
        for config in plan.paged_attn.iter().flatten() {
            assert_eq!(
                config.memory_budget.as_ref().unwrap().memory_goal(),
                MemoryGoal::Utilization(0.5)
            );
        }
        assert_eq!(
            plan.paged_attn[2].as_ref().unwrap().memory_goal(),
            MemoryGoal::KvTokens(1024)
        );
        Ok(())
    }

    #[test]
    fn dense_models_leave_a_share_unallocated_by_paged_models() -> anyhow::Result<()> {
        const GIB: usize = 1024 * 1024 * 1024;
        let plan = plan_paged_kv(
            &[
                PagedKvModelRequest {
                    paged_attn: Some(PagedAttentionConfig::new(
                        None,
                        MemoryGoal::Auto,
                        PagedCacheType::Auto,
                    )?),
                    max_num_seqs: 1,
                },
                PagedKvModelRequest {
                    paged_attn: None,
                    max_num_seqs: 3,
                },
            ],
            Default::default(),
        )?;
        let config = plan.paged_attn[0].as_ref().unwrap();
        let available = config.memory_budget.as_ref().unwrap().allocation_budget(
            candle_core::DeviceLocation::Cuda { gpu_id: 0 },
            crate::utils::memory_usage::DeviceMemory::Discrete {
                total: 100 * GIB,
                free: 100 * GIB,
            },
        )?;
        assert_eq!(available, 95 * GIB / 4);
        assert!(plan.paged_attn[1].is_none());
        Ok(())
    }
    #[test]
    fn total_serving_capacity_overflow_is_rejected() {
        let models = [
            PagedKvModelRequest {
                paged_attn: None,
                max_num_seqs: usize::MAX,
            },
            PagedKvModelRequest {
                paged_attn: None,
                max_num_seqs: 1,
            },
        ];
        assert!(plan_paged_kv(&models, Default::default()).is_err());
    }
}
