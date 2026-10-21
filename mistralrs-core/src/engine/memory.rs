use candle_core::Result;
#[cfg(any(feature = "cuda", feature = "metal"))]
use candle_core::{DType, Device, DeviceLocation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PromptBatchMemoryAction {
    Proceed,
    Retain(usize),
    Reject,
}

pub(super) fn prompt_batch_memory_action(
    current: usize,
    transient_pressure: bool,
) -> PromptBatchMemoryAction {
    if !transient_pressure {
        PromptBatchMemoryAction::Proceed
    } else if current > 1 {
        PromptBatchMemoryAction::Retain(current.div_ceil(2))
    } else {
        PromptBatchMemoryAction::Reject
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(super) struct PromptMemoryStatus {
    pub(super) graph_pressure: bool,
    pub(super) transient_pressure: bool,
    pub(super) insufficient_total_capacity: bool,
    pub(super) maintenance_failed: bool,
}

#[cfg(any(feature = "cuda", feature = "metal"))]
pub(super) fn record_prompt_batch_reduction(previous: usize, retained: usize) {
    metrics::counter!("mistralrs_prompt_batch_reductions_total").increment(1);
    metrics::counter!("mistralrs_prompt_sequences_deferred_total")
        .increment(u64::try_from(previous.saturating_sub(retained)).unwrap_or(u64::MAX));
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ForwardWorkspaceBytes {
    pub(super) primary: usize,
    pub(super) mapped: usize,
}

impl ForwardWorkspaceBytes {
    pub(super) fn for_role(self, primary: bool) -> usize {
        if primary {
            self.primary
        } else {
            self.mapped
        }
    }

    #[cfg(any(feature = "cuda", test))]
    pub(super) fn component_max(self, other: Self) -> Self {
        Self {
            primary: self.primary.max(other.primary),
            mapped: self.mapped.max(other.mapped),
        }
    }

    pub(super) fn with_pending(self, primary: bool, pending: usize) -> Result<usize> {
        self.for_role(primary)
            .checked_add(pending)
            .ok_or_else(|| candle_core::Error::msg("forward pending-weight reservation overflow"))
    }
}

impl std::fmt::Display for ForwardWorkspaceBytes {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} bytes on the primary device and {} bytes on each mapped device",
            self.primary, self.mapped
        )
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ForwardMemoryShape {
    pub query_tokens: usize,
    pub sequences: usize,
    pub has_encoder_input: bool,
    pub extra_logit_tokens: usize,
}

pub(super) fn forward_query_tokens(query_lens: &[usize], packed: bool) -> Result<usize> {
    let tokens = if packed {
        query_lens
            .iter()
            .try_fold(0usize, |total, len| total.checked_add(*len))
    } else {
        query_lens
            .iter()
            .copied()
            .max()
            .unwrap_or_default()
            .checked_mul(query_lens.len())
    };
    tokens.ok_or_else(|| candle_core::Error::msg("forward query token count overflow"))
}

pub(super) fn extra_logit_tokens(
    query_tokens: usize,
    sequences: usize,
    is_prompt: bool,
    return_raw_logits: bool,
) -> usize {
    if !is_prompt || return_raw_logits {
        query_tokens.saturating_sub(sequences)
    } else {
        0
    }
}

pub(super) fn forward_workspace_bytes(
    plan: Option<crate::memory_plan::RuntimeMemoryPlan>,
    shape: ForwardMemoryShape,
    attention_bytes: usize,
    attention_is_complete: bool,
) -> Result<ForwardWorkspaceBytes> {
    let Some(plan) = plan else {
        return Ok(ForwardWorkspaceBytes {
            primary: attention_bytes,
            mapped: attention_bytes,
        });
    };
    let attention_is_complete = attention_is_complete
        && (!plan.custom_prefill_mask || plan.metal_prefill_mask_is_structured);
    let reserve = |primary| -> Result<usize> {
        let non_attention = plan
            .non_attention_bytes(primary, shape.query_tokens, shape.sequences)
            .map_err(candle_core::Error::wrap)?;
        let combined = non_attention
            .checked_add(attention_bytes)
            .ok_or_else(|| candle_core::Error::msg("forward workspace size overflow"))?;
        let text = if attention_is_complete {
            combined
        } else {
            plan.activation_bytes(primary, shape.query_tokens)
                .map_err(candle_core::Error::wrap)?
                .max(combined)
        };
        if primary && shape.has_encoder_input {
            let encoder = non_attention
                .checked_add(plan.encoder_activation_bytes)
                .ok_or_else(|| candle_core::Error::msg("encoder workspace size overflow"))?;
            Ok(text.max(encoder))
        } else {
            Ok(text)
        }
    };
    let extra_logits = shape
        .extra_logit_tokens
        .checked_mul(plan.logits_bytes_per_query_token)
        .ok_or_else(|| candle_core::Error::msg("logits workspace overflow"))?;
    Ok(ForwardWorkspaceBytes {
        primary: reserve(true)?
            .checked_add(extra_logits)
            .ok_or_else(|| candle_core::Error::msg("forward logits workspace overflow"))?,
        mapped: reserve(false)?,
    })
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
fn forward_available_bytes(budget: crate::memory_plan::DeviceMemoryBudget) -> usize {
    budget
        .engine_limit_bytes
        .saturating_sub(budget.occupied_bytes)
        .min(budget.available_bytes)
}

#[cfg(any(feature = "cuda", feature = "metal"))]
pub(super) fn enforce_memory_goal(
    devices: &[Device],
    primary: &Device,
    goal: crate::MemoryGoal,
    workspace: ForwardWorkspaceBytes,
    activation_dtype: DType,
) -> (PromptMemoryStatus, Vec<(DeviceLocation, usize)>) {
    let mut status = PromptMemoryStatus::default();
    let mut metal_allocation_limits = Vec::new();
    for device in devices.iter().filter(|device| !device.is_cpu()) {
        let capacity = || -> Result<_> {
            let snapshot = crate::MemoryUsage
                .query(device)
                .map_err(candle_core::Error::wrap)?;
            let budget = crate::memory_plan::resolve_device_budget(goal, snapshot)
                .map_err(candle_core::Error::wrap)?;
            let available = forward_available_bytes(budget);
            Ok((available, budget.engine_limit_bytes, snapshot.allocated()))
        };
        let result = (|| -> Result<_> {
            let pending = if device.is_cuda() {
                mistralrs_quant::gguf_affine_reserved_bytes(device, activation_dtype)
            } else {
                0
            };
            let bytes = workspace.with_pending(device.same_device(primary), pending)?;
            let mut budget = capacity()?;
            if bytes > budget.0 {
                #[cfg(feature = "cuda")]
                if device.is_cuda() {
                    crate::MemoryUsage
                        .synchronize_cuda_context(device)
                        .map_err(candle_core::Error::wrap)?;
                    crate::MemoryUsage
                        .trim_cuda_memory_pool(device, 0)
                        .map_err(candle_core::Error::wrap)?;
                }
                #[cfg(feature = "metal")]
                if device.is_metal() {
                    device.synchronize()?;
                }
                budget = capacity()?;
            }
            Ok((bytes, budget))
        })();
        match result {
            Ok((bytes, (available, total, allocated))) => {
                if bytes > available {
                    tracing::debug!(
                        device = ?device.location(),
                        required_bytes = bytes,
                        available_bytes = available,
                        allocated_bytes = allocated,
                        engine_limit_bytes = total,
                        "Forward workspace exceeds the device memory budget"
                    );
                }
                status.transient_pressure |= bytes > available;
                status.insufficient_total_capacity |= bytes > total;
                if device.is_metal() {
                    let limit = allocated.saturating_add(available);
                    metal_allocation_limits.push((device.location(), limit));
                }
            }
            Err(err) => {
                tracing::warn!("Memory preflight failed: {err}");
                status.transient_pressure = true;
                status.maintenance_failed = true;
            }
        }
    }
    (status, metal_allocation_limits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_prompt_batches_until_single_request() {
        assert_eq!(
            prompt_batch_memory_action(3, true),
            PromptBatchMemoryAction::Retain(2)
        );
        assert_eq!(
            prompt_batch_memory_action(2, true),
            PromptBatchMemoryAction::Retain(1)
        );
        assert_eq!(
            prompt_batch_memory_action(1, true),
            PromptBatchMemoryAction::Reject
        );
        assert_eq!(
            prompt_batch_memory_action(8, false),
            PromptBatchMemoryAction::Proceed
        );
    }

    #[test]
    fn allocated_kv_leaves_the_full_reserved_forward_budget_admissible() {
        const GIB: usize = 1024 * 1024 * 1024;
        let empty = crate::utils::memory_usage::DeviceMemory::Unified {
            total: 64 * GIB,
            available: 64 * GIB,
            allocated: 0,
            working_set_limit: 48 * GIB,
        };
        let target = crate::memory_plan::resolve_device_budget(crate::MemoryGoal::Auto, empty)
            .unwrap()
            .allocatable_bytes;
        let activation = 2 * GIB;
        let loaded = crate::utils::memory_usage::DeviceMemory::Unified {
            total: 64 * GIB,
            available: 64 * GIB - (target - activation),
            allocated: target - activation,
            working_set_limit: 48 * GIB,
        };
        let available = forward_available_bytes(
            crate::memory_plan::resolve_device_budget(crate::MemoryGoal::Auto, loaded).unwrap(),
        );
        assert_eq!(available, activation);
        assert_eq!(
            prompt_batch_memory_action(1, activation > available),
            PromptBatchMemoryAction::Proceed
        );
    }

    #[test]
    fn auto_forward_uses_planning_headroom_after_external_memory_growth() {
        const GIB: usize = 1024 * 1024 * 1024;
        const ACTIVATION: usize = 2 * GIB;
        let snapshot = |available| crate::utils::memory_usage::DeviceMemory::Unified {
            total: 64 * GIB,
            available,
            allocated: 46 * GIB,
            working_set_limit: 48 * GIB,
        };
        let before =
            crate::memory_plan::resolve_device_budget(crate::MemoryGoal::Auto, snapshot(18 * GIB))
                .unwrap();
        let after =
            crate::memory_plan::resolve_device_budget(crate::MemoryGoal::Auto, snapshot(17 * GIB))
                .unwrap();
        assert_eq!(before.allocatable_bytes, ACTIVATION);
        assert_eq!(after.allocatable_bytes, ACTIVATION);
        assert_eq!(forward_available_bytes(after), ACTIVATION);
        assert_eq!(
            prompt_batch_memory_action(1, ACTIVATION > forward_available_bytes(after)),
            PromptBatchMemoryAction::Proceed
        );
        let exhausted =
            crate::memory_plan::resolve_device_budget(crate::MemoryGoal::Auto, snapshot(GIB))
                .unwrap();
        assert_eq!(forward_available_bytes(exhausted), GIB);
        assert_eq!(
            prompt_batch_memory_action(1, ACTIVATION > forward_available_bytes(exhausted)),
            PromptBatchMemoryAction::Reject
        );
        let capped = crate::memory_plan::resolve_device_budget(
            crate::MemoryGoal::Utilization(0.5),
            snapshot(18 * GIB),
        )
        .unwrap();
        assert_eq!(forward_available_bytes(capped), 0);
    }

    #[test]
    fn discrete_forward_admission_keeps_the_planning_headroom() {
        const GIB: usize = 1024 * 1024 * 1024;
        for goal in [crate::MemoryGoal::Auto, crate::MemoryGoal::Utilization(0.9)] {
            for free in [2 * GIB, 8 * GIB, 64 * GIB] {
                let budget = crate::memory_plan::resolve_device_budget(
                    goal,
                    crate::utils::memory_usage::DeviceMemory::Discrete {
                        total: 64 * GIB,
                        free,
                    },
                )
                .unwrap();
                assert_eq!(forward_available_bytes(budget), budget.allocatable_bytes);
            }
        }
    }

    fn asymmetric_plan() -> crate::memory_plan::RuntimeMemoryPlan {
        crate::memory_plan::RuntimeMemoryPlan {
            memory_goal: Default::default(),
            logits_bytes_per_query_token: 0,
            limits: Default::default(),
            primary_activation_bytes: 100,
            mapped_activation_bytes: 80,
            primary_non_attention_bytes: 60,
            mapped_non_attention_bytes: 50,
            encoder_activation_bytes: 0,
            custom_prefill_mask: false,
            metal_prefill_mask_is_structured: false,
            planned_query_tokens: 10,
        }
    }

    fn text_shape(query_tokens: usize, extra_logit_tokens: usize) -> ForwardMemoryShape {
        ForwardMemoryShape {
            query_tokens,
            sequences: 1,
            extra_logit_tokens,
            ..Default::default()
        }
    }

    #[test]
    fn packed_forward_counts_real_tokens_and_keeps_padded_fallback() {
        let mut lengths = vec![1; 16];
        lengths[0] = 512;
        assert_eq!(forward_query_tokens(&lengths, true).unwrap(), 527);
        assert_eq!(forward_query_tokens(&lengths, false).unwrap(), 8192);
        assert!(forward_query_tokens(&[usize::MAX, 1], true).is_err());
        assert!(forward_query_tokens(&[usize::MAX, 1], false).is_err());
    }

    #[test]
    fn text_forward_does_not_inherit_encoder_or_full_cache_workspace() {
        const GIB: usize = 1024 * 1024 * 1024;
        let mut plan = asymmetric_plan();
        plan.primary_activation_bytes = 36 * GIB;
        plan.mapped_activation_bytes = 20 * GIB;
        plan.encoder_activation_bytes = 16 * GIB;
        plan.logits_bytes_per_query_token = 7;
        let shape = ForwardMemoryShape {
            query_tokens: 2,
            sequences: 2,
            ..Default::default()
        };
        let text = forward_workspace_bytes(Some(plan), shape, 20, true).unwrap();
        assert_eq!(
            text,
            ForwardWorkspaceBytes {
                primary: 46,
                mapped: 30
            }
        );
        let multimodal = forward_workspace_bytes(
            Some(plan),
            ForwardMemoryShape {
                has_encoder_input: true,
                ..shape
            },
            20,
            true,
        )
        .unwrap();
        assert_eq!(multimodal.primary, 16 * GIB + 26);
        assert_eq!(multimodal.mapped, text.mapped);
    }

    #[test]
    fn logits_follow_sequence_count_independently_of_query_tokens() {
        let mut plan = asymmetric_plan();
        plan.logits_bytes_per_query_token = 7;
        let shape = ForwardMemoryShape {
            query_tokens: 10,
            sequences: 4,
            ..Default::default()
        };
        assert_eq!(
            forward_workspace_bytes(Some(plan), shape, 20, true).unwrap(),
            ForwardWorkspaceBytes {
                primary: 108,
                mapped: 70
            }
        );
    }

    #[test]
    fn speculative_decode_reserves_logits_for_every_query_without_raw_logits() {
        let mut plan = asymmetric_plan();
        plan.logits_bytes_per_query_token = 7;
        for (is_prompt, return_raw_logits, expected_primary) in
            [(true, false, 108), (true, true, 150), (false, false, 150)]
        {
            let shape = ForwardMemoryShape {
                query_tokens: 10,
                sequences: 4,
                extra_logit_tokens: extra_logit_tokens(10, 4, is_prompt, return_raw_logits),
                ..Default::default()
            };
            assert_eq!(
                forward_workspace_bytes(Some(plan), shape, 20, true).unwrap(),
                ForwardWorkspaceBytes {
                    primary: expected_primary,
                    mapped: 70
                }
            );
        }
    }

    #[test]
    fn activation_reserve_replaces_overlapping_attention_per_device() {
        let plan = asymmetric_plan();
        for (tokens, attention, expected) in [
            (
                10,
                20,
                ForwardWorkspaceBytes {
                    primary: 100,
                    mapped: 80,
                },
            ),
            (
                10,
                70,
                ForwardWorkspaceBytes {
                    primary: 130,
                    mapped: 120,
                },
            ),
            (
                20,
                70,
                ForwardWorkspaceBytes {
                    primary: 200,
                    mapped: 170,
                },
            ),
        ] {
            assert_eq!(
                forward_workspace_bytes(Some(plan), text_shape(tokens, 0), attention, false)
                    .unwrap(),
                expected
            );
        }
        assert_eq!(
            forward_workspace_bytes(None, text_shape(10, 0), 70, false).unwrap(),
            ForwardWorkspaceBytes {
                primary: 70,
                mapped: 70
            }
        );
        assert!(forward_workspace_bytes(Some(plan), text_shape(10, 0), usize::MAX, false).is_err());
    }

    #[test]
    fn complete_attention_scales_text_workspace_with_the_scheduled_batch() {
        let mut plan = asymmetric_plan();
        plan.primary_activation_bytes = 10_000;
        plan.mapped_activation_bytes = 8_000;
        for tokens in [1, 10] {
            assert_eq!(
                forward_workspace_bytes(Some(plan), text_shape(tokens, 0), 20, true).unwrap(),
                ForwardWorkspaceBytes {
                    primary: 20 + tokens * 6,
                    mapped: 20 + tokens * 5
                }
            );
        }
        assert_eq!(
            forward_workspace_bytes(Some(plan), text_shape(20, 0), 20, true).unwrap(),
            ForwardWorkspaceBytes {
                primary: 140,
                mapped: 120
            }
        );
        assert_eq!(
            forward_workspace_bytes(Some(plan), text_shape(1, 0), 20, false).unwrap(),
            ForwardWorkspaceBytes {
                primary: 10_000,
                mapped: 8_000
            }
        );
        plan.logits_bytes_per_query_token = 7;
        assert_eq!(
            forward_workspace_bytes(Some(plan), text_shape(1, 8), 20, true).unwrap(),
            ForwardWorkspaceBytes {
                primary: 89,
                mapped: 25
            }
        );
        assert!(forward_workspace_bytes(Some(plan), text_shape(1, 0), usize::MAX, true).is_err());
    }

    #[test]
    fn arbitrary_custom_attention_preserves_additional_model_workspace() {
        let mut plan = asymmetric_plan();
        plan.primary_activation_bytes = 10_000;
        plan.mapped_activation_bytes = 8_000;
        plan.custom_prefill_mask = true;
        assert_eq!(
            forward_workspace_bytes(Some(plan), text_shape(1, 0), 20, true).unwrap(),
            ForwardWorkspaceBytes {
                primary: 10_000,
                mapped: 8_000
            }
        );
        assert_eq!(
            forward_workspace_bytes(Some(plan), text_shape(1, 0), 15_000, true).unwrap(),
            ForwardWorkspaceBytes {
                primary: 15_006,
                mapped: 15_005
            }
        );
        plan.metal_prefill_mask_is_structured = true;
        assert_eq!(
            forward_workspace_bytes(Some(plan), text_shape(1, 0), 20, true).unwrap(),
            ForwardWorkspaceBytes {
                primary: 26,
                mapped: 25
            }
        );
    }

    #[test]
    fn asymmetric_loaded_devices_admit_their_own_reservations() {
        let workspace =
            forward_workspace_bytes(Some(asymmetric_plan()), text_shape(10, 0), 20, false).unwrap();
        let fits = |devices: &[(bool, usize)]| {
            let pressure = devices.iter().any(|&(primary, free)| {
                let snapshot =
                    crate::utils::memory_usage::DeviceMemory::Discrete { total: 1000, free };
                let budget = crate::memory_plan::resolve_device_budget(
                    crate::MemoryGoal::Utilization(1.0),
                    snapshot,
                )
                .unwrap();
                workspace.for_role(primary) > forward_available_bytes(budget)
            });
            prompt_batch_memory_action(1, pressure)
        };
        assert_eq!(
            fits(&[(false, 80), (true, 100)]),
            PromptBatchMemoryAction::Proceed
        );
        assert_eq!(
            fits(&[(true, 100), (false, 79)]),
            PromptBatchMemoryAction::Reject
        );
        assert_eq!(
            fits(&[(true, 99), (false, 80)]),
            PromptBatchMemoryAction::Reject
        );
        assert_eq!(fits(&[(true, 100)]), PromptBatchMemoryAction::Proceed);
    }

    #[test]
    fn raw_logits_are_reserved_only_on_the_primary_device() {
        let mut plan = asymmetric_plan();
        plan.logits_bytes_per_query_token = 7;
        let workspace = forward_workspace_bytes(Some(plan), text_shape(10, 8), 20, false).unwrap();
        assert_eq!(
            workspace,
            ForwardWorkspaceBytes {
                primary: 156,
                mapped: 80
            }
        );
        assert!(
            forward_workspace_bytes(Some(plan), text_shape(10, usize::MAX), 20, false).is_err()
        );
        plan.logits_bytes_per_query_token = usize::MAX;
        assert!(forward_workspace_bytes(Some(plan), text_shape(10, 1), 20, false).is_err());
    }

    #[test]
    fn pending_weights_and_deferred_prompts_preserve_device_roles() {
        let first = ForwardWorkspaceBytes {
            primary: 100,
            mapped: 20,
        };
        let second = ForwardWorkspaceBytes {
            primary: 80,
            mapped: 30,
        };
        let deferred = first.component_max(second);
        assert_eq!(
            deferred,
            ForwardWorkspaceBytes {
                primary: 100,
                mapped: 30
            }
        );
        assert_eq!(deferred.with_pending(true, 7).unwrap(), 107);
        assert_eq!(deferred.with_pending(false, 3).unwrap(), 33);
        assert_eq!(deferred.with_pending(false, 0).unwrap(), 30);
        assert!(deferred.with_pending(false, usize::MAX).is_err());
    }
}
