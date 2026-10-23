use anyhow::{Context, Result};
#[cfg(any(target_family = "unix", test))]
use std::mem::size_of;

use crate::{
    memory_plan::{DeviceMemoryBudget, RuntimeMemoryPlan},
    utils::memory_usage::DeviceMemory,
};

#[cfg(any(target_family = "unix", test))]
const FA3_SCHEDULER_BATCH_ALIGNMENT: usize = 4;

pub(super) struct CaptureMemoryShape {
    pub query_tokens: usize,
    pub input_bytes: usize,
    pub attention_bytes: usize,
    pub attention_is_complete: bool,
}

pub(super) fn capture_workspace_bytes(
    plan: RuntimeMemoryPlan,
    primary: bool,
    shape: &CaptureMemoryShape,
) -> Result<usize> {
    let (baseline, non_attention) = if primary {
        (
            plan.primary_activation_bytes,
            plan.primary_non_attention_bytes,
        )
    } else {
        (
            plan.mapped_activation_bytes,
            plan.mapped_non_attention_bytes,
        )
    };
    let attention = if shape.attention_is_complete {
        shape.attention_bytes
    } else {
        baseline
            .saturating_sub(non_attention)
            .max(shape.attention_bytes)
    };
    plan.non_attention_bytes(primary, shape.query_tokens, shape.query_tokens)?
        .checked_add(attention)
        .and_then(|bytes| bytes.checked_add(shape.input_bytes))
        .context("CUDA capture workspace size overflow")
}

#[derive(Clone, Copy)]
#[cfg(any(target_family = "unix", test))]
pub(super) struct Fa3CaptureMemoryShape {
    pub batch: usize,
    pub query_len: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub splits: usize,
    pub pages_per_sequence: usize,
}

#[cfg(any(target_family = "unix", test))]
pub(super) fn fa3_capture_pool_bytes(shape: Fa3CaptureMemoryShape) -> Result<usize> {
    let product = |parts: &[usize]| {
        parts.iter().try_fold(1usize, |total, part| {
            total
                .checked_mul(*part)
                .context("FA3 capture size overflow")
        })
    };
    let rows = product(&[shape.batch, shape.query_len])?;
    let rounded_batch = shape
        .batch
        .div_ceil(FA3_SCHEDULER_BATCH_ALIGNMENT)
        .checked_mul(FA3_SCHEDULER_BATCH_ALIGNMENT)
        .context("FA3 capture batch size overflow")?;
    let scheduler_rows = product(&[2 + usize::from(shape.query_len > 1), rounded_batch])?
        .checked_add(1)
        .context("FA3 capture scheduler size overflow")?;
    let cumulative_rows = shape
        .batch
        .checked_add(1)
        .context("FA3 capture rows overflow")?;
    [
        product(&[rows, shape.heads, shape.head_dim])?,
        product(&[
            shape.splits,
            rows,
            shape.heads,
            shape.head_dim,
            size_of::<f32>(),
        ])?,
        product(&[shape.splits, rows, shape.heads, size_of::<f32>()])?,
        product(&[rows, shape.heads, size_of::<f32>()])?,
        product(&[shape.batch, shape.pages_per_sequence, size_of::<i32>()])?,
        product(&[scheduler_rows, size_of::<i32>()])?,
        product(&[cumulative_rows, size_of::<i32>()])?,
        product(&[shape.batch, size_of::<i32>()])?,
    ]
    .into_iter()
    .try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .context("FA3 capture pool size overflow")
    })
}

pub(super) fn capture_memory_available(
    plan: RuntimeMemoryPlan,
    memory: DeviceMemory,
    workspace: usize,
    pending_weights: usize,
    other_workspace: usize,
) -> Result<bool> {
    memory_available(
        crate::memory_plan::resolve_device_budget(plan.memory_goal, memory)?,
        workspace,
        pending_weights,
        other_workspace,
    )
}

pub(super) fn eager_memory_available(
    plan: RuntimeMemoryPlan,
    memory: DeviceMemory,
    workspace: usize,
    pending_weights: usize,
    other_workspace: usize,
) -> Result<bool> {
    memory_available(
        crate::memory_plan::resolve_forward_budget(plan.memory_goal, memory)?,
        workspace,
        pending_weights,
        other_workspace,
    )
}

fn memory_available(
    budget: DeviceMemoryBudget,
    workspace: usize,
    pending_weights: usize,
    other_workspace: usize,
) -> Result<bool> {
    let required = workspace
        .checked_add(pending_weights)
        .and_then(|bytes| bytes.checked_add(other_workspace))
        .context("CUDA capture pending memory size overflow")?;
    Ok(budget.occupied_bytes <= budget.engine_limit_bytes && required <= budget.allocatable_bytes)
}

pub(super) fn capture_attention_bytes(
    fa3_pool: usize,
    flashinfer_pool: usize,
    transient: usize,
) -> Result<usize> {
    // Graph-owned pools stay live while layers with another backend execute.
    fa3_pool
        .checked_add(flashinfer_pool)
        .and_then(|bytes| bytes.checked_add(transient))
        .context("CUDA capture attention size overflow")
}

pub(super) fn partitioned_attention_bytes(
    rows: usize,
    heads: usize,
    head_dim: usize,
    partitions: usize,
    dtype_bytes: usize,
) -> Result<usize> {
    let values = head_dim
        .checked_mul(dtype_bytes)
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<f32>()))
        .context("CUDA attention partition size overflow")?;
    rows.checked_mul(heads)
        .and_then(|elements| elements.checked_mul(partitions))
        .and_then(|elements| elements.checked_mul(values))
        .context("CUDA attention workspace size overflow")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryGoal;

    const MIB: usize = 1024 * 1024;
    const GIB: usize = 1024 * MIB;

    fn plan() -> RuntimeMemoryPlan {
        RuntimeMemoryPlan {
            memory_goal: MemoryGoal::Utilization(0.8),
            limits: Default::default(),
            primary_activation_bytes: 10 * GIB,
            mapped_activation_bytes: 8 * GIB,
            primary_non_attention_bytes: 8 * GIB,
            mapped_non_attention_bytes: 6 * GIB,
            encoder_activation_bytes: 16 * GIB,
            planned_query_tokens: 4096,
            logits_bytes_per_query_token: 1024 * 1024,
            custom_prefill_mask: false,
            metal_prefill_mask_is_structured: false,
        }
    }

    #[test]
    fn complete_capture_uses_decode_workspace_and_excludes_encoder_reservations() -> Result<()> {
        let shape = CaptureMemoryShape {
            query_tokens: 8,
            input_bytes: 32,
            attention_bytes: GIB,
            attention_is_complete: true,
        };
        assert_eq!(
            capture_workspace_bytes(plan(), true, &shape)?,
            GIB + 24 * 1024 * 1024 + 32
        );
        let mut larger_prefill = plan();
        larger_prefill.primary_activation_bytes = 100 * GIB;
        larger_prefill.encoder_activation_bytes = 64 * GIB;
        assert_eq!(
            capture_workspace_bytes(larger_prefill, true, &shape)?,
            capture_workspace_bytes(plan(), true, &shape)?
        );
        let shape = CaptureMemoryShape {
            attention_is_complete: false,
            ..shape
        };
        assert_eq!(
            capture_workspace_bytes(plan(), true, &shape)?,
            2 * GIB + 24 * 1024 * 1024 + 32
        );
        Ok(())
    }

    #[test]
    fn capture_adds_all_query_logits_to_text_activations() -> Result<()> {
        let mut plan = plan();
        plan.primary_non_attention_bytes = GIB;
        plan.primary_activation_bytes = GIB;
        let shape = CaptureMemoryShape {
            query_tokens: 8,
            input_bytes: 0,
            attention_bytes: 0,
            attention_is_complete: true,
        };
        assert_eq!(
            capture_workspace_bytes(plan, true, &shape)?,
            10 * 1024 * 1024
        );
        Ok(())
    }

    #[test]
    fn capture_reserves_logits_only_on_the_primary_device() -> Result<()> {
        let mut plan = plan();
        plan.primary_activation_bytes = GIB;
        plan.primary_non_attention_bytes = GIB;
        plan.mapped_activation_bytes = GIB;
        plan.mapped_non_attention_bytes = GIB;
        let shape = CaptureMemoryShape {
            query_tokens: 8,
            input_bytes: 32,
            attention_bytes: 1024,
            attention_is_complete: true,
        };
        assert_eq!(
            capture_workspace_bytes(plan, true, &shape)?,
            10 * 1024 * 1024 + shape.input_bytes + shape.attention_bytes
        );
        assert_eq!(
            capture_workspace_bytes(plan, false, &shape)?,
            2 * 1024 * 1024 + shape.input_bytes + shape.attention_bytes
        );
        plan.logits_bytes_per_query_token = usize::MAX;
        assert!(capture_workspace_bytes(plan, false, &shape).is_ok());
        assert!(capture_workspace_bytes(plan, true, &shape).is_err());
        Ok(())
    }

    #[test]
    fn pending_weights_become_occupancy_without_double_counting() -> Result<()> {
        let memory = DeviceMemory::Discrete {
            total: 100 * GIB,
            free: 30 * GIB,
        };
        assert!(capture_memory_available(
            plan(),
            memory,
            6 * GIB,
            3 * GIB,
            0
        )?);
        assert!(!capture_memory_available(
            plan(),
            memory,
            8 * GIB,
            3 * GIB,
            0
        )?);
        let materialized = DeviceMemory::Discrete {
            total: 100 * GIB,
            free: 27 * GIB,
        };
        assert!(capture_memory_available(
            plan(),
            materialized,
            6 * GIB,
            0,
            0
        )?);
        assert!(!capture_memory_available(
            plan(),
            materialized,
            8 * GIB,
            0,
            0
        )?);
        let over_limit = DeviceMemory::Discrete {
            total: 100 * GIB,
            free: 19 * GIB,
        };
        assert!(!capture_memory_available(plan(), over_limit, 0, 0, 0)?);
        Ok(())
    }

    #[test]
    fn capture_and_retention_preserve_other_models_workspace() -> Result<()> {
        let memory = DeviceMemory::Discrete {
            total: 100 * GIB,
            free: 26 * GIB,
        };
        assert!(capture_memory_available(plan(), memory, 3 * GIB, 0, 0)?);
        assert!(!capture_memory_available(
            plan(),
            memory,
            3 * GIB,
            0,
            4 * GIB
        )?);
        let retained = DeviceMemory::Discrete {
            total: 100 * GIB,
            free: 23 * GIB,
        };
        assert!(!capture_memory_available(plan(), retained, 0, 0, 4 * GIB)?);
        let within_own_reserve = DeviceMemory::Discrete {
            total: 100 * GIB,
            free: 24 * GIB,
        };
        assert!(capture_memory_available(
            plan(),
            within_own_reserve,
            0,
            0,
            4 * GIB
        )?);
        assert!(capture_memory_available(plan(), memory, 1, 0, usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn unified_auto_capture_preserves_system_reserve_and_pending_allocations() -> Result<()> {
        let plan = RuntimeMemoryPlan {
            memory_goal: MemoryGoal::Auto,
            ..plan()
        };
        let memory = DeviceMemory::Unified {
            total: 128 * GIB,
            available: 18 * GIB,
            allocated: 110 * GIB,
            working_set_limit: 128 * GIB,
        };
        for (workspace, pending_weights, other_workspace, expected) in [
            (GIB, 0, 0, true),
            (2 * GIB, 0, 0, true),
            (2 * GIB + 1, 0, 0, false),
            (GIB, GIB, 0, true),
            (GIB + 1, GIB, 0, false),
            (GIB, 0, GIB, true),
            (GIB + 1, 0, GIB, false),
        ] {
            assert_eq!(
                capture_memory_available(
                    plan,
                    memory,
                    workspace,
                    pending_weights,
                    other_workspace,
                )?,
                expected,
            );
        }
        Ok(())
    }

    #[test]
    fn unified_auto_capture_and_retention_track_available_memory_and_occupancy() -> Result<()> {
        let plan = RuntimeMemoryPlan {
            memory_goal: MemoryGoal::Auto,
            ..plan()
        };
        for (available, allocated, workspace, expected) in [
            (17 * GIB, 111 * GIB, GIB, true),
            (17 * GIB, 111 * GIB, GIB + 1, false),
            (16 * GIB, 112 * GIB, 0, true),
            (16 * GIB, 112 * GIB, 1, false),
            (15 * GIB, 113 * GIB, 0, false),
            (GIB, 100 * GIB, GIB, true),
            (GIB, 100 * GIB, GIB + 1, false),
        ] {
            let memory = DeviceMemory::Unified {
                total: 128 * GIB,
                available,
                allocated,
                working_set_limit: 128 * GIB,
            };
            assert_eq!(
                capture_memory_available(plan, memory, workspace, 0, 0)?,
                expected,
            );
        }
        Ok(())
    }

    #[test]
    fn eager_retry_can_use_auto_headroom_without_retaining_a_graph() -> Result<()> {
        for memory in [
            DeviceMemory::Unified {
                total: 128 * GIB,
                available: 15 * GIB,
                allocated: 113 * GIB,
                working_set_limit: 128 * GIB,
            },
            DeviceMemory::Discrete {
                total: 24 * GIB,
                free: GIB,
            },
        ] {
            for memory_goal in [
                MemoryGoal::Auto,
                MemoryGoal::KvBytes(4 * GIB),
                MemoryGoal::KvTokens(1024),
            ] {
                let plan = RuntimeMemoryPlan {
                    memory_goal,
                    ..plan()
                };
                assert!(!capture_memory_available(plan, memory, 0, 0, 0)?);
                assert!(!capture_memory_available(plan, memory, 12 * MIB, 0, 0)?);
                assert!(eager_memory_available(plan, memory, 12 * MIB, 0, 0)?);
            }
        }
        Ok(())
    }

    #[test]
    fn eager_retry_respects_pending_allocations_and_physical_limits() -> Result<()> {
        let plan = RuntimeMemoryPlan {
            memory_goal: MemoryGoal::Auto,
            ..plan()
        };
        let memory = DeviceMemory::Unified {
            total: 128 * GIB,
            available: GIB,
            allocated: 113 * GIB,
            working_set_limit: 113 * GIB + 96 * MIB,
        };
        for (workspace, pending_weights, other_workspace, expected) in [
            (64 * MIB, 32 * MIB, 0, true),
            (64 * MIB + 1, 32 * MIB, 0, false),
            (32 * MIB, 32 * MIB, 32 * MIB, true),
            (32 * MIB + 1, 32 * MIB, 32 * MIB, false),
        ] {
            assert_eq!(
                eager_memory_available(plan, memory, workspace, pending_weights, other_workspace)?,
                expected,
            );
        }
        let memory = DeviceMemory::Unified {
            total: 128 * GIB,
            available: 64 * MIB,
            allocated: 113 * GIB,
            working_set_limit: 128 * GIB,
        };
        assert!(eager_memory_available(plan, memory, 32 * MIB, 32 * MIB, 0)?);
        assert!(!eager_memory_available(
            plan,
            memory,
            32 * MIB + 1,
            32 * MIB,
            0,
        )?);
        assert!(eager_memory_available(plan, memory, 1, usize::MAX, 0).is_err());
        Ok(())
    }

    #[test]
    fn eager_retry_preserves_explicit_utilization_limits() -> Result<()> {
        let plan = RuntimeMemoryPlan {
            memory_goal: MemoryGoal::Utilization(0.875),
            ..plan()
        };
        for (available, allocated, workspace, expected) in [
            (17 * GIB, 111 * GIB, GIB, true),
            (17 * GIB, 111 * GIB, GIB + 1, false),
            (16 * GIB, 112 * GIB, 1, false),
            (15 * GIB, 113 * GIB, 0, false),
        ] {
            let memory = DeviceMemory::Unified {
                total: 128 * GIB,
                available,
                allocated,
                working_set_limit: 128 * GIB,
            };
            assert_eq!(
                eager_memory_available(plan, memory, workspace, 0, 0)?,
                expected,
            );
        }
        Ok(())
    }

    #[test]
    fn capture_workspace_rejects_arithmetic_overflow() {
        let shape = CaptureMemoryShape {
            query_tokens: usize::MAX,
            input_bytes: 0,
            attention_bytes: 0,
            attention_is_complete: true,
        };
        assert!(capture_workspace_bytes(plan(), true, &shape).is_err());
        assert!(partitioned_attention_bytes(usize::MAX, 8, 128, 2, 2).is_err());
    }

    #[test]
    fn capture_retains_fa3_pools_during_standard_attention() -> Result<()> {
        let fa3_pool = partitioned_attention_bytes(8, 16, 128, 33, 4)?;
        let flashinfer_pool = partitioned_attention_bytes(64, 16, 128, 1, 2)?;
        let standard_layer = partitioned_attention_bytes(8, 16, 256, 9, 4)?;
        let bytes = capture_attention_bytes(fa3_pool, flashinfer_pool, standard_layer)?;
        assert_eq!(bytes, fa3_pool + flashinfer_pool + standard_layer);
        assert!(bytes > fa3_pool.max(standard_layer) + flashinfer_pool);
        assert_eq!(
            capture_attention_bytes(0, 0, standard_layer)?,
            standard_layer
        );
        assert!(capture_attention_bytes(usize::MAX, 1, 0).is_err());
        assert!(capture_attention_bytes(usize::MAX, 0, 1).is_err());
        Ok(())
    }

    #[test]
    fn fa3_capture_scratch_uses_schedule_splits_and_only_page_tables_grow_with_context(
    ) -> Result<()> {
        const SPLITS: usize = 32;
        const SHORT_PAGES: usize = 128;
        const LONG_PAGES: usize = 31_563;
        let shape = Fa3CaptureMemoryShape {
            batch: 16,
            query_len: 8,
            heads: 24,
            head_dim: 256,
            splits: SPLITS,
            pages_per_sequence: SHORT_PAGES,
        };
        let short = fa3_capture_pool_bytes(shape)?;
        let long = fa3_capture_pool_bytes(Fa3CaptureMemoryShape {
            pages_per_sequence: LONG_PAGES,
            ..shape
        })?;
        assert_eq!(
            long - short,
            shape.batch * (LONG_PAGES - SHORT_PAGES) * size_of::<i32>()
        );
        let accumulator = shape.batch
            * shape.query_len
            * shape.heads
            * shape.head_dim
            * SPLITS
            * size_of::<f32>();
        assert!(long > accumulator);
        assert!(long < accumulator + 4 * 1024 * 1024);
        assert!(fa3_capture_pool_bytes(Fa3CaptureMemoryShape {
            batch: usize::MAX,
            ..shape
        })
        .is_err());
        Ok(())
    }
}
