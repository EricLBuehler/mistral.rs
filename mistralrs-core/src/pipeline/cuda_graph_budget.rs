use anyhow::{Context, Result};

use crate::{memory_plan::RuntimeMemoryPlan, utils::memory_usage::DeviceMemory};

pub(super) struct CaptureMemoryShape {
    pub query_tokens: usize,
    pub input_bytes: usize,
    pub attention_bytes: usize,
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
    let scaled_non_attention = non_attention
        .checked_mul(shape.query_tokens)
        .context("CUDA capture activation size overflow")?
        .div_ceil(plan.planned_query_tokens.max(1));
    let logits = if primary {
        plan.logits_bytes_per_query_token
            .checked_mul(shape.query_tokens)
            .context("CUDA capture logits size overflow")?
    } else {
        0
    };
    let attention = baseline
        .saturating_sub(non_attention)
        .max(shape.attention_bytes);
    scaled_non_attention
        .max(logits)
        .checked_add(attention)
        .and_then(|bytes| bytes.checked_add(shape.input_bytes))
        .context("CUDA capture workspace size overflow")
}

pub(super) fn capture_memory_available(
    plan: RuntimeMemoryPlan,
    memory: DeviceMemory,
    workspace: usize,
    pending_weights: usize,
    other_workspace: usize,
) -> Result<bool> {
    let required = workspace
        .checked_add(pending_weights)
        .and_then(|bytes| bytes.checked_add(other_workspace))
        .context("CUDA capture pending memory size overflow")?;
    let budget = crate::memory_plan::resolve_device_budget(plan.memory_goal, memory)?;
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

    const GIB: usize = 1024 * 1024 * 1024;

    fn plan() -> RuntimeMemoryPlan {
        RuntimeMemoryPlan {
            memory_goal: MemoryGoal::Utilization(0.8),
            limits: Default::default(),
            primary_activation_bytes: 10 * GIB,
            mapped_activation_bytes: 8 * GIB,
            primary_non_attention_bytes: 8 * GIB,
            mapped_non_attention_bytes: 6 * GIB,
            planned_query_tokens: 4096,
            logits_bytes_per_query_token: 1024 * 1024,
            custom_prefill_mask: false,
            metal_prefill_mask_is_structured: false,
        }
    }

    #[test]
    fn capture_preserves_attention_scratch_but_scales_decode_activations() -> Result<()> {
        let shape = CaptureMemoryShape {
            query_tokens: 8,
            input_bytes: 32,
            attention_bytes: GIB,
        };
        assert_eq!(
            capture_workspace_bytes(plan(), true, &shape)?,
            2 * GIB + 16 * 1024 * 1024 + 32
        );
        let shape = CaptureMemoryShape {
            attention_bytes: 3 * GIB,
            ..shape
        };
        assert_eq!(
            capture_workspace_bytes(plan(), true, &shape)?,
            3 * GIB + 16 * 1024 * 1024 + 32
        );
        Ok(())
    }

    #[test]
    fn capture_reserves_full_logits_rows_when_prefill_scaling_is_smaller() -> Result<()> {
        let mut plan = plan();
        plan.primary_non_attention_bytes = GIB;
        plan.primary_activation_bytes = GIB;
        let shape = CaptureMemoryShape {
            query_tokens: 8,
            input_bytes: 0,
            attention_bytes: 0,
        };
        assert_eq!(
            capture_workspace_bytes(plan, true, &shape)?,
            8 * 1024 * 1024
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
        };
        assert_eq!(
            capture_workspace_bytes(plan, true, &shape)?,
            8 * 1024 * 1024 + shape.input_bytes + shape.attention_bytes
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
    fn capture_workspace_rejects_arithmetic_overflow() {
        let shape = CaptureMemoryShape {
            query_tokens: usize::MAX,
            input_bytes: 0,
            attention_bytes: 0,
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
}
