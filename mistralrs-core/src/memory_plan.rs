use std::{fmt, str::FromStr};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::utils::memory_usage::DeviceMemory;

const GIB: usize = 1024 * 1024 * 1024;
const DISCRETE_RESERVE_FRACTION: f64 = 0.05;
const UNIFIED_RESERVE_FRACTION: f64 = 0.25;
const MIN_DISCRETE_RESERVE: usize = GIB;
const MAX_UNIFIED_RESERVE: usize = 16 * GIB;
pub const DEFAULT_MAX_NUM_SEQS: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryGoal {
    #[default]
    Auto,
    Utilization(f32),
    KvBytes(usize),
    KvTokens(usize),
}

impl MemoryGoal {
    pub fn validate(self) -> Result<()> {
        match self {
            Self::Utilization(f) if !f.is_finite() || f <= 0.0 || f > 1.0 => {
                bail!("GPU memory utilization must be finite and in (0, 1]")
            }
            Self::KvBytes(0) | Self::KvTokens(0) => bail!("KV cache capacity must be nonzero"),
            _ => Ok(()),
        }
    }

    pub fn is_capacity(self) -> bool {
        matches!(self, Self::KvBytes(_) | Self::KvTokens(_))
    }
}

impl FromStr for MemoryGoal {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        let split = value
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(value.len());
        let amount: usize = value[..split]
            .parse()
            .context("KV capacity must start with a positive integer")?;
        let unit = value[split..].trim().to_ascii_lowercase();
        let goal = if matches!(unit.as_str(), "token" | "tokens") {
            Self::KvTokens(amount)
        } else {
            let multiplier: usize = match unit.as_str() {
                "b" => 1,
                "kib" => 1024,
                "mib" => 1024 * 1024,
                "gib" => GIB,
                "tib" => 1024 * GIB,
                "kb" => 1000,
                "mb" => 1000 * 1000,
                "gb" => 1000 * 1000 * 1000,
                "tb" => 1000 * 1000 * 1000 * 1000,
                _ => bail!(
                    "KV capacity requires a byte unit or 'tokens', for example 4GiB or 65536tokens"
                ),
            };
            Self::KvBytes(
                amount
                    .checked_mul(multiplier)
                    .context("KV capacity overflows usize")?,
            )
        };
        goal.validate()?;
        Ok(goal)
    }
}

impl fmt::Display for MemoryGoal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str("automatic memory budget"),
            Self::Utilization(value) => write!(f, "{:.1}% GPU memory utilization", value * 100.0),
            Self::KvBytes(bytes) => write!(f, "{bytes} bytes of shared KV cache"),
            Self::KvTokens(tokens) => write!(f, "{tokens} shared KV cache tokens"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionMemoryLimits {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub max_prefill_chunk_tokens: usize,
}

impl Default for ExecutionMemoryLimits {
    fn default() -> Self {
        Self {
            max_num_seqs: DEFAULT_MAX_NUM_SEQS,
            max_num_batched_tokens: crate::DEFAULT_MAX_NUM_BATCHED_TOKENS,
            max_prefill_chunk_tokens: crate::DEFAULT_MAX_PREFILL_CHUNK_TOKENS,
        }
    }
}

impl ExecutionMemoryLimits {
    pub fn validate(self) -> Result<()> {
        if self.max_num_seqs == 0
            || self.max_num_batched_tokens == 0
            || self.max_prefill_chunk_tokens == 0
        {
            bail!("Execution sequence, batch-token and prefill-chunk limits must be nonzero");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeMemoryPlan {
    pub memory_goal: MemoryGoal,
    pub limits: ExecutionMemoryLimits,
    pub primary_activation_bytes: usize,
    pub mapped_activation_bytes: usize,
    pub primary_non_attention_bytes: usize,
    pub mapped_non_attention_bytes: usize,
    pub encoder_activation_bytes: usize,
    pub planned_query_tokens: usize,
    pub logits_bytes_per_query_token: usize,
    pub custom_prefill_mask: bool,
    pub metal_prefill_mask_is_structured: bool,
}

impl RuntimeMemoryPlan {
    pub fn activation_bytes(self, primary: bool, query_tokens: usize) -> Result<usize> {
        self.scale(
            if primary {
                self.primary_activation_bytes
            } else {
                self.mapped_activation_bytes
            },
            query_tokens,
        )
    }

    pub fn non_attention_bytes(
        self,
        primary: bool,
        query_tokens: usize,
        sequences: usize,
    ) -> Result<usize> {
        let bytes = if primary {
            self.primary_non_attention_bytes
        } else {
            self.mapped_non_attention_bytes
        };
        let text = bytes
            .checked_mul(query_tokens)
            .context("Runtime text activation memory size overflow")?
            .div_ceil(self.planned_query_tokens.max(1));
        let logits = if primary {
            self.logits_bytes_per_query_token
                .checked_mul(sequences)
                .context("Runtime logits memory size overflow")?
        } else {
            0
        };
        text.checked_add(logits)
            .context("Runtime non-attention memory size overflow")
    }

    fn scale(self, bytes: usize, query_tokens: usize) -> Result<usize> {
        let planned = self.planned_query_tokens.max(1);
        if query_tokens <= planned {
            return Ok(bytes);
        }
        bytes
            .checked_mul(query_tokens)
            .context("Runtime activation memory size overflow")
            .map(|scaled| scaled.div_ceil(planned))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DeviceMemoryBudget {
    pub physical_bytes: usize,
    pub engine_limit_bytes: usize,
    pub occupied_bytes: usize,
    pub available_bytes: usize,
    pub allocatable_bytes: usize,
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
pub fn resolve_device_budget(goal: MemoryGoal, memory: DeviceMemory) -> Result<DeviceMemoryBudget> {
    goal.validate()?;
    let total = memory.total();
    let reserve = if memory.is_unified() {
        ((total as f64 * UNIFIED_RESERVE_FRACTION) as usize).min(MAX_UNIFIED_RESERVE)
    } else {
        ((total as f64 * DISCRETE_RESERVE_FRACTION) as usize).max(MIN_DISCRETE_RESERVE)
    }
    .min(total);
    let target = match goal {
        MemoryGoal::Utilization(fraction) => (total as f64 * f64::from(fraction)) as usize,
        _ => total.saturating_sub(reserve),
    };
    let engine_limit_bytes = target.min(memory.working_set_limit());
    let occupied_bytes = memory.allocated();
    let available_bytes = memory.available();
    let allocatable_bytes = engine_limit_bytes
        .saturating_sub(occupied_bytes)
        .min(available_bytes);
    Ok(DeviceMemoryBudget {
        physical_bytes: total,
        engine_limit_bytes,
        occupied_bytes,
        available_bytes,
        allocatable_bytes,
    })
}

pub fn device_memory_budget(goal: MemoryGoal, memory: DeviceMemory) -> Result<usize> {
    Ok(resolve_device_budget(goal, memory)?.allocatable_bytes)
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) fn resolve_forward_budget(
    goal: MemoryGoal,
    memory: DeviceMemory,
) -> Result<DeviceMemoryBudget> {
    let mut budget = resolve_device_budget(goal, memory)?;
    if !matches!(goal, MemoryGoal::Utilization(_)) {
        // Automatic headroom cushions transient work; only explicit utilization is a runtime cap.
        budget.engine_limit_bytes = memory.working_set_limit();
        budget.allocatable_bytes = budget
            .engine_limit_bytes
            .saturating_sub(budget.occupied_bytes)
            .min(budget.available_bytes);
    }
    Ok(budget)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_units_and_validation() {
        assert_eq!(
            "4GiB".parse::<MemoryGoal>().unwrap(),
            MemoryGoal::KvBytes(4 * GIB)
        );
        assert_eq!(
            "65536tokens".parse::<MemoryGoal>().unwrap(),
            MemoryGoal::KvTokens(65536)
        );
        for invalid in [
            "0tokens",
            "42",
            "-1GiB",
            "4percent",
            "18446744073709551615TiB",
        ] {
            assert!(invalid.parse::<MemoryGoal>().is_err(), "{invalid}");
        }
        for invalid in [f32::NAN, f32::INFINITY, 0.0, -0.1, 1.1] {
            assert!(MemoryGoal::Utilization(invalid).validate().is_err());
        }
    }

    #[test]
    fn discrete_budget_subtracts_existing_occupancy_once() {
        let memory = DeviceMemory::Discrete {
            total: 100 * GIB,
            free: 60 * GIB,
        };
        assert_eq!(
            device_memory_budget(MemoryGoal::Auto, memory).unwrap(),
            55 * GIB
        );
        let explicit = device_memory_budget(MemoryGoal::Utilization(0.5), memory).unwrap();
        assert_eq!(explicit, 10 * GIB);
    }

    #[test]
    fn discrete_auto_budget_preserves_fraction_and_minimum_headroom() {
        for (total, free, expected) in [
            (100 * GIB, 100 * GIB, 95 * GIB),
            (100 * GIB, 60 * GIB, 55 * GIB),
            (100 * GIB, 5 * GIB, 0),
            (100 * GIB, GIB, 0),
            (8 * GIB, 8 * GIB, 7 * GIB),
            (8 * GIB, 4 * GIB, 3 * GIB),
            (8 * GIB, GIB, 0),
        ] {
            assert_eq!(
                device_memory_budget(MemoryGoal::Auto, DeviceMemory::Discrete { total, free })
                    .unwrap(),
                expected,
                "total={total}, free={free}"
            );
        }
    }

    #[test]
    fn unified_budget_respects_pressure_and_working_set() {
        let small = DeviceMemory::Unified {
            total: 8 * GIB,
            available: 8 * GIB,
            allocated: 0,
            working_set_limit: 8 * GIB,
        };
        assert_eq!(
            device_memory_budget(MemoryGoal::Auto, small).unwrap(),
            6 * GIB
        );
        let memory = DeviceMemory::Unified {
            total: 64 * GIB,
            available: 50 * GIB,
            allocated: 8 * GIB,
            working_set_limit: 48 * GIB,
        };
        assert_eq!(
            device_memory_budget(MemoryGoal::Auto, memory).unwrap(),
            40 * GIB
        );
        assert_eq!(
            device_memory_budget(MemoryGoal::Utilization(1.0), memory).unwrap(),
            40 * GIB
        );
        let large = DeviceMemory::Unified {
            total: 128 * GIB,
            available: 128 * GIB,
            allocated: 0,
            working_set_limit: 128 * GIB,
        };
        assert_eq!(
            device_memory_budget(MemoryGoal::Auto, large).unwrap(),
            112 * GIB
        );
    }

    #[test]
    fn unified_auto_headroom_is_applied_once_to_the_engine_limit() {
        for (available, allocated, working_set_limit, expected) in [
            (50 * GIB, 8 * GIB, 64 * GIB, 40 * GIB),
            (26 * GIB, 8 * GIB, 48 * GIB, 26 * GIB),
            (50 * GIB, 8 * GIB, 40 * GIB, 32 * GIB),
            (GIB, 8 * GIB, 48 * GIB, GIB),
            (16 * GIB, 48 * GIB, 64 * GIB, 0),
            (16 * GIB, 49 * GIB, 64 * GIB, 0),
            (16 * GIB, 41 * GIB, 40 * GIB, 0),
        ] {
            let memory = DeviceMemory::Unified {
                total: 64 * GIB,
                available,
                allocated,
                working_set_limit,
            };
            let auto = resolve_device_budget(MemoryGoal::Auto, memory).unwrap();
            let utilization = resolve_device_budget(MemoryGoal::Utilization(0.75), memory).unwrap();
            assert_eq!(auto.engine_limit_bytes, (48 * GIB).min(working_set_limit));
            assert_eq!(auto.allocatable_bytes, expected);
            assert_eq!(auto.engine_limit_bytes, utilization.engine_limit_bytes);
            assert_eq!(auto.allocatable_bytes, utilization.allocatable_bytes);
            assert!(auto.allocatable_bytes <= available);
            assert!(auto.allocatable_bytes <= working_set_limit.saturating_sub(allocated));
        }
    }

    #[test]
    fn automatic_and_capacity_goals_leave_planning_headroom_available_to_forward() {
        let memory = DeviceMemory::Unified {
            total: 128 * GIB,
            available: 15 * GIB,
            allocated: 113 * GIB,
            working_set_limit: 128 * GIB,
        };
        for goal in [
            MemoryGoal::Auto,
            MemoryGoal::KvBytes(GIB),
            MemoryGoal::KvTokens(65536),
        ] {
            assert_eq!(
                resolve_device_budget(goal, memory)
                    .unwrap()
                    .allocatable_bytes,
                0
            );
            let forward = resolve_forward_budget(goal, memory).unwrap();
            assert_eq!(forward.engine_limit_bytes, 128 * GIB);
            assert_eq!(forward.allocatable_bytes, 15 * GIB);
        }
    }

    #[test]
    fn explicit_utilization_remains_a_forward_limit() {
        for memory in [
            DeviceMemory::Discrete {
                total: 128 * GIB,
                free: 15 * GIB,
            },
            DeviceMemory::Unified {
                total: 128 * GIB,
                available: 15 * GIB,
                allocated: 113 * GIB,
                working_set_limit: 128 * GIB,
            },
        ] {
            let forward = resolve_forward_budget(MemoryGoal::Utilization(0.875), memory).unwrap();
            assert_eq!(forward.engine_limit_bytes, 112 * GIB);
            assert_eq!(forward.allocatable_bytes, 0);
            let forward = resolve_forward_budget(MemoryGoal::Utilization(1.0), memory).unwrap();
            assert_eq!(forward.allocatable_bytes, 15 * GIB);
        }
        assert!(resolve_forward_budget(
            MemoryGoal::Utilization(f32::NAN),
            DeviceMemory::Discrete {
                total: GIB,
                free: GIB
            },
        )
        .is_err());
    }

    #[test]
    fn forward_budget_keeps_physical_and_metal_working_set_limits() {
        for (available, allocated, expected) in [
            (40 * GIB, 80 * GIB, 16 * GIB),
            (8 * GIB, 80 * GIB, 8 * GIB),
            (40 * GIB, 96 * GIB, 0),
            (40 * GIB, 97 * GIB, 0),
            (0, 80 * GIB, 0),
        ] {
            let memory = DeviceMemory::Unified {
                total: 128 * GIB,
                available,
                allocated,
                working_set_limit: 96 * GIB,
            };
            let forward = resolve_forward_budget(MemoryGoal::Auto, memory).unwrap();
            assert_eq!(forward.engine_limit_bytes, 96 * GIB);
            assert_eq!(forward.allocatable_bytes, expected);
        }
    }
}
