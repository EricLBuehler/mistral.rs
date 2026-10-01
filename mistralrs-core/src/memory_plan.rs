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
    pub planned_query_tokens: usize,
    pub logits_bytes_per_query_token: usize,
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

    pub fn non_attention_bytes(self, primary: bool, query_tokens: usize) -> Result<usize> {
        self.scale(
            if primary {
                self.primary_non_attention_bytes
            } else {
                self.mapped_non_attention_bytes
            },
            query_tokens,
        )
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
    let (target, free_reserve) = match goal {
        MemoryGoal::Utilization(fraction) => ((total as f64 * f64::from(fraction)) as usize, 0),
        _ => (total.saturating_sub(reserve), reserve),
    };
    let engine_limit_bytes = target.min(memory.working_set_limit());
    let occupied_bytes = memory.allocated();
    let available_bytes = memory.available();
    let allocatable_bytes = engine_limit_bytes
        .saturating_sub(occupied_bytes)
        .min(memory.physical_available().saturating_sub(free_reserve))
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
            34 * GIB
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
}
