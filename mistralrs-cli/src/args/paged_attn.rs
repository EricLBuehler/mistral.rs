//! PagedAttention configuration options

use clap::{Args, ValueEnum};
use mistralrs_core::{MemoryGoal, PagedCacheType};
use serde::Deserialize;

/// Cache and attention configuration
#[derive(Args, Clone, Deserialize, Default)]
pub struct CacheOptions {
    #[command(flatten)]
    pub paged_attn: PagedAttentionOptions,
}

/// PagedAttention configuration
#[derive(Args, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PagedAttentionOptions {
    /// PagedAttention mode
    /// - auto: enabled on CUDA and Metal, disabled on CPU (default)
    /// - on: force enable (fails if unsupported)
    /// - off: force disable
    #[arg(long = "paged-attn", default_value = "auto", value_enum)]
    #[serde(default)]
    pub mode: PagedAttnMode,

    /// Fraction of physical GPU memory available to model weights, workspace, and KV cache
    #[arg(long, conflicts_with = "kv_cache")]
    pub gpu_memory_utilization: Option<f32>,

    /// Explicit KV cache capacity, such as 4GiB or 65536tokens
    #[arg(long, value_parser = parse_kv_cache)]
    pub kv_cache: Option<String>,

    /// Tokens per block (default: 32 on CUDA and Metal)
    #[arg(long = "pa-block-size")]
    pub block_size: Option<usize>,

    /// KV cache quantization type
    #[arg(long = "pa-cache-type", default_value = "auto", value_parser = parse_cache_type)]
    #[serde(default)]
    pub cache_type: PagedCacheType,
}

impl Default for PagedAttentionOptions {
    fn default() -> Self {
        Self {
            mode: PagedAttnMode::Auto,
            gpu_memory_utilization: None,
            kv_cache: None,
            block_size: None,
            cache_type: PagedCacheType::Auto,
        }
    }
}

/// PagedAttention operation mode
#[derive(Clone, Copy, ValueEnum, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PagedAttnMode {
    /// Automatic: enabled on CUDA and Metal, disabled on CPU
    #[default]
    Auto,
    /// Force enable (error if device doesn't support it)
    On,
    /// Force disable
    Off,
}

impl PagedAttentionOptions {
    /// Convert to the flags expected by MistralRsForServerBuilder
    pub fn into_builder_flags(self) -> anyhow::Result<PagedAttnBuilderFlags> {
        let enable = match self.mode {
            PagedAttnMode::Auto => None,
            PagedAttnMode::On => Some(true),
            PagedAttnMode::Off => Some(false),
        };

        anyhow::ensure!(
            self.gpu_memory_utilization.is_none() || self.kv_cache.is_none(),
            "gpu_memory_utilization and kv_cache are mutually exclusive"
        );
        let memory_goal = match (self.gpu_memory_utilization, self.kv_cache) {
            (Some(value), None) => MemoryGoal::Utilization(value),
            (None, Some(value)) => parse_kv_cache(&value)
                .map_err(anyhow::Error::msg)?
                .parse()?,
            _ => MemoryGoal::Auto,
        };
        memory_goal.validate()?;
        anyhow::ensure!(
            self.mode != PagedAttnMode::Off || memory_goal == MemoryGoal::Auto,
            "an explicit memory goal requires paged attention"
        );
        Ok((enable, memory_goal, self.block_size, self.cache_type))
    }
}

fn parse_cache_type(s: &str) -> Result<PagedCacheType, String> {
    s.parse()
}

fn parse_kv_cache(value: &str) -> Result<String, String> {
    match value.parse::<MemoryGoal>().map_err(|err| err.to_string())? {
        MemoryGoal::KvBytes(_) | MemoryGoal::KvTokens(_) => Ok(value.to_owned()),
        _ => Err("expected a KV cache capacity such as 4GiB or 65536tokens".to_owned()),
    }
}

pub type PagedAttnBuilderFlags = (Option<bool>, MemoryGoal, Option<usize>, PagedCacheType);

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    const TEST_TOKENS: usize = 65536;

    #[derive(Parser)]
    struct TestArgs {
        #[command(flatten)]
        paging: PagedAttentionOptions,
    }

    #[test]
    fn cli_rejects_conflicting_and_removed_options() {
        assert!(TestArgs::try_parse_from([
            "test",
            "--kv-cache",
            "4GiB",
            "--gpu-memory-utilization",
            "0.9"
        ])
        .is_err());
        assert!(TestArgs::try_parse_from(["test", "--pa-context-len", "4096"]).is_err());
        assert!(TestArgs::try_parse_from(["test", "--kv-cache", "4"]).is_err());
        assert!(TestArgs::try_parse_from(["test", "--kv-cache", "65536tokens"]).is_ok());
    }

    #[test]
    fn rejects_conflicting_or_disabled_memory_goals() {
        let mut options = PagedAttentionOptions {
            gpu_memory_utilization: Some(0.9),
            kv_cache: Some("4GiB".to_owned()),
            ..Default::default()
        };
        assert!(options.clone().into_builder_flags().is_err());
        options.kv_cache = None;
        options.mode = PagedAttnMode::Off;
        assert!(options.into_builder_flags().is_err());
        assert!(parse_kv_cache("auto").is_err());
    }

    #[test]
    fn shares_default_and_explicit_capacity_policy() -> anyhow::Result<()> {
        assert_eq!(
            PagedAttentionOptions::default().into_builder_flags()?.1,
            MemoryGoal::Auto
        );
        let options = PagedAttentionOptions {
            kv_cache: Some("65536tokens".to_owned()),
            ..Default::default()
        };
        assert_eq!(
            options.into_builder_flags()?.1,
            MemoryGoal::KvTokens(TEST_TOKENS)
        );
        Ok(())
    }
}
