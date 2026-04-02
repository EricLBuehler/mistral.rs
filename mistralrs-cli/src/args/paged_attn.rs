//! PagedAttention configuration options
//!
//! Unified design replacing the confusing 5-flag system with clear semantics.

use clap::{Args, ValueEnum};
use mistralrs_core::PagedCacheType;
use serde::Deserialize;

/// Cache and attention configuration
#[derive(Args, Clone, Deserialize, Default)]
pub struct CacheOptions {
    #[command(flatten)]
    pub paged_attn: PagedAttentionOptions,

    /// Bits per coordinate for TurboQuant KV-cache compression (2, 3, or 4).
    /// Requires the `kvcache-compression` feature. Disabled by default.
    /// Can also be set via the MISTRALRS_KV_CACHE_BITS environment variable.
    #[arg(long = "kv-cache-bits", value_parser = clap::value_parser!(u8).range(2..=4), env = "MISTRALRS_KV_CACHE_BITS")]
    #[serde(default)]
    pub kv_compression_bits: Option<u8>,

    /// Minimum number of tokens to accumulate before compressing.
    /// Only takes effect when `--kv-cache-bits` is set (default: 128).
    /// Can also be set via the MISTRALRS_KV_CACHE_THRESHOLD environment variable.
    #[arg(long = "kv-cache-threshold", default_value = "128", requires = "kv_compression_bits", env = "MISTRALRS_KV_CACHE_THRESHOLD")]
    #[serde(default = "default_kv_cache_threshold")]
    pub kv_compression_threshold: usize,
}

fn default_kv_cache_threshold() -> usize {
    128
}

/// PagedAttention configuration
#[derive(Args, Clone, Deserialize)]
pub struct PagedAttentionOptions {
    /// PagedAttention mode
    /// - auto: enabled on CUDA, disabled on Metal/CPU (default)
    /// - on: force enable (fails if unsupported)
    /// - off: force disable
    #[arg(long = "paged-attn", default_value = "auto", value_enum)]
    #[serde(default)]
    pub mode: PagedAttnMode,

    /// Allocate KV cache for this context length.
    /// If not specified, defaults to using 90% of available VRAM.
    #[arg(long = "pa-context-len")]
    pub context_len: Option<usize>,

    /// GPU memory to allocate in MBs (alternative to context-len)
    #[arg(long = "pa-memory-mb", conflicts_with = "context_len")]
    pub memory_mb: Option<usize>,

    /// GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb)
    #[arg(long = "pa-memory-fraction", conflicts_with_all = ["context_len", "memory_mb"])]
    pub memory_fraction: Option<f32>,

    /// Tokens per block (default: 32 on CUDA)
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
            context_len: None,
            memory_mb: None,
            memory_fraction: None,
            block_size: None,
            cache_type: PagedCacheType::Auto,
        }
    }
}

/// PagedAttention operation mode
#[derive(Clone, Copy, ValueEnum, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PagedAttnMode {
    /// Automatic: enabled on CUDA, disabled on Metal/CPU
    #[default]
    Auto,
    /// Force enable (error if device doesn't support it)
    On,
    /// Force disable
    Off,
}

impl PagedAttentionOptions {
    /// Convert to the flags expected by MistralRsForServerBuilder
    pub fn into_builder_flags(self) -> PagedAttnBuilderFlags {
        let enable = match self.mode {
            PagedAttnMode::Auto => None,
            PagedAttnMode::On => Some(true),
            PagedAttnMode::Off => Some(false),
        };

        (
            enable,
            self.memory_mb,
            self.memory_fraction,
            self.context_len,
            self.block_size,
            self.cache_type,
        )
    }
}

fn parse_cache_type(s: &str) -> Result<PagedCacheType, String> {
    s.parse()
}

/// PagedAttention builder flags type alias
pub type PagedAttnBuilderFlags = (
    Option<bool>,   // paged_attn enable flag
    Option<usize>,  // gpu_mem (MBs)
    Option<f32>,    // gpu_mem_usage (fraction)
    Option<usize>,  // context_len
    Option<usize>,  // block_size
    PagedCacheType, // cache_type
);
