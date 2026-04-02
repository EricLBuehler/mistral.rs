use serde::{Deserialize, Serialize};

/// Bit-width for TurboQuant KV-cache compression.
///
/// - `Two`   → ~15× smaller KV cache, cosine similarity > 0.95
/// - `Three` → ~5×  smaller KV cache, cosine similarity > 0.98 (recommended)
/// - `Four`  → ~3.5× smaller KV cache, cosine similarity > 0.99
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompressionBits {
    #[serde(rename = "2")]
    Two = 2,
    #[serde(rename = "3")]
    Three = 3,
    #[serde(rename = "4")]
    Four = 4,
}

impl Default for CompressionBits {
    fn default() -> Self {
        Self::Three
    }
}

impl CompressionBits {
    pub fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Policy controlling when TurboQuant compression is applied to KV vectors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompressionPolicy {
    /// Compress all tokens from the first token onward.
    Always,
    /// Keep the first `threshold_tokens` tokens uncompressed; compress the tail.
    /// This avoids accuracy loss in the early (warm-up) context window.
    ThresholdTokens(usize),
    /// No compression — disabled.
    Disabled,
}

impl Default for CompressionPolicy {
    fn default() -> Self {
        Self::ThresholdTokens(4096)
    }
}

/// Top-level configuration for TurboQuant KV-cache compression.
///
/// Add to your model config or pass via CLI flags:
/// ```yaml
/// kvcache_compression:
///   bits: 3
///   policy:
///     threshold_tokens: 4096
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCompressionConfig {
    /// Compression bit-width. Default: 3 bits.
    #[serde(default)]
    pub bits: CompressionBits,
    /// Policy for when to start compressing. Default: after 4096 tokens.
    #[serde(default)]
    pub policy: CompressionPolicy,
}

impl Default for KvCompressionConfig {
    fn default() -> Self {
        Self {
            bits: CompressionBits::default(),
            policy: CompressionPolicy::default(),
        }
    }
}

impl KvCompressionConfig {
    /// Returns `true` if compression is disabled entirely.
    pub fn is_disabled(&self) -> bool {
        self.policy == CompressionPolicy::Disabled
    }

    /// Returns the number of uncompressed "warm-up" tokens before compression
    /// kicks in. Returns `0` for `Always` and `usize::MAX` for `Disabled`.
    pub fn threshold_tokens(&self) -> usize {
        match self.policy {
            CompressionPolicy::Always => 0,
            CompressionPolicy::ThresholdTokens(n) => n,
            CompressionPolicy::Disabled => usize::MAX,
        }
    }
}
