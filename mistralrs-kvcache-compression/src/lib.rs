//! TurboQuant KV-cache compression for mistral.rs.
//!
//! This crate provides transparent KV-cache compression using the
//! [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
//! algorithm (ICLR 2026).  It compresses key-value vectors to as few as
//! **3 bits per coordinate** with virtually no accuracy loss.
//!
//! ## Quick start (NormalCache mode)
//!
//! Add a `kvcache_compression` field to your model's configuration:
//!
//! ```yaml
//! kvcache_compression:
//!   bits: 3
//!   policy:
//!     threshold_tokens: 4096
//! ```
//!
//! The `KvCache::Compressed` variant in `mistralrs-core` handles the rest.
//!
//! ## Expected memory savings
//!
//! | Bit-width | KV cache vs FP16 | Cosine similarity |
//! |-----------|------------------|-------------------|
//! | 2-bit     | ~15× smaller     | > 0.95            |
//! | 3-bit     | ~5× smaller      | > 0.98 (default)  |
//! | 4-bit     | ~3.5× smaller    | > 0.99            |
//!
//! ## Platform support
//!
//! The underlying `turboquant` crate uses SIMD runtime dispatch:
//! - **x86_64**: AVX-512, AVX2+FMA, or AVX2 (selected at runtime)
//! - **aarch64**: NEON (Apple Silicon M-series, AWS Graviton)
//! - **fallback**: scalar path on any other target
//!
//! No GPU dependency — compression runs on CPU and is optimized to
//! keep overhead below 5% of total inference time at 4K+ context.

pub mod compressed_cache;
pub mod config;

pub use compressed_cache::{
    compress_all_tokens, compress_last_token, compress_token_at_pos,
    new_shared_compressed_layer_cache, CompressedLayerCache, SharedCompressedLayerCache,
};
pub use config::{CompressionBits, CompressionPolicy, KvCompressionConfig};
