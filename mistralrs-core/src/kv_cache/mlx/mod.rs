//! MLX-accelerated KV cache backend for macOS Apple Silicon.
//!
//! This module routes KV cache storage and TurboQuant compression/decompression
//! through Apple's MLX framework via `mlx-rs`, enabling fused Metal kernel dispatch
//! for the compress/decompress pipeline. The forward pass stays on Candle+Metal.
//!
//! Gated by `#[cfg(feature = "mlx")]` — compiles out entirely when disabled.

pub mod bridge;
pub mod cache;
pub mod turboquant;

pub use cache::MlxCompressedCache;
pub use turboquant::MlxTurboQuantCompressor;
