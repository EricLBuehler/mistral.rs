//! Pluggable KV-cache compression hook.
//!
//! A `KvCacheCodec` lets callers intercept every tensor written to a
//! `SingleCache` / `RotatingCache` and every tensor read back, so they can
//! apply quantization (fp8, int8, TurboQuant, etc.) without the cache having
//! to know what the codec does.
//!
//! # Shape & dtype contract
//!
//! `encode` and `decode` MUST return a tensor with the same shape and dtype as
//! their input. This preserves the existing `slice_set` / `narrow` paths in
//! the cache implementations: the cache still sees a uniform-shape,
//! uniform-dtype buffer; the codec simply degrades precision within that
//! envelope (e.g. fp16 values rounded to an fp8 grid but stored as fp16).
//!
//! True packed-storage codecs (sub-byte layouts) need a richer interface than
//! this one exposes — they're intentionally out of scope for the first
//! landing. Start here, prove the hook, then extend.
//!
//! # Default behaviour
//!
//! When no codec is installed (`codec: None`, the default), the caches
//! short-circuit to the existing bit-exact behaviour. Installing
//! `PassthroughCodec` is semantically identical but exercises the codec path
//! — useful for tests.

use std::sync::Arc;

use candle_core::{Result, Tensor};

/// Encode/decode hook for KV-cache tensors.
///
/// Implementors must preserve shape and dtype. See the module-level docs for
/// the rationale.
pub trait KvCacheCodec: Send + Sync + std::fmt::Debug {
    /// Quantize a tensor before it is written into the cache buffer.
    fn encode(&self, tensor: &Tensor) -> Result<Tensor>;

    /// Dequantize a tensor read from the cache buffer before it's used for
    /// attention.
    fn decode(&self, tensor: &Tensor) -> Result<Tensor>;

    /// Human-readable name for diagnostics and logging.
    fn name(&self) -> &str;
}

/// Identity codec — encode and decode both clone the input tensor.
///
/// Installing this is semantically equivalent to `codec: None` but runs the
/// codec dispatch path. Useful for round-trip tests and as a skeleton for
/// real codecs.
#[derive(Debug, Default, Clone, Copy)]
pub struct PassthroughCodec;

impl KvCacheCodec for PassthroughCodec {
    fn encode(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn decode(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn name(&self) -> &str {
        "passthrough"
    }
}

/// Type alias for the stored codec — `None` means "no codec, bit-exact
/// passthrough".
pub type KvCacheCodecRef = Option<Arc<dyn KvCacheCodec>>;
