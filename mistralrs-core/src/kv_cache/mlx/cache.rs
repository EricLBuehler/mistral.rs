//! MLX-backed KV cache with TurboQuant compression.
//!
//! Stores K/V tensors as MLX arrays with optional WHT + PolarQuant compression.
//! Accepts Candle tensors via `append()` (bridging internally) and returns
//! Candle tensors via `k()` / `v()` (bridging back).

use candle_core::{DType, Device, Tensor};
use mlx_rs::{Array, ops};

use super::bridge;
use super::turboquant::MlxTurboQuantCompressor;
use crate::kv_cache::KvCompressionConfig;

/// KV cache backed by MLX arrays with optional TurboQuant compression.
///
/// The cache stores compressed K/V arrays per-layer. The sequence dimension
/// is dimension 0 in the stored MLX arrays (shape: `[seq, heads, head_dim/2]`
/// when compressed, `[seq, heads, head_dim]` when uncompressed).
///
/// The `dim` field tracks which dimension of the *Candle* input tensor is the
/// sequence dimension (typically 2 for `[batch, heads, seq, head_dim]`).
pub struct MlxCompressedCache {
    /// Cached key states as MLX array: [total_seq, heads, compressed_dim]
    keys: Option<Array>,
    /// Cached value states as MLX array: [total_seq, heads, compressed_dim]
    values: Option<Array>,
    /// Total number of cached tokens
    total_seq_len: usize,
    /// Sequence dimension in the Candle input tensors
    dim: usize,
    /// Original dtype of input tensors (for bridge-back conversion)
    dtype: DType,
    /// Device to return Candle tensors on
    device: Option<Device>,
    /// TurboQuant compressor
    compressor: MlxTurboQuantCompressor,
}

impl std::fmt::Debug for MlxCompressedCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MlxCompressedCache")
            .field("total_seq_len", &self.total_seq_len)
            .field("dim", &self.dim)
            .field("dtype", &self.dtype)
            .field("has_keys", &self.keys.is_some())
            .field("has_values", &self.values.is_some())
            .finish()
    }
}

impl Clone for MlxCompressedCache {
    fn clone(&self) -> Self {
        // MLX arrays support clone (they are reference-counted internally)
        Self {
            keys: self.keys.clone(),
            values: self.values.clone(),
            total_seq_len: self.total_seq_len,
            dim: self.dim,
            dtype: self.dtype,
            device: self.device.clone(),
            compressor: MlxTurboQuantCompressor::new(self.compressor.head_dim())
                .expect("head_dim was validated at construction"),
        }
    }
}

impl MlxTurboQuantCompressor {
    /// Accessor for the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl MlxCompressedCache {
    /// Create a new MLX compressed cache.
    ///
    /// `dim` is the sequence dimension in the Candle input tensors (typically 2).
    /// `head_dim` must be a power of two.
    pub fn new(dim: usize, head_dim: usize, _config: &KvCompressionConfig) -> candle_core::Result<Self> {
        let compressor = MlxTurboQuantCompressor::new(head_dim)
            .map_err(|e| candle_core::Error::Msg(format!("MLX compressor init: {e}")))?;

        Ok(Self {
            keys: None,
            values: None,
            total_seq_len: 0,
            dim,
            dtype: DType::F32, // will be set on first append
            device: None,
            compressor,
        })
    }

    /// Append new K/V tensors to the cache.
    ///
    /// Input tensors should be Candle tensors of shape `[batch, heads, new_seq, head_dim]`.
    /// The batch dimension is expected to be 1 (squeezed internally).
    ///
    /// The tensors are:
    /// 1. Bridged to MLX arrays
    /// 2. Reshaped to [new_seq, heads, head_dim]
    /// 3. Compressed via TurboQuant
    /// 4. Concatenated with existing cache along the seq dimension
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> candle_core::Result<()> {
        self.dtype = k.dtype();
        self.device = Some(k.device().clone());

        let new_tokens = k.dim(self.dim)?;

        // Convert to MLX: first permute to [seq, heads, head_dim]
        // Input is [batch, heads, seq, head_dim] (dim=2 means seq is at index 2)
        // Squeeze batch, then transpose to [seq, heads, head_dim]
        let k_squeezed = k.squeeze(0)?; // [heads, seq, head_dim]
        let v_squeezed = v.squeeze(0)?;
        let k_perm = k_squeezed.permute((1, 0, 2))?; // [seq, heads, head_dim]
        let v_perm = v_squeezed.permute((1, 0, 2))?;

        let k_mlx = bridge::candle_to_mlx(&k_perm)?;
        let v_mlx = bridge::candle_to_mlx(&v_perm)?;

        // Compress
        let k_comp = self
            .compressor
            .compress(&k_mlx)
            .map_err(|e| candle_core::Error::Msg(format!("MLX compress K: {e}")))?;
        let v_comp = self
            .compressor
            .compress(&v_mlx)
            .map_err(|e| candle_core::Error::Msg(format!("MLX compress V: {e}")))?;

        // Concatenate with existing cache
        self.keys = Some(match self.keys.take() {
            None => k_comp,
            Some(existing) => ops::concatenate(&[&existing, &k_comp], 0)
                .map_err(|e| candle_core::Error::Msg(format!("MLX concat K: {e}")))?,
        });
        self.values = Some(match self.values.take() {
            None => v_comp,
            Some(existing) => ops::concatenate(&[&existing, &v_comp], 0)
                .map_err(|e| candle_core::Error::Msg(format!("MLX concat V: {e}")))?,
        });

        // Force evaluation to prevent lazy eval stalls
        if let Some(ref k) = self.keys {
            k.eval()
                .map_err(|e| candle_core::Error::Msg(format!("MLX eval K: {e}")))?;
        }
        if let Some(ref v) = self.values {
            v.eval()
                .map_err(|e| candle_core::Error::Msg(format!("MLX eval V: {e}")))?;
        }

        self.total_seq_len += new_tokens;
        Ok(())
    }

    /// Retrieve the full K tensor, decompressed and converted back to Candle.
    ///
    /// Returns shape `[1, heads, total_seq, head_dim]` on the original device.
    pub fn k(&self) -> candle_core::Result<Option<Tensor>> {
        self.get_tensor(&self.keys)
    }

    /// Retrieve the full V tensor, decompressed and converted back to Candle.
    pub fn v(&self) -> candle_core::Result<Option<Tensor>> {
        self.get_tensor(&self.values)
    }

    fn get_tensor(&self, arr: &Option<Array>) -> candle_core::Result<Option<Tensor>> {
        let arr = match arr {
            Some(a) => a,
            None => return Ok(None),
        };
        let device = match &self.device {
            Some(d) => d,
            None => return Ok(None),
        };

        // Force eval before reading
        arr.eval()
            .map_err(|e| candle_core::Error::Msg(format!("MLX eval: {e}")))?;

        // Decompress
        let decompressed = self
            .compressor
            .decompress(arr)
            .map_err(|e| candle_core::Error::Msg(format!("MLX decompress: {e}")))?;

        // Bridge to Candle: [seq, heads, head_dim]
        let candle_tensor = bridge::mlx_to_candle(&decompressed, device, self.dtype)?;

        // Permute back to [heads, seq, head_dim] then unsqueeze to [1, heads, seq, head_dim]
        let permuted = candle_tensor.permute((1, 0, 2))?;
        let unsqueezed = permuted.unsqueeze(0)?;

        Ok(Some(unsqueezed))
    }

    /// Current number of cached tokens.
    pub fn current_seq_len(&self) -> usize {
        self.total_seq_len
    }

    /// Clear all cached data.
    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.total_seq_len = 0;
        self.device = None;
    }

    /// Truncate cache to `len` tokens.
    pub fn set_len(&mut self, len: usize) -> candle_core::Result<()> {
        if len > self.total_seq_len {
            candle_core::bail!(
                "MlxCompressedCache: cannot extend len {} beyond current {}",
                len,
                self.total_seq_len
            );
        }
        if len == 0 {
            self.reset();
            return Ok(());
        }
        if len == self.total_seq_len {
            return Ok(());
        }

        // Narrow the MLX arrays along dim 0 (seq dimension)
        // We need to re-compress from scratch or narrow the compressed representation.
        // Since compression is lossy, we narrow the compressed arrays directly.
        // This truncates at compressed-token granularity.
        if let Some(ref k) = self.keys {
            let k_narrowed = k
                .index(..len as i32)
                .map_err(|e| candle_core::Error::Msg(format!("MLX narrow K: {e}")))?;
            self.keys = Some(k_narrowed);
        }
        if let Some(ref v) = self.values {
            let v_narrowed = v
                .index(..len as i32)
                .map_err(|e| candle_core::Error::Msg(format!("MLX narrow V: {e}")))?;
            self.values = Some(v_narrowed);
        }
        self.total_seq_len = len;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::{CompressionBits, CompressionPolicy, KvCompressionConfig};

    fn test_config() -> KvCompressionConfig {
        KvCompressionConfig {
            bits: CompressionBits::Four,
            policy: CompressionPolicy::Always,
        }
    }

    #[test]
    fn test_append_single_token() {
        let config = test_config();
        let mut cache = MlxCompressedCache::new(2, 4, &config).unwrap();
        // [batch=1, heads=2, seq=1, head_dim=4]
        let k = Tensor::zeros((1, 2, 1, 4), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1, 2, 1, 4), DType::F32, &Device::Cpu).unwrap();
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.current_seq_len(), 1);
    }

    #[test]
    fn test_append_multiple_tokens() {
        let config = test_config();
        let mut cache = MlxCompressedCache::new(2, 4, &config).unwrap();
        let k = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.current_seq_len(), 3);

        // Append more
        let k2 = Tensor::zeros((1, 2, 2, 4), DType::F32, &Device::Cpu).unwrap();
        let v2 = Tensor::zeros((1, 2, 2, 4), DType::F32, &Device::Cpu).unwrap();
        cache.append(&k2, &v2).unwrap();
        assert_eq!(cache.current_seq_len(), 5);
    }

    #[test]
    fn test_get_returns_correct_shape() {
        let config = test_config();
        let mut cache = MlxCompressedCache::new(2, 4, &config).unwrap();
        let k = Tensor::randn(0.0f32, 1.0, (1, 2, 3, 4), &Device::Cpu).unwrap();
        let v = Tensor::randn(0.0f32, 1.0, (1, 2, 3, 4), &Device::Cpu).unwrap();
        cache.append(&k, &v).unwrap();

        let k_out = cache.k().unwrap().unwrap();
        let v_out = cache.v().unwrap().unwrap();
        assert_eq!(k_out.dims(), &[1, 2, 3, 4]);
        assert_eq!(v_out.dims(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_reset() {
        let config = test_config();
        let mut cache = MlxCompressedCache::new(2, 4, &config).unwrap();
        let k = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        cache.append(&k, &v).unwrap();
        cache.reset();
        assert_eq!(cache.current_seq_len(), 0);
        assert!(cache.k().unwrap().is_none());
    }

    #[test]
    fn test_set_len_truncation() {
        let config = test_config();
        let mut cache = MlxCompressedCache::new(2, 4, &config).unwrap();
        let k = Tensor::zeros((1, 2, 5, 4), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1, 2, 5, 4), DType::F32, &Device::Cpu).unwrap();
        cache.append(&k, &v).unwrap();
        assert_eq!(cache.current_seq_len(), 5);

        cache.set_len(3).unwrap();
        assert_eq!(cache.current_seq_len(), 3);
    }

    #[test]
    fn test_set_len_extend_fails() {
        let config = test_config();
        let mut cache = MlxCompressedCache::new(2, 4, &config).unwrap();
        let k = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        let v = Tensor::zeros((1, 2, 3, 4), DType::F32, &Device::Cpu).unwrap();
        cache.append(&k, &v).unwrap();
        assert!(cache.set_len(10).is_err());
    }
}
