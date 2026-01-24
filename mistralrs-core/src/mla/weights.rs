//! MLA weight caching for efficient decode operations.

#[cfg(all(feature = "cuda", target_family = "unix"))]
use std::sync::Mutex;

#[cfg(all(feature = "cuda", target_family = "unix"))]
use candle_core::{Device, Result, Tensor, D};

#[cfg(all(feature = "cuda", target_family = "unix"))]
use mistralrs_quant::QuantMethod;

/// Cached MLA weight matrices for efficient decode operations.
///
/// Stores the precomputed w_uk and w_uv_t matrices extracted from kv_b_proj.
/// These are computed lazily on first use and cached for subsequent calls.
pub struct MlaWeights {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    weights: Option<Mutex<Option<(Tensor, Tensor)>>>,
    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
    _phantom: std::marker::PhantomData<()>,
}

impl MlaWeights {
    /// Create a new MlaWeights instance.
    ///
    /// If `paged_attn_enabled` is true and we're on CUDA, allocates the mutex for caching.
    /// Otherwise, the weights are not cached (MLA decode won't be used).
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub fn new(paged_attn_enabled: bool, device: Option<&Device>) -> Self {
        let weights = if paged_attn_enabled {
            if let Some(device) = device {
                if matches!(device, Device::Cuda(_)) {
                    Some(Mutex::new(None))
                } else {
                    None
                }
            } else {
                // If no device is provided, assume we may need it
                Some(Mutex::new(None))
            }
        } else {
            None
        };
        Self { weights }
    }

    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
    pub fn new(_paged_attn_enabled: bool, _device: Option<&candle_core::Device>) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute MLA weights from the kv_b_proj layer.
    ///
    /// Extracts w_uk (for K nope projection) and w_uv_t (transposed V projection)
    /// from the kv_b_proj weight matrix.
    ///
    /// # Arguments
    /// * `kv_b_proj` - The kv_b_proj quantized layer
    /// * `device` - Target device for the weights
    /// * `num_attention_heads` - Number of attention heads
    /// * `kv_lora_rank` - KV latent dimension
    /// * `qk_nope_head_dim` - Non-positional head dimension
    /// * `v_head_dim` - Value head dimension
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub fn compute_weights(
        kv_b_proj: &dyn QuantMethod,
        device: &Device,
        num_attention_heads: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        v_head_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let mut w = kv_b_proj.dequantize_w()?;
        if !w.device().same_device(device) {
            w = w.to_device(device)?;
        }
        let (out_dim, in_dim) = w.dims2()?;
        if in_dim != kv_lora_rank {
            candle_core::bail!(
                "kv_b_proj weight in_dim mismatch: expected {}, got {}",
                kv_lora_rank,
                in_dim
            );
        }
        let per_head_dim = qk_nope_head_dim + v_head_dim;
        if out_dim != num_attention_heads * per_head_dim {
            candle_core::bail!(
                "kv_b_proj weight out_dim mismatch: expected {}, got {}",
                num_attention_heads * per_head_dim,
                out_dim
            );
        }
        let w = w.reshape((num_attention_heads, per_head_dim, kv_lora_rank))?;
        let w_uk = w.narrow(D::Minus2, 0, qk_nope_head_dim)?.contiguous()?;
        let w_uv = w
            .narrow(D::Minus2, qk_nope_head_dim, v_head_dim)?
            .contiguous()?;
        let w_uv_t = w_uv.transpose(1, 2)?.contiguous()?;
        Ok((w_uk, w_uv_t))
    }

    /// Get or compute the MLA weights.
    ///
    /// Returns cached weights if available, otherwise computes and caches them.
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    pub fn get_or_compute(
        &self,
        kv_b_proj: &dyn QuantMethod,
        device: &Device,
        num_attention_heads: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        v_head_dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let Some(mla_weights) = &self.weights else {
            candle_core::bail!("MLA weights are not initialized on this device");
        };
        let mut guard = mla_weights.lock().expect("MLA weights mutex was poisoned");
        if let Some((w_uk, w_uv_t)) = guard.as_ref() {
            return Ok((w_uk.clone(), w_uv_t.clone()));
        }
        let (w_uk, w_uv_t) = Self::compute_weights(
            kv_b_proj,
            device,
            num_attention_heads,
            kv_lora_rank,
            qk_nope_head_dim,
            v_head_dim,
        )?;
        *guard = Some((w_uk.clone(), w_uv_t.clone()));
        Ok((w_uk, w_uv_t))
    }

    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
    #[allow(dead_code)]
    pub fn get_or_compute(
        &self,
        _kv_b_proj: &dyn mistralrs_quant::QuantMethod,
        _device: &candle_core::Device,
        _num_attention_heads: usize,
        _kv_lora_rank: usize,
        _qk_nope_head_dim: usize,
        _v_head_dim: usize,
    ) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
        candle_core::bail!("MLA weights require CUDA support")
    }
}
