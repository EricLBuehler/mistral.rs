//! Hybrid cache for models that mix attention and Mamba layers (e.g., GraniteMoeHybrid)

use candle_core::{Device, Result, Tensor};

use super::KvCache;

/// Cache for a single Mamba layer - stores conv state and SSM state
#[derive(Clone, Debug)]
pub struct MambaLayerCache {
    /// Convolution state: (batch, conv_dim, d_conv)
    pub conv_state: Tensor,
    /// SSM state: (batch, n_heads, head_dim, d_state)
    pub ssm_state: Tensor,
    /// Current sequence length offset
    pub seqlen_offset: usize,
}

impl MambaLayerCache {
    pub fn new(
        batch_size: usize,
        conv_dim: usize,
        d_conv: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        dtype: candle_core::DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            conv_state: Tensor::zeros((batch_size, conv_dim, d_conv), dtype, device)?,
            ssm_state: Tensor::zeros((batch_size, n_heads, head_dim, d_state), dtype, device)?,
            seqlen_offset: 0,
        })
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.ssm_state = self.ssm_state.zeros_like()?;
        self.seqlen_offset = 0;
        Ok(())
    }
}

/// Per-layer cache that can be either attention (KV) or Mamba
#[derive(Clone, Debug)]
pub enum HybridLayerCache {
    Attention(KvCache),
    Mamba(MambaLayerCache),
}

impl HybridLayerCache {
    pub fn current_seq_len(&self) -> usize {
        match self {
            Self::Attention(kv) => kv.current_seq_len(),
            Self::Mamba(mamba) => mamba.seqlen_offset,
        }
    }

    pub fn reset(&mut self) {
        match self {
            Self::Attention(kv) => kv.reset(),
            Self::Mamba(mamba) => {
                let _ = mamba.reset();
            }
        }
    }

    pub fn as_kv_cache(&self) -> Option<&KvCache> {
        match self {
            Self::Attention(kv) => Some(kv),
            Self::Mamba(_) => None,
        }
    }

    pub fn as_kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        match self {
            Self::Attention(kv) => Some(kv),
            Self::Mamba(_) => None,
        }
    }

    pub fn as_mamba_cache(&self) -> Option<&MambaLayerCache> {
        match self {
            Self::Attention(_) => None,
            Self::Mamba(mamba) => Some(mamba),
        }
    }

    pub fn as_mamba_cache_mut(&mut self) -> Option<&mut MambaLayerCache> {
        match self {
            Self::Attention(_) => None,
            Self::Mamba(mamba) => Some(mamba),
        }
    }
}

/// Layer type indicator for hybrid models
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridLayerType {
    Attention,
    Mamba,
}

/// Configuration for creating a hybrid cache
#[derive(Clone, Debug)]
pub struct HybridCacheConfig {
    pub layer_types: Vec<HybridLayerType>,
    pub max_seq_len: usize,
    // Mamba-specific config
    pub mamba_conv_dim: usize,
    pub mamba_d_conv: usize,
    pub mamba_n_heads: usize,
    pub mamba_head_dim: usize,
    pub mamba_d_state: usize,
}

/// Hybrid cache that stores per-layer caches for mixed attention/Mamba models
#[derive(Clone, Debug)]
pub struct HybridCache {
    pub caches: Vec<HybridLayerCache>,
    config: HybridCacheConfig,
}

impl HybridCache {
    pub const CACHE_GROW_SIZE: usize = 512;

    pub fn new(
        config: HybridCacheConfig,
        dtype: candle_core::DType,
        device: &Device,
    ) -> Result<Self> {
        let mut caches = Vec::with_capacity(config.layer_types.len());

        for layer_type in &config.layer_types {
            let cache = match layer_type {
                HybridLayerType::Attention => HybridLayerCache::Attention(KvCache::new_normal(
                    2,
                    config.max_seq_len,
                    Self::CACHE_GROW_SIZE,
                )),
                HybridLayerType::Mamba => HybridLayerCache::Mamba(MambaLayerCache::new(
                    1, // batch_size = 1 initially, will be resized
                    config.mamba_conv_dim,
                    config.mamba_d_conv,
                    config.mamba_n_heads,
                    config.mamba_head_dim,
                    config.mamba_d_state,
                    dtype,
                    device,
                )?),
            };
            caches.push(cache);
        }

        Ok(Self { caches, config })
    }

    pub fn seqlen(&self) -> usize {
        // Return the seqlen from the first attention layer
        for cache in &self.caches {
            if let HybridLayerCache::Attention(kv) = cache {
                return kv.current_seq_len();
            }
        }
        // If no attention layers, check Mamba layers
        for cache in &self.caches {
            if let HybridLayerCache::Mamba(mamba) = cache {
                return mamba.seqlen_offset;
            }
        }
        0
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }

    pub fn layer_types(&self) -> &[HybridLayerType] {
        &self.config.layer_types
    }

    pub fn config(&self) -> &HybridCacheConfig {
        &self.config
    }

    /// Get a mutable reference to a specific layer's cache
    pub fn get_mut(&mut self, layer: usize) -> Option<&mut HybridLayerCache> {
        self.caches.get_mut(layer)
    }

    /// Get a reference to a specific layer's cache
    pub fn get(&self, layer: usize) -> Option<&HybridLayerCache> {
        self.caches.get(layer)
    }
}
