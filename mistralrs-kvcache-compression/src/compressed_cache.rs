use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use turboquant::{TurboQuant, TurboVectorMse};

use crate::config::KvCompressionConfig;

/// A single compressed KV vector slot (one token, one head).
type CompressedSlot = TurboVectorMse;

/// Per-head TurboQuant quantizer shared across all tokens of a layer.
/// One instance per key, one per value.
#[derive(Debug, Clone)]
struct HeadQuantizer {
    tq: TurboQuant,
    /// Compressed token slots in sequence order.
    slots: Vec<CompressedSlot>,
}

impl HeadQuantizer {
    fn new(head_dim: usize, bits: u8, key_seed: u64) -> turboquant::Result<Self> {
        Ok(Self {
            tq: TurboQuant::new(head_dim, bits, key_seed)?,
            slots: Vec::new(),
        })
    }

    /// Compress and append one token's vector.
    fn push(&mut self, vec: &[f32]) -> turboquant::Result<()> {
        let compressed = self.tq.compress_mse(vec)?;
        self.slots.push(compressed);
        Ok(())
    }

    /// Decompress all stored tokens into a flat `Vec<f32>` of shape
    /// `[n_tokens, head_dim]` (row-major).
    fn decompress_all(&self) -> turboquant::Result<Vec<f32>> {
        let head_dim = self.tq.dim();
        let mut out = Vec::with_capacity(self.slots.len() * head_dim);
        for slot in &self.slots {
            let decoded = self.tq.decompress_mse(slot)?;
            out.extend_from_slice(&decoded);
        }
        Ok(out)
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Truncate to `n` slots (used when the cache is reset to a shorter length).
    fn truncate(&mut self, n: usize) {
        self.slots.truncate(n);
    }
}

/// Compressed KV cache for a single transformer layer.
///
/// Holds one `HeadQuantizer` per KV head, for both keys and values.
/// A small "tail" buffer stores the most recent `threshold_tokens` vectors
/// uncompressed to preserve accuracy in the warm-up window.
#[derive(Debug, Clone)]
pub struct CompressedLayerCache {
    /// One quantizer per KV head for keys.
    key_heads: Vec<HeadQuantizer>,
    /// One quantizer per KV head for values.
    val_heads: Vec<HeadQuantizer>,
    /// Number of compressed tokens in the cache.
    n_compressed: usize,
    /// Configuration (bits, threshold policy).
    config: KvCompressionConfig,
    /// Head dimension (must be a power of two: 64, 128, 256...).
    head_dim: usize,
    /// Number of KV heads.
    num_kv_heads: usize,
}

impl CompressedLayerCache {
    /// Create a new compressed layer cache.
    ///
    /// # Arguments
    /// * `num_kv_heads` – number of KV heads for this layer.
    /// * `head_dim` – per-head dimension (must be a power of two ≥ 16).
    /// * `config` – compression configuration.
    /// * `layer_idx` – used to derive unique rotation seeds per layer.
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        config: KvCompressionConfig,
        layer_idx: usize,
    ) -> turboquant::Result<Self> {
        let bits = config.bits.as_u8();
        let mut key_heads = Vec::with_capacity(num_kv_heads);
        let mut val_heads = Vec::with_capacity(num_kv_heads);
        for h in 0..num_kv_heads {
            // Seeds are deterministic and unique per (layer, head) combination.
            let key_seed = (layer_idx as u64) * 1_000_000 + (h as u64) * 2;
            let val_seed = key_seed + 1;
            key_heads.push(HeadQuantizer::new(head_dim, bits, key_seed)?);
            val_heads.push(HeadQuantizer::new(head_dim, bits, val_seed)?);
        }
        Ok(Self {
            key_heads,
            val_heads,
            n_compressed: 0,
            config,
            head_dim,
            num_kv_heads,
        })
    }

    /// Compress and store one new token's KV vectors across all heads.
    ///
    /// `k_slice` shape: `[num_kv_heads, head_dim]` (flat row-major).
    /// `v_slice` shape: `[num_kv_heads, head_dim]` (flat row-major).
    pub fn push_token(
        &mut self,
        k_slice: &[f32],
        v_slice: &[f32],
    ) -> turboquant::Result<()> {
        for h in 0..self.num_kv_heads {
            let k_start = h * self.head_dim;
            let k_end = k_start + self.head_dim;
            self.key_heads[h].push(&k_slice[k_start..k_end])?;
            self.val_heads[h].push(&v_slice[k_start..k_end])?;
        }
        self.n_compressed += 1;
        Ok(())
    }

    /// Decompress the entire compressed history into candle `Tensor`s.
    ///
    /// Returns `(keys, values)` each of shape `[1, num_kv_heads, n_compressed, head_dim]`.
    pub fn decompress_to_tensors(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let n = self.n_compressed;
        if n == 0 {
            let shape = (1usize, self.num_kv_heads, 0usize, self.head_dim);
            let k = Tensor::zeros(shape, DType::F32, device)?;
            let v = Tensor::zeros(shape, DType::F32, device)?;
            return Ok((k, v));
        }

        // Collect flat key/value data: [num_kv_heads, n_compressed, head_dim]
        let mut k_data = Vec::with_capacity(self.num_kv_heads * n * self.head_dim);
        let mut v_data = Vec::with_capacity(self.num_kv_heads * n * self.head_dim);

        for h in 0..self.num_kv_heads {
            let kd = self.key_heads[h].decompress_all().map_err(|e| {
                candle_core::Error::Msg(format!("TurboQuant key decompress error: {e}"))
            })?;
            let vd = self.val_heads[h].decompress_all().map_err(|e| {
                candle_core::Error::Msg(format!("TurboQuant value decompress error: {e}"))
            })?;
            k_data.extend_from_slice(&kd);
            v_data.extend_from_slice(&vd);
        }

        // Shape: [num_kv_heads, n_compressed, head_dim] → transpose to [1, n, num_kv_heads, head_dim]
        // mistral.rs convention for normal cache: [batch=1, num_kv_heads, seq_len, head_dim]
        let k = Tensor::from_vec(k_data, (self.num_kv_heads, n, self.head_dim), device)?
            .unsqueeze(0)?; // [1, num_kv_heads, n, head_dim]
        let v = Tensor::from_vec(v_data, (self.num_kv_heads, n, self.head_dim), device)?
            .unsqueeze(0)?; // [1, num_kv_heads, n, head_dim]

        Ok((k, v))
    }

    /// Number of compressed token slots stored.
    pub fn n_compressed(&self) -> usize {
        self.n_compressed
    }

    /// Reset the compressed cache (drop all slots).
    pub fn reset(&mut self) {
        for h in &mut self.key_heads {
            h.slots.clear();
        }
        for h in &mut self.val_heads {
            h.slots.clear();
        }
        self.n_compressed = 0;
    }

    /// Truncate stored slots to `n` tokens.
    pub fn truncate(&mut self, n: usize) {
        for h in &mut self.key_heads {
            h.truncate(n);
        }
        for h in &mut self.val_heads {
            h.truncate(n);
        }
        self.n_compressed = self.n_compressed.min(n);
    }

    /// Returns the compression policy threshold token count.
    pub fn threshold_tokens(&self) -> usize {
        self.config.threshold_tokens()
    }
}

/// Thread-safe shared handle for a `CompressedLayerCache`.
pub type SharedCompressedLayerCache = Arc<parking_lot::Mutex<CompressedLayerCache>>;

/// Construct a `SharedCompressedLayerCache`.
pub fn new_shared_compressed_layer_cache(
    num_kv_heads: usize,
    head_dim: usize,
    config: KvCompressionConfig,
    layer_idx: usize,
) -> turboquant::Result<SharedCompressedLayerCache> {
    Ok(Arc::new(parking_lot::Mutex::new(CompressedLayerCache::new(
        num_kv_heads,
        head_dim,
        config,
        layer_idx,
    )?)))
}

/// Compress the token at `seq_pos` from a 4-D KV tensor and push it into `cache`.
///
/// `k` and `v` are expected in NormalCache format:
/// `[batch_size, num_kv_heads, seq_len, head_dim]`.
///
/// `seq_dim` is the sequence-length axis (typically 2).
pub fn compress_token_at_pos(
    cache: &mut CompressedLayerCache,
    k: &Tensor,
    v: &Tensor,
    seq_pos: usize,
    seq_dim: usize,
) -> Result<()> {
    let num_kv_heads = k.dim(1)?;
    let head_dim = k.dim(3)?;

    // Extract token at seq_pos: [batch, num_kv_heads, 1, head_dim] → [num_kv_heads * head_dim]
    let k_tok = k
        .narrow(seq_dim, seq_pos, 1)?
        .squeeze(seq_dim)?   // [batch, num_kv_heads, head_dim]
        .squeeze(0)?         // [num_kv_heads, head_dim]
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?;
    let v_tok = v
        .narrow(seq_dim, seq_pos, 1)?
        .squeeze(seq_dim)?
        .squeeze(0)?
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?;

    let k_vec = k_tok.to_vec1::<f32>()?;
    let v_vec = v_tok.to_vec1::<f32>()?;

    if k_vec.len() != num_kv_heads * head_dim {
        candle_core::bail!(
            "compress_token_at_pos: key slice length {} != num_kv_heads * head_dim ({} * {})",
            k_vec.len(), num_kv_heads, head_dim
        );
    }

    cache.push_token(&k_vec, &v_vec).map_err(|e| {
        candle_core::Error::Msg(format!("TurboQuant compress error: {e}"))
    })?;
    Ok(())
}

/// Compress all tokens from a full KV tensor into a fresh `CompressedLayerCache`.
///
/// Compresses tokens `[0, seq_len)` from `k` / `v` (shape
/// `[1, num_kv_heads, seq_len, head_dim]`) into `cache`.  The caller is
/// responsible for constructing `cache` with matching `num_kv_heads` /
/// `head_dim` / `config`.
pub fn compress_all_tokens(
    cache: &mut CompressedLayerCache,
    k: &Tensor,
    v: &Tensor,
    seq_dim: usize,
) -> Result<()> {
    let seq_len = k.dim(seq_dim)?;
    for pos in 0..seq_len {
        compress_token_at_pos(cache, k, v, pos, seq_dim)?;
    }
    Ok(())
}

/// Extract the per-head KV vectors from a candle `Tensor` and compress them.
///
/// `k` and `v` tensors are expected in mistral.rs NormalCache format:
/// `[batch_size, num_kv_heads, seq_len, head_dim]` in F32 or BF16/F16.
///
/// Only the **last** token (seq dimension) is compressed on each call.
pub fn compress_last_token(
    cache: &mut CompressedLayerCache,
    k: &Tensor,
    v: &Tensor,
) -> Result<()> {
    let seq_len = k.dim(2)?;
    if seq_len == 0 {
        return Ok(());
    }
    compress_token_at_pos(cache, k, v, seq_len - 1, 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CompressionBits, CompressionPolicy, KvCompressionConfig};
    use candle_core::{DType, Device, Tensor};

    fn default_config(bits: CompressionBits) -> KvCompressionConfig {
        KvCompressionConfig {
            bits,
            policy: CompressionPolicy::Always,
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b + 1e-8)
    }

    /// Build a random-ish deterministic KV tensor of shape [1, num_heads, seq_len, head_dim].
    fn make_kv_tensor(
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        seed: f32,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let total = num_heads * seq_len * head_dim;
        let data: Vec<f32> = (0..total)
            .map(|i| (i as f32 * 0.1 + seed).sin())
            .collect();
        Tensor::from_vec(data, (1, num_heads, seq_len, head_dim), device)
    }

    #[test]
    fn test_round_trip_single_token_3bit() {
        let device = Device::Cpu;
        let num_heads = 2;
        let head_dim = 64;
        let config = default_config(CompressionBits::Three);
        let mut cache = CompressedLayerCache::new(num_heads, head_dim, config, 0).unwrap();

        let k = make_kv_tensor(num_heads, 1, head_dim, 0.0, &device).unwrap();
        let v = make_kv_tensor(num_heads, 1, head_dim, 1.0, &device).unwrap();

        compress_last_token(&mut cache, &k, &v).unwrap();
        assert_eq!(cache.n_compressed(), 1);

        let (dk, dv) = cache.decompress_to_tensors(&device).unwrap();
        assert_eq!(dk.dims(), &[1, num_heads, 1, head_dim]);
        assert_eq!(dv.dims(), &[1, num_heads, 1, head_dim]);

        let orig_k = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let dec_k = dk.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sim = cosine_similarity(&orig_k, &dec_k);
        assert!(sim > 0.95, "3-bit key cosine similarity {sim:.4} < 0.95");
    }

    #[test]
    fn test_round_trip_4bit_higher_fidelity() {
        let device = Device::Cpu;
        let num_heads = 4;
        let head_dim = 128;
        let config = default_config(CompressionBits::Four);
        let mut cache = CompressedLayerCache::new(num_heads, head_dim, config, 1).unwrap();

        let k = make_kv_tensor(num_heads, 1, head_dim, 2.0, &device).unwrap();
        let v = make_kv_tensor(num_heads, 1, head_dim, 3.0, &device).unwrap();

        compress_last_token(&mut cache, &k, &v).unwrap();

        let (dk, dv) = cache.decompress_to_tensors(&device).unwrap();

        let orig_k = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let dec_k = dk.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let orig_v = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let dec_v = dv.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        let sim_k = cosine_similarity(&orig_k, &dec_k);
        let sim_v = cosine_similarity(&orig_v, &dec_v);
        assert!(sim_k > 0.98, "4-bit key cosine similarity {sim_k:.4} < 0.98");
        assert!(sim_v > 0.98, "4-bit val cosine similarity {sim_v:.4} < 0.98");
    }

    #[test]
    fn test_multi_token_compress_all() {
        let device = Device::Cpu;
        let num_heads = 2;
        let seq_len = 8;
        let head_dim = 64;
        let config = default_config(CompressionBits::Three);
        let mut cache = CompressedLayerCache::new(num_heads, head_dim, config, 2).unwrap();

        let k = make_kv_tensor(num_heads, seq_len, head_dim, 4.0, &device).unwrap();
        let v = make_kv_tensor(num_heads, seq_len, head_dim, 5.0, &device).unwrap();

        compress_all_tokens(&mut cache, &k, &v, 2).unwrap();
        assert_eq!(cache.n_compressed(), seq_len);

        let (dk, dv) = cache.decompress_to_tensors(&device).unwrap();
        assert_eq!(dk.dims(), &[1, num_heads, seq_len, head_dim]);
        assert_eq!(dv.dims(), &[1, num_heads, seq_len, head_dim]);

        let orig_k = k.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let dec_k = dk.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sim = cosine_similarity(&orig_k, &dec_k);
        assert!(sim > 0.93, "multi-token 3-bit key cosine similarity {sim:.4} < 0.93");
    }

    #[test]
    fn test_reset_clears_cache() {
        let device = Device::Cpu;
        let num_heads = 1;
        let head_dim = 64;
        let config = default_config(CompressionBits::Three);
        let mut cache = CompressedLayerCache::new(num_heads, head_dim, config, 3).unwrap();

        let k = make_kv_tensor(num_heads, 4, head_dim, 6.0, &device).unwrap();
        let v = make_kv_tensor(num_heads, 4, head_dim, 7.0, &device).unwrap();
        compress_all_tokens(&mut cache, &k, &v, 2).unwrap();
        assert_eq!(cache.n_compressed(), 4);

        cache.reset();
        assert_eq!(cache.n_compressed(), 0);

        let (dk, dv) = cache.decompress_to_tensors(&device).unwrap();
        assert_eq!(dk.dims()[2], 0);
        assert_eq!(dv.dims()[2], 0);
    }

    #[test]
    fn test_truncate_reduces_count() {
        let device = Device::Cpu;
        let num_heads = 2;
        let head_dim = 64;
        let config = default_config(CompressionBits::Three);
        let mut cache = CompressedLayerCache::new(num_heads, head_dim, config, 4).unwrap();

        let k = make_kv_tensor(num_heads, 6, head_dim, 8.0, &device).unwrap();
        let v = make_kv_tensor(num_heads, 6, head_dim, 9.0, &device).unwrap();
        compress_all_tokens(&mut cache, &k, &v, 2).unwrap();
        assert_eq!(cache.n_compressed(), 6);

        cache.truncate(3);
        assert_eq!(cache.n_compressed(), 3);

        let (dk, _dv) = cache.decompress_to_tensors(&device).unwrap();
        assert_eq!(dk.dims()[2], 3);
    }

    #[test]
    fn test_empty_cache_decompress() {
        let device = Device::Cpu;
        let num_heads = 4;
        let head_dim = 128;
        let config = default_config(CompressionBits::Four);
        let cache = CompressedLayerCache::new(num_heads, head_dim, config, 5).unwrap();
        assert_eq!(cache.n_compressed(), 0);

        let (dk, dv) = cache.decompress_to_tensors(&device).unwrap();
        assert_eq!(dk.dims(), &[1, num_heads, 0, head_dim]);
        assert_eq!(dv.dims(), &[1, num_heads, 0, head_dim]);
    }
}
