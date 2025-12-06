//! Dual Language Model architecture for VibeVoice.
//!
//! VibeVoice uses a split Qwen2 architecture:
//! - Base LM (layers 0-3): Text encoding only
//! - TTS LM (layers 4-23): Text + speech generation
//!
//! The TTS LM receives the last hidden state from the base LM and
//! adds type embeddings to distinguish text from speech positions.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{Embedding, Linear};
use mistralrs_quant::ShardedVarBuilder;
use std::sync::Arc;

use super::config::{DecoderConfig, VibeVoiceConfig};

/// Helper to create a linear layer with bias from ShardedVarBuilder
fn linear(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

/// Helper to create a linear layer without bias from ShardedVarBuilder
fn linear_no_bias(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

/// Helper to create an embedding layer from ShardedVarBuilder
fn embedding(vocab_size: usize, hidden_size: usize, vb: ShardedVarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}

/// RMS Normalization layer
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn new(size: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dtype = xs.dtype();
        let xs = xs.to_dtype(DType::F32)?;
        let variance = xs.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        xs.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .to_dtype(dtype)
    }
}

/// RoPE for VibeVoice (simplified version)
pub struct RotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(
        theta: f32,
        head_dim: usize,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, head_dim / 2, device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, max_seq_len, device)?;

        // (seq_len, head_dim/2)
        let freqs = positions
            .unsqueeze(1)?
            .broadcast_mul(&inv_freq.unsqueeze(0)?)?;

        // Interleave cos and sin: (seq_len, head_dim)
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        // Duplicate for full head_dim
        let cos_cache = Tensor::cat(&[&cos, &cos], 1)?.to_dtype(dtype)?;
        let sin_cache = Tensor::cat(&[&sin, &sin], 1)?.to_dtype(dtype)?;

        Ok(Self {
            cos_cache,
            sin_cache,
            head_dim,
        })
    }

    fn rotate_half(xs: &Tensor) -> Result<Tensor> {
        let last_dim = xs.dim(candle_core::D::Minus1)?;
        let half = last_dim / 2;
        let x1 = xs.narrow(candle_core::D::Minus1, 0, half)?;
        let x2 = xs.narrow(candle_core::D::Minus1, half, half)?;
        Tensor::cat(&[&x2.neg()?, &x1], candle_core::D::Minus1)
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, positions: &[usize]) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;

        // Get position-specific cos/sin
        let start_pos = positions.first().copied().unwrap_or(0);
        let cos = self.cos_cache.narrow(0, start_pos, seq_len)?;
        let sin = self.sin_cache.narrow(0, start_pos, seq_len)?;

        // Expand for batch and heads: (1, 1, seq_len, head_dim)
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

        // Apply rotary: q * cos + rotate_half(q) * sin
        let q_rot = (q.broadcast_mul(&cos)? + Self::rotate_half(q)?.broadcast_mul(&sin)?)?;
        let k_rot = (k.broadcast_mul(&cos)? + Self::rotate_half(k)?.broadcast_mul(&sin)?)?;

        Ok((q_rot, k_rot))
    }
}

/// Qwen2-style attention layer
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: Arc<RotaryEmbedding>,
    softmax_scale: f32,
}

impl Attention {
    fn new(cfg: &DecoderConfig, rope: Arc<RotaryEmbedding>, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_size / num_heads;

        // Qwen2 has bias on q/k/v but not on o_proj
        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_ids: &[usize],
        kv_cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape to (batch, heads, seq, head_dim)
        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = self.rope.forward(&q, &k, position_ids)?;

        // Handle KV cache
        let (k, v) = match kv_cache {
            Some(cache) => cache.append(&k, &v)?,
            None => (k, v),
        };

        // Repeat KV for GQA
        let k = repeat_kv(k, self.num_heads / self.num_kv_heads)?;
        let v = repeat_kv(v, self.num_heads / self.num_kv_heads)?;

        // Attention
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.softmax_scale as f64)?;

        // Causal mask
        let attn_weights = apply_causal_mask(&attn_weights)?;

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

/// Repeat KV heads for grouped query attention
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b, num_kv_heads, seq_len, head_dim) = xs.dims4()?;
    xs.unsqueeze(2)?
        .expand((b, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((b, num_kv_heads * n_rep, seq_len, head_dim))
}

/// Apply causal attention mask
fn apply_causal_mask(attn_weights: &Tensor) -> Result<Tensor> {
    let seq_len = attn_weights.dim(2)?;
    let k_len = attn_weights.dim(3)?;
    let device = attn_weights.device();
    let dtype = attn_weights.dtype();

    // Create causal mask - 1 means mask out (j > i), 0 means keep (j <= i)
    // Accounting for KV cache: the query positions are offset by (k_len - seq_len)
    let offset = k_len.saturating_sub(seq_len);
    let mask: Vec<u8> = (0..seq_len)
        .flat_map(|i| {
            (0..k_len).map(move |j| {
                // j > i + offset means future position, should mask
                u8::from(j > i + offset)
            })
        })
        .collect();
    let mask = Tensor::from_slice(&mask, (seq_len, k_len), device)?;

    // Expand to batch and heads
    let mask = mask.unsqueeze(0)?.unsqueeze(0)?;
    let mask = mask.broadcast_as(attn_weights.shape())?;

    // Apply mask (set masked positions to -inf)
    let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?.to_dtype(dtype)?;
    let zeros = Tensor::zeros(attn_weights.shape(), dtype, device)?;
    mask.where_cond(&neg_inf.broadcast_as(attn_weights.shape())?, &zeros)?
        .broadcast_add(attn_weights)
}

/// Qwen2-style MLP with SwiGLU
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &DecoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Decoder layer (Qwen2-style)
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &DecoderConfig, rope: Arc<RotaryEmbedding>, vb: ShardedVarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, rope, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_ids: &[usize],
        kv_cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        // Self attention with residual
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, position_ids, kv_cache)?;
        let xs = (xs + residual)?;

        // MLP with residual
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

/// KV Cache for attention
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let (k, v) = match (&self.k, &self.v) {
            (Some(prev_k), Some(prev_v)) => {
                let k = Tensor::cat(&[prev_k, k], 2)?;
                let v = Tensor::cat(&[prev_v, v], 2)?;
                (k, v)
            }
            _ => (k.clone(), v.clone()),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    pub fn len(&self) -> usize {
        self.k.as_ref().map(|k| k.dim(2).unwrap_or(0)).unwrap_or(0)
    }
}

impl Default for KvCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Layer-wise KV cache for the entire model
pub struct LayerKvCache {
    caches: Vec<KvCache>,
}

impl LayerKvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            caches: (0..num_layers).map(|_| KvCache::new()).collect(),
        }
    }

    pub fn get_mut(&mut self, layer_idx: usize) -> Option<&mut KvCache> {
        self.caches.get_mut(layer_idx)
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    pub fn len(&self) -> usize {
        self.caches.first().map(|c| c.len()).unwrap_or(0)
    }
}

/// Base Language Model (lower layers for text encoding)
pub struct BaseLM {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    num_layers: usize,
}

impl BaseLM {
    pub fn new(cfg: &VibeVoiceConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let decoder_cfg = &cfg.decoder_config;
        let num_layers = cfg.base_lm_layers();
        let device = vb.device();
        let dtype = vb.dtype();

        let embed_tokens = embedding(
            decoder_cfg.vocab_size,
            decoder_cfg.hidden_size,
            vb.pp("embed_tokens"),
        )?;

        let rope = Arc::new(RotaryEmbedding::new(
            decoder_cfg.rope_theta as f32,
            decoder_cfg.hidden_size / decoder_cfg.num_attention_heads,
            decoder_cfg.max_position_embeddings,
            device,
            dtype,
        )?);

        let mut layers_vec = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = DecoderLayer::new(decoder_cfg, rope.clone(), vb.pp("layers").pp(i))?;
            layers_vec.push(layer);
        }

        Ok(Self {
            embed_tokens,
            layers: layers_vec,
            num_layers,
        })
    }

    /// Forward pass through base LM layers
    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_ids: &[usize],
        kv_cache: &mut Option<LayerKvCache>,
    ) -> Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = kv_cache.as_mut().and_then(|c| c.get_mut(i));
            xs = layer.forward(&xs, position_ids, cache)?;
        }

        Ok(xs)
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

/// TTS Language Model (upper layers for speech generation)
pub struct TtsLM {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    num_layers: usize,
}

impl TtsLM {
    pub fn new(cfg: &VibeVoiceConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let decoder_cfg = &cfg.decoder_config;
        let num_layers = cfg.tts_backbone_num_hidden_layers;
        let device = vb.device();
        let dtype = vb.dtype();

        // TTS LM has its own embedding layer
        let embed_tokens = embedding(
            decoder_cfg.vocab_size,
            decoder_cfg.hidden_size,
            vb.pp("embed_tokens"),
        )?;

        let rope = Arc::new(RotaryEmbedding::new(
            decoder_cfg.rope_theta as f32,
            decoder_cfg.hidden_size / decoder_cfg.num_attention_heads,
            decoder_cfg.max_position_embeddings,
            device,
            dtype,
        )?);

        let mut layers_vec = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = DecoderLayer::new(decoder_cfg, rope.clone(), vb.pp("layers").pp(i))?;
            layers_vec.push(layer);
        }

        let norm = RmsNorm::new(
            decoder_cfg.hidden_size,
            decoder_cfg.rms_norm_eps,
            vb.pp("norm"),
        )?;

        Ok(Self {
            embed_tokens,
            layers: layers_vec,
            norm,
            num_layers,
        })
    }

    /// Forward pass with base LM hidden states injected
    pub fn forward(
        &self,
        input_embeds: &Tensor,
        type_embeds: Option<&Tensor>,
        position_ids: &[usize],
        kv_cache: &mut Option<LayerKvCache>,
    ) -> Result<Tensor> {
        let mut xs = input_embeds.clone();

        // Add type embeddings if provided
        if let Some(type_emb) = type_embeds {
            xs = (xs + type_emb)?;
        }

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = kv_cache.as_mut().and_then(|c| c.get_mut(i));
            xs = layer.forward(&xs, position_ids, cache)?;
        }

        self.norm.forward(&xs)
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

/// Acoustic Connector - projects acoustic latents to LM hidden size
pub struct AcousticConnector {
    fc1: Linear,
    fc2: Linear,
    norm: RmsNorm,
}

impl AcousticConnector {
    pub fn new(
        latent_dim: usize,
        hidden_size: usize,
        eps: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let fc1 = linear(latent_dim, hidden_size, vb.pp("fc1"))?;
        let fc2 = linear(hidden_size, hidden_size, vb.pp("fc2"))?;
        let norm = RmsNorm::new(hidden_size, eps, vb.pp("norm"))?;
        Ok(Self { fc1, fc2, norm })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.silu()?;
        let xs = self.fc2.forward(&xs)?;
        self.norm.forward(&xs)
    }
}

/// Type embeddings to distinguish text from speech positions
pub struct TypeEmbeddings {
    embedding: Embedding,
}

impl TypeEmbeddings {
    pub fn new(hidden_size: usize, vb: ShardedVarBuilder) -> Result<Self> {
        // 2 types: 0 = text, 1 = speech
        let embedding = embedding(2, hidden_size, vb)?;
        Ok(Self { embedding })
    }

    pub fn forward(&self, type_ids: &Tensor) -> Result<Tensor> {
        self.embedding.forward(type_ids)
    }
}

/// EOS Classifier - predicts end of speech
pub struct EosClassifier {
    fc1: Linear,
    fc2: Linear,
}

impl EosClassifier {
    pub fn new(hidden_size: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let fc1 = linear(hidden_size, hidden_size, vb.pp("fc1"))?;
        let fc2 = linear(hidden_size, 1, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.silu()?;
        self.fc2.forward(&xs)
    }

    /// Predict EOS probability (sigmoid of logit)
    pub fn predict_eos(&self, xs: &Tensor) -> Result<bool> {
        let logit = self.forward(xs)?;
        let prob = candle_nn::ops::sigmoid(&logit)?;
        // Get last position
        let seq_len = prob.dim(1)?;
        let last_prob = prob.i((.., seq_len - 1, 0))?;
        let prob_val: f32 = last_prob.mean_all()?.to_scalar()?;
        Ok(prob_val > 0.5)
    }
}
