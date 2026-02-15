//! Qwen3 Text Encoder for FLUX.2
//!
//! FLUX.2 uses a Qwen3ForCausalLM model as its text encoder.
//! Unlike FLUX.1 which uses T5-XXL (4096 dim output), FLUX.2 extracts hidden states
//! from layers 8, 17, 26 (0-indexed) and concatenates them to produce a 12288-dim embedding.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear};
use mistralrs_quant::ShardedVarBuilder;
use serde::Deserialize;
use std::sync::Arc;

use crate::attention::SdpaParams;
use crate::layers::{embedding, linear_no_bias, Activation, RmsNorm, RotaryEmbedding, Sdpa};

/// Layers after which to extract hidden states.
/// Diffusers uses hidden_states indices (9, 18, 27), which correspond to outputs
/// after layers 8, 17, 26 in HF convention.
const HIDDEN_STATE_LAYERS: &[usize] = &[8, 17, 26];

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TextEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// Head dimension - may be called head_dim or attention_head_dim in config
    #[serde(alias = "attention_head_dim", alias = "head_size")]
    pub head_dim: Option<usize>,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_max_position_embeddings() -> usize {
    40960
}

impl Qwen3TextEncoderConfig {
    pub fn head_dim(&self) -> usize {
        // FLUX.2's Qwen3 has non-standard config: hidden_size=3072, num_heads=32, head_dim=128
        // The explicit head_dim (128) doesn't equal hidden_size/num_heads (96)
        // So we MUST use the explicit value if provided
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn new(
        cfg: &Qwen3TextEncoderConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();

        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;

        let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
            },
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply QK-norm
        q = q.apply(&self.q_norm)?;
        k = k.apply(&self.k_norm)?;

        // Apply rotary embeddings - use zero offsets for text encoder
        let seqlen_offsets = vec![0usize; b_sz];
        let (q, k) = self.rotary_emb.forward(&q, &k, &seqlen_offsets)?;

        // Attention
        // Note: On the CUDA flash-attn path, the explicit mask is ignored and
        // flash_params=None defaults to causal=true (seq_len > 1). This is
        // acceptable because we use right-padding: real tokens are always to the
        // left, so the causal mask alone prevents them from attending to padding.
        // Padding token outputs are never used downstream.
        let attn_output =
            Sdpa.run_attention(&q, &k, &v, Some(attention_mask), None, &self.sdpa_params)?;

        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(cfg: &Qwen3TextEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;

        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = Activation::Silu.forward(&self.gate_proj.forward(xs)?)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &Qwen3TextEncoderConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(cfg, rotary_emb, vb.pp("self_attn"))?;
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

    fn forward(&self, xs: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

/// Qwen3 Text Encoder for FLUX.2
///
/// Extracts hidden states after layers 8, 17, 26 (0-indexed) and concatenates them
/// to produce a `3 * hidden_size` embedding (e.g. 3 * 4096 = 12288 for 4096-dim models).
pub struct Qwen3TextEncoder {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    hidden_size: usize,
    device: Device,
    dtype: DType,
}

impl Qwen3TextEncoder {
    pub fn new(cfg: &Qwen3TextEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_m.pp("embed_tokens"),
            &None,
        )?;

        // Create shared rotary embeddings
        // Qwen3 uses GPT-NeoX style rotation (is_gpt_neox = true)
        let head_dim = cfg.head_dim();
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta as f32,
            head_dim,
            cfg.max_position_embeddings,
            &device,
            true, // is_gpt_neox - Qwen3 uses GPT-NeoX style rotation
            dtype,
        )?);

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary_emb.clone(), vb_l.pp(i))?);
        }

        // Note: We don't use the final norm layer since we extract from intermediate layers

        Ok(Self {
            embed_tokens,
            layers,
            hidden_size: cfg.hidden_size,
            device,
            dtype,
        })
    }

    /// Forward pass that returns concatenated hidden states from layers 10, 20, 30
    ///
    /// Output shape: (batch, seq_len, 3 * hidden_size) = (batch, seq, 12288)
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Create attention mask: causal + padding
        let attention_mask = self.create_attention_mask(seq_len, attention_mask)?;

        // Embed tokens
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;

        // Collect hidden states from target layers
        let mut collected_states = Vec::with_capacity(HIDDEN_STATE_LAYERS.len());

        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, &attention_mask)?;

            // Collect hidden states from target layers (after layer forward)
            if HIDDEN_STATE_LAYERS.contains(&i) {
                collected_states.push(hidden_states.clone());
            }
        }

        // Stack: (batch, 3, seq, hidden_size)
        let stacked = Tensor::stack(&collected_states, 1)?;

        // Permute: (batch, seq, 3, hidden_size)
        let permuted = stacked.permute((0, 2, 1, 3))?;

        // Reshape: (batch, seq, 3 * hidden_size)
        let num_layers = HIDDEN_STATE_LAYERS.len();
        permuted.reshape((batch_size, seq_len, num_layers * self.hidden_size))
    }

    fn create_attention_mask(&self, seq_len: usize, padding_mask: &Tensor) -> Result<Tensor> {
        // Causal mask where positions can only attend to previous positions
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
            .collect();

        let causal = Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), &self.device)?
            .to_dtype(self.dtype)?;

        // Padding mask: 1 for real tokens, 0 for pad tokens -> 0 / -1e9 bias
        // Using -1e9 instead of NEG_INFINITY to avoid NaN from 0*inf in arithmetic
        // and to sidestep Metal's missing where_cond F32 kernel.
        let padding = padding_mask
            .to_dtype(DType::F32)?
            .unsqueeze(1)?
            .unsqueeze(1)?;
        let pad_bias = ((padding - 1.0)? * 1e9)?.to_dtype(self.dtype)?;
        causal.broadcast_add(&pad_bias)
    }

    #[allow(dead_code)]
    pub fn device(&self) -> &Device {
        &self.device
    }

    #[allow(dead_code)]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Returns the output dimension of the text encoder (3 * hidden_size = 12288)
    pub fn output_dim(&self) -> usize {
        HIDDEN_STATE_LAYERS.len() * self.hidden_size
    }
}
