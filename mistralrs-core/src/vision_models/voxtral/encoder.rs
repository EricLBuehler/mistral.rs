#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use candle_core::{DType, Module, Result, Tensor};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    attention::SdpaParams,
    layers::{CausalMasker, MatMul, RmsNorm, RotaryEmbedding, Sdpa},
    layers_masker::PastKvLenCache,
    pipeline::{KvCache, NormalCache},
};

use super::config::WhisperEncoderArgs;

pub(super) struct EncoderAttention {
    pub(super) wq: Arc<dyn QuantMethod>,
    pub(super) wk: Arc<dyn QuantMethod>,
    pub(super) wv: Arc<dyn QuantMethod>,
    pub(super) wo: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    sdpa_params: SdpaParams,
}

impl EncoderAttention {
    fn new(
        cfg: &WhisperEncoderArgs,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let dim = cfg.dim;
        let num_heads = cfg.n_heads;
        let num_kv_heads = cfg.n_kv_heads;
        let head_dim = cfg.head_dim;
        let use_bias = cfg.use_biases;

        // Per-linear bias flags matching actual weight structure:
        // wq, wv, wo have bias; wk does NOT
        let wq =
            mistralrs_quant::linear_b(dim, num_heads * head_dim, use_bias, &None, vb.pp("wq"))?;
        let wk =
            mistralrs_quant::linear_b(dim, num_kv_heads * head_dim, false, &None, vb.pp("wk"))?;
        let wv =
            mistralrs_quant::linear_b(dim, num_kv_heads * head_dim, use_bias, &None, vb.pp("wv"))?;
        let wo =
            mistralrs_quant::linear_b(num_heads * head_dim, dim, use_bias, &None, vb.pp("wo"))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            num_kv_heads,
            head_dim,
            rotary_emb,
            sdpa_params: SdpaParams {
                n_kv_groups: num_heads / num_kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: cfg.sliding_window,
                sinks: None,
            },
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let q = MatMul.qmethod_matmul(xs, &*self.wq)?;
        let k = MatMul.qmethod_matmul(xs, &*self.wk)?;
        let v = MatMul.qmethod_matmul(xs, &*self.wv)?;

        let (q, k, v) = if q_len != 1 {
            let q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.num_heads, q_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.num_kv_heads, q_len, self.head_dim))?;
            (q, k, v)
        };

        let (q, k) = self.rotary_emb.forward(&q, &k, seqlen_offsets)?;

        let (k, v) = kv_cache.append(&k, &v)?;

        let attn_output = Sdpa.run_attention(
            &q,
            &k,
            &v,
            attention_mask,
            None, // no flash params for encoder (causal via mask)
            &self.sdpa_params,
        )?;

        let attn_output = if attention_mask.is_some() {
            attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?
        } else {
            attn_output.reshape((b_sz, q_len, ()))?
        };
        MatMul.qmethod_matmul(&attn_output, &*self.wo)
    }
}

pub(super) struct EncoderMlp {
    pub(super) w1: Arc<dyn QuantMethod>, // gate
    pub(super) w2: Arc<dyn QuantMethod>, // down
    pub(super) w3: Arc<dyn QuantMethod>, // up
}

impl EncoderMlp {
    fn new(cfg: &WhisperEncoderArgs, vb: ShardedVarBuilder) -> Result<Self> {
        let dim = cfg.dim;
        let hidden_dim = cfg.hidden_dim;
        let use_bias = cfg.use_biases;

        // Per-linear bias flags: only w2 has bias; w1, w3 do not
        let w1 = mistralrs_quant::linear_b(dim, hidden_dim, false, &None, vb.pp("w1"))?;
        let w2 = mistralrs_quant::linear_b(hidden_dim, dim, use_bias, &None, vb.pp("w2"))?;
        let w3 = mistralrs_quant::linear_b(dim, hidden_dim, false, &None, vb.pp("w3"))?;

        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // SwiGLU: silu(w1(x)) * w3(x), then w2
        let gate = MatMul.qmethod_matmul(xs, &*self.w1)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let up = MatMul.qmethod_matmul(xs, &*self.w3)?;
        let xs = (gate * up)?;
        MatMul.qmethod_matmul(&xs, &*self.w2)
    }
}

pub(super) struct EncoderLayer {
    pub(super) attention: EncoderAttention,
    pub(super) feed_forward: EncoderMlp,
    pub(super) attention_norm: RmsNorm,
    pub(super) ffn_norm: RmsNorm,
}

impl EncoderLayer {
    fn new(
        cfg: &WhisperEncoderArgs,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let attention = EncoderAttention::new(cfg, rotary_emb, vb.pp("attention"))?;
        let feed_forward = EncoderMlp::new(cfg, vb.pp("feed_forward"))?;
        let attention_norm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("attention_norm"))?;
        let ffn_norm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("ffn_norm"))?;
        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.attention_norm.forward(xs)?;
        let xs = self
            .attention
            .forward(&xs, attention_mask, seqlen_offsets, kv_cache)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self.ffn_norm.forward(&xs)?;
        let xs = self.feed_forward.forward(&xs)?;
        residual + xs
    }
}

/// Causal Whisper-based audio encoder for Voxtral.
///
/// Unlike standard Whisper, this uses:
/// - Two Conv1d layers to project mel features to encoder dim
/// - Causal attention with sliding window (750 frames)
/// - RoPE positional embeddings
/// - SwiGLU FFN
/// - RMSNorm
pub struct VoxtralEncoder {
    pub(super) conv1: candle_nn::Conv1d,
    pub(super) conv2: candle_nn::Conv1d,
    pub(super) layers: Vec<EncoderLayer>,
    pub(super) norm: RmsNorm,
    cache: Arc<Mutex<NormalCache>>,
    num_heads: usize,
    sliding_window: Option<usize>,
    n_layers: usize,
    /// Model dtype (e.g. BF16) for the transformer layers.
    /// Conv1d weights are stored as F32 for CUDA compatibility.
    model_dtype: DType,
}

impl VoxtralEncoder {
    pub fn new(cfg: &WhisperEncoderArgs, vb: ShardedVarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let n_mels = cfg.audio_encoding_args.num_mel_bins;

        // Conv1d weights stored as F32 (CUDA Conv1d does not support BF16).
        // Causal padding: left-pad by (kernel_size - 1) * dilation, padding=0 in Conv1d.
        let vb_c1 = vb.pp("conv_layers").pp("0").pp("conv");
        let conv1 = candle_nn::Conv1d::new(
            vb_c1
                .get((cfg.dim, n_mels, 3), "weight")?
                .to_dtype(DType::F32)?,
            Some(vb_c1.get(cfg.dim, "bias")?.to_dtype(DType::F32)?),
            candle_nn::Conv1dConfig {
                padding: 0,
                stride: 1,
                ..Default::default()
            },
        );
        let vb_c2 = vb.pp("conv_layers").pp("1").pp("conv");
        let conv2 = candle_nn::Conv1d::new(
            vb_c2
                .get((cfg.dim, cfg.dim, 3), "weight")?
                .to_dtype(DType::F32)?,
            Some(vb_c2.get(cfg.dim, "bias")?.to_dtype(DType::F32)?),
            candle_nn::Conv1dConfig {
                padding: 0,
                stride: 2,
                ..Default::default()
            },
        );

        // Create RoPE embeddings for encoder
        let mut ropes = HashMap::new();
        ropes.insert(
            device.location(),
            Arc::new(RotaryEmbedding::new(
                cfg.rope_theta as f32,
                cfg.head_dim,
                1_000_000, // large max_position for encoder
                &device,
                false, // !is_gptx: consolidated.safetensors stores Q/K in interleaved layout
                dtype,
            )?),
        );

        let vb_layers = vb.pp("transformer").pp("layers");
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            layers.push(EncoderLayer::new(cfg, rotary_emb, vb_layers.pp(i))?);
        }

        let norm = RmsNorm::new(cfg.dim, cfg.norm_eps, vb.pp("transformer").pp("norm"))?;

        Ok(Self {
            conv1,
            conv2,
            layers,
            norm,
            cache: NormalCache::new_sliding(cfg.n_layers, 1_000_000, cfg.sliding_window),
            num_heads: cfg.n_heads,
            sliding_window: cfg.sliding_window,
            n_layers: cfg.n_layers,
            model_dtype: dtype,
        })
    }

    /// Forward pass through the encoder.
    /// Input: mel features [B, T, mel_bins]
    /// Output: [B, T/2, dim]
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, _t, _mel) = xs.dims3()?;

        let xs = xs.to_dtype(DType::F32)?;

        // Transpose [B, T, mel] -> [B, mel, T] for Conv1d
        let xs = xs.transpose(1, 2)?;

        // Causal Conv1: left-pad by 2, then conv(kernel=3, stride=1, padding=0)
        let xs = xs.pad_with_zeros(2, 2, 0)?;
        let xs = self.conv1.forward(&xs)?.gelu_erf()?;

        // Causal Conv2: left-pad by 1, then conv(kernel=3, stride=2, padding=0)
        // HF VoxtralRealtimeCausalConv1d stores left_pad=1 for this layer.
        let xs = xs.pad_with_zeros(2, 1, 0)?;
        let xs = self.conv2.forward(&xs)?.gelu_erf()?;

        // Transpose back [B, dim, T/2] -> [B, T/2, dim]
        let xs = xs.transpose(1, 2)?.contiguous()?;
        // Cast from F32 to model dtype for transformer layers
        let xs = xs.to_dtype(self.model_dtype)?;

        let seq_len = xs.dim(1)?;

        let mut cache = self.cache.lock().expect("Encoder cache lock poisoned");

        // Create causal mask with sliding window for the encoder
        let seqlen_offsets = vec![0usize; b_sz];
        let dummy_toks = Tensor::zeros((b_sz, seq_len), DType::U32, xs.device())?;
        let attention_mask = CausalMasker.make_sliding_window_causal_mask_matrix(
            &dummy_toks,
            &cache.0 as &dyn PastKvLenCache,
            self.sliding_window,
            xs.dtype(),
            self.num_heads,
        )?;

        let mut hidden = xs;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                &hidden,
                attention_mask.as_ref(),
                &seqlen_offsets,
                &mut cache.0[i],
            )?;
        }

        self.norm.forward(&hidden)
    }

    /// Reset the encoder KV cache (call between different audio inputs).
    pub fn reset_cache(&self) {
        let fresh = NormalCache::new_sliding(self.n_layers, 1_000_000, self.sliding_window);
        let inner = fresh.lock().expect("New cache lock poisoned").clone();
        *self.cache.lock().expect("Encoder cache lock poisoned") = inner;
    }
}
