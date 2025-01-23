use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::{
    Activation, Conv1d, Conv1dConfig, Embedding, LayerNorm, Linear, Module, VarBuilder,
};

use crate::{
    attention::SdpaParams,
    layers::{clamp_for_f16, Sdpa},
};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct WhisperEncoderConfig {
    pub num_mel_bins: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub activation_function: Activation,
    pub d_model: usize,
    pub max_source_positions: usize,
}

pub struct WhisperAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl WhisperAttention {
    fn new(cfg: &WhisperEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            q_proj: candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear_no_bias(cfg.d_model, cfg.d_model, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear(cfg.d_model, cfg.d_model, vb.pp("o_proj"))?,
            num_heads: cfg.encoder_attention_heads,
            head_dim: cfg.d_model / cfg.encoder_attention_heads,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut q = self.q_proj.forward(&xs)?;
        let mut k = self.k_proj.forward(&xs)?;
        let mut v = self.v_proj.forward(&xs)?;

        // Should be same, no caching...
        let (bs, q_sq, _) = q.dims3()?;

        q = q
            .reshape((bs, q_sq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((bs, q_sq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((bs, q_sq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let attn_output = Sdpa
            .run_attention(
                &q.contiguous()?,
                &k.contiguous()?,
                &v.contiguous()?,
                attention_mask,
                None,
                &SdpaParams {
                    n_kv_groups: 1,
                    use_flash_attn: false,
                    sliding_window: None,
                    softcap: None,
                    softmax_scale: 1. / (self.head_dim as f32).sqrt(),
                },
            )?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bs, q_sq, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

pub struct WhisperEncoderLayer {
    attn: WhisperAttention,
    self_attn_layer_norm: LayerNorm,
    act: Activation,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl WhisperEncoderLayer {
    fn new(cfg: &WhisperEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn_layer_norm: candle_nn::layer_norm(
                cfg.d_model,
                1e-6,
                vb.pp("self_attn_layer_norm"),
            )?,
            final_layer_norm: candle_nn::layer_norm(cfg.d_model, 1e-6, vb.pp("final_layer_norm"))?,
            fc1: candle_nn::linear(cfg.d_model, cfg.encoder_ffn_dim, vb.pp("fc1"))?,
            fc2: candle_nn::linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?,
            act: cfg.activation_function.clone(),
            attn: WhisperAttention::new(cfg, vb.pp("self_attn"))?,
        })
    }

    fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = self.self_attn_layer_norm.forward(xs)?;
        xs = self.attn.forward(&xs, attention_mask)?;
        xs = (residual + xs)?;

        let residual = xs.clone();
        xs = self.final_layer_norm.forward(&xs)?;
        xs = self.fc1.forward(&xs)?.apply(&self.act)?;
        xs = self.fc2.forward(&xs)?;
        xs = (residual + xs)?;

        if xs.dtype() == DType::F16 {
            xs = clamp_for_f16(&xs)?;
        }

        Ok(xs)
    }
}

pub struct WhisperEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    embed_positions: Embedding,
    layer_norm: LayerNorm,
    layers: Vec<WhisperEncoderLayer>,
}

impl WhisperEncoder {
    pub fn new(cfg: &WhisperEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let conv1 = candle_nn::conv1d(
            cfg.num_mel_bins,
            cfg.d_model,
            3,
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = candle_nn::conv1d(
            cfg.d_model,
            cfg.d_model,
            3,
            Conv1dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        let embed_positions = candle_nn::embedding(
            cfg.max_source_positions,
            cfg.d_model,
            vb.pp("embed_positions"),
        )?;
        let layer_norm = candle_nn::layer_norm(cfg.d_model, 1e-6, vb.pp("layer_norm"))?;

        let vb_l = vb.pp("layers");
        let mut layers = Vec::new();
        for i in 0..cfg.encoder_layers {
            layers.push(WhisperEncoderLayer::new(cfg, vb_l.pp(i))?);
        }

        Ok(Self {
            conv1,
            conv2,
            embed_positions,
            layer_norm,
            layers,
        })
    }

    pub fn forward(&self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = self.conv1.forward(xs)?.gelu()?;
        xs = self.conv2.forward(&xs)?;

        xs = xs.permute((0, 2, 1))?;

        let mut embed_pos = self.embed_positions.embeddings().clone();
        // No cache because there is no streaming.
        embed_pos = embed_pos.i((.., ..xs.dim(1)?, ..))?;

        xs = xs.broadcast_add(&embed_pos)?;

        for layer in &self.layers {
            xs = layer.forward(&xs, attention_mask)?;
        }

        self.layer_norm.forward(&xs)
    }
}
