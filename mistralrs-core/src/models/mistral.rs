#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

/// Mistral LLM, https://github.com/mistralai/mistral-src
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{
    layer_norm::RmsNormNonQuantized, Activation, RotaryEmbedding as RotaryEmbeddingNN, VarBuilder,
};
use candle_transformers::models::with_tracing::{linear_no_bias, Linear};

use crate::{
    pa::{
        layers::{attention::PagedAttention, rope::RotaryEmbedding},
        InputMetadata,
    },
    pipeline::{ConfigLike, MISTRAL_IS_GPTX},
};

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) sliding_window: usize,
    pub(crate) use_flash_attn: bool,
}

impl ConfigLike for Config {
    fn get_hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn get_num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn get_num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn get_num_kv_heads(&self) -> usize {
        self.num_key_value_heads
    }
    fn get_sliding_window(&self) -> Option<usize> {
        Some(self.sliding_window)
    }
    fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    inner: candle_nn::RmsNorm<RmsNormNonQuantized>,
    span: tracing::Span,
}

impl RmsNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm_non_quant(size, eps, vb)?;
        Ok(Self { inner, span })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    hidden_size: usize,
    rotary_emb: RotaryEmbedding,
    attn: PagedAttention,
    nnrope: RotaryEmbeddingNN,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(
        rotary_emb: RotaryEmbedding,
        cfg: &Config,
        vb: VarBuilder,
        nnrope: RotaryEmbeddingNN,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = hidden_sz / num_heads;
        let q_proj = linear_no_bias(hidden_sz, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_sz, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_sz, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            hidden_size: hidden_sz,
            rotary_emb,
            attn: PagedAttention::new(
                vb.device(),
                num_heads,
                head_dim,
                1. / ((head_dim as f32).sqrt()),
                Some(num_kv_heads),
                Some(cfg.sliding_window),
                None,
            )?,
            nnrope,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &mut self,
        input_tokens: Tensor,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = input_tokens.dims3()?;

        let q = self.q_proj.forward(&input_tokens)?;
        let k = self.k_proj.forward(&input_tokens)?;
        let v = self.v_proj.forward(&input_tokens)?;

        //dbg!(&input_positions.to_vec2::<i64>());
        //dbg!(&input_metadata.slot_mapping.to_vec2::<i64>());

        /*let mut q =
            q.reshape((b_sz * q_len, self.num_heads, self.head_dim))?;
        let mut k =
            k.reshape((b_sz * q_len, self.num_kv_heads, self.head_dim))?;

        self.nnrope.forward(
            &[usize::MAX],
            &input_positions,
            &mut q,
            &mut k,
            b_sz,
        )?;

        if q.rank() == 3 {
            q = q
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
            k = k
                .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()?;
        }*/
        self.rotary_emb.forward(input_positions, &q, &k)?;

        let dtype = q.dtype();
        let attn_output = self.attn.forward(
            &q,
            &k,
            &v,
            cache.map(|(k, _)| k),
            cache.map(|(_, v)| v),
            input_metadata,
            dtype,
        )?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: RotaryEmbedding,
        cfg: &Config,
        vb: VarBuilder,
        nnrope: RotaryEmbeddingNN,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"), nnrope)?;
        let mlp = MLP::new(cfg, vb.pp("mlp"))?;
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
        &mut self,
        input_tokens: &Tensor,
        input_positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &mut InputMetadata,
    ) -> Result<Tensor> {
        let residual = input_tokens;
        let xs = self.input_layernorm.forward(&input_tokens)?;
        let xs = self
            .self_attn
            .forward(xs, input_positions, cache, input_metadata)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        residual + xs
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    pub device: Device,
    pub max_seq_len: usize,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary_emb = RotaryEmbedding::new(
            vb.device(),
            vb.dtype(),
            head_dim,
            head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta as f32,
            MISTRAL_IS_GPTX,
        )?;
        let nnrotary_emb = RotaryEmbeddingNN::new(
            cfg.rope_theta as f32,
            head_dim,
            cfg.max_position_embeddings,
            vb.device(),
            MISTRAL_IS_GPTX,
            vb.dtype(),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
                nnrotary_emb.clone(),
            )?;
            layers.push(layer)
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: vb.device().clone(),
            max_seq_len: cfg.max_position_embeddings,
        })
    }

    pub fn forward(
        &mut self,
        input_tokens: &Tensor,
        input_positions: &Tensor,
        kv_caches: Option<&[(Tensor, Tensor)]>,
        mut input_metadata: InputMetadata,
    ) -> Result<Tensor> {
        let (_b_sz, seq_len) = input_tokens.dims2()?;
        let mut xs = self.embed_tokens.forward(&input_tokens)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(
                &xs,
                &input_positions,
                kv_caches.map(|x| (&x[i].0, &x[i].1)),
                &mut input_metadata,
            )?
        }
        xs.apply(&self.norm)?
            .apply(&self.lm_head)?
            .narrow(1, seq_len - 1, 1)
    }
}
