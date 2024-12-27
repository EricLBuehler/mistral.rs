use std::{collections::HashMap, sync::Arc};

use candle_core::{Result, Tensor, D};
use candle_nn::{embedding, Embedding, Linear, Module, VarBuilder};
use serde::Deserialize;

use crate::{
    attention::SdpaParams,
    layers::{
        Activation, DeepSeekV2RopeConfig, DeepSeekV2RopeScaling, DeepSeekV2RotaryEmbedding, RmsNorm,
    },
    ops::SplitOp,
    paged_attention::AttentionImplementation,
    pipeline::{KvCache, NormalLoadingMetadata},
    serde_default_fn,
    utils::progress::NiceProgressBar,
};

serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(TopkMethod, topk_method, TopkMethod::Greedy);
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(bool, norm_topk_prob, false);
serde_default_fn!(ScoringFunc, scoring_func, ScoringFunc::Softmax);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(bool, tie_word_embeddings, false);

#[derive(Deserialize, Clone)]
enum TopkMethod {
    #[serde(rename = "greedy")]
    Greedy,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

#[derive(Deserialize, Clone)]
enum ScoringFunc {
    #[serde(rename = "softmax")]
    Softmax,
}

#[derive(Deserialize, Clone)]
pub struct DeepSeekV2Config {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    moe_intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    n_shared_experts: Option<usize>,
    n_routed_experts: Option<usize>,
    #[serde(default = "routed_scaling_factor")]
    routed_scaling_factor: f64,
    #[serde(default = "topk_method")]
    topk_method: TopkMethod,
    num_experts_per_tok: Option<usize>,
    #[serde(default = "moe_layer_freq")]
    moe_layer_freq: usize,
    #[serde(default = "first_k_dense_replace")]
    first_k_dense_replace: usize,
    // k dense layers
    #[serde(default = "norm_topk_prob")]
    norm_topk_prob: bool,
    #[serde(default = "scoring_func")]
    scoring_func: ScoringFunc,
    #[serde(default = "hidden_act")]
    hidden_act: Activation,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    #[serde(default = "tie_word_embeddings")]
    tie_word_embeddings: bool,
    rope_theta: f32,
    rope_scaling: Option<DeepSeekV2RopeScaling>,
    attention_bias: bool,
    q_lora_rank: Option<usize>,
    qk_rope_head_dim: usize,
    kv_lora_rank: usize,
    v_head_dim: usize,
    qk_nope_head_dim: usize,
}

enum QProj {
    Plain(Linear),
    Lora { a: Linear, norm: RmsNorm, b: Linear },
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Lora { a, norm, b } => b.forward(&norm.forward(&a.forward(xs)?)?),
            Self::Plain(lin) => lin.forward(xs),
        }
    }
}

struct Attention {
    q: QProj,
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    o_proj: Linear,
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    softmax_scale: f32,
    cfg: DeepSeekV2Config,
    q_head_dim: usize,
}

impl Attention {
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        let q = match cfg.q_lora_rank {
            Some(lora_rank) => {
                let a = candle_nn::linear_b(
                    cfg.hidden_size,
                    lora_rank,
                    cfg.attention_bias,
                    vb.pp("q_a_proj"),
                )?;
                let norm = RmsNorm::new(lora_rank, cfg.rms_norm_eps, vb.pp("q_a_layernorm"))?;
                let b = candle_nn::linear_no_bias(
                    lora_rank,
                    cfg.num_attention_heads * q_head_dim,
                    vb.pp("q_b_proj"),
                )?;
                QProj::Lora { a, norm, b }
            }
            None => QProj::Plain(candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * q_head_dim,
                vb.pp("q_proj"),
            )?),
        };

        let kv_a_proj_with_mqa = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            cfg.attention_bias,
            vb.pp("kv_a_proj_with_mqa"),
        )?;
        let kv_a_layernorm =
            RmsNorm::new(cfg.kv_lora_rank, cfg.rms_norm_eps, vb.pp("kv_a_layernorm"))?;
        let kv_b_proj = candle_nn::linear_no_bias(
            cfg.kv_lora_rank,
            cfg.num_attention_heads * (q_head_dim - cfg.qk_rope_head_dim + cfg.v_head_dim),
            vb.pp("kv_b_proj"),
        )?;

        let o_proj = candle_nn::linear_b(
            cfg.num_attention_heads * cfg.v_head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        let mut softmax_scale = 1.0 / (q_head_dim as f32).sqrt();
        if let Some(DeepSeekV2RopeScaling::Yarn {
            mscale_all_dim,
            factor,
            ..
        }) = cfg.rope_scaling
        {
            let mscale = DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, mscale_all_dim);
            softmax_scale = softmax_scale * mscale * mscale;
        }

        Ok(Self {
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            softmax_scale,
            cfg: cfg.clone(),
            q_head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;

        let mut q = self.q.forward(xs)?;
        q = q
            .reshape((bs, seq_len, self.cfg.num_attention_heads, self.q_head_dim))?
            .transpose(1, 2)?;
        let q_split = q.split(
            &[self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        let q_nope = q_split[0].clone();
        let q_pe = q_split[1].clone();

        let mut compressed_kv = self.kv_a_proj_with_mqa.forward(xs)?;
        let ckv_split = compressed_kv.split(
            &[self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        compressed_kv = ckv_split[0].clone();
        let mut k_pe = ckv_split[1].clone();
        k_pe = k_pe
            .reshape((bs, seq_len, 1, self.cfg.qk_rope_head_dim))?
            .transpose(1, 2)?;
        let mut kv = self
            .kv_b_proj
            .forward(&self.kv_a_layernorm.forward(&compressed_kv)?)?;
        kv = kv
            .reshape((
                bs,
                seq_len,
                self.cfg.num_attention_heads,
                self.cfg.qk_nope_head_dim + self.cfg.v_head_dim,
            ))?
            .transpose(1, 2)?;

        let kv_split = kv.split(&[self.cfg.qk_nope_head_dim, self.cfg.v_head_dim], D::Minus1)?;
        let k_nope = kv_split[0].clone();
        let mut v = kv_split[1].clone();

        let (q_pe, k_pe) = self.rotary_emb.forward(&q_pe, &k_pe, seqlen_offsets)?;

        let mut q = Tensor::zeros(
            (bs, self.cfg.num_attention_heads, seq_len, self.q_head_dim),
            q_pe.dtype(),
            q_pe.device(),
        )?;
        q = q.slice_assign(&[&.., &.., &.., &(..self.cfg.qk_nope_head_dim)], &q_nope)?;
        q = q.slice_assign(&[&.., &.., &.., &(self.cfg.qk_nope_head_dim..)], &q_pe)?;

        let mut k = Tensor::zeros(
            (bs, self.cfg.num_attention_heads, seq_len, self.q_head_dim),
            k_pe.dtype(),
            k_pe.device(),
        )?;
        k = k.slice_assign(&[&.., &.., &.., &(..self.cfg.qk_nope_head_dim)], &k_nope)?;
        k = k.slice_assign(&[&.., &.., &.., &(self.cfg.qk_nope_head_dim..)], &k_pe)?;

        (k, v) = kv_cache.append(&k, &v)?;

        let mut attn_out = {
            let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * self.softmax_scale as f64)?;
            attn_weights = match attention_mask {
                Some(mask) => (attn_weights + mask)?,
                None => attn_weights,
            };
            attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&v)?
        };
        attn_out = if attention_mask.is_some() {
            attn_out.transpose(1, 2)?.reshape((bs, seq_len, ()))?
        } else {
            attn_out.reshape((bs, seq_len, ()))?
        };

        self.o_proj.forward(&attn_out)
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn: Attention,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        todo!()
    }
}

pub struct DeepSeekV2 {
    lm_head: Linear,
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
}

impl DeepSeekV2 {
    pub fn new(
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,
        _is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let lm_head = if !cfg.tie_word_embeddings {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        };
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let mut ropes = HashMap::new();
        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        for _i in 0..cfg.num_hidden_layers {
            let device = &normal_loading_metadata.real_device;
            ropes.insert(
                device.location(),
                Arc::new(DeepSeekV2RotaryEmbedding::new(
                    &rope_cfg,
                    vb.dtype(),
                    device,
                )?),
            );
        }
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in
            NiceProgressBar::<_, 'b'>(0..cfg.num_hidden_layers, "Loading repeating layers")
        {
            let device = &normal_loading_metadata.real_device;
            let rotary_emb = ropes
                .get(&device.location())
                .expect("No RoPE for device location!")
                .clone();
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }

        Ok(Self {
            lm_head,
            embed_tokens,
            norm,
            layers,
        })
    }
}
