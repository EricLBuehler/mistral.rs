use std::sync::Arc;

use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Linear};
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};

use crate::{
    attention::SdpaParams,
    layers::{self, Activation, Sdpa},
    vision_models::conformer::{
        nemo::NemoConvSubsampling,
        pos_embed::{AbsolutePositionalEncoding, T5RelativeAttentionLogitBias},
    },
};

use super::config::ConformerEncoderConfig;

struct Attention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    o_proj: Arc<dyn QuantMethod>,
}

impl Attention {
    fn new(cfg: &ConformerEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let embed_dim = cfg.attention_dim;
        let num_heads = cfg.attention_heads;
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_proj = mistralrs_quant::linear(embed_dim, embed_dim, &None, vb.pp("q_proj"))?;
        let k_proj = mistralrs_quant::linear(
            embed_dim,
            embed_dim / cfg.attention_group_size,
            &None,
            vb.pp("k_proj"),
        )?;
        let v_proj = mistralrs_quant::linear(
            embed_dim,
            embed_dim / cfg.attention_group_size,
            &None,
            vb.pp("v_proj"),
        )?;
        let o_proj = mistralrs_quant::linear(
            embed_dim / cfg.attention_group_size,
            embed_dim,
            &None,
            vb.pp("out_proj"),
        )?;

        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        relative_attention_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        todo!("relative_attention_bias");
        let (b_sz, q_len, _) = xs.dims3()?;

        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        let mut v = self.v_proj.forward(xs)?;

        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let attn_weights = Sdpa.run_attention(
            &q,
            &k,
            &v,
            attention_mask,
            None,
            &SdpaParams {
                n_kv_groups: 1,
                sliding_window: None,
                softcap: None,
                softmax_scale: self.scale,
            },
        )?;

        self.o_proj.forward(&attn_weights.transpose(1, 2)?.reshape((
            b_sz,
            q_len,
            self.embed_dim,
        ))?)
    }
}

struct FeedForward {
    layer_norm: LayerNorm,
    act: Activation,
    up: Linear,
    down: Linear,
}

impl FeedForward {
    fn new(cfg: &ConformerEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let layer_norm = layers::layer_norm(cfg.attention_dim, 1e-5, vb.pp("layer_norm"))?;
        let up = layers::linear_b(
            cfg.attention_dim,
            cfg.linear_units,
            cfg.bias_in_glu,
            vb.pp("net.0.linear"),
        )?;
        let down = layers::linear(cfg.linear_units, cfg.attention_dim, vb.pp("net.2"))?;

        Ok(Self {
            layer_norm,
            up,
            down,
            act: cfg.activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.layer_norm)?
            .apply(&self.up)?
            .apply(&self.act)?
            .apply(&self.down)
    }
}

struct EncoderLayer {
    self_attn: Attention,
    feed_forward_in: FeedForward,
    feed_forward_out: FeedForward,
    layer_norm_att: LayerNorm,
    layer_norm: LayerNorm,
}

impl EncoderLayer {
    fn new(cfg: &ConformerEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let feed_forward_in = FeedForward::new(cfg, vb.pp("feed_forward_in"))?;
        let feed_forward_out = FeedForward::new(cfg, vb.pp("feed_forward_out"))?;
        let layer_norm_att = layers::layer_norm(cfg.attention_dim, 1e-5, vb.pp("layer_norm_att"))?;
        let layer_norm = layers::layer_norm(cfg.attention_dim, 1e-5, vb.pp("layer_norm"))?;
        todo!()
    }
}

pub struct Encoder {
    embed: NemoConvSubsampling,
    pos_embed: AbsolutePositionalEncoding,
    relative_attention_bias_layer: T5RelativeAttentionLogitBias,
}

impl Encoder {
    pub fn new(mut cfg: ConformerEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        assert_eq!(cfg.input_layer, "nemo_conv");

        cfg.finish_nemo_config();
        let embed = NemoConvSubsampling::new(&cfg.nemo_conv_settings, vb.pp("embed"))?;

        let pos_emb = AbsolutePositionalEncoding::new(&cfg, vb.device())?;

        assert!(cfg
            .relative_attention_bias_args
            .as_ref()
            .is_some_and(|x| x.tp == "t5"));
        let relative_attention_bias_args = cfg.relative_attention_bias_args.unwrap();
        let relative_attention_bias_layer = T5RelativeAttentionLogitBias::new(
            cfg.attention_heads / cfg.attention_group_size,
            None,
            relative_attention_bias_args
                .t5_bias_max_distance
                .unwrap_or(1000),
            relative_attention_bias_args
                .t5_bias_symmetric
                .unwrap_or(false),
            vb.pp("relative_attention_bias_layer"),
        )?;

        todo!()
    }
}
