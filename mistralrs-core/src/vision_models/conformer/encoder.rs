use std::sync::Arc;

use candle_core::{Result, Tensor};
use candle_nn::{BatchNorm, Conv1d, Conv1dConfig, LayerNorm, Linear};
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

        let attention_mask = match (attention_mask, relative_attention_bias) {
            (Some(attention_mask), Some(relative_attention_bias)) => Some(
                attention_mask
                    .unsqueeze(1)?
                    .broadcast_add(relative_attention_bias)?,
            ),
            (Some(attention_mask), None) => Some(attention_mask.unsqueeze(1)?),
            (None, None) => None,
            (None, Some(_)) => {
                candle_core::bail!("Got `relative_attention_bias` but no `attention_mask`")
            }
        };
        let attn_weights = Sdpa.run_attention(
            &q,
            &k,
            &v,
            attention_mask.as_ref(),
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

struct DepthWiseSeperableConv1d {
    dw_conv: Conv1d,
    pw_conv: Option<Conv1d>,
}

impl DepthWiseSeperableConv1d {
    fn new(cfg: &ConformerEncoderConfig, padding: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let dw_conv = layers::conv1d(
            cfg.attention_dim,
            cfg.attention_dim * cfg.depthwise_multiplier,
            cfg.kernel_size,
            Conv1dConfig {
                padding,
                stride: 1,
                groups: cfg.attention_dim,
                dilation: 1,
            },
            vb.pp("dw_conv"),
        )?;

        let pw_conv = if cfg.depthwise_seperable_out_channel != 0 {
            Some(layers::conv1d(
                cfg.attention_dim * cfg.depthwise_multiplier,
                cfg.attention_dim,
                1,
                Conv1dConfig {
                    padding: 0,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp("pw_conv"),
            )?)
        } else {
            None
        };

        Ok(Self { pw_conv, dw_conv })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.dw_conv)?;
        if let Some(pw_conv) = &self.pw_conv {
            xs = xs.apply(pw_conv)?;
        }

        Ok(xs)
    }
}

struct GLUPointWiseConv {
    ext_pw_conv_1d: Conv1d,
    act: Activation,
    b1_b2: Option<(Tensor, Tensor)>,
}

impl GLUPointWiseConv {
    fn new(cfg: &ConformerEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let ext_pw_conv_1d = if cfg.causal {
            layers::conv1d(
                cfg.attention_dim,
                cfg.ext_pw_out_channel * 2,
                cfg.ext_pw_kernel_size,
                Conv1dConfig {
                    padding: cfg.ext_pw_kernel_size - 1,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp("ext_pw_conv_1d"),
            )?
        } else {
            layers::conv1d(
                cfg.attention_dim,
                cfg.ext_pw_out_channel * 2,
                cfg.ext_pw_kernel_size,
                Conv1dConfig {
                    padding: (cfg.ext_pw_kernel_size - 1) / 2,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp("ext_pw_conv_1d"),
            )?
        };

        let b1_b2 = if cfg.bias_in_glu {
            let b1 = vb.get((1, cfg.ext_pw_out_channel, 1), "b1")?;
            let b2 = vb.get((1, cfg.ext_pw_out_channel, 1), "b2")?;
            Some((b1, b2))
        } else {
            None
        };

        Ok(Self {
            ext_pw_conv_1d,
            act: cfg.conv_glu_type,
            b1_b2,
        })
    }
}

struct ConvModule {
    layer_norm: LayerNorm,
    bn_layer: Option<BatchNorm>,
    ln2: Option<Linear>,
    dw_sep_conv_1d: DepthWiseSeperableConv1d,
    glu: GLUPointWiseConv,
}

impl ConvModule {
    fn new(cfg: &ConformerEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let layer_norm = layers::layer_norm(cfg.attention_dim, 1e-5, vb.pp("layer_norm"))?;

        let bn_layer = if cfg.batch_norm {
            Some(layers::batch_norm(
                cfg.attention_dim,
                1e-5,
                vb.pp("bn_layer"),
            )?)
        } else {
            None
        };

        let padding = if cfg.causal {
            cfg.kernel_size - 1
        } else {
            (cfg.kernel_size - 1) / 2
        };

        let dw_sep_conv_1d = DepthWiseSeperableConv1d::new(cfg, padding, vb.pp("dw_sep_conv_1d"))?;

        let ln2 = if cfg.depthwise_seperable_out_channel != 0
            && cfg.attention_dim != cfg.depthwise_seperable_out_channel
        {
            Some(layers::linear(
                cfg.depthwise_seperable_out_channel,
                cfg.attention_dim,
                vb.pp("ln2"),
            )?)
        } else if cfg.depthwise_multiplier != 1 {
            Some(layers::linear(
                cfg.attention_dim * cfg.depthwise_multiplier,
                cfg.attention_dim,
                vb.pp("ln2"),
            )?)
        } else {
            None
        };

        let mut fix_len1 = false;
        let ext_pw_conv_1d = if cfg.causal {
            if cfg.ext_pw_kernel_size > 1 {
                fix_len1 = true;
            } else {
                fix_len1 = false;
            }
            layers::conv1d(
                cfg.attention_dim,
                cfg.ext_pw_out_channel,
                cfg.ext_pw_kernel_size,
                Conv1dConfig {
                    padding: cfg.ext_pw_kernel_size - 1,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp("ext_pw_conv_1d"),
            )?
        } else {
            fix_len1 = false;
            layers::conv1d(
                cfg.attention_dim,
                cfg.ext_pw_out_channel,
                cfg.ext_pw_kernel_size,
                Conv1dConfig {
                    padding: (cfg.ext_pw_kernel_size - 1) / 2,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                },
                vb.pp("ext_pw_conv_1d"),
            )?
        };

        let ln1 = if cfg.attention_dim != cfg.ext_pw_out_channel {
            Some(layers::linear(
                cfg.ext_pw_out_channel,
                cfg.attention_dim,
                vb.pp("ln1"),
            )?)
        } else {
            None
        };

        assert_eq!(cfg.linear_glu_in_convm, false);
        let glu = GLUPointWiseConv::new(cfg, vb.pp("glu"))?;

        Ok(Self {
            layer_norm,
            bn_layer,
            ln2,
            dw_sep_conv_1d,
            glu,
        })
    }
}

struct EncoderLayer {
    self_attn: Attention,
    feed_forward_in: FeedForward,
    feed_forward_out: FeedForward,
    layer_norm_att: LayerNorm,
    layer_norm: LayerNorm,
    conv: ConvModule,
}

impl EncoderLayer {
    fn new(cfg: &ConformerEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;
        let feed_forward_in = FeedForward::new(cfg, vb.pp("feed_forward_in"))?;
        let feed_forward_out = FeedForward::new(cfg, vb.pp("feed_forward_out"))?;
        let layer_norm_att = layers::layer_norm(cfg.attention_dim, 1e-5, vb.pp("layer_norm_att"))?;
        let layer_norm = layers::layer_norm(cfg.attention_dim, 1e-5, vb.pp("layer_norm"))?;
        let conv = ConvModule::new(cfg, vb.pp("conv"))?;

        Ok(Self {
            self_attn,
            feed_forward_in,
            feed_forward_out,
            layer_norm,
            layer_norm_att,
            conv,
        })
    }
}

pub struct Encoder {
    embed: NemoConvSubsampling,
    pos_embed: AbsolutePositionalEncoding,
    relative_attention_bias_layer: T5RelativeAttentionLogitBias,
    encoders: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn new(mut cfg: ConformerEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        assert_eq!(cfg.input_layer, "nemo_conv");

        cfg.finish_nemo_config();
        let embed = NemoConvSubsampling::new(&cfg.nemo_conv_settings, vb.pp("embed"))?;

        let pos_embed = AbsolutePositionalEncoding::new(&cfg, vb.device())?;

        assert!(cfg
            .relative_attention_bias_args
            .as_ref()
            .is_some_and(|x| x.tp == "t5"));
        let relative_attention_bias_args = cfg.relative_attention_bias_args.as_ref().unwrap();
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

        let mut encoders = Vec::new();
        for i in 0..cfg.num_blocks {
            encoders.push(EncoderLayer::new(&cfg, vb.pp("encoders").pp(i))?);
        }

        Ok(Self {
            embed,
            pos_embed,
            relative_attention_bias_layer,
            encoders,
        })
    }
}
