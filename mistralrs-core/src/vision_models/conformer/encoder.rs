#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{BatchNorm, Conv1d, Conv1dConfig, LayerNorm, Linear, ModuleT};
use mistralrs_quant::{Convolution, QuantMethod, ShardedVarBuilder};

use crate::{
    attention::SdpaParams,
    layers::{self, Activation, Sdpa},
    pipeline::text_models_inputs_processor::FlashParams,
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

        let q_proj = mistralrs_quant::linear(embed_dim, embed_dim, &None, vb.pp("linear_q"))?;
        let k_proj = mistralrs_quant::linear(
            embed_dim,
            embed_dim / cfg.attention_group_size,
            &None,
            vb.pp("linear_k"),
        )?;
        let v_proj = mistralrs_quant::linear(
            embed_dim,
            embed_dim / cfg.attention_group_size,
            &None,
            vb.pp("linear_v"),
        )?;
        let o_proj = mistralrs_quant::linear(
            embed_dim / cfg.attention_group_size,
            embed_dim,
            &None,
            vb.pp("linear_out"),
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
            (None, Some(relative_attention_bias)) => Some(relative_attention_bias.contiguous()?),
        };
        let flash_params = FlashParams {
            max_q: 0,
            max_k: 0,
            cumulative_seqlens_q: HashMap::new(),
            cumulative_seqlens_k: HashMap::new(),
            causal: false,
        };

        let attn_weights = Sdpa.run_attention(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            attention_mask.as_ref(),
            Some(&flash_params),
            &SdpaParams {
                n_kv_groups: 1,
                sliding_window: None,
                softcap: None,
                softmax_scale: self.scale,
                sinks: None,
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
            cfg.linear_units * 2,
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
        let normed = xs.apply(&self.layer_norm)?;
        let projected = normed.apply(&self.up)?;

        // GLU: split in half and gate
        let chunks = projected.chunk(2, D::Minus1)?;
        let x = &chunks[0];
        let gate = chunks[1].apply(&self.act)?;
        let gated = (x * gate)?;

        gated.apply(&self.down)
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
                cudnn_fwd_algo: None,
            },
            vb.pp("dw_conv").set_dtype(DType::F32),
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
                    cudnn_fwd_algo: None,
                },
                vb.pp("pw_conv").set_dtype(DType::F32),
            )?)
        } else {
            None
        };

        Ok(Self { pw_conv, dw_conv })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let original_dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        let mut xs = Convolution
            .forward_1d(&self.dw_conv, &xs_f32)?
            .to_dtype(original_dtype)?;
        if let Some(pw_conv) = &self.pw_conv {
            let xs_f32 = xs.to_dtype(DType::F32)?;
            xs = Convolution
                .forward_1d(pw_conv, &xs_f32)?
                .to_dtype(original_dtype)?;
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
                    cudnn_fwd_algo: None,
                },
                vb.pp("ext_pw_conv_1d").set_dtype(DType::F32),
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
                    cudnn_fwd_algo: None,
                },
                vb.pp("ext_pw_conv_1d").set_dtype(DType::F32),
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Input is (B, T, D), need (B, D, T) for conv1d
        let x = x.transpose(1, 2)?;
        let original_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        let x = Convolution
            .forward_1d(&self.ext_pw_conv_1d, &x_f32)?
            .to_dtype(original_dtype)?;

        // Split for GLU
        let chunks = x.chunk(2, 1)?; // Split along channel dim
        let first_half = &chunks[0];
        let second_half = &chunks[1];

        let result = if let Some((b1, b2)) = &self.b1_b2 {
            let first_with_bias = first_half.broadcast_add(b1)?;
            let second_with_bias = second_half.broadcast_add(b2)?;
            first_with_bias.mul(&second_with_bias.apply(&self.act)?)?
        } else {
            first_half.mul(&second_half.apply(&self.act)?)?
        };

        // Back to (B, T, D)
        result.transpose(1, 2)
    }
}

struct ConvModule {
    layer_norm: LayerNorm,
    bn_layer: Option<BatchNorm>,
    ln1: Option<Linear>,
    ln2: Option<Linear>,
    dw_sep_conv_1d: DepthWiseSeperableConv1d,
    glu: GLUPointWiseConv,
    ext_pw_conv_1d: Conv1d,
    cfg: ConformerEncoderConfig,
    act: Activation,
    fix_len1: bool,
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

        assert_ne!(cfg.ext_pw_out_channel, 0);

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

        let fix_len1;
        let ext_pw_conv_1d = if cfg.causal {
            fix_len1 = cfg.ext_pw_kernel_size > 1;
            layers::conv1d(
                cfg.attention_dim,
                cfg.ext_pw_out_channel,
                cfg.ext_pw_kernel_size,
                Conv1dConfig {
                    padding: cfg.ext_pw_kernel_size - 1,
                    stride: 1,
                    dilation: 1,
                    groups: 1,
                    cudnn_fwd_algo: None,
                },
                vb.pp("ext_pw_conv_1d").set_dtype(DType::F32),
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
                    cudnn_fwd_algo: None,
                },
                vb.pp("ext_pw_conv_1d").set_dtype(DType::F32),
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

        assert!(!cfg.linear_glu_in_convm);
        let glu = GLUPointWiseConv::new(cfg, vb.pp("glu"))?;

        Ok(Self {
            layer_norm,
            bn_layer,
            ln1,
            ln2,
            dw_sep_conv_1d,
            glu,
            ext_pw_conv_1d,
            cfg: cfg.clone(),
            act: cfg.conv_activation,
            fix_len1,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.apply(&self.layer_norm)?;

        // Use GLU
        x = self.glu.forward(&x)?;
        if self.cfg.causal && self.cfg.ext_pw_kernel_size > 1 {
            let seq_len = x.dim(1)?;
            x = x.i((.., ..(seq_len - (self.cfg.ext_pw_kernel_size - 1)), ..))?;
        }
        if let Some(ln1) = &self.ln1 {
            x = x.apply(ln1)?;
        }

        // Apply depthwise separable conv
        x = x.transpose(1, 2)?; // (B, T, D) -> (B, D, T)
        x = self.dw_sep_conv_1d.forward(&x)?;

        if self.cfg.causal && self.cfg.kernel_size > 1 {
            let seq_len = x.dim(2)?;
            x = x.i((.., .., ..(seq_len - (self.cfg.kernel_size - 1))))?;
        }

        if let Some(ln2) = &self.ln2 {
            x = x.transpose(1, 2)?; // (B, D, T) -> (B, T, D)
            x = x.apply(ln2)?;
            x = x.transpose(1, 2)?; // (B, T, D) -> (B, D, T)
        }

        if let Some(bn) = &self.bn_layer {
            x = bn.forward_t(&x, false)?;
        }

        x = x.apply(&self.act)?;

        let original_dtype = x.dtype();
        let x_f32 = x.to_dtype(DType::F32)?;
        x = Convolution
            .forward_1d(&self.ext_pw_conv_1d, &x_f32)?
            .to_dtype(original_dtype)?;
        if self.fix_len1 {
            let seq_len = x.dim(2)?;
            x = x.i((.., .., ..(seq_len - (self.cfg.ext_pw_kernel_size - 1))))?;
        }
        if let Some(ln1) = &self.ln1 {
            x = x.transpose(1, 2)?;
            x = x.apply(ln1)?;
            x = x.transpose(1, 2)?;
        }
        x = x.transpose(1, 2)?; // Back to (B, T, D)

        Ok(x)
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
    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        relative_attention_bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        // First feed forward (0.5x)
        let ff_in_out = self.feed_forward_in.forward(x)?;
        let mut x = (x + (ff_in_out * 0.5)?)?;

        // Self attention with pre-norm
        let norm_x = x.apply(&self.layer_norm_att)?;
        let attn_out = self
            .self_attn
            .forward(&norm_x, mask, relative_attention_bias)?;
        x = (x + attn_out)?;

        // Conv module
        let conv_out = self.conv.forward(&x)?;
        x = (x + conv_out)?;

        // Second feed forward (0.5x)
        let ff_out_out = self.feed_forward_out.forward(&x)?;
        x = (x + (ff_out_out * 0.5)?)?;

        // Final layer norm
        x.apply(&self.layer_norm)
    }
}

struct EncoderEmbedding {
    global_invstd: Tensor,
    global_mean: Tensor,
}

impl EncoderEmbedding {
    fn new(vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            global_invstd: vb.get_unchecked("global_invstd")?,
            global_mean: vb.get_unchecked("global_mean")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_sub(&self.global_mean)?
            .broadcast_mul(&self.global_invstd)
    }
}

pub struct ConformerEncoder {
    encoder_embedding: EncoderEmbedding,
    embed: NemoConvSubsampling,
    #[allow(unused)]
    pos_embed: AbsolutePositionalEncoding,
    relative_attention_bias_layer: T5RelativeAttentionLogitBias,
    encoders: Vec<EncoderLayer>,
}

impl ConformerEncoder {
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

        let encoder_embedding = EncoderEmbedding::new(vb.pp("encoder_embedding"))?;

        Ok(Self {
            encoder_embedding,
            embed,
            pos_embed,
            relative_attention_bias_layer,
            encoders,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)> {
        // Forward through embeddings (subsampling)
        let xs = self.encoder_embedding.forward(xs)?;
        let (mut input_tensor, masks) = self.embed.forward(&xs, mask)?;

        // Handle long sequences with unfolding
        let max_seq_len = 500;
        let (ori_bz, seq_len, d) = input_tensor.dims3()?;
        let unfolded = seq_len > max_seq_len;

        // Outside of the `if` block as it's needed below
        let mut chunk_pad_size = 0;
        if unfolded {
            // Pad to multiple of max_seq_len
            chunk_pad_size = if seq_len % max_seq_len > 0 {
                max_seq_len - (seq_len % max_seq_len)
            } else {
                0
            };

            if chunk_pad_size > 0 {
                input_tensor = input_tensor.pad_with_zeros(D::Minus2, 0, chunk_pad_size)?;
            }

            // Unfold into chunks
            input_tensor = unfold_tensor(&input_tensor, max_seq_len)?;
        }

        // // Apply positional encoding
        // input_tensor = self.pos_embed.forward(&input_tensor)?;

        // Compute relative attention bias if available
        let relative_attention_bias = self.relative_attention_bias_layer.forward(&input_tensor)?;

        // Apply encoder layers
        for layer in &self.encoders {
            input_tensor = layer.forward(
                &input_tensor,
                masks.as_ref(),
                Some(&relative_attention_bias),
            )?;
        }

        // Handle unfolding restoration
        if unfolded {
            input_tensor = input_tensor.reshape((ori_bz, seq_len + chunk_pad_size, d))?;
            if chunk_pad_size > 0 {
                input_tensor = input_tensor.i((.., ..seq_len, ..))?;
            }
        }

        Ok((input_tensor, masks))
    }
}

fn unfold_tensor(xs_pad: &Tensor, max_seq_len: usize) -> Result<Tensor> {
    let (_n, t, _d) = xs_pad.dims3()?;

    // If sequence length is already <= max_seq_len, no need to unfold
    if t <= max_seq_len {
        return Ok(xs_pad.clone());
    }

    // xs_pad.transpose(-1, -2) # convert to N, D, T
    let xs_pad = xs_pad.transpose(1, 2)?; // (N, T, D) -> (N, D, T)

    // Unfold the last dimension (T) with size=max_seq_len and step=max_seq_len
    // This creates sliding windows of size max_seq_len with step max_seq_len
    let xs_pad = xs_pad.unfold(2, max_seq_len, max_seq_len)?;
    // Shape is now (N, D, T', max_seq_len) where T' = T // max_seq_len

    let (n, d, t_prime, _) = xs_pad.dims4()?;

    // Permute to (N, T', max_seq_len, D) - equivalent to permute(0, 2, 3, 1)
    let xs_pad = xs_pad.permute((0, 2, 3, 1))?;

    // Reshape to (N*T', max_seq_len, D)
    let xs_pad = xs_pad.reshape((n * t_prime, max_seq_len, d))?;

    Ok(xs_pad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_unfold_tensor() -> Result<()> {
        let device = Device::Cpu;

        // Test case 1: T > max_seq_len
        let xs = Tensor::arange(0f32, 24f32, &device)?.reshape((2, 6, 2))?; // (N=2, T=6, D=2)
        let result = unfold_tensor(&xs, 3)?; // max_seq_len=3
        assert_eq!(result.dims(), &[4, 3, 2]); // (N*T'=2*2, max_seq_len=3, D=2)

        // Test case 2: T <= max_seq_len
        let xs = Tensor::arange(0f32, 12f32, &device)?.reshape((2, 3, 2))?; // (N=2, T=3, D=2)
        let result = unfold_tensor(&xs, 5)?; // max_seq_len=5
        assert_eq!(result.dims(), &[2, 3, 2]); // Should return original shape

        // Test case 3: T == max_seq_len
        let xs = Tensor::arange(0f32, 12f32, &device)?.reshape((2, 3, 2))?; // (N=2, T=3, D=2)
        let result = unfold_tensor(&xs, 3)?; // max_seq_len=3
        assert_eq!(result.dims(), &[2, 3, 2]); // (N*T'=2*1, max_seq_len=3, D=2)

        Ok(())
    }

    #[test]
    fn test_unfold_tensor_larger() -> Result<()> {
        let device = Device::Cpu;

        // Test with larger tensor
        let xs = Tensor::arange(0f32, 120f32, &device)?.reshape((2, 10, 6))?; // (N=2, T=10, D=6)
        let result = unfold_tensor(&xs, 4)?; // max_seq_len=4, T'=10//4=2
        assert_eq!(result.dims(), &[4, 4, 6]); // (N*T'=2*2, max_seq_len=4, D=6)

        Ok(())
    }
}
