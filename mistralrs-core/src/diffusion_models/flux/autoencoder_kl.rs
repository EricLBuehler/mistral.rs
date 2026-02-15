//! AutoencoderKL implementation for FLUX.2 (diffusers-style naming)
//!
//! This implements the AutoencoderKLFlux2 architecture used by FLUX.2 models,
//! which uses diffusers-style weight naming conventions.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, GroupNorm, Linear};
use mistralrs_quant::{Convolution, ShardedVarBuilder};
use serde::Deserialize;

use crate::layers::{conv2d, group_norm, linear_b};

pub use super::common::DiagonalGaussian;

fn default_scaling_factor() -> f64 {
    1.0
}

fn default_shift_factor() -> f64 {
    0.0
}

fn default_batch_norm_eps() -> f64 {
    1e-5
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    #[serde(default = "default_scaling_factor")]
    pub scaling_factor: f64,
    #[serde(default = "default_shift_factor")]
    pub shift_factor: f64,
    #[serde(default = "default_batch_norm_eps")]
    pub batch_norm_eps: f64,
    pub norm_num_groups: usize,
}

impl From<&super::autoencoder::Config> for Config {
    fn from(cfg: &super::autoencoder::Config) -> Self {
        Self {
            in_channels: cfg.in_channels,
            out_channels: cfg.out_channels,
            block_out_channels: cfg.block_out_channels.clone(),
            layers_per_block: cfg.layers_per_block,
            latent_channels: cfg.latent_channels,
            scaling_factor: cfg.scaling_factor,
            shift_factor: cfg.shift_factor,
            batch_norm_eps: cfg.batch_norm_eps,
            norm_num_groups: cfg.norm_num_groups,
        }
    }
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    super::common::scaled_dot_product_attention_simple(q, k, v)
}

/// Attention block with diffusers-style naming (to_q, to_k, to_v, to_out, group_norm)
/// FLUX.2 VAE uses Linear layers for attention projections, not Conv2d
#[derive(Debug, Clone)]
struct AttnBlock {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    group_norm: GroupNorm,
}

impl AttnBlock {
    fn new(in_c: usize, vb: ShardedVarBuilder, cfg: &Config) -> Result<Self> {
        let to_q = linear_b(in_c, in_c, true, vb.pp("to_q"))?;
        let to_k = linear_b(in_c, in_c, true, vb.pp("to_k"))?;
        let to_v = linear_b(in_c, in_c, true, vb.pp("to_v"))?;
        let to_out = linear_b(in_c, in_c, true, vb.pp("to_out.0"))?;
        let group_norm = group_norm(cfg.norm_num_groups, in_c, 1e-6, vb.pp("group_norm"))?;
        Ok(Self {
            to_q,
            to_k,
            to_v,
            to_out,
            group_norm,
        })
    }
}

impl candle_core::Module for AttnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let init_xs = xs;
        let (b, c, h, w) = xs.dims4()?;

        // Group norm expects (b, c, h, w)
        let normed = self.group_norm.forward(xs)?;

        // Reshape from (b, c, h, w) to (b, h*w, c) for linear projections
        let normed = normed.permute((0, 2, 3, 1))?.reshape((b, h * w, c))?;

        // Apply linear projections
        let q = self.to_q.forward(&normed)?;
        let k = self.to_k.forward(&normed)?;
        let v = self.to_v.forward(&normed)?;

        // Add head dimension: (b, seq, c) -> (b, 1, seq, c)
        let q = q.unsqueeze(1)?;
        let k = k.unsqueeze(1)?;
        let v = v.unsqueeze(1)?;

        // Attention
        let attended = scaled_dot_product_attention(&q, &k, &v)?;

        // Remove head dimension: (b, 1, seq, c) -> (b, seq, c)
        let attended = attended.squeeze(1)?;

        // Apply output projection
        let projected = self.to_out.forward(&attended)?;

        // Reshape back to (b, c, h, w)
        let projected = projected.reshape((b, h, w, c))?.permute((0, 3, 1, 2))?;

        projected + init_xs
    }
}

/// ResNet block with diffusers-style naming (norm1, conv1, norm2, conv2, conv_shortcut)
#[derive(Debug, Clone)]
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn new(in_c: usize, out_c: usize, vb: ShardedVarBuilder, cfg: &Config) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let norm1 = group_norm(cfg.norm_num_groups, in_c, 1e-6, vb.pp("norm1"))?;
        let conv1 = conv2d(in_c, out_c, 3, conv_cfg, vb.pp("conv1"))?;
        let norm2 = group_norm(cfg.norm_num_groups, out_c, 1e-6, vb.pp("norm2"))?;
        let conv2 = conv2d(out_c, out_c, 3, conv_cfg, vb.pp("conv2"))?;
        let conv_shortcut = if in_c == out_c {
            None
        } else {
            Some(conv2d(
                in_c,
                out_c,
                1,
                Default::default(),
                vb.pp("conv_shortcut"),
            )?)
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
        })
    }
}

impl candle_core::Module for ResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = self.norm1.forward(xs)?;
        h = candle_nn::Activation::Silu.forward(&h)?;
        h = Convolution.forward_2d(&self.conv1, &h)?;
        h = self.norm2.forward(&h)?;
        h = candle_nn::Activation::Silu.forward(&h)?;
        h = Convolution.forward_2d(&self.conv2, &h)?;
        match self.conv_shortcut.as_ref() {
            None => xs + h,
            Some(c) => Convolution.forward_2d(c, xs)? + h,
        }
    }
}

/// Downsampler with diffusers naming (downsamplers.0.conv)
#[derive(Debug, Clone)]
struct Downsample {
    conv: Conv2d,
}

impl Downsample {
    fn new(in_c: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(in_c, in_c, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl candle_core::Module for Downsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Convolution.forward_2d(&self.conv, xs)
    }
}

/// Upsampler with diffusers naming (upsamplers.0.conv)
#[derive(Debug, Clone)]
struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    fn new(in_c: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let conv = conv2d(in_c, in_c, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl candle_core::Module for Upsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let upsampled = xs.upsample_nearest2d(h * 2, w * 2)?;
        Convolution.forward_2d(&self.conv, &upsampled)
    }
}

/// Down block containing resnets and optional downsampler
#[derive(Debug, Clone)]
struct DownBlock {
    resnets: Vec<ResnetBlock>,
    downsamplers: Option<Downsample>,
}

/// Mid block containing resnets and attention
#[derive(Debug, Clone)]
struct MidBlock {
    resnets: Vec<ResnetBlock>,
    attentions: Vec<AttnBlock>,
}

impl MidBlock {
    fn new(in_c: usize, vb: ShardedVarBuilder, cfg: &Config) -> Result<Self> {
        let vb_r = vb.pp("resnets");
        let resnets = vec![
            ResnetBlock::new(in_c, in_c, vb_r.pp(0), cfg)?,
            ResnetBlock::new(in_c, in_c, vb_r.pp(1), cfg)?,
        ];
        let vb_a = vb.pp("attentions");
        let attentions = vec![AttnBlock::new(in_c, vb_a.pp(0), cfg)?];
        Ok(Self {
            resnets,
            attentions,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = self.resnets[0].forward(xs)?;
        for attn in &self.attentions {
            h = attn.forward(&h)?;
        }
        for resnet in self.resnets.iter().skip(1) {
            h = resnet.forward(&h)?;
        }
        Ok(h)
    }
}

/// Up block containing resnets and optional upsampler
#[derive(Debug, Clone)]
struct UpBlock {
    resnets: Vec<ResnetBlock>,
    upsamplers: Option<Upsample>,
}

/// Encoder with diffusers-style naming
#[derive(Debug, Clone)]
pub struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownBlock>,
    mid_block: MidBlock,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Encoder {
    pub fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let base_ch = cfg.block_out_channels[0];
        let conv_in = conv2d(cfg.in_channels, base_ch, 3, conv_cfg, vb.pp("conv_in"))?;

        let mut down_blocks = Vec::with_capacity(cfg.block_out_channels.len());
        let vb_d = vb.pp("down_blocks");
        let mut block_in = base_ch;

        for (i_level, &block_out) in cfg.block_out_channels.iter().enumerate() {
            let vb_d = vb_d.pp(i_level);
            let vb_r = vb_d.pp("resnets");

            let mut resnets = Vec::with_capacity(cfg.layers_per_block);
            for i_block in 0..cfg.layers_per_block {
                let in_c = if i_block == 0 { block_in } else { block_out };
                resnets.push(ResnetBlock::new(in_c, block_out, vb_r.pp(i_block), cfg)?);
            }
            block_in = block_out;

            let downsamplers = if i_level != cfg.block_out_channels.len() - 1 {
                Some(Downsample::new(block_out, vb_d.pp("downsamplers.0"))?)
            } else {
                None
            };

            down_blocks.push(DownBlock {
                resnets,
                downsamplers,
            });
        }

        let mid_block = MidBlock::new(block_in, vb.pp("mid_block"), cfg)?;
        let conv_norm_out =
            group_norm(cfg.norm_num_groups, block_in, 1e-6, vb.pp("conv_norm_out"))?;
        let conv_out = conv2d(
            block_in,
            2 * cfg.latent_channels,
            3,
            conv_cfg,
            vb.pp("conv_out"),
        )?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_out,
        })
    }
}

impl candle_nn::Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = Convolution.forward_2d(&self.conv_in, xs)?;
        for block in &self.down_blocks {
            for resnet in &block.resnets {
                h = resnet.forward(&h)?;
            }
            if let Some(ds) = &block.downsamplers {
                h = ds.forward(&h)?;
            }
        }
        h = self.mid_block.forward(&h)?;
        h = self.conv_norm_out.forward(&h)?;
        h = candle_nn::Activation::Silu.forward(&h)?;
        Convolution.forward_2d(&self.conv_out, &h)
    }
}

/// Decoder with diffusers-style naming
#[derive(Debug, Clone)]
pub struct Decoder {
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    pub fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let block_in = cfg
            .block_out_channels
            .last()
            .copied()
            .ok_or_else(|| candle_core::Error::Msg("block_out_channels must not be empty".into()))?;
        let conv_in = conv2d(cfg.latent_channels, block_in, 3, conv_cfg, vb.pp("conv_in"))?;

        let mid_block = MidBlock::new(block_in, vb.pp("mid_block"), cfg)?;

        // Decoder has layers_per_block + 1 resnets per up_block
        let num_resnets = cfg.layers_per_block + 1;
        let mut up_blocks = Vec::with_capacity(cfg.block_out_channels.len());
        let vb_u = vb.pp("up_blocks");
        let mut prev_out = block_in;

        // up_blocks go in order 0..n (reversed channel order from encoder)
        let reversed_channels: Vec<usize> = cfg.block_out_channels.iter().copied().rev().collect();

        for (i_level, &block_out) in reversed_channels.iter().enumerate() {
            let vb_u = vb_u.pp(i_level);
            let vb_r = vb_u.pp("resnets");

            let mut resnets = Vec::with_capacity(num_resnets);
            for i_block in 0..num_resnets {
                let in_c = if i_block == 0 { prev_out } else { block_out };
                resnets.push(ResnetBlock::new(in_c, block_out, vb_r.pp(i_block), cfg)?);
            }
            prev_out = block_out;

            let upsamplers = if i_level != reversed_channels.len() - 1 {
                Some(Upsample::new(block_out, vb_u.pp("upsamplers.0"))?)
            } else {
                None
            };

            up_blocks.push(UpBlock {
                resnets,
                upsamplers,
            });
        }

        let conv_norm_out =
            group_norm(cfg.norm_num_groups, prev_out, 1e-6, vb.pp("conv_norm_out"))?;
        let conv_out = conv2d(prev_out, cfg.out_channels, 3, conv_cfg, vb.pp("conv_out"))?;

        Ok(Self {
            conv_in,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_out,
        })
    }
}

impl candle_nn::Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = Convolution.forward_2d(&self.conv_in, xs)?;
        h = self.mid_block.forward(&h)?;
        for block in &self.up_blocks {
            for resnet in &block.resnets {
                h = resnet.forward(&h)?;
            }
            if let Some(us) = &block.upsamplers {
                h = us.forward(&h)?;
            }
        }
        h = self.conv_norm_out.forward(&h)?;
        h = candle_nn::Activation::Silu.forward(&h)?;
        Convolution.forward_2d(&self.conv_out, &h)
    }
}

/// AutoencoderKL for FLUX.2 with diffusers-style naming
///
/// Includes quant_conv, post_quant_conv, and BatchNorm (affine=False) layers.
#[derive(Debug, Clone)]
pub struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    reg: DiagonalGaussian,
    quant_conv: Conv2d,
    post_quant_conv: Conv2d,
    bn_mean_4d: Tensor,
    bn_std_4d: Tensor,
    shift_factor: f64,
    scale_factor: f64,
}

impl AutoEncoderKL {
    pub fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let decoder = Decoder::new(cfg, vb.pp("decoder"))?;
        let reg = DiagonalGaussian::new(true, 1)?;

        // quant_conv: 2*latent_channels -> 2*latent_channels (for mean and logvar)
        let quant_conv = conv2d(
            2 * cfg.latent_channels,
            2 * cfg.latent_channels,
            1,
            Default::default(),
            vb.pp("quant_conv"),
        )?;

        // post_quant_conv: latent_channels -> latent_channels
        let post_quant_conv = conv2d(
            cfg.latent_channels,
            cfg.latent_channels,
            1,
            Default::default(),
            vb.pp("post_quant_conv"),
        )?;

        // BatchNorm with affine=False (no weight/bias, just running stats)
        // FLUX.2 VAE uses this to normalize latents
        // Features = patch_size^2 * latent_channels = 2*2 * 32 = 128 for FLUX.2
        let bn_features = 4 * cfg.latent_channels; // patch_size=2, so 2*2=4
        let bn_vb = vb.pp("bn");
        let bn_mean = bn_vb.get(bn_features, "running_mean")?;
        let bn_var = bn_vb.get(bn_features, "running_var")?;

        // Precompute reshaped mean and std to avoid per-call reshape+sqrt
        let bn_mean_4d = bn_mean.flatten_all()?.reshape((1, (), 1, 1))?;
        let bn_std_4d =
            (bn_var.flatten_all()?.reshape((1, (), 1, 1))? + cfg.batch_norm_eps)?.sqrt()?;

        Ok(Self {
            encoder,
            decoder,
            reg,
            quant_conv,
            post_quant_conv,
            bn_mean_4d,
            bn_std_4d,
            scale_factor: cfg.scaling_factor,
            shift_factor: cfg.shift_factor,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs.apply(&self.encoder)?;
        let h = Convolution.forward_2d(&self.quant_conv, &h)?;
        let z = h.apply(&self.reg)?;
        (z - self.shift_factor)? * self.scale_factor
    }

    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = ((xs / self.scale_factor)? + self.shift_factor)?;
        let xs = Convolution.forward_2d(&self.post_quant_conv, &xs)?;
        xs.apply(&self.decoder)
    }

    pub fn denormalize_packed(&self, xs: &Tensor) -> Result<Tensor> {
        let mean = self.bn_mean_4d.to_dtype(xs.dtype())?;
        let std = self.bn_std_4d.to_dtype(xs.dtype())?;
        xs.broadcast_mul(&std)?.broadcast_add(&mean)
    }

    pub fn normalize_packed(&self, xs: &Tensor) -> Result<Tensor> {
        let mean = self.bn_mean_4d.to_dtype(xs.dtype())?;
        let std = self.bn_std_4d.to_dtype(xs.dtype())?;
        xs.broadcast_sub(&mean)?.broadcast_div(&std)
    }
}

impl candle_core::Module for AutoEncoderKL {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.decode(&self.encode(xs)?)
    }
}
