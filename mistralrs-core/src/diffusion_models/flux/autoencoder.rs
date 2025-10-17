#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Result, Tensor, D};
use candle_nn::{Conv2d, GroupNorm};
use mistralrs_quant::{Convolution, ShardedVarBuilder};
use serde::Deserialize;

use crate::layers::{conv2d, group_norm, MatMul};

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub in_channels: usize,
    pub out_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub latent_channels: usize,
    pub scaling_factor: f64,
    pub shift_factor: f64,
    pub norm_num_groups: usize,
}

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (MatMul.matmul(q, &k.t()?)? * scale_factor)?;
    MatMul.matmul(&candle_nn::ops::softmax_last_dim(&attn_weights)?, v)
}

#[derive(Debug, Clone)]
struct AttnBlock {
    q: Conv2d,
    k: Conv2d,
    v: Conv2d,
    proj_out: Conv2d,
    norm: GroupNorm,
}

impl AttnBlock {
    fn new(in_c: usize, vb: ShardedVarBuilder, cfg: &Config) -> Result<Self> {
        let q = conv2d(in_c, in_c, 1, Default::default(), vb.pp("q"))?;
        let k = conv2d(in_c, in_c, 1, Default::default(), vb.pp("k"))?;
        let v = conv2d(in_c, in_c, 1, Default::default(), vb.pp("v"))?;
        let proj_out = conv2d(in_c, in_c, 1, Default::default(), vb.pp("proj_out"))?;
        let norm = group_norm(cfg.norm_num_groups, in_c, 1e-6, vb.pp("norm"))?;
        Ok(Self {
            q,
            k,
            v,
            proj_out,
            norm,
        })
    }
}

impl candle_core::Module for AttnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let init_xs = xs;
        let normed = self.norm.forward(xs)?;
        let q = Convolution.forward_2d(&self.q, &normed)?;
        let k = Convolution.forward_2d(&self.k, &normed)?;
        let v = Convolution.forward_2d(&self.v, &normed)?;
        let (b, c, h, w) = q.dims4()?;
        let q = q.flatten_from(2)?.t()?.unsqueeze(1)?;
        let k = k.flatten_from(2)?.t()?.unsqueeze(1)?;
        let v = v.flatten_from(2)?.t()?.unsqueeze(1)?;
        let attended = scaled_dot_product_attention(&q, &k, &v)?;
        let attended = attended.squeeze(1)?.t()?.reshape((b, c, h, w))?;
        let projected = Convolution.forward_2d(&self.proj_out, &attended)?;
        projected + init_xs
    }
}

#[derive(Debug, Clone)]
struct ResnetBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    nin_shortcut: Option<Conv2d>,
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
        let nin_shortcut = if in_c == out_c {
            None
        } else {
            Some(conv2d(
                in_c,
                out_c,
                1,
                Default::default(),
                vb.pp("nin_shortcut"),
            )?)
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            nin_shortcut,
        })
    }
}

impl candle_core::Module for ResnetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = self.norm1.forward(xs)?;
        h = candle_nn::Activation::Swish.forward(&h)?;
        h = Convolution.forward_2d(&self.conv1, &h)?;
        h = self.norm2.forward(&h)?;
        h = candle_nn::Activation::Swish.forward(&h)?;
        h = Convolution.forward_2d(&self.conv2, &h)?;
        match self.nin_shortcut.as_ref() {
            None => xs + h,
            Some(c) => Convolution.forward_2d(c, xs)? + h,
        }
    }
}

#[derive(Debug, Clone)]
struct Downsample {
    conv: Conv2d,
}

impl Downsample {
    fn new(in_c: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            stride: 2,
            ..Default::default()
        };
        let conv = conv2d(in_c, in_c, 3, conv_cfg, vb.pp("conv"))?;
        Ok(Self { conv })
    }
}

impl candle_core::Module for Downsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.pad_with_zeros(D::Minus1, 0, 1)?;
        let xs = xs.pad_with_zeros(D::Minus2, 0, 1)?;
        Convolution.forward_2d(&self.conv, &xs)
    }
}

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

#[derive(Debug, Clone)]
struct DownBlock {
    block: Vec<ResnetBlock>,
    downsample: Option<Downsample>,
}

#[derive(Debug, Clone)]
pub struct Encoder {
    conv_in: Conv2d,
    mid_block_1: ResnetBlock,
    mid_attn_1: AttnBlock,
    mid_block_2: ResnetBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    down: Vec<DownBlock>,
}

impl Encoder {
    pub fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let base_ch = cfg.block_out_channels[0];
        let mut block_in = base_ch;
        let conv_in = conv2d(cfg.in_channels, block_in, 3, conv_cfg, vb.pp("conv_in"))?;

        let mut down = Vec::with_capacity(cfg.block_out_channels.len());
        let vb_d = vb.pp("down");
        for (i_level, out_channels) in cfg.block_out_channels.iter().enumerate() {
            let mut block = Vec::with_capacity(cfg.layers_per_block);
            let vb_d = vb_d.pp(i_level);
            let vb_b = vb_d.pp("block");
            block_in = if i_level == 0 {
                base_ch
            } else {
                cfg.block_out_channels[i_level - 1]
            };
            let block_out = *out_channels;
            for i_block in 0..cfg.layers_per_block {
                let b = ResnetBlock::new(block_in, block_out, vb_b.pp(i_block), cfg)?;
                block.push(b);
                block_in = block_out;
            }
            let downsample = if i_level != cfg.block_out_channels.len() - 1 {
                Some(Downsample::new(block_in, vb_d.pp("downsample"))?)
            } else {
                None
            };
            let block = DownBlock { block, downsample };
            down.push(block)
        }

        let mid_block_1 = ResnetBlock::new(block_in, block_in, vb.pp("mid.block_1"), cfg)?;
        let mid_attn_1 = AttnBlock::new(block_in, vb.pp("mid.attn_1"), cfg)?;
        let mid_block_2 = ResnetBlock::new(block_in, block_in, vb.pp("mid.block_2"), cfg)?;
        let conv_out = conv2d(
            block_in,
            2 * cfg.latent_channels,
            3,
            conv_cfg,
            vb.pp("conv_out"),
        )?;
        let norm_out = group_norm(cfg.norm_num_groups, block_in, 1e-6, vb.pp("norm_out"))?;
        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn_1,
            mid_block_2,
            norm_out,
            conv_out,
            down,
        })
    }
}

impl candle_nn::Module for Encoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = Convolution.forward_2d(&self.conv_in, xs)?;
        for block in self.down.iter() {
            for b in block.block.iter() {
                h = b.forward(&h)?
            }
            if let Some(ds) = block.downsample.as_ref() {
                h = ds.forward(&h)?
            }
        }
        h = self.mid_block_1.forward(&h)?;
        h = self.mid_attn_1.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;
        h = self.norm_out.forward(&h)?;
        h = candle_nn::Activation::Swish.forward(&h)?;
        Convolution.forward_2d(&self.conv_out, &h)
    }
}

#[derive(Debug, Clone)]
struct UpBlock {
    block: Vec<ResnetBlock>,
    upsample: Option<Upsample>,
}

#[derive(Debug, Clone)]
pub struct Decoder {
    conv_in: Conv2d,
    mid_block_1: ResnetBlock,
    mid_attn_1: AttnBlock,
    mid_block_2: ResnetBlock,
    norm_out: GroupNorm,
    conv_out: Conv2d,
    up: Vec<UpBlock>,
}

impl Decoder {
    pub fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_cfg = candle_nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let base_ch = cfg.block_out_channels[0];
        let mut block_in = cfg.block_out_channels.last().copied().unwrap_or(base_ch);
        let conv_in = conv2d(cfg.latent_channels, block_in, 3, conv_cfg, vb.pp("conv_in"))?;
        let mid_block_1 = ResnetBlock::new(block_in, block_in, vb.pp("mid.block_1"), cfg)?;
        let mid_attn_1 = AttnBlock::new(block_in, vb.pp("mid.attn_1"), cfg)?;
        let mid_block_2 = ResnetBlock::new(block_in, block_in, vb.pp("mid.block_2"), cfg)?;

        let mut up = Vec::with_capacity(cfg.block_out_channels.len());
        let vb_u = vb.pp("up");
        for (i_level, out_channels) in cfg.block_out_channels.iter().enumerate().rev() {
            let block_out = *out_channels;
            let vb_u = vb_u.pp(i_level);
            let vb_b = vb_u.pp("block");
            let mut block = Vec::with_capacity(cfg.layers_per_block + 1);
            for i_block in 0..=cfg.layers_per_block {
                let b = ResnetBlock::new(block_in, block_out, vb_b.pp(i_block), cfg)?;
                block.push(b);
                block_in = block_out;
            }
            let upsample = if i_level != 0 {
                Some(Upsample::new(block_in, vb_u.pp("upsample"))?)
            } else {
                None
            };
            let block = UpBlock { block, upsample };
            up.push(block)
        }
        up.reverse();

        let norm_out = group_norm(cfg.norm_num_groups, block_in, 1e-6, vb.pp("norm_out"))?;
        let conv_out = conv2d(block_in, cfg.out_channels, 3, conv_cfg, vb.pp("conv_out"))?;
        Ok(Self {
            conv_in,
            mid_block_1,
            mid_attn_1,
            mid_block_2,
            norm_out,
            conv_out,
            up,
        })
    }
}

impl candle_nn::Module for Decoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut h = Convolution.forward_2d(&self.conv_in, xs)?;
        h = self.mid_block_1.forward(&h)?;
        h = self.mid_attn_1.forward(&h)?;
        h = self.mid_block_2.forward(&h)?;
        for block in self.up.iter().rev() {
            for b in block.block.iter() {
                h = b.forward(&h)?
            }
            if let Some(us) = block.upsample.as_ref() {
                h = us.forward(&h)?
            }
        }
        h = self.norm_out.forward(&h)?;
        h = candle_nn::Activation::Swish.forward(&h)?;
        Convolution.forward_2d(&self.conv_out, &h)
    }
}

#[derive(Debug, Clone)]
pub struct DiagonalGaussian {
    sample: bool,
    chunk_dim: usize,
}

impl DiagonalGaussian {
    pub fn new(sample: bool, chunk_dim: usize) -> Result<Self> {
        Ok(Self { sample, chunk_dim })
    }
}

impl candle_nn::Module for DiagonalGaussian {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let chunks = xs.chunk(2, self.chunk_dim)?;
        if self.sample {
            let std = (&chunks[1] * 0.5)?.exp()?;
            &chunks[0] + (std * chunks[0].randn_like(0., 1.))?
        } else {
            Ok(chunks[0].clone())
        }
    }
}

#[derive(Debug, Clone)]
pub struct AutoEncoder {
    encoder: Encoder,
    decoder: Decoder,
    reg: DiagonalGaussian,
    shift_factor: f64,
    scale_factor: f64,
}

impl AutoEncoder {
    pub fn new(cfg: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        let encoder = Encoder::new(cfg, vb.pp("encoder"))?;
        let decoder = Decoder::new(cfg, vb.pp("decoder"))?;
        let reg = DiagonalGaussian::new(true, 1)?;
        Ok(Self {
            encoder,
            decoder,
            reg,
            scale_factor: cfg.scaling_factor,
            shift_factor: cfg.shift_factor,
        })
    }

    pub fn encode(&self, xs: &Tensor) -> Result<Tensor> {
        let z = xs.apply(&self.encoder)?.apply(&self.reg)?;
        (z - self.shift_factor)? * self.scale_factor
    }
    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = ((xs / self.scale_factor)? + self.shift_factor)?;
        xs.apply(&self.decoder)
    }
}

impl candle_core::Module for AutoEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.decode(&self.encode(xs)?)
    }
}
