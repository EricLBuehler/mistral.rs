use candle_core::{Result, Tensor};
use candle_nn::{Activation, Conv2d, Conv2dConfig, Module};
use mistralrs_quant::{Convolution, ShardedVarBuilder};
use tracing::warn;

use crate::{
    attention::SdpaParams,
    layers::{conv2d, conv2d_no_bias, Sdpa},
    utils::unvarbuilder::UnVarBuilder,
};

use std::fmt::Debug;

#[derive(Debug, Clone)]
pub enum BlockType {
    EdgeResidual {
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        expand_ratio: f64,
        // Odd. Neither MLX nor timm use it.
        #[allow(unused)]
        is_multiscale: bool,
    },
    UniversalInvertedResidual {
        out_channels: usize,
        start_kernel_size: usize,
        mid_kernel_size: usize,
        stride: usize,
        expand_ratio: f64,
        // Odd. Neither MLX nor timm use it.
        #[allow(unused)]
        is_multiscale: bool,
    },
    MultiQueryAttention {
        num_heads: usize,
        kv_dim: usize,
        kv_stride: usize,
        // Odd. Neither MLX nor timm use it.
        #[allow(unused)]
        is_multiscale: bool,
    },
}

// Helper function to calculate same padding
fn pad_same(x: &Tensor, kernel_size: usize, stride: usize, dilation: usize) -> Result<Tensor> {
    let (_, _, ih, iw) = x.dims4()?;
    let oh = ih.div_ceil(stride);
    let ow = iw.div_ceil(stride);

    // Calculate effective kernel size
    let effective_kernel_h = dilation * (kernel_size - 1) + 1;
    let effective_kernel_w = dilation * (kernel_size - 1) + 1;

    let pad_h = ((oh - 1) * stride + effective_kernel_h).saturating_sub(ih);
    let pad_w = ((ow - 1) * stride + effective_kernel_w).saturating_sub(iw);

    let pad_top = pad_h / 2;
    let pad_bottom = pad_h - pad_top;
    let pad_left = pad_w / 2;
    let pad_right = pad_w - pad_left;

    if pad_h > 0 || pad_w > 0 {
        x.pad_with_zeros(2, pad_top, pad_bottom)?
            .pad_with_zeros(3, pad_left, pad_right)
    } else {
        Ok(x.clone())
    }
}

// Conv2d with same padding
#[derive(Debug, Clone)]
struct Conv2dSame {
    conv: Conv2d,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
}

impl Conv2dSame {
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 0, // We'll handle padding manually
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };

        let conv = if bias {
            conv2d(in_channels, out_channels, kernel_size, cfg, vb)?
        } else {
            conv2d_no_bias(in_channels, out_channels, kernel_size, cfg, vb)?
        };

        Ok(Self {
            conv,
            kernel_size,
            stride,
            dilation,
        })
    }
}

impl Module for Conv2dSame {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = pad_same(x, self.kernel_size, self.stride, self.dilation)?;
        Convolution.forward_2d(&self.conv, &x)
    }
}

#[derive(Debug, Clone)]
struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    fn new(dims: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let weight = vb.get(dims, "weight")?;
        Ok(Self { weight, eps })
    }
}

#[derive(Debug, Clone)]
struct RMSNormAct2d {
    norm: RMSNorm,
    activation: Option<Activation>,
}

impl RMSNormAct2d {
    fn new(num_channels: usize, eps: f64, apply_act: bool, vb: ShardedVarBuilder) -> Result<Self> {
        let norm = RMSNorm::new(num_channels, eps, vb)?;
        let activation = if apply_act {
            Some(Activation::Gelu)
        } else {
            None
        };
        Ok(Self { norm, activation })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = candle_nn::ops::rms_norm(
            &x.permute((0, 2, 3, 1))?.contiguous()?,
            &self.norm.weight,
            self.norm.eps as f32,
        )?
        .permute((0, 3, 1, 2))?;

        // Optional activation
        if let Some(act) = &self.activation {
            x = x.apply(act)?;
        }

        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct LayerScale2d {
    gamma: Tensor,
}

impl LayerScale2d {
    fn new(dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self { gamma })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let c = self.gamma.dims1()?;
        let gamma = self.gamma.reshape((1, c, 1, 1))?;
        x.broadcast_mul(&gamma)
    }
}

#[derive(Debug, Clone)]
enum ConvType {
    Regular(Conv2d),
    Same(Conv2dSame),
}

impl Module for ConvType {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            ConvType::Regular(conv) => Convolution.forward_2d(conv, x),
            ConvType::Same(conv) => conv.forward(x),
        }
    }
}

#[derive(Debug, Clone)]
struct ConvNormAct {
    conv: ConvType,
    norm: Option<RMSNormAct2d>,
}

impl ConvNormAct {
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_chs: usize,
        out_chs: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        apply_act: bool,
        eps: f64,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        // Use Conv2dSame for depthwise convolutions (groups == in_chs or groups == out_chs)
        // and for convolutions with kernel_size > 1 where padding would be needed
        let use_same_padding = groups == in_chs || (kernel_size > 1 && padding > 0);

        let conv = if use_same_padding {
            ConvType::Same(Conv2dSame::new(
                in_chs,
                out_chs,
                kernel_size,
                stride,
                1, // dilation
                groups,
                bias,
                vb.pp("conv"),
            )?)
        } else {
            let conv_cfg = Conv2dConfig {
                stride,
                padding,
                groups,
                ..Default::default()
            };
            let conv = if bias {
                conv2d(in_chs, out_chs, kernel_size, conv_cfg, vb.pp("conv"))?
            } else {
                conv2d_no_bias(in_chs, out_chs, kernel_size, conv_cfg, vb.pp("conv"))?
            };
            ConvType::Regular(conv)
        };

        let norm = Some(RMSNormAct2d::new(out_chs, eps, apply_act, vb.pp("bn"))?);

        Ok(Self { conv, norm })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv.forward(x)?;
        if let Some(norm) = &self.norm {
            x = norm.forward(&x)?;
        }
        Ok(x)
    }
}

impl Module for ConvNormAct {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

#[derive(Debug, Clone)]
struct EdgeResidual {
    conv_exp: Conv2dSame,
    bn1: RMSNormAct2d,
    conv_pwl: Conv2d,
    bn2: RMSNormAct2d,
    has_skip: bool,
}

impl EdgeResidual {
    fn new(
        in_chs: usize,
        out_chs: usize,
        exp_kernel_size: usize,
        stride: usize,
        expand_ratio: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let mid_chs = make_divisible(in_chs as f64 * expand_ratio, 8);
        let has_skip = in_chs == out_chs && stride == 1;

        let conv_exp = Conv2dSame::new(
            in_chs,
            mid_chs,
            exp_kernel_size,
            stride,
            1, // dilation
            1, // groups
            false,
            vb.pp("conv_exp"),
        )?;

        let bn1 = RMSNormAct2d::new(mid_chs, 1e-5, true, vb.pp("bn1"))?;

        let conv_pwl_cfg = Conv2dConfig {
            ..Default::default()
        };

        let conv_pwl = conv2d_no_bias(mid_chs, out_chs, 1, conv_pwl_cfg, vb.pp("conv_pwl"))?;

        let bn2 = RMSNormAct2d::new(out_chs, 1e-5, false, vb.pp("bn2"))?;

        Ok(Self {
            conv_exp,
            bn1,
            conv_pwl,
            bn2,
            has_skip,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = x.clone();
        let mut x = self.conv_exp.forward(x)?;
        x = self.bn1.forward(&x)?;
        x = Convolution.forward_2d(&self.conv_pwl, &x)?;
        x = self.bn2.forward(&x)?;

        if self.has_skip {
            x = (x + shortcut)?;
        }

        Ok(x)
    }
}

impl Module for EdgeResidual {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

#[derive(Debug, Clone)]
struct UniversalInvertedResidual {
    dw_start: Option<ConvNormAct>,
    pw_exp: ConvNormAct,
    dw_mid: Option<ConvNormAct>,
    pw_proj: ConvNormAct,
    layer_scale: Option<LayerScale2d>,
    has_skip: bool,
}

impl UniversalInvertedResidual {
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_chs: usize,
        out_chs: usize,
        dw_kernel_size_start: usize,
        dw_kernel_size_mid: usize,
        stride: usize,
        exp_ratio: f64,
        layer_scale_init_value: Option<f64>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let has_skip = in_chs == out_chs && stride == 1;
        let mid_chs = make_divisible(in_chs as f64 * exp_ratio, 8);

        // DW start (optional)
        let dw_start = if dw_kernel_size_start > 0 {
            let dw_start_stride = if dw_kernel_size_mid > 0 { 1 } else { stride };
            Some(ConvNormAct::new(
                in_chs,
                in_chs,
                dw_kernel_size_start,
                dw_start_stride,
                dw_kernel_size_start / 2,
                in_chs, // Depthwise
                false,
                1e-5,
                false,
                vb.pp("dw_start"),
            )?)
        } else {
            None
        };

        // PW expansion
        let pw_exp = ConvNormAct::new(
            in_chs,
            mid_chs,
            1,
            1,
            0,
            1,
            true,
            1e-5,
            false,
            vb.pp("pw_exp"),
        )?;

        // DW mid (optional)
        let dw_mid = if dw_kernel_size_mid > 0 {
            Some(ConvNormAct::new(
                mid_chs,
                mid_chs,
                dw_kernel_size_mid,
                stride,
                dw_kernel_size_mid / 2,
                mid_chs, // Depthwise
                true,
                1e-5,
                false,
                vb.pp("dw_mid"),
            )?)
        } else {
            None
        };

        // PW projection
        let pw_proj = ConvNormAct::new(
            mid_chs,
            out_chs,
            1,
            1,
            0,
            1,
            false,
            1e-5,
            false,
            vb.pp("pw_proj"),
        )?;

        // Layer scale
        let layer_scale = if layer_scale_init_value.is_some() {
            Some(LayerScale2d::new(out_chs, vb.pp("layer_scale"))?)
        } else {
            None
        };

        Ok(Self {
            dw_start,
            pw_exp,
            dw_mid,
            pw_proj,
            layer_scale,
            has_skip,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = x.clone();

        let mut x = x.clone();
        if let Some(dw) = &self.dw_start {
            x = dw.forward(&x)?;
        }

        x = self.pw_exp.forward(&x)?;

        if let Some(dw) = &self.dw_mid {
            x = dw.forward(&x)?;
        }

        x = self.pw_proj.forward(&x)?;

        if let Some(ls) = &self.layer_scale {
            x = ls.forward(&x)?;
        }

        if self.has_skip {
            x = (x + shortcut)?;
        }

        Ok(x)
    }
}

impl Module for UniversalInvertedResidual {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

#[derive(Debug, Clone)]
struct MultiQueryAttention2d {
    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
    scale: f64,
    query_proj: Conv2d,
    key_down_conv: Option<Conv2dSame>,
    key_norm: Option<RMSNormAct2d>,
    key_proj: Conv2d,
    value_down_conv: Option<Conv2dSame>,
    value_norm: Option<RMSNormAct2d>,
    value_proj: Conv2d,
    output_proj: Conv2d,
}

impl MultiQueryAttention2d {
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        dim_out: usize,
        num_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kv_stride: usize,
        dw_kernel_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let scale = (key_dim as f64).powf(-0.5);

        // Query projection
        let query_proj = conv2d_no_bias(
            dim,
            num_heads * key_dim,
            1,
            Conv2dConfig::default(),
            vb.pp("query").pp("proj"),
        )?;

        // Key path
        let (key_down_conv, key_norm) = if kv_stride > 1 {
            let down_conv = Conv2dSame::new(
                dim,
                dim,
                dw_kernel_size,
                kv_stride,
                1,   // dilation
                dim, // Depthwise
                false,
                vb.pp("key").pp("down_conv"),
            )?;
            let norm = RMSNormAct2d::new(dim, 1e-6, false, vb.pp("key").pp("norm"))?;
            (Some(down_conv), Some(norm))
        } else {
            (None, None)
        };

        let key_proj = conv2d_no_bias(
            dim,
            key_dim,
            1,
            Conv2dConfig::default(),
            vb.pp("key").pp("proj"),
        )?;

        // Value path
        let (value_down_conv, value_norm) = if kv_stride > 1 {
            let down_conv = Conv2dSame::new(
                dim,
                dim,
                dw_kernel_size,
                kv_stride,
                1,   // dilation
                dim, // Depthwise
                false,
                vb.pp("value").pp("down_conv"),
            )?;
            let norm = RMSNormAct2d::new(dim, 1e-6, false, vb.pp("value").pp("norm"))?;
            (Some(down_conv), Some(norm))
        } else {
            (None, None)
        };

        let value_proj = conv2d_no_bias(
            dim,
            value_dim,
            1,
            Conv2dConfig::default(),
            vb.pp("value").pp("proj"),
        )?;

        // Output projection
        let output_proj = conv2d_no_bias(
            value_dim * num_heads,
            dim_out,
            1,
            Conv2dConfig::default(),
            vb.pp("output").pp("proj"),
        )?;

        Ok(Self {
            num_heads,
            key_dim,
            value_dim,
            scale,
            query_proj,
            key_down_conv,
            key_norm,
            key_proj,
            value_down_conv,
            value_norm,
            value_proj,
            output_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, _c, h, w) = x.dims4()?;

        // Query projection and reshape
        // [B, H, W, C] -> [B, H, W, num_heads * key_dim] -> [B, H*W, num_heads, key_dim] -> [B, num_heads, H*W, key_dim]
        let mut q = Convolution.forward_2d(&self.query_proj, x)?;
        q = q
            .permute((0, 2, 3, 1))? // NCHW -> NHWC
            .reshape((b, h * w, self.num_heads, self.key_dim))?
            .permute((0, 2, 1, 3))?; // [B, num_heads, H*W, key_dim]

        // Key projection and reshape
        let mut k = x.clone();
        if let (Some(down_conv), Some(norm)) = (&self.key_down_conv, &self.key_norm) {
            k = down_conv.forward(&k)?;
            k = norm.forward(&k)?;
        }
        k = Convolution.forward_2d(&self.key_proj, &k)?;
        let (_, _, kh, kw) = k.dims4()?;
        // [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C] -> [B, 1, H*W, C]
        k = k
            .permute((0, 2, 3, 1))? // NCHW -> NHWC
            .reshape((b, kh * kw, self.key_dim))?
            .unsqueeze(1)?; // [B, 1, kh*kw, key_dim]

        // Value projection and reshape
        let mut v = x.clone();
        if let (Some(down_conv), Some(norm)) = (&self.value_down_conv, &self.value_norm) {
            v = down_conv.forward(&v)?;
            v = norm.forward(&v)?;
        }
        v = Convolution.forward_2d(&self.value_proj, &v)?;
        let (_, _, vh, vw) = v.dims4()?;
        // [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C] -> [B, 1, H*W, C]
        v = v
            .permute((0, 2, 3, 1))? // NCHW -> NHWC
            .reshape((b, vh * vw, self.value_dim))?
            .unsqueeze(1)?; // [B, 1, vh*vw, value_dim]

        let sdpa_params = SdpaParams {
            n_kv_groups: self.num_heads,
            softcap: None,
            softmax_scale: self.scale as f32,
            sliding_window: None,
            sinks: None,
        };
        let mut o = Sdpa.run_attention_noflash(&q, &k, &v, None, &sdpa_params)?;

        // Reshape output back
        // [B, num_heads, H*W, value_dim] -> [B, H*W, num_heads, value_dim] -> [B, H, W, num_heads * value_dim]
        o = o
            .permute((0, 2, 1, 3))? // [B, H*W, num_heads, value_dim]
            .reshape((b, h, w, self.num_heads * self.value_dim))?
            .permute((0, 3, 1, 2))?; // NHWC -> NCHW

        o = Convolution.forward_2d(&self.output_proj, &o)?;

        Ok(o)
    }
}

#[derive(Debug, Clone)]
struct MobileAttention {
    norm: RMSNormAct2d,
    attn: MultiQueryAttention2d,
    layer_scale: Option<LayerScale2d>,
    has_skip: bool,
}

impl MobileAttention {
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_chs: usize,
        out_chs: usize,
        stride: usize,
        num_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kv_stride: usize,
        dw_kernel_size: usize,
        layer_scale_init_value: Option<f64>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let has_skip = stride == 1 && in_chs == out_chs;

        let norm = RMSNormAct2d::new(in_chs, 1e-5, false, vb.pp("norm"))?;

        let attn = MultiQueryAttention2d::new(
            in_chs,
            out_chs,
            num_heads,
            key_dim,
            value_dim,
            kv_stride,
            dw_kernel_size,
            vb.pp("attn"),
        )?;

        let layer_scale = if layer_scale_init_value.is_some() {
            Some(LayerScale2d::new(out_chs, vb.pp("layer_scale"))?)
        } else {
            None
        };

        Ok(Self {
            norm,
            attn,
            layer_scale,
            has_skip,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shortcut = x.clone();

        let mut x = self.norm.forward(x)?;
        x = self.attn.forward(&x)?;

        if let Some(ls) = &self.layer_scale {
            x = ls.forward(&x)?;
        }

        if self.has_skip {
            x = (x + shortcut)?;
        }

        Ok(x)
    }
}

impl Module for MobileAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

// Helper functions
pub fn make_divisible(v: f64, divisor: usize) -> usize {
    let divisor = divisor as f64;
    let min_value = divisor;
    let new_v = ((v + divisor / 2.0) / divisor).floor() * divisor;
    let new_v = new_v.max(min_value) as usize;
    // Make sure that round down does not go down by more than 10%.
    if (new_v as f64) < 0.9 * v {
        new_v + divisor as usize
    } else {
        new_v
    }
}

// Multi-scale fusion adapter
#[derive(Debug, Clone)]
struct MobileNetV5MultiScaleFusionAdapter {
    output_resolution: (usize, usize),
    ffn: UniversalInvertedResidual,
    norm: RMSNormAct2d,
}

impl MobileNetV5MultiScaleFusionAdapter {
    fn new(
        in_chs: Vec<usize>,
        out_chs: usize,
        output_resolution: (usize, usize),
        expansion_ratio: f64,
        use_layer_scale: bool,
        layer_scale_init_value: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let in_channels = in_chs.iter().sum();

        let layer_scale = if use_layer_scale {
            Some(layer_scale_init_value)
        } else {
            None
        };

        let ffn = UniversalInvertedResidual::new(
            in_channels,
            out_chs,
            0, // dw_kernel_size_start
            0, // dw_kernel_size_mid
            1, // stride
            expansion_ratio,
            layer_scale,
            vb.pp("ffn"),
        )?;

        let norm = RMSNormAct2d::new(out_chs, 1e-6, false, vb.pp("norm"))?;

        Ok(Self {
            output_resolution,
            ffn,
            norm,
        })
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor> {
        // Get the highest resolution from the first input
        let (_, _, h0, w0) = inputs[0].dims4()?;

        // Resize inputs to match highest resolution
        let mut resized_inputs = Vec::with_capacity(inputs.len());
        for img in inputs {
            let (_, _, h, w) = img.dims4()?;
            if h < h0 || w < w0 {
                // Use bilinear interpolation as nearest equivalent
                let resized = img.upsample_nearest2d(h0, w0)?;
                resized_inputs.push(resized);
            } else {
                resized_inputs.push(img.clone());
            }
        }

        // Concatenate along channel dimension
        let channel_cat_imgs = Tensor::cat(&resized_inputs, 1)?;

        // Apply FFN first
        let mut img = self.ffn.forward(&channel_cat_imgs)?;

        // Resize / pool to target output resolution *before* final normalisation
        let (out_h, out_w) = self.output_resolution;
        if h0 != out_h || w0 != out_w {
            if h0 % out_h != 0 || w0 % out_w != 0 {
                // Fallback to bilinear interpolation if input spatial dims are
                // not integer multiples of the desired output size. We use
                // Candle’s nearest-neighbour upsample as the closest available op.
                img = img.upsample_nearest2d(out_h, out_w)?;
            } else {
                let h_stride = h0 / out_h;
                let w_stride = w0 / out_w;
                img = img.avg_pool2d((h_stride, w_stride))?;
            }
        }

        img = self.norm.forward(&img)?;

        Ok(img)
    }
}

// Constants for vision tower configuration
pub const INPUT_CHANNELS: usize = 3;
pub const STEM_OUT_CHANNELS: usize = 64;
pub const STEM_KERNEL_SIZE: usize = 3;
pub const MSFA_IN_CHANNELS: &[usize] = &[640, 1280];
pub const MSFA_OUT_CHANNELS: usize = 2048;
pub const MSFA_EXPANSION_RATIO: f64 = 2.0;

// Architecture definition for Gemma3n
pub fn gemma3n_mobilenet_def() -> Vec<Vec<BlockType>> {
    vec![
        // Stage 1: Edge Residuals
        vec![
            BlockType::EdgeResidual {
                out_channels: 128,
                kernel_size: 3,
                stride: 2,
                expand_ratio: 4.0,
                is_multiscale: false,
            },
            BlockType::EdgeResidual {
                out_channels: 128,
                kernel_size: 3,
                stride: 1,
                expand_ratio: 4.0,
                is_multiscale: false,
            },
            BlockType::EdgeResidual {
                out_channels: 128,
                kernel_size: 3,
                stride: 1,
                expand_ratio: 4.0,
                is_multiscale: false,
            },
        ],
        // Stage 2: Universal Inverted Residuals
        vec![
            BlockType::UniversalInvertedResidual {
                out_channels: 256,
                start_kernel_size: 3,
                mid_kernel_size: 5,
                stride: 2,
                expand_ratio: 6.0,
                is_multiscale: false,
            },
            BlockType::UniversalInvertedResidual {
                out_channels: 256,
                start_kernel_size: 5,
                mid_kernel_size: 0,
                stride: 1,
                expand_ratio: 4.0,
                is_multiscale: false,
            },
            BlockType::UniversalInvertedResidual {
                out_channels: 256,
                start_kernel_size: 3,
                mid_kernel_size: 0,
                stride: 1,
                expand_ratio: 4.0,
                is_multiscale: false,
            },
            BlockType::UniversalInvertedResidual {
                out_channels: 256,
                start_kernel_size: 5,
                mid_kernel_size: 0,
                stride: 1,
                expand_ratio: 4.0,
                is_multiscale: false,
            },
            BlockType::UniversalInvertedResidual {
                out_channels: 256,
                start_kernel_size: 3,
                mid_kernel_size: 0,
                stride: 1,
                expand_ratio: 4.0,
                is_multiscale: false,
            },
        ],
        // Stage 3: Universal Inverted Residuals with Multi-Query Attention
        {
            let mut blocks = vec![BlockType::UniversalInvertedResidual {
                out_channels: 640,
                start_kernel_size: 5,
                mid_kernel_size: 5,
                stride: 2,
                expand_ratio: 6.0,
                is_multiscale: false,
            }];
            // Add 7 UIR blocks
            for _ in 0..7 {
                blocks.push(BlockType::UniversalInvertedResidual {
                    out_channels: 640,
                    start_kernel_size: 5,
                    mid_kernel_size: 0,
                    stride: 1,
                    expand_ratio: 4.0,
                    is_multiscale: false,
                });
            }
            // Add one special UIR block
            blocks.push(BlockType::UniversalInvertedResidual {
                out_channels: 640,
                start_kernel_size: 0,
                mid_kernel_size: 0,
                stride: 1,
                expand_ratio: 1.0,
                is_multiscale: false,
            });
            // Add 13 pairs of MMQA + UIR
            for _ in 0..13 {
                blocks.push(BlockType::MultiQueryAttention {
                    num_heads: 12,
                    kv_dim: 64,
                    kv_stride: 2,
                    is_multiscale: false,
                });
                blocks.push(BlockType::UniversalInvertedResidual {
                    out_channels: 640,
                    start_kernel_size: 0,
                    mid_kernel_size: 0,
                    stride: 1,
                    expand_ratio: 2.0,
                    is_multiscale: false,
                });
            }
            // Final pair with multiscale
            blocks.push(BlockType::MultiQueryAttention {
                num_heads: 12,
                kv_dim: 64,
                kv_stride: 2,
                is_multiscale: false,
            });
            blocks.push(BlockType::UniversalInvertedResidual {
                out_channels: 640,
                start_kernel_size: 0,
                mid_kernel_size: 0,
                stride: 1,
                expand_ratio: 2.0,
                is_multiscale: true,
            });
            blocks
        },
        // Stage 4: Universal Inverted Residuals with Multi-Query Attention
        {
            let mut blocks = vec![BlockType::UniversalInvertedResidual {
                out_channels: 1280,
                start_kernel_size: 5,
                mid_kernel_size: 5,
                stride: 2,
                expand_ratio: 6.0,
                is_multiscale: false,
            }];
            // Add 18 pairs of MMQA + UIR
            for _ in 0..18 {
                blocks.push(BlockType::MultiQueryAttention {
                    num_heads: 16,
                    kv_dim: 96,
                    kv_stride: 1,
                    is_multiscale: false,
                });
                blocks.push(BlockType::UniversalInvertedResidual {
                    out_channels: 1280,
                    start_kernel_size: 0,
                    mid_kernel_size: 0,
                    stride: 1,
                    expand_ratio: 2.0,
                    is_multiscale: false,
                });
            }
            // Final pair with multiscale
            blocks.push(BlockType::MultiQueryAttention {
                num_heads: 16,
                kv_dim: 96,
                kv_stride: 1,
                is_multiscale: false,
            });
            blocks.push(BlockType::UniversalInvertedResidual {
                out_channels: 1280,
                start_kernel_size: 0,
                mid_kernel_size: 0,
                stride: 1,
                expand_ratio: 2.0,
                is_multiscale: true,
            });
            blocks
        },
    ]
}

// Enum wrapper for blocks to enable dynamic dispatch
#[derive(Debug, Clone)]
enum Block {
    EdgeResidual(EdgeResidual),
    UniversalInvertedResidual(UniversalInvertedResidual),
    MobileAttention(MobileAttention),
}

impl Module for Block {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Block::EdgeResidual(m) => m.forward(x),
            Block::UniversalInvertedResidual(m) => m.forward(x),
            Block::MobileAttention(m) => m.forward(x),
        }
    }
}

// Main vision tower
#[derive(Debug, Clone)]
pub struct VisionTower {
    conv_stem: ConvNormAct,
    blocks: Vec<Vec<Block>>,
    msfa: MobileNetV5MultiScaleFusionAdapter,
    msfa_indices: Vec<usize>,
    old_vision_tower: bool,
}

impl VisionTower {
    pub fn new(vb: ShardedVarBuilder) -> Result<Self> {
        // Some models have invalid vision tower weights from the old gemma 3n upload
        // https://github.com/EricLBuehler/mistral.rs/issues/1592
        let old_vision_tower = !vb.contains_tensor("conv_stem.conv.bias");
        if old_vision_tower {
            warn!(
                "This model contains invalid vision tower weights from an old Gemma 3n upload.
See: https://github.com/EricLBuehler/mistral.rs/issues/1592

The vision tower for this model will still be loaded, but you might experience degraded quality."
            );
        }
        let conv_stem_bias = !old_vision_tower;
        // Initial stem convolution
        let conv_stem = ConvNormAct::new(
            3,              // in_chs
            64,             // out_chs
            3,              // kernel_size
            2,              // stride
            1,              // padding
            1,              // groups
            true,           // apply_act
            1e-5,           // eps
            conv_stem_bias, // bias
            vb.pp("conv_stem"),
        )?;

        // Build blocks according to architecture definition
        let block_defs = gemma3n_mobilenet_def();
        let mut blocks = Vec::with_capacity(block_defs.len());
        let mut in_chs = 64;

        for (stage_idx, stage_blocks) in block_defs.iter().enumerate() {
            let mut stage = Vec::with_capacity(stage_blocks.len());

            for (block_idx, block_type) in stage_blocks.iter().enumerate() {
                let block = match block_type {
                    BlockType::EdgeResidual {
                        out_channels,
                        kernel_size,
                        stride,
                        expand_ratio,
                        ..
                    } => {
                        let edge_res = EdgeResidual::new(
                            in_chs,
                            *out_channels,
                            *kernel_size,
                            *stride,
                            *expand_ratio,
                            vb.pp(format!("blocks.{stage_idx}.{block_idx}")),
                        )?;
                        in_chs = *out_channels;
                        Block::EdgeResidual(edge_res)
                    }
                    BlockType::UniversalInvertedResidual {
                        out_channels,
                        start_kernel_size,
                        mid_kernel_size,
                        stride,
                        expand_ratio,
                        ..
                    } => {
                        let uir = UniversalInvertedResidual::new(
                            in_chs,
                            *out_channels,
                            *start_kernel_size,
                            *mid_kernel_size,
                            *stride,
                            *expand_ratio,
                            Some(1e-5), // layer_scale_init_value
                            vb.pp(format!("blocks.{stage_idx}.{block_idx}")),
                        )?;
                        in_chs = *out_channels;
                        Block::UniversalInvertedResidual(uir)
                    }
                    BlockType::MultiQueryAttention {
                        num_heads,
                        kv_dim,
                        kv_stride,
                        ..
                    } => {
                        let ma = MobileAttention::new(
                            in_chs,
                            in_chs, // out_chs same as in_chs
                            1,      // stride
                            *num_heads,
                            *kv_dim,
                            *kv_dim, // value_dim same as key_dim
                            *kv_stride,
                            3,          // dw_kernel_size
                            Some(1e-5), // layer_scale_init_value
                            vb.pp(format!("blocks.{stage_idx}.{block_idx}")),
                        )?;
                        Block::MobileAttention(ma)
                    }
                };
                stage.push(block);
            }
            blocks.push(stage);
        }

        // Multi-scale fusion adapter
        // Collecting from stages 3 and 4 (after 640 and 1280 channel blocks)
        let msfa = MobileNetV5MultiScaleFusionAdapter::new(
            vec![640, 1280], // in_chs from stages 3 and 4
            2048,            // out_chs
            (16, 16),        // output_resolution
            2.0,             // expansion_ratio
            false,           // use_layer_scale
            1e-5,            // layer_scale_init_value
            vb.pp("msfa"),
        )?;

        Ok(Self {
            conv_stem,
            blocks,
            msfa,
            msfa_indices: vec![3, 4], // Indices for multi-scale features
            old_vision_tower,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = if self.old_vision_tower {
            // Some models have invalid vision tower weights from the old gemma 3n upload
            // https://github.com/EricLBuehler/mistral.rs/issues/1592

            // This is a hack necessary because the weights for Gemma 3n are broken and require the image to be rotated.
            x.t()?
        } else {
            x.clone()
        };

        // Apply stem
        x = self.conv_stem.forward(&x)?;

        let mut intermediates = Vec::new();

        // Process blocks stage by stage
        for (stage_idx, stage) in self.blocks.iter().enumerate() {
            for block in stage {
                x = block.forward(&x)?;
            }

            // Collect intermediate features for multi-scale fusion
            if self.msfa_indices.contains(&(stage_idx + 1)) {
                intermediates.push(x.clone());
            }
        }

        // Apply multi-scale fusion adapter
        let x = self.msfa.forward(&intermediates)?;

        Ok(x)
    }

    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();

        // Add conv_stem tensors
        add_conv_norm_act(&uvb.pp("conv_stem"), &self.conv_stem);

        // Add blocks tensors
        for (stage_idx, stage) in self.blocks.iter().enumerate() {
            for (block_idx, block) in stage.iter().enumerate() {
                let uvb_block = uvb.pp(format!("blocks.{stage_idx}.{block_idx}"));
                match block {
                    Block::EdgeResidual(edge) => add_edge_residual(&uvb_block, edge),
                    Block::UniversalInvertedResidual(uir) => {
                        add_universal_inverted_residual(&uvb_block, uir)
                    }
                    Block::MobileAttention(ma) => add_mobile_attention(&uvb_block, ma),
                }
            }
        }

        // Add MSFA tensors
        add_msfa(&uvb.pp("msfa"), &self.msfa);

        uvb.to_safetensors()
    }
}

// Helper functions for adding residual tensors
fn add_conv_norm_act(uvb: &UnVarBuilder, cna: &ConvNormAct) {
    // Add conv layer
    match &cna.conv {
        ConvType::Regular(conv) => uvb.pp("conv").add(conv),
        ConvType::Same(conv) => uvb.pp("conv").add(&conv.conv),
    }

    // Add norm layer
    if let Some(norm) = &cna.norm {
        uvb.pp("bn").add_tensor("weight", norm.norm.weight.clone());
    }
}

fn add_edge_residual(uvb: &UnVarBuilder, edge: &EdgeResidual) {
    uvb.pp("conv_exp").add(&edge.conv_exp.conv);
    uvb.pp("bn1")
        .add_tensor("weight", edge.bn1.norm.weight.clone());
    uvb.pp("conv_pwl").add(&edge.conv_pwl);
    uvb.pp("bn2")
        .add_tensor("weight", edge.bn2.norm.weight.clone());
}

fn add_universal_inverted_residual(uvb: &UnVarBuilder, uir: &UniversalInvertedResidual) {
    // Add dw_start if present
    if let Some(dw_start) = &uir.dw_start {
        add_conv_norm_act(&uvb.pp("dw_start"), dw_start);
    }

    // Add pw_exp
    add_conv_norm_act(&uvb.pp("pw_exp"), &uir.pw_exp);

    // Add dw_mid if present
    if let Some(dw_mid) = &uir.dw_mid {
        add_conv_norm_act(&uvb.pp("dw_mid"), dw_mid);
    }

    // Add pw_proj
    add_conv_norm_act(&uvb.pp("pw_proj"), &uir.pw_proj);

    // Add layer_scale if present
    if let Some(layer_scale) = &uir.layer_scale {
        uvb.pp("layer_scale")
            .add_tensor("gamma", layer_scale.gamma.clone());
    }
}

fn add_mobile_attention(uvb: &UnVarBuilder, ma: &MobileAttention) {
    // Add norm
    uvb.pp("norm")
        .add_tensor("weight", ma.norm.norm.weight.clone());

    // Add attention components
    let uvb_attn = uvb.pp("attn");

    // Query projection
    uvb_attn.pp("query").pp("proj").add(&ma.attn.query_proj);

    // Key components
    if let Some(key_down_conv) = &ma.attn.key_down_conv {
        uvb_attn.pp("key").pp("down_conv").add(&key_down_conv.conv);
    }
    if let Some(key_norm) = &ma.attn.key_norm {
        uvb_attn
            .pp("key")
            .pp("norm")
            .add_tensor("weight", key_norm.norm.weight.clone());
    }
    uvb_attn.pp("key").pp("proj").add(&ma.attn.key_proj);

    // Value components
    if let Some(value_down_conv) = &ma.attn.value_down_conv {
        uvb_attn
            .pp("value")
            .pp("down_conv")
            .add(&value_down_conv.conv);
    }
    if let Some(value_norm) = &ma.attn.value_norm {
        uvb_attn
            .pp("value")
            .pp("norm")
            .add_tensor("weight", value_norm.norm.weight.clone());
    }
    uvb_attn.pp("value").pp("proj").add(&ma.attn.value_proj);

    // Output projection
    uvb_attn.pp("output").pp("proj").add(&ma.attn.output_proj);

    // Layer scale if present
    if let Some(layer_scale) = &ma.layer_scale {
        uvb.pp("layer_scale")
            .add_tensor("gamma", layer_scale.gamma.clone());
    }
}

fn add_msfa(uvb: &UnVarBuilder, msfa: &MobileNetV5MultiScaleFusionAdapter) {
    // Add FFN (UniversalInvertedResidual)
    add_universal_inverted_residual(&uvb.pp("ffn"), &msfa.ffn);

    // Add norm
    uvb.pp("norm")
        .add_tensor("weight", msfa.norm.norm.weight.clone());
}
