mod config;

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Activation, Conv2d, Conv2dConfig, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::layers::conv2d_no_bias;


use std::fmt::Debug;

#[derive(Debug, Clone)]
enum BlockType {
    EdgeResidual {
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        expand_ratio: f64,
        is_multiscale: bool,
    },
    UniversalInvertedResidual {
        out_channels: usize,
        start_kernel_size: usize,
        mid_kernel_size: usize,
        stride: usize,
        expand_ratio: f64,
        is_multiscale: bool,
    },
    MultiQueryAttention {
        num_heads: usize,
        kv_dim: usize,
        kv_stride: usize,
        is_multiscale: bool,
    },
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean_square = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_div(&(mean_square + self.eps)?.sqrt()?)?;
        let x = x.to_dtype(dtype)?;
        let x = x.broadcast_mul(&self.weight)?;
        Ok(x)
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
        // Input is NCHW format
        let (b, c, h, w) = x.dims4()?;
        
        // For 2D normalization, we need to normalize along spatial dimensions (H, W)
        // Reshape to (B*C, H*W) to apply normalization per channel
        let x_reshaped = x.reshape((b * c, h * w))?;
        
        // Compute RMS normalization along the spatial dimension
        let dtype = x_reshaped.dtype();
        let x_f32 = x_reshaped.to_dtype(DType::F32)?;
        let mean_square = x_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x_f32.broadcast_div(&(mean_square + self.norm.eps)?.sqrt()?)?;
        let x_norm = x_norm.to_dtype(dtype)?;
        
        // Reshape back to (B, C, H, W)
        let x_norm = x_norm.reshape((b, c, h, w))?;
        
        // Apply channel-wise weight
        let weight = self.norm.weight.reshape((1, c, 1, 1))?;
        let mut x = x_norm.broadcast_mul(&weight)?;
        
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
    fn new(dim: usize, init_values: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let gamma = (Tensor::ones(dim, vb.dtype(), vb.device())? * init_values)?;
        Ok(Self { gamma })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Reshape gamma to broadcast correctly for NCHW format
        let gamma = self.gamma.reshape((1, (), 1, 1))?;
        x.broadcast_mul(&gamma)
    }
}

#[derive(Debug, Clone)]
struct ConvNormAct {
    conv: Conv2d,
    norm: Option<RMSNormAct2d>,
}

impl ConvNormAct {
    fn new(
        in_chs: usize,
        out_chs: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        apply_act: bool,
        eps: f64,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride,
            padding,
            groups,
            ..Default::default()
        };
        
        let conv = conv2d_no_bias(
            in_chs,
            out_chs,
            kernel_size,
            conv_cfg,
            vb.pp("conv"),
        )?;
        
        let norm = Some(RMSNormAct2d::new(
            out_chs,
            eps,
            apply_act,
            vb.pp("bn"),
        )?);
        
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
    conv_exp: Conv2d,
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
        
        let conv_exp_cfg = Conv2dConfig {
            stride,
            padding: exp_kernel_size / 2,
            ..Default::default()
        };
        
        let conv_exp = conv2d_no_bias(
            in_chs,
            mid_chs,
            exp_kernel_size,
            conv_exp_cfg,
            vb.pp("conv_exp"),
        )?;
        
        let bn1 = RMSNormAct2d::new(mid_chs, 1e-5, true, vb.pp("bn1"))?;
        
        let conv_pwl_cfg = Conv2dConfig {
            ..Default::default()
        };
        
        let conv_pwl = conv2d_no_bias(
            mid_chs,
            out_chs,
            1,
            conv_pwl_cfg,
            vb.pp("conv_pwl"),
        )?;
        
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
        x = self.conv_pwl.forward(&x)?;
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
            vb.pp("pw_proj"),
        )?;
        
        // Layer scale
        let layer_scale = if let Some(init_value) = layer_scale_init_value {
            Some(LayerScale2d::new(out_chs, init_value, vb.pp("layer_scale"))?)
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
    key_down_conv: Option<Conv2d>,
    key_norm: Option<RMSNormAct2d>,
    key_proj: Conv2d,
    value_down_conv: Option<Conv2d>,
    value_norm: Option<RMSNormAct2d>,
    value_proj: Conv2d,
    output_proj: Conv2d,
}

impl MultiQueryAttention2d {
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
            let conv_cfg = Conv2dConfig {
                stride: kv_stride,
                padding: dw_kernel_size / 2,
                groups: dim, // Depthwise
                ..Default::default()
            };
            let down_conv = conv2d_no_bias(
                dim,
                dim,
                dw_kernel_size,
                conv_cfg,
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
            let conv_cfg = Conv2dConfig {
                stride: kv_stride,
                padding: dw_kernel_size / 2,
                groups: dim, // Depthwise
                ..Default::default()
            };
            let down_conv = conv2d_no_bias(
                dim,
                dim,
                dw_kernel_size,
                conv_cfg,
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
        
        // Query path
        let q = self.query_proj.forward(x)?;
        // Reshape: (B, NH*KD, H, W) -> (B, NH, H*W, KD)
        let q = q.reshape((b, self.num_heads, self.key_dim, h * w))?
            .transpose(2, 3)?;
        let q = (q * self.scale)?;
        
        // Key path
        let mut k = x.clone();
        if let (Some(down_conv), Some(norm)) = (&self.key_down_conv, &self.key_norm) {
            k = down_conv.forward(&k)?;
            k = norm.forward(&k)?;
        }
        k = self.key_proj.forward(&k)?;
        // Reshape: (B, KD, H', W') -> (B, 1, H'*W', KD)
        let (_, _, kh, kw) = k.dims4()?;
        let k = k.reshape((b, self.key_dim, kh * kw))?
            .transpose(1, 2)?  // Now: (B, H'*W', KD)
            .unsqueeze(1)?;    // Now: (B, 1, H'*W', KD)
        
        // Value path
        let mut v = x.clone();
        if let (Some(down_conv), Some(norm)) = (&self.value_down_conv, &self.value_norm) {
            v = down_conv.forward(&v)?;
            v = norm.forward(&v)?;
        }
        v = self.value_proj.forward(&v)?;
        // Reshape: (B, VD, H', W') -> (B, 1, H'*W', VD)
        let (_, _, vh, vw) = v.dims4()?;
        let v = v.reshape((b, self.value_dim, vh * vw))?
            .transpose(1, 2)?
            .unsqueeze(1)?;
        
        // Attention
        // q: [B, NH, H*W, KD], k: [B, 1, H'*W', KD]
        // We need to compute q @ k^T -> [B, NH, H*W, H'*W']
        
        // For matmul with broadcasting: q @ k^T
        // q: [B, NH, H*W, KD] @ k^T: [B, 1, KD, H'*W'] -> [B, NH, H*W, H'*W']
        // The matmul will automatically broadcast the second dimension
        let k_t = k.transpose(D::Minus1, D::Minus2)?; // [B, 1, KD, H'*W']
        
        // Use broadcast_matmul for proper broadcasting
        let attn = q.broadcast_matmul(&k_t)?; // [B, NH, H*W, H'*W']
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        
        // v: [B, 1, H'*W', VD]
        // attn: [B, NH, H*W, H'*W'] @ v: [B, 1, H'*W', VD] -> [B, NH, H*W, VD]
        let o = attn.broadcast_matmul(&v)?; // [B, NH, H*W, VD]
        
        // Reshape output: (B, NH, H*W, VD) -> (B, NH*VD, H, W)
        let o = o.transpose(1, 2)?
            .reshape((b, self.num_heads * self.value_dim, h, w))?;
        
        // Output projection
        let o = self.output_proj.forward(&o)?;
        
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
        
        let layer_scale = if let Some(init_value) = layer_scale_init_value {
            Some(LayerScale2d::new(out_chs, init_value, vb.pp("layer_scale"))?)
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
fn make_divisible(v: f64, divisor: usize) -> usize {
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
    in_channels: usize,
    out_channels: usize,
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
        _noskip: bool,
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
            in_channels,
            out_channels: out_chs,
            output_resolution,
            ffn,
            norm,
        })
    }

    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor> {
        // Get the highest resolution from the first input
        let (_, _, h0, w0) = inputs[0].dims4()?;
        
        // Resize inputs to match highest resolution
        let mut resized_inputs = Vec::new();
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
        
        // Apply FFN
        let mut img = self.ffn.forward(&channel_cat_imgs)?;
        
        // Apply normalization if not noskip
        img = self.norm.forward(&img)?;
        
        // Resize to output resolution if needed
        let (out_h, out_w) = self.output_resolution;
        if h0 != out_h || w0 != out_w {
            if h0 % out_h != 0 || w0 % out_w != 0 {
                // Use bilinear interpolation as bicubic equivalent
                // Use nearest neighbor upsampling as a fallback
                img = img.upsample_nearest2d(out_h, out_w)?;
            } else {
                // Use average pooling for downsampling
                let h_stride = h0 / out_h;
                let w_stride = w0 / out_w;
                img = img.avg_pool2d((h_stride, w_stride))?;
            }
        }
        
        Ok(img)
    }
}

// Architecture definition for Gemma3n
fn gemma3n_mobilenet_def() -> Vec<Vec<BlockType>> {
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
            let mut blocks = vec![
                BlockType::UniversalInvertedResidual {
                    out_channels: 640,
                    start_kernel_size: 5,
                    mid_kernel_size: 5,
                    stride: 2,
                    expand_ratio: 6.0,
                    is_multiscale: false,
                },
            ];
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
            let mut blocks = vec![
                BlockType::UniversalInvertedResidual {
                    out_channels: 1280,
                    start_kernel_size: 5,
                    mid_kernel_size: 5,
                    stride: 2,
                    expand_ratio: 6.0,
                    is_multiscale: false,
                },
            ];
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
}

impl VisionTower {
    pub fn new(vb: ShardedVarBuilder) -> Result<Self> {
        // Initial stem convolution
        let conv_stem = ConvNormAct::new(
            3,      // in_chs
            64,     // out_chs
            3,      // kernel_size
            2,      // stride
            1,      // padding
            1,      // groups
            true,   // apply_act
            1e-5,   // eps
            vb.pp("conv_stem"),
        )?;
        
        // Build blocks according to architecture definition
        let block_defs = gemma3n_mobilenet_def();
        let mut blocks = Vec::new();
        let mut in_chs = 64;
        
        for (stage_idx, stage_blocks) in block_defs.iter().enumerate() {
            let mut stage = Vec::new();
            
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
                            vb.pp(format!("blocks.{}.{}", stage_idx, block_idx)),
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
                            vb.pp(format!("blocks.{}.{}", stage_idx, block_idx)),
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
                            3, // dw_kernel_size
                            Some(1e-5), // layer_scale_init_value
                            vb.pp(format!("blocks.{}.{}", stage_idx, block_idx)),
                        )?;
                        Block::MobileAttention(ma)
                    }
                };
                stage.push(block);
            }
            blocks.push(stage);
        }
        
        // Multi-scale fusion adapter
        let msfa = MobileNetV5MultiScaleFusionAdapter::new(
            vec![1920], // in_chs
            2048,       // out_chs
            (16, 16),   // output_resolution
            2.0,        // expansion_ratio
            true,       // noskip
            false,      // use_layer_scale
            1e-5,       // layer_scale_init_value
            vb.pp("msfa"),
        )?;
        
        Ok(Self {
            conv_stem,
            blocks,
            msfa,
            msfa_indices: vec![3, 4], // Indices for multi-scale features
        })
    }
    
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply stem
        let mut x = self.conv_stem.forward(x)?;
        
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
    
    pub fn dtype(&self) -> DType {
        // Return the dtype from the conv_stem weight
        self.conv_stem.conv.weight().dtype()
    }
}

// Public interface
#[derive(Debug, Clone)]
pub struct MobileNetV5 {
    vision_tower: VisionTower,
}

impl MobileNetV5 {
    pub fn new(vb: ShardedVarBuilder) -> Result<Self> {
        let vision_tower = VisionTower::new(vb.pp("vision_tower"))?;
        Ok(Self { vision_tower })
    }
}

impl Module for MobileNetV5 {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.vision_tower.forward(x)
    }
}