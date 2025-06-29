#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module};
use mistralrs_quant::ShardedVarBuilder;

use crate::{
    layers::{conv2d_no_bias, Sdpa},
    attention::SdpaParams,
};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Gemma3nVisionConfig {
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_attention_dropout")]
    pub attention_dropout: f32,
    #[serde(default = "default_residual_dropout")]
    pub residual_dropout: f32,
}

fn default_image_size() -> usize { 256 }
fn default_patch_size() -> usize { 16 }
fn default_num_channels() -> usize { 3 }
fn default_hidden_size() -> usize { 2048 }  // This should match text hidden size if no projector
fn default_intermediate_size() -> usize { 8192 }
fn default_num_hidden_layers() -> usize { 4 }
fn default_num_attention_heads() -> usize { 16 }
fn default_layer_norm_eps() -> f64 { 1e-6 }
fn default_attention_dropout() -> f32 { 0.0 }
fn default_residual_dropout() -> f32 { 0.0 }

impl Default for Gemma3nVisionConfig {
    fn default() -> Self {
        Self {
            image_size: default_image_size(),
            patch_size: default_patch_size(),
            num_channels: default_num_channels(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            num_key_value_heads: None,
            layer_norm_eps: default_layer_norm_eps(),
            attention_dropout: default_attention_dropout(),
            residual_dropout: default_residual_dropout(),
        }
    }
}

fn make_divisible(v: usize, divisor: usize, min_value: Option<usize>) -> usize {
    let min_value = min_value.unwrap_or(divisor);
    let new_v = std::cmp::max(min_value, ((v + divisor / 2) / divisor) * divisor);
    // Make sure that round down does not go down by more than 10%.
    if (new_v as f32) < 0.9 * (v as f32) {
        new_v + divisor
    } else {
        new_v
    }
}

fn num_groups(group_size: Option<usize>, channels: usize) -> usize {
    match group_size {
        None | Some(0) => 1,
        Some(size) => {
            assert_eq!(channels % size, 0, "channels must be divisible by group_size");
            channels / size
        }
    }
}

#[derive(Debug, Clone)]
struct ConvNormAct {
    conv: Conv2d,
    norm: RmsNormAct2d,
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
        let conv_config = Conv2dConfig {
            padding,
            stride,
            groups,
            dilation: 1,
        };
        
        let conv = conv2d_no_bias(
            in_chs,
            out_chs,
            kernel_size,
            conv_config,
            vb.pp("conv"),
        )?;
        
        let norm = RmsNormAct2d::new(out_chs, eps, apply_act, vb.pp("bn"))?;
        
        Ok(Self { conv, norm })
    }
}

impl Module for ConvNormAct {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(xs)?;
        self.norm.forward(&x)
    }
}

#[derive(Debug, Clone)]
struct RmsNormAct2d {
    weight: Tensor,
    eps: f64,
    apply_act: bool,
}

impl RmsNormAct2d {
    fn new(num_channels: usize, eps: f64, apply_act: bool, vb: ShardedVarBuilder) -> Result<Self> {
        let weight = vb.get(num_channels, "weight")?;
        Ok(Self { weight, eps, apply_act })
    }
}

impl Module for RmsNormAct2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Convert from NHWC to NCHW for normalization
        let x = xs.transpose(1, 3)?.transpose(2, 3)?;
        
        // RMS normalization over channel dimension
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(1)?;
        let x = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        
        // Apply weight
        let x = x.to_dtype(dtype)?;
        let weight = self.weight.reshape((1, (), 1, 1))?;
        let x = x.broadcast_mul(&weight)?;
        
        // Apply activation if needed
        let x = if self.apply_act {
            candle_nn::Activation::Gelu.forward(&x)?
        } else {
            x
        };
        
        // Convert back to NHWC
        x.transpose(2, 3)?.transpose(1, 3)
    }
}

#[derive(Debug, Clone)]
struct LayerScale2d {
    gamma: Tensor,
}

impl LayerScale2d {
    fn new(dim: usize, init_value: f32, vb: ShardedVarBuilder) -> Result<Self> {
        let gamma = (vb.get(dim, "gamma")? * init_value as f64)?;
        Ok(Self { gamma })
    }
}

impl Module for LayerScale2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.gamma)
    }
}

#[derive(Debug, Clone)]
struct UniversalInvertedResidual {
    has_skip: bool,
    dw_start: Option<ConvNormAct>,
    pw_exp: ConvNormAct,
    dw_mid: Option<ConvNormAct>,
    pw_proj: ConvNormAct,
    layer_scale: Option<LayerScale2d>,
}

impl UniversalInvertedResidual {
    fn new(
        in_chs: usize,
        out_chs: usize,
        dw_kernel_size_start: usize,
        dw_kernel_size_mid: usize,
        stride: usize,
        exp_ratio: f32,
        layer_scale_init_value: Option<f32>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let has_skip = in_chs == out_chs && stride == 1;
        let mid_chs = make_divisible((in_chs as f32 * exp_ratio) as usize, 8, None);
        
        let dw_start = if dw_kernel_size_start > 0 {
            let dw_start_stride = if dw_kernel_size_mid > 0 { 1 } else { stride };
            Some(ConvNormAct::new(
                in_chs,
                in_chs,
                dw_kernel_size_start,
                dw_start_stride,
                (dw_kernel_size_start - 1) / 2,
                in_chs, // depthwise
                false, // no activation
                1e-5,
                vb.pp("dw_start"),
            )?)
        } else {
            None
        };
        
        let pw_exp = ConvNormAct::new(
            in_chs,
            mid_chs,
            1,
            1,
            0,
            1,
            true, // with activation
            1e-5,
            vb.pp("pw_exp"),
        )?;
        
        let dw_mid = if dw_kernel_size_mid > 0 {
            Some(ConvNormAct::new(
                mid_chs,
                mid_chs,
                dw_kernel_size_mid,
                stride,
                (dw_kernel_size_mid - 1) / 2,
                mid_chs, // depthwise
                true, // with activation
                1e-5,
                vb.pp("dw_mid"),
            )?)
        } else {
            None
        };
        
        let pw_proj = ConvNormAct::new(
            mid_chs,
            out_chs,
            1,
            1,
            0,
            1,
            false, // no activation
            1e-5,
            vb.pp("pw_proj"),
        )?;
        
        let layer_scale = layer_scale_init_value.map(|v| LayerScale2d::new(out_chs, v, vb.pp("layer_scale")));
        let layer_scale = layer_scale.transpose()?;
        
        Ok(Self {
            has_skip,
            dw_start,
            pw_exp,
            dw_mid,
            pw_proj,
            layer_scale,
        })
    }
}

impl Module for UniversalInvertedResidual {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shortcut = xs.clone();
        
        let mut x = xs.clone();
        if let Some(dw_start) = &self.dw_start {
            x = dw_start.forward(&x)?;
        }
        
        x = self.pw_exp.forward(&x)?;
        
        if let Some(dw_mid) = &self.dw_mid {
            x = dw_mid.forward(&x)?;
        }
        
        x = self.pw_proj.forward(&x)?;
        
        if let Some(layer_scale) = &self.layer_scale {
            x = layer_scale.forward(&x)?;
        }
        
        if self.has_skip {
            x = (x + shortcut)?;
        }
        
        Ok(x)
    }
}

#[derive(Debug, Clone)]
struct EdgeResidual {
    has_skip: bool,
    conv_exp: Conv2d,
    bn1: RmsNormAct2d,
    conv_pwl: Conv2d,
    bn2: RmsNormAct2d,
}

impl EdgeResidual {
    fn new(
        in_chs: usize,
        out_chs: usize,
        exp_kernel_size: usize,
        stride: usize,
        expand_ratio: f32,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let has_skip = in_chs == out_chs && stride == 1;
        let mid_chs = make_divisible((in_chs as f32 * expand_ratio) as usize, 8, None);
        
        let padding = (exp_kernel_size - 1) / 2;
        let conv_exp = conv2d_no_bias(
            in_chs,
            mid_chs,
            exp_kernel_size,
            Conv2dConfig {
                padding,
                stride,
                groups: 1,
                dilation: 1,
            },
            vb.pp("conv_exp"),
        )?;
        
        let bn1 = RmsNormAct2d::new(mid_chs, 1e-5, true, vb.pp("bn1"))?;
        
        let conv_pwl = conv2d_no_bias(
            mid_chs,
            out_chs,
            1,
            Conv2dConfig::default(),
            vb.pp("conv_pwl"),
        )?;
        
        let bn2 = RmsNormAct2d::new(out_chs, 1e-5, false, vb.pp("bn2"))?;
        
        Ok(Self {
            has_skip,
            conv_exp,
            bn1,
            conv_pwl,
            bn2,
        })
    }
}

impl Module for EdgeResidual {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shortcut = xs.clone();
        
        let x = self.conv_exp.forward(xs)?;
        let x = self.bn1.forward(&x)?;
        let x = self.conv_pwl.forward(&x)?;
        let x = self.bn2.forward(&x)?;
        
        if self.has_skip {
            Ok((x + shortcut)?)
        } else {
            Ok(x)
        }
    }
}

#[derive(Debug, Clone)]
struct MultiQueryAttention2d {
    num_heads: usize,
    key_dim: usize,
    value_dim: usize,
    scale: f32,
    query_proj: Conv2d,
    key_proj: Conv2d,
    key_norm: Option<RmsNormAct2d>,
    value_proj: Conv2d,
    value_norm: Option<RmsNormAct2d>,
    output_proj: Conv2d,
}

impl MultiQueryAttention2d {
    fn new(
        dim: usize,
        dim_out: Option<usize>,
        num_heads: usize,
        key_dim: usize,
        value_dim: usize,
        kv_stride: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let dim_out = dim_out.unwrap_or(dim);
        let scale = (key_dim as f32).powf(-0.5);
        
        let query_proj = conv2d_no_bias(
            dim,
            num_heads * key_dim,
            1,
            Conv2dConfig::default(),
            vb.pp("query").pp("proj"),
        )?;
        
        let (key_proj, key_norm) = if kv_stride > 1 {
            // Implement downsampling for key
            let down_conv = conv2d_no_bias(
                dim,
                dim,
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: kv_stride,
                    groups: dim,
                    dilation: 1,
                },
                vb.pp("key").pp("down_conv"),
            )?;
            let norm = RmsNormAct2d::new(dim, 1e-6, false, vb.pp("key").pp("norm"))?;
            (down_conv, Some(norm))
        } else {
            let proj = conv2d_no_bias(
                dim,
                key_dim,
                1,
                Conv2dConfig::default(),
                vb.pp("key").pp("proj"),
            )?;
            (proj, None)
        };
        
        let (value_proj, value_norm) = if kv_stride > 1 {
            // Implement downsampling for value
            let down_conv = conv2d_no_bias(
                dim,
                dim,
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: kv_stride,
                    groups: dim,
                    dilation: 1,
                },
                vb.pp("value").pp("down_conv"),
            )?;
            let norm = RmsNormAct2d::new(dim, 1e-6, false, vb.pp("value").pp("norm"))?;
            (down_conv, Some(norm))
        } else {
            let proj = conv2d_no_bias(
                dim,
                value_dim,
                1,
                Conv2dConfig::default(),
                vb.pp("value").pp("proj"),
            )?;
            (proj, None)
        };
        
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
            key_proj,
            key_norm,
            value_proj,
            value_norm,
            output_proj,
        })
    }
}

impl Module for MultiQueryAttention2d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_sz, h, w, _) = xs.dims4()?;
        
        // Project queries
        let q = self.query_proj.forward(xs)?;
        let q = q.reshape((b_sz, h * w, self.num_heads, self.key_dim))?
            .transpose(1, 2)?; // [B, NH, L, D]
        
        // Project keys
        let mut k = self.key_proj.forward(xs)?;
        if let Some(norm) = &self.key_norm {
            k = norm.forward(&k)?;
        }
        let k = k.flatten(1, 2)?  // [B, H*W, C]
            .unsqueeze(1)?; // [B, 1, H*W, C]
        
        // Project values  
        let mut v = self.value_proj.forward(xs)?;
        if let Some(norm) = &self.value_norm {
            v = norm.forward(&v)?;
        }
        let v = v.flatten(1, 2)?  // [B, H*W, C]
            .unsqueeze(1)?; // [B, 1, H*W, C]
        
        // Attention
        let sdpa = Sdpa;
        let sdpa_params = SdpaParams {
            n_kv_groups: self.num_heads,
            softcap: None,
            softmax_scale: self.scale,
            sliding_window: None,
        };
        
        let attn = sdpa.run_attention(&q, &k, &v, None, None, &sdpa_params)?;
        
        // Reshape and project output
        let attn = attn.transpose(1, 2)?  // [B, L, NH, D]
            .reshape((b_sz, h, w, self.num_heads * self.value_dim))?;
        
        self.output_proj.forward(&attn)
    }
}

#[derive(Debug, Clone)]
struct MobileAttention {
    has_skip: bool,
    norm: RmsNormAct2d,
    attn: MultiQueryAttention2d,
    layer_scale: Option<LayerScale2d>,
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
        layer_scale_init_value: Option<f32>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let has_skip = stride == 1 && in_chs == out_chs;
        
        let norm = RmsNormAct2d::new(in_chs, 1e-5, false, vb.pp("norm"))?;
        
        let attn = MultiQueryAttention2d::new(
            in_chs,
            Some(out_chs),
            num_heads,
            key_dim,
            value_dim,
            kv_stride,
            vb.pp("attn"),
        )?;
        
        let layer_scale = layer_scale_init_value.map(|v| LayerScale2d::new(out_chs, v, vb.pp("layer_scale")));
        let layer_scale = layer_scale.transpose()?;
        
        Ok(Self {
            has_skip,
            norm,
            attn,
            layer_scale,
        })
    }
}

impl Module for MobileAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shortcut = xs.clone();
        
        let x = self.norm.forward(xs)?;
        let x = self.attn.forward(&x)?;
        
        let x = if let Some(layer_scale) = &self.layer_scale {
            layer_scale.forward(&x)?
        } else {
            x
        };
        
        if self.has_skip {
            Ok((x + shortcut)?)
        } else {
            Ok(x)
        }
    }
}

#[derive(Debug, Clone)]
struct MobileNetV5MultiScaleFusionAdapter {
    ffn: UniversalInvertedResidual,
    norm: RmsNormAct2d,
    output_resolution: (usize, usize),
}

impl MobileNetV5MultiScaleFusionAdapter {
    fn new(
        in_chs: Vec<usize>,
        out_chs: usize,
        output_resolution: (usize, usize),
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let in_channels = in_chs.iter().sum();
        
        let ffn = UniversalInvertedResidual::new(
            in_channels,
            out_chs,
            0, // no start dw
            0, // no mid dw
            1,
            2.0, // expansion ratio
            None, // no layer scale
            vb.pp("ffn"),
        )?;
        
        let norm = RmsNormAct2d::new(out_chs, 1e-6, false, vb.pp("norm"))?;
        
        Ok(Self {
            ffn,
            norm,
            output_resolution,
        })
    }
}

impl MobileNetV5MultiScaleFusionAdapter {
    fn forward(&self, inputs: &[Tensor]) -> Result<Tensor> {
        // Assuming inputs are in NHWC format
        let high_resolution = (inputs[0].dim(1)?, inputs[0].dim(2)?);
        
        // Resize all inputs to highest resolution
        let mut resized_inputs = Vec::new();
        for img in inputs {
            let (_, h, w, _) = img.dims4()?;
            if h < high_resolution.0 || w < high_resolution.1 {
                // Upsample using nearest neighbor
                let upsampled = img.transpose(1, 3)?.transpose(2, 3)? // Convert to NCHW
                    .upsample_nearest2d(high_resolution.0, high_resolution.1)?;
                resized_inputs.push(upsampled.transpose(2, 3)?.transpose(1, 3)?); // Back to NHWC
            } else {
                resized_inputs.push(img.clone());
            }
        }
        
        // Concatenate along channel dimension
        let cat_imgs = Tensor::cat(&resized_inputs, 3)?;
        
        // Apply FFN
        let x = self.ffn.forward(&cat_imgs)?;
        
        // Downsample if needed
        if high_resolution != self.output_resolution {
            let (h_out, w_out) = self.output_resolution;
            let h_stride = high_resolution.0 / h_out;
            let w_stride = high_resolution.1 / w_out;
            
            // Convert to NCHW for pooling
            let x = x.transpose(1, 3)?.transpose(2, 3)?;
            
            // Average pooling
            let x = x.avg_pool2d_with_stride((h_stride, w_stride), (h_stride, w_stride))?;
            
            // Back to NHWC
            let x = x.transpose(2, 3)?.transpose(1, 3)?;
            self.norm.forward(&x)
        } else {
            Ok(x)
        }
    }
}

enum BlockConfig {
    EdgeResidual {
        kernel_size: usize,
        filters: usize,
        strides: usize,
        expand_ratio: f32,
    },
    UniversalInvertedResidual {
        start_dw_kernel_size: usize,
        mid_dw_kernel_size: usize,
        filters: usize,
        strides: usize,
        expand_ratio: f32,
        is_multiscale: bool,
    },
    MultiQueryAttention {
        num_heads: usize,
        kv_dim: usize,
        kv_strides: usize,
    },
}

fn gemma3n_mobilenet_blocks() -> Vec<Vec<BlockConfig>> {
    use BlockConfig::*;
    
    vec![
        // Stage 1: Edge Residuals
        vec![
            EdgeResidual { kernel_size: 3, filters: 128, strides: 2, expand_ratio: 4.0 },
            EdgeResidual { kernel_size: 3, filters: 128, strides: 1, expand_ratio: 4.0 },
            EdgeResidual { kernel_size: 3, filters: 128, strides: 1, expand_ratio: 4.0 },
        ],
        // Stage 2: Universal Inverted Residuals
        vec![
            UniversalInvertedResidual { start_dw_kernel_size: 3, mid_dw_kernel_size: 5, filters: 256, strides: 2, expand_ratio: 6.0, is_multiscale: false },
            UniversalInvertedResidual { start_dw_kernel_size: 5, mid_dw_kernel_size: 0, filters: 256, strides: 1, expand_ratio: 4.0, is_multiscale: false },
            UniversalInvertedResidual { start_dw_kernel_size: 3, mid_dw_kernel_size: 0, filters: 256, strides: 1, expand_ratio: 4.0, is_multiscale: false },
            UniversalInvertedResidual { start_dw_kernel_size: 5, mid_dw_kernel_size: 0, filters: 256, strides: 1, expand_ratio: 4.0, is_multiscale: false },
            UniversalInvertedResidual { start_dw_kernel_size: 3, mid_dw_kernel_size: 0, filters: 256, strides: 1, expand_ratio: 4.0, is_multiscale: false },
        ],
        // Stage 3: Universal Inverted Residuals with Multi-Query Attention
        {
            let mut stage3 = vec![
                UniversalInvertedResidual { start_dw_kernel_size: 5, mid_dw_kernel_size: 5, filters: 640, strides: 2, expand_ratio: 6.0, is_multiscale: false },
            ];
            for _ in 0..7 {
                stage3.push(UniversalInvertedResidual { start_dw_kernel_size: 5, mid_dw_kernel_size: 0, filters: 640, strides: 1, expand_ratio: 4.0, is_multiscale: false });
            }
            stage3.push(UniversalInvertedResidual { start_dw_kernel_size: 0, mid_dw_kernel_size: 0, filters: 640, strides: 1, expand_ratio: 1.0, is_multiscale: false });
            
            for _ in 0..13 {
                stage3.push(MultiQueryAttention { num_heads: 12, kv_dim: 64, kv_strides: 2 });
                stage3.push(UniversalInvertedResidual { start_dw_kernel_size: 0, mid_dw_kernel_size: 0, filters: 640, strides: 1, expand_ratio: 2.0, is_multiscale: false });
            }
            stage3.push(MultiQueryAttention { num_heads: 12, kv_dim: 64, kv_strides: 2 });
            stage3.push(UniversalInvertedResidual { start_dw_kernel_size: 0, mid_dw_kernel_size: 0, filters: 640, strides: 1, expand_ratio: 2.0, is_multiscale: true });
            stage3
        },
        // Stage 4: Universal Inverted Residuals with Multi-Query Attention
        {
            let mut stage4 = vec![
                UniversalInvertedResidual { start_dw_kernel_size: 5, mid_dw_kernel_size: 5, filters: 1280, strides: 2, expand_ratio: 6.0, is_multiscale: false },
            ];
            for _ in 0..18 {
                stage4.push(MultiQueryAttention { num_heads: 16, kv_dim: 96, kv_strides: 1 });
                stage4.push(UniversalInvertedResidual { start_dw_kernel_size: 0, mid_dw_kernel_size: 0, filters: 1280, strides: 1, expand_ratio: 2.0, is_multiscale: false });
            }
            stage4.push(MultiQueryAttention { num_heads: 16, kv_dim: 96, kv_strides: 1 });
            stage4.push(UniversalInvertedResidual { start_dw_kernel_size: 0, mid_dw_kernel_size: 0, filters: 1280, strides: 1, expand_ratio: 2.0, is_multiscale: true });
            stage4
        },
    ]
}

enum VisionBlock {
    EdgeResidual(EdgeResidual),
    UniversalInvertedResidual(UniversalInvertedResidual),
    MobileAttention(MobileAttention),
}

impl Module for VisionBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            VisionBlock::EdgeResidual(b) => b.forward(xs),
            VisionBlock::UniversalInvertedResidual(b) => b.forward(xs),
            VisionBlock::MobileAttention(b) => b.forward(xs),
        }
    }
}

/// Gemma3n Vision Tower implementation based on MobileNetV5 architecture
/// 
/// Weight format: This implementation expects weights in PyTorch format:
/// - Conv2d weights: [out_channels, in_channels, kernel_height, kernel_width]  
/// - This matches the HuggingFace model format, no transposition needed
/// 
/// The MLX implementation transposes weights to [out_channels, kH, kW, in_channels],
/// but we use PyTorch format directly.
pub struct Gemma3nVisionTower {
    conv_stem: ConvNormAct,
    blocks: Vec<Vec<VisionBlock>>,
    msfa: MobileNetV5MultiScaleFusionAdapter,
    msfa_indices: Vec<usize>,
}

impl Gemma3nVisionTower {
    pub fn new(_config: &Gemma3nVisionConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let conv_stem = ConvNormAct::new(
            3,      // in_channels
            64,     // out_channels
            3,      // kernel_size
            2,      // stride
            1,      // padding
            1,      // groups
            true,   // apply_act
            1e-5,   // eps
            vb.pp("conv_stem"),
        )?;
        
        let msfa_indices = vec![2, 3];
        let msfa_output_resolution = (16, 16);
        
        let mut blocks = Vec::new();
        let mut in_chs = 64; // output of conv_stem
        
        for (stage_idx, stage_configs) in gemma3n_mobilenet_blocks().into_iter().enumerate() {
            let mut stage_blocks: Vec<VisionBlock> = Vec::new();
            
            for (block_idx, config) in stage_configs.into_iter().enumerate() {
                let block_vb = vb.pp(format!("blocks.{}.{}", stage_idx, block_idx));
                
                match config {
                    BlockConfig::EdgeResidual { kernel_size, filters, strides, expand_ratio } => {
                        let block = EdgeResidual::new(
                            in_chs,
                            filters,
                            kernel_size,
                            strides,
                            expand_ratio,
                            block_vb,
                        )?;
                        in_chs = filters;
                        stage_blocks.push(VisionBlock::EdgeResidual(block));
                    }
                    BlockConfig::UniversalInvertedResidual { 
                        start_dw_kernel_size, 
                        mid_dw_kernel_size, 
                        filters, 
                        strides, 
                        expand_ratio,
                        is_multiscale: _,
                    } => {
                        let block = UniversalInvertedResidual::new(
                            in_chs,
                            filters,
                            start_dw_kernel_size,
                            mid_dw_kernel_size,
                            strides,
                            expand_ratio,
                            Some(1e-5),
                            block_vb,
                        )?;
                        in_chs = filters;
                        stage_blocks.push(VisionBlock::UniversalInvertedResidual(block));
                    }
                    BlockConfig::MultiQueryAttention { num_heads, kv_dim, kv_strides } => {
                        let block = MobileAttention::new(
                            in_chs,
                            in_chs,
                            1,
                            num_heads,
                            kv_dim,
                            kv_dim,
                            kv_strides,
                            Some(1e-5),
                            block_vb,
                        )?;
                        stage_blocks.push(VisionBlock::MobileAttention(block));
                    }
                }
            }
            
            blocks.push(stage_blocks);
        }
        
        let msfa = MobileNetV5MultiScaleFusionAdapter::new(
            vec![640, 1280], // Channels from stages 2 and 3
            2048,            // output channels
            msfa_output_resolution,
            vb.pp("msfa"),
        )?;
        
        Ok(Self {
            conv_stem,
            blocks,
            msfa,
            msfa_indices,
        })
    }
    
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Convert from NCHW to NHWC for processing
        let x = pixel_values.transpose(1, 3)?.transpose(1, 2)?;
        
        let mut x = self.conv_stem.forward(&x)?;
        let mut intermediates = Vec::new();
        
        for (stage_idx, stage_blocks) in self.blocks.iter().enumerate() {
            for block in stage_blocks {
                x = block.forward(&x)?;
            }
            
            // Capture features after processing the stage
            if self.msfa_indices.contains(&stage_idx) {
                intermediates.push(x.clone());
            }
        }
        
        // Apply multi-scale fusion adapter
        self.msfa.forward(&intermediates)
    }
    
    pub fn dtype(&self) -> DType {
        self.conv_stem.conv.weight().dtype()
    }
}

impl Gemma3nVisionTower {
    pub fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        Vec::new() // Vision models typically don't have residual tensors for quantization
    }
}