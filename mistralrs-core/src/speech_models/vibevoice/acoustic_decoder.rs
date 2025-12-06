//! Acoustic tokenizer decoder (σ-VAE) for converting latent codes to audio waveforms.
//!
//! The decoder uses a multi-stage upsampling architecture with transformer blocks.
//! Upsampling ratios: [8, 5, 5, 4, 2, 2] = 3200x total, converting 7.5Hz latents to 24kHz audio.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Linear};
use mistralrs_quant::ShardedVarBuilder;

use super::config::AcousticTokenizerConfig;

/// Helper to create a linear layer with bias from ShardedVarBuilder
fn linear(in_dim: usize, out_dim: usize, vb: ShardedVarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

/// RMS Normalization for 1D convolutions (operates on channel dimension)
struct ConvRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl ConvRmsNorm {
    fn new(dim: usize, eps: f64, vb: ShardedVarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Input: (batch, channels, time)
        // Normalize over channel dimension
        let (b, c, t) = xs.dims3()?;

        // Transpose to (batch, time, channels) for normalization
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let xs = xs.reshape((b * t, c))?;

        // RMS normalization
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let xs = xs.broadcast_mul(&self.weight)?;

        // Reshape back to (batch, channels, time)
        xs.reshape((b, t, c))?.transpose(1, 2)?.contiguous()
    }
}

/// Causal 1D convolution with padding
struct CausalConv1d {
    conv: Conv1d,
    kernel_size: usize,
}

impl CausalConv1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        // Model has nested conv.conv structure
        let conv_vb = vb.pp("conv").pp("conv");
        let conv = Conv1d::new(
            conv_vb.get((out_channels, in_channels / groups, kernel_size), "weight")?,
            if bias {
                Some(conv_vb.get(out_channels, "bias")?)
            } else {
                None
            },
            Conv1dConfig {
                padding: 0,
                stride,
                dilation: 1,
                groups,
                ..Default::default()
            },
        );
        Ok(Self { conv, kernel_size })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Apply causal padding (left padding only)
        let padding = self.kernel_size - 1;
        let xs = xs.pad_with_zeros(2, padding, 0)?;
        self.conv.forward(&xs)
    }
}

/// Causal 1D transposed convolution (upsampling)
struct CausalConvTranspose1d {
    conv: ConvTranspose1d,
    stride: usize,
}

impl CausalConvTranspose1d {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        // Model has nested convtr.convtr structure
        let convtr_vb = vb.pp("convtr").pp("convtr");
        let conv = ConvTranspose1d::new(
            convtr_vb.get((in_channels, out_channels, kernel_size), "weight")?,
            if bias {
                Some(convtr_vb.get(out_channels, "bias")?)
            } else {
                None
            },
            ConvTranspose1dConfig {
                padding: 0,
                output_padding: 0,
                stride,
                dilation: 1,
                groups: 1,
            },
        );
        Ok(Self { conv, stride })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.conv.forward(xs)?;
        // Trim output to maintain causal property
        let trim = out.dim(2)? - (xs.dim(2)? * self.stride);
        if trim > 0 {
            out.narrow(2, 0, out.dim(2)? - trim)
        } else {
            Ok(out)
        }
    }
}

/// Depthwise causal 1D convolution mixer
struct DepthwiseConvMixer {
    conv: CausalConv1d,
}

impl DepthwiseConvMixer {
    fn new(dim: usize, kernel_size: usize, bias: bool, vb: ShardedVarBuilder) -> Result<Self> {
        let conv = CausalConv1d::new(
            dim,
            dim,
            kernel_size,
            1,
            dim, // groups = dim for depthwise
            bias,
            vb.pp("conv"),
        )?;
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs)
    }
}

/// Feed-forward network block
struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    fn new(dim: usize, hidden_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let linear1 = linear(dim, hidden_dim, vb.pp("linear1"))?;
        let linear2 = linear(hidden_dim, dim, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Input: (batch, channels, time)
        // Need to transpose for linear layers
        let (b, c, t) = xs.dims3()?;
        let xs = xs.transpose(1, 2)?.reshape((b * t, c))?;
        let xs = self.linear1.forward(&xs)?;
        let xs = xs.gelu()?;
        let xs = self.linear2.forward(&xs)?;
        xs.reshape((b, t, ()))?.transpose(1, 2)?.contiguous()
    }
}

/// Transformer block for the decoder stages
struct Block1d {
    norm: ConvRmsNorm,
    mixer: DepthwiseConvMixer,
    gamma: Option<Tensor>,
    ffn_norm: ConvRmsNorm,
    ffn: FeedForward,
    ffn_gamma: Option<Tensor>,
}

impl Block1d {
    fn new(
        dim: usize,
        kernel_size: usize,
        ffn_ratio: usize,
        eps: f64,
        layer_scale_init: f64,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let norm = ConvRmsNorm::new(dim, eps, vb.pp("norm"))?;
        let mixer = DepthwiseConvMixer::new(dim, kernel_size, bias, vb.pp("mixer"))?;

        let gamma = if layer_scale_init > 0.0 {
            Some(vb.get(dim, "gamma")?)
        } else {
            None
        };

        let ffn_norm = ConvRmsNorm::new(dim, eps, vb.pp("ffn_norm"))?;
        let ffn = FeedForward::new(dim, dim * ffn_ratio, vb.pp("ffn"))?;

        let ffn_gamma = if layer_scale_init > 0.0 {
            Some(vb.get(dim, "ffn_gamma")?)
        } else {
            None
        };

        Ok(Self {
            norm,
            mixer,
            gamma,
            ffn_norm,
            ffn,
            ffn_gamma,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Mixer block with residual
        let residual = xs;
        let mut xs = self.norm.forward(xs)?;
        xs = self.mixer.forward(&xs)?;
        if let Some(gamma) = &self.gamma {
            // gamma is (dim,), need to broadcast over (batch, dim, time)
            xs = xs.broadcast_mul(&gamma.unsqueeze(0)?.unsqueeze(D::Minus1)?)?;
        }
        let xs = (residual + xs)?;

        // FFN block with residual
        let residual = &xs;
        let mut ffn_out = self.ffn_norm.forward(&xs)?;
        ffn_out = self.ffn.forward(&ffn_out)?;
        if let Some(gamma) = &self.ffn_gamma {
            ffn_out = ffn_out.broadcast_mul(&gamma.unsqueeze(0)?.unsqueeze(D::Minus1)?)?;
        }
        residual + ffn_out
    }
}

/// Decoder stage with upsampling and transformer blocks
struct DecoderStage {
    upsample: Option<Box<dyn UpsampleLayer>>,
    blocks: Vec<Block1d>,
}

trait UpsampleLayer: Send + Sync {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

/// Initial stem layer (regular convolution)
struct StemUpsample {
    conv: CausalConv1d,
}

impl StemUpsample {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let conv = CausalConv1d::new(in_channels, out_channels, kernel_size, 1, 1, bias, vb)?;
        Ok(Self { conv })
    }
}

impl UpsampleLayer for StemUpsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs)
    }
}

/// Transposed convolution upsampling layer
struct TransposedUpsample {
    conv: CausalConvTranspose1d,
}

impl TransposedUpsample {
    fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        // kernel_size = 2 * stride for smooth upsampling
        let conv =
            CausalConvTranspose1d::new(in_channels, out_channels, 2 * stride, stride, bias, vb)?;
        Ok(Self { conv })
    }
}

impl UpsampleLayer for TransposedUpsample {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs)
    }
}

impl DecoderStage {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = if let Some(upsample) = &self.upsample {
            upsample.forward(xs)?
        } else {
            xs.clone()
        };

        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }

        Ok(xs)
    }
}

/// Output head - final convolution to produce audio
struct OutputHead {
    conv: CausalConv1d,
}

impl OutputHead {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let conv = CausalConv1d::new(in_channels, out_channels, kernel_size, 1, 1, true, vb)?;
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.conv.forward(xs)
    }
}

/// Acoustic Tokenizer Decoder (σ-VAE decoder)
///
/// Converts 64-dimensional latent codes at 7.5Hz to 24kHz audio waveforms.
/// Architecture:
/// - Stem: Conv1d(latent_dim, channels * 2^(stages-1))
/// - Stages: Each stage has upsampling + transformer blocks
/// - Head: Conv1d(channels, audio_channels)
pub struct AcousticDecoder {
    stages: Vec<DecoderStage>,
    head: OutputHead,
    fix_std: f32,
}

impl AcousticDecoder {
    pub fn new(cfg: &AcousticTokenizerConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let depths = cfg.get_depths();
        let ratios = &cfg.decoder_ratios;
        let n_filters = cfg.decoder_n_filters;
        let kernel_size = 7;
        let ffn_ratio = 4;

        // Decoder depths are reversed from encoder depths
        let decoder_depths: Vec<usize> = depths.iter().rev().cloned().collect();

        // Calculate channel dimensions for each stage (reverse of encoder)
        // Total stages = stem + upsampling stages = 1 + ratios.len()
        // Encoder: [n_filters, n_filters*2, n_filters*4, ..., n_filters*2^num_ratios]
        // Decoder: [n_filters*2^num_ratios, ..., n_filters*4, n_filters*2, n_filters]
        let num_ratios = ratios.len();

        let mut stages = Vec::new();

        // Stage 0: Stem (input projection from latent dim)
        // First stage channel dim is n_filters * 2^num_ratios
        let first_channel_dim = n_filters * (1 << num_ratios);
        {
            let upsample = StemUpsample::new(
                cfg.vae_dim,
                first_channel_dim,
                kernel_size,
                cfg.conv_bias,
                vb.pp("upsample_layers").pp(0).pp(0),
            )?;

            let mut blocks = Vec::new();
            for block_idx in 0..decoder_depths[0] {
                let block = Block1d::new(
                    first_channel_dim,
                    kernel_size,
                    ffn_ratio,
                    cfg.layernorm_eps,
                    cfg.layer_scale_init_value,
                    cfg.conv_bias,
                    vb.pp("stages").pp(0).pp(block_idx),
                )?;
                blocks.push(block);
            }

            stages.push(DecoderStage {
                upsample: Some(Box::new(upsample)),
                blocks,
            });
        }

        // Remaining stages with transposed convolution upsampling
        for (stage_idx, &ratio) in ratios.iter().enumerate() {
            let stage_num = stage_idx + 1;

            // Channel dimensions decrease as we upsample
            // in_channels = n_filters * 2^(num_ratios - stage_idx)
            // out_channels = n_filters * 2^(num_ratios - stage_idx - 1)
            let in_channels = n_filters * (1 << (num_ratios - stage_idx));
            let out_channels = n_filters * (1 << (num_ratios - stage_idx - 1));

            let upsample = TransposedUpsample::new(
                in_channels,
                out_channels,
                ratio,
                cfg.conv_bias,
                vb.pp("upsample_layers").pp(stage_num).pp(0),
            )?;

            let depth_idx = stage_num.min(decoder_depths.len() - 1);
            let mut blocks = Vec::new();
            for block_idx in 0..decoder_depths[depth_idx] {
                let block = Block1d::new(
                    out_channels,
                    kernel_size,
                    ffn_ratio,
                    cfg.layernorm_eps,
                    cfg.layer_scale_init_value,
                    cfg.conv_bias,
                    vb.pp("stages").pp(stage_num).pp(block_idx),
                )?;
                blocks.push(block);
            }

            stages.push(DecoderStage {
                upsample: Some(Box::new(upsample)),
                blocks,
            });
        }

        // Output head
        let head = OutputHead::new(n_filters, cfg.channels, kernel_size, vb.pp("head"))?;

        Ok(Self {
            stages,
            head,
            fix_std: cfg.fix_std,
        })
    }

    /// Decode latent codes to audio waveform
    ///
    /// Input: (batch, seq_len, latent_dim) - latent codes at 7.5Hz
    /// Output: (batch, audio_channels, audio_samples) - audio at 24kHz
    pub fn forward(&self, latents: &Tensor) -> Result<Tensor> {
        // Scale latents by fixed std
        let latents = (latents * self.fix_std as f64)?;

        // Transpose to (batch, latent_dim, seq_len) for conv layers
        let mut xs = latents.transpose(1, 2)?;

        // Pass through all stages
        for stage in &self.stages {
            xs = stage.forward(&xs)?;
        }

        // Output head
        self.head.forward(&xs)
    }

    /// Decode latent codes to audio waveform and squeeze to 1D if single channel
    pub fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let audio = self.forward(latents)?;
        // If single channel, squeeze the channel dimension
        if audio.dim(1)? == 1 {
            audio.squeeze(1)
        } else {
            Ok(audio)
        }
    }
}
