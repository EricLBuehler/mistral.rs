use candle_core::{DType, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Conv2d, Conv2dConfig, LayerNorm, Linear, Module};
use mistralrs_quant::{lfm3_causal_conv1d, ShardedVarBuilder};
use serde::{Deserialize, Serialize};

use crate::{
    attention::{AttentionMask, Sdpa, SdpaParams},
    layers::{conv1d, conv2d, layer_norm, linear, linear_b, linear_no_bias, RmsNorm},
    pipeline::text_models_inputs_processor::FlashParams,
    utils::unvarbuilder::UnVarBuilder,
};

use super::audio_processing::ENCODER_STRIDE;

const DOWNSAMPLE_KERNEL: usize = 3;
const DOWNSAMPLE_STRIDE: usize = 2;
const DOWNSAMPLE_PADDING: usize = 1;
const PROJECTION_PARTS: usize = 3;
const PROJECTOR_NORM_EPS: f64 = 1e-5;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct AudioEncoderConfig {
    pub conv_channels: usize,
    pub hidden_size: usize,
    pub num_mel_bins: usize,
    pub sliding_window: usize,
    pub text_config: AudioTextConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct AudioTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub norm_eps: f64,
    pub rope_parameters: crate::models::lfm2::RopeParameters,
    pub conv_bias: bool,
    #[serde(rename = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub block_multiple_of: usize,
    pub block_ffn_dim_multiplier: Option<f64>,
    pub block_auto_adjust_ff_dim: bool,
    pub layer_types: Vec<String>,
}

impl AudioTextConfig {
    pub(crate) fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub(crate) fn intermediate_size(&self) -> usize {
        if !self.block_auto_adjust_ff_dim {
            return self.intermediate_size;
        }
        let size = 2 * self.intermediate_size / 3;
        match self.block_ffn_dim_multiplier {
            Some(multiplier) => {
                #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                let size = (size as f64 * multiplier) as usize;
                size.div_ceil(self.block_multiple_of) * self.block_multiple_of
            }
            None => size,
        }
    }
}

impl AudioEncoderConfig {
    pub(crate) fn parameter_count(&self) -> usize {
        let channels = self.conv_channels;
        let hidden = self.hidden_size;
        let mut count = 3 * (channels * DOWNSAMPLE_KERNEL.pow(2) + channels)
            + 2 * (channels * channels + channels)
            + hidden * (self.num_mel_bins * channels / ENCODER_STRIDE)
            + hidden;
        let config = &self.text_config;
        let hidden = config.hidden_size;
        count += hidden;
        for layer_type in &config.layer_types {
            count += 2 * hidden + 3 * hidden * config.intermediate_size();
            count += if layer_type == "full_attention" {
                2 * hidden * hidden
                    + 2 * config.num_key_value_heads * config.head_dim() * hidden
                    + 2 * config.head_dim()
            } else {
                (PROJECTION_PARTS + 1) * hidden * hidden
                    + hidden * config.conv_l_cache
                    + if config.conv_bias { 5 * hidden } else { 0 }
            };
        }
        count
    }
}

struct Downsampler {
    conv_in: Conv2d,
    conv_mid: Conv2d,
    lin_mid: Conv2d,
    conv_end: Conv2d,
    lin_end: Conv2d,
    lin_out: Conv1d,
}

fn mask_padding(xs: &Tensor, lengths: &[usize]) -> Result<Tensor> {
    let (batch, _, _, time) = xs.dims4()?;
    let mask = lengths
        .iter()
        .flat_map(|&length| {
            (0..time).map(move |index| if index < length { 1.0f32 } else { 0.0f32 })
        })
        .collect::<Vec<_>>();
    xs.broadcast_mul(
        &Tensor::from_vec(mask, (batch, 1, 1, time), xs.device())?.to_dtype(xs.dtype())?,
    )
}

impl Downsampler {
    fn new(config: &AudioEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let channels = config.conv_channels;
        let convolution = Conv2dConfig {
            padding: DOWNSAMPLE_PADDING,
            stride: DOWNSAMPLE_STRIDE,
            ..Default::default()
        };
        let depthwise = Conv2dConfig {
            groups: channels,
            ..convolution
        };
        Ok(Self {
            conv_in: conv2d(
                1,
                channels,
                DOWNSAMPLE_KERNEL,
                convolution,
                vb.pp("conv_in"),
            )?,
            conv_mid: conv2d(
                channels,
                channels,
                DOWNSAMPLE_KERNEL,
                depthwise,
                vb.pp("conv_mid"),
            )?,
            lin_mid: conv2d(
                channels,
                channels,
                1,
                Conv2dConfig::default(),
                vb.pp("lin_mid"),
            )?,
            conv_end: conv2d(
                channels,
                channels,
                DOWNSAMPLE_KERNEL,
                depthwise,
                vb.pp("conv_end"),
            )?,
            lin_end: conv2d(
                channels,
                channels,
                1,
                Conv2dConfig::default(),
                vb.pp("lin_end"),
            )?,
            lin_out: conv1d(
                config.num_mel_bins * channels / ENCODER_STRIDE,
                config.hidden_size,
                1,
                Conv1dConfig::default(),
                vb.pp("lin_out"),
            )?,
        })
    }

    fn forward(&self, features: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        let mut lengths = lengths.to_vec();
        let features = features.to_dtype(self.conv_in.weight().dtype())?;
        let xs = mask_padding(&features.unsqueeze(1)?, &lengths)?;
        let xs = self.conv_in.forward(&xs)?.silu()?;
        lengths
            .iter_mut()
            .for_each(|length| *length = length.div_ceil(DOWNSAMPLE_STRIDE));
        let xs = mask_padding(&xs, &lengths)?;
        let xs = self.lin_mid.forward(&self.conv_mid.forward(&xs)?)?.silu()?;
        lengths
            .iter_mut()
            .for_each(|length| *length = length.div_ceil(DOWNSAMPLE_STRIDE));
        let xs = mask_padding(&xs, &lengths)?;
        let xs = self.lin_end.forward(&self.conv_end.forward(&xs)?)?.silu()?;
        self.lin_out.forward(&xs.flatten(1, 2)?)?.transpose(1, 2)
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        uvb.pp("conv_in").add(&self.conv_in);
        uvb.pp("conv_mid").add(&self.conv_mid);
        uvb.pp("lin_mid").add(&self.lin_mid);
        uvb.pp("conv_end").add(&self.conv_end);
        uvb.pp("lin_end").add(&self.lin_end);
        uvb.pp("lin_out").add(&self.lin_out);
    }
}

struct AudioAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    q_layernorm: RmsNorm,
    k_layernorm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
}

impl AudioAttention {
    fn new(config: &AudioTextConfig, window: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;
        let head_dim = config.head_dim();
        let kv_size = config.num_key_value_heads * head_dim;
        Ok(Self {
            q_proj: linear_no_bias(hidden, hidden, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(hidden, kv_size, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(hidden, kv_size, vb.pp("v_proj"))?,
            out_proj: linear_no_bias(hidden, hidden, vb.pp("out_proj"))?,
            q_layernorm: RmsNorm::new(head_dim, config.norm_eps, vb.pp("q_layernorm"))?,
            k_layernorm: RmsNorm::new(head_dim, config.norm_eps, vb.pp("k_layernorm"))?,
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim,
            window,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (batch, time, hidden) = xs.dims3()?;
        let query = self
            .q_layernorm
            .forward(&self.q_proj.forward(xs)?.reshape((
                batch,
                time,
                self.heads,
                self.head_dim,
            ))?)?
            .transpose(1, 2)?;
        let key = self
            .k_layernorm
            .forward(&self.k_proj.forward(xs)?.reshape((
                batch,
                time,
                self.kv_heads,
                self.head_dim,
            ))?)?
            .transpose(1, 2)?;
        let value = self
            .v_proj
            .forward(xs)?
            .reshape((batch, time, self.kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let (query, key) = mistralrs_quant::rotary::apply_rotary_qk_preselected(
            &query.to_dtype(DType::F32)?,
            &key.to_dtype(DType::F32)?,
            cos,
            sin,
            true,
        )?;
        #[allow(clippy::cast_precision_loss)]
        let softmax_scale = 1.0 / (self.head_dim as f32).sqrt();
        let output = Sdpa.run_attention(
            &query.to_dtype(xs.dtype())?.contiguous()?,
            &key.to_dtype(xs.dtype())?.contiguous()?,
            &value,
            &AttentionMask::CausalFlash,
            Some(&FlashParams::empty(true)),
            &SdpaParams {
                n_kv_groups: self.heads / self.kv_heads,
                sliding_window: Some(self.window),
                softcap: None,
                softmax_scale,
                sinks: None,
            },
        )?;
        self.out_proj
            .forward(&output.transpose(1, 2)?.reshape((batch, time, hidden))?)
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        uvb.pp("q_proj").add(&self.q_proj);
        uvb.pp("k_proj").add(&self.k_proj);
        uvb.pp("v_proj").add(&self.v_proj);
        uvb.pp("out_proj").add(&self.out_proj);
        uvb.pp("q_layernorm").add(&self.q_layernorm);
        uvb.pp("k_layernorm").add(&self.k_layernorm);
    }
}

struct AudioConv {
    in_proj: Linear,
    out_proj: Linear,
    weight: Tensor,
    bias: Option<Tensor>,
}

impl AudioConv {
    fn new(config: &AudioTextConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;
        Ok(Self {
            in_proj: linear_b(
                hidden,
                PROJECTION_PARTS * hidden,
                config.conv_bias,
                vb.pp("in_proj"),
            )?,
            out_proj: linear_b(hidden, hidden, config.conv_bias, vb.pp("out_proj"))?,
            weight: vb
                .pp("conv")
                .get((hidden, 1, config.conv_l_cache), "weight")?,
            bias: if config.conv_bias {
                Some(vb.pp("conv").get(hidden, "bias")?)
            } else {
                None
            },
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, _, hidden) = xs.dims3()?;
        let state = Tensor::zeros(
            (batch, hidden, self.weight.dim(2)?),
            xs.dtype(),
            xs.device(),
        )?;
        let (output, _) = lfm3_causal_conv1d(
            &self.in_proj.forward(xs)?,
            &self.weight,
            self.bias.as_ref(),
            &state,
            false,
        )?;
        self.out_proj.forward(&output)
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        uvb.pp("in_proj").add(&self.in_proj);
        uvb.pp("out_proj").add(&self.out_proj);
        uvb.pp("conv").add_tensor("weight", self.weight.clone());
        if let Some(bias) = &self.bias {
            uvb.pp("conv").add_tensor("bias", bias.clone());
        }
    }
}

enum AudioOperator {
    Attention(AudioAttention),
    Conv(AudioConv),
}

struct AudioLayer {
    operator: AudioOperator,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl AudioLayer {
    fn new(config: &AudioEncoderConfig, index: usize, vb: ShardedVarBuilder) -> Result<Self> {
        let text = &config.text_config;
        let hidden = text.hidden_size;
        let intermediate = text.intermediate_size();
        let operator = match text.layer_types[index].as_str() {
            "full_attention" => AudioOperator::Attention(AudioAttention::new(
                text,
                config.sliding_window,
                vb.pp("self_attn"),
            )?),
            "conv" => AudioOperator::Conv(AudioConv::new(text, vb.pp("conv"))?),
            other => candle_core::bail!("Unsupported LFM3 audio encoder layer type {other}"),
        };
        Ok(Self {
            operator,
            operator_norm: RmsNorm::new(hidden, text.norm_eps, vb.pp("operator_norm"))?,
            ffn_norm: RmsNorm::new(hidden, text.norm_eps, vb.pp("ffn_norm"))?,
            w1: linear_no_bias(hidden, intermediate, vb.pp("feed_forward.w1"))?,
            w2: linear_no_bias(intermediate, hidden, vb.pp("feed_forward.w2"))?,
            w3: linear_no_bias(hidden, intermediate, vb.pp("feed_forward.w3"))?,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let normalized = self.operator_norm.forward(xs)?;
        let output = match &self.operator {
            AudioOperator::Attention(attention) => attention.forward(&normalized, cos, sin)?,
            AudioOperator::Conv(conv) => conv.forward(&normalized)?,
        };
        let xs = (xs + output)?;
        let normalized = self.ffn_norm.forward(&xs)?;
        let feed_forward = self
            .w2
            .forward(&(self.w1.forward(&normalized)?.silu()? * self.w3.forward(&normalized)?)?)?;
        xs + feed_forward
    }

    fn residual_tensors(&self, uvb: UnVarBuilder) {
        match &self.operator {
            AudioOperator::Attention(attention) => attention.residual_tensors(uvb.pp("self_attn")),
            AudioOperator::Conv(conv) => conv.residual_tensors(uvb.pp("conv")),
        }
        uvb.pp("operator_norm").add(&self.operator_norm);
        uvb.pp("ffn_norm").add(&self.ffn_norm);
        uvb.pp("feed_forward.w1").add(&self.w1);
        uvb.pp("feed_forward.w2").add(&self.w2);
        uvb.pp("feed_forward.w3").add(&self.w3);
    }
}

pub(crate) struct AudioEncoder {
    downsample: Downsampler,
    layers: Vec<AudioLayer>,
    norm: RmsNorm,
    head_dim: usize,
    rope_theta: f64,
}

impl AudioEncoder {
    pub(crate) fn new(config: &AudioEncoderConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let text = &config.text_config;
        Ok(Self {
            downsample: Downsampler::new(config, vb.pp("downsample"))?,
            layers: (0..text.num_hidden_layers)
                .map(|index| AudioLayer::new(config, index, vb.pp("lfm.layers").pp(index)))
                .collect::<Result<_>>()?,
            norm: RmsNorm::new(text.hidden_size, text.norm_eps, vb.pp("lfm.embedding_norm"))?,
            head_dim: text.head_dim(),
            rope_theta: text.rope_parameters.rope_theta,
        })
    }

    pub(crate) fn forward(&self, features: &Tensor, lengths: &[usize]) -> Result<Tensor> {
        let mut hidden = self.downsample.forward(features, lengths)?;
        let time = hidden.dim(1)?;
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let frequencies = (0..time)
            .flat_map(|position| {
                (0..self.head_dim).step_by(2).map(move |index| {
                    position as f32
                        / self.rope_theta.powf(index as f64 / self.head_dim as f64) as f32
                })
            })
            .collect::<Vec<_>>();
        let frequencies =
            Tensor::from_vec(frequencies, (time, self.head_dim / 2), hidden.device())?;
        let cos = frequencies.cos()?;
        let sin = frequencies.sin()?;
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin)?;
        }
        self.norm.forward(&hidden)
    }

    pub(crate) fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        self.downsample.residual_tensors(uvb.pp("downsample"));
        for (index, layer) in self.layers.iter().enumerate() {
            layer.residual_tensors(uvb.pp("lfm.layers").pp(index));
        }
        uvb.pp("lfm.embedding_norm").add(&self.norm);
        uvb.to_safetensors()
    }
}

pub(crate) struct AudioProjector {
    norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
}

impl AudioProjector {
    pub(crate) fn new(
        input_size: usize,
        output_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(input_size, PROJECTOR_NORM_EPS, vb.pp("layer_norm"))?,
            linear_1: linear(input_size, output_size, vb.pp("linear_1"))?,
            linear_2: linear(output_size, output_size, vb.pp("linear_2"))?,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.linear_2
            .forward(&self.linear_1.forward(&self.norm.forward(xs)?)?.gelu_erf()?)
    }

    pub(crate) fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        let uvb = UnVarBuilder::new();
        uvb.pp("layer_norm").add(&self.norm);
        uvb.pp("linear_1").add(&self.linear_1);
        uvb.pp("linear_2").add(&self.linear_2);
        uvb.to_safetensors()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::Device;
    use mistralrs_quant::ShardedSafeTensors;

    use super::*;

    fn tiny_encoder() -> Result<(AudioEncoderConfig, HashMap<String, Tensor>)> {
        let config: AudioEncoderConfig = serde_json::from_value(serde_json::json!({
            "conv_channels": 2, "hidden_size": 8, "num_mel_bins": 8, "sliding_window": 2,
            "text_config": {
                "hidden_size": 8, "intermediate_size": 12, "num_hidden_layers": 3,
                "num_attention_heads": 2, "num_key_value_heads": 1, "norm_eps": 1e-5,
                "rope_parameters": {"rope_theta": 1000000.0}, "conv_bias": false,
                "conv_L_cache": 3, "block_multiple_of": 4, "block_ffn_dim_multiplier": 1.0,
                "block_auto_adjust_ff_dim": false, "layer_types": ["conv", "full_attention", "conv"]
            }
        }))
        .map_err(candle_core::Error::wrap)?;
        let mut shapes = Vec::<(String, Vec<usize>)>::new();
        for (name, input, output, kernel) in [
            ("conv_in", 1, 2, 3),
            ("conv_mid", 1, 2, 3),
            ("lin_mid", 2, 2, 1),
            ("conv_end", 1, 2, 3),
            ("lin_end", 2, 2, 1),
        ] {
            shapes.push((
                format!("downsample.{name}.weight"),
                vec![output, input, kernel, kernel],
            ));
            shapes.push((format!("downsample.{name}.bias"), vec![output]));
        }
        shapes.push(("downsample.lin_out.weight".into(), vec![8, 2, 1]));
        shapes.push(("downsample.lin_out.bias".into(), vec![8]));
        shapes.push(("lfm.embedding_norm.weight".into(), vec![8]));
        for layer in 0..3 {
            let prefix = format!("lfm.layers.{layer}");
            for norm in ["operator_norm", "ffn_norm"] {
                shapes.push((format!("{prefix}.{norm}.weight"), vec![8]));
            }
            for (name, shape) in [
                ("w1", vec![12, 8]),
                ("w2", vec![8, 12]),
                ("w3", vec![12, 8]),
            ] {
                shapes.push((format!("{prefix}.feed_forward.{name}.weight"), shape));
            }
            if layer == 1 {
                for (name, shape) in [
                    ("q_proj", vec![8, 8]),
                    ("k_proj", vec![4, 8]),
                    ("v_proj", vec![4, 8]),
                    ("out_proj", vec![8, 8]),
                    ("q_layernorm", vec![4]),
                    ("k_layernorm", vec![4]),
                ] {
                    shapes.push((format!("{prefix}.self_attn.{name}.weight"), shape));
                }
            } else {
                for (name, shape) in [
                    ("in_proj", vec![24, 8]),
                    ("out_proj", vec![8, 8]),
                    ("conv", vec![8, 1, 3]),
                ] {
                    shapes.push((format!("{prefix}.conv.{name}.weight"), shape));
                }
            }
        }
        let weights = shapes
            .into_iter()
            .map(|(name, shape)| {
                let values = (0..shape.iter().product())
                    .map(|index| {
                        #[allow(clippy::cast_precision_loss)]
                        let value = ((index % 17) as f32 - 8.0) * 0.02;
                        if name.contains("norm.weight") {
                            value + 1.0
                        } else {
                            value
                        }
                    })
                    .collect::<Vec<_>>();
                Ok((name, Tensor::from_vec(values, shape, &Device::Cpu)?))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok((config, weights))
    }

    #[test]
    fn encoder_matches_torch_with_padding_and_sliding_attention() -> Result<()> {
        let (config, weights) = tiny_encoder()?;
        assert_eq!(config.parameter_count(), 1776);
        assert_eq!(
            weights.values().map(Tensor::elem_count).sum::<usize>(),
            config.parameter_count()
        );
        let encoder = AudioEncoder::new(
            &config,
            ShardedSafeTensors::wrap(weights, DType::F32, Device::Cpu),
        )?;
        let features = (0u16..2 * 8 * 25)
            .map(|index| (f32::from(index % 29) - 14.0) * 0.03)
            .collect::<Vec<_>>();
        let features = Tensor::from_vec(features, (2, 8, 25), &Device::Cpu)?;
        let actual = encoder
            .forward(&features, &[25, 9])?
            .flatten_all()?
            .to_vec1::<f32>()?;
        // Torch reference includes padded mel frames, grouped convolutions, RoPE, and a two-token attention window.
        let expected = [
            -2.1035428,
            -0.22359908,
            -0.3348824,
            -0.92683053,
            -0.06432849,
            -0.5565901,
            -0.23636098,
            0.18790962,
            -2.0994163,
            -0.22005293,
            -0.35688516,
            -0.92834437,
            -0.07152019,
            -0.5530484,
            -0.24464183,
            0.19431165,
            -2.0946853,
            -0.21479097,
            -0.3823968,
            -0.9296684,
            -0.080650635,
            -0.54813546,
            -0.25482303,
            0.20105894,
            -2.0949192,
            -0.21495602,
            -0.38371792,
            -0.9285674,
            -0.082611635,
            -0.54802287,
            -0.25295007,
            0.20232214,
            -2.103538,
            -0.22358926,
            -0.3348615,
            -0.92685294,
            -0.06432034,
            -0.55659276,
            -0.236376,
            0.18788296,
            -2.0994601,
            -0.22065242,
            -0.35739145,
            -0.9278192,
            -0.071668155,
            -0.5531442,
            -0.24392547,
            0.19518612,
            -2.0947654,
            -0.21419334,
            -0.38230175,
            -0.929721,
            -0.08069349,
            -0.5478939,
            -0.2554154,
            0.20059992,
            -2.094806,
            -0.21416229,
            -0.38271523,
            -0.9295621,
            -0.082268886,
            -0.54799366,
            -0.25394785,
            0.20087834,
        ];
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 3e-5,
                "index {index}: {actual} != {expected}"
            );
        }
        assert_eq!(
            encoder
                .residual_tensors()
                .iter()
                .map(|(_, tensor)| tensor.elem_count())
                .sum::<usize>(),
            config.parameter_count()
        );
        Ok(())
    }
}
