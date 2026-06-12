#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{Result, Tensor};
use candle_nn::{
    conv1d, conv1d_no_bias, Activation, Conv1d, Conv1dConfig, Embedding, Linear, VarBuilder,
};
use mistralrs_quant::{linear_b, QuantMethod, QuantMethodConfig, QuantizedConfig, UnquantLinear};
use serde::Deserialize;

use crate::{
    layers::{GatedRmsNorm, RmsNorm},
    paged_attention::AttentionImplementation,
    pipeline::NormalLoadingMetadata,
    serde_default_fn,
};

serde_default_fn!(usize, num_heads_default, 128);
serde_default_fn!(usize, head_dim_default, 64);
serde_default_fn!(usize, vocab_size_default, 32768);
serde_default_fn!(usize, hidden_size_default, 4096);
serde_default_fn!(usize, state_size_default, 128);
serde_default_fn!(usize, num_hidden_layers_default, 64);
serde_default_fn!(f64, layer_norm_epsilon_default, 1e-5);
serde_default_fn!(f64, expand_default, 2.0);
serde_default_fn!(usize, conv_kernel_default, 4);
serde_default_fn!(usize, n_groups_default, 2);
serde_default_fn!(bool, use_bias_default, false);
serde_default_fn!(bool, use_conv_bias_default, true);
serde_default_fn!(Activation, hidden_act_default, Activation::Silu);
serde_default_fn!(bool, residual_in_fp32_default, true);
serde_default_fn!(f64, time_step_min_default, 0.001);
serde_default_fn!(f64, time_step_max_default, 0.1);
serde_default_fn!(f64, time_step_floor_default, 0.0001);
serde_default_fn!((f64, f64), time_step_limit_default, (0.0, f64::INFINITY));
serde_default_fn!(bool, rescale_prenorm_residual_default, false);
serde_default_fn!(bool, norm_before_gate_default, true);
serde_default_fn!(bool, rms_norm_default, true);
serde_default_fn!(usize, chunk_size_default, 256);

#[derive(Debug, Clone, Deserialize, Default)]
pub enum TimeStepRank {
    #[default]
    Auto,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Mamba2Config {
    #[serde(default = "num_heads_default")]
    num_heads: usize,
    #[serde(default = "head_dim_default")]
    head_dim: usize,
    #[serde(default = "vocab_size_default")]
    vocab_size: usize,
    #[serde(default = "hidden_size_default")]
    hidden_size: usize,
    #[serde(default = "state_size_default")]
    state_size: usize,
    #[serde(default = "num_hidden_layers_default")]
    num_hidden_layers: usize,
    #[serde(default = "layer_norm_epsilon_default")]
    layer_norm_epsilon: f64,
    #[serde(default = "expand_default")]
    expand: f64,
    #[serde(default = "conv_kernel_default")]
    conv_kernel: usize,
    #[serde(default = "n_groups_default")]
    n_groups: usize,
    #[serde(default = "use_bias_default")]
    use_bias: bool,
    #[serde(default = "use_conv_bias_default")]
    use_conv_bias: bool,
    #[serde(default = "hidden_act_default")]
    hidden_act: Activation,
    #[serde(default = "residual_in_fp32_default")]
    residual_in_fp32: bool,
    #[serde(default = "Default::default")]
    time_step_rank: TimeStepRank,
    #[serde(default = "time_step_min_default")]
    time_step_min: f64,
    #[serde(default = "time_step_max_default")]
    time_step_max: f64,
    #[serde(default = "time_step_floor_default")]
    time_step_floor: f64,
    #[serde(default = "time_step_limit_default")]
    time_step_limit: (f64, f64),
    #[serde(default = "rescale_prenorm_residual_default")]
    rescale_prenorm_residual: bool,
    #[serde(default = "norm_before_gate_default")]
    norm_before_gate: bool,
    #[serde(default = "rms_norm_default")]
    rms_norm: bool,
    #[serde(default = "chunk_size_default")]
    chunk_size: usize,
    quantization_config: Option<QuantizedConfig>,
}

// https://github.com/huggingface/transformers/blob/main/src/transformers/models/mamba2/modeling_mamba2.py#L406
struct Mixer {
    conv1d: Conv1d,
    in_proj: Arc<dyn QuantMethod>,
    dt_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
    norm: GatedRmsNorm,
    out_proj: Arc<dyn QuantMethod>,
}

impl Mixer {
    fn new(cfg: &Mamba2Config, vb: VarBuilder) -> Result<Self> {
        let intermediate_size = (cfg.expand * cfg.hidden_size as f64) as usize;
        let conv_dim = intermediate_size + 2 * cfg.n_groups * cfg.state_size;
        let projection_size = intermediate_size + conv_dim + cfg.num_heads;

        let conv1d_fn = if cfg.use_conv_bias {
            conv1d
        } else {
            conv1d_no_bias
        };

        let conv1d = conv1d_fn(
            conv_dim,
            conv_dim,
            cfg.conv_kernel,
            Conv1dConfig {
                padding: cfg.conv_kernel - 1,
                groups: conv_dim,
                stride: 1,
                dilation: 1,
            },
            vb.pp("conv1d"),
        )?;

        let in_proj = linear_b(
            cfg.hidden_size,
            projection_size,
            cfg.use_bias,
            &cfg.quantization_config,
            vb.pp("in_proj"),
        )?;

        let out_proj = linear_b(
            intermediate_size,
            cfg.hidden_size,
            cfg.use_bias,
            &cfg.quantization_config,
            vb.pp("out_proj"),
        )?;

        // Time step proj, discretization
        let dt_bias = vb.get((cfg.num_heads,), "dt_bias")?;

        // S4D real init, not discretized
        let a_log = vb.get((1, cfg.num_heads + 1), "A_log")?;
        let d = vb.get((cfg.num_heads,), "D")?;

        let norm = GatedRmsNorm::new(intermediate_size, cfg.layer_norm_epsilon, vb.pp("norm"))?;

        Ok(Self {
            conv1d,
            in_proj,
            out_proj,
            dt_bias,
            a_log,
            d,
            norm,
        })
    }
}

struct Layer {
    norm: RmsNorm,
    mixer: Mixer,
    res_in_f32: bool,
}

impl Layer {
    fn new(cfg: &Mamba2Config, vb: VarBuilder) -> Result<Self> {
        let norm = RmsNorm::new(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("norm"))?;
        let mixer = Mixer::new(cfg, vb.pp("mixer"))?;
        Ok(Self {
            norm,
            mixer,
            res_in_f32: cfg.residual_in_fp32,
        })
    }
}

pub struct Model {
    lm_head: Arc<dyn QuantMethod>,
    embeddings: Embedding,
    norm_f: RmsNorm,
    layers: Vec<Layer>,
}

impl Model {
    pub fn new(
        cfg: &Mamba2Config,
        vb: VarBuilder,
        _is_gptx: bool,
        _normal_loading_metadata: NormalLoadingMetadata,
        _attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        if let Some(ref quant_cfg) = &cfg.quantization_config {
            tracing::info!(
                "Using {} quantization in {} bits.",
                quant_cfg.quant_method.to_string(),
                quant_cfg.bits
            );
        }

        let vb = vb.pp("backbone");

        let embeddings =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embeddings"))?;
        // Tied to lm_head...
        let lm_head = Arc::new(UnquantLinear::new(QuantMethodConfig::Unquantized(
            Linear::new(embeddings.embeddings().clone(), None),
        ))?);
        let norm_f = RmsNorm::new(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("norm_f"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(Layer::new(cfg, vb.pp(idx))?);
        }

        Ok(Self {
            lm_head,
            embeddings,
            norm_f,
            layers,
        })
    }
}
