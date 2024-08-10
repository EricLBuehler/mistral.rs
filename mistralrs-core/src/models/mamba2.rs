use std::sync::Arc;

use candle_core::Tensor;
use candle_nn::{Activation, Conv1d, Embedding, Linear};
use mistralrs_quant::QuantMethod;
use serde::Deserialize;

use crate::{
    layers::{GatedRmsNorm, RmsNorm},
    serde_default_fn,
};

serde_default_fn!(usize, num_heads_default, 128);
serde_default_fn!(usize, head_dim_default, 64);
serde_default_fn!(usize, vocab_size_default, 32768);
serde_default_fn!(usize, hidden_size_default, 4096);
serde_default_fn!(usize, state_size_default, 128);
serde_default_fn!(usize, num_hidden_layers_default, 64);
serde_default_fn!(f64, layer_norm_epsilon_default, 1e-5);
serde_default_fn!(usize, expand_default, 2);
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
    expand: usize,
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

struct Layer {
    norm: RmsNorm,
    mixer: Mixer,
    res_in_f32: bool,
}

pub struct Model {
    lm_head: Linear,
    embeddings: Embedding,
    norm_f: RmsNorm,
    layers: Vec<Layer>,
}
