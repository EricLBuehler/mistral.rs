use std::{collections::HashSet, fmt::Debug, sync::Arc};

use candle_core::{quantized::QTensor, IndexOp, Result, Shape, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
use loralinear::LoraLinear;
pub use qloralinear::QLoraLinear;
use serde::Deserialize;

mod frozenlinear;
mod loralinear;
mod qloralinear;

use std::collections::HashMap;

#[derive(Clone, Debug, Deserialize)]
pub struct Ordering {
    #[serde(rename = "order")]
    pub adapters: Option<Vec<String>>,
    pub layers: HashMap<String, usize>,
}

#[derive(Clone, Debug)]
/// Configuration for LoraLinear
pub struct LoraLinearConfig {
    in_features: usize,
    out_features: usize,
}

impl LoraLinearConfig {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        LoraLinearConfig {
            in_features,
            out_features,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct LoraConfig {
    #[serde(rename = "r")]
    rank: usize,
    #[serde(rename = "lora_alpha")]
    alpha: f64,
    #[serde(rename = "lora_dropout")]
    dropout: Option<f32>,
    target_modules: HashSet<String>,
}

fn apply_scalings_to_x(x: Tensor, scalings_layer: &Tensor, adapter: usize) -> Result<Tensor> {
    let scalings = scalings_layer.i((.., .., adapter))?.unsqueeze(D::Minus1)?;
    let res = x.broadcast_mul(&scalings)?;
    Ok(res)
}

impl LoraConfig {
    pub const fn new(
        rank: usize,
        alpha: f64,
        dropout: Option<f32>,
        target_modules: HashSet<String>,
    ) -> Self {
        Self {
            rank,
            alpha,
            dropout,
            target_modules,
        }
    }
}

/// Any layer that is linear-like.
pub trait LinearLayerLike: Debug {
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn shape(&self) -> &Shape;
    fn lora_forward(
        &self,
        x: &Tensor,
        scalings_layer: Tensor,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor>;
}

impl LinearLayerLike for Linear {
    fn weight(&self) -> &Tensor {
        self.weight()
    }
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
    fn shape(&self) -> &Shape {
        self.weight().shape()
    }
    fn lora_forward(
        &self,
        x: &Tensor,
        _scalings_layer: Tensor,
        _global_scaling_weight: f64,
        _is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        self.forward(x)
    }
}

pub fn linear(
    d1: usize,
    d2: usize,
    vb: VarBuilder,
    lora_config: &Vec<(String, LoraConfig)>,
    count: &mut usize,
    ord: &Ordering,
) -> Result<Arc<dyn LinearLayerLike + Send + Sync>> {
    let prefix = vb.prefix();
    let module = prefix.split('.').last().unwrap();

    let linear_config = LoraLinearConfig::new(d1, d2);
    let inner = candle_nn::linear(d1, d2, vb.clone())?;

    let target_modules = &lora_config[0].1.target_modules;
    for (_, cfg) in lora_config {
        if &cfg.target_modules != target_modules {
            candle_core::bail!("Expected all target modules to be the same.");
        }
    }
    if !target_modules.contains(module) {
        return Ok(Arc::new(inner));
    }
    let name = prefix.split("lora_A").last().unwrap();
    let layer = ord.layers.get(name).unwrap();

    let lorainner = LoraLinear::new(&inner, &linear_config, lora_config, &vb, *layer)?;
    *count += 1;
    Ok(Arc::new(lorainner))
}

pub fn linear_no_bias(
    d1: usize,
    d2: usize,
    vb: VarBuilder,
    lora_config: &Vec<(String, LoraConfig)>,
    count: &mut usize,
    ord: &Ordering,
) -> Result<Arc<dyn LinearLayerLike + Send + Sync>> {
    let prefix = vb.prefix();
    let module = prefix.split('.').last().unwrap();

    let linear_config = LoraLinearConfig::new(d1, d2);
    let inner = candle_nn::linear_no_bias(d1, d2, vb.clone())?;

    let target_modules = &lora_config[0].1.target_modules;
    for (_, cfg) in lora_config {
        if &cfg.target_modules != target_modules {
            candle_core::bail!("Expected all target modules to be the same.");
        }
    }
    if !target_modules.contains(module) {
        return Ok(Arc::new(inner));
    }
    let name = prefix.split("lora_A").last().unwrap();
    let layer = ord.layers.get(name).unwrap();

    let lorainner = LoraLinear::new(&inner, &linear_config, lora_config, &vb, *layer)?;
    *count += 1;
    Ok(Arc::new(lorainner))
}

fn get_maybe_topk_scalings(scalings: Tensor, layer: usize) -> Result<Tensor> {
    scalings.i((.., .., layer, ..))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: crate::VarBuilder,
    lora_config: &Vec<(String, LoraConfig)>,
    count: &mut usize,
    ord: &Ordering,
) -> Result<Arc<dyn LinearLayerLike + Send + Sync>> {
    if bias {
        linear(in_dim, out_dim, vb, lora_config, count, ord)
    } else {
        linear_no_bias(in_dim, out_dim, vb, lora_config, count, ord)
    }
}

pub fn get_lora_cfg(tensor: &QTensor) -> LoraLinearConfig {
    LoraLinearConfig::new(tensor.shape().dims()[1], tensor.shape().dims()[0])
}

/// Batch matrix multiplication.
///
/// input = `[n_adapters, bs*seqlen, in]`
///
/// a = `[n_adapters, out, in]`
fn bmm(input: &Tensor, a: &Tensor) -> Result<Tensor> {
    let input_reshaped = input.t()?;
    let a_reshaped = a.t()?;

    let res = a_reshaped.broadcast_matmul(&input_reshaped)?;

    res.reshape(&[input.dim(0)?, input.dim(1)?, *a.dims().last().unwrap()])
}
