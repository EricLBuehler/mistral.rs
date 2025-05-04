#![allow(clippy::cast_precision_loss)]

use std::{collections::HashSet, fmt::Debug, sync::Arc};

use candle_core::{quantized::QTensor, DType, IndexOp, Result, Tensor, D};
use candle_nn::{Linear, Module};
use loralinear::LoraLinear;
use mistralrs_quant::{QuantMethod, ShardedVarBuilder};
pub use qloralinear::QLoraLinear;
use serde::Deserialize;

mod loralinear;
mod qloralinear;

use std::collections::HashMap;

use crate::layers;

#[derive(Clone, Debug, Deserialize)]
pub struct PreloadAdapter {
    pub name: String,
    pub adapter_model_id: String,
}

#[derive(Clone, Debug, Deserialize)]
/// Adapter model ordering information.
pub struct Ordering {
    #[serde(rename = "order")]
    pub adapters: Option<Vec<String>>,
    pub layers: Option<HashMap<String, usize>>,
    pub base_model_id: String,
    pub preload_adapters: Option<Vec<PreloadAdapter>>,
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

#[derive(Debug)]
struct Adapter {
    a: Linear,
    b: Linear,
    scale: f64,
}

fn make_adapter(
    a_vb: ShardedVarBuilder,
    b_vb: ShardedVarBuilder,
    cfg: &LoraConfig,
    linear_cfg: &LoraLinearConfig,
) -> Result<Adapter> {
    assert!(a_vb.contains_tensor("weight"));
    let a = a_vb.get((cfg.rank, linear_cfg.in_features), "weight")?;
    assert!(b_vb.contains_tensor("weight"));
    let b = b_vb.get((linear_cfg.out_features, cfg.rank), "weight")?;
    let a = Linear::new(a, None);
    let b = Linear::new(b, None);
    let scale = if cfg.rank > 0 {
        cfg.alpha / cfg.rank as f64
    } else {
        1.0
    };
    Ok(Adapter { a, b, scale })
}

/// Any layer that is linear-like.
pub trait LinearLayerLike: Merge {
    fn quantized_act_type(&self) -> Option<DType>;
    fn quant_inner(&mut self) -> &mut Arc<dyn QuantMethod>;
    fn is_lora(&self) -> bool;
    fn weight(&self) -> &Tensor;
    fn bias(&self) -> Option<&Tensor>;
    fn lora_forward(
        &self,
        x: &Tensor,
        scalings_layer: Option<Tensor>,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor>;
}

pub trait Merge {
    /// Get the delta weight of the LoRA layer. This is meant to be an internal method.
    fn get_delta_weight(&self, adapter: usize) -> Result<Tensor>;
    /// Merge the LoRA weights.
    fn merge_weights(&mut self) -> Result<()>;
}

impl Merge for Linear {
    fn merge_weights(&mut self) -> Result<()> {
        Ok(())
    }
    fn get_delta_weight(&self, _adapter: usize) -> Result<Tensor> {
        unreachable!()
    }
}

impl LinearLayerLike for Linear {
    fn bias(&self) -> Option<&Tensor> {
        self.bias()
    }
    fn quant_inner(&mut self) -> &mut Arc<dyn QuantMethod> {
        unimplemented!("Linear layer has no reasonable quant inner!")
    }
    fn weight(&self) -> &Tensor {
        self.weight()
    }
    fn lora_forward(
        &self,
        x: &Tensor,
        _scalings_layer: Option<Tensor>,
        _global_scaling_weight: f64,
        _is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        self.forward(x)
    }
    fn quantized_act_type(&self) -> Option<DType> {
        None
    }
    fn is_lora(&self) -> bool {
        false
    }
}

#[allow(clippy::too_many_arguments)]
pub fn linear(
    d1: usize,
    d2: usize,
    base_vb: ShardedVarBuilder,
    vb: ShardedVarBuilder,
    lora_config: &[((String, String), LoraConfig)],
    count: &mut usize,
    ord: &Ordering,
    preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
) -> Result<Arc<dyn LinearLayerLike + Send + Sync>> {
    let prefix = vb.prefix();
    let module = prefix.split('.').next_back().unwrap();

    let linear_config = LoraLinearConfig::new(d1, d2);
    let inner = layers::linear(d1, d2, base_vb.clone())?;

    let target_modules = &lora_config.first().map(|c| &c.1.target_modules);
    for (_, cfg) in lora_config {
        if target_modules
            .as_ref()
            .is_some_and(|target_modules| &cfg.target_modules != *target_modules)
        {
            candle_core::bail!("Expected all target modules to be the same.");
        }
    }

    if !target_modules
        .as_ref()
        .is_some_and(|target_modules| target_modules.contains(module))
    {
        return Ok(Arc::new(inner));
    }
    let name = prefix.split("lora_A").last().unwrap();
    let layer = if let Some(ref layers) = ord.layers {
        *layers.get(name).unwrap()
    } else {
        0
    };

    let lorainner = LoraLinear::new(
        &inner,
        &linear_config,
        lora_config,
        &vb,
        layer,
        preload_adapters,
    )?;
    *count += 1;
    Ok(Arc::new(lorainner))
}

#[allow(clippy::too_many_arguments)]
pub fn linear_no_bias(
    d1: usize,
    d2: usize,
    base_vb: ShardedVarBuilder,
    vb: ShardedVarBuilder,
    lora_config: &[((String, String), LoraConfig)],
    count: &mut usize,
    ord: &Ordering,
    preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
) -> Result<Arc<dyn LinearLayerLike + Send + Sync>> {
    let prefix = vb.prefix();
    let module = prefix.split('.').next_back().unwrap();

    let linear_config = LoraLinearConfig::new(d1, d2);
    let inner = layers::linear_no_bias(d1, d2, base_vb.clone())?;

    let target_modules = &lora_config.first().map(|c| &c.1.target_modules);
    for (_, cfg) in lora_config {
        if target_modules
            .as_ref()
            .is_some_and(|target_modules| &cfg.target_modules != *target_modules)
        {
            candle_core::bail!("Expected all target modules to be the same.");
        }
    }

    if !target_modules
        .as_ref()
        .is_some_and(|target_modules| target_modules.contains(module))
    {
        return Ok(Arc::new(inner));
    }
    let name = prefix.split("lora_A").last().unwrap();
    let layer = if let Some(ref layers) = ord.layers {
        *layers.get(name).unwrap()
    } else {
        0
    };

    let lorainner = LoraLinear::new(
        &inner,
        &linear_config,
        lora_config,
        &vb,
        layer,
        preload_adapters,
    )?;
    *count += 1;
    Ok(Arc::new(lorainner))
}

fn get_maybe_topk_scalings(scalings: Tensor, layer: usize) -> Result<Tensor> {
    scalings.i((.., .., layer, ..))
}

#[allow(clippy::too_many_arguments)]
pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    base_vb: ShardedVarBuilder,
    vb: ShardedVarBuilder,
    lora_config: &[((String, String), LoraConfig)],
    count: &mut usize,
    ord: &Ordering,
    preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
) -> Result<Arc<dyn LinearLayerLike + Send + Sync>> {
    if bias {
        linear(
            in_dim,
            out_dim,
            base_vb,
            vb,
            lora_config,
            count,
            ord,
            preload_adapters,
        )
    } else {
        linear_no_bias(
            in_dim,
            out_dim,
            base_vb,
            vb,
            lora_config,
            count,
            ord,
            preload_adapters,
        )
    }
}

pub fn get_lora_cfg(tensor: &QTensor) -> LoraLinearConfig {
    LoraLinearConfig::new(tensor.shape().dims()[1], tensor.shape().dims()[0])
}
