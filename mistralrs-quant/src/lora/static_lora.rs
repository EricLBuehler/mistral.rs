use std::{collections::HashMap, sync::Arc};

use candle_core::Result;
use candle_nn::Linear;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::{
    lora::{get_adapter_delta, load_adapter, LoraConfigLike},
    DummyLayer, QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear,
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StaticLoraConfig {
    pub layer: String,
    pub lora_alpha: f64,
    pub r: usize,
}

impl LoraConfigLike for StaticLoraConfig {
    fn rank(&self) -> usize {
        self.r
    }
    fn alpha(&self) -> f64 {
        self.lora_alpha
    }
}

/// Static LoRA in the style of Phi-4 multimodal. Only when the layer regex for the specific LoRA matches.
///
/// Structure:
/// - `prefix.base_layer.weight`
/// - `prefix.lora_A.<lora name>.weight`
/// - `prefix.lora_B.<lora name>.weight`
pub fn linear_no_bias_static_lora(
    in_dim: usize,
    out_dim: usize,
    loras: HashMap<String, StaticLoraConfig>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let layer = {
        // Handle the case where the layer is dummy (no tensors)
        if !vb.contains_tensor("base_layer.weight") {
            let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        } else {
            let mut weight =
                vb.get_with_hints((out_dim, in_dim), "base_layer.weight", Default::default())?;

            for (name, lora_cfg) in loras {
                let regex = Regex::new(&lora_cfg.layer).map_err(candle_core::Error::msg)?;
                if !regex.is_match(&vb.prefix()) {
                    continue;
                }

                let adapter = load_adapter(
                    in_dim,
                    out_dim,
                    Some(name),
                    vb.clone(),
                    Default::default(),
                    &lora_cfg,
                )?;
                let delta_weight = get_adapter_delta(adapter)?;

                weight = (weight + delta_weight)?;
            }

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    Ok(layer)
}
