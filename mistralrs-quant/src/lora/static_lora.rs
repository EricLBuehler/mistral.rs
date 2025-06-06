use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Result};
use candle_nn::Linear;
use regex::Regex;

use crate::{DummyLayer, QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear};

use super::StaticLoraConfig;

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

                let a = vb.get((lora_cfg.r, in_dim), &format!("lora_A.{name}.weight"))?;
                let b = vb.get((out_dim, lora_cfg.r), &format!("lora_B.{name}.weight"))?;
                let scale = if lora_cfg.r > 0 {
                    lora_cfg.lora_alpha / lora_cfg.r as f64
                } else {
                    1.0
                };

                let ab = if a.device().is_cpu() {
                    b.to_dtype(DType::F32)?.matmul(&a.to_dtype(DType::F32)?)?
                } else {
                    b.matmul(&a)?
                };

                let delta_weight = (ab * scale)?;
                weight = (weight + delta_weight.to_dtype(a.dtype())?)?;
            }

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    Ok(layer)
}
