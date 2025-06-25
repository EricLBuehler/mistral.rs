use candle_core::{DType, Result, Tensor};
use regex::Regex;

use crate::{LoraAdapter, Shard, ShardedVarBuilder, APPLIED_LORAS};

pub fn merge_lora_weights(
    vb: &ShardedVarBuilder,
    mut weight: Tensor,
    in_dim: usize,
    out_dim: usize,
    shard: Shard,
) -> Result<Tensor> {
    for LoraAdapter { config, weights } in &*APPLIED_LORAS.lock().expect("No loras initialized.") {
        let target_modules = config
            .target_modules
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("|");
        let regex = Regex::new(&target_modules).map_err(candle_core::Error::msg)?;
        if !regex.is_match(&vb.prefix()) {
            continue;
        }

        // Handle base_model.model things from peft
        let weights = if weights
            .pp("base_model.model")
            .pp(vb.prefix())
            .contains_tensor("lora_A.weight")
        {
            weights.pp("base_model.model").pp(vb.prefix())
        } else {
            weights.pp(vb.prefix())
        };

        let a = weights.get_with_hints((config.rank, in_dim), "lora_A.weight", shard)?;
        let b = weights.get_with_hints((out_dim, config.rank), "lora_B.weight", shard)?;
        let scale = if config.rank > 0 {
            config.alpha / config.rank as f64
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

    Ok(weight)
}
