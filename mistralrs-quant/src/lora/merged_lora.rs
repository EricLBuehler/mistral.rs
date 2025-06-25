use candle_core::{Result, Tensor};
use regex::Regex;

use crate::{
    get_applied_loras,
    lora::{get_adapter_delta, load_adapter, AppliedLoraKind},
    LoraAdapter, Shard, ShardedVarBuilder,
};

pub fn merge_lora_weights(
    vb: &ShardedVarBuilder,
    mut weight: Tensor,
    in_dim: usize,
    out_dim: usize,
    shard: Shard,
) -> Result<Tensor> {
    let Some(applied_loras) = get_applied_loras() else {
        return Ok(weight);
    };

    if !matches!(applied_loras.kind, AppliedLoraKind::Merged) {
        return Ok(weight);
    }

    for LoraAdapter { config, weights } in applied_loras.adapters {
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

        let adapter = load_adapter(in_dim, out_dim, None, weights, shard, &config)?;
        let delta_weight = get_adapter_delta(adapter)?;

        weight = (weight + delta_weight)?;
    }

    Ok(weight)
}
