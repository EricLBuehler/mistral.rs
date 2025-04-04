mod static_lora;

use std::{
    collections::HashSet,
    sync::{Arc, LazyLock, Mutex},
};

use candle_core::{DType, Result, Tensor};
use regex::Regex;
use serde::{Deserialize, Serialize};
pub use static_lora::linear_no_bias_static_lora;

use crate::{Shard, ShardedVarBuilder};

pub static APPLIED_LORAS: LazyLock<Arc<Mutex<Vec<LoraAdapter>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(Vec::new())));

pub const MULTI_LORA_DELIMITER: &str = ";";

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct StaticLoraConfig {
    pub layer: String,
    pub lora_alpha: f64,
    pub r: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoraConfig {
    #[serde(rename = "r")]
    pub rank: usize,
    #[serde(rename = "lora_alpha")]
    pub alpha: f64,
    pub target_modules: HashSet<String>,
}

pub struct LoraAdapter {
    pub config: LoraConfig,
    pub weights: ShardedVarBuilder,
}

pub(crate) fn merge_lora_weights(
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
        let weights = weights.set_prefix(vb.prefix());

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
