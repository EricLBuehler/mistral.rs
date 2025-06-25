use std::{cell::RefCell, collections::HashSet};

use crate::{Shard, ShardedVarBuilder};
use candle_core::{DType, Result, Tensor};
use serde::{Deserialize, Serialize};

thread_local! {
    static ENGINE_APPLIED_LORAS: RefCell<Vec<LoraAdapter>> = const { RefCell::new(Vec::new()) };
}

/// Get the LoRA adapters for the current engine thread
pub fn get_applied_loras() -> Vec<LoraAdapter> {
    ENGINE_APPLIED_LORAS.with(|loras| loras.borrow().clone())
}

/// Push a LoRA adapter for the current engine thread
pub fn push_applied_lora(adapter: LoraAdapter) {
    ENGINE_APPLIED_LORAS.with(|loras| loras.borrow_mut().push(adapter));
}

pub const MULTI_LORA_DELIMITER: &str = ";";

pub(crate) trait LoraConfigLike {
    fn rank(&self) -> usize;
    fn alpha(&self) -> f64;
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LoraConfig {
    #[serde(rename = "r")]
    pub rank: usize,
    #[serde(rename = "lora_alpha")]
    pub alpha: f64,
    pub target_modules: HashSet<String>,
}

impl LoraConfigLike for LoraConfig {
    fn rank(&self) -> usize {
        self.rank
    }
    fn alpha(&self) -> f64 {
        self.alpha
    }
}

#[derive(Clone)]
pub struct LoraAdapter {
    pub config: LoraConfig,
    pub weights: ShardedVarBuilder,
}

pub struct InstantiatedLoraAdapter {
    pub a: Tensor,
    pub b: Tensor,
    pub scale: f64,
}

pub(crate) fn load_adapter<C: LoraConfigLike>(
    in_dim: usize,
    out_dim: usize,
    sub_adapter_name: Option<String>,
    vb: ShardedVarBuilder,
    shard: Shard,
    cfg: &C,
) -> Result<InstantiatedLoraAdapter> {
    let (a, b) = if let Some(name) = sub_adapter_name {
        let a = vb.get_with_hints(
            (cfg.rank(), in_dim),
            &format!("lora_A.{name}.weight"),
            shard,
        )?;
        let b = vb.get_with_hints(
            (out_dim, cfg.rank()),
            &format!("lora_B.{name}.weight"),
            shard,
        )?;
        (a, b)
    } else {
        let a = vb.get_with_hints((cfg.rank(), in_dim), "lora_A.weight", shard)?;
        let b = vb.get_with_hints((out_dim, cfg.rank()), "lora_B.weight", shard)?;
        (a, b)
    };
    let scale = if cfg.rank() > 0 {
        cfg.alpha() / cfg.rank() as f64
    } else {
        1.0
    };

    Ok(InstantiatedLoraAdapter { a, b, scale })
}

pub(crate) fn get_adapter_delta(
    InstantiatedLoraAdapter { a, b, scale }: InstantiatedLoraAdapter,
) -> Result<Tensor> {
    let ab = if a.device().is_cpu() {
        b.to_dtype(DType::F32)?.matmul(&a.to_dtype(DType::F32)?)?
    } else {
        b.matmul(&a)?
    };

    (ab * scale)?.to_dtype(a.dtype())
}
