use std::sync::Arc;

use candle_core::{Result, Tensor};

use crate::ShardedVarBuilder;

use super::{add_delta, current_lora_execution, LoraLinearSpec, LoraSiteHandle, LoraSiteKey};

pub fn register_dynamic_lora_site(
    vb: &ShardedVarBuilder,
    spec: LoraLinearSpec,
) -> Result<Option<Arc<LoraSiteHandle>>> {
    let Some(registry) = vb.lora_registry() else {
        return Ok(None);
    };
    registry
        .register(
            LoraSiteKey::new(vb.prefix()),
            spec,
            vb.dtype(),
            vb.device().clone(),
        )
        .map(Some)
}

pub fn apply_dynamic_lora_delta(
    site: &LoraSiteHandle,
    input: &Tensor,
    base_output: Tensor,
) -> Result<Tensor> {
    let Some(execution) = current_lora_execution(site.runtime_id()) else {
        return Ok(base_output);
    };
    if !execution.site_is_active(site)? {
        return Ok(base_output);
    }
    add_delta(&execution, site, input, base_output)
}
