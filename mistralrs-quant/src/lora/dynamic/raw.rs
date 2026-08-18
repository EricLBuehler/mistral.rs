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

pub fn is_dynamic_lora_site_active(site: &LoraSiteHandle) -> bool {
    current_lora_execution(site.runtime_id())
        .is_some_and(|execution| execution.site_is_active(site).unwrap_or(true))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        with_lora_execution, LoraExecution, LoraLayerRegistry, LoraLinearSpec, LoraWeights,
    };
    use candle_core::{DType, Device};

    #[test]
    fn raw_site_activity_matches_delta_execution() -> Result<()> {
        let device = Device::Cpu;
        let registry = LoraLayerRegistry::new();
        let site = registry.register(
            LoraSiteKey::new("model.layers.0.altup.modality_router"),
            LoraLinearSpec::replicated(2, 2),
            DType::F32,
            device.clone(),
        )?;
        registry.finalize()?;
        let input = Tensor::new(&[[1f32, 2.]], &device)?;
        let base = Tensor::new(&[[3f32, 4.]], &device)?;

        assert!(!is_dynamic_lora_site_active(&site));
        assert_eq!(
            apply_dynamic_lora_delta(&site, &input, base.clone())?.to_vec2::<f32>()?,
            vec![vec![3., 4.]]
        );

        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(0)]);
        execution.insert(
            &site,
            0,
            LoraWeights::new(
                Tensor::new(&[[1f32, 0.], [0., 1.]], &device)?,
                Tensor::new(&[[1f32, 0.], [0., 1.]], &device)?,
                2.0,
            )?,
        )?;
        let output = with_lora_execution(Some(Arc::new(execution)), || {
            assert!(is_dynamic_lora_site_active(&site));
            apply_dynamic_lora_delta(&site, &input, base)
        })?;
        assert_eq!(output.to_vec2::<f32>()?, vec![vec![5., 8.]]);
        assert!(!is_dynamic_lora_site_active(&site));
        Ok(())
    }
}
