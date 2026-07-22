use candle_core::{Result, Tensor};

use super::{LoraExecution, LoraSiteHandle, LoraWeights};

fn validate_weights(
    weights: &LoraWeights,
    input: &Tensor,
    input_features: usize,
    output_features: usize,
) -> Result<()> {
    let (rank, a_input) = weights.a.dims2()?;
    let (b_output, b_rank) = weights.b.dims2()?;
    if a_input != input_features || b_output != output_features || b_rank != rank {
        candle_core::bail!(
            "LoRA weight shape mismatch: A={:?}, B={:?}, expected input={input_features}, output={output_features}",
            weights.a.dims(),
            weights.b.dims()
        );
    }
    if weights.a.dtype() != input.dtype() || weights.b.dtype() != input.dtype() {
        candle_core::bail!(
            "LoRA weights must use input dtype {:?}, got A={:?}, B={:?}",
            input.dtype(),
            weights.a.dtype(),
            weights.b.dtype()
        );
    }
    if weights.a.device().location() != input.device().location()
        || weights.b.device().location() != input.device().location()
    {
        candle_core::bail!("LoRA weights and input must be on the same device");
    }
    Ok(())
}

pub(crate) fn add_delta(
    execution: &LoraExecution,
    site: &LoraSiteHandle,
    input: &Tensor,
    base_output: Tensor,
) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if let Some(output) = super::cuda::try_add_delta_cuda(execution, site, input, &base_output)? {
        return Ok(output);
    }
    add_delta_reference(execution, site, input, base_output)
}

pub(super) fn add_delta_reference(
    execution: &LoraExecution,
    site: &LoraSiteHandle,
    input: &Tensor,
    base_output: Tensor,
) -> Result<Tensor> {
    let input_features = input.dim(candle_core::D::Minus1)?;
    let output_features = base_output.dim(candle_core::D::Minus1)?;
    if input_features == 0 || output_features == 0 {
        candle_core::bail!("LoRA input and output feature dimensions must be nonzero");
    }
    let rows = input.elem_count() / input_features;
    if execution.row_slots().len() != rows {
        candle_core::bail!(
            "LoRA route count {} does not match input row count {rows}",
            execution.row_slots().len()
        );
    }
    if input.dims()[..input.rank() - 1] != base_output.dims()[..base_output.rank() - 1]
        || base_output.elem_count() != rows * output_features
    {
        candle_core::bail!("LoRA base output leading dimensions do not match input");
    }

    let mut active_slots = Vec::new();
    for slot in execution.rows_by_slot().keys() {
        if execution.weights(site, *slot)?.is_none() {
            continue;
        }
        active_slots.push(*slot);
    }
    if active_slots.is_empty() {
        return Ok(base_output);
    }

    let input = input.reshape((rows, input_features))?.contiguous()?;
    let mut index_parts = Vec::with_capacity(active_slots.len());
    let mut delta_parts = Vec::with_capacity(active_slots.len());
    for slot in active_slots {
        let weights = execution
            .weights(site, slot)?
            .expect("slot was filtered by weight presence");
        validate_weights(weights, &input, input_features, output_features)?;
        let indices = execution
            .row_indices(slot, input.device())?
            .expect("active LoRA slot has routed rows");
        let selected = input.index_select(&indices, 0)?;
        let hidden = selected.matmul(&weights.a.t()?)?;
        let mut selected_delta = hidden.matmul(&weights.b.t()?)?;
        if weights.scale != 1.0 {
            selected_delta = (selected_delta * weights.scale)?;
        }
        index_parts.push(indices);
        delta_parts.push(selected_delta);
    }

    let index_refs = index_parts.iter().collect::<Vec<_>>();
    let delta_refs = delta_parts.iter().collect::<Vec<_>>();
    let indices = Tensor::cat(&index_refs, 0)?;
    let delta = Tensor::cat(&delta_refs, 0)?;
    let output_shape = base_output.shape().clone();
    base_output
        .reshape((rows, output_features))?
        .index_add(&indices, &delta, 0)?
        .reshape(output_shape)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use candle_core::{DType, Device, Tensor};

    use super::*;
    use crate::{LoraLayerRegistry, LoraLinearSpec, LoraRuntimeId, LoraSiteKey, LoraWeights};

    fn site() -> Result<(LoraRuntimeId, Arc<LoraSiteHandle>)> {
        let registry = LoraLayerRegistry::new();
        let site = registry.register(
            LoraSiteKey::new("model.layers.0.self_attn.q_proj"),
            LoraLinearSpec::replicated(2, 2),
            DType::F32,
            Device::Cpu,
        )?;
        registry.finalize()?;
        Ok((registry.runtime_id(), site))
    }

    #[test]
    fn mixed_sequence_slots_match_reference_math() -> Result<()> {
        let device = Device::Cpu;
        let (runtime_id, site) = site()?;
        let mut execution =
            LoraExecution::from_sequence_slots(runtime_id, &[Some(0), None, Some(1)], 2);
        execution.insert(
            &site,
            0,
            LoraWeights::new(
                Tensor::new(&[[1f32, 0.], [0., 1.]], &device)?,
                Tensor::new(&[[1f32, 0.], [0., 1.]], &device)?,
                1.0,
            )?,
        )?;
        execution.insert(
            &site,
            1,
            LoraWeights::new(
                Tensor::new(&[[1f32, 1.]], &device)?,
                Tensor::new(&[[2f32], [3.]], &device)?,
                0.5,
            )?,
        )?;

        let input = Tensor::new(
            &[
                [[1f32, 2.], [3., 4.]],
                [[5., 6.], [7., 8.]],
                [[2., 4.], [1., 3.]],
            ],
            &device,
        )?;
        let base = Tensor::zeros((3, 2, 2), DType::F32, &device)?;
        let output = add_delta_reference(&execution, &site, &input, base)?;
        assert_eq!(
            output.to_vec3::<f32>()?,
            vec![
                vec![vec![1., 2.], vec![3., 4.]],
                vec![vec![0., 0.], vec![0., 0.]],
                vec![vec![6., 9.], vec![4., 6.]],
            ]
        );
        Ok(())
    }

    #[test]
    fn weights_for_another_site_leave_base_output_unchanged() -> Result<()> {
        let device = Device::Cpu;
        let registry = LoraLayerRegistry::new();
        let active_site = registry.register(
            LoraSiteKey::new("active"),
            LoraLinearSpec::replicated(2, 2),
            DType::F32,
            Device::Cpu,
        )?;
        let inactive_site = registry.register(
            LoraSiteKey::new("inactive"),
            LoraLinearSpec::replicated(2, 2),
            DType::F32,
            Device::Cpu,
        )?;
        registry.finalize()?;
        let mut execution = LoraExecution::new(registry.runtime_id(), vec![Some(0), Some(0)]);
        execution.insert(
            &active_site,
            0,
            LoraWeights::new(
                Tensor::new(&[[1f32, 0.]], &device)?,
                Tensor::new(&[[1f32], [1.]], &device)?,
                1.0,
            )?,
        )?;
        let input = Tensor::new(&[[1f32, 2.], [3., 4.]], &device)?;
        let base = Tensor::new(&[[5f32, 6.], [7., 8.]], &device)?;

        let output = add_delta_reference(&execution, &inactive_site, &input, base.clone())?;
        assert_eq!(output.to_vec2::<f32>()?, base.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn reference_rejects_route_shape_weight_and_dtype_mismatches() -> Result<()> {
        let device = Device::Cpu;
        let (runtime_id, site) = site()?;
        let input = Tensor::zeros((1, 2), DType::F32, &device)?;
        let base = Tensor::zeros((1, 2), DType::F32, &device)?;

        let route_mismatch = LoraExecution::new(runtime_id, vec![Some(0), Some(0)]);
        assert!(add_delta_reference(&route_mismatch, &site, &input, base.clone()).is_err());

        let mut shape_mismatch = LoraExecution::new(runtime_id, vec![Some(0)]);
        shape_mismatch.insert(
            &site,
            0,
            LoraWeights::new(
                Tensor::zeros((1, 3), DType::F32, &device)?,
                Tensor::zeros((2, 1), DType::F32, &device)?,
                1.0,
            )?,
        )?;
        assert!(add_delta_reference(&shape_mismatch, &site, &input, base.clone()).is_err());

        let mut dtype_mismatch = LoraExecution::new(runtime_id, vec![Some(0)]);
        dtype_mismatch.insert(
            &site,
            0,
            LoraWeights::new(
                Tensor::zeros((1, 2), DType::F64, &device)?,
                Tensor::zeros((2, 1), DType::F64, &device)?,
                1.0,
            )?,
        )?;
        assert!(add_delta_reference(&dtype_mismatch, &site, &input, base).is_err());
        Ok(())
    }

    #[test]
    fn reference_rejects_different_leading_dimensions_with_equal_row_counts() -> Result<()> {
        let device = Device::Cpu;
        let (runtime_id, site) = site()?;
        let execution = LoraExecution::new(runtime_id, vec![None, None]);
        let input = Tensor::zeros((1, 2, 2), DType::F32, &device)?;
        let base = Tensor::zeros((2, 1, 2), DType::F32, &device)?;
        assert!(add_delta_reference(&execution, &site, &input, base).is_err());
        Ok(())
    }
}
