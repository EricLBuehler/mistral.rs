use super::*;

pub(crate) fn tensor_prefix(vb: &ShardedVarBuilder) -> String {
    let prefix = vb.prefix();
    if prefix.is_empty() {
        "<root>".to_string()
    } else {
        prefix
    }
}

pub(crate) fn missing_required_tensors(vb: &ShardedVarBuilder, required: &[&str]) -> Vec<String> {
    required
        .iter()
        .copied()
        .filter(|name| !vb.contains_tensor(name))
        .map(|name| safetensors::full_tensor_name(vb, name))
        .collect()
}

pub(crate) fn has_missing_required_tensors(vb: &ShardedVarBuilder, required: &[&str]) -> bool {
    required.iter().any(|name| !vb.contains_tensor(name))
}

pub(crate) fn make_dummy_or_error(
    context: &str,
    vb: &ShardedVarBuilder,
    required: &[&str],
) -> Result<Arc<dyn QuantMethod>> {
    let missing = missing_required_tensors(vb, required);
    if missing.is_empty() {
        candle_core::bail!(
            "Internal error: requested DummyLayer for {context} without missing tensors"
        );
    }

    let has_uqff_placeholder = required
        .iter()
        .any(|name| safetensors::is_uqff_dummy_tensor(vb, name));
    if !has_uqff_placeholder {
        candle_core::bail!(
            "Missing required tensor(s) for {context} at prefix `{}`: {}. Dummy layers are only allowed for tensors intentionally omitted while loading UQFF artifacts.",
            tensor_prefix(vb),
            missing.join(", ")
        );
    }

    Ok(Arc::new(DummyLayer::placeholder(DummyLayerInfo {
        context: context.to_string(),
        prefix: tensor_prefix(vb),
        missing_tensors: missing,
    })))
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let base_vb = vb.clone();
    if let Some(source) = base_vb.weight_source() {
        let load_device = weight_source_load_device(&base_vb);
        if let Some(layer) =
            source.load_linear(&base_vb.prefix(), &load_device, Shard::default())?
        {
            let layer = maybe_wrap_dynamic_lora(
                &base_vb,
                layer,
                LoraLinearSpec::replicated(in_dim, out_dim),
            )?;
            return apply_immediate_isq_sharded(layer, base_vb, Some(Shard::default()));
        }
    }
    let vb = if should_apply_immediate_isq(&vb) {
        vb.set_device(Device::Cpu)
    } else {
        vb
    };

    let layer = if let Some(quant_conf) = &config {
        match quant_conf {
            QuantizedConfig::GptqAwq { .. } => gptq_linear(in_dim, out_dim, quant_conf, vb)?,
            QuantizedConfig::Fp8 { .. }
            | QuantizedConfig::CompressedTensors { .. }
            | QuantizedConfig::ModelOpt { .. } => fp8_config::checkpoint_linear_b(
                in_dim,
                out_dim,
                quant_conf,
                false,
                Default::default(),
                vb,
            )?,
            QuantizedConfig::Bitsandbytes { .. } => {
                Arc::new(BnbLinear::linear_b(in_dim, out_dim, false, vb)?) as Arc<_>
            }
            QuantizedConfig::Afq { .. } => {
                AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, false, vb)?
            }
            QuantizedConfig::MXFP4 {} => {
                MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, false, vb)?
            }
        }
    } else {
        if !vb.contains_tensor("weight") {
            make_dummy_or_error("linear_no_bias", &vb, &["weight"])?
        } else {
            let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    let layer =
        maybe_wrap_dynamic_lora(&base_vb, layer, LoraLinearSpec::replicated(in_dim, out_dim))?;
    apply_immediate_isq_sharded(layer, base_vb, Some(Shard::default()))
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let base_vb = vb.clone();
    if let Some(source) = base_vb.weight_source() {
        let load_device = weight_source_load_device(&base_vb);
        if let Some(layer) =
            source.load_linear(&base_vb.prefix(), &load_device, Shard::default())?
        {
            let layer = maybe_wrap_dynamic_lora(
                &base_vb,
                layer,
                LoraLinearSpec::replicated(in_dim, out_dim),
            )?;
            return apply_immediate_isq_sharded(layer, base_vb, Some(Shard::default()));
        }
    }
    let vb = if should_apply_immediate_isq(&vb) {
        vb.set_device(Device::Cpu)
    } else {
        vb
    };

    let layer = if let Some(quant_conf) = &config {
        match quant_conf {
            QuantizedConfig::GptqAwq { .. } => gptq_linear(in_dim, out_dim, quant_conf, vb)?,
            QuantizedConfig::Fp8 { .. }
            | QuantizedConfig::CompressedTensors { .. }
            | QuantizedConfig::ModelOpt { .. } => fp8_config::checkpoint_linear_b(
                in_dim,
                out_dim,
                quant_conf,
                true,
                Default::default(),
                vb,
            )?,
            QuantizedConfig::Bitsandbytes { .. } => {
                Arc::new(BnbLinear::linear_b(in_dim, out_dim, true, vb)?) as Arc<_>
            }
            QuantizedConfig::Afq { .. } => {
                AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, true, vb)?
            }
            QuantizedConfig::MXFP4 {} => {
                MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, true, vb)?
            }
        }
    } else {
        if has_missing_required_tensors(&vb, &["weight", "bias"]) {
            make_dummy_or_error("linear", &vb, &["weight", "bias"])?
        } else {
            let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;
            let bias = vb.get_with_hints((out_dim,), "bias", Default::default())?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, Some(bias)),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    let layer =
        maybe_wrap_dynamic_lora(&base_vb, layer, LoraLinearSpec::replicated(in_dim, out_dim))?;
    apply_immediate_isq_sharded(layer, base_vb, Some(Shard::default()))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    config: &Option<QuantizedConfig>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    if bias {
        linear(in_dim, out_dim, config, vb)
    } else {
        linear_no_bias(in_dim, out_dim, config, vb)
    }
}
