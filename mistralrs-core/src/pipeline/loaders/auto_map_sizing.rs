use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use candle_core::DType;
use mistralrs_quant::{IsqType, QuantizedWeightSource};
use tracing::info;

use super::{
    AutoDeviceMapQuantization, CheckpointTensors, DeviceMappedModelLoader, QuantizationConfigShim,
};
use crate::matformer::MatformerSliceConfig;
use crate::pipeline::IsqOrganization;
use crate::Topology;

/// Everything automatic device mapping needs to estimate per-layer weight bytes.
pub(crate) struct AutoMapSizingInput<'a> {
    pub config: &'a str,
    pub dtype: DType,
    pub weight_files: &'a [PathBuf],
    pub checkpoint: &'a CheckpointTensors<'a>,
    pub uqff: bool,
    pub weight_source: Option<&'a Arc<dyn QuantizedWeightSource>>,
    pub prepared_weight_source: Option<&'a Arc<dyn QuantizedWeightSource>>,
    pub in_situ_quant: Option<IsqType>,
    pub topology: Option<&'a Topology>,
    pub organization: IsqOrganization,
    pub has_lora_adapters: bool,
    pub matformer: Option<&'a MatformerSliceConfig>,
    /// Text loaders size non-mapped weights at the source pack factor; multimodal ones assume unpacked.
    pub packed_non_mapped: bool,
}

pub(crate) fn exact_weight_source_sizes(
    source: &dyn QuantizedWeightSource,
    dtype: DType,
    num_layers: usize,
) -> Result<Option<(Vec<usize>, usize, usize)>> {
    let Some(exact) = source.resident_layer_bytes(dtype)? else {
        return Ok(None);
    };
    let layers = (0..num_layers)
        .map(|layer| exact.layers.get(&layer).copied().unwrap_or(0))
        .collect::<Vec<_>>();
    let total = layers.iter().sum::<usize>() + exact.outside;
    Ok(Some((layers, exact.outside, total)))
}

/// Returns per-layer bytes, non-mapped bytes and their total, from the most exact source available.
pub(crate) fn auto_device_map_sizes<L: DeviceMappedModelLoader + ?Sized>(
    inner: &L,
    input: AutoMapSizingInput<'_>,
) -> Result<(Vec<usize>, usize, usize)> {
    let config = input.config;
    let dtype = input.dtype;
    let source_quantization = QuantizationConfigShim::get_quantization_config(config)?;
    let sizes = match crate::pipeline::isq_flow::resolve_auto_device_map_sizing(
        input.uqff,
        input.prepared_weight_source.is_some(),
        input.in_situ_quant,
    ) {
        sizing @ (crate::pipeline::isq_flow::AutoDeviceMapSizing::Uqff
        | crate::pipeline::isq_flow::AutoDeviceMapSizing::PreparedWeightSource) => {
            let source = input
                .weight_source
                .expect("selected weight-source sizing requires a weight source");
            let exact = if input.topology.is_none() && input.matformer.is_none() {
                exact_weight_source_sizes(source.as_ref(), dtype, inner.num_layers(config)?)?
            } else {
                None
            };
            if let Some(sizes) = exact {
                info!(
                    model_mib = sizes.2 / (1024 * 1024),
                    "Using quantized weight-source inventory for automatic device mapping"
                );
                sizes
            } else {
                let quantization =
                    if matches!(sizing, crate::pipeline::isq_flow::AutoDeviceMapSizing::Uqff) {
                        AutoDeviceMapQuantization::weight_source(source.as_ref())
                    } else {
                        AutoDeviceMapQuantization::weight_source_with_topology(
                            source.as_ref(),
                            input.topology,
                        )
                    };
                let quantization = quantization
                    .with_checkpoint(input.checkpoint)
                    .with_source_quantization(source_quantization.as_ref());
                let weight_pack_factor =
                    quantization.conservative_pack_factor(dtype, source.pack_factor(dtype)?);
                let layer_sizes_in_bytes = inner.layer_sizes_in_bytes_with_quantization(
                    config,
                    dtype,
                    weight_pack_factor,
                    Some(&quantization),
                    input.matformer,
                )?;
                let non_mapped_pack_factor = if input.packed_non_mapped
                    || matches!(sizing, crate::pipeline::isq_flow::AutoDeviceMapSizing::Uqff)
                {
                    weight_pack_factor
                } else {
                    1
                };
                let non_mapped_size_in_bytes = inner.non_mapped_size_in_bytes(
                    config,
                    dtype,
                    non_mapped_pack_factor,
                    Some(&quantization),
                    input.matformer,
                )?;
                let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                (
                    layer_sizes_in_bytes,
                    non_mapped_size_in_bytes,
                    layer_sizes_sum + non_mapped_size_in_bytes,
                )
            }
        }
        crate::pipeline::isq_flow::AutoDeviceMapSizing::Isq(isq) => {
            let moqe = matches!(input.organization, IsqOrganization::MoeExpertsOnly);
            let source_pack_factor = if let Some(source) = input.prepared_weight_source {
                source.pack_factor(dtype)?
            } else {
                QuantizationConfigShim::get_quant_config_pack_factor(config, dtype)?
            };
            let target_pack_factor = isq.pack_factor(dtype);
            let (weight_pack_factor, non_mapped_pack_factor, quantization) = if moqe {
                let quantization = input.prepared_weight_source.map_or_else(
                    || AutoDeviceMapQuantization::isq(None, input.topology),
                    |source| {
                        AutoDeviceMapQuantization::weight_source_with_topology(
                            source.as_ref(),
                            input.topology,
                        )
                    },
                );
                (
                    quantization.conservative_moqe_pack_factor(dtype, source_pack_factor, isq),
                    if input.packed_non_mapped {
                        source_pack_factor
                    } else {
                        1
                    },
                    quantization,
                )
            } else {
                let quantization = AutoDeviceMapQuantization::isq(Some(isq), input.topology);
                (
                    quantization.conservative_pack_factor(dtype, target_pack_factor),
                    target_pack_factor,
                    quantization,
                )
            };
            let quantization = quantization
                .with_checkpoint(input.checkpoint)
                .with_source_quantization(source_quantization.as_ref());
            let layer_sizes_in_bytes = inner.layer_sizes_in_bytes_with_quantization(
                config,
                dtype,
                weight_pack_factor,
                (!moqe).then_some(&quantization),
                input.matformer,
            )?;
            let non_mapped_size_in_bytes = inner.non_mapped_size_in_bytes(
                config,
                dtype,
                non_mapped_pack_factor,
                Some(&quantization),
                input.matformer,
            )?;
            let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
            (
                layer_sizes_in_bytes,
                non_mapped_size_in_bytes,
                layer_sizes_sum + non_mapped_size_in_bytes,
            )
        }
        crate::pipeline::isq_flow::AutoDeviceMapSizing::Checkpoint => {
            let inventory = if input.topology.is_none()
                && !input.has_lora_adapters
                && input.matformer.is_none()
                && inner.supports_checkpoint_weight_inventory(config)?
            {
                let num_layers = inner.num_layers(config)?;
                crate::pipeline::loaders::checkpoint_device_map_sizes(
                    input.weight_files,
                    num_layers,
                    dtype,
                    |name| inner.checkpoint_layer_index(config, name),
                )?
            } else {
                None
            };
            if let Some(inventory) = inventory {
                info!(
                    model_mib = inventory.total_model_size_in_bytes / (1024 * 1024),
                    "Using checkpoint tensor inventory for automatic device mapping"
                );
                (
                    inventory.layer_sizes_in_bytes,
                    inventory.non_mapped_size_in_bytes,
                    inventory.total_model_size_in_bytes,
                )
            } else {
                // Be sure to get the weight pack factor here; we might be loading a prequantized model.
                let weight_pack_factor =
                    QuantizationConfigShim::get_quant_config_pack_factor(config, dtype)?;
                let checkpoint_quantization =
                    QuantizationConfigShim::get_quantization_config(config)?;
                let quantization = AutoDeviceMapQuantization::checkpoint(
                    checkpoint_quantization.as_ref(),
                    input.topology,
                )
                .map(|quantization| {
                    quantization
                        .with_checkpoint(input.checkpoint)
                        .with_source_quantization(source_quantization.as_ref())
                });
                let weight_pack_factor = quantization
                    .as_ref()
                    .map_or(weight_pack_factor, |quantization| {
                        quantization.conservative_pack_factor(dtype, weight_pack_factor)
                    });
                let layer_sizes_in_bytes = inner.layer_sizes_in_bytes_with_quantization(
                    config,
                    dtype,
                    weight_pack_factor,
                    quantization.as_ref(),
                    input.matformer,
                )?;
                let non_mapped_size_in_bytes = inner.non_mapped_size_in_bytes(
                    config,
                    dtype,
                    weight_pack_factor,
                    quantization.as_ref(),
                    input.matformer,
                )?;
                let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                (
                    layer_sizes_in_bytes,
                    non_mapped_size_in_bytes,
                    layer_sizes_sum + non_mapped_size_in_bytes,
                )
            }
        }
    };
    Ok(sizes)
}
