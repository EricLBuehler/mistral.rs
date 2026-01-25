//! Quantize command implementation for UQFF generation

use anyhow::Result;
use tracing::info;

use mistralrs_core::{initialize_logging, ModelSelected};
use mistralrs_server_core::mistralrs_for_server_builder::{
    defaults, MistralRsForServerBuilder,
};

use crate::args::{GlobalOptions, QuantizeModelType};

/// Run UQFF quantization and generation
pub async fn run_quantize(model_type: QuantizeModelType, global: GlobalOptions) -> Result<()> {
    initialize_logging();

    // Convert quantize args to ModelSelected with write_uqff set
    let (model_selected, cpu, device_layers, isq) = convert_to_model_selected(&model_type)?;

    info!("Starting UQFF generation...");

    // Build the MistralRs instance - this triggers model loading and UQFF generation
    let _mistralrs = MistralRsForServerBuilder::new()
        .with_model(model_selected)
        .with_max_seqs(1)
        .with_no_kv_cache(defaults::NO_KV_CACHE)
        .with_token_source(global.token_source)
        .with_interactive_mode(defaults::INTERACTIVE_MODE)
        .with_prefix_cache_n(0)
        .with_cpu(cpu)
        .with_num_device_layers_optional(device_layers)
        .with_in_situ_quant_optional(Some(isq))
        .build()
        .await?;

    info!("UQFF generation complete!");

    Ok(())
}

/// Convert QuantizeModelType to ModelSelected with write_uqff set
fn convert_to_model_selected(
    model_type: &QuantizeModelType,
) -> Result<(ModelSelected, bool, Option<Vec<String>>, String)> {
    match model_type {
        QuantizeModelType::Auto {
            model,
            quantization,
            device,
            output,
            vision,
        } => {
            let model_selected = ModelSelected::Run {
                model_id: model.model_id.clone(),
                tokenizer_json: model.tokenizer.as_ref().map(|p| p.to_string_lossy().to_string()),
                dtype: model.dtype,
                topology: device.topology.as_ref().map(|p| p.to_string_lossy().to_string()),
                organization: quantization.isq_organization,
                write_uqff: Some(output.output_path.clone()),
                from_uqff: None,
                imatrix: quantization.imatrix.clone(),
                calibration_file: quantization.calibration_file.clone(),
                max_edge: vision.max_edge,
                max_seq_len: device.max_seq_len,
                max_batch_size: device.max_batch_size,
                max_num_images: vision.max_num_images,
                max_image_length: vision.max_image_length,
                hf_cache_path: device.hf_cache.clone(),
                matformer_config_path: None,
                matformer_slice_name: None,
            };
            Ok((
                model_selected,
                device.cpu,
                device.device_layers.clone(),
                quantization.in_situ_quant.clone(),
            ))
        }

        QuantizeModelType::Text {
            model,
            arch,
            quantization,
            device,
            output,
        } => {
            let model_selected = ModelSelected::Plain {
                model_id: model.model_id.clone(),
                tokenizer_json: model.tokenizer.as_ref().map(|p| p.to_string_lossy().to_string()),
                arch: arch.clone(),
                dtype: model.dtype,
                topology: device.topology.as_ref().map(|p| p.to_string_lossy().to_string()),
                organization: quantization.isq_organization,
                write_uqff: Some(output.output_path.clone()),
                from_uqff: None,
                imatrix: quantization.imatrix.clone(),
                calibration_file: quantization.calibration_file.clone(),
                max_seq_len: device.max_seq_len,
                max_batch_size: device.max_batch_size,
                hf_cache_path: device.hf_cache.clone(),
                matformer_config_path: None,
                matformer_slice_name: None,
            };
            Ok((
                model_selected,
                device.cpu,
                device.device_layers.clone(),
                quantization.in_situ_quant.clone(),
            ))
        }

        QuantizeModelType::Vision {
            model,
            quantization,
            device,
            output,
            vision,
        } => {
            let model_selected = ModelSelected::VisionPlain {
                model_id: model.model_id.clone(),
                tokenizer_json: model.tokenizer.as_ref().map(|p| p.to_string_lossy().to_string()),
                arch: None,
                dtype: model.dtype,
                topology: device.topology.as_ref().map(|p| p.to_string_lossy().to_string()),
                write_uqff: Some(output.output_path.clone()),
                from_uqff: None,
                max_edge: vision.max_edge,
                calibration_file: quantization.calibration_file.clone(),
                imatrix: quantization.imatrix.clone(),
                max_seq_len: device.max_seq_len,
                max_batch_size: device.max_batch_size,
                max_num_images: vision.max_num_images.unwrap_or(1),
                max_image_length: vision.max_image_length.unwrap_or(1024),
                hf_cache_path: device.hf_cache.clone(),
                matformer_config_path: None,
                matformer_slice_name: None,
            };
            Ok((
                model_selected,
                device.cpu,
                device.device_layers.clone(),
                quantization.in_situ_quant.clone(),
            ))
        }

        QuantizeModelType::Embedding {
            model,
            quantization,
            device,
            output,
        } => {
            let model_selected = ModelSelected::Embedding {
                model_id: model.model_id.clone(),
                tokenizer_json: model.tokenizer.as_ref().map(|p| p.to_string_lossy().to_string()),
                arch: None,
                dtype: model.dtype,
                topology: device.topology.as_ref().map(|p| p.to_string_lossy().to_string()),
                write_uqff: Some(output.output_path.clone()),
                from_uqff: None,
                hf_cache_path: device.hf_cache.clone(),
            };
            Ok((
                model_selected,
                device.cpu,
                device.device_layers.clone(),
                quantization.in_situ_quant.clone(),
            ))
        }
    }
}
