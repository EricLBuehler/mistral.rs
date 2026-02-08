//! Quantize command implementation for UQFF generation

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Result;
use tracing::{info, warn};

use mistralrs_core::{initialize_logging, parse_isq_value, ModelSelected};
use mistralrs_server_core::mistralrs_for_server_builder::{defaults, MistralRsForServerBuilder};

use crate::args::{GlobalOptions, QuantizeModelType};

/// Extract ISQ values from the QuantizeModelType
fn get_isq_values(model_type: &QuantizeModelType) -> &[String] {
    match model_type {
        QuantizeModelType::Auto { quantization, .. } => &quantization.in_situ_quant,
        QuantizeModelType::Text { quantization, .. } => &quantization.in_situ_quant,
        QuantizeModelType::Vision { quantization, .. } => &quantization.in_situ_quant,
        QuantizeModelType::Embedding { quantization, .. } => &quantization.in_situ_quant,
    }
}

/// Extract the output path from the QuantizeModelType
fn get_output_path(model_type: &QuantizeModelType) -> &PathBuf {
    match model_type {
        QuantizeModelType::Auto { output, .. } => &output.output_path,
        QuantizeModelType::Text { output, .. } => &output.output_path,
        QuantizeModelType::Vision { output, .. } => &output.output_path,
        QuantizeModelType::Embedding { output, .. } => &output.output_path,
    }
}

/// Run UQFF quantization and generation, supporting multiple ISQ types.
pub async fn run_quantize(model_type: QuantizeModelType, global: GlobalOptions) -> Result<()> {
    initialize_logging();

    let isq_values = get_isq_values(&model_type);
    let base_output = get_output_path(&model_type);
    let file_mode = base_output.extension().map_or(false, |ext| ext == "uqff");

    // Multiple ISQ values require directory output mode
    if isq_values.len() > 1 && file_mode {
        anyhow::bail!(
            "Cannot use multiple --isq values with a .uqff output path. \
             Use a directory path (e.g., -o output/) to auto-name files per ISQ type."
        );
    }

    // Deduplicate ISQ values, preserving order
    let mut seen = HashSet::new();
    let unique_isq: Vec<&String> = isq_values
        .iter()
        .filter(|v| seen.insert(v.to_lowercase()))
        .collect();
    if unique_isq.len() < isq_values.len() {
        warn!("Duplicate --isq values detected; deduplicating.");
    }

    // Validate all ISQ strings upfront (fail fast before loading any models)
    for isq in &unique_isq {
        parse_isq_value(isq, None)
            .map_err(|e| anyhow::anyhow!("Invalid --isq value '{}': {}", isq, e))?;
    }

    let total = unique_isq.len();

    for (i, isq) in unique_isq.iter().enumerate() {
        let effective_output = if file_mode {
            base_output.clone()
        } else {
            std::fs::create_dir_all(base_output)?;
            base_output.join(format!("{}.uqff", isq.to_lowercase()))
        };

        info!(
            "[{}/{}] Starting UQFF generation for ISQ={} -> `{}`",
            i + 1,
            total,
            isq,
            effective_output.display()
        );

        let (model_selected, cpu, device_layers) =
            convert_to_model_selected(&model_type, effective_output)?;

        // Build the MistralRs instance - this triggers model loading and UQFF generation
        let _mistralrs = MistralRsForServerBuilder::new()
            .with_model(model_selected)
            .with_max_seqs(1)
            .with_no_kv_cache(defaults::NO_KV_CACHE)
            .with_token_source(global.token_source.clone())
            .with_interactive_mode(defaults::INTERACTIVE_MODE)
            .with_prefix_cache_n(0)
            .with_cpu(cpu)
            .with_num_device_layers_optional(device_layers)
            .with_in_situ_quant_optional(Some(isq.to_string()))
            .build()
            .await?;

        info!(
            "[{}/{}] UQFF generation for ISQ={} complete!",
            i + 1,
            total,
            isq
        );
    }

    if total > 1 {
        info!("All {} UQFF quantizations complete!", total);
    }

    Ok(())
}

/// Convert QuantizeModelType to ModelSelected with write_uqff set.
/// The `output_path` parameter overrides the output from the struct (to support per-ISQ paths).
fn convert_to_model_selected(
    model_type: &QuantizeModelType,
    output_path: PathBuf,
) -> Result<(ModelSelected, bool, Option<Vec<String>>)> {
    match model_type {
        QuantizeModelType::Auto {
            model,
            quantization,
            device,
            vision,
            ..
        } => {
            let model_selected = ModelSelected::Run {
                model_id: model.model_id.clone(),
                tokenizer_json: model
                    .tokenizer
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                dtype: model.dtype,
                topology: device
                    .topology
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                organization: quantization.isq_organization,
                write_uqff: Some(output_path),
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
            Ok((model_selected, device.cpu, device.device_layers.clone()))
        }

        QuantizeModelType::Text {
            model,
            arch,
            quantization,
            device,
            ..
        } => {
            let model_selected = ModelSelected::Plain {
                model_id: model.model_id.clone(),
                tokenizer_json: model
                    .tokenizer
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                arch: arch.clone(),
                dtype: model.dtype,
                topology: device
                    .topology
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                organization: quantization.isq_organization,
                write_uqff: Some(output_path),
                from_uqff: None,
                imatrix: quantization.imatrix.clone(),
                calibration_file: quantization.calibration_file.clone(),
                max_seq_len: device.max_seq_len,
                max_batch_size: device.max_batch_size,
                hf_cache_path: device.hf_cache.clone(),
                matformer_config_path: None,
                matformer_slice_name: None,
            };
            Ok((model_selected, device.cpu, device.device_layers.clone()))
        }

        QuantizeModelType::Vision {
            model,
            quantization,
            device,
            vision,
            ..
        } => {
            let model_selected = ModelSelected::VisionPlain {
                model_id: model.model_id.clone(),
                tokenizer_json: model
                    .tokenizer
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                arch: None,
                dtype: model.dtype,
                topology: device
                    .topology
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                write_uqff: Some(output_path),
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
                organization: quantization.isq_organization,
            };
            Ok((model_selected, device.cpu, device.device_layers.clone()))
        }

        QuantizeModelType::Embedding { model, device, .. } => {
            let model_selected = ModelSelected::Embedding {
                model_id: model.model_id.clone(),
                tokenizer_json: model
                    .tokenizer
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                arch: None,
                dtype: model.dtype,
                topology: device
                    .topology
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                write_uqff: Some(output_path),
                from_uqff: None,
                hf_cache_path: device.hf_cache.clone(),
            };
            Ok((model_selected, device.cpu, device.device_layers.clone()))
        }
    }
}
