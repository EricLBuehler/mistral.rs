//! Quantize command implementation for UQFF generation

use std::collections::{BTreeMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

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

/// Extract the model ID from the QuantizeModelType
fn get_model_id(model_type: &QuantizeModelType) -> &str {
    match model_type {
        QuantizeModelType::Auto { model, .. } => &model.model_id,
        QuantizeModelType::Text { model, .. } => &model.model_id,
        QuantizeModelType::Vision { model, .. } => &model.model_id,
        QuantizeModelType::Embedding { model, .. } => &model.model_id,
    }
}

/// Extract the no_readme flag from the QuantizeModelType
fn get_no_readme(model_type: &QuantizeModelType) -> bool {
    match model_type {
        QuantizeModelType::Auto { output, .. } => output.no_readme,
        QuantizeModelType::Text { output, .. } => output.no_readme,
        QuantizeModelType::Vision { output, .. } => output.no_readme,
        QuantizeModelType::Embedding { output, .. } => output.no_readme,
    }
}

/// Run UQFF quantization and generation, supporting multiple ISQ types.
pub async fn run_quantize(model_type: QuantizeModelType, global: GlobalOptions) -> Result<()> {
    initialize_logging();

    let isq_values = get_isq_values(&model_type);
    let base_output = get_output_path(&model_type).clone();
    let file_mode = base_output.extension().is_some_and(|ext| ext == "uqff");
    let model_id = get_model_id(&model_type).to_string();
    let is_vision = matches!(&model_type, QuantizeModelType::Vision { .. });
    let no_readme = get_no_readme(&model_type);

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
            std::fs::create_dir_all(&base_output)?;
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

    // Generate README.md model card in directory mode (unless disabled)
    if !file_mode && !no_readme {
        if let Err(e) = generate_model_card(&base_output, &model_id, is_vision) {
            warn!("Failed to generate README.md: {}", e);
        }
    }

    // Print upload hint in directory mode
    if !file_mode {
        print_upload_hint(&base_output, &model_id);
    }

    Ok(())
}

/// Generate a README.md model card in the UQFF output directory.
fn generate_model_card(output_dir: &Path, model_id: &str, is_vision: bool) -> Result<()> {
    // Scan the output directory for .uqff files and group by prefix
    let mut groups: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for entry in std::fs::read_dir(output_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("uqff") {
                let stem = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or_default();
                // Group shards: strip trailing numeric suffix (e.g., "q4k-0" -> "q4k")
                let key = if let Some((pre, suf)) = stem.rsplit_once('-') {
                    if suf.chars().all(|c| c.is_ascii_digit()) {
                        pre.to_string()
                    } else {
                        stem.to_string()
                    }
                } else {
                    stem.to_string()
                };
                groups.entry(key).or_default().push(path);
            }
        }
    }

    if groups.is_empty() {
        warn!("No .uqff files found in output directory, skipping README.md generation");
        return Ok(());
    }

    let mut output = format!(
        r#"---
tags:
  - uqff
  - mistral.rs
base_model: {model_id}
base_model_relation: quantized
---

# `{model_id}`, UQFF quantization

Run with [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Documentation: [UQFF docs](https://github.com/EricLBuehler/mistral.rs/blob/master/docs/UQFF.md).

1) **Flexible**: Multiple quantization formats in *one* file format with *one* framework to run them all.
2) **Reliable**: Compatibility ensured with *embedded* and *checked* semantic versioning information from day 1.
3) **Easy**: Download UQFF models *easily* and *quickly* from Hugging Face, or use a local file.
4) **Customizable**: Make and publish your own UQFF files in minutes.

## Examples

|Quantization|Command|
|--|--|
"#
    );

    let model_type_flag = if is_vision { " vision-plain" } else { "" };

    for (prefix, paths) in &groups {
        // Sort shards by numeric suffix
        let mut paths_sorted = paths.clone();
        paths_sorted.sort_by_key(|p| {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
            if let Some((_, suf)) = stem.rsplit_once('-') {
                suf.parse::<u64>().unwrap_or(u64::MAX)
            } else {
                u64::MAX
            }
        });

        // Use only the first shard file (auto-discovery handles the rest)
        let first_file = paths_sorted[0]
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default();

        let quant_name = prefix.to_uppercase();
        output += &format!(
            "|{quant_name}|`mistralrs run -m <REPO_ID>{model_type_flag} --from-uqff {first_file}`|\n"
        );
    }

    let readme_path = output_dir.join("README.md");
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&readme_path)?;
    file.write_all(output.as_bytes())?;

    info!("Generated model card at `{}`", readme_path.display());
    Ok(())
}

/// Print the huggingface-cli upload command for the user.
fn print_upload_hint(output_dir: &Path, model_id: &str) {
    let model_name = Path::new(model_id)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model_id);

    info!("To upload your UQFF to Hugging Face, run:");
    info!(
        "  huggingface-cli upload <YOUR_USERNAME>/{model_name}-UQFF {} --repo-type model --private",
        output_dir.display()
    );
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
