//! Quantize command implementation for UQFF generation

use std::collections::{BTreeMap, HashSet};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::Result;
use tracing::{info, warn};

use mistralrs_core::{expand_isq_value, initialize_logging, IsqType, ModelSelected};
use mistralrs_server_core::mistralrs_for_server_builder::{defaults, MistralRsForServerBuilder};

use crate::args::{GlobalOptions, QuantizeModelType};

/// Extract ISQ values from the QuantizeModelType
fn get_isq_values(model_type: &QuantizeModelType) -> &[String] {
    match model_type {
        QuantizeModelType::Auto { quantization, .. } => &quantization.in_situ_quant,
        QuantizeModelType::Text { quantization, .. } => &quantization.in_situ_quant,
        QuantizeModelType::Multimodal { quantization, .. } => &quantization.in_situ_quant,
        QuantizeModelType::Embedding { quantization, .. } => &quantization.in_situ_quant,
    }
}

/// Extract the output path from the QuantizeModelType
fn get_output_path(model_type: &QuantizeModelType) -> &PathBuf {
    match model_type {
        QuantizeModelType::Auto { output, .. } => &output.output_path,
        QuantizeModelType::Text { output, .. } => &output.output_path,
        QuantizeModelType::Multimodal { output, .. } => &output.output_path,
        QuantizeModelType::Embedding { output, .. } => &output.output_path,
    }
}

/// Extract the model ID from the QuantizeModelType
fn get_model_id(model_type: &QuantizeModelType) -> &str {
    match model_type {
        QuantizeModelType::Auto { model, .. } => &model.model_id,
        QuantizeModelType::Text { model, .. } => &model.model_id,
        QuantizeModelType::Multimodal { model, .. } => &model.model_id,
        QuantizeModelType::Embedding { model, .. } => &model.model_id,
    }
}

/// Extract the no_readme flag from the QuantizeModelType
fn get_no_readme(model_type: &QuantizeModelType) -> bool {
    match model_type {
        QuantizeModelType::Auto { output, .. } => output.no_readme,
        QuantizeModelType::Text { output, .. } => output.no_readme,
        QuantizeModelType::Multimodal { output, .. } => output.no_readme,
        QuantizeModelType::Embedding { output, .. } => output.no_readme,
    }
}

/// Extract the README override flags from the QuantizeModelType
fn get_readme_overrides(model_type: &QuantizeModelType) -> (Option<String>, Option<String>) {
    match model_type {
        QuantizeModelType::Auto { output, .. }
        | QuantizeModelType::Text { output, .. }
        | QuantizeModelType::Multimodal { output, .. }
        | QuantizeModelType::Embedding { output, .. } => {
            (output.uqff_base_model.clone(), output.uqff_repo_id.clone())
        }
    }
}

/// Run UQFF quantization and generation, supporting multiple ISQ types.
pub async fn run_quantize(model_type: QuantizeModelType, global: GlobalOptions) -> Result<()> {
    initialize_logging();

    let isq_values = get_isq_values(&model_type);
    let base_output = get_output_path(&model_type).clone();
    let file_mode = base_output.extension().is_some_and(|ext| ext == "uqff");
    let model_id = get_model_id(&model_type).to_string();
    let is_multimodal = matches!(&model_type, QuantizeModelType::Multimodal { .. });
    let no_readme = get_no_readme(&model_type);
    let (flag_base_model, flag_repo_id) = get_readme_overrides(&model_type);

    // Expand numeric ISQ shorthands into concrete variants (both Metal and non-Metal),
    // then deduplicate by IsqType.
    let mut seen_strings = HashSet::new();
    let mut seen_types = HashSet::new();
    let mut expanded_isq: Vec<IsqType> = Vec::new();
    for val in isq_values {
        if !seen_strings.insert(val.to_lowercase()) {
            warn!("Duplicate --isq value '{}'; skipping.", val);
            continue;
        }
        let types = expand_isq_value(val)?;
        for tp in types {
            if seen_types.insert(tp) {
                expanded_isq.push(tp);
            }
        }
    }

    // Multiple expanded ISQ types require directory output mode
    if expanded_isq.len() > 1 && file_mode {
        anyhow::bail!(
            "Cannot use multiple --isq values with a .uqff output path. \
             Use a directory path (e.g., -o output/) to auto-name files per ISQ type."
        );
    }

    let total = expanded_isq.len();

    for (i, isq) in expanded_isq.iter().enumerate() {
        let effective_output = if file_mode {
            base_output.clone()
        } else {
            std::fs::create_dir_all(&base_output)?;
            base_output.join(format!("{isq}.uqff"))
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
            .set_paged_attn(Some(false))
            .with_cpu(cpu)
            .with_num_device_layers_optional(device_layers)
            .with_in_situ_quant(isq.to_string())
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

    // Generate README.md model card and upload hint in directory mode
    if !file_mode {
        let (base_model, repo_id) = if no_readme {
            (model_id.clone(), flag_repo_id)
        } else if flag_base_model.is_some() || flag_repo_id.is_some() {
            // CLI flags provided, skip interactive prompts
            (
                flag_base_model.unwrap_or_else(|| model_id.clone()),
                flag_repo_id,
            )
        } else {
            prompt_readme_details(&model_id)
        };

        if !no_readme {
            if let Err(e) =
                generate_model_card(&base_output, &base_model, repo_id.as_deref(), is_multimodal)
            {
                warn!("Failed to generate README.md: {}", e);
            }
        }

        print_upload_hint(&base_output, repo_id.as_deref(), &model_id);
    }

    Ok(())
}

/// Prompt the user for base model and upload destination to populate the README.
fn prompt_readme_details(default_model_id: &str) -> (String, Option<String>) {
    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    // Ask for base model
    eprintln!();
    eprint!("Base model for the README (press Enter for '{default_model_id}'): ",);
    io::stderr().flush().ok();
    let base_model = lines
        .next()
        .and_then(|l| l.ok())
        .map(|l| l.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| default_model_id.to_string());

    // Ask for upload destination
    eprint!("HF repo where this will be uploaded (e.g. 'user/model-UQFF', press Enter to skip): ");
    io::stderr().flush().ok();
    let repo_id = lines
        .next()
        .and_then(|l| l.ok())
        .map(|l| l.trim().to_string())
        .filter(|s| !s.is_empty());

    eprintln!();
    (base_model, repo_id)
}

/// Generate a README.md model card in the UQFF output directory.
fn generate_model_card(
    output_dir: &Path,
    base_model: &str,
    repo_id: Option<&str>,
    is_multimodal: bool,
) -> Result<()> {
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

    let repo_display = repo_id.unwrap_or("<REPO_ID>");

    let has_afq = groups.keys().any(|k| k.to_lowercase().starts_with("afq"));
    let afq_note = if has_afq {
        "**Note:** AFQ variants are optimized for Apple Silicon / Metal."
    } else {
        ""
    };

    let mut output = format!(
        r#"---
tags:
  - uqff
  - mistral.rs
base_model: {base_model}
base_model_relation: quantized
---

# `{base_model}`, UQFF quantization

Run with [mistral.rs](https://github.com/EricLBuehler/mistral.rs). Documentation: [UQFF docs](https://ericlbuehler.github.io/mistral.rs/UQFF.html).

1) **Flexible** 🌀: Multiple quantization formats in *one* file format with *one* framework to run them all.
2) **Reliable** 🔒: Compatibility ensured with *embedded* and *checked* semantic versioning information from day 1.
3) **Easy** 🤗: Download UQFF models *easily* and *quickly* from Hugging Face, or use a local file.
4) **Customizable** 🛠️: Make and publish your own UQFF files in minutes.

## Install

Install [mistral.rs](https://github.com/EricLBuehler/mistral.rs) ([full guide](https://ericlbuehler.github.io/mistral.rs/INSTALLATION.html)):

**Linux/macOS:**
```
curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh | sh
```

**Windows (PowerShell):**
```
irm https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1 | iex
```

## Examples

{afq_note}

|Quantization|Command|
|--|--|
"#
    );

    let model_type_flag = if is_multimodal {
        " multimodal-plain"
    } else {
        ""
    };

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
            "|{quant_name}|`mistralrs run -m {repo_display}{model_type_flag} --from-uqff {first_file}`|\n"
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

/// Print the hf cli upload command for the user.
fn print_upload_hint(output_dir: &Path, repo_id: Option<&str>, model_id: &str) {
    let repo = if let Some(id) = repo_id {
        id.to_string()
    } else {
        let model_name = Path::new(model_id)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(model_id);
        format!("<YOUR_USERNAME>/{model_name}-UQFF")
    };

    info!("To upload your UQFF to Hugging Face, run:");
    info!(
        "  hf upload {repo} {} --repo-type model --private",
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
            multimodal,
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
                max_edge: multimodal.max_edge,
                max_seq_len: device.max_seq_len,
                max_batch_size: device.max_batch_size,
                max_num_images: multimodal.max_num_images,
                max_image_length: multimodal.max_image_length,
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

        QuantizeModelType::Multimodal {
            model,
            quantization,
            device,
            multimodal,
            ..
        } => {
            let model_selected = ModelSelected::MultimodalPlain {
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
                max_edge: multimodal.max_edge,
                calibration_file: quantization.calibration_file.clone(),
                imatrix: quantization.imatrix.clone(),
                max_seq_len: device.max_seq_len,
                max_batch_size: device.max_batch_size,
                max_num_images: multimodal.max_num_images.unwrap_or(1),
                max_image_length: multimodal.max_image_length.unwrap_or(1024),
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
