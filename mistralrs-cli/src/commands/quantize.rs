//! Quantize command implementation for UQFF generation

use std::collections::{BTreeMap, HashSet};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::{info, warn};

use mistralrs_core::{
    expand_isq_value, initialize_logging, IsqType, ModelSelected, TokenSource, UqffWriteConfig,
};
use mistralrs_server_core::mistralrs_for_server_builder::{defaults, MistralRsForServerBuilder};

use crate::args::{
    GlobalOptions, QuantizeDeviceOptions, QuantizeModelFormat, QuantizeModelSourceOptions,
    QuantizeModelType, QuantizeMultimodalOptions, QuantizeQuantizationOptions,
};

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
        QuantizeModelType::Auto { model, .. }
        | QuantizeModelType::Text { model, .. }
        | QuantizeModelType::Multimodal { model, .. } => model
            .model_id
            .as_deref()
            .expect("quantize model source was normalized"),
        QuantizeModelType::Embedding { model, .. } => &model.model_id,
    }
}

fn get_model_source(model_type: &QuantizeModelType) -> Option<&QuantizeModelSourceOptions> {
    match model_type {
        QuantizeModelType::Auto { model, .. }
        | QuantizeModelType::Text { model, .. }
        | QuantizeModelType::Multimodal { model, .. } => Some(model),
        QuantizeModelType::Embedding { .. } => None,
    }
}

fn get_model_source_mut(
    model_type: &mut QuantizeModelType,
) -> Option<&mut QuantizeModelSourceOptions> {
    match model_type {
        QuantizeModelType::Auto { model, .. }
        | QuantizeModelType::Text { model, .. }
        | QuantizeModelType::Multimodal { model, .. } => Some(model),
        QuantizeModelType::Embedding { .. } => None,
    }
}

fn get_device_options(model_type: &QuantizeModelType) -> &QuantizeDeviceOptions {
    match model_type {
        QuantizeModelType::Auto { device, .. }
        | QuantizeModelType::Text { device, .. }
        | QuantizeModelType::Multimodal { device, .. }
        | QuantizeModelType::Embedding { device, .. } => device,
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

fn resolve_gguf_source(
    model_type: &mut QuantizeModelType,
    token_source: &TokenSource,
) -> Result<()> {
    if let Some(path) = get_device_options(model_type).hf_cache.clone() {
        mistralrs_core::set_hf_cache_path(path);
    }
    let explicit_multimodal = matches!(model_type, QuantizeModelType::Multimodal { .. });
    let Some(model) = get_model_source_mut(model_type) else {
        return Ok(());
    };
    let model_id = model
        .model_id
        .as_deref()
        .expect("quantize model source was normalized")
        .to_string();
    let requested = model.quant.clone();
    let exact_file = model.format.quantized_file.clone();
    let explicit_gguf = matches!(model.format.format, Some(QuantizeModelFormat::Gguf));

    let should_inspect = requested.is_some() || (explicit_gguf && exact_file.is_some());
    let files = should_inspect
        .then(|| selected_model_files(&model_id, exact_file.as_deref(), token_source))
        .transpose()?
        .flatten();
    let confident_gguf = files
        .as_ref()
        .is_some_and(|files| is_confident_gguf_artifact_repo(&model_id, files));

    if let Some(requested) = requested.as_deref() {
        let files = files.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Could not inspect GGUF artifacts for `{model_id}`. Pass `-f <filename.gguf>` \
                 explicitly or check repository access."
            )
        })?;
        if !crate::commands::quant::has_gguf_model_files(files) {
            anyhow::bail!(
                "`--quant {requested}` selects an input GGUF artifact, but `{model_id}` has no \
                 model GGUF files"
            );
        }
        if !explicit_gguf && !confident_gguf {
            anyhow::bail!(
                "`{model_id}` contains GGUF files alongside another model format. Pass \
                 `--format gguf` to use `--quant {requested}` as the input artifact selector."
            );
        }

        let artifact = crate::commands::quant::resolve_gguf_quant(files, requested)?;
        info!(
            "quantize: --quant {requested} -> input GGUF {} from `{model_id}`",
            artifact.label
        );
        model.format.format = Some(QuantizeModelFormat::Gguf);
        model.format.quantized_file = Some(artifact.file_spec());
    }

    if matches!(model.format.format, Some(QuantizeModelFormat::Gguf)) {
        let quantized_file = model.format.quantized_file.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "GGUF input requires a model file. Pass `-f <model.gguf>`, or use \
                 `-m <GGUF-repo> --quant <level>` to select one automatically."
            )
        })?;

        if model.format.mmproj.is_none()
            && (confident_gguf || explicit_multimodal || model.format.direct_file_only)
        {
            if let Some(files) = files.as_ref() {
                if let Some(projector) =
                    crate::commands::quant::resolve_gguf_projector(files, model.dtype)?
                {
                    info!(
                        "GGUF: selected {} projector `{}`",
                        projector.label,
                        projector.file_spec()
                    );
                    model.format.mmproj = Some(projector.file_spec());
                }
            }
        }

        if quantized_file.is_empty() {
            anyhow::bail!("`--quantized-file` must contain nonempty filenames");
        }
    }

    Ok(())
}

fn selected_model_files(
    model_id: &str,
    exact_file: Option<&str>,
    token_source: &TokenSource,
) -> Result<Option<Vec<String>>> {
    let path = Path::new(model_id);
    if path.exists() {
        if let Some(exact_file) = exact_file {
            return crate::commands::quant::list_local_gguf_companions(path, exact_file).map(Some);
        }
        return crate::commands::quant::list_local_files_recursive(path).map(Some);
    }
    Ok(mistralrs_core::probe_hf_repo_files(
        model_id,
        "main",
        token_source,
    ))
}

fn model_name_looks_gguf(model_id: &str) -> bool {
    model_id
        .rsplit_once('/')
        .map_or(model_id, |(_, name)| name)
        .to_ascii_lowercase()
        .ends_with("-gguf")
}

fn is_confident_gguf_artifact_repo(model_id: &str, files: &[String]) -> bool {
    if model_name_looks_gguf(model_id) {
        return true;
    }
    !files.iter().any(|file| {
        let lower = file.to_ascii_lowercase();
        lower.ends_with(".uqff")
            || lower.ends_with(".safetensors")
            || lower.ends_with(".pth")
            || lower.ends_with(".pt")
            || lower.ends_with(".bin")
    })
}

/// Run UQFF quantization and generation, supporting multiple ISQ types.
pub async fn run_quantize(mut model_type: QuantizeModelType, global: GlobalOptions) -> Result<()> {
    initialize_logging();
    resolve_gguf_source(&mut model_type, &global.token_source)?;

    let isq_values = get_isq_values(&model_type);
    let base_output = get_output_path(&model_type).clone();
    let file_mode = base_output.extension().is_some_and(|ext| ext == "uqff");
    let model_id = get_model_id(&model_type).to_string();
    let is_multimodal = matches!(&model_type, QuantizeModelType::Multimodal { .. })
        || get_model_source(&model_type).is_some_and(|model| model.format.mmproj.is_some());
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
            "Cannot use multiple --isq values with a .uqff output path (ISQ setting produced multiple expanded ISQ values). \
             Use a directory path (e.g., -o output/) to auto-name files per ISQ type."
        );
    }

    let effective_output = if file_mode {
        base_output.clone()
    } else if expanded_isq.len() == 1 {
        std::fs::create_dir_all(&base_output)?;
        base_output.join(format!("{}.uqff", expanded_isq[0]))
    } else {
        std::fs::create_dir_all(&base_output)?;
        base_output.clone()
    };
    let write_uqff = UqffWriteConfig::with_types(effective_output.clone(), expanded_isq.clone())
        .with_report_metadata(
            Some(flag_base_model.clone().unwrap_or_else(|| model_id.clone())),
            flag_repo_id.clone(),
        );
    let requested = expanded_isq
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");

    info!(
        "Starting UQFF generation for ISQ=[{}] -> `{}`",
        requested,
        effective_output.display()
    );

    let (model_selected, cpu, device_layers) = convert_to_model_selected(&model_type, write_uqff)?;

    let mistralrs = MistralRsForServerBuilder::new()
        .with_model(model_selected)
        .with_max_seqs(1)
        .with_no_kv_cache(defaults::NO_KV_CACHE)
        .with_token_source(global.token_source.clone())
        .with_interactive_mode(defaults::INTERACTIVE_MODE)
        .with_prefix_cache_n(0)
        .set_paged_attn(Some(false))
        .with_cpu(cpu)
        .with_num_device_layers_optional(device_layers)
        .build()
        .await?;
    mistralrs.shutdown().await.map_err(anyhow::Error::msg)?;

    info!("UQFF generation for ISQ=[{}] complete!", requested);

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

Generated with [mistral.rs](https://github.com/EricLBuehler/mistral.rs) {mistralrs_version}. Documentation: [UQFF docs](https://ericlbuehler.github.io/mistral.rs/guides/quantization/uqff/).

1) **Flexible** 🌀: Multiple quantization formats in *one* file format with *one* framework to run them all.
2) **Versioned**: Embedded semantic-version metadata lets mistral.rs detect incompatible artifacts before loading.
3) **Easy** 🤗: Download UQFF models *easily* and *quickly* from Hugging Face, or use a local file.
4) **Customizable** 🛠️: Make and publish your own UQFF files in minutes.

## Install

Install [mistral.rs](https://github.com/EricLBuehler/mistral.rs) ([full guide](https://ericlbuehler.github.io/mistral.rs/guides/install/)):

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
"#,
        mistralrs_version = mistralrs_core::MISTRALRS_VERSION,
    );

    let model_type = if is_multimodal { "multimodal " } else { "" };

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
            "|{quant_name}|`mistralrs run {model_type}-m {repo_display} --from-uqff {first_file}`|\n"
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
fn convert_to_model_selected(
    model_type: &QuantizeModelType,
    write_uqff: UqffWriteConfig,
) -> Result<(ModelSelected, bool, Option<Vec<String>>)> {
    match model_type {
        QuantizeModelType::Auto {
            model,
            quantization,
            device,
            multimodal,
            ..
        } => {
            match model.format.format.unwrap_or(QuantizeModelFormat::Plain) {
                QuantizeModelFormat::Gguf => {
                    let selected = convert_gguf_source(
                        model,
                        quantization,
                        device,
                        Some(multimodal),
                        write_uqff,
                    )?;
                    return Ok((selected, device.cpu, device.device_layers.clone()));
                }
                QuantizeModelFormat::Plain => {}
            }
            let model_selected = ModelSelected::Run {
                model_id: model
                    .model_id
                    .clone()
                    .expect("quantize model source was normalized"),
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
                write_uqff: Some(write_uqff),
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
            match model.format.format.unwrap_or(QuantizeModelFormat::Plain) {
                QuantizeModelFormat::Gguf => {
                    let selected =
                        convert_gguf_source(model, quantization, device, None, write_uqff)?;
                    return Ok((selected, device.cpu, device.device_layers.clone()));
                }
                QuantizeModelFormat::Plain => {}
            }
            let model_selected = ModelSelected::Plain {
                model_id: model
                    .model_id
                    .clone()
                    .expect("quantize model source was normalized"),
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
                write_uqff: Some(write_uqff),
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
            match model.format.format.unwrap_or(QuantizeModelFormat::Plain) {
                QuantizeModelFormat::Gguf => {
                    if model.format.mmproj.is_none() {
                        anyhow::bail!(
                            "No companion projector was found for this multimodal GGUF; pass \
                             `--mmproj <filename>` to select one explicitly"
                        );
                    }
                    let selected = convert_gguf_source(
                        model,
                        quantization,
                        device,
                        Some(multimodal),
                        write_uqff,
                    )?;
                    return Ok((selected, device.cpu, device.device_layers.clone()));
                }
                QuantizeModelFormat::Plain => {}
            }
            let model_selected = ModelSelected::MultimodalPlain {
                model_id: model
                    .model_id
                    .clone()
                    .expect("quantize model source was normalized"),
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
                write_uqff: Some(write_uqff),
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

        QuantizeModelType::Embedding {
            model,
            device,
            quantization,
            ..
        } => {
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
                write_uqff: Some(write_uqff),
                from_uqff: None,
                imatrix: quantization.imatrix.clone(),
                calibration_file: quantization.calibration_file.clone(),
                hf_cache_path: device.hf_cache.clone(),
            };
            Ok((model_selected, device.cpu, device.device_layers.clone()))
        }
    }
}

fn convert_gguf_source(
    model: &QuantizeModelSourceOptions,
    quantization: &QuantizeQuantizationOptions,
    device: &QuantizeDeviceOptions,
    multimodal: Option<&QuantizeMultimodalOptions>,
    write_uqff: UqffWriteConfig,
) -> Result<ModelSelected> {
    Ok(ModelSelected::GGUF {
        tok_model_id: model.format.tok_model_id.clone(),
        quantized_model_id: model
            .model_id
            .clone()
            .expect("quantize model source was normalized"),
        quantized_filename: model
            .format
            .quantized_file
            .clone()
            .context("GGUF input requires `--quantized-file`/`-f`")?,
        tokenizer_json: model
            .tokenizer
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        mmproj_filename: model.format.mmproj.clone(),
        lora_adapters: Vec::new(),
        lora_runtime_config: None,
        dtype: model.dtype,
        topology: device
            .topology
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        organization: quantization.isq_organization,
        write_uqff: Some(write_uqff),
        imatrix: quantization.imatrix.clone(),
        calibration_file: quantization.calibration_file.clone(),
        max_edge: multimodal.and_then(|options| options.max_edge),
        max_seq_len: device.max_seq_len,
        max_batch_size: device.max_batch_size,
        max_num_images: multimodal.and_then(|options| options.max_num_images),
        max_image_length: multimodal.and_then(|options| options.max_image_length),
        hf_cache_path: device.hf_cache.clone(),
        matformer_config_path: None,
        matformer_slice_name: None,
    })
}

#[cfg(test)]
mod tests {
    use std::fs;

    use clap::Parser;

    use super::*;
    use crate::args::{resolve_quantize_model_type, Cli, Command, ModelType};

    fn parse(args: &[&str]) -> QuantizeModelType {
        let cli = Cli::try_parse_from(
            ["mistralrs", "quantize"]
                .into_iter()
                .chain(args.iter().copied()),
        )
        .unwrap();
        let Command::Quantize {
            model_type,
            default_quantize,
        } = cli.command
        else {
            unreachable!()
        };
        resolve_quantize_model_type(model_type, default_quantize).unwrap()
    }

    #[test]
    fn multimodal_model_card_commands_parse() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("q4k.uqff"), []).unwrap();

        generate_model_card(&root, "org/base", Some("org/quantized"), true).unwrap();
        let readme = fs::read_to_string(root.join("README.md")).unwrap();
        let command = readme
            .lines()
            .find(|line| line.starts_with("|Q4K|"))
            .and_then(|line| line.split('`').nth(1))
            .expect("Q4K command in generated model card");
        assert_eq!(
            command,
            "mistralrs run multimodal -m org/quantized --from-uqff q4k.uqff"
        );

        let cli = Cli::try_parse_from(command.split_whitespace()).unwrap();
        let Command::Run {
            model_type:
                Some(ModelType::Multimodal {
                    model,
                    quantization,
                    ..
                }),
            ..
        } = cli.command
        else {
            panic!("expected a multimodal run command")
        };
        assert_eq!(model.model_id, "org/quantized");
        assert_eq!(quantization.from_uqff.as_deref(), Some("q4k.uqff"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn gguf_artifact_and_projector_are_selected_for_uqff_output() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        for file in [
            "model-Q4_K_S.gguf",
            "model-Q4_K_M.gguf",
            "mmproj-F16.gguf",
            "mmproj-BF16.gguf",
        ] {
            fs::write(root.join(file), []).unwrap();
        }
        let output = root.join("output.uqff");
        let mut model_type = parse(&[
            "-m",
            &root.to_string_lossy(),
            "--quant",
            "4",
            "--isq",
            "q8_0",
            "--dtype",
            "bf16",
            "-o",
            &output.to_string_lossy(),
        ]);

        resolve_gguf_source(&mut model_type, &TokenSource::None).unwrap();
        let source = get_model_source(&model_type).expect("GGUF model source");
        assert_eq!(source.format.format, Some(QuantizeModelFormat::Gguf));
        assert_eq!(
            source.format.quantized_file.as_deref(),
            Some("model-Q4_K_M.gguf")
        );
        assert_eq!(source.format.mmproj.as_deref(), Some("mmproj-BF16.gguf"));
        assert_eq!(get_isq_values(&model_type), ["q8_0"]);

        let write_uqff = UqffWriteConfig::with_types(output.clone(), vec![IsqType::Q8_0]);
        let (selected, _, _) = convert_to_model_selected(&model_type, write_uqff).unwrap();
        let ModelSelected::GGUF {
            quantized_filename,
            mmproj_filename,
            write_uqff,
            ..
        } = selected
        else {
            panic!("expected GGUF source")
        };
        assert_eq!(quantized_filename, "model-Q4_K_M.gguf");
        assert_eq!(mmproj_filename.as_deref(), Some("mmproj-BF16.gguf"));
        let write_uqff = write_uqff.expect("GGUF source should write UQFF");
        assert_eq!(write_uqff.output, output);
        assert_eq!(write_uqff.types, [IsqType::Q8_0]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn exact_gguf_preserves_asset_and_projector_overrides() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model-Q4_K_M.gguf"), []).unwrap();
        fs::write(root.join("custom-mmproj.gguf"), []).unwrap();
        let output = root.join("output.uqff");
        let mut model_type = parse(&[
            "-m",
            &root.to_string_lossy(),
            "-f",
            "model-Q4_K_M.gguf",
            "--mmproj",
            "custom-mmproj.gguf",
            "--tok-model-id",
            "org/base",
            "--isq",
            "q5k",
            "-o",
            &output.to_string_lossy(),
        ]);

        resolve_gguf_source(&mut model_type, &TokenSource::None).unwrap();
        let selected = convert_to_model_selected(
            &model_type,
            UqffWriteConfig::with_types(output, vec![IsqType::Q5K]),
        )
        .unwrap()
        .0;
        let ModelSelected::GGUF {
            tok_model_id,
            quantized_filename,
            mmproj_filename,
            ..
        } = selected
        else {
            panic!("expected GGUF source")
        };
        assert_eq!(tok_model_id.as_deref(), Some("org/base"));
        assert_eq!(quantized_filename, "model-Q4_K_M.gguf");
        assert_eq!(mmproj_filename.as_deref(), Some("custom-mmproj.gguf"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn direct_local_gguf_discovers_only_a_sibling_projector() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        let unrelated = root.join("unrelated");
        fs::create_dir_all(&unrelated).unwrap();
        let model_path = root.join("model.gguf");
        fs::write(&model_path, []).unwrap();
        fs::write(root.join("mmproj-BF16.gguf"), []).unwrap();
        fs::write(root.join("model.safetensors"), []).unwrap();
        fs::write(unrelated.join("mmproj-BF16.gguf"), []).unwrap();
        let output = root.join("output.uqff");
        let mut model_type = parse(&[
            "-f",
            &model_path.to_string_lossy(),
            "--isq",
            "q4k",
            "-o",
            &output.to_string_lossy(),
        ]);

        let resolved = resolve_gguf_source(&mut model_type, &TokenSource::None);

        resolved.unwrap();
        let source = get_model_source(&model_type).expect("GGUF model source");
        assert_eq!(source.format.quantized_file.as_deref(), Some("model.gguf"));
        assert_eq!(source.format.mmproj.as_deref(), Some("mmproj-BF16.gguf"));
        assert!(source.format.direct_file_only);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn multimodal_gguf_requires_a_projector() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model-Q4_K_M.gguf"), []).unwrap();
        let output = root.join("output.uqff");
        let mut model_type = parse(&[
            "multimodal",
            "-m",
            &root.to_string_lossy(),
            "-f",
            "model-Q4_K_M.gguf",
            "--isq",
            "q4k",
            "-o",
            &output.to_string_lossy(),
        ]);

        resolve_gguf_source(&mut model_type, &TokenSource::None).unwrap();
        let error = convert_to_model_selected(
            &model_type,
            UqffWriteConfig::with_types(output, vec![IsqType::Q4K]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("projector"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn mixed_source_directory_requires_explicit_gguf_format() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model-Q4_K_M.gguf"), []).unwrap();
        fs::write(root.join("model.safetensors"), []).unwrap();
        let output = root.join("output.uqff");
        let root_arg = root.to_string_lossy().into_owned();
        let output_arg = output.to_string_lossy().into_owned();
        let args = [
            "-m",
            root_arg.as_str(),
            "--quant",
            "4",
            "--isq",
            "q8_0",
            "-o",
            output_arg.as_str(),
        ];

        let mut ambiguous = parse(&args);
        let error = resolve_gguf_source(&mut ambiguous, &TokenSource::None).unwrap_err();
        assert!(error.to_string().contains("--format gguf"));

        let mut explicit_args = args.to_vec();
        explicit_args.extend(["--format", "gguf"]);
        let mut explicit = parse(&explicit_args);
        resolve_gguf_source(&mut explicit, &TokenSource::None).unwrap();
        assert_eq!(
            get_model_source(&explicit)
                .expect("GGUF model source")
                .format
                .quantized_file
                .as_deref(),
            Some("model-Q4_K_M.gguf")
        );

        fs::remove_dir_all(root).unwrap();
    }
}
