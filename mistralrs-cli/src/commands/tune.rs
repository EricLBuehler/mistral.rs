use std::path::PathBuf;

use anyhow::Result;
use comfy_table::{presets::UTF8_FULL, Cell, Color, ContentArrangement, Table};

use mistralrs_core::{auto_tune, AutoTuneRequest, FitStatus, ModelSelected, QualityTier};

use crate::args::{GlobalOptions, ModelType, TuneProfileArg};

use super::serve::{convert_to_model_selected, extract_device_settings, extract_isq_setting};

pub async fn run_tune(
    model_type: ModelType,
    global: GlobalOptions,
    profile: TuneProfileArg,
    json: bool,
    emit_config: Option<PathBuf>,
) -> Result<()> {
    let model_selected = convert_to_model_selected(&model_type)?;
    let (cpu, _device_layers) = extract_device_settings(&model_type);
    let requested_isq = extract_isq_setting(&model_type)
        .as_deref()
        .map(|s| {
            mistralrs_core::parse_isq_value(s, None)
                .map_err(|err| anyhow::anyhow!("Invalid --isq value: {err}"))
        })
        .transpose()?;

    let request = AutoTuneRequest {
        model: model_selected.clone(),
        token_source: global.token_source,
        hf_revision: None,
        force_cpu: cpu,
        profile: profile.into(),
        requested_isq,
    };

    let result = auto_tune(request)?;

    if let Some(ref path) = emit_config {
        let toml = emit_toml_config(&model_type, &model_selected, &result)?;
        std::fs::write(path, toml)?;
        println!("Wrote config to {}", path.display());
    }

    if json {
        let out = serde_json::to_string_pretty(&result)?;
        println!("{out}");
        return Ok(());
    }

    // Header
    println!();
    println!("Tuning Analysis");
    println!("===============");
    println!();
    println!("Model: {}", result.model_id);
    println!("Profile: {:?}", result.profile);
    println!("Backend: {}", result.backend);
    if result.total_vram_bytes > 0 {
        println!("Total VRAM: {:.1} GB", result.total_vram_bytes as f64 / 1e9);
    }
    println!();

    // Notes
    if !result.notes.is_empty() {
        for note in &result.notes {
            println!("[INFO] {note}");
        }
        println!();
    }

    // Analysis Matrix Table
    println!("Quantization Options");
    println!("--------------------");

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Quant"),
            Cell::new("Est. Size"),
            Cell::new("VRAM %"),
            Cell::new("Context Room"),
            Cell::new("Quality"),
            Cell::new("Status"),
        ]);

    for candidate in &result.candidates {
        let size_str = format!("{:.2} GB", candidate.estimated_size_bytes as f64 / 1e9);
        let vram_pct = format!("{:.0}%", candidate.vram_usage_percent * 100.0);
        let context_str = {
            let base = if candidate.max_context_tokens >= 1000 {
                format!("{}k", candidate.max_context_tokens / 1000)
            } else {
                format!("{}", candidate.max_context_tokens)
            };
            if candidate.context_is_model_max {
                format!("{base} (max)")
            } else {
                base
            }
        };

        let quality_str = match candidate.quality {
            QualityTier::Baseline => "Baseline",
            QualityTier::NearLossless => "Near-lossless",
            QualityTier::Good => "Good",
            QualityTier::Acceptable => "Acceptable",
            QualityTier::Degraded => "Degraded",
        };

        let (status_str, status_color) = match candidate.fit_status {
            FitStatus::Fits if candidate.recommended => ("🚀 Recommended", Color::Green),
            FitStatus::Fits => ("✅ Fits", Color::Green),
            FitStatus::Hybrid if candidate.recommended => {
                ("🚀 Recommended (Hybrid)", Color::Yellow)
            }
            FitStatus::Hybrid => ("⚠️ Hybrid", Color::Yellow),
            FitStatus::TooLarge => ("❌ Too Large", Color::Red),
        };

        table.add_row(vec![
            Cell::new(&candidate.isq_name),
            Cell::new(&size_str),
            Cell::new(&vram_pct),
            Cell::new(&context_str),
            Cell::new(quality_str),
            Cell::new(status_str).fg(status_color),
        ]);
    }

    println!("{table}");
    println!();

    // Warnings
    if !result.warnings.is_empty() {
        println!();
        for warning in &result.warnings {
            println!("[WARN] {warning}");
        }
    }

    // Recommended command
    println!();
    println!("Recommended Command");
    println!("-------------------");
    if let Some(ref path) = emit_config {
        println!("  mistralrs from-config --file {}", path.display());
        println!();
        println!("Or equivalently:");
    }
    println!("  {}", result.recommended_command);
    println!();

    if let Some(mode) = &result.paged_attn_mode {
        if mode != "off" {
            println!("[INFO] PagedAttention is available (mode: {mode})");
        }
    }

    Ok(())
}

fn emit_toml_config(
    model_type: &ModelType,
    model_selected: &ModelSelected,
    result: &mistralrs_core::AutoTuneResult,
) -> Result<String> {
    let mut out = String::new();
    out.push_str("command = \"serve\"\n\n");
    out.push_str("[server]\n");
    out.push_str("host = \"0.0.0.0\"\n");
    out.push_str("port = 1234\n\n");
    out.push_str("[runtime]\n");
    out.push_str("max_seqs = 32\n\n");

    out.push_str("[[models]]\n");
    out.push_str(&format!("kind = \"{}\"\n", model_kind(model_type)));
    out.push_str(&format!("model_id = \"{}\"\n", result.model_id));
    if let Some(dtype) = model_dtype(model_selected) {
        out.push_str(&format!("dtype = \"{}\"\n", dtype));
    }

    if let Some(isq) = result.recommended_isq {
        out.push_str("\n[models.quantization]\n");
        out.push_str(&format!("in_situ_quant = \"{:?}\"\n", isq));
    }

    if let Some(spec) = &result.device_layers_cli {
        let items = spec
            .split(';')
            .filter(|s| !s.trim().is_empty())
            .map(|s| format!("\"{}\"", s))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str("\n[models.device]\n");
        out.push_str(&format!("device_layers = [{items}]\n"));
    }

    Ok(out)
}

fn model_kind(model_type: &ModelType) -> &'static str {
    match model_type {
        ModelType::Auto { .. } => "auto",
        ModelType::Text { .. } => "text",
        ModelType::Vision { .. } => "vision",
        ModelType::Diffusion { .. } => "diffusion",
        ModelType::Speech { .. } => "speech",
        ModelType::Embedding { .. } => "embedding",
    }
}

fn model_dtype(model_selected: &ModelSelected) -> Option<&'static str> {
    use mistralrs_core::ModelDType;
    match model_selected {
        ModelSelected::Plain { dtype, .. }
        | ModelSelected::Lora { dtype, .. }
        | ModelSelected::XLora { dtype, .. }
        | ModelSelected::GGUF { dtype, .. }
        | ModelSelected::GGML { dtype, .. }
        | ModelSelected::LoraGGUF { dtype, .. }
        | ModelSelected::XLoraGGUF { dtype, .. }
        | ModelSelected::LoraGGML { dtype, .. }
        | ModelSelected::XLoraGGML { dtype, .. }
        | ModelSelected::VisionPlain { dtype, .. }
        | ModelSelected::DiffusionPlain { dtype, .. }
        | ModelSelected::Run { dtype, .. }
        | ModelSelected::Speech { dtype, .. }
        | ModelSelected::Embedding { dtype, .. } => Some(match dtype {
            ModelDType::Auto => "auto",
            ModelDType::F16 => "f16",
            ModelDType::BF16 => "bf16",
            ModelDType::F32 => "f32",
        }),
        ModelSelected::Toml { .. } | ModelSelected::MultiModel { .. } => None,
    }
}
