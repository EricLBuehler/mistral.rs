use std::path::PathBuf;

use anyhow::Result;

use mistralrs_core::{auto_tune, AutoTuneRequest, ModelSelected};

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

    if let Some(path) = emit_config {
        let toml = emit_toml_config(&model_type, &model_selected, &result)?;
        std::fs::write(&path, toml)?;
        println!("Wrote config to {}", path.display());
    }

    if json {
        let out = serde_json::to_string_pretty(&result)?;
        println!("{out}");
        return Ok(());
    }

    println!("mistralrs tune");
    println!("Model: {}", result.model_id);
    println!("Profile: {:?}", result.profile);
    println!("Backend: {}", result.backend);
    if let Some(isq) = result.recommended_isq {
        println!("Recommended ISQ: {:?}", isq);
    }
    if let Some(spec) = &result.device_layers_cli {
        println!("Device layers: {spec}");
    }
    if let Some(mode) = result.paged_attn_mode {
        println!("PagedAttention: {mode}");
    }
    if !result.notes.is_empty() {
        println!();
        for note in &result.notes {
            println!("note: {note}");
        }
    }
    if !result.warnings.is_empty() {
        println!();
        for warning in &result.warnings {
            println!("warning: {warning}");
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
    out.push_str("port = 8080\n\n");
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
        ModelType::Config { .. } => "config",
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
