//! Resolution for the `--quant` front-door.

use std::{fs, path::Path};

use anyhow::{anyhow, Result};
use tracing::{debug, info, warn};

use mistralrs_core::{
    auto_tune, parse_isq_value, parse_uqff_shard, probe_hf_repo_files, resolve_uqff_shorthand,
    AutoTuneRequest, ModelSelected, TokenSource, TuneProfile,
};

const UQFF_REPO_ORG: &str = "mistralrs-community";
const UQFF_REPO_SUFFIX: &str = "-UQFF";
const UQFF_REPO_SUFFIX_LOWER: &str = "-uqff";
const UQFF_RESIDUAL_SAFETENSORS: &str = "residual.safetensors";

#[derive(Default, Debug, Clone)]
pub struct ResolvedQuant {
    pub model_id_swap: Option<String>,
    pub from_uqff: Option<String>,
    pub in_situ_quant: Option<String>,
}

pub async fn resolve_quant(
    raw: &str,
    model_id: &str,
    token_source: &TokenSource,
    model_selected: &ModelSelected,
    force_cpu: bool,
) -> Result<ResolvedQuant> {
    let lowered = raw.trim().to_lowercase();
    if lowered == "auto" {
        return resolve_auto(model_id, token_source, model_selected, force_cpu).await;
    }
    resolve_explicit(&lowered, model_id, token_source).await
}

async fn resolve_auto(
    model_id: &str,
    token_source: &TokenSource,
    model_selected: &ModelSelected,
    force_cpu: bool,
) -> Result<ResolvedQuant> {
    debug!("quant: auto, probing hardware via `tune`");
    let result = auto_tune(AutoTuneRequest {
        model: model_selected.clone(),
        token_source: token_source.clone(),
        hf_revision: None,
        force_cpu,
        profile: TuneProfile::Balanced,
        requested_isq: None,
    })
    .map_err(|e| anyhow!("`--quant auto` failed during tune analysis: {e}"))?;

    let Some(isq) = result.recommended_isq else {
        info!("quant: --quant auto -> full precision (model fits)");
        return Ok(ResolvedQuant::default());
    };
    let isq_name = format!("{isq:?}").to_lowercase();
    info!(
        "quant: --quant auto -> {isq_name} (backend={}, vram={:.1} GB)",
        result.backend,
        result.total_vram_bytes as f64 / 1e9,
    );
    resolve_explicit(&isq_name, model_id, token_source).await
}

async fn resolve_explicit(
    raw: &str,
    model_id: &str,
    token_source: &TokenSource,
) -> Result<ResolvedQuant> {
    parse_isq_value(raw, None)
        .map_err(|e| anyhow!("`--quant {raw}` is not a recognized quant level: {e}"))?;

    let selected_files = selected_repo_files(model_id, token_source)?;
    if let Some(files) = &selected_files {
        if let Some(resolved) = resolve_selected_uqff(raw, model_id, files)? {
            return Ok(resolved);
        }
    } else if model_name_looks_uqff(model_id) {
        anyhow::bail!(
            "Model `{model_id}` appears to be a UQFF artifact repo, but its file listing could not \
             be inspected. Use `--from-uqff {raw}` to load a known local/cached artifact, or check \
             repository access."
        );
    }

    if Path::new(model_id).exists() {
        info!("quant: --quant {raw} -> ISQ {raw} (local model)");
        return Ok(fallback_isq(raw));
    }

    let uqff_repo = sibling_uqff_repo(model_id);
    debug!("quant: probing prebuilt UQFF at `{uqff_repo}`");
    let Some(files) = probe_hf_repo_files(&uqff_repo, "main", token_source) else {
        debug!("quant: no UQFF repo at `{uqff_repo}` (or unreachable)");
        info!("quant: --quant {raw} -> ISQ {raw}");
        return Ok(fallback_isq(raw));
    };

    let Some(matched) = resolve_uqff_shorthand(raw, &files) else {
        let available: Vec<&String> = files.iter().filter(|f| f.ends_with(".uqff")).collect();
        warn!(
            "quant: `{uqff_repo}` has no shard matching `{raw}` (available: {available:?}); falling back to ISQ {raw}"
        );
        return Ok(fallback_isq(raw));
    };

    let shorthand = uqff_shorthand_from_match(&matched);
    info!(
        "quant: --quant {raw} -> UQFF {shorthand} from `{uqff_repo}`; use `--isq {raw}` to \
         quantize the selected model source instead"
    );
    Ok(ResolvedQuant {
        model_id_swap: Some(uqff_repo),
        from_uqff: Some(shorthand),
        in_situ_quant: None,
    })
}

fn sibling_uqff_repo(model_id: &str) -> String {
    let base = model_id.rsplit_once('/').map_or(model_id, |(_, n)| n);
    format!("{UQFF_REPO_ORG}/{base}{UQFF_REPO_SUFFIX}")
}

fn selected_repo_files(model_id: &str, token_source: &TokenSource) -> Result<Option<Vec<String>>> {
    let path = Path::new(model_id);
    if path.exists() {
        return Ok(Some(local_model_files(path)?));
    }
    Ok(probe_hf_repo_files(model_id, "main", token_source))
}

fn local_model_files(model_path: &Path) -> Result<Vec<String>> {
    if !model_path.is_dir() {
        return Ok(Vec::new());
    }

    let mut files = Vec::new();
    for entry in fs::read_dir(model_path).map_err(|err| {
        anyhow!(
            "Cannot list local model directory `{}`: {err}",
            model_path.display()
        )
    })? {
        let entry = entry?;
        if let Some(name) = entry.path().file_name().and_then(|name| name.to_str()) {
            files.push(name.to_string());
        }
    }
    Ok(files)
}

fn resolve_selected_uqff(
    raw: &str,
    model_id: &str,
    files: &[String],
) -> Result<Option<ResolvedQuant>> {
    if let Some(matched) = resolve_uqff_shorthand(raw, files) {
        let shorthand = uqff_shorthand_from_match(&matched);
        info!("quant: --quant {raw} -> UQFF {shorthand} from selected model `{model_id}`");
        return Ok(Some(ResolvedQuant {
            model_id_swap: None,
            from_uqff: Some(shorthand),
            in_situ_quant: None,
        }));
    }

    if is_uqff_artifact_repo(model_id, files) {
        anyhow::bail!(
            "Model `{model_id}` appears to be a UQFF artifact repo, but no UQFF shard matched \
             `--quant {raw}`. Available UQFF files: {}. Use `--from-uqff <name>` to choose an \
             available artifact, or select the base model and use `--isq {raw}` to quantize it at \
             load time.",
            format_available_uqff(files)
        );
    }

    Ok(None)
}

fn uqff_shorthand_from_match(matched: &str) -> String {
    parse_uqff_shard(matched)
        .map(|(name, _)| name)
        .unwrap_or_else(|| matched.to_string())
}

fn model_name_looks_uqff(model_id: &str) -> bool {
    model_id
        .rsplit_once('/')
        .map_or(model_id, |(_, name)| name)
        .to_ascii_lowercase()
        .ends_with(UQFF_REPO_SUFFIX_LOWER)
}

fn is_uqff_artifact_repo(model_id: &str, files: &[String]) -> bool {
    if model_name_looks_uqff(model_id) {
        return true;
    }

    let has_uqff = files.iter().any(|file| file.ends_with(".uqff"));
    let has_uqff_metadata = files
        .iter()
        .any(|file| file == mistralrs_quant::UQFF_REPORT_JSON || file == UQFF_RESIDUAL_SAFETENSORS);
    let has_source_weights = files.iter().any(|file| {
        let lower = file.to_ascii_lowercase();
        (lower.ends_with(".safetensors") && lower != UQFF_RESIDUAL_SAFETENSORS)
            || lower.ends_with(".pth")
            || lower.ends_with(".pt")
            || lower.ends_with(".bin")
    });

    has_uqff_metadata || (has_uqff && !has_source_weights)
}

fn format_available_uqff(files: &[String]) -> String {
    let available = files
        .iter()
        .filter(|file| file.ends_with(".uqff"))
        .cloned()
        .collect::<Vec<_>>();
    if available.is_empty() {
        "none".to_string()
    } else {
        available.join(", ")
    }
}

fn fallback_isq(raw: &str) -> ResolvedQuant {
    ResolvedQuant {
        in_situ_quant: Some(raw.to_string()),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_uqff_repo_resolves_matching_quant() {
        let files = vec![
            "q8_0-0.uqff".to_string(),
            UQFF_RESIDUAL_SAFETENSORS.to_string(),
            "config.json".to_string(),
        ];

        let resolved = resolve_selected_uqff("8", "mistralrs-community/foo-UQFF", &files)
            .unwrap()
            .unwrap();

        assert_eq!(resolved.model_id_swap, None);
        assert_eq!(resolved.from_uqff, Some("q8_0".to_string()));
        assert_eq!(resolved.in_situ_quant, None);
    }

    #[test]
    fn selected_uqff_repo_errors_on_unmatched_quant() {
        let files = vec![
            "q4k-0.uqff".to_string(),
            UQFF_RESIDUAL_SAFETENSORS.to_string(),
            "config.json".to_string(),
        ];

        let err = resolve_selected_uqff("8", "mistralrs-community/foo-UQFF", &files)
            .unwrap_err()
            .to_string();

        assert!(err.contains("appears to be a UQFF artifact repo"));
        assert!(err.contains("q4k-0.uqff"));
    }

    #[test]
    fn selected_mixed_source_repo_uses_matching_uqff() {
        let files = vec![
            "model.safetensors".to_string(),
            "q8_0-0.uqff".to_string(),
            "config.json".to_string(),
        ];

        let resolved = resolve_selected_uqff("8", "org/foo", &files)
            .unwrap()
            .unwrap();

        assert_eq!(resolved.from_uqff, Some("q8_0".to_string()));
        assert_eq!(resolved.in_situ_quant, None);
    }

    #[test]
    fn selected_mixed_source_repo_allows_isq_fallback_without_match() {
        let files = vec![
            "model.safetensors".to_string(),
            "q4k-0.uqff".to_string(),
            "config.json".to_string(),
        ];

        assert!(resolve_selected_uqff("8", "org/foo", &files)
            .unwrap()
            .is_none());
    }

    #[test]
    fn selected_uqff_metadata_marks_artifact_repo() {
        let files = vec![
            "q4k-0.uqff".to_string(),
            mistralrs_quant::UQFF_REPORT_JSON.to_string(),
            "config.json".to_string(),
        ];

        assert!(is_uqff_artifact_repo("org/foo", &files));
    }
}
