//! Resolution for the `--quant` front-door.

use std::path::Path;

use anyhow::{anyhow, Result};
use tracing::{debug, info, warn};

use mistralrs_core::{
    auto_tune, parse_isq_value, parse_uqff_shard, probe_hf_repo_files, resolve_uqff_shorthand,
    AutoTuneRequest, ModelSelected, TokenSource, TuneProfile,
};

const UQFF_REPO_ORG: &str = "mistralrs-community";
const UQFF_REPO_SUFFIX: &str = "-UQFF";

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

    let shorthand = parse_uqff_shard(&matched)
        .map(|(name, _)| name)
        .unwrap_or_else(|| matched.clone());
    info!("quant: --quant {raw} -> UQFF {shorthand} from `{uqff_repo}`");
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

fn fallback_isq(raw: &str) -> ResolvedQuant {
    ResolvedQuant {
        in_situ_quant: Some(raw.to_string()),
        ..Default::default()
    }
}
