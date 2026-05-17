//! `--quant` front-door resolution: prebuilt UQFF if available, otherwise ISQ,
//! plus `auto` mode powered by `mistralrs tune`.
//!
//! When a sibling `mistralrs-community/<model>-UQFF` repo is found, we swap the
//! user's `--model-id` to that repo and set `from_uqff = <shorthand>`. The repo
//! is self-contained (config, tokenizer, residual, shards), so the existing
//! `-m <repo>-UQFF --from-uqff <shorthand>` load path handles everything without
//! any further changes to the loader.

use std::path::Path;

use anyhow::{anyhow, Result};
use tracing::{info, warn};

use mistralrs_core::{
    auto_tune, parse_isq_value, probe_hf_repo_files, AutoTuneRequest, ModelSelected, TokenSource,
    TuneProfile,
};

/// Outcome of resolving `--quant <value>`. Either points at a prebuilt UQFF repo
/// (and gives the shorthand), or applies in-situ quantization to the base model.
#[derive(Default, Debug, Clone)]
pub struct ResolvedQuant {
    /// If set, replace the user's `--model-id` with this self-contained UQFF repo.
    pub model_id_swap: Option<String>,
    /// `from_uqff` shorthand (e.g. `"q4k"`) to load from the (possibly swapped) repo.
    pub from_uqff: Option<String>,
    /// In-situ quantization level to apply to the base model.
    pub in_situ_quant: Option<String>,
}

/// Resolve `--quant <raw>` into either a UQFF reference or an ISQ string.
/// Logs a clear decision trace at INFO so the user can see what was chosen.
pub async fn resolve_quant(
    raw: &str,
    model_id: &str,
    token_source: &TokenSource,
    model_selected: &ModelSelected,
    force_cpu: bool,
) -> Result<ResolvedQuant> {
    let lowered = raw.trim().to_lowercase();

    if lowered == "auto" {
        info!("quant: auto -- probing hardware via `tune`");
        let req = AutoTuneRequest {
            model: model_selected.clone(),
            token_source: token_source.clone(),
            hf_revision: None,
            force_cpu,
            profile: TuneProfile::Balanced,
            requested_isq: None,
        };
        let result = auto_tune(req)
            .map_err(|e| anyhow!("`--quant auto` failed during tune analysis: {e}"))?;
        let Some(isq) = result.recommended_isq else {
            info!("quant: auto -> no quantization recommended (model fits at full precision)");
            return Ok(ResolvedQuant::default());
        };
        let isq_name = format!("{:?}", isq).to_lowercase();
        info!(
            "quant: auto -> {} (backend={}, vram={:.1} GB)",
            isq_name,
            result.backend,
            result.total_vram_bytes as f64 / 1e9,
        );
        return resolve_explicit(&isq_name, model_id, token_source).await;
    }

    resolve_explicit(&lowered, model_id, token_source).await
}

async fn resolve_explicit(
    raw: &str,
    model_id: &str,
    token_source: &TokenSource,
) -> Result<ResolvedQuant> {
    // Validate up front against the known ISQ vocabulary so a typo doesn't
    // silently fall through to "no UQFF, no ISQ".
    parse_isq_value(raw, None)
        .map_err(|e| anyhow!("`--quant {raw}` is not a recognized quant level: {e}"))?;

    if Path::new(model_id).exists() {
        info!("quant: model_id is a local path, skipping UQFF probe; using ISQ {raw}");
        return Ok(ResolvedQuant {
            in_situ_quant: Some(raw.to_string()),
            ..Default::default()
        });
    }

    // The CLI's own UQFF naming convention: `mistralrs-community/<base>-UQFF`.
    let base = model_id.rsplit_once('/').map_or(model_id, |(_, n)| n);
    let uqff_repo = format!("mistralrs-community/{base}-UQFF");
    info!("quant: probing prebuilt UQFF at `{uqff_repo}`");

    let token = read_hf_token(token_source);
    let Some(files) = probe_hf_repo_files(&uqff_repo, "main", token) else {
        info!("quant: no UQFF repo at `{uqff_repo}` (or unreachable); using ISQ {raw}");
        return Ok(ResolvedQuant {
            in_situ_quant: Some(raw.to_string()),
            ..Default::default()
        });
    };

    match mistralrs_core::resolve_uqff_shorthand(raw, &files) {
        Some(matched) => {
            let shorthand = mistralrs_core::parse_uqff_shard(&matched)
                .map(|(name, _)| name)
                .unwrap_or(matched.clone());
            info!("quant: using prebuilt `{uqff_repo}` (shard `{matched}`)");
            Ok(ResolvedQuant {
                model_id_swap: Some(uqff_repo),
                from_uqff: Some(shorthand),
                in_situ_quant: None,
            })
        }
        None => {
            let uqff_shards: Vec<&String> = files.iter().filter(|f| f.ends_with(".uqff")).collect();
            warn!(
                "quant: `{uqff_repo}` exists but has no shard matching `{raw}` (available: {:?}); falling back to ISQ {raw}",
                uqff_shards
            );
            Ok(ResolvedQuant {
                in_situ_quant: Some(raw.to_string()),
                ..Default::default()
            })
        }
    }
}

fn read_hf_token(source: &TokenSource) -> Option<String> {
    use std::env;
    match source {
        TokenSource::Literal(s) => Some(s.clone()),
        TokenSource::EnvVar(v) => env::var(v).ok(),
        TokenSource::Path(p) => std::fs::read_to_string(p)
            .ok()
            .map(|s| s.trim().to_string()),
        TokenSource::CacheToken => mistralrs_core::hf_token_path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|s| s.trim().to_string()),
        TokenSource::None => None,
    }
}
