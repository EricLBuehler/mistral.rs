//! `--quant` front-door resolution: prebuilt UQFF if available, otherwise ISQ,
//! plus `auto` mode powered by `mistralrs tune`.

use std::path::Path;

use anyhow::{anyhow, Result};
use tracing::{info, warn};

use mistralrs_core::{
    auto_tune, parse_isq_value, probe_hf_repo_files, AutoTuneRequest, ModelSelected, TokenSource,
    TuneProfile,
};

/// Result of resolving `--quant <value>`. At most one of these is `Some`.
#[derive(Default, Debug, Clone)]
pub struct ResolvedQuant {
    /// `from_uqff` payload to set on the model. Format: `"<repo>::<isq-name>"`.
    pub from_uqff: Option<String>,
    /// `in_situ_quant` value to set on the model.
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
            from_uqff: None,
            in_situ_quant: Some(raw.to_string()),
        });
    }

    let base = match model_id.rsplit_once('/') {
        Some((_, name)) => name,
        None => model_id,
    };
    let uqff_repo = format!("mistralrs-community/{base}-UQFF");
    info!("quant: probing prebuilt UQFF at `{uqff_repo}`");

    let token = mistralrs_core_token(token_source);
    let files = probe_hf_repo_files(&uqff_repo, "main", token);

    let Some(files) = files else {
        info!("quant: no UQFF repo at `{uqff_repo}` (or unreachable); using ISQ {raw}");
        return Ok(ResolvedQuant {
            from_uqff: None,
            in_situ_quant: Some(raw.to_string()),
        });
    };

    match mistralrs_core::resolve_uqff_shorthand(raw, &files) {
        Some(matched) => {
            let isq_name = parse_uqff_isq(&matched).unwrap_or_else(|| matched.clone());
            info!("quant: using prebuilt `{uqff_repo}` (shard `{matched}`)");
            Ok(ResolvedQuant {
                from_uqff: Some(format!("{uqff_repo}::{isq_name}")),
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
                from_uqff: None,
                in_situ_quant: Some(raw.to_string()),
            })
        }
    }
}

/// `"q4k-0.uqff"` -> `Some("q4k")`. Returns `None` for non-sharded names.
fn parse_uqff_isq(filename: &str) -> Option<String> {
    mistralrs_core::parse_uqff_shard(filename).map(|(name, _)| name)
}

fn mistralrs_core_token(source: &TokenSource) -> Option<String> {
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
