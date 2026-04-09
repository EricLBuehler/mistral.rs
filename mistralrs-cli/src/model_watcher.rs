//! Filesystem watcher for model directory changes.
//! Uses polling-based scanning for portability.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, warn};

use crate::commands::serve::discovered_to_pending_config;
use crate::model_scanner::scan_models_dir;

/// Spawns a background task that watches the models directory for changes.
/// Re-scans every 10 seconds and registers/unregisters pending models.
pub fn spawn_model_watcher(
    mistralrs: Arc<mistralrs_core::MistralRs>,
    models_dir: PathBuf,
    device: candle_core::Device,
    paged_attn_config: Option<mistralrs_core::PagedAttentionConfig>,
) {
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_secs(5)).await;

        let mut last_scan: Vec<String> = scan_models_dir(&models_dir)
            .iter()
            .map(|m| m.name.clone())
            .collect();

        info!("Model watcher started for: {}", models_dir.display());
        info!("Watching {} models", last_scan.len());

        loop {
            tokio::time::sleep(Duration::from_secs(10)).await;

            let current = scan_models_dir(&models_dir);
            let current_names: Vec<String> = current.iter().map(|m| m.name.clone()).collect();

            for old_name in &last_scan {
                if !current_names.contains(old_name) {
                    info!("Model directory removed: {}", old_name);
                    if let Err(e) = mistralrs.unregister_pending_model(old_name) {
                        warn!("Failed to unregister pending model '{}': {}", old_name, e);
                    }
                }
            }

            for model in &current {
                if !last_scan.contains(&model.name) {
                    info!("Model directory added: {}", model.name);
                    match discovered_to_pending_config(model, &device, paged_attn_config.clone()) {
                        Ok(config) => {
                            if let Err(e) = mistralrs.register_pending_model(&model.name, config) {
                                warn!("Failed to register pending model '{}': {}", model.name, e);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to build config for '{}': {}", model.name, e);
                        }
                    }
                }
            }

            if last_scan.len() != current_names.len() {
                info!(
                    "Model directory changed: {} -> {} models",
                    last_scan.len(),
                    current_names.len()
                );
            }

            last_scan = current_names;
        }
    });
}
