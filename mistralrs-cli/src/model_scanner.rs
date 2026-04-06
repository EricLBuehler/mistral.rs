//! Model directory scanner for auto-discovery.

use std::path::Path;
use tracing::{info, warn};

/// Detected model file format
#[derive(Debug, Clone, PartialEq)]
pub enum ModelFileFormat {
    /// GGUF quantized file
    Gguf { filename: String },
    /// GGML quantized file
    Ggml { filename: String },
    /// Plain/safetensors model (directory contains config.json or model.safetensors)
    Plain,
}

/// A discovered model from disk scanning
#[derive(Debug, Clone)]
pub struct DiscoveredModel {
    /// Directory name (used as model ID)
    pub name: String,
    /// Full path to the model directory
    pub path: std::path::PathBuf,
    /// Detected file format
    pub format: ModelFileFormat,
}

/// Scan a models directory for model subdirectories.
/// Returns a list of discovered models sorted by name.
pub fn scan_models_dir(models_dir: &Path) -> Vec<DiscoveredModel> {
    let mut models = Vec::new();

    let entries = match std::fs::read_dir(models_dir) {
        Ok(entries) => entries,
        Err(e) => {
            warn!(
                "Failed to read models directory {}: {}",
                models_dir.display(),
                e
            );
            return models;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };

        if name.starts_with('.') {
            continue;
        }

        if let Some(fmt) = detect_model_format(&path) {
            info!("Discovered model: {} (format: {:?})", name, fmt);
            models.push(DiscoveredModel {
                name,
                path,
                format: fmt,
            });
        } else {
            warn!(
                "Skipping directory '{}' — no recognized model files found",
                name
            );
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}

/// Detect the model format from files in a directory.
/// Returns None if no model files are found.
fn detect_model_format(dir: &Path) -> Option<ModelFileFormat> {
    let entries = std::fs::read_dir(dir).ok()?;

    let mut has_gguf: Option<String> = None;
    let mut has_ggml: Option<String> = None;
    let mut has_config = false;
    let mut has_safetensors = false;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let filename = match path.file_name().and_then(|n| n.to_str()) {
            Some(f) => f.to_lowercase(),
            None => continue,
        };

        if filename.ends_with(".gguf") {
            if has_gguf.is_none() {
                has_gguf = Some(path.file_name().unwrap().to_string_lossy().to_string());
            }
        } else if filename.ends_with(".ggml") {
            if has_ggml.is_none() {
                has_ggml = Some(path.file_name().unwrap().to_string_lossy().to_string());
            }
        } else if filename == "config.json" {
            has_config = true;
        } else if filename.ends_with(".safetensors") || filename == "model.safetensors.index.json" {
            has_safetensors = true;
        }
    }

    if let Some(filename) = has_gguf {
        Some(ModelFileFormat::Gguf { filename })
    } else if let Some(filename) = has_ggml {
        Some(ModelFileFormat::Ggml { filename })
    } else if has_config || has_safetensors {
        Some(ModelFileFormat::Plain)
    } else {
        None
    }
}
