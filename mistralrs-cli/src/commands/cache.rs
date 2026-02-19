//! HuggingFace cache management commands

use anyhow::Result;
use comfy_table::{presets::UTF8_FULL, Cell, ContentArrangement, Table};
use std::path::{Path, PathBuf};

/// Get the HuggingFace hub directory
fn hf_hub_dir() -> Result<PathBuf> {
    mistralrs_core::hf_hub_cache_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine Hugging Face hub cache directory"))
}

/// List all cached models
pub fn run_cache_list() -> Result<()> {
    let hub_dir = hf_hub_dir()?;

    println!();
    println!("HuggingFace Model Cache");
    println!("-----------------------");
    println!();

    if !hub_dir.exists() {
        println!("No cached models found.");
        println!();
        println!("Cache directory: {}", hub_dir.display());
        return Ok(());
    }

    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("Model"),
            Cell::new("Size"),
            Cell::new("Last Used"),
        ]);

    let mut total_size = 0u64;
    let mut model_count = 0;

    let mut models: Vec<_> = std::fs::read_dir(&hub_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("models--"))
        .collect();

    // Sort by name for consistent output
    models.sort_by_key(|a| a.file_name());

    for entry in models {
        let name = entry.file_name().to_string_lossy().to_string();
        // Convert "models--org--name" to "org/name"
        let model_id = name.trim_start_matches("models--").replace("--", "/");
        let size = dir_size(&entry.path()).unwrap_or(0);
        let last_used = get_last_modified(&entry.path()).unwrap_or_else(|_| "unknown".to_string());

        table.add_row(vec![
            Cell::new(&model_id),
            Cell::new(format_size(size)),
            Cell::new(&last_used),
        ]);
        total_size += size;
        model_count += 1;
    }

    if model_count == 0 {
        println!("No cached models found.");
    } else {
        println!("{table}");
        println!();
        println!(
            "Total: {} model{}, {}",
            model_count,
            if model_count == 1 { "" } else { "s" },
            format_size(total_size)
        );
    }
    println!();
    println!("Cache directory: {}", hub_dir.display());
    println!();

    Ok(())
}

/// Delete a specific model from cache
pub fn run_cache_delete(model_id: &str) -> Result<()> {
    let hub_dir = hf_hub_dir()?;
    // Convert "org/name" to "models--org--name"
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let model_path = hub_dir.join(&dir_name);

    if !model_path.exists() {
        anyhow::bail!(
            "Model '{}' not found in cache.\nRun 'mistralrs cache list' to see cached models.",
            model_id
        );
    }

    let size = dir_size(&model_path).unwrap_or(0);
    std::fs::remove_dir_all(&model_path)?;
    println!("✅ Deleted '{}' ({})", model_id, format_size(size));
    Ok(())
}

/// Calculate total size of a directory recursively
fn dir_size(path: &Path) -> Result<u64> {
    let mut size = 0;
    for entry in walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            size += entry.metadata().map(|m| m.len()).unwrap_or(0);
        }
    }
    Ok(size)
}

/// Get human-readable last modified time
fn get_last_modified(path: &Path) -> Result<String> {
    let metadata = std::fs::metadata(path)?;
    let modified = metadata.modified()?;
    let duration = modified.elapsed().unwrap_or_default();
    let days = duration.as_secs() / 86400;

    Ok(if days == 0 {
        "today".to_string()
    } else if days == 1 {
        "yesterday".to_string()
    } else if days < 7 {
        format!("{} days ago", days)
    } else if days < 30 {
        format!("{} weeks ago", days / 7)
    } else if days < 365 {
        format!("{} months ago", days / 30)
    } else {
        format!("{} years ago", days / 365)
    })
}

/// Format bytes as human-readable size
fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1e6)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1e3)
    } else {
        format!("{} B", bytes)
    }
}
