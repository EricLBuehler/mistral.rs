//! Scheduler configuration loading and management.
//!
//! This module provides functionality for loading parking-lot scheduler
//! configuration from YAML files with a priority chain:
//! 1. CLI --scheduler-config flag
//! 2. MISTRALRS_SCHEDULER_CONFIG environment variable
//! 3. ~/.mistralrs-server/scheduler.yaml (default location)
//! 4. Built-in defaults

use anyhow::Result;
use mistralrs_core::parking_lot::ParkingLotSchedulerConfig;
use std::path::PathBuf;
use tracing::{info, warn};

/// CLI scheduler configuration overrides.
///
/// These values take precedence over YAML configuration when specified.
pub struct CliSchedulerConfig {
    pub worker_threads: Option<usize>,
    pub thread_stack_size: Option<usize>,
    pub max_units: Option<u32>,
    pub max_queue_depth: Option<usize>,
    pub timeout_secs: Option<u64>,
}

/// Load scheduler configuration with priority chain.
///
/// Priority order:
/// 1. CLI --scheduler-config flag
/// 2. MISTRALRS_SCHEDULER_CONFIG env var
/// 3. ~/.mistralrs-server/scheduler.yaml
/// 4. None (use defaults)
///
/// # Arguments
///
/// * `cli_config_path` - Optional path from CLI flag
///
/// # Returns
///
/// * `Ok(Some(config))` - Configuration loaded successfully
/// * `Ok(None)` - No configuration file found (will use defaults)
/// * `Err(e)` - Configuration file exists but failed to load/parse
///
/// # Errors
///
/// Returns an error if a configuration file is specified but cannot be
/// loaded or parsed, or if validation fails.
pub fn load_scheduler_config(
    cli_config_path: Option<&str>,
) -> Result<Option<ParkingLotSchedulerConfig>> {
    let config_path = resolve_config_path(cli_config_path)?;

    if let Some(path) = config_path {
        info!("📋 Loading scheduler configuration from: {}", path.display());

        match ParkingLotSchedulerConfig::from_file(path.to_str().unwrap()) {
            Ok(config) => {
                info!("✅ Scheduler configuration loaded successfully");
                Ok(Some(config))
            }
            Err(e) => {
                anyhow::bail!("Failed to load scheduler config: {}", e);
            }
        }
    } else {
        info!("ℹ️  No scheduler config file found, using defaults");
        Ok(None)
    }
}

/// Resolve config file path with priority chain.
///
/// # Arguments
///
/// * `cli_path` - Optional path from CLI --scheduler-config flag
///
/// # Returns
///
/// Returns the first existing configuration file found in the priority chain,
/// or `None` if no configuration file is found.
///
/// # Errors
///
/// Returns an error if the CLI path is specified but does not exist.
fn resolve_config_path(cli_path: Option<&str>) -> Result<Option<PathBuf>> {
    // Priority 1: CLI flag
    if let Some(path) = cli_path {
        let p = PathBuf::from(path);
        if !p.exists() {
            anyhow::bail!("Scheduler config file not found: {}", path);
        }
        info!("📌 Using scheduler config from CLI: {}", path);
        return Ok(Some(p));
    }

    // Priority 2: Environment variable
    if let Ok(env_path) = std::env::var("MISTRALRS_SCHEDULER_CONFIG") {
        let p = PathBuf::from(&env_path);
        if p.exists() {
            info!(
                "📌 Using scheduler config from MISTRALRS_SCHEDULER_CONFIG: {}",
                env_path
            );
            return Ok(Some(p));
        } else {
            warn!(
                "⚠️  MISTRALRS_SCHEDULER_CONFIG points to non-existent file: {}",
                env_path
            );
        }
    }

    // Priority 3: Default location ~/.mistralrs-server/scheduler.yaml
    if let Some(user_dirs) = directories::UserDirs::new() {
        let default_path = user_dirs
            .home_dir()
            .join(".mistralrs-server")
            .join("scheduler.yaml");

        if default_path.exists() {
            info!(
                "📌 Using default scheduler config: {}",
                default_path.display()
            );
            return Ok(Some(default_path));
        }
    }

    Ok(None)
}

/// Merge YAML config with CLI overrides.
///
/// CLI values take precedence over YAML values. If no YAML config is provided,
/// creates a default config with CLI overrides applied.
///
/// # Arguments
///
/// * `yaml_config` - Optional configuration loaded from YAML
/// * `cli` - CLI override values
///
/// # Returns
///
/// A merged configuration with CLI overrides applied.
pub fn merge_configs(
    yaml_config: Option<ParkingLotSchedulerConfig>,
    cli: CliSchedulerConfig,
) -> ParkingLotSchedulerConfig {
    let mut config = yaml_config.unwrap_or_default();

    // CLI overrides take precedence
    if cli.worker_threads.is_some() {
        config.pool.worker_threads = cli.worker_threads;
        info!(
            "🔧 CLI override: worker_threads = {}",
            cli.worker_threads.unwrap()
        );
    }
    if cli.thread_stack_size.is_some() {
        config.pool.thread_stack_size = cli.thread_stack_size;
        info!(
            "🔧 CLI override: thread_stack_size = {} bytes",
            cli.thread_stack_size.unwrap()
        );
    }
    if cli.max_units.is_some() {
        config.limits.max_units = cli.max_units;
        info!("🔧 CLI override: max_units = {}", cli.max_units.unwrap());
    }
    if cli.max_queue_depth.is_some() {
        config.limits.max_queue_depth = cli.max_queue_depth;
        info!(
            "🔧 CLI override: max_queue_depth = {}",
            cli.max_queue_depth.unwrap()
        );
    }
    if cli.timeout_secs.is_some() {
        config.limits.timeout_secs = cli.timeout_secs;
        info!(
            "🔧 CLI override: timeout_secs = {}",
            cli.timeout_secs.unwrap()
        );
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_configs_with_cli_overrides() {
        let mut yaml_config = ParkingLotSchedulerConfig::default();
        yaml_config.pool.worker_threads = Some(4);
        yaml_config.limits.max_units = Some(2048);

        let cli = CliSchedulerConfig {
            worker_threads: Some(8), // Override
            thread_stack_size: None,
            max_units: None, // Don't override
            max_queue_depth: Some(50),
            timeout_secs: None,
        };

        let merged = merge_configs(Some(yaml_config), cli);

        assert_eq!(merged.pool.worker_threads, Some(8)); // CLI override
        assert_eq!(merged.limits.max_units, Some(2048)); // YAML value kept
        assert_eq!(merged.limits.max_queue_depth, Some(50)); // CLI override
    }

    #[test]
    fn test_merge_configs_no_yaml() {
        let cli = CliSchedulerConfig {
            worker_threads: Some(8),
            thread_stack_size: Some(2097152),
            max_units: Some(4096),
            max_queue_depth: Some(100),
            timeout_secs: Some(300),
        };

        let merged = merge_configs(None, cli);

        assert_eq!(merged.pool.worker_threads, Some(8));
        assert_eq!(merged.pool.thread_stack_size, Some(2097152));
        assert_eq!(merged.limits.max_units, Some(4096));
        assert_eq!(merged.limits.max_queue_depth, Some(100));
        assert_eq!(merged.limits.timeout_secs, Some(300));
    }

    #[test]
    fn test_resolve_config_path_nonexistent_cli() {
        let result = resolve_config_path(Some("/nonexistent/config.yaml"));
        assert!(result.is_err());
    }
}
