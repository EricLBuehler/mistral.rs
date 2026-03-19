//! Configuration types for parking lot scheduler.
//!
//! This module provides YAML-serializable configuration structures for the
//! parking-lot scheduler feature, supporting loading from files and merging
//! with CLI overrides.

use serde::{Deserialize, Serialize};

/// Parking lot scheduler configuration loaded from YAML.
///
/// This configuration can be loaded from:
/// - `~/.mistralrs-server/scheduler.yaml` (default location)
/// - Environment variable `MISTRALRS_SCHEDULER_CONFIG`
/// - CLI flag `--scheduler-config <path>`
///
/// # Example YAML
///
/// ```yaml
/// pool:
///   worker_threads: 4
///   thread_stack_size: 2097152  # 2MB
///
/// limits:
///   max_units: 4096
///   max_queue_depth: 100
///   timeout_secs: 300
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParkingLotSchedulerConfig {
    /// Thread pool configuration
    #[serde(default)]
    pub pool: PoolConfig,
    
    /// Resource limits and queue configuration
    #[serde(default)]
    pub limits: LimitsConfig,
}

/// Thread pool configuration.
///
/// Controls the number of worker threads and thread-specific settings.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoolConfig {
    /// Number of worker threads for inference (default: num_cpus).
    ///
    /// Each worker thread can process one inference request at a time.
    /// Recommended:
    /// - 4-8 for M1/M2 Macs
    /// - `num_cpus::get()` for multi-GPU setups
    pub worker_threads: Option<usize>,
    
    /// Stack size per worker thread in bytes (default: 2MB).
    ///
    /// This setting is only supported on native (non-WASM) platforms.
    /// Minimum recommended: 64KB (65536 bytes)
    /// Default: 2MB (2097152 bytes)
    pub thread_stack_size: Option<usize>,
}

/// Resource limits and queue configuration.
///
/// Controls resource allocation, queue depth, and timeouts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LimitsConfig {
    /// Maximum resource units (GPU VRAM blocks or KV cache blocks).
    ///
    /// Set based on GPU memory: roughly (GPU_MEM_MB / 16) for 16-token blocks.
    /// Examples:
    /// - 4096 units = ~4GB KV cache for small models
    /// - 16384 units = ~16GB KV cache for larger models
    ///
    /// Default: 16384 (~256K tokens with 16-token blocks)
    pub max_units: Option<u32>,
    
    /// Maximum number of queued requests before rejection.
    ///
    /// Requests beyond this limit will be rejected with a queue-full error.
    /// Lower for memory-constrained systems, higher for high-throughput.
    ///
    /// Default: 1000
    pub max_queue_depth: Option<usize>,
    
    /// Request timeout in seconds.
    ///
    /// Requests that don't complete within this time are cancelled.
    /// Increase for very long context or slow models.
    ///
    /// Default: 120 (2 minutes)
    pub timeout_secs: Option<u64>,
}

impl ParkingLotSchedulerConfig {
    /// Load configuration from a YAML file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the YAML configuration file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be read
    /// - YAML parsing fails
    /// - Configuration validation fails
    pub fn from_file(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;
        
        let config: Self = serde_yaml::from_str(&content)
            .map_err(|e| format!("Failed to parse YAML: {}", e))?;
        
        config.validate()?;
        
        Ok(config)
    }
    
    /// Validate configuration values are within reasonable ranges.
    ///
    /// # Errors
    ///
    /// Returns an error if any configuration value is invalid.
    pub fn validate(&self) -> Result<(), String> {
        // Validate worker threads
        if let Some(threads) = self.pool.worker_threads {
            if threads == 0 {
                return Err("worker_threads must be > 0".into());
            }
            if threads > 1024 {
                return Err(format!(
                    "worker_threads={} is too high (max 1024)",
                    threads
                ));
            }
        }
        
        // Validate thread stack size
        if let Some(stack_size) = self.pool.thread_stack_size {
            const MIN_STACK_SIZE: usize = 64 * 1024; // 64KB minimum
            if stack_size < MIN_STACK_SIZE {
                return Err(format!(
                    "thread_stack_size={} is too small (min {} bytes / 64KB)",
                    stack_size, MIN_STACK_SIZE
                ));
            }
        }
        
        // Validate max_units
        if let Some(units) = self.limits.max_units {
            if units == 0 {
                return Err("max_units must be > 0".into());
            }
        }
        
        // Validate queue depth
        if let Some(depth) = self.limits.max_queue_depth {
            if depth == 0 {
                return Err("max_queue_depth must be > 0".into());
            }
            if depth > 100_000 {
                return Err(format!(
                    "max_queue_depth={} is too high (max 100,000)",
                    depth
                ));
            }
        }
        
        // Validate timeout
        if let Some(timeout) = self.limits.timeout_secs {
            if timeout == 0 {
                return Err("timeout_secs must be > 0".into());
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_yaml_config() {
        let yaml = r#"
pool:
  worker_threads: 8
  thread_stack_size: 2097152
limits:
  max_units: 2048
  max_queue_depth: 50
  timeout_secs: 60
"#;
        let config: ParkingLotSchedulerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.pool.worker_threads, Some(8));
        assert_eq!(config.pool.thread_stack_size, Some(2097152));
        assert_eq!(config.limits.max_units, Some(2048));
        assert_eq!(config.limits.max_queue_depth, Some(50));
        assert_eq!(config.limits.timeout_secs, Some(60));
    }
    
    #[test]
    fn test_default_config() {
        let config = ParkingLotSchedulerConfig::default();
        assert!(config.pool.worker_threads.is_none());
        assert!(config.limits.max_units.is_none());
        config.validate().unwrap();
    }
    
    #[test]
    fn test_validation_zero_workers() {
        let mut config = ParkingLotSchedulerConfig::default();
        config.pool.worker_threads = Some(0);
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_validation_zero_max_units() {
        let mut config = ParkingLotSchedulerConfig::default();
        config.limits.max_units = Some(0);
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_validation_small_stack_size() {
        let mut config = ParkingLotSchedulerConfig::default();
        config.pool.thread_stack_size = Some(1024); // Too small
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_validation_valid_config() {
        let mut config = ParkingLotSchedulerConfig::default();
        config.pool.worker_threads = Some(4);
        config.pool.thread_stack_size = Some(2 * 1024 * 1024);
        config.limits.max_units = Some(4096);
        config.limits.max_queue_depth = Some(100);
        config.limits.timeout_secs = Some(300);
        assert!(config.validate().is_ok());
    }
}
