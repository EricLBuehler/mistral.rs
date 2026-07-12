use candle_core::{Device, DeviceLocation};
use tracing_subscriber::EnvFilter;

use crate::DEBUG;

static LOGGER: std::sync::OnceLock<()> = std::sync::OnceLock::new();

pub const MISTRALRS_LOG_TARGET_PREFIX: &str = "mistralrs";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LogVerbosity {
    Info,
    Debug,
    Trace,
}

impl LogVerbosity {
    pub fn from_count(count: u8) -> Self {
        match count {
            0 => Self::Info,
            1 => Self::Debug,
            _ => Self::Trace,
        }
    }
}

/// This should be called to initialize the debug flag and logging.
/// This should not be called in mistralrs-core code due to Rust usage.
pub fn initialize_logging() {
    let verbosity = if is_debug_env() {
        LogVerbosity::Debug
    } else {
        LogVerbosity::Info
    };
    initialize_mistralrs_logging(verbosity);
}

pub fn initialize_mistralrs_logging(verbosity: LogVerbosity) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| default_mistralrs_filter(verbosity));
    initialize_logging_with_filter(filter);
}

pub fn initialize_logging_with_filter(filter: EnvFilter) {
    DEBUG.store(is_debug_env(), std::sync::atomic::Ordering::Relaxed);
    LOGGER.get_or_init(|| {
        let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
    });
}

pub fn default_mistralrs_filter(verbosity: LogVerbosity) -> EnvFilter {
    let (base, level) = match verbosity {
        LogVerbosity::Info => ("warn", "info"),
        LogVerbosity::Debug => ("warn", "debug"),
        LogVerbosity::Trace => ("warn,hf_hub=info", "trace"),
    };
    EnvFilter::new(base).add_directive(
        format!("{MISTRALRS_LOG_TARGET_PREFIX}={level}")
            .parse()
            .expect("valid default log directive"),
    )
}

fn is_debug_env() -> bool {
    std::env::var("MISTRALRS_DEBUG")
        .unwrap_or_default()
        .contains('1')
}

pub(crate) trait DeviceRepr {
    fn device_pretty_repr(&self) -> String;
}

impl DeviceRepr for Device {
    fn device_pretty_repr(&self) -> String {
        match self.location() {
            DeviceLocation::Cpu => "cpu".to_string(),
            DeviceLocation::Cuda { gpu_id } => format!("cuda[{gpu_id}]"),
            DeviceLocation::Metal { gpu_id } => format!("metal[{gpu_id}]"),
        }
    }
}
