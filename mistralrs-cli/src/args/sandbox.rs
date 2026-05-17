//! Sandbox configuration options.
//!
//! Mirrors the shape of [`crate::args::PagedAttentionOptions`]: one tri-state
//! `mode` plus a handful of optional numeric overrides, exposed identically
//! via clap (CLI) and serde (TOML).

use clap::{Args, ValueEnum};
use mistralrs_sandbox::NetworkMode;
use serde::Deserialize;

/// Sandbox configuration applied to model-spawned subprocesses (currently:
/// the Python code-execution session). Defaults to `auto` (enabled on Linux
/// and macOS, no-op elsewhere).
#[derive(Args, Clone, Deserialize)]
pub struct SandboxOptions {
    /// Sandbox mode.
    /// - auto: enabled on Linux/macOS, no-op on other platforms (default)
    /// - on: force enable (warn if no platform impl)
    /// - off: disable, model-generated code has full host access
    #[arg(id = "sandbox_mode", long = "sandbox", default_value = "auto", value_enum)]
    #[serde(default)]
    pub mode: SandboxMode,

    /// Per-session memory cap in MiB (default: 2048).
    #[arg(id = "sandbox_max_memory_mb", long = "sb-max-memory-mb")]
    pub max_memory_mb: Option<u64>,

    /// Per-session CPU time cap in seconds (default: 300).
    #[arg(id = "sandbox_max_cpu_secs", long = "sb-max-cpu-secs")]
    pub max_cpu_secs: Option<u64>,

    /// Per-session process/thread cap (default: 64).
    #[arg(id = "sandbox_max_procs", long = "sb-max-procs")]
    pub max_procs: Option<u32>,

    /// Network access permitted to the sandboxed session.
    /// - none: no sockets at all
    /// - loopback: 127.0.0.1 only (default)
    /// - full: unrestricted
    #[arg(id = "sandbox_network", long = "sandbox-network", default_value = "loopback", value_enum)]
    #[serde(default)]
    pub network: SandboxNetworkMode,
}

impl Default for SandboxOptions {
    fn default() -> Self {
        Self {
            mode: SandboxMode::Auto,
            max_memory_mb: None,
            max_cpu_secs: None,
            max_procs: None,
            network: SandboxNetworkMode::Loopback,
        }
    }
}

/// Sandbox operation mode (CLI/TOML representation).
#[derive(Clone, Copy, ValueEnum, Default, PartialEq, Eq, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxMode {
    /// Enabled on Linux/macOS, no-op on Windows.
    #[default]
    Auto,
    /// Force enable.
    On,
    /// Force disable.
    Off,
}

/// CLI/TOML wrapper over [`mistralrs_sandbox::NetworkMode`].
/// Wrapped so clap's `ValueEnum` derive can target it without modifying the
/// sandbox crate's public types.
#[derive(Clone, Copy, ValueEnum, Default, PartialEq, Eq, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxNetworkMode {
    None,
    #[default]
    Loopback,
    Full,
}

impl From<SandboxNetworkMode> for NetworkMode {
    fn from(m: SandboxNetworkMode) -> Self {
        match m {
            SandboxNetworkMode::None => NetworkMode::None,
            SandboxNetworkMode::Loopback => NetworkMode::Loopback,
            SandboxNetworkMode::Full => NetworkMode::Full,
        }
    }
}

