//! Sandbox configuration options.

use clap::{Args, ValueEnum};
use mistralrs_sandbox::NetworkMode;
use serde::Deserialize;

#[derive(Args, Clone, Deserialize)]
pub struct SandboxOptions {
    /// Sandbox mode.
    #[arg(
        id = "sandbox_mode",
        long = "sandbox",
        default_value = "auto",
        value_enum
    )]
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
    #[arg(
        id = "sandbox_network",
        long = "sandbox-network",
        default_value = "loopback",
        value_enum
    )]
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
