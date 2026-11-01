//! Sandbox configuration options.

use clap::{Args, ValueEnum};
use mistralrs_sandbox::{NetworkMode, SandboxProfile};
use serde::Deserialize;

#[derive(Args, Clone, Deserialize)]
pub struct SandboxOptions {
    /// Sandbox for the processes started by the code and shell tools.
    #[arg(
        id = "sandbox_mode",
        long = "sandbox",
        value_name = "MODE",
        default_value = "auto",
        value_enum
    )]
    #[serde(default)]
    pub mode: SandboxMode,

    /// Files and network the sandbox allows. Defaults to `developer` when code or shell tools are on,
    /// otherwise `restricted`.
    #[arg(
        id = "sandbox_profile",
        long = "sandbox-profile",
        value_name = "PROFILE",
        value_enum
    )]
    #[serde(default)]
    pub profile: Option<SandboxProfileArg>,

    /// Per-session memory cap in MiB (default: 2048).
    #[arg(
        id = "sandbox_max_memory_mb",
        long = "sb-max-memory-mb",
        value_name = "MEMORY_MB"
    )]
    pub max_memory_mb: Option<u64>,

    /// Per-session CPU time cap in seconds (default: 600). Raised to at least enabled code/shell timeouts.
    #[arg(
        id = "sandbox_max_cpu_secs",
        long = "sb-max-cpu-secs",
        value_name = "CPU_SECS"
    )]
    pub max_cpu_secs: Option<u64>,

    /// Per-session process/thread cap (default: 64).
    #[arg(id = "sandbox_max_procs", long = "sb-max-procs", value_name = "PROCS")]
    pub max_procs: Option<u32>,

    /// Network access inside the sandbox. Defaults to `loopback` for the restricted profile and `full`
    /// for developer.
    #[arg(
        id = "sandbox_network",
        long = "sandbox-network",
        value_name = "NETWORK",
        value_enum
    )]
    #[serde(default)]
    pub network: Option<SandboxNetworkMode>,
}

impl Default for SandboxOptions {
    fn default() -> Self {
        Self {
            mode: SandboxMode::Auto,
            profile: None,
            max_memory_mb: None,
            max_cpu_secs: None,
            max_procs: None,
            network: None,
        }
    }
}

#[derive(Clone, Copy, ValueEnum, Default, PartialEq, Eq, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxMode {
    /// Sandbox on Linux and macOS, skipping protections the OS lacks; no sandbox on Windows.
    #[default]
    Auto,
    /// Always sandbox; fail if any protection is unavailable.
    On,
    /// No sandbox.
    Off,
}

#[derive(Clone, Copy, ValueEnum, Default, PartialEq, Eq, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxNetworkMode {
    /// No network access.
    None,
    /// Localhost only.
    #[default]
    Loopback,
    /// Unrestricted network access.
    Full,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
pub enum SandboxProfileArg {
    /// Session directory and system files only.
    Restricted,
    /// Also reads installed toolchains (PATH, Homebrew, nvm, conda, and similar) with full network.
    Developer,
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

impl From<SandboxProfileArg> for SandboxProfile {
    fn from(profile: SandboxProfileArg) -> Self {
        match profile {
            SandboxProfileArg::Restricted => SandboxProfile::Restricted,
            SandboxProfileArg::Developer => SandboxProfile::Developer,
        }
    }
}
