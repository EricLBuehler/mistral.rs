//! OS-level sandbox for subprocesses spawned on behalf of LLMs.
//!
//! The threat model is misbehavior by model-generated code (fs damage, secret
//! exfiltration, fork/memory bombs, subprocess pivots), not determined kernel
//! exploits. Defenses are layered: rlimits + seccomp + env scrub on every
//! supported OS, with namespaces and mount layout added on Linux.
//!
//! ```ignore
//! use mistralrs_sandbox::{detect, SandboxPolicy};
//! let sandbox = detect();
//! let policy = SandboxPolicy::default();
//! sandbox.harden(&mut cmd, &policy)?;
//! ```

mod null;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "macos")]
mod macos;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use null::NullSandbox;

/// Environment variable that overrides the configured sandbox mode at runtime.
/// Accepted values: `auto`, `on`, `off`. Case-insensitive.
pub const SANDBOX_ENV_VAR: &str = "MISTRALRS_SANDBOX";

#[derive(Debug, Error)]
pub enum SandboxError {
    #[error("sandbox setup failed: {0}")]
    Setup(String),
    #[error("sandbox not supported on this platform")]
    Unsupported,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Network access permitted to sandboxed processes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NetworkMode {
    /// No sockets at all (denied at syscall level on Linux).
    None,
    /// Loopback only. Default - many libs probe 127.0.0.1 on import.
    #[default]
    Loopback,
    /// Unrestricted - matches pre-sandbox behavior.
    Full,
}

/// Policy applied to a sandboxed process. Defaults are tuned for
/// model-generated Python code execution; tighten via the CLI/TOML overrides.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SandboxPolicy {
    pub max_memory_mb: u64,
    pub max_cpu_secs: u64,
    pub max_procs: u32,
    pub max_open_fds: u32,
    pub max_file_sz_mb: u64,
    pub network: NetworkMode,
    /// Per-session writable directory. Filled in by the caller right before
    /// `harden()`. Bind-mounted to `/work` on Linux when namespaces are used.
    pub session_workdir: Option<PathBuf>,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            max_memory_mb: 2048,
            max_cpu_secs: 300,
            max_procs: 64,
            max_open_fds: 1024,
            max_file_sz_mb: 256,
            network: NetworkMode::Loopback,
            session_workdir: None,
        }
    }
}

/// Applied to a `tokio::process::Command` before spawn and to the resulting
/// PID after spawn. Implementations are platform-specific.
pub trait Sandbox: Send + Sync {
    /// Harden the command. Called before `spawn()`. May install `pre_exec`
    /// hooks, clear env, set up wrapping argv, etc.
    fn harden(
        &self,
        cmd: &mut tokio::process::Command,
        policy: &SandboxPolicy,
    ) -> Result<(), SandboxError>;

    /// Post-spawn attachment. Called after `spawn()` returns. Used for
    /// cgroup membership writes on Linux; no-op elsewhere.
    fn attach(&self, _pid: u32, _policy: &SandboxPolicy) -> Result<(), SandboxError> {
        Ok(())
    }

    /// Human-readable name for logging.
    fn name(&self) -> &'static str;
}

/// Return the best sandbox implementation for the current platform.
/// On unsupported platforms returns [`NullSandbox`] with no isolation.
pub fn detect() -> Box<dyn Sandbox> {
    #[cfg(target_os = "linux")]
    {
        Box::new(linux::LinuxSandbox::new())
    }
    #[cfg(target_os = "macos")]
    {
        Box::new(macos::MacosSandbox::new())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        tracing::warn!(
            "no sandbox implementation for this platform - model-generated code has full host access"
        );
        Box::new(NullSandbox)
    }
}

/// Explicit no-op sandbox. Prints a warning on first call.
pub fn null() -> Box<dyn Sandbox> {
    static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    WARNED.get_or_init(|| {
        tracing::warn!(
            "sandbox disabled - model-generated code has full filesystem, network, and subprocess access"
        );
    });
    Box::new(NullSandbox)
}
