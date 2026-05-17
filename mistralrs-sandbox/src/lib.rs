//! OS-level sandbox for subprocesses spawned on behalf of LLMs.
//!
//! Linux uses env scrub, namespaces, Landlock, rlimits, seccomp, and optional
//! cgroup v2 limits. macOS uses env scrub and Seatbelt.
//!
//! ```ignore
//! use mistralrs_sandbox::{detect, SandboxPolicy};
//! let sandbox = detect();
//! let policy = SandboxPolicy::default();
//! sandbox.harden(&mut cmd, &policy)?;
//! ```

mod null;

#[cfg(any(target_os = "linux", target_os = "macos"))]
mod env_scrub;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "macos")]
mod macos;

#[cfg(all(test, not(target_os = "macos")))]
#[path = "macos/profile.rs"]
mod macos_profile;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use null::NullSandbox;

/// Environment variable that overrides the configured sandbox mode at runtime.
/// Accepted values: `auto`, `on`, `off`. Case-insensitive.
pub const SANDBOX_ENV_VAR: &str = "MISTRALRS_SANDBOX";
pub const DEFAULT_MAX_MEMORY_MB: u64 = 2048;
pub const DEFAULT_MAX_CPU_SECS: u64 = 300;
pub const DEFAULT_MAX_PROCS: u32 = 64;
pub const DEFAULT_MAX_OPEN_FDS: u32 = 1024;
pub const DEFAULT_MAX_FILE_SZ_MB: u64 = 256;

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

/// Policy applied to a sandboxed process.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SandboxPolicy {
    /// Address-space cap in MB where resource limits are supported.
    pub max_memory_mb: u64,
    /// CPU-time cap in seconds where resource limits are supported.
    pub max_cpu_secs: u64,
    /// Child process cap where resource limits are supported.
    pub max_procs: u32,
    /// Open file descriptor cap where resource limits are supported.
    pub max_open_fds: u32,
    /// Maximum file size in MB where resource limits are supported.
    pub max_file_sz_mb: u64,
    /// Network access granted to sandboxed subprocesses.
    pub network: NetworkMode,
    /// Additional filesystem paths the sandboxed process may read.
    /// Appended to the built-in system allowlist.
    #[serde(default)]
    pub extra_fs_read: Vec<PathBuf>,
    /// Additional filesystem paths the sandboxed process may read and write.
    /// Appended to the per-session workdir.
    #[serde(default)]
    pub extra_fs_write: Vec<PathBuf>,
    /// Additional environment variable names allowed through the env scrub.
    /// Appended to the built-in allowlist (PATH, LANG, ...). Names only,
    /// values come from the parent process's environment.
    #[serde(default)]
    pub extra_env: Vec<String>,
    /// When true, missing requested layers become hard errors.
    #[serde(default)]
    pub strict: bool,
    /// Per-session writable directory. Filled in by the caller right before
    /// `harden()`.
    pub session_workdir: Option<PathBuf>,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            max_memory_mb: DEFAULT_MAX_MEMORY_MB,
            max_cpu_secs: DEFAULT_MAX_CPU_SECS,
            max_procs: DEFAULT_MAX_PROCS,
            max_open_fds: DEFAULT_MAX_OPEN_FDS,
            max_file_sz_mb: DEFAULT_MAX_FILE_SZ_MB,
            network: NetworkMode::Loopback,
            extra_fs_read: Vec::new(),
            extra_fs_write: Vec::new(),
            extra_env: Vec::new(),
            strict: false,
            session_workdir: None,
        }
    }
}

/// What a `Sandbox` can enforce for a given policy after OS feature detection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EffectiveProtection {
    /// Filesystem reads/writes outside the allowlist are denied at the OS
    /// level (Landlock on Linux, Seatbelt on macOS). False on NullSandbox
    /// or when Landlock is unavailable.
    pub fs_isolated: bool,
    /// Network access is restricted (`network=none` blocks socket(); netns
    /// scopes routing for `loopback`; Seatbelt denies non-local for macOS).
    /// False on NullSandbox or when network=Full.
    pub network_isolated: bool,
    /// Resource limits will be applied. On Linux this also means the seccomp
    /// deny-list will be installed. False on macOS and NullSandbox.
    pub rlimits_applied: bool,
}

impl EffectiveProtection {
    /// True if any layer is actually enforcing something.
    pub fn any(&self) -> bool {
        self.fs_isolated || self.network_isolated || self.rlimits_applied
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

    /// Probe the OS once and report what layers will actually fire for the
    /// given policy. Implementations should be cheap and side-effect-free.
    fn effective(&self, policy: &SandboxPolicy) -> EffectiveProtection;
}

/// Return the best sandbox implementation for the current platform.
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

/// Explicit no-op sandbox.
pub fn null() -> Box<dyn Sandbox> {
    static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    WARNED.get_or_init(|| {
        tracing::warn!(
            "sandbox disabled - model-generated code has full filesystem, network, and subprocess access"
        );
    });
    Box::new(NullSandbox)
}
