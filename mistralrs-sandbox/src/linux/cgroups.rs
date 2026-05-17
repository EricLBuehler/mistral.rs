//! cgroup v2 best-effort attach. Belt-and-suspenders for `RLIMIT_AS`:
//! `memory.max` kills the cgroup on OOM, while `RLIMIT_AS` only fails malloc
//! (and leaves Python in a half-broken state).
//!
//! Silently disabled when cgroup v2 isn't delegated to the calling user.
//! This is the common case in non-systemd setups, inside containers, and
//! on most CI runners.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::SandboxPolicy;

const ROOT: &str = "/sys/fs/cgroup";

/// Create a new scope for this session and write the memory/pid caps.
/// Returns the scope's path so `write_pid` can drop the child PID into it.
pub(crate) fn create_scope(policy: &SandboxPolicy) -> Option<PathBuf> {
    let controllers = fs::read_to_string(format!("{ROOT}/cgroup.controllers")).ok()?;
    if !controllers.contains("memory") || !controllers.contains("pids") {
        return None;
    }

    // Pick an instance id from the current pid + a counter so concurrent
    // managers don't collide. The kernel rejects creating a duplicate scope.
    let id = format!(
        "mistralrs-sandbox-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let scope = PathBuf::from(format!("{ROOT}/{id}"));

    if let Err(e) = fs::create_dir(&scope) {
        tracing::debug!("cgroup v2 scope create failed at {scope:?}: {e}");
        return None;
    }

    let memory_bytes = policy.max_memory_mb.saturating_mul(1024 * 1024);
    if let Err(e) = fs::write(scope.join("memory.max"), memory_bytes.to_string()) {
        tracing::debug!("cgroup memory.max write failed: {e}");
    }
    if let Err(e) = fs::write(scope.join("pids.max"), policy.max_procs.to_string()) {
        tracing::debug!("cgroup pids.max write failed: {e}");
    }

    Some(scope)
}

/// Move a freshly-spawned PID into the scope. Returns Ok even on benign
/// failures (cgroup may already be torn down if the child exited fast).
pub(crate) fn write_pid(scope: &Path, pid: u32) -> std::io::Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .open(scope.join("cgroup.procs"))?;
    writeln!(f, "{pid}")?;
    Ok(())
}

/// Remove the scope. Safe to call after the child exits - `rmdir` succeeds
/// once the cgroup is empty.
pub(crate) fn remove_scope(scope: &Path) {
    let _ = fs::remove_dir(scope);
}
