//! Best-effort cgroup v2 limits.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::SandboxPolicy;

const ROOT: &str = "/sys/fs/cgroup";

pub(crate) fn create_scope(policy: &SandboxPolicy) -> Option<PathBuf> {
    let controllers = fs::read_to_string(format!("{ROOT}/cgroup.controllers")).ok()?;
    if !controllers.contains("memory") || !controllers.contains("pids") {
        return None;
    }

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

pub(crate) fn write_pid(scope: &Path, pid: u32) -> std::io::Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .open(scope.join("cgroup.procs"))?;
    writeln!(f, "{pid}")?;
    Ok(())
}

pub(crate) fn remove_scope(scope: &Path) {
    let _ = fs::remove_dir(scope);
}
