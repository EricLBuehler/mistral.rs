//! Linux sandbox composed of env scrub + namespaces + Landlock + rlimits +
//! seccomp + (optional) cgroup v2. Low-level work is delegated to crates:
//! [`landlock`] for the FS LSM, [`nix`] for `unshare`/`setrlimit`/fd ops,
//! [`seccompiler`] for the BPF deny-list, [`itoa`] for the one stack-only
//! formatter we still need.
//!
//! Order inside `pre_exec` is intentional:
//!   1. namespaces + Landlock (need fs/socket syscalls available)
//!   2. setrlimit (still needs unrestricted ability to call)
//!   3. seccomp install (deny-list goes live, including `unshare`/`mount`)
//!
//! If you add new pre_exec steps, do it before step 3 unless they're
//! explicitly in the seccomp allowlist.

mod cgroups;
mod env;
mod namespaces;
mod rlimits;
mod seccomp;

use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use nix::sys::resource::Resource;
use seccompiler::BpfProgram;

use crate::{Sandbox, SandboxError, SandboxPolicy};

pub struct LinuxSandbox {
    /// Track cgroup scopes we've created so they can be cleaned up. Keyed by
    /// PID since `attach()` is the only point where we know which scope went
    /// with which child.
    scopes: Mutex<std::collections::HashMap<u32, PathBuf>>,
    /// Pending scope created in `harden()`, consumed by the next `attach()`.
    /// Single-slot because spawns are serialized per-session.
    pending_scope: Mutex<Option<PathBuf>>,
}

impl LinuxSandbox {
    pub fn new() -> Self {
        Self {
            scopes: Mutex::new(std::collections::HashMap::new()),
            pending_scope: Mutex::new(None),
        }
    }
}

impl Default for LinuxSandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Sandbox for LinuxSandbox {
    fn harden(
        &self,
        cmd: &mut tokio::process::Command,
        policy: &SandboxPolicy,
    ) -> Result<(), SandboxError> {
        env::apply(cmd, policy);

        // cgroup scope creation is best-effort; failures fall through to
        // rlimit-only enforcement.
        if let Ok(mut slot) = self.pending_scope.lock() {
            *slot = cgroups::create_scope(policy);
        }

        // Build seccomp filter in the parent so the child's pre_exec stays
        // async-signal-safe (no allocations beyond moving the Arc).
        let bpf = Arc::new(
            seccomp::build(policy.network)
                .map_err(|e| SandboxError::Setup(format!("seccomp build: {e}")))?,
        );

        let mut ns_plan = namespaces::plan(policy)
            .map_err(|e| SandboxError::Setup(format!("namespace plan: {e}")))?;

        let policy = policy.clone();
        let bpf_for_child: Arc<BpfProgram> = Arc::clone(&bpf);

        unsafe {
            cmd.pre_exec(move || apply_in_child(&policy, &mut ns_plan, &bpf_for_child));
        }

        Ok(())
    }

    fn attach(&self, pid: u32, _policy: &SandboxPolicy) -> Result<(), SandboxError> {
        let scope = self.pending_scope.lock().ok().and_then(|mut s| s.take());
        let Some(scope) = scope else {
            return Ok(());
        };

        if let Err(e) = cgroups::write_pid(&scope, pid) {
            tracing::debug!("cgroup attach pid {pid} -> {scope:?}: {e}");
            cgroups::remove_scope(&scope);
            return Ok(());
        }

        if let Ok(mut map) = self.scopes.lock() {
            map.insert(pid, scope);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "linux"
    }
}

impl Drop for LinuxSandbox {
    fn drop(&mut self) {
        if let Ok(map) = self.scopes.lock() {
            for scope in map.values() {
                cgroups::remove_scope(scope);
            }
        }
    }
}

/// Runs inside the forked child between fork and exec. Async-signal-safe:
/// the crates we call (`landlock`, `nix`, `seccompiler`) are all thin
/// syscall wrappers at this point.
fn apply_in_child(
    policy: &SandboxPolicy,
    ns_plan: &mut namespaces::Plan,
    bpf: &BpfProgram,
) -> io::Result<()> {
    tagged(b"ns", namespaces::apply(ns_plan))?;

    let mb = |n: u64| n.saturating_mul(1024 * 1024);
    tagged(
        b"as",
        rlimits::set(Resource::RLIMIT_AS, mb(policy.max_memory_mb)),
    )?;
    tagged(
        b"cpu",
        rlimits::set(Resource::RLIMIT_CPU, policy.max_cpu_secs),
    )?;
    tagged(
        b"nofile",
        rlimits::set(Resource::RLIMIT_NOFILE, policy.max_open_fds as u64),
    )?;
    tagged(
        b"nproc",
        rlimits::set(Resource::RLIMIT_NPROC, policy.max_procs as u64),
    )?;
    tagged(
        b"fsize",
        rlimits::set(Resource::RLIMIT_FSIZE, mb(policy.max_file_sz_mb)),
    )?;
    tagged(b"core", rlimits::zero(Resource::RLIMIT_CORE))?;

    tagged(b"seccomp", seccomp::install(bpf))?;

    Ok(())
}

/// Write a short error tag to stderr before propagating so the parent can
/// see which step in pre_exec failed (errno alone isn't enough). Only fires
/// on Err; success path is silent. Async-signal-safe.
fn tagged<T>(tag: &[u8], r: io::Result<T>) -> io::Result<T> {
    if r.is_err() {
        const PREFIX: &[u8] = b"[mistralrs-sandbox] pre_exec failed at: ";
        unsafe {
            libc::write(2, PREFIX.as_ptr() as *const _, PREFIX.len());
            libc::write(2, tag.as_ptr() as *const _, tag.len());
            libc::write(2, b"\n".as_ptr() as *const _, 1);
        }
    }
    r
}
