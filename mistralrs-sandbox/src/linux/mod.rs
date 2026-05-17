//! Linux sandbox layers for model-generated subprocesses.

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
    scopes: Mutex<std::collections::HashMap<u32, PathBuf>>,
    seccomp_supported: bool,
}

impl LinuxSandbox {
    pub fn new() -> Self {
        Self {
            scopes: Mutex::new(std::collections::HashMap::new()),
            seccomp_supported: seccomp::supported(),
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

        let bpf = if self.seccomp_supported {
            Some(Arc::new(seccomp::build(policy.network).map_err(|e| {
                SandboxError::Setup(format!("seccomp build: {e}"))
            })?))
        } else if policy.strict {
            return Err(SandboxError::Setup(
                "sandbox=on but seccomp is unavailable on this host".to_string(),
            ));
        } else {
            tracing::warn!("seccomp unavailable - syscall and socket deny-list is OFF");
            None
        };

        let mut ns_plan = namespaces::plan(policy)
            .map_err(|e| SandboxError::Setup(format!("namespace plan: {e}")))?;
        let nproc_limit = rlimits::nproc_limit(policy.max_procs as u64);

        let policy = policy.clone();
        let bpf_for_child = bpf.clone();

        unsafe {
            cmd.pre_exec(move || {
                apply_in_child(&policy, &mut ns_plan, bpf_for_child.as_deref(), nproc_limit)
            });
        }

        Ok(())
    }

    fn attach(&self, pid: u32, policy: &SandboxPolicy) -> Result<(), SandboxError> {
        let Some(scope) = cgroups::create_scope(policy) else {
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

    fn effective(&self, policy: &SandboxPolicy) -> crate::EffectiveProtection {
        crate::EffectiveProtection {
            fs_isolated: namespaces::landlock_supported(),
            network_isolated: match policy.network {
                crate::NetworkMode::None => self.seccomp_supported,
                crate::NetworkMode::Loopback => namespaces::netns_supported(),
                crate::NetworkMode::Full => false,
            },
            rlimits_applied: true,
        }
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

fn apply_in_child(
    policy: &SandboxPolicy,
    ns_plan: &mut namespaces::Plan,
    bpf: Option<&BpfProgram>,
    nproc_limit: u64,
) -> io::Result<()> {
    tagged(b"nproc", rlimits::set(Resource::RLIMIT_NPROC, nproc_limit))?;
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
        b"fsize",
        rlimits::set(Resource::RLIMIT_FSIZE, mb(policy.max_file_sz_mb)),
    )?;
    tagged(b"core", rlimits::zero(Resource::RLIMIT_CORE))?;

    if let Some(bpf) = bpf {
        tagged(b"seccomp", seccomp::install(bpf))?;
    }

    Ok(())
}

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
