//! macOS Seatbelt sandbox. We wrap the user's argv with `sandbox-exec -p
//! <profile>`. Same mechanism Chromium and other macOS-native sandboxes use.
//!
//! Resource limits (memory/cpu/file size) are applied via `setrlimit` in a
//! pre_exec hook just like Linux. macOS doesn't have a cgroup-equivalent
//! that's accessible without root, so memory bombs are bounded only by
//! RLIMIT_AS (best-effort).

mod profile;

use std::io;
use std::os::unix::process::CommandExt;
use std::path::Path;

use crate::{env_scrub, EffectiveProtection, NetworkMode, Sandbox, SandboxError, SandboxPolicy};

const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

pub struct MacosSandbox;

impl MacosSandbox {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MacosSandbox {
    fn default() -> Self {
        Self::new()
    }
}

impl Sandbox for MacosSandbox {
    fn harden(
        &self,
        cmd: &mut tokio::process::Command,
        policy: &SandboxPolicy,
    ) -> Result<(), SandboxError> {
        if !Path::new(SANDBOX_EXEC).exists() {
            return Err(SandboxError::Setup(format!(
                "{SANDBOX_EXEC} not found; macOS sandbox requires the system sandbox-exec binary"
            )));
        }

        // sandbox-exec takes -p <profile> followed by argv. We rewrite the
        // command: original program becomes the first positional arg.
        let original_program = cmd.as_std().get_program().to_os_string();
        let original_args: Vec<_> = cmd.as_std().get_args().map(|a| a.to_os_string()).collect();
        let workdir = cmd.as_std().get_current_dir().map(|d| d.to_path_buf());

        let sbpl = profile::render(policy);

        // tokio::process::Command doesn't expose a way to replace the program
        // in place, so we build a fresh std::process::Command and swap into it.
        let mut new_cmd = tokio::process::Command::new(SANDBOX_EXEC);
        new_cmd.arg("-p").arg(sbpl);
        new_cmd.arg(original_program);
        for a in original_args {
            new_cmd.arg(a);
        }

        // Carry stdio configuration over. The tokio::process::Command stdio
        // setters return self; on the rebuilt command we just re-apply piped
        // because that's what code-exec uses. If we ever expose this for
        // generic callers we'll need to clone the std Stdio settings.
        new_cmd
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        // Apply the same env scrub as Linux: clear inherited env, replay the
        // allowlist + `extra_env` from the parent process, re-point HOME and
        // XDG dirs at the workdir. Reading from the original cmd's env-overrides
        // (the previous behavior) would drop PATH/HOME entirely whenever the
        // caller hadn't pre-set them.
        env_scrub::apply(&mut new_cmd, policy);
        if let Some(dir) = workdir {
            new_cmd.current_dir(dir);
        }

        let policy = policy.clone();
        unsafe {
            new_cmd.pre_exec(move || rlimit_pre_exec(&policy));
        }

        *cmd = new_cmd;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "macos"
    }

    fn effective(&self, policy: &SandboxPolicy) -> EffectiveProtection {
        let sandbox_exec_available = Path::new(SANDBOX_EXEC).exists();
        EffectiveProtection {
            // Seatbelt enforces FS + network rules through sandbox-exec; if
            // that binary is missing, harden() returns Err and no layer
            // actually runs.
            fs_isolated: sandbox_exec_available,
            network_isolated: sandbox_exec_available
                && !matches!(policy.network, NetworkMode::Full),
            // rlimits are applied via pre_exec but pre_exec only runs if
            // spawn() succeeds, which requires sandbox-exec.
            rlimits_applied: sandbox_exec_available,
        }
    }
}

fn rlimit_pre_exec(policy: &SandboxPolicy) -> io::Result<()> {
    set_rlimit(
        libc::RLIMIT_AS,
        policy.max_memory_mb.saturating_mul(1024 * 1024),
    )?;
    set_rlimit(libc::RLIMIT_CPU, policy.max_cpu_secs)?;
    set_rlimit(libc::RLIMIT_NOFILE, policy.max_open_fds as u64)?;
    set_rlimit(libc::RLIMIT_NPROC, policy.max_procs as u64)?;
    set_rlimit(
        libc::RLIMIT_FSIZE,
        policy.max_file_sz_mb.saturating_mul(1024 * 1024),
    )?;
    set_rlimit(libc::RLIMIT_CORE, 0)?;
    Ok(())
}

fn set_rlimit(resource: i32, value: u64) -> io::Result<()> {
    let lim = libc::rlimit {
        rlim_cur: value as libc::rlim_t,
        rlim_max: value as libc::rlim_t,
    };
    let rc = unsafe { libc::setrlimit(resource as _, &lim) };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}
