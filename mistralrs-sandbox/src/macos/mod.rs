//! macOS Seatbelt sandbox.

mod profile;

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

        let original_program = cmd.as_std().get_program().to_os_string();
        let original_args: Vec<_> = cmd.as_std().get_args().map(|a| a.to_os_string()).collect();
        let workdir = cmd.as_std().get_current_dir().map(|d| d.to_path_buf());

        let sbpl = profile::render(policy);

        let mut new_cmd = tokio::process::Command::new(SANDBOX_EXEC);
        new_cmd.arg("-p").arg(sbpl);
        new_cmd.arg(original_program);
        for a in original_args {
            new_cmd.arg(a);
        }

        new_cmd
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true);

        env_scrub::apply(&mut new_cmd, policy);
        if let Some(dir) = workdir {
            new_cmd.current_dir(dir);
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
            fs_isolated: sandbox_exec_available,
            network_isolated: sandbox_exec_available
                && !matches!(policy.network, NetworkMode::Full),
            rlimits_applied: false,
        }
    }
}
