use tokio::process::Command;

use crate::SandboxPolicy;

pub(crate) const ALLOWED: &[&str] = &[
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "HOME",
    "TMPDIR",
    "PYTHONHASHSEED",
    "PYTHONIOENCODING",
    "PYTHONUNBUFFERED",
];

pub(crate) fn apply(cmd: &mut Command, policy: &SandboxPolicy) {
    cmd.env_clear();
    for var in ALLOWED {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
    }
    for var in &policy.extra_env {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
    }
    if let Some(workdir) = policy.session_workdir.as_ref() {
        cmd.env("HOME", workdir);
        cmd.env("XDG_CACHE_HOME", workdir);
        cmd.env("XDG_CONFIG_HOME", workdir);
        cmd.env("XDG_DATA_HOME", workdir);
        cmd.env("TMPDIR", workdir);
    }
}
