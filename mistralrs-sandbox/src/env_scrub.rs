//! Environment scrub shared by Linux and macOS sandboxes.
//!
//! Clears the parent env, replays a short hardcoded allowlist plus
//! `policy.extra_env`, and re-points `HOME` + XDG dirs at the session
//! workdir so libraries that expect a writable `$HOME` (matplotlib font
//! cache, click config) work inside the sandbox without leaking into the
//! real user home.

use tokio::process::Command;

use crate::SandboxPolicy;

/// Env vars allowed to propagate into the sandboxed child by default.
/// Anything else is dropped (including `LD_PRELOAD` and secrets like
/// `AWS_*`, `HF_TOKEN`, `OPENAI_API_KEY`). To pass additional vars to a
/// session, list them in `SandboxPolicy::extra_env`.
///
/// `HF_TOKEN`/`HF_HOME`/`HF_HUB_CACHE` are deliberately NOT in the default
/// allowlist: model-generated code could `print(os.environ["HF_TOKEN"])`
/// and exfiltrate before any network restriction kicks in.
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
