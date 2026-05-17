//! Environment scrub: clear the inherited env and replay only a small
//! allowlist of variables that are commonly needed for Python tooling.

use tokio::process::Command;

use crate::SandboxPolicy;

/// Env vars allowed to propagate into the sandboxed child by default.
/// Anything else is dropped (including `LD_PRELOAD` and secrets like
/// `AWS_*`, `HF_TOKEN`, `OPENAI_API_KEY`). To pass additional vars to a
/// session, list them in `SandboxPolicy::extra_env`.
///
/// HF_TOKEN/HF_HOME/HF_HUB_CACHE are deliberately NOT in the default
/// allowlist: model-generated code could `print(os.environ["HF_TOKEN"])`
/// and exfiltrate it before any network restriction kicks in.
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
    // Point HOME and the XDG dirs at the session workdir so libraries that
    // expect a writable home (matplotlib font cache, click config, etc.)
    // don't hit EACCES on the user's real ~. The workdir is already in
    // Landlock's write allowlist.
    if let Some(workdir) = policy.session_workdir.as_ref() {
        cmd.env("HOME", workdir);
        cmd.env("XDG_CACHE_HOME", workdir);
        cmd.env("XDG_CONFIG_HOME", workdir);
        cmd.env("XDG_DATA_HOME", workdir);
        cmd.env("TMPDIR", workdir);
    }
}
