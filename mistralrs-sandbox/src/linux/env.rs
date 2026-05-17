//! Environment scrub: clear the inherited env and replay only a small
//! allowlist of variables that are commonly needed for Python tooling.

use tokio::process::Command;

/// Env vars allowed to propagate into the sandboxed child. Anything else is
/// dropped (including `LD_PRELOAD`, secrets in `AWS_*`, etc.).
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
    "HF_TOKEN",
    "HF_HOME",
    "HF_HUB_CACHE",
];

pub(crate) fn apply(cmd: &mut Command) {
    cmd.env_clear();
    for var in ALLOWED {
        if let Ok(val) = std::env::var(var) {
            cmd.env(var, val);
        }
    }
}
