//! Seatbelt SBPL (Sandbox Profile Language) generator. SBPL is a Scheme-ish
//! DSL Apple ships for `sandbox-exec`. We deny everything by default and
//! allow the minimum needed for Python to run: read of system libraries,
//! write to the session workdir, and (optionally) loopback or full network.

use crate::{NetworkMode, SandboxPolicy};

const READ_ALLOWED: &[&str] = &[
    "/usr",
    "/System",
    "/Library",
    "/private/etc",
    "/private/var/db/timezone",
    "/opt/homebrew",
    "/opt/local",
    "/bin",
    "/sbin",
    "/dev/null",
    "/dev/urandom",
    "/dev/random",
    "/dev/zero",
];

pub(crate) fn render(policy: &SandboxPolicy) -> String {
    let mut profile = String::new();
    profile.push_str("(version 1)\n(deny default)\n");
    profile.push_str("(allow process-fork)\n");
    profile.push_str("(allow process-exec)\n");
    profile.push_str("(allow signal (target self))\n");
    profile.push_str("(allow mach-lookup)\n");
    profile.push_str("(allow sysctl-read)\n");
    profile.push_str("(allow file-ioctl)\n");

    for path in READ_ALLOWED {
        profile.push_str(&format!("(allow file-read* (subpath \"{path}\"))\n"));
    }

    if let Some(workdir) = policy.session_workdir.as_ref() {
        let escaped = workdir.display().to_string().replace('"', "\\\"");
        profile.push_str(&format!(
            "(allow file-read* (subpath \"{escaped}\"))\n(allow file-write* (subpath \"{escaped}\"))\n"
        ));
    }

    // tmp + tempdir for python.
    profile.push_str("(allow file-write* (subpath \"/private/tmp\"))\n");
    profile.push_str("(allow file-write* (subpath \"/private/var/folders\"))\n");

    match policy.network {
        NetworkMode::None => {
            // No network rules = deny by default.
        }
        NetworkMode::Loopback => {
            profile.push_str("(allow network* (local ip \"localhost:*\"))\n");
            profile.push_str("(allow network* (remote ip \"localhost:*\"))\n");
            profile.push_str("(allow network* (local ip \"127.0.0.1:*\"))\n");
            profile.push_str("(allow network* (remote ip \"127.0.0.1:*\"))\n");
            profile.push_str("(allow network* (local ip \"[::1]:*\"))\n");
            profile.push_str("(allow network* (remote ip \"[::1]:*\"))\n");
        }
        NetworkMode::Full => {
            profile.push_str("(allow network*)\n");
        }
    }

    profile
}
