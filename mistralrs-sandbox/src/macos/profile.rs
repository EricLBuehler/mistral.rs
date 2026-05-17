//! Seatbelt profile generation.

use std::path::Path;

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
        allow_read(&mut profile, Path::new(path));
    }

    for path in &policy.extra_fs_read {
        allow_read(&mut profile, path);
    }

    for path in &policy.extra_fs_write {
        allow_read_write(&mut profile, path);
    }

    if let Some(workdir) = policy.session_workdir.as_ref() {
        allow_read_write(&mut profile, workdir);
    }

    match policy.network {
        NetworkMode::None => {
            // No network rules = deny by default.
        }
        NetworkMode::Loopback => {
            profile.push_str("(allow network* (local ip \"localhost:*\"))\n");
            profile.push_str("(allow network* (remote ip \"localhost:*\"))\n");
        }
        NetworkMode::Full => {
            profile.push_str("(allow network*)\n");
        }
    }

    profile
}

fn allow_read(profile: &mut String, path: &Path) {
    for path in path_variants(path) {
        let path = sbpl_string(&path);
        profile.push_str(&format!("(allow file-read* (subpath \"{path}\"))\n"));
        profile.push_str(&format!(
            "(allow file-map-executable (subpath \"{path}\"))\n"
        ));
    }
}

fn allow_read_write(profile: &mut String, path: &Path) {
    for path in path_variants(path) {
        let path = sbpl_string(&path);
        profile.push_str(&format!(
            "(allow file-read* (subpath \"{path}\"))\n(allow file-write* (subpath \"{path}\"))\n"
        ));
    }
}

fn path_variants(path: &Path) -> Vec<String> {
    let mut paths = vec![path.display().to_string()];
    if let Ok(canonical) = path.canonicalize() {
        let canonical = canonical.display().to_string();
        if !paths.contains(&canonical) {
            paths.push(canonical);
        }
    }
    paths
}

fn sbpl_string(path: &str) -> String {
    path.replace('\\', "\\\\").replace('"', "\\\"")
}
