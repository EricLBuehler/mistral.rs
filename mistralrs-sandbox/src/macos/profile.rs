//! Seatbelt profile generation.

use std::path::Path;

use crate::{NetworkMode, SandboxPolicy};

const BASE_RULES: &str = "\
(version 1)
(deny default)
(allow process-fork)
(allow process-exec)
(allow signal (target self))
(allow mach*)
(allow iokit-open)
(allow sysctl-read)
(allow file-ioctl)
(allow file-read-metadata)
(allow file-read-data (literal \"/\"))
";

const READ_ALLOWED_SUBPATHS: &[&str] = &[
    "/usr",
    "/System",
    "/Library",
    "/private/etc",
    "/private/var/db/dyld",
    "/private/var/db/timezone",
    "/opt/homebrew",
    "/opt/local",
    "/bin",
    "/sbin",
    "/dev/fd",
];

const READ_ALLOWED_FILES: &[&str] = &["/dev/null", "/dev/urandom", "/dev/random", "/dev/zero"];

pub(crate) fn render(policy: &SandboxPolicy) -> String {
    let mut profile = String::from(BASE_RULES);

    for path in READ_ALLOWED_SUBPATHS {
        allow_read_subpath(&mut profile, Path::new(path));
    }

    for path in READ_ALLOWED_FILES {
        allow_read_literal(&mut profile, Path::new(path));
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
        allow_file_read_literal(profile, &path);
        allow_file_read_subpath(profile, &path);
        allow_file_map_literal(profile, &path);
        allow_file_map_subpath(profile, &path);
    }
}

fn allow_read_write(profile: &mut String, path: &Path) {
    for path in path_variants(path) {
        let path = sbpl_string(&path);
        allow_file_read_literal(profile, &path);
        allow_file_read_subpath(profile, &path);
        profile.push_str(&format!("(allow file-write* (literal \"{path}\"))\n"));
        profile.push_str(&format!("(allow file-write* (subpath \"{path}\"))\n"));
    }
}

fn allow_read_literal(profile: &mut String, path: &Path) {
    for path in path_variants(path) {
        let path = sbpl_string(&path);
        allow_file_read_literal(profile, &path);
    }
}

fn allow_read_subpath(profile: &mut String, path: &Path) {
    for path in path_variants(path) {
        let path = sbpl_string(&path);
        allow_file_read_subpath(profile, &path);
        allow_file_map_subpath(profile, &path);
    }
}

fn allow_file_read_literal(profile: &mut String, path: &str) {
    profile.push_str(&format!("(allow file-read* (literal \"{path}\"))\n"));
}

fn allow_file_read_subpath(profile: &mut String, path: &str) {
    profile.push_str(&format!("(allow file-read* (subpath \"{path}\"))\n"));
}

fn allow_file_map_literal(profile: &mut String, path: &str) {
    profile.push_str(&format!(
        "(allow file-map-executable (literal \"{path}\"))\n"
    ));
}

fn allow_file_map_subpath(profile: &mut String, path: &str) {
    profile.push_str(&format!(
        "(allow file-map-executable (subpath \"{path}\"))\n"
    ));
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn loopback_uses_sandbox_exec_hostnames() {
        let profile = render(&SandboxPolicy::default());

        assert!(profile.contains("(allow network* (local ip \"localhost:*\"))"));
        assert!(profile.contains("(allow network* (remote ip \"localhost:*\"))"));
        assert!(!profile.contains("127.0.0.1"));
        assert!(!profile.contains("::1"));
    }

    #[test]
    fn none_network_emits_no_network_rules() {
        let profile = render(&SandboxPolicy {
            network: NetworkMode::None,
            ..SandboxPolicy::default()
        });

        assert!(!profile.contains("(allow network"));
    }

    #[test]
    fn workdir_gets_read_write_rules() {
        let profile = render(&SandboxPolicy {
            session_workdir: Some(PathBuf::from("/tmp/mistralrs workdir")),
            ..SandboxPolicy::default()
        });

        assert!(profile.contains("(allow file-read* (literal \"/tmp/mistralrs workdir\"))"));
        assert!(profile.contains("(allow file-write* (literal \"/tmp/mistralrs workdir\"))"));
        assert!(profile.contains("(allow file-write* (subpath \"/tmp/mistralrs workdir\"))"));
        assert!(
            !profile.contains("(allow file-map-executable (literal \"/tmp/mistralrs workdir\"))")
        );
        assert!(
            !profile.contains("(allow file-map-executable (subpath \"/tmp/mistralrs workdir\"))")
        );
    }

    #[test]
    fn paths_are_escaped_for_sbpl_strings() {
        let profile = render(&SandboxPolicy {
            extra_fs_read: vec![PathBuf::from("/tmp/quote\"and\\slash")],
            ..SandboxPolicy::default()
        });

        assert!(profile.contains("/tmp/quote\\\"and\\\\slash"));
    }
}
