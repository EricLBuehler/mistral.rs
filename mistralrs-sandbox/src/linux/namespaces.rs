//! Linux namespaces and Landlock filesystem isolation.

use std::ffi::CStr;
use std::io;
use std::mem::MaybeUninit;
use std::path::Path;
use std::sync::OnceLock;

use landlock::{
    path_beneath_rules, Access, AccessFs, Ruleset, RulesetAttr, RulesetCreated, RulesetCreatedAttr,
    RulesetError, ABI,
};
use nix::fcntl::{open, OFlag};
use nix::sched::{unshare, CloneFlags};
use nix::sys::stat::Mode;
use nix::unistd::{write, Gid, Uid};

use crate::{NetworkMode, SandboxPolicy};

pub(crate) struct Plan {
    unshare_flags: CloneFlags,
    outer_uid: u32,
    outer_gid: u32,
    bring_up_lo: bool,
    landlock_ruleset: Option<RulesetCreated>,
}

pub(crate) fn plan(policy: &SandboxPolicy) -> io::Result<Plan> {
    let supports_userns = probe_namespace_supported(base_userns_flags());
    let supports_netns = supports_userns
        && probe_namespace_supported(base_userns_flags() | CloneFlags::CLONE_NEWNET);
    let (flags, bring_up_lo) = namespace_flags(policy, supports_userns, supports_netns)?;

    let landlock_ruleset = if landlock_supported() {
        match build_landlock_ruleset(policy) {
            Ok(r) => Some(r),
            Err(e) => {
                if policy.strict {
                    return Err(io::Error::other(format!(
                        "sandbox=on but Landlock is unavailable on this kernel: {e}. \
                         Landlock requires Linux 5.13+ with CONFIG_SECURITY_LANDLOCK=y."
                    )));
                }
                tracing::warn!(
                    "Landlock unavailable, filesystem isolation is OFF (other layers still apply): {e}"
                );
                None
            }
        }
    } else if policy.strict {
        return Err(io::Error::other(
            "sandbox=on but Landlock cannot restrict this process on this host. \
             Landlock requires Linux 5.13+ with CONFIG_SECURITY_LANDLOCK=y and permission \
             to set no_new_privs.",
        ));
    } else {
        tracing::warn!(
            "Landlock cannot restrict this process on this host - filesystem isolation is OFF"
        );
        None
    };

    Ok(Plan {
        unshare_flags: flags,
        outer_uid: Uid::effective().as_raw(),
        outer_gid: Gid::effective().as_raw(),
        bring_up_lo,
        landlock_ruleset,
    })
}

pub(crate) fn netns_supported() -> bool {
    probe_namespace_supported(base_userns_flags() | CloneFlags::CLONE_NEWNET)
}

pub(crate) fn landlock_supported() -> bool {
    static SUPPORTED: OnceLock<bool> = OnceLock::new();
    *SUPPORTED.get_or_init(probe_landlock_supported)
}

fn probe_namespace_supported(flags: CloneFlags) -> bool {
    use nix::sys::wait::{waitpid, WaitStatus};
    use nix::unistd::{fork, ForkResult};

    match unsafe { fork() } {
        Ok(ForkResult::Parent { child }) => {
            matches!(waitpid(child, None), Ok(WaitStatus::Exited(_, 0)))
        }
        Ok(ForkResult::Child) => {
            let ok = probe_namespace_child(flags).is_ok();
            unsafe { libc::_exit(if ok { 0 } else { 1 }) };
        }
        Err(_) => false,
    }
}

fn probe_namespace_child(flags: CloneFlags) -> io::Result<()> {
    let outer_uid = Uid::effective().as_raw();
    let outer_gid = Gid::effective().as_raw();
    unshare(flags).map_err(|e| io::Error::from_raw_os_error(e as i32))?;
    if flags.contains(CloneFlags::CLONE_NEWUSER) {
        write_uid_gid_maps(outer_uid, outer_gid)?;
    }
    if flags.contains(CloneFlags::CLONE_NEWNET) {
        bring_up_loopback()?;
    }
    Ok(())
}

fn probe_landlock_supported() -> bool {
    use nix::sys::wait::{waitpid, WaitStatus};
    use nix::unistd::{fork, ForkResult};

    match unsafe { fork() } {
        Ok(ForkResult::Parent { child }) => {
            matches!(waitpid(child, None), Ok(WaitStatus::Exited(_, 0)))
        }
        Ok(ForkResult::Child) => {
            let ok = Ruleset::default()
                .handle_access(AccessFs::from_all(ABI::V1))
                .and_then(|r| r.create())
                .and_then(|r| r.restrict_self().map(|_| ()))
                .is_ok();
            unsafe { libc::_exit(if ok { 0 } else { 1 }) };
        }
        Err(_) => false,
    }
}

fn base_userns_flags() -> CloneFlags {
    CloneFlags::CLONE_NEWUSER | CloneFlags::CLONE_NEWIPC | CloneFlags::CLONE_NEWUTS
}

fn namespace_flags(
    policy: &SandboxPolicy,
    supports_userns: bool,
    supports_netns: bool,
) -> io::Result<(CloneFlags, bool)> {
    let mut flags = CloneFlags::empty();
    if supports_userns {
        flags |= base_userns_flags();
        if matches!(policy.network, NetworkMode::Loopback) {
            if supports_netns {
                flags |= CloneFlags::CLONE_NEWNET;
                return Ok((flags, true));
            }
            if policy.strict {
                return Err(io::Error::other(
                    "sandbox=on with network=loopback requires unprivileged user and network \
                     namespaces, but network namespaces are disabled on this host",
                ));
            }
            tracing::warn!(
                "network namespaces disabled - network=loopback falls back to no network \
                 restriction beyond seccomp. Set network=none for strict isolation, \
                 or use --sandbox on to make this a hard error."
            );
        }
        return Ok((flags, false));
    }

    if matches!(policy.network, NetworkMode::Loopback) {
        if policy.strict {
            return Err(io::Error::other(
                "sandbox=on with network=loopback requires unprivileged user namespaces, \
                 which are disabled on this host. Use network=none, or enable userns \
                 (sysctl kernel.unprivileged_userns_clone=1 on Debian)",
            ));
        }
        tracing::warn!(
            "unprivileged user namespaces disabled - network=loopback falls back to no \
             network restriction beyond seccomp. Set network=none for strict isolation, \
             or use --sandbox on to make this a hard error."
        );
    }

    Ok((flags, false))
}

fn build_landlock_ruleset(policy: &SandboxPolicy) -> Result<RulesetCreated, RulesetError> {
    let abi = ABI::V3;
    let mut ruleset = Ruleset::default()
        .handle_access(AccessFs::from_all(abi))?
        .create()?
        .add_rules(path_beneath_rules(read_paths(), AccessFs::from_read(abi)))?
        .add_rules(path_beneath_rules(
            policy.extra_fs_read.iter().map(|p| p.as_path()),
            AccessFs::from_read(abi),
        ))?;

    let mut write_paths: Vec<&Path> = policy.extra_fs_write.iter().map(|p| p.as_path()).collect();
    if let Some(workdir) = policy.session_workdir.as_ref() {
        write_paths.push(workdir.as_path());
    }
    ruleset = ruleset.add_rules(path_beneath_rules(write_paths, AccessFs::from_all(abi)))?;
    Ok(ruleset)
}

fn read_paths() -> Vec<&'static str> {
    let candidates: &[&str] = &[
        "/usr",
        "/lib",
        "/lib64",
        "/bin",
        "/sbin",
        "/etc",
        "/opt",
        "/proc/self",
        "/sys/devices/system/cpu",
        "/dev/null",
        "/dev/urandom",
        "/dev/random",
        "/dev/zero",
    ];
    candidates
        .iter()
        .copied()
        .filter(|p| Path::new(p).exists())
        .collect()
}

pub(crate) fn apply(plan: &mut Plan) -> io::Result<()> {
    if !plan.unshare_flags.is_empty() {
        unshare(plan.unshare_flags).map_err(|e| io::Error::from_raw_os_error(e as i32))?;
        write_uid_gid_maps(plan.outer_uid, plan.outer_gid)?;
    }

    if plan.bring_up_lo {
        bring_up_loopback()?;
    }

    if let Some(ruleset) = plan.landlock_ruleset.take() {
        ruleset
            .restrict_self()
            .map_err(|e| io::Error::other(e.to_string()))?;
    }

    Ok(())
}

fn write_uid_gid_maps(uid: u32, gid: u32) -> io::Result<()> {
    let mut ubuf = [0u8; 32];
    let uid_line = build_map_line(&mut ubuf, uid);
    write_proc_file(c"/proc/self/uid_map", uid_line)?;

    write_proc_file(c"/proc/self/setgroups", b"deny")?;

    let mut gbuf = [0u8; 32];
    let gid_line = build_map_line(&mut gbuf, gid);
    write_proc_file(c"/proc/self/gid_map", gid_line)?;
    Ok(())
}

fn build_map_line(buf: &mut [u8; 32], id: u32) -> &[u8] {
    let mut ibuf = itoa::Buffer::new();
    let id_str = ibuf.format(id);
    let id_bytes = id_str.as_bytes();
    let mut pos = 0;
    buf[pos] = b'0';
    pos += 1;
    buf[pos] = b' ';
    pos += 1;
    buf[pos..pos + id_bytes.len()].copy_from_slice(id_bytes);
    pos += id_bytes.len();
    buf[pos] = b' ';
    pos += 1;
    buf[pos] = b'1';
    pos += 1;
    buf[pos] = b'\n';
    pos += 1;
    &buf[..pos]
}

fn write_proc_file(path: &CStr, contents: &[u8]) -> io::Result<()> {
    let fd = open(path, OFlag::O_WRONLY | OFlag::O_CLOEXEC, Mode::empty())
        .map_err(|e| io::Error::from_raw_os_error(e as i32))?;
    let mut written = 0;
    while written < contents.len() {
        let n =
            write(&fd, &contents[written..]).map_err(|e| io::Error::from_raw_os_error(e as i32))?;
        if n == 0 {
            return Err(io::Error::new(io::ErrorKind::WriteZero, "short proc write"));
        }
        written += n;
    }
    Ok(())
}

fn bring_up_loopback() -> io::Result<()> {
    let sock = unsafe { libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0) };
    if sock < 0 {
        return Err(io::Error::last_os_error());
    }

    let mut ifr: libc::ifreq = unsafe { MaybeUninit::zeroed().assume_init() };
    ifr.ifr_name[0] = b'l' as libc::c_char;
    ifr.ifr_name[1] = b'o' as libc::c_char;
    ifr.ifr_name[2] = 0;
    ifr.ifr_ifru.ifru_flags = (libc::IFF_UP | libc::IFF_RUNNING) as i16;

    let rc = unsafe { libc::ioctl(sock, libc::SIOCSIFFLAGS, &ifr) };
    let err = if rc == 0 {
        None
    } else {
        Some(io::Error::last_os_error())
    };
    unsafe { libc::close(sock) };

    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_none_does_not_require_netns() {
        let policy = SandboxPolicy {
            network: NetworkMode::None,
            ..SandboxPolicy::default()
        };

        let (flags, bring_up_lo) = namespace_flags(&policy, true, false).unwrap();

        assert!(flags.contains(base_userns_flags()));
        assert!(!flags.contains(CloneFlags::CLONE_NEWNET));
        assert!(!bring_up_lo);
    }

    #[test]
    fn loopback_falls_back_without_netns_in_auto_mode() {
        let policy = SandboxPolicy::default();

        let (flags, bring_up_lo) = namespace_flags(&policy, true, false).unwrap();

        assert!(flags.contains(base_userns_flags()));
        assert!(!flags.contains(CloneFlags::CLONE_NEWNET));
        assert!(!bring_up_lo);
    }

    #[test]
    fn loopback_requires_netns_in_strict_mode() {
        let policy = SandboxPolicy {
            strict: true,
            ..SandboxPolicy::default()
        };

        assert!(namespace_flags(&policy, true, false).is_err());
    }

    #[test]
    fn loopback_uses_netns_when_available() {
        let policy = SandboxPolicy::default();

        let (flags, bring_up_lo) = namespace_flags(&policy, true, true).unwrap();

        assert!(flags.contains(CloneFlags::CLONE_NEWNET));
        assert!(bring_up_lo);
    }
}
