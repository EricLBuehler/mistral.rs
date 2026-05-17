//! Linux namespace isolation + Landlock-based FS access control.
//!
//! Built on top of `landlock`, `nix`, and `itoa` so this file works with
//! typed APIs rather than raw `libc::syscall` calls. The only remaining
//! hand-rolled bits are (1) the uid/gid map line we render into a stack
//! buffer to stay pre_exec-safe, and (2) the loopback ioctl (the `ifreq`
//! union nix doesn't model cleanly).
//!
//! FS isolation uses Landlock rather than mount-namespace + `pivot_root`.
//! Landlock applies via one `landlock_restrict_self` syscall (handled by
//! the crate) which is safe in pre_exec. For our threat model
//! (model-generated code, not kernel escapes) it provides equivalent
//! practical isolation with far less hand-rolled code.

use std::ffi::CStr;
use std::io;
use std::mem::MaybeUninit;
use std::path::Path;

use landlock::{
    path_beneath_rules, Access, AccessFs, Ruleset, RulesetAttr, RulesetCreated,
    RulesetCreatedAttr, RulesetError, ABI,
};
use nix::fcntl::{open, OFlag};
use nix::sched::{unshare, CloneFlags};
use nix::sys::stat::Mode;
use nix::unistd::{write, Gid, Uid};

use crate::{NetworkMode, SandboxPolicy};

pub(crate) struct Plan {
    unshare_flags: CloneFlags,
    /// Caller's UID/GID captured in the parent. After `CLONE_NEWUSER` the
    /// child sees 65534 (nobody) until uid_map is set up, so we can't ask
    /// the kernel for it later.
    outer_uid: u32,
    outer_gid: u32,
    bring_up_lo: bool,
    /// Built in the parent. `restrict_self` consumes `self`, so apply()
    /// calls `.take()` and moves it out.
    landlock_ruleset: Option<RulesetCreated>,
}

pub(crate) fn plan(policy: &SandboxPolicy) -> io::Result<Plan> {
    // Non-user-ns flags require CAP_SYS_ADMIN unless we also unshare USER.
    // Probe support so we don't blow up on hosts that disabled unprivileged
    // user namespaces (Debian default, hardened containers).
    //
    // We deliberately do NOT include CLONE_NEWPID: unshare(CLONE_NEWPID) only
    // affects future children, not the calling thread. Since we're already
    // the forked-but-not-exec'd child here, the exec'd python ends up in the
    // parent's PID namespace anyway. Real PID isolation would require an
    // extra fork after unshare (i.e. a launcher binary). IPC/UTS still apply
    // to us directly.
    let supports_userns = probe_userns_supported();

    let mut flags = CloneFlags::empty();
    if supports_userns {
        flags |= CloneFlags::CLONE_NEWUSER | CloneFlags::CLONE_NEWIPC | CloneFlags::CLONE_NEWUTS;
        if !matches!(policy.network, NetworkMode::Full) {
            flags |= CloneFlags::CLONE_NEWNET;
        }
    } else if matches!(policy.network, NetworkMode::Loopback) {
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

    let landlock_ruleset = match build_landlock_ruleset(policy) {
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
    };

    Ok(Plan {
        unshare_flags: flags,
        outer_uid: Uid::effective().as_raw(),
        outer_gid: Gid::effective().as_raw(),
        bring_up_lo: supports_userns && matches!(policy.network, NetworkMode::Loopback),
        landlock_ruleset,
    })
}

/// Fork a tiny probe child that attempts `unshare(CLONE_NEWUSER)` and
/// reports back via its exit code. ~1 ms per `harden()`.
fn probe_userns_supported() -> bool {
    use nix::sys::wait::{waitpid, WaitStatus};
    use nix::unistd::{fork, ForkResult};

    match unsafe { fork() } {
        Ok(ForkResult::Parent { child }) => {
            matches!(waitpid(child, None), Ok(WaitStatus::Exited(_, 0)))
        }
        Ok(ForkResult::Child) => {
            let ok = unshare(CloneFlags::CLONE_NEWUSER).is_ok();
            unsafe { libc::_exit(if ok { 0 } else { 1 }) };
        }
        Err(_) => false,
    }
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

    let mut write_paths: Vec<&Path> = policy
        .extra_fs_write
        .iter()
        .map(|p| p.as_path())
        .collect();
    if let Some(workdir) = policy.session_workdir.as_ref() {
        write_paths.push(workdir.as_path());
    }
    ruleset = ruleset.add_rules(path_beneath_rules(write_paths, AccessFs::from_all(abi)))?;
    Ok(ruleset)
}

fn read_paths() -> Vec<&'static str> {
    // Conservatively allow read of /etc as a whole. Per-file Unix permissions
    // already protect actually-sensitive entries (/etc/shadow, /etc/ssh/*_key,
    // /etc/sudoers are mode 600/640). Trying to allowlist only "safe" /etc
    // paths breaks too many libraries (matplotlib reads /etc/matplotlibrc, apt
    // hooks read /etc/apt/*, openssl reads /etc/ssl/openssl.cnf, etc.).
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

/// Runs inside the child's pre_exec. Async-signal-safe.
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

/// After `unshare(CLONE_NEWUSER)`, the new user ns has no UID mapping at
/// all - we can't even open files until we set one up. Map UID 0 inside
/// the ns to the caller's real UID (captured in the parent before unshare).
/// Order is fixed by the kernel: write `setgroups=deny` BEFORE `gid_map`.
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

/// Render `"0 <id> 1\n"` into a stack buffer. itoa handles the only
/// non-trivial part (u32 to decimal) without allocation.
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

/// Open an AF_INET dgram socket, ioctl SIOCSIFFLAGS with IFF_UP|IFF_RUNNING,
/// close. Async-signal-safe (only syscalls). Kept on raw libc because nix
/// doesn't model the `ifr_ifru` union cleanly.
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
