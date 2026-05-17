//! seccomp-bpf deny-list filter.
//!
//! Built in the parent process (the `BpfProgram` is just a `Vec<sock_filter>`)
//! and installed inside the child's `pre_exec` hook via `apply_filter_all_threads`.
//! Install LAST — anything we deny here cannot be undone, so all setup
//! syscalls (unshare, mount, setrlimit, etc.) must run before this.

use std::collections::BTreeMap;
use std::io;

use seccompiler::{
    apply_filter_all_threads, BackendError, BpfProgram, SeccompAction, SeccompCmpArgLen,
    SeccompCmpOp, SeccompCondition, SeccompFilter, SeccompRule, TargetArch,
};

use crate::NetworkMode;

/// Detect the running architecture for the seccomp filter at compile time.
#[cfg(target_arch = "x86_64")]
const ARCH: TargetArch = TargetArch::x86_64;
#[cfg(target_arch = "aarch64")]
const ARCH: TargetArch = TargetArch::aarch64;

/// Syscalls denied unconditionally on all supported arches.
const COMMON_DENIED: &[i64] = &[
    libc::SYS_ptrace,
    libc::SYS_mount,
    libc::SYS_umount2,
    libc::SYS_pivot_root,
    libc::SYS_chroot,
    libc::SYS_unshare,
    libc::SYS_setns,
    libc::SYS_keyctl,
    libc::SYS_add_key,
    libc::SYS_request_key,
    libc::SYS_bpf,
    libc::SYS_perf_event_open,
    libc::SYS_kexec_load,
    libc::SYS_init_module,
    libc::SYS_finit_module,
    libc::SYS_delete_module,
    libc::SYS_reboot,
    libc::SYS_swapon,
    libc::SYS_swapoff,
    libc::SYS_clock_settime,
    libc::SYS_settimeofday,
    libc::SYS_setdomainname,
    libc::SYS_sethostname,
    libc::SYS_acct,
    libc::SYS_quotactl,
    libc::SYS_io_uring_setup,
];

/// Arch-specific denials. `ioperm`/`iopl` only exist on x86_64;
/// `nfsservctl` was removed in modern aarch64 libc.
#[cfg(target_arch = "x86_64")]
const ARCH_DENIED: &[i64] = &[libc::SYS_ioperm, libc::SYS_iopl, libc::SYS_nfsservctl];

#[cfg(target_arch = "aarch64")]
const ARCH_DENIED: &[i64] = &[];

/// Build the BPF program for a given network mode.
///
/// Default action = Allow (allowlist-by-deny). Denied syscalls return
/// EPERM rather than SIGKILL so Python can surface them as OSError instead
/// of disappearing the interpreter mid-session.
pub(crate) fn build(network: NetworkMode) -> Result<BpfProgram, io::Error> {
    let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();
    for nr in COMMON_DENIED.iter().chain(ARCH_DENIED.iter()) {
        rules.insert(*nr, vec![]);
    }

    if network == NetworkMode::None {
        // No sockets at all. socket(2) is the gatekeeper; deny it outright.
        // socketpair(2) is still allowed because Python's stdlib uses it for
        // some internal IPC and it can't reach the network.
        rules.insert(libc::SYS_socket, vec![]);
    } else if network == NetworkMode::Loopback {
        // Permit AF_UNIX and AF_INET/AF_INET6 (the netns will scope routing
        // to lo). Deny AF_NETLINK + AF_PACKET so the child can't poke kernel
        // routing or sniff packets even inside its own netns.
        let denied_families = [libc::AF_NETLINK, libc::AF_PACKET];
        for family in denied_families {
            let cond =
                SeccompCondition::new(0, SeccompCmpArgLen::Dword, SeccompCmpOp::Eq, family as u64)
                    .map_err(|e| io::Error::other(e.to_string()))?;
            let rule = SeccompRule::new(vec![cond]).map_err(|e| io::Error::other(e.to_string()))?;
            rules.entry(libc::SYS_socket).or_default().push(rule);
        }
    }

    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,
        SeccompAction::Errno(libc::EPERM as u32),
        ARCH,
    )
    .map_err(|e| io::Error::other(e.to_string()))?;

    filter
        .try_into()
        .map_err(|e: BackendError| io::Error::other(e.to_string()))
}

/// Install a prebuilt filter in the calling thread. Async-signal-safe
/// because the program is already constructed; this just makes prctl + seccomp
/// syscalls.
pub(crate) fn install(program: &BpfProgram) -> io::Result<()> {
    let rc = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    apply_filter_all_threads(program).map_err(|e| io::Error::other(e.to_string()))
}
