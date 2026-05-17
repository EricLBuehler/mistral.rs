//! seccomp-bpf deny-list filter.

use std::collections::BTreeMap;
use std::io;

use seccompiler::{
    apply_filter_all_threads, BackendError, BpfProgram, SeccompAction, SeccompCmpArgLen,
    SeccompCmpOp, SeccompCondition, SeccompFilter, SeccompRule, TargetArch,
};

use crate::NetworkMode;

#[cfg(target_arch = "x86_64")]
const ARCH: TargetArch = TargetArch::x86_64;
#[cfg(target_arch = "aarch64")]
const ARCH: TargetArch = TargetArch::aarch64;

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

#[cfg(target_arch = "x86_64")]
const ARCH_DENIED: &[i64] = &[libc::SYS_ioperm, libc::SYS_iopl, libc::SYS_nfsservctl];

#[cfg(target_arch = "aarch64")]
const ARCH_DENIED: &[i64] = &[];

pub(crate) fn build(network: NetworkMode) -> Result<BpfProgram, io::Error> {
    let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();
    for nr in COMMON_DENIED.iter().chain(ARCH_DENIED.iter()) {
        rules.insert(*nr, vec![]);
    }

    if network == NetworkMode::None {
        rules.insert(libc::SYS_socket, vec![]);
    } else if network == NetworkMode::Loopback {
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

pub(crate) fn install(program: &BpfProgram) -> io::Result<()> {
    let rc = unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
    if rc != 0 {
        return Err(io::Error::last_os_error());
    }
    apply_filter_all_threads(program).map_err(|e| io::Error::other(e.to_string()))
}

pub(crate) fn supported() -> bool {
    use nix::sys::wait::{waitpid, WaitStatus};
    use nix::unistd::{fork, ForkResult};

    let Ok(program) = build(NetworkMode::None) else {
        return false;
    };

    match unsafe { fork() } {
        Ok(ForkResult::Parent { child }) => {
            matches!(waitpid(child, None), Ok(WaitStatus::Exited(_, 0)))
        }
        Ok(ForkResult::Child) => {
            let ok = install(&program).is_ok();
            unsafe { libc::_exit(if ok { 0 } else { 1 }) };
        }
        Err(_) => false,
    }
}
