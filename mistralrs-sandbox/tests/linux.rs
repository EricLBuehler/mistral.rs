#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};

use mistralrs_sandbox::{detect, NetworkMode, SandboxPolicy};
use tokio::process::Command;

fn workdir() -> PathBuf {
    let p = std::env::temp_dir().join(format!("mistralrs-sandbox-test-{}", std::process::id()));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn base_policy() -> SandboxPolicy {
    let mut p = SandboxPolicy {
        max_memory_mb: 256,
        max_cpu_secs: 10,
        max_procs: 32,
        max_open_fds: 256,
        max_file_sz_mb: 16,
        network: NetworkMode::Loopback,
        session_workdir: Some(workdir()),
        ..SandboxPolicy::default()
    };
    if let Some(py) = which("python3") {
        for prefix in resolve_python_prefixes(&py) {
            p.extra_fs_read.push(prefix);
        }
    }
    p
}

fn resolve_python_prefixes(python_path: &Path) -> Vec<PathBuf> {
    let out = std::process::Command::new(python_path)
        .args(["-c", PYTHON_PREFIX_PROBE])
        .output();
    let Ok(out) = out else { return Vec::new() };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let s = l.trim();
            if s.is_empty() {
                return None;
            }
            let p = PathBuf::from(s);
            let p = if p.is_file() {
                p.parent()?.parent()?.to_path_buf()
            } else {
                p
            };
            if p.exists() {
                Some(p)
            } else {
                None
            }
        })
        .collect()
}

const PYTHON_PREFIX_PROBE: &str = concat!(
    "import sys, site; ",
    "print(sys.prefix); ",
    "print(sys.base_prefix); ",
    "print(sys.executable); ",
    "print(site.getusersitepackages())",
);

#[tokio::test]
async fn detect_returns_linux_impl() {
    let sb = detect();
    assert_eq!(sb.name(), "linux");
}

#[tokio::test]
async fn rlimit_as_caps_address_space() {
    let sb = detect();
    let mut policy = base_policy();
    policy.max_memory_mb = 128;

    let mut cmd = Command::new("/bin/sh");
    cmd.arg("-c").arg("ulimit -v");
    sb.harden(&mut cmd, &policy).expect("harden");

    let out = cmd.output().await.expect("output");
    let s = String::from_utf8_lossy(&out.stdout);
    let kb: u64 = s.trim().parse().unwrap_or(0);
    assert_eq!(
        kb,
        128 * 1024,
        "RLIMIT_AS in KB should be 128 MiB = 131072 KB"
    );
}

#[tokio::test]
async fn rlimit_nproc_caps_processes() {
    let sb = detect();
    let mut policy = base_policy();
    policy.max_procs = 17;

    if which("python3").is_none() {
        eprintln!("skipping: python3 not on PATH");
        return;
    }
    let mut cmd = Command::new("python3");
    cmd.arg("-c").arg(
        "import resource, threading\n\
         t = threading.Thread(target=lambda: None)\n\
         t.start(); t.join()\n\
         print(resource.getrlimit(resource.RLIMIT_NPROC)[0])",
    );
    sb.harden(&mut cmd, &policy).expect("harden");

    let out = cmd.output().await.expect("output");
    assert!(
        out.status.success(),
        "child should be able to start one thread; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let n: u64 = s.trim().parse().unwrap_or(0);
    assert!(n >= 17, "RLIMIT_NPROC should include test headroom: {s:?}");
}

#[tokio::test]
async fn seccomp_blocks_ptrace() {
    if which("python3").is_none() {
        eprintln!("skipping: python3 not on PATH");
        return;
    }

    let sb = detect();
    if !seccomp_available(sb.as_ref()) {
        eprintln!("skipping: seccomp unavailable on this host");
        return;
    }
    let policy = base_policy();
    let mut cmd = Command::new("python3");
    cmd.arg("-c").arg(
        "import ctypes, sys; libc = ctypes.CDLL('libc.so.6', use_errno=True); \
         r = libc.ptrace(0, 0, 0, 0); print(r, ctypes.get_errno())",
    );
    sb.harden(&mut cmd, &policy).expect("harden");

    let out = cmd.output().await.expect("output");
    let s = String::from_utf8_lossy(&out.stdout);
    let errno = s
        .split_whitespace()
        .nth(1)
        .and_then(|t| t.parse::<i32>().ok())
        .unwrap_or(0);
    assert_eq!(errno, libc::EPERM, "ptrace should be denied with EPERM");
}

#[tokio::test]
async fn network_none_blocks_socket() {
    if which("python3").is_none() {
        eprintln!("skipping: python3 not on PATH");
        return;
    }

    let sb = detect();
    let mut policy = base_policy();
    policy.network = NetworkMode::None;
    if !sb.effective(&policy).network_isolated {
        eprintln!("skipping: network=none isolation unavailable on this host");
        return;
    }
    let mut cmd = Command::new("python3");
    cmd.arg("-c").arg(
        "import socket\n\
         try:\n  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n  print('opened')\n\
         except OSError as e:\n  print('blocked', e.errno)",
    );
    sb.harden(&mut cmd, &policy).expect("harden");

    let out = cmd.output().await.expect("output");
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(
        s.contains("blocked"),
        "socket(AF_INET) should fail; got: {s}"
    );
}

#[tokio::test]
async fn unshare_is_denied_inside_child() {
    if which("python3").is_none() {
        eprintln!("skipping: python3 not on PATH");
        return;
    }

    let sb = detect();
    if !seccomp_available(sb.as_ref()) {
        eprintln!("skipping: seccomp unavailable on this host");
        return;
    }
    let policy = base_policy();
    let mut cmd = Command::new("python3");
    cmd.arg("-c").arg(
        "import ctypes, ctypes.util\n\
         libc = ctypes.CDLL('libc.so.6', use_errno=True)\n\
         r = libc.unshare(0x10000000)  # CLONE_NEWNS\n\
         print(r, ctypes.get_errno())",
    );
    sb.harden(&mut cmd, &policy).expect("harden");

    let out = cmd.output().await.expect("output");
    let s = String::from_utf8_lossy(&out.stdout);
    let errno = s
        .split_whitespace()
        .nth(1)
        .and_then(|t| t.parse::<i32>().ok())
        .unwrap_or(0);
    assert_eq!(errno, libc::EPERM, "unshare should be denied with EPERM");
}

fn which(prog: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join(prog))
            .find(|p| p.is_file())
    })
}

fn seccomp_available(sb: &dyn mistralrs_sandbox::Sandbox) -> bool {
    let mut policy = base_policy();
    policy.network = NetworkMode::None;
    sb.effective(&policy).network_isolated
}
