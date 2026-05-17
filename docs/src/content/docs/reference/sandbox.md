---
title: Sandbox
description: OS-level sandbox applied to model-generated code execution.
---

mistral.rs runs model-generated Python code in a persistent kernel. To keep that code from doing damage, the spawned Python process is hardened with an OS-level sandbox.

This is only supported on macOS and Linux environments.

## Threat model

This is primarily to avoid cases where a confused or jailbroken model generating Python could:

- delete or read arbitrary files (`rm -rf ~`, `~/.ssh/id_rsa`, `.env`)
- exfiltrate data over the network
- fork-bomb, allocate 200 GB, fill the disk
- pivot to other binaries via `subprocess`
- attach to host processes via `ptrace`
- load kernel modules, manipulate mounts, etc.

This implementation should not be relied on 100% in environments where high security is a requirements, and alternative methods should be explored in those cases.

## Defaults

On Linux and macOS the sandbox is `auto` (enabled). On Windows it is a no-op with a one-time warning. The default policy:

| field | default |
|---|---|
| `max_memory_mb` | 2048 |
| `max_cpu_secs` | 300 |
| `max_procs` | 64 |
| `max_open_fds` | 1024 |
| `max_file_sz_mb` | 256 |
| `network` | `loopback` |

## Configuration

Two paths:

**TOML (`mistralrs from-config -f <toml>`):**

```toml
[sandbox]
mode          = "auto"      # "auto" | "on" | "off"
max_memory_mb = 2048
max_cpu_secs  = 300
max_procs     = 64
network       = "loopback"  # "none" | "loopback" | "full"
```

**CLI:**

```
--sandbox {auto|on|off}              default: auto
--sb-max-memory-mb <USIZE>
--sb-max-cpu-secs  <USIZE>
--sb-max-procs     <USIZE>
--sandbox-network  {none|loopback|full}
```

**Env var** (lower precedence than an explicit CLI flag, higher than the
TOML default):

```
MISTRALRS_SANDBOX={auto|on|off}
```

## What each layer does (Linux)

Applied in order:

1. **Env scrub.** All inherited env vars are dropped; only a small allowlist
   (`PATH`, `LANG`, `LC_ALL`, `TERM`, `HOME`, `TMPDIR`, `PYTHONHASHSEED`, `HF_TOKEN`, `HF_HOME`, `HF_HUB_CACHE`) is replayed.
2. **Namespaces** (when unprivileged user namespaces are available).
   `unshare(CLONE_NEWUSER|CLONE_NEWPID|CLONE_NEWIPC|CLONE_NEWUTS)` plus `CLONE_NEWNET` when `network != full`.
   UID 0 inside the ns is mapped to the caller's UID outside.
3. **Loopback up.** If `network = loopback`, `ioctl(SIOCSIFFLAGS)` brings up `lo` inside the new netns.
4. **Landlock** (kernel 5.13+). Read access is allowed to a static set of system paths (`/usr`, `/lib`, `/lib64`, `/etc/ssl/certs`, etc.) and the per-session workdir gets read+write. Anything else returns `EACCES`.
5. **rlimits.** `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_NOFILE`, `RLIMIT_NPROC`, `RLIMIT_FSIZE` per policy. `RLIMIT_CORE = 0`.
6. **seccomp-bpf deny-list.** Returns `EPERM` for: `ptrace`, `mount`, `umount2`, `pivot_root`, `chroot`, `unshare`, `setns`, `keyctl`,
   `add_key`, `request_key`, `bpf`, `perf_event_open`, `kexec_load`, `init_module`, `finit_module`, `delete_module`, `reboot`, `swapon`,
   `swapoff`, `clock_settime`, `settimeofday`, `setdomainname`, `sethostname`, `acct`, `quotactl`, `io_uring_setup`, plus `ioperm`,
   `iopl`, `nfsservctl` on x86_64.
   
   When `network = none`, `socket` is also denied. When `network = loopback`, `AF_NETLINK` and `AF_PACKET` sockets are denied (everything else stays allowed inside the netns).

Best-effort additions:

- **cgroup v2.** When `/sys/fs/cgroup/cgroup.controllers` is delegated,
  a fresh scope is created with `memory.max` and `pids.max` set per
  policy, and the child PID is moved into it. Silently skipped otherwise.

If unprivileged user namespaces are disabled on the host, the sandbox falls back to rlimits + env scrub + seccomp + Landlock (no PID/IPC/UTS/NET
isolation). A warning is logged once.

## What each layer does (macOS)

Argv is wrapped with `sandbox-exec -p <profile>`. The generated SBPL
profile denies by default, allows file-read on system paths and
write on the session workdir, and gates network per policy. rlimits are
applied via the same `setrlimit` calls as Linux.

## Disabling

Set `mode = "off"` in the TOML, `--sandbox off` on the CLI, or `MISTRALRS_SANDBOX=off` in the env.

A startup warning is logged, and this restores pre-sandbox behavior: model-generated code has full filesystem, network, and subprocess access.

## Programmatic use

```rust
use mistralrs_core::CodeExecutionConfig;
use mistralrs_sandbox::{NetworkMode, SandboxPolicy};

let cfg = CodeExecutionConfig {
    sandbox_policy: Some(SandboxPolicy {
        max_memory_mb: 1024,
        network: NetworkMode::None,
        ..SandboxPolicy::default()
    }),
    ..CodeExecutionConfig::default()
};
```
