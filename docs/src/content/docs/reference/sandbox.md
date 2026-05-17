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

For high-assurance deployments, also isolate the mistral.rs process itself with a container or VM, a dedicated low-privilege user, and constrained network egress.

## Defaults

The CLI and TOML configuration default to `auto`: enabled on Linux and macOS, and a no-op with a warning elsewhere. The Python API disables sandboxing unless you pass a `SandboxPolicy`.

The default policy:

| field | default |
|---|---|
| `max_memory_mb` | 2048 |
| `max_cpu_secs` | 300 |
| `max_procs` | 64 |
| `max_open_fds` | 1024 |
| `max_file_sz_mb` | 256 |
| `network` | `loopback` |

## Configuration

CLI/TOML expose the common controls: mode, memory, CPU, process count, and network. Programmatic `SandboxPolicy` also exposes open-file and written-file-size caps.

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

Concrete `mistralrs serve` example:

```bash
mistralrs serve \
  -m mistralrs-community/gemma-4-E4B-it-UQFF \
  --from-uqff 8 \
  --enable-code-execution \
  --sandbox on \
  --sandbox-network none \
  --sb-max-memory-mb 2048 \
  --code-exec-workdir . \
  --enable-search \
  --ui
```

`--sandbox on` makes missing sandbox support a hard error when code execution initializes. `--sandbox-network none` blocks network access from model-generated Python; web search still runs through the server-side search tool. `--code-exec-workdir .` chooses the working/output directory and is made writable inside the sandbox.

**Env var** (lower precedence than an explicit CLI flag, higher than the
default `auto` mode):

```
MISTRALRS_SANDBOX={auto|on|off}
```

## Linux details

Applied in order:

1. **Env scrub.** All inherited env vars are dropped; only a small allowlist (`PATH`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TERM`, `HOME`, `TMPDIR`, `PYTHONHASHSEED`, `PYTHONIOENCODING`, `PYTHONUNBUFFERED`) is replayed. Secrets such as `HF_TOKEN`, `HF_HOME`, `HF_HUB_CACHE`, `AWS_*`, and `OPENAI_API_KEY` are not included by default. `HOME` and the XDG cache/config/data dirs are re-pointed at the session workdir.
2. **Namespaces** (when unprivileged user namespaces are available).
   `unshare(CLONE_NEWUSER|CLONE_NEWIPC|CLONE_NEWUTS)` plus `CLONE_NEWNET` when `network != full`.
   UID 0 inside the ns is mapped to the caller's UID outside.
   PID namespace isolation is not applied: `unshare(CLONE_NEWPID)` only affects future children of the calling thread, and we're already past the fork that became the Python process. Real PID isolation would require a launcher binary.
3. **Loopback up.** If `network = loopback`, `ioctl(SIOCSIFFLAGS)` brings up `lo` inside the new netns.
4. **Landlock** (kernel 5.13+). Read access is allowed to a static set of system paths (`/usr`, `/lib`, `/lib64`, `/bin`, `/sbin`, `/etc`, `/opt`, `/proc/self`, selected `/sys` CPU info, and null/random/zero devices). The per-session workdir gets read+write access. Anything else returns `EACCES`.
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

If unprivileged user namespaces are disabled on the host, the sandbox falls back to rlimits + env scrub + seccomp + Landlock without IPC/UTS namespaces. For `network = "loopback"`, that also means no network namespace; use `network = "none"` to deny `socket(2)` without user namespaces. To make missing filesystem isolation or requested network isolation a hard error during code-execution initialization, set `mode = "on"` instead of the default `"auto"`.

`HF_TOKEN`, `HF_HOME`, and `HF_HUB_CACHE` are deliberately excluded from the default env allowlist: model-generated code can print env vars before any network restriction kicks in. To pass other tokens or secrets through, list them in `extra_env`.

## macOS details

Argv is wrapped with `sandbox-exec -p <profile>`. The generated SBPL profile denies by default, allows read access to system paths and configured read paths, allows read/write access to configured write paths and the session workdir, and gates network per policy.

CPU time, open files, written-file size, and core dumps are controlled with `setrlimit`. Address-space and process-count limits are best-effort on macOS because they are evaluated before Python `exec`, against the forked server process and the current user's process count. If those values already exceed the requested cap, mistral.rs keeps the Seatbelt sandbox and the other rlimits rather than failing the Python launch.

## Disabling

Set `mode = "off"` in the TOML, `--sandbox off` on the CLI, or `MISTRALRS_SANDBOX=off` in the env.

A startup warning is logged, and this restores pre-sandbox behavior: model-generated code has full filesystem, network, and subprocess access.

## Programmatic use

For end-to-end code execution setup, see [enable code execution](/mistral.rs/guides/agents/enable-code-execution/). The checked-in examples cover [Python](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/code_execution.py), [Rust](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/code_execution/main.rs), and [Rust file outputs](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/code_execution_files/main.rs). Python types are documented in the [Python API reference](/mistral.rs/reference/python/code-execution/).

Rust:

```rust
use mistralrs::{CodeExecutionConfig, NetworkMode, SandboxPolicy};

let cfg = CodeExecutionConfig {
    sandbox_policy: Some(SandboxPolicy {
        max_memory_mb: 1024,
        network: NetworkMode::None,
        ..SandboxPolicy::default()
    }),
    ..CodeExecutionConfig::default()
};
```

Python:

```python
from mistralrs import CodeExecutionConfig, Runner, SandboxPolicy, Which

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    code_execution_config=CodeExecutionConfig(
        sandbox_policy=SandboxPolicy(
            max_memory_mb=1024,
            network="none",   # "none" | "loopback" | "full"
        ),
    ),
)
```

Omit `sandbox_policy` (or pass `None`) to disable the sandbox entirely in programmatic use.
