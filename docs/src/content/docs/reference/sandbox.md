---
title: Sandbox
description: OS-level sandbox applied to model-generated code execution.
---

mistral.rs runs model-generated Python code in a persistent kernel. To keep that code from doing damage, the spawned Python process is hardened with an OS-level sandbox.

This is only supported on macOS and Linux environments.

The sandbox is not the same as permissioning. `--agent-permission ask` or `deny` decides whether model-requested agent actions are allowed to start. The sandbox controls what the subprocess can access after it starts. See [agent permissions](/mistral.rs/guides/agents/agentic-runtime/#agent-permissions) for the cross-API approval model.

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
| `max_procs` | 64 additional UID tasks on Linux |
| `max_open_fds` | 1024 |
| `max_file_sz_mb` | 256 |
| `network` | `loopback` |

On macOS, the resource cap fields are accepted for configuration compatibility but are not enforced by Seatbelt. Filesystem and network isolation still apply.

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
  --enable-search
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
2. **Process-count rlimit.** `RLIMIT_NPROC` is applied before namespace setup. Linux counts it per real UID, so `max_procs` is applied as additional task headroom above the current UID task count, then clamped to the inherited hard limit.
3. **Namespaces** (when unprivileged user namespaces are available).
   `unshare(CLONE_NEWUSER|CLONE_NEWIPC|CLONE_NEWUTS)` plus `CLONE_NEWNET` when `network = "loopback"` and network namespaces are available.
   UID 0 inside the ns is mapped to the caller's UID outside.
   PID namespace isolation is not applied: `unshare(CLONE_NEWPID)` only affects future children of the calling thread, and we're already past the fork that became the Python process. Real PID isolation would require a launcher binary.
4. **Loopback up.** If `network = "loopback"` uses a network namespace, `ioctl(SIOCSIFFLAGS)` brings up `lo` inside the new netns. If `network = "none"`, no network namespace is required because seccomp denies `socket(2)`.
5. **Landlock** (kernel 5.13+). Read access is allowed to a static set of system paths (`/usr`, `/lib`, `/lib64`, `/bin`, `/sbin`, `/etc`, `/opt`, `/proc/self`, selected `/sys` CPU info, and null/random/zero devices). The per-session workdir gets read+write access. Anything else returns `EACCES`.
6. **rlimits.** `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_NOFILE`, `RLIMIT_FSIZE` per policy, clamped to the inherited hard limit. `RLIMIT_CORE = 0`.
7. **seccomp-bpf deny-list** (when filter install is available). Returns `EPERM` for: `ptrace`, `mount`, `umount2`, `pivot_root`, `chroot`, `unshare`, `setns`, `keyctl`,
   `add_key`, `request_key`, `bpf`, `perf_event_open`, `kexec_load`, `init_module`, `finit_module`, `delete_module`, `reboot`, `swapon`,
   `swapoff`, `clock_settime`, `settimeofday`, `setdomainname`, `sethostname`, `acct`, `quotactl`, `io_uring_setup`, plus `ioperm`,
   `iopl`, `nfsservctl` on x86_64.

   When `network = none`, `socket` is also denied. When `network = loopback`, `AF_NETLINK` and `AF_PACKET` sockets are denied (everything else stays allowed inside the netns).

Best-effort additions:

- **cgroup v2.** When `/sys/fs/cgroup/cgroup.controllers` is delegated,
  a fresh scope is created with `memory.max` and `pids.max` set per
  policy, and the child PID is moved into it. Silently skipped otherwise.

If unprivileged user or network namespaces are disabled on the host, the sandbox falls back to the remaining available layers without the unavailable namespace layer. For `network = "loopback"`, that also means no network namespace; use `network = "none"` to deny `socket(2)` without network namespaces. If seccomp itself is unavailable, `network = "none"` cannot be enforced by the sandbox. To make missing filesystem isolation or requested network isolation a hard error during code-execution initialization, set `mode = "on"` instead of the default `"auto"`.

`HF_TOKEN`, `HF_HOME`, and `HF_HUB_CACHE` are deliberately excluded from the default env allowlist: model-generated code can print env vars before any network restriction kicks in. To pass other tokens or secrets through, list them in `extra_env`.

## macOS details

Argv is wrapped with `sandbox-exec -p <profile>`. The generated SBPL profile denies by default, allows Python/runtime reads from system paths, dyld and timezone databases, Homebrew/MacPorts prefixes, standard device files, and configured read paths. The session workdir and configured write paths get read/write access.

The profile also allows the native startup operations CPython and extension modules commonly need, including Mach, IOKit, sysctl, file metadata, file ioctl, and executable file maps. These allowances are for process startup and library loading; writes remain limited to the workdir and configured write paths.

Network follows the configured policy: `none` emits no network rules, `loopback` allows localhost endpoints, and `full` allows `network*`.

Resource rlimits are not applied on macOS. Applying them from the server requires a `pre_exec` hook, which forces a fork path from an already-running multithreaded process before Python starts. mistral.rs keeps the Seatbelt sandbox for filesystem and network isolation; use a container or VM when macOS deployments need hard memory, CPU, or process-count caps.

## Disabling

Set `mode = "off"` in the TOML, `--sandbox off` on the CLI, or `MISTRALRS_SANDBOX=off` in the env.

A startup warning is logged. With all sandbox layers off, model-generated code has full filesystem, network, and subprocess access as the mistralrs user.

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
from mistralrs import CodeExecutionConfig, NetworkMode, Runner, SandboxPolicy, Which

runner = Runner(
    which=Which.Plain(model_id="Qwen/Qwen3-4B"),
    code_execution_config=CodeExecutionConfig(
        sandbox_policy=SandboxPolicy(
            max_memory_mb=1024,
            network=NetworkMode.NoNetwork,
        ),
    ),
)
```

Omit `sandbox_policy` (or pass `None`) to disable the sandbox entirely in programmatic use.
