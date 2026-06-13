---
title: Sandbox
description: OS-level sandbox applied to model-generated code execution.
---

mistral.rs runs model-generated Python code in a persistent kernel. To keep that code from doing damage, the spawned Python process is hardened with an OS-level sandbox.

This is only supported on macOS and Linux environments.

:::caution
The CLI and TOML default to sandboxing on Linux and macOS. The Python and Rust SDKs default to **no sandbox** - an embedding application must construct and attach a `SandboxPolicy` itself.
:::

On by default for the CLI on Linux and macOS. To tune it, use `--sandbox` / the `[sandbox]` TOML table (see [Configuration](#configuration)). To turn it off, see [Disabling](#disabling). For a full setup example, see [enable code execution](/mistral.rs/guides/agents/enable-code-execution/).

The sandbox is not the same as permissioning. `--agent-permission ask` or `deny` decides whether model-requested agent actions are allowed to start. The sandbox controls what the subprocess can access after it starts. See [permissions and approvals](/mistral.rs/guides/agents/permissions-and-approvals/) for the cross-API approval model.

## Threat model

The sandbox targets **model misbehavior**: a confused or jailbroken model generating Python that could:

- delete or read arbitrary files (`rm -rf ~`, `~/.ssh/id_rsa`, `.env`)
- exfiltrate data over the network
- fork-bomb, allocate 200 GB, fill the disk
- pivot to other binaries via `subprocess`
- attach to host processes via `ptrace`
- load kernel modules, manipulate mounts, etc.

It is not a substitute for OS-level isolation against a determined attacker who can choose arbitrary code. For high-assurance deployments (multi-tenant, untrusted prompts, regulated data), also isolate the mistral.rs process itself with a container or VM, a dedicated low-privilege user, and constrained network egress. `--tool-dispatch-url` (run code on a separate host) is the alternative when code execution should leave the mistral.rs host entirely.

## Defaults

The CLI and TOML configuration default to `auto`: enabled on Linux and macOS, and a no-op with a warning elsewhere. The three modes:

- `off` - no sandbox.
- `auto` - apply whichever layers the host supports (Landlock, seccomp, namespaces; see [Linux details](#linux-details)).
- `on` - same as `auto`, but a missing layer becomes a hard error at code-execution initialization instead of being skipped.

The programmatic surfaces behave differently: `CodeExecutionConfig` in the Python and Rust SDKs defaults to **no sandbox**. Omitting `sandbox_policy` (or passing `None`) is equivalent to `--sandbox off`; the sandbox engages only when a `SandboxPolicy` is constructed and attached. An application embedding mistral.rs as a library does not inherit the safer CLI default and is responsible for choosing a policy.

The default policy:

| field | default |
|---|---|
| `max_memory_mb` | 2048 |
| `max_cpu_secs` | 300 |
| `max_procs` | 64 (additional tasks for the run UID on Linux; see [Linux details](#linux-details)) |
| `max_open_fds` | 1024 |
| `max_file_sz_mb` | 256 |
| `network` | `loopback` |

On macOS, the resource cap fields are accepted for configuration compatibility but are not enforced by Seatbelt. Filesystem and network isolation still apply.

## Configuration

CLI flags (`--sandbox`, `--sb-max-memory-mb`, `--sb-max-cpu-secs`, `--sb-max-procs`, `--sandbox-network`) and the `[sandbox]` TOML table expose the common controls: mode, memory, CPU, process count, and network. The programmatic `SandboxPolicy` also exposes open-file and written-file-size caps. Schemas: [TOML configuration](/mistral.rs/reference/cli-toml-config/#sandbox-section), [generated CLI reference](/mistral.rs/reference/cli/serve/). A worked `mistralrs serve` example is in [enable code execution](/mistral.rs/guides/agents/enable-code-execution/).

The `MISTRALRS_SANDBOX={auto|on|off}` env var sits between the two: lower precedence than an explicit CLI/TOML mode, higher than the default `auto`.

A working directory chosen with `--code-exec-workdir` is made writable inside the sandbox and shared across sessions: anything written there persists and is visible to subsequent sessions.

## Linux details

Applied in order:

1. **Env scrub.** All inherited env vars are dropped; only a small allowlist (`PATH`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TERM`, `HOME`, `TMPDIR`, `PYTHONHASHSEED`, `PYTHONIOENCODING`, `PYTHONUNBUFFERED`) is replayed. Secrets such as `HF_TOKEN`, `HF_HOME`, `HF_HUB_CACHE`, `AWS_*`, and `OPENAI_API_KEY` are not included by default. `HOME` and the XDG cache/config/data dirs are re-pointed at the session workdir.
2. **Process-count rlimit.** `RLIMIT_NPROC` is applied before namespace setup. Linux counts it per real UID, so `max_procs` is applied as additional task headroom above the current UID task count, then clamped to the inherited hard limit.
3. **Namespaces** (when unprivileged user namespaces are available).
   `unshare(CLONE_NEWUSER|CLONE_NEWIPC|CLONE_NEWUTS)` plus `CLONE_NEWNET` when `network = "loopback"` and network namespaces are available.
   UID 0 inside the ns is mapped to the caller's UID outside.
   PID namespace isolation is not applied: `unshare(CLONE_NEWPID)` only affects future children of the calling thread, and we're already past the fork that became the Python process. Real PID isolation would require a launcher binary.
4. **Bring up loopback.** If `network = "loopback"` uses a network namespace, `ioctl(SIOCSIFFLAGS)` brings up `lo` inside the new netns. If `network = "none"`, no network namespace is required because seccomp denies `socket(2)`.
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

Fallback when namespaces or seccomp are unavailable:

- No unprivileged user or network namespaces: the remaining available layers still apply, minus the missing namespace layer.
- `network = "loopback"` without a network namespace: there is no network isolation; use `network = "none"` to deny `socket(2)` without network namespaces.
- No seccomp: `network = "none"` cannot be enforced by the sandbox.
- Want a hard failure instead of silent fallback: set `mode = "on"` instead of the default `"auto"`, which turns missing filesystem or requested network isolation into a hard error at code-execution initialization.

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

For end-to-end setup and the Rust/Python `SandboxPolicy` snippets, see [enable code execution](/mistral.rs/guides/agents/enable-code-execution/). Python types are documented in the [Python API reference](/mistral.rs/reference/python/code-execution/). Remember the default: programmatic use is unsandboxed until a `SandboxPolicy` is attached.
