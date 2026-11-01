---
title: Sandbox
description: OS-level sandbox applied to model-generated code and shell execution.
---

The sandbox limits filesystem, network, and resource access for model-generated Python and shell commands on Linux and macOS.

:::caution
The CLI and TOML default to sandboxing on Linux and macOS. The Python and Rust SDKs default to **no sandbox** - an embedding application must construct and attach a `SandboxPolicy` itself.
:::

Configure it with `--sandbox` / the `[sandbox]` TOML table (see [Configuration](#configuration)). To turn it off, see [Disabling](#disabling). For full setup examples, see [enable code execution](/guides/agents/enable-code-execution/) and [enable shell execution](/guides/agents/enable-shell/).

The sandbox is not the same as permissioning. `--agent-permission ask` or `deny` decides whether model-requested agent actions are allowed to start. The sandbox controls what the subprocess can access after it starts. See [permissions and approvals](/guides/agents/permissions-and-approvals/) for the cross-API approval model.

## Threat model

The sandbox limits accidental or model-requested access to host files, networks, processes, and resources.

It is not a substitute for OS-level isolation against a determined attacker who can choose arbitrary code or commands. For high-assurance deployments (multi-tenant, untrusted prompts, regulated data), also isolate the mistral.rs process itself with a container or VM, a dedicated low-privilege user, and constrained network egress. `--tool-dispatch-url` is the alternative when execution should leave the mistral.rs host entirely.

## Defaults

The CLI and TOML configuration default to `auto`: enabled on Linux and macOS, and a no-op with a warning elsewhere. The three modes:

- `off` - no sandbox.
- `auto` - apply whichever layers the host supports (Landlock, seccomp, namespaces; see [Linux details](#linux-details)).
- `on` - same as `auto`, but a missing layer becomes a hard error at code-execution initialization instead of being skipped.

The sandbox profile controls the filesystem and environment allowlist inside that mode:

- `developer` - default when agent, code execution, or shell execution is enabled. It keeps writes limited to the session workdir, but allows reads from common local toolchain roots such as Conda/venv, pyenv, rustup, nvm, Homebrew, Nix, Java, Bun, Deno, and selected compiler/library search paths. Its default network mode is `full`.
- `restricted` - for tighter deployments. It allows system/runtime reads plus the session workdir and explicitly configured paths. Its default network mode is `loopback`.

Native Python and Rust applications must attach a `SandboxPolicy` to `CodeExecutionConfig` or `ShellConfig`; CLI defaults are not inherited.

The restricted profile default policy:

| field | default |
|---|---|
| `max_memory_mb` | 2048 |
| `max_cpu_secs` | 600 |
| `max_procs` | 64 (additional tasks for the run UID on Linux; see [Linux details](#linux-details)) |
| `max_open_fds` | 1024 |
| `max_file_sz_mb` | 256 |
| `network` | `loopback` |

The developer profile uses the same limits and defaults to `network = "full"` unless `network` is explicitly set. When rlimits apply and a configured tool timeout exceeds `max_cpu_secs`, code and shell execution raise `max_cpu_secs` to match and log a warning.

On macOS, the resource limits are accepted but not enforced. Filesystem and network isolation still apply.

## Choosing a profile

The examples below are TOML configuration snippets for `mistralrs from-config -f <path>`. See the full [TOML configuration reference](/reference/cli-toml-config/) for where these sections fit in a complete config file.

For local agent use, start with `developer`:

```toml
[runtime]
agent = true

[sandbox]
mode = "auto"
profile = "developer"
```

`developer` is already the default when `agent`, `enable_code_execution`, or `enable_shell` is on, so the `profile` line is optional.

Use `restricted` when you want tighter filesystem and network defaults:

```toml
[sandbox]
mode = "auto"
profile = "restricted"
network = "loopback"
```

Use `mode = "on"` when startup should fail if requested sandbox layers are unavailable:

```toml
[sandbox]
mode = "on"
profile = "restricted"
network = "none"
```

Use `mode = "off"` only when you intentionally want subprocesses to run with full host access:

```toml
[sandbox]
mode = "off"
```

The profile does not change the basic write rule: generated code and shell commands can write the session workdir and explicitly configured write paths, not arbitrary home directories.

| Profile | Best for | Reads | Writes | Default network |
|---|---|---|---|---|
| `developer` | Local agents, Skills, package managers, language runtimes, and project analysis tools. | System/runtime paths, discovered toolchain roots, `PATH` entries, configured read paths, and the session workdir. | Session workdir and configured write paths. | `full` |
| `restricted` | Production-style deployments and prompts from users you do not fully trust. | System/runtime paths, configured read paths, and the session workdir. | Session workdir and configured write paths. | `loopback` |

## Configuration

CLI flags (`--sandbox`, `--sandbox-profile`, `--sb-max-memory-mb`, `--sb-max-cpu-secs`, `--sb-max-procs`, `--sandbox-network`) and the `[sandbox]` TOML table expose the common controls: mode, profile, memory, CPU, process count, and network. The programmatic `SandboxPolicy` also exposes open-file and written-file-size caps. Schemas: [TOML configuration](/reference/cli-toml-config/#sandbox-section), [generated CLI reference](/reference/cli/serve/). Worked `mistralrs serve` examples are in [enable code execution](/guides/agents/enable-code-execution/) and [enable shell execution](/guides/agents/enable-shell/).

The `MISTRALRS_SANDBOX={auto|on|off}` env var overrides only the mode. It has lower precedence than an explicit CLI/TOML mode and higher precedence than the default `auto`. It does not choose the profile or network policy.

A working directory you choose is made writable inside the sandbox, and files written there persist after the session ends:

- `--code-exec-workdir <dir>`: every Python session uses `<dir>` itself, so sessions see each other's files.
- `--shell-workdir <dir>`: each shell session gets its own `<dir>/<session_id>` subdirectory.

## Linux details

Applied in order:

1. **Env scrub.** All inherited env vars are dropped; only a small allowlist (`PATH`, `LANG`, `LC_ALL`, `LC_CTYPE`, `TERM`, `HOME`, `TMPDIR`, `PYTHONHASHSEED`, `PYTHONIOENCODING`, `PYTHONUNBUFFERED`) is replayed. The developer profile also replays non-secret toolchain location variables such as `CONDA_PREFIX`, `VIRTUAL_ENV`, `PYENV_ROOT`, `NVM_DIR`, `RUSTUP_HOME`, and compiler/library search paths. Secrets such as `HF_TOKEN`, `HF_HOME`, `HF_HUB_CACHE`, `AWS_*`, and `OPENAI_API_KEY` are not included by default. `HOME` and the XDG cache/config/data dirs are re-pointed at the session workdir.
2. **Process-count rlimit.** `RLIMIT_NPROC` is applied before namespace setup. Linux counts processes per user, so `max_procs` allows that many processes beyond what the user already runs, capped at the inherited hard limit.
3. **Namespaces** (when unprivileged user namespaces are available).
   `unshare(CLONE_NEWUSER|CLONE_NEWIPC|CLONE_NEWUTS)` plus `CLONE_NEWNET` when `network = "loopback"` and network namespaces are available.
   UID 0 inside the ns is mapped to the caller's UID outside.
   There's no PID namespace, so process IDs aren't isolated from the host. Run mistral.rs in a container if you need that.
4. **Bring up loopback.** If `network = "loopback"` uses a network namespace, `ioctl(SIOCSIFFLAGS)` brings up `lo` inside the new netns. If `network = "none"`, no network namespace is required because seccomp denies `socket(2)`.
5. **Landlock** (kernel 5.13+). Read access is allowed to a static set of system paths (`/usr`, `/lib`, `/lib64`, `/bin`, `/sbin`, `/etc`, `/opt`, `/proc/self`, selected `/sys` CPU info, and null/random/zero devices). The developer profile adds read access to discovered toolchain roots. The per-session workdir gets read+write access. Anything else returns `EACCES`.
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

Argv is wrapped with `sandbox-exec -p <profile>`. The generated SBPL profile denies by default, allows runtime reads from system paths, dyld and timezone databases, Homebrew/MacPorts prefixes, standard device files, and configured read paths. The developer profile adds read access to discovered toolchain roots. The session workdir and configured write paths get read/write access.

The profile also allows native startup operations that Python and shell commands commonly need, including Mach, IOKit, sysctl, file metadata, file ioctl, and executable file maps. These allowances are for process startup and library loading; writes remain limited to the workdir and configured write paths.

Network follows the configured policy: `none` emits no network rules, `loopback` allows localhost endpoints, and `full` allows `network*`.

Resource limits aren't enforced on macOS. If you need hard memory, CPU, or process-count caps there, run mistral.rs in a container or VM.

## Disabling

Set `mode = "off"` in the TOML, `--sandbox off` on the CLI, or `MISTRALRS_SANDBOX=off` in the env.

A startup warning is logged. With all sandbox layers off, model-generated code and commands have full filesystem, network, and subprocess access as the mistralrs user.

## Programmatic use

For end-to-end setup and the Rust/Python `SandboxPolicy` snippets, see [enable code execution](/guides/agents/enable-code-execution/) and [enable shell execution](/guides/agents/enable-shell/). Python types are documented in the [Python API reference](/reference/python/code-execution/). Remember the default: programmatic use is unsandboxed until a `SandboxPolicy` is attached.
