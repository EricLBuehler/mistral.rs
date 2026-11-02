---
title: Installer
description: Installer settings, automation, install locations, and maintenance.
---

The [quickstart](/quickstart/) takes you through your first installation and chat. This page is for choosing installer settings, automating setup, or maintaining an existing installation.

## Guided installation

The installer recommends settings for your hardware. You can review them before installing or choose Customize to make changes. New installations use the latest stable release, including when building from source. Your choices of release, acceleration, and installation method are saved for the next time you run it.

## Command-line options

Use these options to choose settings directly or automate installation. They take precedence over saved settings and legacy environment variables.

| macOS / Linux | Windows PowerShell | Purpose |
|---|---|---|
| `--release VERSION` | `-Release VERSION` | Choose `latest`, `master`, or a release tag such as `v0.9.5`. |
| `--backend BACKEND` | `-Backend BACKEND` | Choose `auto`, `cpu`, or `cuda`; macOS also supports `metal`. |
| `--from-source` | `-FromSource` | Build the selected release instead of downloading a binary. |
| `--with-ffmpeg` | `-WithFFmpeg` | Install optional FFmpeg support for [video input](/guides/models/video-setup/). |
| `--no-nccl` | `-NoNccl` | Omit NCCL from source builds. |
| `--yes`, `-y` | `-Yes` | Accept the selected plan and prerequisite-installation prompts. |
| `--dry-run` | `-DryRun` | Preview settings without downloading packages or changing files. |
| `--verbose` | `-Verbose` | Stream full command output during installation. |
| `--help`, `-h` | `-Help` | Show available options. |

For CUDA builds, use `--allow-cuda-mismatch` (`-AllowCudaMismatch` on Windows) to allow a toolkit newer than the driver supports.

On macOS and Linux, pass options after `sh -s --`:

```bash
curl -fsSL https://mistralrs.dev/install.sh | sh -s -- --release v0.9.5 --yes
```

In PowerShell, use a script block to pass arguments:

```powershell
& ([scriptblock]::Create((irm https://mistralrs.dev/install.ps1))) -Release v0.9.5 -Yes
```

Existing automation can continue using the [installer environment variables](/reference/environment-variables/#installer-automation).

## Acceleration and source builds

By default, the installer selects a prebuilt binary when one is available for your hardware. See [hardware support](/reference/hardware-support/) for the available combinations. Source builds need the [platform prerequisites](/developer/from-source/#platform-prerequisites); prebuilt binaries do not need Rust.

If installation fails, the installer prints the diagnostics and saves a log. The [troubleshooting guide](/reference/troubleshooting/#installer-download-or-build-fails) covers the next steps.

## Install locations and maintenance

Prebuilt and source installs use the same locations:

| Platform | Managed files | Command location |
|---|---|---|
| macOS / Linux | `~/.mistralrs` | `~/.local/bin/mistralrs` |
| Windows | `%USERPROFILE%\.mistralrs` | `%USERPROFILE%\.local\bin\mistralrs.cmd` |

Saved settings live in `install.conf` inside the managed directory.

The installer adds the command directory to PATH. Open a new terminal or follow the activation instructions it prints to use the command in your current shell. If the command is still unavailable, see [PATH troubleshooting](/reference/troubleshooting/#mistralrs-command-not-found-after-install).

On macOS and Linux, run `mistralrs update` to update an installation and `mistralrs uninstall` to remove it. You can also rerun the installer to change your saved settings.

On Windows, stop running `mistralrs` processes before rerunning the installer to update. To remove it, run `mistralrs uninstall` for manual removal instructions.
