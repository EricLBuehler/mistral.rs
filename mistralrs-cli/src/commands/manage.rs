//! Self-management for installer-managed installs: `update` and `uninstall`.

use anyhow::{bail, Context, Result};
use std::path::PathBuf;

const REPO_URL: &str = "https://github.com/EricLBuehler/mistral.rs";
#[cfg(unix)]
const INSTALL_SH_URL: &str =
    "https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.sh";
#[cfg(windows)]
const INSTALL_PS1_URL: &str =
    "https://raw.githubusercontent.com/EricLBuehler/mistral.rs/master/install.ps1";

fn managed_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".mistralrs"))
}

fn bin_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".local").join("bin"))
}

// Managed installs live in ~/.mistralrs; canonicalize so symlinks resolve to the real install path.
fn is_managed_install() -> bool {
    let (Some(pre), Ok(exe)) = (managed_dir(), std::env::current_exe()) else {
        return false;
    };
    let real = std::fs::canonicalize(&exe).unwrap_or(exe);
    real.starts_with(pre)
}

fn print_source_install_hint(action: &str) {
    let exe = std::env::current_exe().unwrap_or_default();
    println!("`mistralrs {action}` only manages installer-managed installs.");
    println!("This binary is at {}.", exe.display());
    match action {
        "update" => println!(
            "Update it with: cargo install --git {REPO_URL} --locked --force mistralrs-cli"
        ),
        _ => println!("Remove it with: cargo uninstall mistralrs-cli"),
    }
}

pub fn run_update(version: Option<String>) -> Result<()> {
    let _ = &version;
    if !is_managed_install() {
        print_source_install_hint("update");
        return Ok(());
    }

    #[cfg(windows)]
    {
        println!("Automatic update is not yet supported on Windows (the running .exe is locked).");
        println!("Re-run the installer in a new PowerShell:");
        println!("  irm {INSTALL_PS1_URL} | iex");
        Ok(())
    }

    #[cfg(unix)]
    {
        println!("Updating mistral.rs (managed install)...");
        let cmd = format!("curl --proto '=https' --tlsv1.2 -sSf {INSTALL_SH_URL} | sh");
        let mut c = std::process::Command::new("sh");
        c.arg("-c")
            .arg(&cmd)
            .env("MISTRALRS_INSTALL_YES", "1")
            .env("MISTRALRS_INSTALL_IGNORE_FFMPEG", "1");
        if let Some(v) = &version {
            c.env("MISTRALRS_INSTALL_TAG", v);
        }
        let status = c.status().context("failed to run the install script")?;
        if !status.success() {
            bail!("update failed (install script exited with {status})");
        }
        Ok(())
    }
}

pub fn run_uninstall(yes: bool) -> Result<()> {
    let _ = yes;
    let Some(pre) = managed_dir() else {
        bail!("could not resolve home directory");
    };
    if !is_managed_install() {
        print_source_install_hint("uninstall");
        return Ok(());
    }

    #[cfg(windows)]
    {
        println!(
            "Automatic uninstall is not yet supported on Windows (the running .exe is locked)."
        );
        println!(
            "Delete {} and the mistralrs launcher on your PATH manually.",
            pre.display()
        );
        Ok(())
    }

    #[cfg(unix)]
    {
        if !yes && !confirm(&format!("Remove {}?", pre.display()))? {
            println!("Cancelled.");
            return Ok(());
        }
        if let Some(bin) = bin_dir() {
            for name in ["mistralrs", "tileiras"] {
                let link = bin.join(name);
                let is_symlink = link
                    .symlink_metadata()
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false);
                if is_symlink {
                    let _ = std::fs::remove_file(&link);
                }
            }
        }
        std::fs::remove_dir_all(&pre).with_context(|| format!("removing {}", pre.display()))?;
        println!(
            "Removed {} and symlinks. mistral.rs uninstalled.",
            pre.display()
        );
        println!("Restart your terminal before reinstalling or switching install methods.");
        Ok(())
    }
}

#[cfg(unix)]
fn confirm(prompt: &str) -> Result<bool> {
    use std::io::Write;
    print!("{prompt} [y/N] ");
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(matches!(input.trim().to_lowercase().as_str(), "y" | "yes"))
}
