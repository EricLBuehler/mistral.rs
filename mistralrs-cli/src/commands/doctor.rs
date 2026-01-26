use anyhow::Result;
use std::process::Command;

use mistralrs_core::run_doctor as run_doctor_report;

pub fn run_doctor(json: bool) -> Result<()> {
    if json {
        let report = run_doctor_report();
        let out = serde_json::to_string_pretty(&report)?;
        println!("{out}");
        return Ok(());
    }

    let report = run_doctor_report();
    let system = &report.system;
    println!("mistralrs doctor");
    println!("git revision: {}", system.build.git_revision);
    println!("OS: {}", system.os.as_deref().unwrap_or("unknown"));
    println!(
        "CPU: {} ({} logical cores)",
        system.cpu.brand.as_deref().unwrap_or("unknown"),
        system.cpu.logical_cores
    );
    println!(
        "RAM: {:.1} GB total, {:.1} GB available",
        system.memory.total_bytes as f64 / 1e9,
        system.memory.available_bytes as f64 / 1e9
    );
    for dev in system.devices.iter().filter(|d| d.kind != "cpu") {
        let label = match dev.ordinal {
            Some(ord) => format!("{}[{}]", dev.kind, ord),
            None => dev.kind.clone(),
        };
        let total = dev
            .total_memory_bytes
            .map(|v| format!("{:.1} GB", v as f64 / 1e9))
            .unwrap_or_else(|| "unknown".to_string());
        let avail = dev
            .available_memory_bytes
            .map(|v| format!("{:.1} GB", v as f64 / 1e9))
            .unwrap_or_else(|| "unknown".to_string());
        println!("Device: {label} (total {total}, free {avail})");
    }
    if cfg!(feature = "cuda") {
        let nvcc = nvcc_version().unwrap_or_else(|| "unknown".to_string());
        let driver = nvidia_driver_version().unwrap_or_else(|| "unknown".to_string());
        println!("CUDA: nvcc {nvcc}, nvidia-smi driver {driver}");
    }
    if cfg!(feature = "metal") {
        let xcode = xcode_version().unwrap_or_else(|| "unknown".to_string());
        println!("Metal: Xcode {xcode}");
    }
    println!();
    for check in &report.checks {
        let status = match check.status {
            mistralrs_core::DoctorStatus::Ok => "OK",
            mistralrs_core::DoctorStatus::Warn => "WARN",
            mistralrs_core::DoctorStatus::Error => "ERROR",
        };
        println!("[{status}] {}: {}", check.name, check.message);
        if let Some(suggestion) = &check.suggestion {
            println!("         hint: {suggestion}");
        }
    }
    Ok(())
}

fn command_stdout(cmd: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(cmd).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

#[cfg(target_os = "windows")]
fn command_stdout_any(cmds: &[&str], args: &[&str]) -> Option<String> {
    for cmd in cmds {
        if let Some(output) = command_stdout(cmd, args) {
            return Some(output);
        }
    }
    None
}

#[cfg(not(target_os = "windows"))]
fn command_stdout_any(cmds: &[&str], args: &[&str]) -> Option<String> {
    let _ = cmds;
    command_stdout(cmds.first()?, args)
}

fn nvcc_version() -> Option<String> {
    let output = command_stdout_any(&["nvcc", "nvcc.exe"], &["--version"])?;
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.contains("release") {
            return Some(trimmed.to_string());
        }
    }
    output.lines().next().map(|line| line.trim().to_string())
}

fn nvidia_driver_version() -> Option<String> {
    let output = command_stdout_any(
        &["nvidia-smi", "nvidia-smi.exe"],
        &["--query-gpu=driver_version", "--format=csv,noheader"],
    )?;
    let mut versions: Vec<&str> = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    versions.sort();
    versions.dedup();
    if versions.is_empty() {
        return None;
    }
    Some(versions.join(", "))
}

fn xcode_version() -> Option<String> {
    let output = command_stdout("xcodebuild", &["-version"])?;
    let mut lines = output.lines().map(str::trim).filter(|line| !line.is_empty());
    let xcode = lines.next()?;
    let build = lines.next();
    let mut version = xcode.to_string();
    if let Some(build) = build {
        version.push_str(" (");
        version.push_str(build);
        version.push(')');
    }
    Some(version)
}
