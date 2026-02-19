use anyhow::Result;
use std::process::Command;

use mistralrs_core::{run_doctor as run_doctor_report, DoctorStatus};

pub fn run_doctor(json: bool) -> Result<()> {
    if json {
        let report = run_doctor_report();
        let out = serde_json::to_string_pretty(&report)?;
        println!("{out}");
        return Ok(());
    }

    let report = run_doctor_report();
    let system = &report.system;

    // Header
    println!();
    println!("Environment Diagnosis");
    println!("---------------------");
    println!(
        "[INFO] OS: {} ({})",
        system.os.as_deref().unwrap_or("unknown"),
        system.kernel.as_deref().unwrap_or("unknown")
    );

    // CPU info with extensions
    let mut cpu_ext = Vec::new();
    if system.cpu.avx {
        cpu_ext.push("AVX");
    }
    if system.cpu.avx2 {
        cpu_ext.push("AVX2");
    }
    if system.cpu.fma {
        cpu_ext.push("FMA");
    }
    if system.cpu.avx512 {
        cpu_ext.push("AVX-512");
    }
    let ext_str = if cpu_ext.is_empty() {
        "none".to_string()
    } else {
        cpu_ext.join(", ")
    };

    println!(
        "[INFO] CPU: {} ({} cores, extensions: {})",
        system.cpu.brand.as_deref().unwrap_or("unknown"),
        system.cpu.logical_cores,
        ext_str
    );
    println!(
        "[INFO] RAM: {:.1} GB total, {:.1} GB available",
        system.memory.total_bytes as f64 / 1e9,
        system.memory.available_bytes as f64 / 1e9
    );

    // Accelerator section
    let gpu_devices: Vec<_> = system.devices.iter().filter(|d| d.kind != "cpu").collect();
    if !gpu_devices.is_empty() {
        println!();
        println!("Accelerator Check");
        println!("-----------------");

        for dev in gpu_devices {
            let label = match dev.ordinal {
                Some(ord) => format!("{}[{}]", dev.kind.to_uppercase(), ord),
                None => dev.kind.to_uppercase(),
            };
            let total = dev
                .total_memory_bytes
                .map(|v| format!("{:.1} GB", v as f64 / 1e9))
                .unwrap_or_else(|| "unknown".to_string());
            let avail = dev
                .available_memory_bytes
                .map(|v| format!("{:.1} GB", v as f64 / 1e9))
                .unwrap_or_else(|| "unknown".to_string());

            // Include compute capability and flash attention status if available
            let cc_str = if let Some((major, minor)) = dev.compute_capability {
                let fa_v2 = if dev.flash_attn_compatible == Some(true) {
                    "✅"
                } else {
                    "❌"
                };
                let fa_v3 = if dev.flash_attn_v3_compatible == Some(true) {
                    "✅"
                } else {
                    "❌"
                };
                format!(" - Compute {major}.{minor} (FA v2: {fa_v2}, v3: {fa_v3})")
            } else {
                String::new()
            };

            println!("[INFO] {label}: {total} total, {avail} free{cc_str}");
        }

        // CUDA version info
        if cfg!(feature = "cuda") {
            let nvcc = nvcc_version().unwrap_or_else(|| "unknown".to_string());
            let driver = nvidia_driver_version().unwrap_or_else(|| "unknown".to_string());
            println!("[INFO] CUDA: nvcc {nvcc}, driver {driver}");
        }
        // Metal/Xcode version info
        if cfg!(feature = "metal") {
            let xcode = xcode_version().unwrap_or_else(|| "unknown".to_string());
            println!("[INFO] Metal: Xcode {xcode}");
        }
    }

    // Installation section
    println!();
    println!("Mistral.rs Installation");
    println!("-----------------------");
    println!("[INFO] Git revision: {}", system.build.git_revision);

    let mut features = Vec::new();
    if system.build.cuda {
        features.push("cuda");
    }
    if system.build.metal {
        features.push("metal");
    }
    if system.build.cudnn {
        features.push("cudnn");
    }
    if system.build.flash_attn {
        features.push("flash-attn");
    }
    if system.build.flash_attn_v3 {
        features.push("flash-attn-v3");
    }
    if system.build.accelerate {
        features.push("accelerate");
    }
    if system.build.mkl {
        features.push("mkl");
    }
    let features_str = if features.is_empty() {
        "none".to_string()
    } else {
        features.join(", ")
    };
    println!("[INFO] Build features: {features_str}");

    // Checks section
    println!();
    println!("System Checks");
    println!("-------------");

    let mut warn_count = 0;
    let mut error_count = 0;

    for check in &report.checks {
        let (status_str, emoji) = match check.status {
            DoctorStatus::Ok => ("PASS", "✅"),
            DoctorStatus::Warn => {
                warn_count += 1;
                ("WARN", "⚠️")
            }
            DoctorStatus::Error => {
                error_count += 1;
                ("ERROR", "❌")
            }
        };
        println!("[{status_str}] {emoji} {}", check.message);
        if let Some(suggestion) = &check.suggestion {
            println!("       hint: {suggestion}");
        }
    }

    // Summary
    println!();
    println!("Summary");
    println!("-------");
    if error_count > 0 {
        println!(
            "❌ {} error(s) found. Please address the issues above.",
            error_count
        );
    } else if warn_count > 0 {
        println!(
            "⚠️ {} warning(s) found. System is functional but may have issues.",
            warn_count
        );
    } else {
        println!("✅ Your system is healthy. Ready to infer! 🚀");
    }
    println!();

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
            // Extract just the version part like "12.2"
            if let Some(idx) = trimmed.find("release") {
                let after = &trimmed[idx + 8..];
                if let Some(end) = after.find(',') {
                    return Some(after[..end].trim().to_string());
                }
            }
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
    let mut lines = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
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
