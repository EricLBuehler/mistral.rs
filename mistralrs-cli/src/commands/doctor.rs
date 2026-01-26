use anyhow::Result;

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
