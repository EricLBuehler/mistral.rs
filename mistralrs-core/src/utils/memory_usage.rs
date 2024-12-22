use std::process::Command;

use candle_core::{Device, Result};
use sysinfo::System;

const KB_TO_BYTES: usize = 1024;

fn get_available_memory_vm_stat() -> Result<usize> {
    // Execute the `vm_stat` command
    let output = Command::new("vm_stat")
        .output()
        .map_err(|e| candle_core::Error::msg(format!("Failed to execute vm_stat: {}", e)))?;

    // Convert output to a string
    let output_str = String::from_utf8(output.stdout)
        .map_err(|e| candle_core::Error::msg(format!("Failed to parse output: {}", e)))?;

    // Initialize variables
    let mut free_pages = 0;
    let mut inactive_pages = 0;
    let mut page_size = 0; // Default page size in bytes for macOS

    // Parse the output line by line
    for line in output_str.lines() {
        if line.starts_with("Pages free:") {
            if let Some(value) = line.split_whitespace().nth(2) {
                free_pages = value.trim_end_matches('.').parse::<usize>().unwrap();
            }
        } else if line.starts_with("Pages inactive:") {
            if let Some(value) = line.split_whitespace().nth(2) {
                inactive_pages = value.trim_end_matches('.').parse::<usize>().unwrap();
            }
        } else if line.starts_with("Mach Virtual Memory Statistics:") {
            if let Some(start) = line.find("of ") {
                if let Some(end) = line.find(" bytes)") {
                    page_size = (line[start + "of ".len()..end].to_string())
                        .parse::<usize>()
                        .unwrap();
                }
            }
        }
    }

    // Calculate available memory
    let available_memory = (free_pages + inactive_pages) * page_size;

    Ok(available_memory)
}

fn get_total_memory_vm_stat() -> Result<usize> {
    // Execute the `vm_stat` command
    let output = Command::new("sysctl")
        .arg("hw.memsize")
        .output()
        .map_err(|e| candle_core::Error::msg(format!("Failed to execute sysctl: {}", e)))?;

    // Convert output to a string
    let output_str = String::from_utf8(output.stdout)
        .map_err(|e| candle_core::Error::msg(format!("Failed to parse output: {}", e)))?;

    Ok(output_str
        .trim_start_matches("hw.memsize: ")
        .trim()
        .parse::<usize>()
        .unwrap())
}

pub struct MemoryUsage;

impl MemoryUsage {
    /// Amount of available memory in bytes.
    pub fn get_memory_available(&self, device: &Device) -> Result<usize> {
        match device {
            Device::Cpu => {
                let mut sys = System::new_all();
                sys.refresh_cpu();
                Ok(usize::try_from(sys.free_memory())? * KB_TO_BYTES)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                use candle_core::cuda_backend::WrapErr;
                Ok(candle_core::cuda::cudarc::driver::result::mem_get_info()
                    .w()?
                    .0)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                candle_core::bail!("Cannot get memory available for CUDA device")
            }
            Device::Metal(_) => get_available_memory_vm_stat(),
        }
    }

    /// Amount of total memory in bytes.
    pub fn get_total_memory(&self, device: &Device) -> Result<usize> {
        match device {
            Device::Cpu => {
                let mut sys = System::new_all();
                sys.refresh_cpu();
                Ok(usize::try_from(sys.total_memory())? * KB_TO_BYTES)
            }
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => {
                use candle_core::cuda_backend::WrapErr;
                Ok(candle_core::cuda::cudarc::driver::result::mem_get_info()
                    .w()?
                    .1)
            }
            #[cfg(not(feature = "cuda"))]
            Device::Cuda(_) => {
                candle_core::bail!("Cannot get total memory for CUDA device")
            }
            Device::Metal(_) => get_total_memory_vm_stat(),
        }
    }
}
