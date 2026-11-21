//! Vulkan device selection and execution reporting.
//!
//! The Vulkan backend lives in the patched `candle-core` (see
//! `docs/vulkan-backend.md`). This module is the single place where
//! mistral.rs opens a Vulkan device, so that:
//!
//! - an explicit Vulkan selection never silently becomes the CPU,
//! - the physical adapter and driver actually used are always logged, and
//! - CPU fallback inside the Vulkan backend is measurable
//!   ([`vulkan_execution_report`]).

use candle_core::{Device, Result};

/// Environment variable selecting the Vulkan adapter ordinal used when no
/// device is given explicitly (default `0`). The ordinal is the index in
/// `vulkaninfo --summary` (`GPU0`, `GPU1`, ...).
pub const VULKAN_DEVICE_ENV: &str = "MISTRALRS_VULKAN_DEVICE";

/// Environment variable controlling automatic device selection in a Vulkan
/// build when no Vulkan device can be opened: unset or `error` fails, `cpu`
/// logs a warning and runs on the CPU.
pub const VULKAN_FALLBACK_ENV: &str = "MISTRALRS_VULKAN_FALLBACK";

/// Opens Vulkan adapter `ordinal` and logs the GPU and driver that will run
/// inference. Fails (never falls back to the CPU) when Vulkan is not compiled
/// in, no loader/driver is installed, the ordinal does not exist, or the
/// adapter is a software implementation such as lavapipe.
pub fn new_vulkan_device(ordinal: usize) -> Result<Device> {
    #[cfg(feature = "vulkan")]
    {
        let device = Device::new_vulkan(ordinal).map_err(|e| {
            candle_core::Error::Msg(format!(
                "failed to open Vulkan device {ordinal}: {e}. Available adapters: {}",
                describe_adapters()
            ))
        })?;
        if let Device::Vulkan(vk) = &device {
            tracing::info!("Using Vulkan device {}", vk.adapter_info());
        }
        Ok(device)
    }
    #[cfg(not(feature = "vulkan"))]
    {
        candle_core::bail!(
            "Vulkan device {ordinal} requested, but mistral.rs was built without the `vulkan` feature"
        )
    }
}

#[cfg(feature = "vulkan")]
fn describe_adapters() -> String {
    match candle_core::vulkan_backend::enumerate_adapters() {
        Ok(list) if list.is_empty() => "none".to_string(),
        Ok(list) => list
            .iter()
            .map(|a| a.to_string())
            .collect::<Vec<_>>()
            .join("; "),
        Err(e) => format!("unknown ({e})"),
    }
}

/// The ordinal from [`VULKAN_DEVICE_ENV`], or `0`.
pub fn default_vulkan_ordinal() -> Result<usize> {
    match std::env::var(VULKAN_DEVICE_ENV) {
        Ok(v) if !v.trim().is_empty() => v.trim().parse::<usize>().map_err(|_| {
            candle_core::Error::Msg(format!(
                "{VULKAN_DEVICE_ENV}={v:?} is not a device ordinal (expected 0, 1, ...)"
            ))
        }),
        _ => Ok(0),
    }
}

/// Automatic device selection for a Vulkan build: the adapter from
/// [`VULKAN_DEVICE_ENV`]. If it cannot be opened this fails, unless
/// [`VULKAN_FALLBACK_ENV`]`=cpu`, in which case a warning is logged and the
/// CPU is returned.
pub fn auto_vulkan_device() -> Result<Device> {
    let ordinal = default_vulkan_ordinal()?;
    match new_vulkan_device(ordinal) {
        Ok(d) => Ok(d),
        Err(e) => {
            let fallback = std::env::var(VULKAN_FALLBACK_ENV).unwrap_or_default();
            if fallback.eq_ignore_ascii_case("cpu") {
                tracing::warn!(
                    "{e}; {VULKAN_FALLBACK_ENV}=cpu is set, so inference will run on the CPU (no GPU acceleration)"
                );
                Ok(Device::Cpu)
            } else {
                Err(candle_core::Error::Msg(format!(
                    "{e}. Pass --cpu to run on the CPU, or set {VULKAN_FALLBACK_ENV}=cpu to fall back automatically"
                )))
            }
        }
    }
}

/// A one-line summary of Vulkan execution since the last
/// [`reset_vulkan_execution_stats`]: native dispatches, CPU fallbacks (by
/// op), host/device traffic and syncs. `None` without the `vulkan` feature.
pub fn vulkan_execution_report() -> Option<String> {
    #[cfg(feature = "vulkan")]
    {
        Some(candle_core::vulkan_backend::stats().to_string())
    }
    #[cfg(not(feature = "vulkan"))]
    {
        None
    }
}

/// Resets the process-wide Vulkan execution counters.
pub fn reset_vulkan_execution_stats() {
    #[cfg(feature = "vulkan")]
    candle_core::vulkan_backend::reset_stats();
}
