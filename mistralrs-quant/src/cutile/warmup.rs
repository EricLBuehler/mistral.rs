//! cuTile JIT warmup driver for registered inference kernels.

use candle_core::{CudaDevice, Device, DeviceLocation, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

const PROGRESS_TICK: Duration = Duration::from_millis(100);

pub(super) struct WarmupProgress {
    bar: ProgressBar,
}

impl WarmupProgress {
    pub(super) fn new(family: &str, total: usize, silent: bool) -> Self {
        let bar = if silent {
            ProgressBar::hidden()
        } else {
            ProgressBar::new(total as u64)
        };
        bar.set_length(total as u64);
        bar.set_prefix(family.to_owned());
        bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} warming cuTile {prefix} ({eta}) {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        if !silent {
            bar.enable_steady_tick(PROGRESS_TICK);
        }
        Self { bar }
    }

    pub(super) fn run<T>(
        &self,
        stage: impl Into<String>,
        compile: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        self.bar.set_message(stage.into());
        self.bar.tick();
        let result = compile();
        if result.is_ok() {
            self.bar.inc(1);
        }
        result
    }
}

impl Drop for WarmupProgress {
    fn drop(&mut self) {
        self.bar.finish_and_clear();
    }
}

/// A cuTile kernel that can pre-compile every JIT key it will hit at inference.
pub(super) trait CutileKernel {
    fn warm(&self, dev: &CudaDevice, silent: bool) -> Result<()>;
}

/// Every cuTile kernel to warm; add a line per new kernel.
fn registered() -> [&'static dyn CutileKernel; 10] {
    [
        &super::fused_moe::FUSED_MOE,
        &super::fused_moe_fp8::FUSED_MOE_FP8,
        &super::fp8_gemm::FP8_GEMM,
        &super::fp8_w8a8::FP8_W8A8,
        &super::fp8_w8a16::FP8_W8A16,
        &super::nvfp4::NVFP4,
        &super::mxfp8_gemm::MXFP8_GEMM,
        &super::gdn_prefill::GDN_PREFILL,
        &super::qsa::QSA,
        &super::hyper_connection::HYPER_CONNECTION,
    ]
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct RegisteredRouting {
    device: DeviceLocation,
    experts: usize,
    topk: usize,
}

static ROUTING: OnceLock<Mutex<Vec<RegisteredRouting>>> = OnceLock::new();

/// Register routing before loading expert weights so fused-MoE warmup covers the model's top-k.
pub fn register_moe_routing(device: &Device, experts: usize, topk: usize) -> Result<()> {
    if experts == 0 || topk == 0 || topk > experts {
        candle_core::bail!("MoE routing requires 1 <= topk <= experts")
    }
    if !device.is_cuda() {
        return Ok(());
    }
    let entry = RegisteredRouting {
        device: device.location(),
        experts,
        topk,
    };
    let mut routing = ROUTING
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if !routing.contains(&entry) {
        routing.push(entry);
        mark_dirty();
    }
    Ok(())
}

pub(super) fn registered_moe_topks(device: &Device, experts: usize) -> Vec<usize> {
    let topks: Vec<_> = ROUTING
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .iter()
        .filter(|entry| entry.device == device.location() && entry.experts == experts)
        .map(|entry| entry.topk)
        .collect();
    if topks.is_empty() {
        vec![1]
    } else {
        topks
    }
}

static WARMED_LOCATIONS: OnceLock<Mutex<HashSet<DeviceLocation>>> = OnceLock::new();

pub(super) fn mark_dirty() {
    if let Some(warmed) = WARMED_LOCATIONS.get() {
        warmed.lock().unwrap().clear();
    }
}

/// Warm every registered cuTile kernel once per device.
pub fn warmup_moe_kernels(device: &Device, silent: bool) -> Result<()> {
    let Device::Cuda(dev) = device else {
        return Ok(());
    };
    let location = device.location();
    {
        let mut warmed = WARMED_LOCATIONS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap();
        if !warmed.insert(location) {
            return Ok(());
        }
    }
    for kernel in registered() {
        if let Err(err) = kernel.warm(dev, silent) {
            WARMED_LOCATIONS
                .get_or_init(|| Mutex::new(HashSet::new()))
                .lock()
                .unwrap()
                .remove(&location);
            return Err(err);
        }
    }
    Ok(())
}
