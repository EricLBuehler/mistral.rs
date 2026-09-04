//! cuTile JIT warmup driver for registered inference kernels.

use candle_core::{CudaDevice, Device, DeviceLocation, Result};
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// A cuTile kernel that can pre-compile every JIT key it will hit at inference.
pub(super) trait CutileKernel {
    fn warm(&self, dev: &CudaDevice) -> Result<()>;
}

/// Every cuTile kernel to warm; add a line per new kernel.
fn registered() -> [&'static dyn CutileKernel; 4] {
    [
        &super::fused_moe::FUSED_MOE,
        &super::fused_moe_fp8::FUSED_MOE_FP8,
        &super::fp8_gemm::FP8_GEMM,
        &super::gdn_prefill::GDN_PREFILL,
    ]
}

static WARMED_LOCATIONS: OnceLock<Mutex<HashSet<DeviceLocation>>> = OnceLock::new();

/// Warm every registered cuTile kernel once per device.
pub fn warmup_moe_kernels(device: &Device) -> Result<()> {
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
        if let Err(err) = kernel.warm(dev) {
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
