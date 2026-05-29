//! cuTile JIT warmup driver: runs every registered [`CutileKernel`] once per device on the engine thread (the JIT cache is thread-local).

use candle_core::{CudaDevice, Device, DeviceLocation, Result};
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// A cuTile kernel that can pre-compile (warm) every JIT key it will hit at inference.
pub(super) trait CutileKernel {
    fn warm(&self, dev: &CudaDevice) -> Result<()>;
}

/// Every cuTile kernel to warm. One today; add a line per new kernel.
fn registered() -> [&'static dyn CutileKernel; 1] {
    [&super::fused_moe::FUSED_MOE]
}

static WARMED_LOCATIONS: OnceLock<Mutex<HashSet<DeviceLocation>>> = OnceLock::new();

/// Warm every registered cuTile kernel for `device`, once per device. Call on the engine thread.
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
    device.synchronize()
}
