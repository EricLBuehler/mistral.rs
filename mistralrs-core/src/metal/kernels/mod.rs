use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::{Error, Result};
use candle_metal_kernels::metal::{Device, Library};
use mistralrs_metal_compile::MetalSourceSet;
use objc2_metal::MTLDevice;

include!("source_set.rs");

macro_rules! embedded_library {
    ($name:ident, $library:literal) => {
        #[cfg(target_os = "macos")]
        const $name: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/", $library, ".metallib"));
        #[cfg(target_os = "ios")]
        const $name: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/", $library, "_ios.metallib"));
        #[cfg(target_os = "tvos")]
        const $name: &[u8] =
            include_bytes!(concat!(env!("OUT_DIR"), "/", $library, "_tvos.metallib"));
    };
}

embedded_library!(GDN_METALLIB, "mistralrs_gdn");
embedded_library!(SSM_METALLIB, "mistralrs_ssm");

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub(super) enum RecurrentLibrary {
    Gdn,
    Ssm,
}

type Libraries = HashMap<(u64, RecurrentLibrary), Arc<Library>>;
static LIBRARIES: OnceLock<Mutex<Libraries>> = OnceLock::new();

pub(super) fn device_key(device: &Device) -> u64 {
    device.registry_id()
}

impl RecurrentLibrary {
    fn source_set(self) -> &'static MetalSourceSet {
        match self {
            Self::Gdn => &GDN_METAL_SOURCE_SET,
            Self::Ssm => &SSM_METAL_SOURCE_SET,
        }
    }

    fn embedded(self) -> &'static [u8] {
        match self {
            Self::Gdn => GDN_METALLIB,
            Self::Ssm => SSM_METALLIB,
        }
    }

    pub(super) fn load(self, device: &Device) -> Result<Arc<Library>> {
        let mut libraries = LIBRARIES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .map_err(|e| Error::msg(format!("Failed to lock recurrent Metal libraries: {e}")))?;
        let key = (device_key(device), self);
        if let Some(library) = libraries.get(&key) {
            return Ok(library.clone());
        }
        let library = if self.embedded().is_empty() {
            mistralrs_metal_compile::compile_runtime_library(device, self.source_set())
                .map_err(Error::msg)?
        } else {
            let data = dispatch2::DispatchData::from_static_bytes(self.embedded());
            let library = device
                .as_ref()
                .newLibraryWithData_error(&data)
                .map_err(|e| {
                    Error::msg(format!(
                        "Failed to load {} Metal library: {e}",
                        self.source_set().library_name
                    ))
                })?;
            Library::new(library)
        };
        let library = Arc::new(library);
        libraries.insert(key, library.clone());
        Ok(library)
    }
}

pub(crate) fn warmup_libraries(device: &Device) -> Result<()> {
    RecurrentLibrary::Gdn.load(device)?;
    RecurrentLibrary::Ssm.load(device)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GDN_FUNCTIONS: [&str; 4] = [
        "gdn_prefill_from_convolved_bfloat_128",
        "gdn_prefill_from_convolved_half_64",
        "causal_conv1d_full_bfloat16_t",
        "gdn_rmsnorm_gated_bfloat",
    ];
    const SSM_FUNCTIONS: [&str; 4] = ["ssm_scan_c1", "ssm_scan_c2", "ssm_scan_c4", "ssm_scan_c8"];

    fn check_functions(library: &Library, names: &[&str]) -> Result<()> {
        for name in names {
            library
                .get_function(name, None)
                .map_err(|e| Error::msg(format!("Missing recurrent Metal function {name}: {e}")))?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn embedded_recurrent_libraries_export_required_functions() -> Result<()> {
        let device = candle_core::Device::new_metal(0)?;
        let raw = device.as_metal_device()?.device();
        for (source, names) in [
            (RecurrentLibrary::Gdn, GDN_FUNCTIONS.as_slice()),
            (RecurrentLibrary::Ssm, SSM_FUNCTIONS.as_slice()),
        ] {
            let library = source.load(raw)?;
            assert!(Arc::ptr_eq(&library, &source.load(raw)?));
            check_functions(&library, names)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn runtime_recurrent_libraries_export_required_functions() -> Result<()> {
        let device = candle_core::Device::new_metal(0)?;
        let raw = device.as_metal_device()?.device();
        for (source_set, names) in [
            (&GDN_METAL_SOURCE_SET, GDN_FUNCTIONS.as_slice()),
            (&SSM_METAL_SOURCE_SET, SSM_FUNCTIONS.as_slice()),
        ] {
            let library = mistralrs_metal_compile::compile_runtime_library(raw, source_set)
                .map_err(Error::msg)?;
            check_functions(&library, names)?;
        }
        Ok(())
    }
}
