use std::collections::HashMap;

use candle_core::{Device, DeviceLocation, Result, Tensor};

use super::{mappers::DeviceMapper, peer::CudaPeerAccess};

/// Pre-creates one copy of an attention mask per unique device used by a `DeviceMapper`.
///
/// Instead of calling `mask.to_device(xs.device())` inside every layer loop iteration
/// (which allocates new GPU storage each time when src != dst device), create a
/// `DeviceMappedMask` once before the loop and call `.get(device)` inside the loop
/// for zero-allocation mask lookup.
pub enum DeviceMappedMask {
    /// No masking.
    None,
    /// Flash attention handles causality. No tensor needed.
    CausalFlash,
    /// Explicit mask tensor, replicated to each device.
    Custom(HashMap<DeviceLocation, Tensor>),
}

impl DeviceMappedMask {
    /// Build a device-mapped mask from an [`AttentionMask`].
    pub fn new(mask: crate::attention::AttentionMask, mapper: &dyn DeviceMapper) -> Result<Self> {
        match mask {
            crate::attention::AttentionMask::None => Ok(Self::None),
            crate::attention::AttentionMask::CausalFlash => Ok(Self::CausalFlash),
            crate::attention::AttentionMask::Custom(tensor) => {
                let mut masks = HashMap::new();
                let devices = mapper.get_unique_devices();
                let cuda_peer_access = CudaPeerAccess::new(&devices)?;
                for device in devices {
                    let loc = device.location();
                    if let std::collections::hash_map::Entry::Vacant(e) = masks.entry(loc) {
                        e.insert(cuda_peer_access.to_device(&tensor, &device)?);
                    }
                }
                Ok(Self::Custom(masks))
            }
        }
    }

    /// Build a device-mapped mask from a single tensor on its current device.
    pub fn from_single(mask: crate::attention::AttentionMask) -> Self {
        match mask {
            crate::attention::AttentionMask::None => Self::None,
            crate::attention::AttentionMask::CausalFlash => Self::CausalFlash,
            crate::attention::AttentionMask::Custom(tensor) => {
                let mut masks = HashMap::new();
                masks.insert(tensor.device().location(), tensor);
                Self::Custom(masks)
            }
        }
    }

    /// Look up the [`AttentionMask`] for the given device.
    pub fn get(&self, device: &Device) -> crate::attention::AttentionMask {
        match self {
            Self::None => crate::attention::AttentionMask::None,
            Self::CausalFlash => crate::attention::AttentionMask::CausalFlash,
            Self::Custom(masks) => {
                let tensor = masks
                    .get(&device.location())
                    .expect("DeviceMappedMask: device not in mapper's unique devices");
                crate::attention::AttentionMask::Custom(tensor.clone())
            }
        }
    }
}
