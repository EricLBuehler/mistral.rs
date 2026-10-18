use candle_core::{Device, Result};

use crate::utils::memory_usage::DeviceMemory;
use crate::{paged_attention::device_memory_cap, MemoryUsage};

#[cfg(any(feature = "metal", test))]
const METAL_WORKSPACE_MULTIPLIER: usize = 2;

pub(crate) fn encoder_memory_usage(device: &Device) -> Result<DeviceMemory> {
    let memory = MemoryUsage.query(device)?;
    #[cfg(feature = "metal")]
    if let Device::Metal(metal) = device {
        // A configured wired limit can be lower than Metal's recommended working-set size.
        return Ok(DeviceMemory::Unified {
            budget: memory
                .total()
                .min(crate::utils::memory_usage::metal_sysctl_floor_bytes()?),
            allocated: metal.current_allocated_size(),
        });
    }
    Ok(memory)
}

pub(crate) struct EncoderMemory {
    devices: Vec<(Device, usize)>,
    bytes_per_token: usize,
}

impl EncoderMemory {
    pub fn new(devices: Vec<Device>, bytes_per_token: usize) -> Result<Self> {
        let devices = devices
            .into_iter()
            .filter(|device| !device.is_cpu())
            .map(|device| {
                let budget = encoder_memory_usage(&device)?.total();
                Ok((device, budget))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            devices,
            bytes_per_token,
        })
    }

    pub fn token_budget(&self, max_tokens: usize) -> Result<usize> {
        let mut limit = max_tokens;
        for (device, _budget) in &self.devices {
            let available = match device {
                #[cfg(feature = "metal")]
                Device::Metal(metal) => _budget.saturating_sub(metal.current_allocated_size()),
                _ => encoder_memory_usage(device)?.available(),
            };
            limit = limit.min(tokens_for_memory(
                device_memory_cap(available, device),
                self.bytes_per_token,
                max_tokens,
            ));
        }
        Ok(limit)
    }

    pub fn workspace_bytes(&self, tokens: usize) -> usize {
        self.bytes_per_token.saturating_mul(tokens)
    }

    #[cfg(all(test, feature = "metal"))]
    pub fn constrained(device: Device) -> Self {
        let Device::Metal(metal) = &device else {
            unreachable!()
        };
        let budget = metal.current_allocated_size();
        Self {
            devices: vec![(device, budget)],
            bytes_per_token: 1,
        }
    }
}

fn tokens_for_memory(available: usize, bytes_per_token: usize, max_tokens: usize) -> usize {
    (available / bytes_per_token).clamp(1, max_tokens)
}

pub(super) struct MetalMemoryGuard {
    #[cfg(feature = "metal")]
    budgets: Vec<(Device, usize)>,
}

impl MetalMemoryGuard {
    pub fn new(devices: &[Device]) -> Result<Self> {
        #[cfg(feature = "metal")]
        {
            let budgets = devices
                .iter()
                .filter(|device| device.is_metal())
                .map(|device| Ok((device.clone(), encoder_memory_usage(device)?.total())))
                .collect::<Result<Vec<_>>>()?;
            Ok(Self { budgets })
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = devices;
            Ok(Self {})
        }
    }

    pub fn reclaim_if_needed(&self, device: &Device, workspace_bytes: usize) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Device::Metal(metal) = device {
            let budget = self
                .budgets
                .iter()
                .find(|(d, _)| d.same_device(device))
                .expect("mapped Metal device")
                .1;
            let available = budget.saturating_sub(metal.current_allocated_size());
            if should_reclaim(device_memory_cap(available, device), workspace_bytes) {
                tracing::debug!(
                    available_bytes = available,
                    workspace_bytes,
                    "Reclaiming Metal encoder scratch buffers"
                );
                device.synchronize()?;
            }
        }
        #[cfg(not(feature = "metal"))]
        let _ = (device, workspace_bytes);
        Ok(())
    }
}

#[cfg(any(feature = "metal", test))]
fn should_reclaim(usable_bytes: usize, workspace_bytes: usize) -> bool {
    usable_bytes < workspace_bytes.saturating_mul(METAL_WORKSPACE_MULTIPLIER)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batches_shrink_only_when_workspace_does_not_fit() {
        assert_eq!(tokens_for_memory(4096, 1, 2048), 2048);
        assert_eq!(tokens_for_memory(1024, 2, 2048), 512);
        assert_eq!(tokens_for_memory(0, 2, 2048), 1);
    }

    #[test]
    fn large_gpu_keeps_async_execution_and_small_gpu_reclaims_scratch() {
        assert!(!should_reclaim(4096, 1024));
        assert!(!should_reclaim(2048, 1024));
        assert!(should_reclaim(2047, 1024));
        assert!(should_reclaim(0, 1024));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn pressure_guard_releases_scratch_and_preserves_live_tensors() -> Result<()> {
        use candle_core::{DType, Tensor};
        const SCRATCH_ELEMS: usize = 1024 * 1024;
        let device = Device::new_metal(0)?;
        let Device::Metal(metal) = &device else {
            unreachable!()
        };
        let live = Tensor::ones(128, DType::F32, &device)?;
        let scratch = Tensor::zeros(SCRATCH_ELEMS, DType::F32, &device)?;
        drop(scratch);
        let allocated = metal.current_allocated_size();
        let large = MetalMemoryGuard {
            budgets: vec![(device.clone(), usize::MAX)],
        };
        large.reclaim_if_needed(&device, SCRATCH_ELEMS)?;
        assert_eq!(metal.current_allocated_size(), allocated);
        let small = MetalMemoryGuard {
            budgets: vec![(device.clone(), allocated)],
        };
        small.reclaim_if_needed(&device, SCRATCH_ELEMS)?;
        assert!(metal.current_allocated_size() < allocated);
        assert_eq!(live.to_vec1::<f32>()?, vec![1.0; 128]);
        Ok(())
    }
}
