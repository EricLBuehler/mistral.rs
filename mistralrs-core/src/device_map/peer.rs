#[cfg(feature = "cuda")]
use candle_core::DeviceLocation;
use candle_core::{Device, Result, Tensor};
#[cfg(feature = "cuda")]
use tracing::{info, warn};

#[derive(Debug, Default, Clone)]
pub(super) struct CudaPeerAccess {
    #[cfg(feature = "cuda")]
    enabled_pairs: std::collections::HashSet<(usize, usize)>,
}

impl CudaPeerAccess {
    pub(super) fn new(devices: &[Device]) -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            Self::new_cuda(devices)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = devices;
            Ok(Self::default())
        }
    }

    pub(super) fn to_device(&self, input: &Tensor, dst: &Device) -> Result<Tensor> {
        if self.needs_host_stage(input.device(), dst) {
            input.to_device(&Device::Cpu)?.to_device(dst)
        } else {
            input.to_device(dst)
        }
    }

    fn needs_host_stage(&self, src: &Device, dst: &Device) -> bool {
        if src.same_device(dst) {
            return false;
        }

        #[cfg(feature = "cuda")]
        {
            let Some(src_ordinal) = Self::cuda_ordinal(src) else {
                return false;
            };
            let Some(dst_ordinal) = Self::cuda_ordinal(dst) else {
                return false;
            };
            !self.enabled_pairs.contains(&(src_ordinal, dst_ordinal))
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = (src, dst);
            false
        }
    }

    #[cfg(feature = "cuda")]
    fn new_cuda(devices: &[Device]) -> Result<Self> {
        use candle_core::cuda::cudarc::driver::{result, sys};

        let mut cuda_devices = Vec::new();
        for device in devices {
            let Some(ordinal) = Self::cuda_ordinal(device) else {
                continue;
            };
            if !cuda_devices
                .iter()
                .any(|(existing, _)| *existing == ordinal)
            {
                cuda_devices.push((ordinal, device));
            }
        }

        if cuda_devices.len() < 2 {
            return Ok(Self::default());
        }

        let mut enabled_pairs = std::collections::HashSet::new();
        for (src_ordinal, src) in &cuda_devices {
            for (dst_ordinal, dst) in &cuda_devices {
                if src_ordinal == dst_ordinal {
                    continue;
                }

                let src_ctx = src.as_cuda_device()?.cuda_stream().context().clone();
                let dst_ctx = dst.as_cuda_device()?.cuda_stream().context().clone();
                let mut can_access = 0;
                let can_access_result = unsafe {
                    sys::cuDeviceCanAccessPeer(
                        &mut can_access,
                        src_ctx.cu_device(),
                        dst_ctx.cu_device(),
                    )
                    .result()
                };
                if let Err(err) = can_access_result {
                    warn!(
                        "Failed to query CUDA peer access from cuda:{src_ordinal} to cuda:{dst_ordinal}: {err:?}. Staging cross-GPU layer transfers through CPU."
                    );
                    continue;
                }

                if can_access == 0 {
                    warn!(
                        "CUDA peer access unavailable from cuda:{src_ordinal} to cuda:{dst_ordinal}; staging cross-GPU layer transfers through CPU."
                    );
                    continue;
                }

                if let Err(err) = src_ctx.bind_to_thread() {
                    warn!(
                        "Failed to bind cuda:{src_ordinal} before enabling peer access to cuda:{dst_ordinal}: {err:?}. Staging cross-GPU layer transfers through CPU."
                    );
                    continue;
                }

                let enable = unsafe { sys::cuCtxEnablePeerAccess(dst_ctx.cu_ctx(), 0).result() };
                match enable {
                    Ok(())
                    | Err(result::DriverError(
                        sys::CUresult::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
                    )) => {
                        enabled_pairs.insert((*src_ordinal, *dst_ordinal));
                    }
                    Err(result::DriverError(sys::CUresult::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED)) => {
                        warn!(
                            "CUDA peer access unsupported from cuda:{src_ordinal} to cuda:{dst_ordinal}; staging cross-GPU layer transfers through CPU."
                        );
                    }
                    Err(err) => {
                        warn!(
                            "Failed to enable CUDA peer access from cuda:{src_ordinal} to cuda:{dst_ordinal}: {err:?}. Staging cross-GPU layer transfers through CPU."
                        );
                    }
                }
            }
        }

        if !enabled_pairs.is_empty() {
            info!(
                "Enabled CUDA peer access for {} directed device pairs.",
                enabled_pairs.len()
            );
        }

        Ok(Self { enabled_pairs })
    }

    #[cfg(feature = "cuda")]
    fn cuda_ordinal(device: &Device) -> Option<usize> {
        match device.location() {
            DeviceLocation::Cuda { gpu_id } => Some(gpu_id),
            _ => None,
        }
    }
}
