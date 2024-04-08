use candle_core::{Device, Result};
use indexmap::IndexMap;
use sysinfo::System;

use crate::{models::LayerCaches, sequence::Sequence};

struct PrefixCacheManager {
    caches: IndexMap<Vec<u32>, LayerCaches>,
    cpu_caches: IndexMap<Vec<u32>, LayerCaches>,
    max_mem: usize,
    cpu_system: System,
    cpu_mem: usize,
    device: Device,
}

enum EvictionStatus {
    DoneEvictedN(usize),
    Unable,
}

impl PrefixCacheManager {
    pub fn new(device: Device) -> Self {
        let mut sys = System::new_all();
        sys.refresh_cpu();
        let cpu_mem = sys.total_memory() as usize;
        let max_mem = match &device {
            Device::Cpu => cpu_mem,
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                        .unwrap()
                        .1
                }
                #[cfg(not(feature = "cuda"))]
                {
                    unreachable!()
                }
            }
            Device::Metal(_) => todo!(),
        };
        PrefixCacheManager {
            caches: IndexMap::new(),
            cpu_caches: IndexMap::new(),
            max_mem,
            cpu_system: sys,
            cpu_mem,
            device,
        }
    }

    /// This always keeps the cache on the device. If later on, a new seq cannot be allocated due to memory shortage,
    /// some caches will be evicted.
    pub fn add_sequence(&mut self, seq: &mut Sequence) {
        self.caches
            .insert(seq.get_toks().to_vec(), seq.cache().clone());
    }

    /// Evict the caches to CPU. This is called when there is not enough space for a new seq.
    pub fn evict_to_cpu(&mut self, target_mem: usize) -> Result<EvictionStatus> {
        let mut n = 0;
        while !self.get_device_free_mem() <= target_mem {
            if self.caches.is_empty() {
                return Ok(EvictionStatus::Unable);
            }
            let mut d = self.caches.drain(0..1).collect::<Vec<_>>();
            let (ids, cache) = d.pop().unwrap();
            let mut new_cache = Vec::new();
            for layer in cache {
                if let Some((ref q, ref k)) = layer {
                    new_cache.push(Some((
                        q.to_device(&Device::Cpu)?,
                        k.to_device(&Device::Cpu)?,
                    )));
                } else {
                    new_cache.push(None);
                }
            }
            self.cpu_caches.insert(ids, new_cache);
            n += 1;
        }
        Ok(EvictionStatus::DoneEvictedN(n))
    }

    /// Get the amount of free memory in bytes.
    fn get_device_free_mem(&mut self) -> usize {
        match self.device {
            Device::Cpu => {
                self.cpu_system.refresh_cpu();
                self.cpu_system.free_memory() as usize
            }
            Device::Cuda(_) => {
                #[cfg(feature = "cuda")]
                {
                    candle_core::cuda_backend::cudarc::driver::result::mem_get_info()
                        .unwrap()
                        .0
                }
                #[cfg(not(feature = "cuda"))]
                {
                    unreachable!()
                }
            }
            Device::Metal(_) => todo!(),
        }
    }
}
