use candle_core::{Device, Result};
use indexmap::IndexMap;

use crate::{models::LayerCaches, sequence::Sequence};

pub struct PrefixCacheManager {
    caches: IndexMap<Vec<u32>, LayerCaches>,
    cpu_caches: IndexMap<Vec<u32>, LayerCaches>,
    device: Device,
    pub n_on_device: usize,
}

#[derive(Clone)]
pub enum MatchingCache {
    Verbatim(LayerCaches),
    Subset(LayerCaches, Vec<u32>),
}

impl PrefixCacheManager {
    pub fn new(device: Device, n_on_device: usize) -> Self {
        PrefixCacheManager {
            caches: IndexMap::new(),
            cpu_caches: IndexMap::new(),
            device,
            n_on_device,
        }
    }

    /// This always keeps the cache on the device. If later on, a new seq cannot be allocated due to memory shortage,
    /// some caches will be evicted.
    pub fn add_sequence(&mut self, seq: &mut Sequence) {
        self.caches
            .insert(seq.get_toks().to_vec(), seq.cache().clone());
    }

    /// Evict the caches to CPU. This will evict the first k seqs such that the number of sequences on device after the copy is
    /// the maximum allowed. Returns the number of evicted sequences.
    pub fn evict_to_cpu(&mut self) -> Result<usize> {
        // Intentionally evict the first ones first, as they are the oldest
        for (ids, cache) in self
            .caches
            .drain(0..self.caches.len().saturating_sub(self.n_on_device))
        {
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
        }
        Ok(self.caches.len() - self.n_on_device)
    }

    pub fn promote_into_device_cache(
        &mut self,
        toks: Vec<u32>,
        cache: &LayerCaches,
    ) -> Result<LayerCaches> {
        let mut new_cache = Vec::new();
        for layer in cache {
            if let Some((ref q, ref k)) = layer {
                new_cache.push(Some((
                    q.to_device(&self.device)?,
                    k.to_device(&self.device)?,
                )));
            } else {
                new_cache.push(None);
            }
        }
        // Load it into the cache
        self.caches.insert(toks, new_cache.clone());
        Ok(new_cache)
    }

    /// Search for a matching cache given some toks
    pub fn search_for_matching_cache(&mut self, toks: &[u32]) -> Result<Option<MatchingCache>> {
        if let Some(cache) = self.caches.get(toks) {
            Ok(Some(MatchingCache::Verbatim(cache.clone())))
        } else if let Some(cache) = self.cpu_caches.get(toks).cloned() {
            Ok(Some(MatchingCache::Verbatim(
                self.promote_into_device_cache(toks.to_vec(), &cache)?,
            )))
        } else {
            // Look for token ids such that they begins with `toks`
            for (ids, cache) in &self.caches {
                if ids.len() <= toks.len() && &toks[0..ids.len()] == ids {
                    return Ok(Some(MatchingCache::Subset(
                        cache.clone(),
                        toks[ids.len()..].to_vec(),
                    )));
                }
                if ids.len() >= toks.len() && &ids[0..toks.len()] == toks {
                    return Ok(Some(MatchingCache::Verbatim(cache.clone())));
                }
            }
            for (ids, cache) in self.cpu_caches.clone() {
                if ids.len() <= toks.len() && &toks[0..ids.len()] == ids {
                    return Ok(Some(MatchingCache::Subset(
                        self.promote_into_device_cache(toks.to_vec(), &cache)?,
                        toks[ids.len()..].to_vec(),
                    )));
                }
                if ids.len() >= toks.len() && &ids[0..toks.len()] == toks {
                    return Ok(Some(MatchingCache::Verbatim(
                        self.promote_into_device_cache(toks.to_vec(), &cache)?,
                    )));
                }
            }
            Ok(None)
        }
    }
}
