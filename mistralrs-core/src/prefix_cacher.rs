#![allow(unused_variables, dead_code, unreachable_code)]

use candle_core::{Device, Result};
use indexmap::IndexMap;

use crate::{models::LayerCaches, sequence::Sequence};

pub struct PrefixCacheManager {
    caches: IndexMap<Vec<u32>, LayerCaches>,
    cpu_caches: IndexMap<Vec<u32>, LayerCaches>,
    xlora_caches: Option<IndexMap<Vec<u32>, LayerCaches>>,
    xlora_cpu_caches: Option<IndexMap<Vec<u32>, LayerCaches>>,
    device: Device,
    pub n_on_device: usize,
    no_prefix_cache: bool,
}

#[derive(Clone)]
pub struct MatchingCache {
    pub normal: LayerCaches,
    pub xlora: Option<LayerCaches>,
    pub toks: Vec<u32>,
}

impl PrefixCacheManager {
    pub fn new(device: Device, n_on_device: usize, is_xlora: bool, no_prefix_cache: bool) -> Self {
        PrefixCacheManager {
            caches: IndexMap::new(),
            cpu_caches: IndexMap::new(),
            xlora_caches: if is_xlora {
                Some(IndexMap::new())
            } else {
                None
            },
            xlora_cpu_caches: if is_xlora {
                Some(IndexMap::new())
            } else {
                None
            },
            device,
            n_on_device,
            no_prefix_cache,
        }
    }

    /// This always keeps the cache on the device. If later on, a new seq cannot be allocated due to memory shortage,
    /// some caches will be evicted.
    pub fn add_sequence(&mut self, seq: &mut Sequence) {
        if self.no_prefix_cache {
            return;
        }
        self.caches
            .insert(seq.get_toks().to_vec(), seq.cache().clone());
        if seq.is_xlora() {
            self.caches
                .insert(seq.get_toks().to_vec(), seq.xlora_cache().clone());
        }
    }

    /// Evict the caches to CPU. This will evict the first k seqs such that the number of sequences on device after the copy is
    /// the maximum allowed. Returns the number of evicted sequences.
    pub fn evict_to_cpu(&mut self) -> Result<usize> {
        if self.no_prefix_cache {
            return Ok(0);
        }
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
        Ok(self.caches.len().saturating_sub(self.n_on_device))
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
        self.cpu_caches.swap_remove(&toks);
        self.caches.insert(toks, new_cache.clone());
        Ok(new_cache)
    }

    pub fn promote_into_device_xlora_cache(
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
        self.xlora_cpu_caches.as_mut().unwrap().swap_remove(&toks);
        self.xlora_caches
            .as_mut()
            .unwrap()
            .insert(toks, new_cache.clone());
        Ok(new_cache)
    }

    /// Search for a matching cache given some toks
    pub fn search_for_matching_cache(&mut self, toks: &[u32]) -> Result<Option<MatchingCache>> {
        return Ok(None);

        if self.no_prefix_cache {
            return Ok(None);
        }

        // Look for token ids such that they begins with `toks`
        let mut candidates = Vec::new();
        // Search the device cache
        for (ids, cache) in &self.caches {
            if ids.len() <= toks.len() && &toks[0..ids.len()] == ids {
                if let Some(xlora_cache) = &self.xlora_caches {
                    candidates.push(MatchingCache {
                        normal: cache.clone(),
                        xlora: Some(xlora_cache.get(ids).unwrap().clone()),
                        toks: toks[ids.len()..].to_vec(),
                    });
                } else {
                    candidates.push(MatchingCache {
                        normal: cache.clone(),
                        xlora: None,
                        toks: toks[ids.len()..].to_vec(),
                    });
                }
            }
        }
        // Sort the candidates by ascending `toks` length
        candidates.sort_by_key(|x| x.toks.len());
        if !candidates.is_empty() {
            // The first one has the shortest `toks` length and therefore maximizes cache usage
            return Ok(Some(candidates.swap_remove(0)));
        }

        let mut candidates = Vec::new();
        // Search the CPU cache and promote if needed
        for (ids, cache) in self.cpu_caches.clone() {
            if ids.len() <= toks.len() && toks[0..ids.len()] == ids {
                if self.xlora_cpu_caches.is_some() {
                    candidates.push(MatchingCache {
                        normal: self.promote_into_device_cache(toks.to_vec(), &cache)?,
                        xlora: Some(
                            self.promote_into_device_xlora_cache(
                                toks.to_vec(),
                                &self
                                    .xlora_cpu_caches
                                    .as_ref()
                                    .unwrap()
                                    .get(toks)
                                    .unwrap()
                                    .clone(),
                            )?,
                        ),
                        toks: toks[ids.len()..].to_vec(),
                    });
                } else {
                    candidates.push(MatchingCache {
                        normal: self.promote_into_device_cache(toks.to_vec(), &cache)?,
                        xlora: None,
                        toks: toks[ids.len()..].to_vec(),
                    });
                }
            }
        }
        // Sort the candidates by ascending `toks` length
        candidates.sort_by_key(|x| x.toks.len());
        if !candidates.is_empty() {
            // The first one has the shortest `toks` length and therefore maximizes cache usage
            return Ok(Some(candidates.swap_remove(0)));
        }

        Ok(None)
    }
}
