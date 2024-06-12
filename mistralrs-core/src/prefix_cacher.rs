use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use candle_core::{Device, Result, Tensor};

use crate::{get_mut_arcmutex, pipeline::LayerCaches, sequence::Sequence};

#[derive(PartialEq, Eq, Hash)]
struct Tokens(Vec<u32>);

impl From<Vec<u32>> for Tokens {
    fn from(value: Vec<u32>) -> Self {
        Self(value)
    }
}

type EvictionCacheGroup = (Arc<Mutex<LayerCaches>>, Option<Arc<Mutex<LayerCaches>>>);

trait SubsetCacheUtils
where
    Self: Sized,
{
    fn narrow(self, found_idx: usize) -> Result<Self>;
    fn to(self, device: &Device) -> Result<Self>;
}

impl SubsetCacheUtils for Option<(Tensor, Tensor)> {
    fn narrow(self, found_idx: usize) -> Result<Self> {
        if let Some((k, v)) = self {
            Ok(Some((
                k.narrow(2, 0, found_idx)?,
                v.narrow(2, 0, found_idx)?,
            )))
        } else {
            Ok(None)
        }
    }
    fn to(self, device: &Device) -> Result<Self> {
        if let Some((k, v)) = self {
            Ok(Some((k.to_device(device)?, v.to_device(device)?)))
        } else {
            Ok(None)
        }
    }
}

pub struct PrefixCacheManager {
    caches: HashMap<Tokens, Arc<Mutex<LayerCaches>>>,
    xlora_caches: Option<HashMap<Tokens, Arc<Mutex<LayerCaches>>>>,
    device: Device,
    pub n_on_device: usize,
    no_prefix_cache: bool,
    eviction_cache_ptrs: Vec<EvictionCacheGroup>,
}

#[derive(Clone)]
pub struct MatchingCache {
    pub normal: LayerCaches,
    pub xlora: Option<LayerCaches>,
    pub current_toks: Vec<u32>,
    pub remaining_toks: Option<Vec<u32>>,
}

impl PrefixCacheManager {
    pub fn new(device: Device, n_on_device: usize, is_xlora: bool, no_prefix_cache: bool) -> Self {
        PrefixCacheManager {
            caches: HashMap::new(),
            xlora_caches: if is_xlora { Some(HashMap::new()) } else { None },
            device,
            n_on_device,
            no_prefix_cache,
            eviction_cache_ptrs: Vec::new(),
        }
    }

    /// This always keeps the cache on the device. If later on, a new seq cannot be allocated due to memory shortage,
    /// some caches will be evicted.
    pub fn add_sequence(&mut self, seq: &mut Sequence) {
        if self.no_prefix_cache {
            return;
        }
        let cache = Arc::new(Mutex::new(seq.cache_mut().clone()));
        self.caches
            .insert(seq.get_toks().to_vec().into(), cache.clone());
        if seq.is_xlora() {
            let xlora_cache = Arc::new(Mutex::new(seq.xlora_cache().clone()));
            self.xlora_caches
                .as_mut()
                .unwrap()
                .insert(seq.get_toks().to_vec().into(), xlora_cache.clone());
            self.eviction_cache_ptrs.push((cache, Some(xlora_cache)));
        } else {
            self.eviction_cache_ptrs.push((cache, None));
        }
    }

    fn cache_to<'a>(
        cache: impl Iterator<Item = &'a mut Option<(Tensor, Tensor)>>,
        device: &Device,
    ) -> Result<()> {
        for layer in cache {
            if let Some((ref q, ref k)) = layer {
                *layer = Some((q.to_device(device)?, k.to_device(device)?));
            }
        }
        Ok(())
    }

    /// Evict the caches to CPU. This will evict the first k seqs such that the number of sequences on device after the copy is
    /// the maximum allowed. Returns the number of evicted sequences.
    pub fn evict_to_cpu(&mut self) -> Result<usize> {
        if self.no_prefix_cache {
            return Ok(0);
        }
        let mut n_on_device = 0;
        for (cache, _) in &self.eviction_cache_ptrs {
            if !matches!(
                get_mut_arcmutex!(cache.as_ref())[0]
                    .as_ref()
                    .unwrap()
                    .0
                    .device(),
                Device::Cpu
            ) {
                n_on_device += 1;
            }
        }
        let mut n_evicted = 0;
        // Intentionally evict the first ones first, as they are the oldest
        for (cache, xlora_cache) in &self.eviction_cache_ptrs {
            if n_on_device - n_evicted == self.n_on_device {
                break;
            }
            if !matches!(
                get_mut_arcmutex!(cache.as_ref())[0]
                    .as_ref()
                    .unwrap()
                    .0
                    .device(),
                Device::Cpu
            ) {
                let mut cache = get_mut_arcmutex!(cache);
                let mut xlora_cache = xlora_cache.as_ref().map(|c| get_mut_arcmutex!(c));

                Self::cache_to(cache.iter_mut(), &Device::Cpu)?;
                if let Some(ref mut xlora_cache) = xlora_cache {
                    Self::cache_to(xlora_cache.iter_mut(), &Device::Cpu)?;
                }
                n_evicted += 1;
            }
        }
        Ok(self.caches.len().saturating_sub(self.n_on_device))
    }

    /// Evict all the caches to CPU.
    pub fn evict_all_to_cpu(&mut self) -> Result<usize> {
        if self.no_prefix_cache {
            return Ok(0);
        }
        // Intentionally evict the first ones first, as they are the oldest
        for (cache, xlora_cache) in &self.eviction_cache_ptrs {
            if !matches!(
                get_mut_arcmutex!(cache.as_ref())[0]
                    .as_ref()
                    .unwrap()
                    .0
                    .device(),
                Device::Cpu
            ) {
                let mut cache = get_mut_arcmutex!(cache);
                let mut xlora_cache = xlora_cache.as_ref().map(|c| get_mut_arcmutex!(c));

                Self::cache_to(cache.iter_mut(), &Device::Cpu)?;
                if let Some(ref mut xlora_cache) = xlora_cache {
                    Self::cache_to(xlora_cache.iter_mut(), &Device::Cpu)?;
                }
            }
        }
        Ok(self.caches.len())
    }

    /// Search for a matching cache given some toks
    pub fn search_for_matching_cache(&mut self, toks: &[u32]) -> Result<Option<MatchingCache>> {
        if self.no_prefix_cache {
            return Ok(None);
        }

        let toks = Tokens(toks.to_vec());
        // Try to get a verbatim match
        if let Some(cache) = self.caches.get(&toks) {
            Self::cache_to(get_mut_arcmutex!(cache.as_ref()).iter_mut(), &self.device)?;
            let cache = get_mut_arcmutex!(cache.as_ref()).clone();
            let xlora_cache = if let Some(ref xlora_caches) = self.xlora_caches {
                let mut xlora_cache = get_mut_arcmutex!(xlora_caches.get(&toks).unwrap().as_ref());
                Self::cache_to(xlora_cache.iter_mut(), &self.device)?;
                Some(xlora_cache.clone())
            } else {
                None
            };
            Ok(Some(MatchingCache {
                normal: cache,
                xlora: xlora_cache,
                current_toks: toks.0,
                remaining_toks: None,
            }))
        } else {
            // We could not find a verbatim match, so now look for matches where the target toks are a superset
            // of the found tokens (ie there are some tokens added)
            // TODO(EricLBuehler): n^2 performance here...
            let mut found = None;
            for candidate in self.caches.keys() {
                for i in 1..(candidate.0.len().min(toks.0.len())) {
                    let candidate_tokens = &candidate.0[0..i];
                    let needle_tokens = &toks.0[0..i];
                    if candidate_tokens == needle_tokens && !toks.0[i..].is_empty() {
                        found = Some((i, candidate));
                    }
                }
            }
            if let Some((found_idx, candidate)) = found {
                // We now have the index `i` of a candidate whose cache up to `i` can be used.
                // Get the caches
                let cache = self.caches[candidate].clone();
                let cache = get_mut_arcmutex!(cache.as_ref()).clone();
                let xlora_cache = self
                    .xlora_caches
                    .as_ref()
                    .map(|cache| cache[candidate].clone())
                    .map(|cache| get_mut_arcmutex!(cache.as_ref()).clone());
                // Tokens for which the cache has been computed
                let current_toks = candidate.0[..found_idx].to_vec();
                // Narrow the caches
                let mut new_cache = vec![None; cache.len()];
                for (i, layer) in cache.into_iter().enumerate() {
                    new_cache[i] = layer.narrow(current_toks.len())?.to(&self.device)?;
                }
                let xlora_cache = if let Some(xlora_cache) = xlora_cache {
                    let mut new_cache = vec![None; xlora_cache.len()];
                    for (i, layer) in xlora_cache.into_iter().enumerate() {
                        new_cache[i] = layer.narrow(current_toks.len())?.to(&self.device)?;
                    }
                    Some(new_cache)
                } else {
                    None
                };
                // These tokens will be run on the next step.
                let next_toks = toks.0[found_idx..].to_vec();
                Ok(Some(MatchingCache {
                    normal: new_cache,
                    xlora: xlora_cache,
                    current_toks,
                    remaining_toks: Some(next_toks),
                }))
            } else {
                Ok(None)
            }
        }
    }
}
