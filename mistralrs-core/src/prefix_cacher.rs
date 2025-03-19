use std::collections::HashMap;

use candle_core::{Device, Result};
use either::Either;
use itertools::Itertools;
use tracing::info;

use crate::{
    pipeline::{KvCache, RotatingCache, SingleCache},
    sequence::Sequence,
};

#[derive(PartialEq, Eq, Debug, Hash)]
struct Tokens(Vec<u32>);

impl Tokens {
    /// Maximum index to where the two sets of tokens match
    fn find_max_index(&self, x: &Self) -> Option<usize> {
        self.0
            .iter()
            .zip(x.0.iter())
            .enumerate()
            .take_while(|(_, (a, b))| a == b)
            .map(|(index, _)| index)
            .last()
    }
}

impl From<Vec<u32>> for Tokens {
    fn from(value: Vec<u32>) -> Self {
        Self(value)
    }
}

#[derive(Clone)]
struct CacheElement {
    cache: Vec<Option<KvCache>>,
    devices: Vec<Option<Device>>,
}

pub struct PrefixCacheManagerV2 {
    caches: HashMap<Tokens, CacheElement>,
    n_on_device: usize,
    no_prefix_cache: bool,
}

#[derive(Clone)]
pub struct MatchingCache {
    pub normal: Vec<Option<KvCache>>,
    pub toks: Vec<u32>,
    pub offset: usize,
}

impl PrefixCacheManagerV2 {
    pub fn new(n_on_device: usize, no_prefix_cache: bool) -> Self {
        if !no_prefix_cache {
            info!("PrefixCacherV2 is enabled! Expect higher multi-turn prompt throughput.");
        }
        PrefixCacheManagerV2 {
            caches: HashMap::new(),
            n_on_device,
            no_prefix_cache,
        }
    }

    /// This always keeps the cache on the device.
    pub fn add_sequence(&mut self, seq: &mut Sequence) {
        if self.no_prefix_cache || seq.has_images() {
            return;
        }
        let cache = seq.normal_cache().to_vec();
        let devices = cache
            .iter()
            .map(|x| x.as_ref().map(|x| x.k().unwrap().unwrap().device().clone()))
            .collect::<Vec<_>>();
        self.caches.insert(
            seq.get_toks().to_vec().into(),
            CacheElement { cache, devices },
        );
    }

    fn cache_to(
        cache: &mut [Option<KvCache>],
        devices: Either<&Device, &Vec<Option<Device>>>,
    ) -> Result<()> {
        for (i, layer) in cache
            .iter_mut()
            .enumerate()
            .flat_map(|(i, x)| x.as_mut().map(|x| (i, x)))
        {
            let device = devices.left_or_else(|layers| layers[i].as_ref().unwrap());

            match layer {
                KvCache::Normal { k, v } => {
                    *layer = KvCache::Normal {
                        k: SingleCache {
                            all_data: k.all_data.as_ref().map(|x| x.to_device(device).unwrap()),
                            dim: k.dim,
                            current_seq_len: k.current_seq_len,
                            max_seq_len: k.max_seq_len,
                            capacity_seq_len: k.capacity_seq_len,
                        },
                        v: SingleCache {
                            all_data: v.all_data.as_ref().map(|x| x.to_device(device).unwrap()),
                            dim: v.dim,
                            current_seq_len: v.current_seq_len,
                            max_seq_len: v.max_seq_len,
                            capacity_seq_len: v.capacity_seq_len,
                        },
                    }
                }
                KvCache::Rotating { k, v } => {
                    *layer = KvCache::Rotating {
                        k: RotatingCache {
                            all_data: k.all_data.as_ref().map(|x| x.to_device(device).unwrap()),
                            dim: k.dim,
                            current_seq_len: k.current_seq_len,
                            max_seq_len: k.max_seq_len,
                            offset: k.offset,
                            capacity_seq_len: k.capacity_seq_len,
                        },
                        v: RotatingCache {
                            all_data: v.all_data.as_ref().map(|x| x.to_device(device).unwrap()),
                            dim: v.dim,
                            current_seq_len: v.current_seq_len,
                            max_seq_len: v.max_seq_len,
                            offset: v.offset,
                            capacity_seq_len: v.capacity_seq_len,
                        },
                    }
                }
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
        for cache in self.caches.values() {
            let first_non_none = cache.cache.iter().find_or_first(|x| x.is_some());
            let Some(Some(first_non_none)) = first_non_none else {
                continue;
            };

            let cache_device = match first_non_none {
                KvCache::Normal { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Rotating { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
            };

            if !matches!(cache_device, Device::Cpu) {
                n_on_device += 1;
            }
        }
        let mut n_evicted = 0;
        // Intentionally evict the first ones first, as they are the oldest
        for cache in self.caches.values_mut() {
            if n_on_device - n_evicted == self.n_on_device {
                break;
            }
            let first_non_none = cache.cache.iter().find_or_first(|x| x.is_some());
            let Some(Some(first_non_none)) = first_non_none else {
                continue;
            };

            let cache_device = match first_non_none {
                KvCache::Normal { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Rotating { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
            };

            if !matches!(cache_device, Device::Cpu) {
                Self::cache_to(&mut cache.cache, Either::Left(&Device::Cpu))?;
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
        for cache in self.caches.values_mut() {
            let first_non_none = cache.cache.iter().find_or_first(|x| x.is_some());
            let Some(Some(first_non_none)) = first_non_none else {
                continue;
            };

            let cache_device = match first_non_none {
                KvCache::Normal { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
                KvCache::Rotating { k, .. } => {
                    k.all_data().as_ref().expect("No KV cache data").device()
                }
            };

            if !matches!(cache_device, Device::Cpu) {
                Self::cache_to(&mut cache.cache, Either::Left(&Device::Cpu))?;
            }
        }
        Ok(self.caches.len())
    }

    /// Search for a matching cache given some toks
    pub fn search_for_matching_cache(
        &mut self,
        toks: &[u32],
        contains_images: bool,
    ) -> Result<Option<MatchingCache>> {
        if self.no_prefix_cache || toks.is_empty() || contains_images {
            return Ok(None);
        }

        let toks = Tokens(toks.to_vec());

        let mut longest_match = (0, None);
        for (k, v) in self.caches.iter() {
            let match_len = toks.find_max_index(k);
            if let Some(match_len) = match_len {
                if match_len > longest_match.0 {
                    longest_match = (match_len, Some(v));
                }
            }
        }
        if let (match_len, Some(longest_match)) = longest_match {
            let mut cache = longest_match.clone();
            Self::cache_to(&mut cache.cache, Either::Right(&cache.devices))?;
            for layer in cache.cache.iter_mut().flatten() {
                match layer.set_len(match_len) {
                    Ok(_) => (),
                    Err(_) => return Ok(None),
                }
            }
            Ok(Some(MatchingCache {
                normal: cache.cache,
                toks: toks.0[match_len..].to_vec(),
                offset: match_len,
            }))
        } else {
            Ok(None)
        }
    }
}
