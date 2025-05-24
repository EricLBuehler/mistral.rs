use std::sync::Arc;

use candle_core::{Device, Result};
use indexmap::IndexMap;
use itertools::Itertools;
use tracing::info;

use crate::{
    get_mut_arcmutex,
    paged_attention::{BlockEngine, BlockEngineSequence},
    pipeline::KvCache,
    sequence::Sequence,
};

#[derive(PartialEq, Eq, Debug, Hash)]
struct Tokens(Vec<u32>);

impl Tokens {
    /// Returns the length of the common prefix shared with `other`.
    fn shared_prefix_len(&self, other: &Self) -> usize {
        self.0
            .iter()
            .zip(other.0.iter())
            .take_while(|(a, b)| a == b)
            .count()
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
    image_hashes: Option<Vec<u64>>,
}

pub struct PrefixCacheManagerV2 {
    caches: IndexMap<Tokens, CacheElement>,
    n_on_device: usize,
    no_prefix_cache: bool,
    block_engine: Option<Arc<tokio::sync::Mutex<BlockEngine>>>,
}

#[derive(Clone)]
pub struct MatchingCache {
    pub normal: Vec<Option<KvCache>>,
    pub images_to_keep: usize,
    pub toks: Vec<u32>,
    pub offset: usize,
}

impl PrefixCacheManagerV2 {
    pub fn new(
        n_on_device: usize,
        no_prefix_cache: bool,
        block_engine: Option<Arc<tokio::sync::Mutex<BlockEngine>>>,
    ) -> Self {
        if !no_prefix_cache {
            info!("PrefixCacherV2 is enabled. Expect higher multi-turn throughput for both text and multimodal.");
        }
        PrefixCacheManagerV2 {
            caches: IndexMap::new(),
            n_on_device,
            no_prefix_cache,
            block_engine,
        }
    }

    /// This always keeps the cache on the device.
    pub fn add_sequence(&mut self, seq: &mut Sequence) {
        // Do not cache if prefix caching disabled
        if self.no_prefix_cache {
            return;
        }

        if let Some(block_engine) = &self.block_engine {
            dbg!(&seq.logical_token_blocks());
            let block_engine = get_mut_arcmutex!(block_engine);
            dbg!(&block_engine.block_tables[seq.id()]);
        }

        let cache = seq.normal_cache().to_vec();

        self.caches.insert(
            seq.get_toks().to_vec().into(),
            CacheElement {
                cache,
                image_hashes: seq.image_hashes().map(|x| x.to_vec()),
            },
        );
    }

    /// Evict the caches. This will evict the first k seqs such that the number of sequences on device after the copy is
    /// the maximum allowed. Returns the number of evicted sequences.
    pub fn evict_caches(&mut self) -> Result<usize> {
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
            if n_on_device - n_evicted <= self.n_on_device {
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
                cache.cache.clear();
                n_evicted += 1;
            }
        }

        self.caches.retain(|_tokens, cache| !cache.cache.is_empty());

        Ok(n_evicted)
    }

    /// Evict all the caches.
    pub fn evict_all_caches(&mut self) -> Result<usize> {
        let len = self.caches.len();
        self.caches.clear();
        Ok(len)
    }

    /// Search for a matching cache given some tokens. Image-containing sequences are now cached too.
    pub fn search_for_matching_cache(
        &mut self,
        toks: &[u32],
        image_hashes: Option<&[u64]>,
        _contains_images: bool,
    ) -> Result<Option<MatchingCache>> {
        // Do not search if prefix caching disabled or no tokens
        if self.no_prefix_cache || toks.is_empty() {
            return Ok(None);
        }

        let toks = Tokens(toks.to_vec());

        let mut best_match: Option<(usize, &CacheElement, usize)> = None;
        for (k, v) in &self.caches {
            let match_len = toks.shared_prefix_len(k);
            if match_len == 0 {
                continue;
            }

            let images_match_until = match image_hashes {
                Some(input_hashes) => match &v.image_hashes {
                    Some(cached_hashes) => input_hashes
                        .iter()
                        .zip(cached_hashes)
                        .take_while(|(a, b)| a == b)
                        .count(),
                    None => 0,
                },
                None => 0,
            };

            if best_match
                .as_ref()
                .is_none_or(|(len, _, _)| match_len > *len)
            {
                best_match = Some((match_len, v, images_match_until));
            }
        }

        if let Some((match_len, cache_element, images_match_until)) = best_match {
            let mut cache = cache_element.clone();
            // Count how many input images are not already cached
            let images_to_keep = if let Some(input_hashes) = image_hashes {
                input_hashes.len().saturating_sub(images_match_until)
            } else {
                0
            };
            for layer in cache.cache.iter_mut().flatten() {
                if let Err(e) = layer.set_len(match_len) {
                    tracing::warn!("Failed to set cache length to {}: {:?}", match_len, e);
                    return Ok(None);
                }
            }
            return Ok(Some(MatchingCache {
                normal: cache.cache,
                images_to_keep,
                toks: toks.0[match_len..].to_vec(),
                offset: match_len,
            }));
        }

        Ok(None)
    }
}
