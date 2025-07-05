use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

use candle_core::{Device, Result};
use indexmap::IndexMap;
use itertools::Itertools;
use tracing::info;

use crate::{
    get_mut_arcmutex,
    paged_attention::{BlockEngine, LogicalTokenBlock, PhysicalTokenBlock},
    pipeline::KvCache,
    sequence::{self, Sequence},
};

type BlockBestMatch<'a> = (
    usize,                         // matched_len
    &'a [LogicalTokenBlock],       // logical blocks
    &'a [Arc<PhysicalTokenBlock>], // physical blocks
    usize,                         // audios_match_until
    usize,                         // images_match_until
);

fn hash_logical_blocks(logical_blocks: &[LogicalTokenBlock]) -> Vec<u64> {
    logical_blocks
        .iter()
        .map(|block| {
            let mut hasher = DefaultHasher::new();
            block.hash(&mut hasher);
            hasher.finish()
        })
        .collect::<Vec<_>>()
}

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
    audio_hashes: Option<Vec<u64>>,
    image_hashes: Option<Vec<u64>>,
}

#[derive(Clone)]
struct BlockCacheElement {
    logical_blocks: Vec<LogicalTokenBlock>,
    physical_blocks: Vec<Arc<PhysicalTokenBlock>>,
    image_hashes: Option<Vec<u64>>,
    audio_hashes: Option<Vec<u64>>,
}

pub struct PrefixCacheManagerV2 {
    caches: IndexMap<Tokens, CacheElement>,
    block_caches: IndexMap<Vec<u64>, BlockCacheElement>, // (hashed logical blocks) => BlockCacheElement
    n_on_device: usize,
    no_prefix_cache: bool,
    block_engine: Option<Arc<tokio::sync::Mutex<BlockEngine>>>,
}

#[derive(Clone)]
pub enum MatchingCache {
    Normal {
        normal: Vec<Option<KvCache>>,
        images_to_keep: usize,
        audios_to_keep: usize,
        toks: Vec<u32>,
        offset: usize,
    },
    Paged {
        logical_blocks: Vec<LogicalTokenBlock>,
        physical_blocks: Vec<Arc<PhysicalTokenBlock>>,
        toks: Vec<u32>,
        offset: usize,
        images_to_keep: usize,
        audios_to_keep: usize,
    },
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
            block_caches: IndexMap::new(),
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

        if let Some(_block_engine) = &self.block_engine {
            // let logical_token_blocks = seq.logical_token_blocks();
            // let block_engine = get_mut_arcmutex!(block_engine);
            // let block_table = &block_engine.block_tables[seq.id()];
            // let hashed_logical_blocks = hash_logical_blocks(logical_token_blocks);

            // if !self.block_caches.contains_key(&hashed_logical_blocks) {
            //     for block in block_table {
            //         block.deref_mut().increment_refcount();
            //     }

            //     self.block_caches.insert(
            //         hashed_logical_blocks,
            //         BlockCacheElement {
            //             logical_blocks: logical_token_blocks.to_vec(),
            //             physical_blocks: block_table.clone(),
            //             image_hashes: seq.image_hashes().map(|x| x.to_vec()),
            //             audio_hashes: seq.audio_hashes().map(|x| x.to_vec()),
            //         },
            //     );
            // }
        } else {
            let cache = seq.normal_cache().to_vec();

            self.caches.insert(
                seq.get_toks().to_vec().into(),
                CacheElement {
                    cache,
                    image_hashes: seq.image_hashes().map(|x| x.to_vec()),
                    audio_hashes: seq.audio_hashes().map(|x| x.to_vec()),
                },
            );
        }
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
        // Count block‑caches that still reside on‑device.
        for cache in self.block_caches.values() {
            if !cache.physical_blocks.is_empty() {
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

        // Now evict block‑caches if we still exceed the on‑device limit.
        for cache in self.block_caches.values_mut() {
            if n_on_device - n_evicted <= self.n_on_device {
                break;
            }
            if !cache.physical_blocks.is_empty() {
                // Drop our strong references and decrement ref‑counts so the
                // BlockEngine can reclaim the KV blocks.
                for block in &cache.physical_blocks {
                    block.deref_mut().decrement_refcount();
                }
                cache.physical_blocks.clear();
                n_evicted += 1;
            }
        }

        self.caches.retain(|_tokens, cache| !cache.cache.is_empty());
        self.block_caches
            .retain(|_key, cache| !cache.physical_blocks.is_empty());

        Ok(n_evicted)
    }

    /// Evict all the caches.
    pub fn evict_all_caches(&mut self) -> Result<usize> {
        let len = self.caches.len();

        self.caches.clear();

        for cache in self.block_caches.values_mut() {
            for block in &cache.physical_blocks {
                block.deref_mut().decrement_refcount();
            }
        }
        self.block_caches.clear();
        Ok(len)
    }

    /// Search for a matching cache given some tokens. Image-containing sequences are now cached too.
    pub fn search_for_matching_cache(
        &mut self,
        toks: &[u32],
        image_hashes: Option<&[u64]>,
        audio_hashes: Option<&[u64]>,
    ) -> Result<Option<MatchingCache>> {
        // Do not search if prefix caching disabled or no tokens
        if self.no_prefix_cache || toks.is_empty() {
            return Ok(None);
        }

        if let Some(block_engine) = &self.block_engine {
            let block_engine = get_mut_arcmutex!(block_engine);
            let block_size = block_engine.block_size();
            let mut test_logical_blocks = Vec::new();
            for tok in toks {
                sequence::util_append_token_to_blocks(
                    *tok as usize,
                    &mut test_logical_blocks,
                    block_size,
                );
            }
            let hashed_logical_blocks = hash_logical_blocks(&test_logical_blocks);

            let mut best_match: Option<BlockBestMatch> = None;
            for (logical, cache_elem) in &self.block_caches {
                let logical_matches_until = logical
                    .iter()
                    .zip(&hashed_logical_blocks)
                    .take_while(|(a, b)| **a == **b)
                    .count();
                let matched_len: usize = cache_elem.logical_blocks[..logical_matches_until]
                    .iter()
                    .map(|block| block.num_tokens())
                    .sum();

                let images_match_until = if let (Some(input_hashes), Some(cached_hashes)) =
                    (image_hashes, cache_elem.image_hashes.as_ref())
                {
                    input_hashes
                        .iter()
                        .zip(cached_hashes)
                        .take_while(|(a, b)| a == b)
                        .count()
                } else {
                    0
                };

                let audios_match_until = if let (Some(input_hashes), Some(cached_hashes)) =
                    (audio_hashes, cache_elem.audio_hashes.as_ref())
                {
                    input_hashes
                        .iter()
                        .zip(cached_hashes)
                        .take_while(|(a, b)| a == b)
                        .count()
                } else {
                    0
                };

                if best_match
                    .is_some_and(|(best_match_len, _, _, _, _)| best_match_len < matched_len)
                    || best_match.is_none()
                {
                    best_match = Some((
                        matched_len,
                        &cache_elem.logical_blocks,
                        &cache_elem.physical_blocks,
                        images_match_until,
                        audios_match_until,
                    ))
                }
            }

            let Some((
                match_len,
                logical_blocks,
                physical_blocks,
                images_match_until,
                audios_match_until,
            )) = best_match
            else {
                return Ok(None);
            };

            // Determine how many blocks cover the matched prefix
            let mut n_blocks = match_len.div_ceil(block_size);
            n_blocks = n_blocks.min(logical_blocks.len());

            if n_blocks == 0 {
                return Ok(None);
            }

            // Take the first n_blocks of both logical and physical blocks
            let mut logical_prefix = logical_blocks[..n_blocks].to_vec();
            let physical_prefix = physical_blocks[..n_blocks].to_vec();
            for block in &physical_prefix {
                block.deref_mut().increment_refcount();
            }

            // If the last reused block is full, reserve an extra empty block for new tokens
            let new_toks = toks[match_len..].to_vec();
            logical_prefix.push(LogicalTokenBlock::new(block_size));
            for tok in &new_toks {
                sequence::util_append_token_to_blocks(
                    *tok as usize,
                    &mut logical_prefix,
                    block_size,
                );
            }
            if logical_prefix.last().is_some_and(|last| last.is_full()) {
                logical_prefix.push(LogicalTokenBlock::new(block_size));
            }
            let images_to_keep = if let Some(input_hashes) = image_hashes {
                input_hashes.len().saturating_sub(images_match_until)
            } else {
                0
            };
            let audios_to_keep = if let Some(input_hashes) = audio_hashes {
                input_hashes.len().saturating_sub(audios_match_until)
            } else {
                0
            };
            return Ok(Some(MatchingCache::Paged {
                logical_blocks: logical_prefix,
                physical_blocks: physical_prefix,
                toks: new_toks,
                offset: match_len,
                images_to_keep,
                audios_to_keep,
            }));
        }

        let toks = Tokens(toks.to_vec());

        let mut best_match: Option<(usize, &CacheElement, usize, usize)> = None;
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

            let audios_match_until = match audio_hashes {
                Some(input_hashes) => match &v.audio_hashes {
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
                .is_none_or(|(len, _, _, _)| match_len > *len)
            {
                best_match = Some((match_len, v, images_match_until, audios_match_until));
            }
        }

        if let Some((match_len, cache_element, images_match_until, audios_match_until)) = best_match
        {
            let new_toks = toks.0[match_len..].to_vec();
            if new_toks.is_empty() {
                return Ok(None);
            }

            let mut cache = cache_element.clone();
            // Count how many input images are not already cached
            let images_to_keep = if let Some(input_hashes) = image_hashes {
                input_hashes.len().saturating_sub(images_match_until)
            } else {
                0
            };
            let audios_to_keep = if let Some(input_hashes) = audio_hashes {
                input_hashes.len().saturating_sub(audios_match_until)
            } else {
                0
            };
            for layer in cache.cache.iter_mut().flatten() {
                if layer.try_set_len(match_len).is_err() {
                    return Ok(None);
                }
            }
            for layer in cache.cache.iter_mut().flatten() {
                layer.set_len(match_len)?;
            }
            return Ok(Some(MatchingCache::Normal {
                normal: cache.cache,
                images_to_keep,
                audios_to_keep,
                toks: new_toks,
                offset: match_len,
            }));
        }

        Ok(None)
    }
}
