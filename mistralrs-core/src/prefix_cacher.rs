use candle_core::{Device, Result};
use indexmap::IndexMap;
use itertools::Itertools;
use tracing::info;

use crate::{
    kv_cache::RecurrentStateSnapshot,
    paged_attention::block_hash::{BlockHash, MultiModalFeature, MultimodalKind},
    pipeline::KvCache,
    sequence::Sequence,
    AdapterGenerationId,
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

#[derive(PartialEq, Eq, Debug, Hash)]
struct CacheKey {
    tokens: Tokens,
    adapter_generation: Option<AdapterGenerationId>,
}

impl CacheKey {
    fn new(tokens: Vec<u32>, adapter_generation: Option<AdapterGenerationId>) -> Self {
        Self {
            tokens: tokens.into(),
            adapter_generation,
        }
    }
}

impl From<Vec<u32>> for CacheKey {
    fn from(tokens: Vec<u32>) -> Self {
        Self::new(tokens, None)
    }
}

#[derive(Clone)]
struct CacheElement {
    cache: Vec<Option<KvCache>>,
    recurrent_snapshots: Option<CachedRecurrentState>,
    audio_hashes: Option<Vec<u64>>,
    image_hashes: Option<Vec<u64>>,
    video_hashes: Option<Vec<u64>>,
}

#[derive(Clone)]
struct CachedRecurrentState {
    len: usize,
    snapshots: Vec<RecurrentStateSnapshot>,
}

impl CacheElement {
    fn can_rewind_to(&self, len: usize) -> bool {
        self.cache
            .iter()
            .flatten()
            .all(|layer| layer.try_set_len(len).is_ok())
    }

    fn coverage_len(&self) -> usize {
        self.cache
            .iter()
            .flatten()
            .filter(|layer| !matches!(layer, KvCache::Shared { .. }))
            .map(KvCache::current_seq_len)
            .min()
            .unwrap_or(0)
    }
}

fn clamp_normal_prefix_len(prefix_len: usize, features: &[MultiModalFeature]) -> usize {
    let mut prefix_len = prefix_len;
    loop {
        let next = features
            .iter()
            .filter(|feature| {
                matches!(feature.kind, MultimodalKind::Image | MultimodalKind::Audio)
                    && feature.offset < prefix_len
                    && prefix_len < feature.end()
            })
            .map(|feature| feature.offset)
            .min()
            .unwrap_or(prefix_len);
        if next == prefix_len {
            return prefix_len;
        }
        prefix_len = next;
    }
}

fn clamp_prefix_len_for_media_hashes(
    prefix_len: usize,
    features: &[MultiModalFeature],
    kind: MultimodalKind,
    matching_items: usize,
    input_items: usize,
) -> usize {
    if matching_items >= input_items {
        return prefix_len;
    }
    features
        .iter()
        .filter(|feature| {
            feature.kind == kind
                && feature.item_range.end > matching_items
                && feature.item_range.start < input_items
        })
        .map(|feature| feature.offset)
        .min()
        .map_or(prefix_len, |offset| prefix_len.min(offset))
}

fn fully_cached_item_count(
    prefix_len: usize,
    features: &[MultiModalFeature],
    kind: MultimodalKind,
    item_count: usize,
) -> usize {
    (0..item_count)
        .take_while(|&item| {
            let mut item_features = features
                .iter()
                .filter(|feature| feature.kind == kind && feature.item_range.contains(&item))
                .peekable();
            item_features.peek().is_some()
                && item_features.all(|feature| feature.end() <= prefix_len)
        })
        .count()
}

fn features_cover_items(
    features: &[MultiModalFeature],
    kind: MultimodalKind,
    item_count: usize,
) -> bool {
    (0..item_count).all(|item| {
        features
            .iter()
            .any(|feature| feature.kind == kind && feature.item_range.contains(&item))
    })
}

pub struct PrefixCacheManagerV2 {
    caches: IndexMap<CacheKey, CacheElement>,
    paged_recurrent_caches: IndexMap<Vec<BlockHash>, Vec<RecurrentStateSnapshot>>,
    paged_recurrent_bytes: usize,
    paged_recurrent_reported: bool,
    n_on_device: usize,
    no_prefix_cache: bool,
    has_paged_attention: bool,
}

#[derive(Clone)]
pub enum MatchingCache {
    Normal {
        normal: Vec<Option<KvCache>>,
        recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
        images_to_keep: usize,
        audios_to_keep: usize,
        video_frames_to_keep: usize,
        toks: Vec<u32>,
        offset: usize,
    },
}

impl PrefixCacheManagerV2 {
    pub fn new(n_on_device: usize, no_prefix_cache: bool, has_paged_attention: bool) -> Self {
        if !no_prefix_cache && !has_paged_attention {
            info!("Prefix caching enabled (sequence-level, non-paged attention). Expect higher multi-turn throughput for both text and multimodal.");
        }
        PrefixCacheManagerV2 {
            caches: IndexMap::new(),
            paged_recurrent_caches: IndexMap::new(),
            paged_recurrent_bytes: 0,
            paged_recurrent_reported: false,
            n_on_device,
            no_prefix_cache,
            has_paged_attention,
        }
    }

    /// Whether recurrent prefix snapshots would be kept; callers skip the device copy otherwise.
    pub fn accepts_paged_recurrent_prefix(&self) -> bool {
        !self.no_prefix_cache && self.has_paged_attention && self.paged_recurrent_capacity() > 0
    }

    fn paged_recurrent_capacity(&self) -> usize {
        self.n_on_device.max(1)
    }

    fn snapshot_bytes(snapshots: &[RecurrentStateSnapshot]) -> usize {
        snapshots
            .iter()
            .map(|snapshot| {
                snapshot.conv_state.elem_count() * snapshot.conv_state.dtype().size_in_bytes()
                    + snapshot.recurrent_state.elem_count()
                        * snapshot.recurrent_state.dtype().size_in_bytes()
            })
            .sum()
    }

    /// This always keeps the cache on the device.
    pub fn add_sequence(
        &mut self,
        seq: &mut Sequence,
        recurrent_snapshots: Option<Vec<RecurrentStateSnapshot>>,
    ) {
        // Do not cache if prefix caching disabled
        if self.no_prefix_cache {
            return;
        }

        // For paged attention, prefix caching is handled by the KVCacheManager.
        // PrefixCacheManagerV2 only handles non-paged attention caching.
        if !self.has_paged_attention {
            let cache = seq.normal_cache().to_vec();
            let recurrent_snapshots = recurrent_snapshots.map(|snapshots| CachedRecurrentState {
                len: cache
                    .iter()
                    .flatten()
                    .filter(|layer| !matches!(layer, KvCache::Shared { .. }))
                    .map(KvCache::current_seq_len)
                    .min()
                    .unwrap_or(0),
                snapshots,
            });

            self.caches.insert(
                CacheKey::new(seq.get_toks().to_vec(), seq.adapter_generation()),
                CacheElement {
                    cache,
                    recurrent_snapshots,
                    image_hashes: seq.image_hashes().map(|x| x.to_vec()),
                    audio_hashes: seq.audio_hashes().map(|x| x.to_vec()),
                    video_hashes: seq.video_hashes().map(|x| x.to_vec()),
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
            let first_non_none = cache
                .cache
                .iter()
                .find_or_first(|x| x.as_ref().is_some_and(|kv| kv.k().ok().flatten().is_some()));
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
                KvCache::Shared { .. } => continue,
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
            let first_non_none = cache
                .cache
                .iter()
                .find_or_first(|x| x.as_ref().is_some_and(|kv| kv.k().ok().flatten().is_some()));
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
                KvCache::Shared { .. } => continue,
            };

            if !matches!(cache_device, Device::Cpu) {
                cache.cache.clear();
                n_evicted += 1;
            }
        }

        self.caches.retain(|_tokens, cache| !cache.cache.is_empty());

        if n_evicted > 0 {
            metrics::counter!("mistralrs_prefix_cache_evictions_total").increment(n_evicted as u64);
        }
        Ok(n_evicted)
    }

    /// Evict all the caches.
    pub fn evict_all_caches(&mut self) -> Result<usize> {
        // caches is empty under paged attention, where the prefix cache lives in the block pool
        let len = self.caches.len() + self.paged_recurrent_caches.len();
        self.caches.clear();
        self.paged_recurrent_caches.clear();
        self.paged_recurrent_bytes = 0;
        if len > 0 {
            metrics::counter!("mistralrs_prefix_cache_evictions_total").increment(len as u64);
        }
        Ok(len)
    }

    /// Add a recurrent-state snapshot for a paged-attention block-hash prefix key.
    /// This is used by hybrid models to restore recurrent states alongside paged KV prefix hits.
    pub fn add_paged_recurrent_prefix(
        &mut self,
        key: Vec<BlockHash>,
        snapshots: Vec<RecurrentStateSnapshot>,
    ) {
        if self.no_prefix_cache
            || !self.has_paged_attention
            || key.is_empty()
            || snapshots.is_empty()
        {
            return;
        }

        // Maintain LRU order by reinserting on update.
        if let Some(prev) = self.paged_recurrent_caches.shift_remove(&key) {
            self.paged_recurrent_bytes = self
                .paged_recurrent_bytes
                .saturating_sub(Self::snapshot_bytes(&prev));
        }
        self.paged_recurrent_bytes += Self::snapshot_bytes(&snapshots);
        self.paged_recurrent_caches.insert(key, snapshots);

        while self.paged_recurrent_caches.len() > self.paged_recurrent_capacity() {
            let Some((_, evicted)) = self.paged_recurrent_caches.shift_remove_index(0) else {
                break;
            };
            self.paged_recurrent_bytes = self
                .paged_recurrent_bytes
                .saturating_sub(Self::snapshot_bytes(&evicted));
        }

        // One snapshot spans every recurrent layer at once, so the entry count says nothing about
        // what the store costs. Report it rather than leaving the ceiling invisible.
        if self.paged_recurrent_caches.len() == self.paged_recurrent_capacity()
            && !self.paged_recurrent_reported
        {
            self.paged_recurrent_reported = true;
            info!(
                "Recurrent prefix cache full at {} entries, {} MB. Adjust with `--prefix-cache-n`.",
                self.paged_recurrent_caches.len(),
                self.paged_recurrent_bytes / (1024 * 1024),
            );
        }
    }

    /// Lookup a recurrent-state snapshot for a paged-attention block-hash prefix key.
    /// Returns a cloned snapshot and updates LRU order.
    pub fn get_paged_recurrent_prefix(
        &mut self,
        key: &[BlockHash],
    ) -> Option<Vec<RecurrentStateSnapshot>> {
        if self.no_prefix_cache || !self.has_paged_attention || key.is_empty() {
            return None;
        }

        let key = key.to_vec();
        let snapshots = self.paged_recurrent_caches.shift_remove(&key)?;
        let out = snapshots.clone();
        self.paged_recurrent_caches.insert(key, snapshots);
        Some(out)
    }

    pub fn get_longest_paged_recurrent_prefix(
        &mut self,
        block_hashes: &[BlockHash],
        max_blocks: usize,
    ) -> Option<(usize, Vec<RecurrentStateSnapshot>)> {
        if self.no_prefix_cache || !self.has_paged_attention {
            return None;
        }

        let max_blocks = max_blocks.min(block_hashes.len());
        for n_blocks in (1..=max_blocks).rev() {
            if let Some(snapshots) = self.get_paged_recurrent_prefix(&block_hashes[..n_blocks]) {
                return Some((n_blocks, snapshots));
            }
        }
        None
    }

    /// Search for a matching cache given some tokens. Image-containing sequences are now cached too.
    pub fn search_for_matching_cache(
        &mut self,
        toks: &[u32],
        adapter_generation: Option<AdapterGenerationId>,
        mm_features: &[MultiModalFeature],
        image_hashes: Option<&[u64]>,
        audio_hashes: Option<&[u64]>,
        video_hashes: Option<&[u64]>,
    ) -> Result<Option<MatchingCache>> {
        // Do not search if prefix caching disabled or no tokens
        if self.no_prefix_cache || toks.is_empty() {
            return Ok(None);
        }

        if self.has_paged_attention {
            // For paged attention, prefix caching is handled by the KVCacheManager.
            // PrefixCacheManagerV2 only handles non-paged attention caching.
            return Ok(None);
        }
        if video_hashes.is_some_and(|hashes| !hashes.is_empty()) {
            return Ok(None);
        }
        if image_hashes.is_some_and(|hashes| {
            !features_cover_items(mm_features, MultimodalKind::Image, hashes.len())
        }) || audio_hashes.is_some_and(|hashes| {
            !features_cover_items(mm_features, MultimodalKind::Audio, hashes.len())
        }) {
            return Ok(None);
        }

        let toks = Tokens(toks.to_vec());

        let mut best_match: Option<(usize, &CacheElement)> = None;
        for (k, v) in &self.caches {
            if k.adapter_generation != adapter_generation {
                continue;
            }
            if v.video_hashes
                .as_ref()
                .is_some_and(|hashes| !hashes.is_empty())
            {
                continue;
            }
            let match_len = toks.shared_prefix_len(&k.tokens);
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

            let input_image_count = image_hashes.map_or(0, |h| h.len());
            let input_audio_count = audio_hashes.map_or(0, |h| h.len());
            if v.image_hashes.as_ref().map_or(0, Vec::len) > input_image_count
                || v.audio_hashes.as_ref().map_or(0, Vec::len) > input_audio_count
            {
                continue;
            }

            // The cache holds kv only for forwarded positions; a finished sequence's
            // final sampled token has no kv row, so clamp the text match to coverage.
            // Shared layers mirror their owner and always report zero, so skip them.
            let cache_len = v.coverage_len();
            let match_len = match_len.min(cache_len);
            let match_len = clamp_prefix_len_for_media_hashes(
                match_len,
                mm_features,
                MultimodalKind::Image,
                images_match_until,
                input_image_count,
            );
            let match_len = clamp_prefix_len_for_media_hashes(
                match_len,
                mm_features,
                MultimodalKind::Audio,
                audios_match_until,
                input_audio_count,
            );
            let match_len = clamp_normal_prefix_len(match_len, mm_features);
            if match_len == 0 {
                continue;
            }
            let cached_input_images = image_hashes.map_or(0, |hashes| {
                fully_cached_item_count(match_len, mm_features, MultimodalKind::Image, hashes.len())
            });
            if images_match_until < cached_input_images {
                continue;
            }
            let cached_input_audios = audio_hashes.map_or(0, |hashes| {
                fully_cached_item_count(match_len, mm_features, MultimodalKind::Audio, hashes.len())
            });
            if audios_match_until < cached_input_audios {
                continue;
            }

            // Sliding/rotating caches only retain a fixed tail. If a cache has already
            // truncated older tokens, it can still safely serve an exact extension of the
            // cached prefix, but it cannot be rewound to an earlier logical length. Skip such
            // candidates here so a rolled-over cache does not block a shorter valid prefix hit.
            if !v.can_rewind_to(match_len) {
                continue;
            }
            if v.recurrent_snapshots
                .as_ref()
                .is_some_and(|state| state.len != match_len)
            {
                continue;
            }

            if best_match.as_ref().is_none_or(|(len, _)| match_len > *len) {
                best_match = Some((match_len, v));
            }
        }

        if let Some((match_len, cache_element)) = best_match {
            let new_toks = toks.0[match_len..].to_vec();
            if new_toks.is_empty() {
                return Ok(None);
            }

            let mut cache = cache_element.clone();
            let images_to_keep = if let Some(input_hashes) = image_hashes {
                input_hashes.len().saturating_sub(fully_cached_item_count(
                    match_len,
                    mm_features,
                    MultimodalKind::Image,
                    input_hashes.len(),
                ))
            } else {
                0
            };
            let audios_to_keep = if let Some(input_hashes) = audio_hashes {
                input_hashes.len().saturating_sub(fully_cached_item_count(
                    match_len,
                    mm_features,
                    MultimodalKind::Audio,
                    input_hashes.len(),
                ))
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
                recurrent_snapshots: cache.recurrent_snapshots.map(|state| state.snapshots),
                images_to_keep,
                audios_to_keep,
                video_frames_to_keep: 0,
                toks: new_toks,
                offset: match_len,
            }));
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::{
        CacheElement, CacheKey, CachedRecurrentState, MatchingCache, PrefixCacheManagerV2,
    };
    use crate::{
        kv_cache::{KvCache, RecurrentStateSnapshot, RotatingCache, SingleCache},
        paged_attention::block_hash::{
            MultiModalFeature, MultimodalAttentionPolicy, MultimodalKind,
        },
        AdapterGenerationId,
    };

    fn make_cache_tensor(len: usize) -> candle_core::Result<Tensor> {
        Tensor::zeros((1, 1, len, 1), DType::F32, &Device::Cpu)
    }

    fn make_rotating_kv_cache(
        logical_len: usize,
        sliding_window: usize,
    ) -> candle_core::Result<KvCache> {
        let src = make_cache_tensor(logical_len)?;
        let mut k = RotatingCache::new(2, sliding_window, sliding_window);
        let mut v = RotatingCache::new(2, sliding_window, sliding_window);
        let _ = k.append(&src)?;
        let _ = v.append(&src)?;
        Ok(KvCache::Rotating { k, v })
    }

    fn make_normal_kv_cache(logical_len: usize) -> candle_core::Result<KvCache> {
        let src = make_cache_tensor(logical_len)?;
        let mut k = SingleCache::new(2, logical_len, logical_len);
        let mut v = SingleCache::new(2, logical_len, logical_len);
        k.append(&src)?;
        v.append(&src)?;
        Ok(KvCache::Normal { k, v })
    }

    fn make_recurrent_snapshot() -> candle_core::Result<RecurrentStateSnapshot> {
        Ok(RecurrentStateSnapshot {
            conv_state: Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu)?,
            recurrent_state: Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu)?,
            state_layout: crate::kv_cache::RecurrentStateLayout::Opaque,
        })
    }

    fn generation(value: u8) -> AdapterGenerationId {
        AdapterGenerationId::from_bytes([value; 32])
    }

    #[test]
    fn adapter_generations_do_not_cross_hit() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);
        let generation_a = generation(1);
        let generation_b = generation(2);

        prefix_cacher.caches.insert(
            CacheKey::new(vec![1, 2, 3], Some(generation_a)),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(3)?)],
                recurrent_snapshots: None,
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );

        let query = [1, 2, 3, 4];
        assert!(prefix_cacher
            .search_for_matching_cache(&query, None, &[], None, None, None)?
            .is_none());
        assert!(prefix_cacher
            .search_for_matching_cache(&query, Some(generation_b), &[], None, None, None)?
            .is_none());
        assert!(prefix_cacher
            .search_for_matching_cache(&query, Some(generation_a), &[], None, None, None)?
            .is_some());

        Ok(())
    }

    #[test]
    fn skips_rolled_over_rotating_candidate_that_cannot_rewind() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);

        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into(),
            CacheElement {
                cache: vec![Some(make_rotating_kv_cache(10, 4)?)],
                recurrent_snapshots: None,
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );
        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 9].into(),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(6)?)],
                recurrent_snapshots: None,
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );

        let hit = prefix_cacher.search_for_matching_cache(
            &[1, 2, 3, 4, 5, 6, 7, 99],
            None,
            &[],
            None,
            None,
            None,
        )?;

        match hit {
            Some(MatchingCache::Normal { toks, offset, .. }) => {
                assert_eq!(offset, 5);
                assert_eq!(toks, vec![6, 7, 99]);
            }
            None => panic!("expected a shorter valid prefix-cache hit"),
        }

        Ok(())
    }

    #[test]
    fn allows_exact_extension_from_rolled_over_rotating_cache() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);

        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into(),
            CacheElement {
                cache: vec![Some(make_rotating_kv_cache(10, 4)?)],
                recurrent_snapshots: None,
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );

        let hit = prefix_cacher.search_for_matching_cache(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            None,
            &[],
            None,
            None,
            None,
        )?;

        match hit {
            Some(MatchingCache::Normal { toks, offset, .. }) => {
                assert_eq!(offset, 10);
                assert_eq!(toks, vec![11]);
            }
            None => panic!("expected exact-extension prefix-cache hit"),
        }

        Ok(())
    }

    #[test]
    fn hybrid_snapshot_only_matches_its_exact_boundary() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);
        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 6, 7, 8].into(),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(8)?)],
                recurrent_snapshots: Some(CachedRecurrentState {
                    len: 8,
                    snapshots: vec![make_recurrent_snapshot()?],
                }),
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );
        prefix_cacher.caches.insert(
            vec![1, 2, 3, 9].into(),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(4)?)],
                recurrent_snapshots: None,
                audio_hashes: None,
                image_hashes: None,
                video_hashes: None,
            },
        );

        let exact = prefix_cacher.search_for_matching_cache(
            &[1, 2, 3, 4, 5, 6, 7, 8, 10],
            None,
            &[],
            None,
            None,
            None,
        )?;
        match exact {
            Some(MatchingCache::Normal {
                recurrent_snapshots,
                offset,
                ..
            }) => {
                assert_eq!(offset, 8);
                assert!(recurrent_snapshots.is_some());
            }
            None => panic!("expected an exact hybrid snapshot hit"),
        }

        let partial = prefix_cacher.search_for_matching_cache(
            &[1, 2, 3, 4, 10],
            None,
            &[],
            None,
            None,
            None,
        )?;
        match partial {
            Some(MatchingCache::Normal {
                recurrent_snapshots,
                offset,
                ..
            }) => {
                assert_eq!(offset, 3);
                assert!(recurrent_snapshots.is_none());
            }
            None => panic!("expected the shorter non-hybrid cache hit"),
        }

        Ok(())
    }

    #[test]
    fn normal_multimodal_hit_clamps_and_retains_boundary_items() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);
        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 6, 7, 8].into(),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(8)?)],
                recurrent_snapshots: None,
                image_hashes: Some(vec![11, 22]),
                audio_hashes: Some(vec![33]),
                video_hashes: None,
            },
        );
        let features = vec![
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![11],
                offset: 1,
                length: 2,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Audio,
                item_range: 0..1,
                hashes: vec![33],
                offset: 5,
                length: 2,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 1..2,
                hashes: vec![22],
                offset: 6,
                length: 3,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
        ];

        let hit = prefix_cacher.search_for_matching_cache(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9],
            None,
            &features,
            Some(&[11, 22]),
            Some(&[33]),
            None,
        )?;

        match hit {
            Some(MatchingCache::Normal {
                images_to_keep,
                audios_to_keep,
                toks,
                offset,
                ..
            }) => {
                assert_eq!(offset, 5);
                assert_eq!(toks, vec![6, 7, 8, 9]);
                assert_eq!(images_to_keep, 1);
                assert_eq!(audios_to_keep, 1);
            }
            None => panic!("expected a boundary-clamped multimodal prefix-cache hit"),
        }

        Ok(())
    }

    #[test]
    fn normal_multimodal_hit_reuses_text_before_a_hash_mismatch() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);
        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4, 5, 6, 7, 8].into(),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(8)?)],
                recurrent_snapshots: None,
                image_hashes: Some(vec![11, 22]),
                audio_hashes: None,
                video_hashes: None,
            },
        );
        let features = vec![
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![11],
                offset: 2,
                length: 2,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 1..2,
                hashes: vec![99],
                offset: 6,
                length: 2,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
        ];

        let hit = prefix_cacher.search_for_matching_cache(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9],
            None,
            &features,
            Some(&[11, 99]),
            None,
            None,
        )?;

        match hit {
            Some(MatchingCache::Normal {
                images_to_keep,
                toks,
                offset,
                ..
            }) => {
                assert_eq!(offset, 6);
                assert_eq!(toks, vec![7, 8, 9]);
                assert_eq!(images_to_keep, 1);
            }
            None => panic!("expected reuse before the mismatched image"),
        }

        Ok(())
    }

    #[test]
    fn normal_prefix_cache_rejects_video_hits() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);
        prefix_cacher.caches.insert(
            vec![1, 2, 3].into(),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(3)?)],
                recurrent_snapshots: None,
                image_hashes: None,
                audio_hashes: None,
                video_hashes: Some(vec![44]),
            },
        );

        assert!(prefix_cacher
            .search_for_matching_cache(&[1, 2, 3, 4], None, &[], None, None, Some(&[44]))?
            .is_none());
        assert!(prefix_cacher
            .search_for_matching_cache(&[1, 2, 3, 4], None, &[], None, None, None)?
            .is_none());

        Ok(())
    }

    #[test]
    fn normal_multimodal_hit_rejects_empty_layout() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);
        prefix_cacher.caches.insert(
            vec![1, 2, 3].into(),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(3)?)],
                recurrent_snapshots: None,
                image_hashes: Some(vec![11]),
                audio_hashes: Some(vec![22]),
                video_hashes: None,
            },
        );

        assert!(prefix_cacher
            .search_for_matching_cache(&[1, 2, 3, 4], None, &[], Some(&[11]), Some(&[22]), None,)?
            .is_none());

        Ok(())
    }

    #[test]
    fn normal_multimodal_hit_rejects_incomplete_layout() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);
        prefix_cacher.caches.insert(
            vec![1, 2, 3, 4].into(),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(4)?)],
                recurrent_snapshots: None,
                image_hashes: Some(vec![11, 22]),
                audio_hashes: Some(vec![33]),
                video_hashes: None,
            },
        );
        let image_features = vec![
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 0..1,
                hashes: vec![11],
                offset: 1,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
            MultiModalFeature {
                kind: MultimodalKind::Image,
                item_range: 1..2,
                hashes: vec![22],
                offset: 2,
                length: 1,
                attention_policy: MultimodalAttentionPolicy::Causal,
                splittable: false,
            },
        ];

        assert!(prefix_cacher
            .search_for_matching_cache(
                &[1, 2, 3, 4, 5],
                None,
                &image_features[..1],
                Some(&[11, 22]),
                Some(&[33]),
                None,
            )?
            .is_none());
        assert!(prefix_cacher
            .search_for_matching_cache(
                &[1, 2, 3, 4, 5],
                None,
                &image_features,
                Some(&[11, 22]),
                Some(&[33]),
                None,
            )?
            .is_none());

        Ok(())
    }

    #[test]
    fn normal_multimodal_hit_without_hashes_stops_before_first_item() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, false);
        prefix_cacher.caches.insert(
            vec![1, 2, 3].into(),
            CacheElement {
                cache: vec![Some(make_normal_kv_cache(3)?)],
                recurrent_snapshots: None,
                image_hashes: None,
                audio_hashes: None,
                video_hashes: None,
            },
        );
        let features = vec![MultiModalFeature {
            kind: MultimodalKind::Image,
            item_range: 0..1,
            hashes: vec![11],
            offset: 1,
            length: 1,
            attention_policy: MultimodalAttentionPolicy::Causal,
            splittable: false,
        }];

        let hit = prefix_cacher.search_for_matching_cache(
            &[1, 2, 3, 4],
            None,
            &features,
            Some(&[11]),
            None,
            None,
        )?;
        match hit {
            Some(MatchingCache::Normal {
                images_to_keep,
                toks,
                offset,
                ..
            }) => {
                assert_eq!(offset, 1);
                assert_eq!(toks, vec![2, 3, 4]);
                assert_eq!(images_to_keep, 1);
            }
            None => panic!("expected reuse before the unverifiable image"),
        }

        let leading_feature = [MultiModalFeature {
            offset: 0,
            ..features[0].clone()
        }];
        assert!(prefix_cacher
            .search_for_matching_cache(
                &[1, 2, 3, 4],
                None,
                &leading_feature,
                Some(&[11]),
                None,
                None,
            )?
            .is_none());

        Ok(())
    }
}
