use candle_core::{Device, Result};
use indexmap::IndexMap;
use itertools::Itertools;
use std::{any::Any, collections::HashSet, sync::Arc};
use tracing::info;

use crate::{
    kv_cache::RecurrentStateSnapshot,
    paged_attention::{
        block_hash::{BlockHash, MultiModalFeature, MultimodalKind},
        block_pool::{PrefixBlockRetention, PrefixBlockRetentionLease},
    },
    pipeline::KvCache,
    sequence::Sequence,
    AdapterGenerationId,
};

const PAGED_RECURRENT_PREFIX_OWNERS_CAPACITY_METRIC: &str =
    "mistralrs_paged_recurrent_prefix_owners_capacity";
const PAGED_RECURRENT_PREFIX_OWNERS_USED_METRIC: &str =
    "mistralrs_paged_recurrent_prefix_owners_used";
const PAGED_RECURRENT_PREFIX_OWNER_EVICTIONS_METRIC: &str =
    "mistralrs_paged_recurrent_prefix_owner_evictions_total";
const CAPACITY_EVICTION_REASON: &str = "capacity";
const BLOCK_PRESSURE_EVICTION_REASON: &str = "block_pressure";

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
    paged_recurrent_caches: IndexMap<Vec<BlockHash>, PagedRecurrentCacheEntry>,
    paged_recurrent_sequence_keys: IndexMap<BlockHash, Vec<BlockHash>>,
    paged_recurrent_bytes: usize,
    paged_recurrent_reported: bool,
    paged_block_retention: Option<PrefixBlockRetention>,
    n_on_device: usize,
    no_prefix_cache: bool,
    has_paged_attention: bool,
}

pub trait PagedAuxiliaryPrefixState: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn bytes(&self) -> usize;
}

#[derive(Clone)]
pub struct PagedPrefixCheckpoint {
    pub recurrent_snapshots: Vec<RecurrentStateSnapshot>,
    pub auxiliary: Option<Arc<dyn PagedAuxiliaryPrefixState>>,
}

struct PagedRecurrentCacheEntry {
    snapshots: Vec<RecurrentStateSnapshot>,
    auxiliary: Option<Arc<dyn PagedAuxiliaryPrefixState>>,
    owners: HashSet<BlockHash>,
    retention: Option<PrefixBlockRetentionLease>,
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
        let manager = PrefixCacheManagerV2 {
            caches: IndexMap::new(),
            paged_recurrent_caches: IndexMap::new(),
            paged_recurrent_sequence_keys: IndexMap::new(),
            paged_recurrent_bytes: 0,
            paged_recurrent_reported: false,
            paged_block_retention: None,
            n_on_device,
            no_prefix_cache,
            has_paged_attention,
        };
        manager.publish_paged_recurrent_owner_metrics();
        metrics::counter!(
            PAGED_RECURRENT_PREFIX_OWNER_EVICTIONS_METRIC,
            "reason" => CAPACITY_EVICTION_REASON
        )
        .increment(0);
        metrics::counter!(
            PAGED_RECURRENT_PREFIX_OWNER_EVICTIONS_METRIC,
            "reason" => BLOCK_PRESSURE_EVICTION_REASON
        )
        .increment(0);
        manager
    }

    /// Whether recurrent prefix snapshots would be kept; callers skip the device copy otherwise.
    pub fn accepts_paged_recurrent_prefix(&self) -> bool {
        !self.no_prefix_cache && self.has_paged_attention && self.paged_recurrent_capacity() > 0
    }

    pub(crate) fn attach_paged_block_retention(&mut self, retention: PrefixBlockRetention) {
        assert!(
            self.paged_recurrent_caches.is_empty(),
            "paged block retention must be attached before caching prefixes"
        );
        if self.accepts_paged_recurrent_prefix() {
            retention.enable();
            self.paged_block_retention = Some(retention);
        }
    }

    fn paged_recurrent_capacity(&self) -> usize {
        self.n_on_device
    }

    fn paged_recurrent_owner_metric_values(&self) -> (u32, u32) {
        let used = u32::try_from(self.paged_recurrent_sequence_keys.len())
            .expect("paged recurrent prefix owner usage exceeds u32");
        let capacity = if self.no_prefix_cache || !self.has_paged_attention {
            0
        } else {
            u32::try_from(self.paged_recurrent_capacity())
                .expect("paged recurrent prefix owner capacity exceeds u32")
        };
        (used, capacity)
    }

    fn publish_paged_recurrent_owner_metrics(&self) {
        let (used, capacity) = self.paged_recurrent_owner_metric_values();
        metrics::gauge!(PAGED_RECURRENT_PREFIX_OWNERS_USED_METRIC).set(f64::from(used));
        metrics::gauge!(PAGED_RECURRENT_PREFIX_OWNERS_CAPACITY_METRIC).set(f64::from(capacity));
    }

    pub(crate) fn prune_revoked_paged_recurrent_entries(&mut self) -> usize {
        let revoked_keys = self
            .paged_recurrent_caches
            .iter()
            .filter_map(|(key, entry)| {
                entry
                    .retention
                    .as_ref()
                    .is_some_and(|retention| !retention.is_active())
                    .then_some(key.clone())
            })
            .collect::<Vec<_>>();
        let mut removed_owners = 0;
        for key in revoked_keys {
            let Some(entry) = self.paged_recurrent_caches.shift_remove(&key) else {
                continue;
            };
            self.paged_recurrent_bytes =
                self.paged_recurrent_bytes
                    .saturating_sub(Self::checkpoint_bytes(
                        &entry.snapshots,
                        entry.auxiliary.as_deref(),
                    ));
            for owner in entry.owners {
                if self
                    .paged_recurrent_sequence_keys
                    .get(&owner)
                    .is_some_and(|owner_key| owner_key == &key)
                {
                    self.paged_recurrent_sequence_keys.shift_remove(&owner);
                    removed_owners += 1;
                }
            }
        }
        if removed_owners > 0 {
            metrics::counter!("mistralrs_prefix_cache_evictions_total")
                .increment(removed_owners as u64);
            metrics::counter!(
                PAGED_RECURRENT_PREFIX_OWNER_EVICTIONS_METRIC,
                "reason" => BLOCK_PRESSURE_EVICTION_REASON
            )
            .increment(removed_owners as u64);
            self.publish_paged_recurrent_owner_metrics();
        }
        removed_owners
    }

    fn checkpoint_bytes(
        snapshots: &[RecurrentStateSnapshot],
        auxiliary: Option<&dyn PagedAuxiliaryPrefixState>,
    ) -> usize {
        let recurrent = snapshots
            .iter()
            .map(|snapshot| {
                snapshot.conv_state.elem_count() * snapshot.conv_state.dtype().size_in_bytes()
                    + snapshot.recurrent_state.elem_count()
                        * snapshot.recurrent_state.dtype().size_in_bytes()
            })
            .sum::<usize>();
        recurrent.saturating_add(auxiliary.map_or(0, |state| state.bytes()))
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
        let len = self.caches.len() + self.paged_recurrent_sequence_keys.len();
        self.caches.clear();
        self.paged_recurrent_caches.clear();
        self.paged_recurrent_sequence_keys.clear();
        self.paged_recurrent_bytes = 0;
        self.publish_paged_recurrent_owner_metrics();
        if len > 0 {
            metrics::counter!("mistralrs_prefix_cache_evictions_total").increment(len as u64);
        }
        Ok(len)
    }

    /// Add a recurrent-state snapshot for a paged-attention block-hash prefix key.
    /// This is used by hybrid models to restore recurrent states alongside paged KV prefix hits.
    pub fn add_paged_recurrent_prefix(
        &mut self,
        owner: BlockHash,
        key: Vec<BlockHash>,
        snapshots: Vec<RecurrentStateSnapshot>,
        auxiliary: Option<Arc<dyn PagedAuxiliaryPrefixState>>,
    ) {
        if self.no_prefix_cache
            || !self.has_paged_attention
            || self.paged_recurrent_capacity() == 0
            || key.is_empty()
            || snapshots.is_empty()
        {
            return;
        }
        self.prune_revoked_paged_recurrent_entries();

        if let Some(stale_key) = self.paged_recurrent_sequence_keys.shift_remove(&owner) {
            if stale_key != key {
                self.remove_paged_recurrent_owner(owner, &stale_key);
            }
        }

        let previous = self.paged_recurrent_caches.shift_remove(&key);
        if let Some(entry) = previous.as_ref() {
            self.paged_recurrent_bytes =
                self.paged_recurrent_bytes
                    .saturating_sub(Self::checkpoint_bytes(
                        &entry.snapshots,
                        entry.auxiliary.as_deref(),
                    ));
        }
        let (mut owners, previous_auxiliary, retention) = previous.map_or_else(
            || (HashSet::new(), None, None),
            |entry| (entry.owners, entry.auxiliary, entry.retention),
        );
        let auxiliary = auxiliary.or(previous_auxiliary);
        let retention = retention.or_else(|| {
            self.paged_block_retention
                .as_ref()
                .map(|retention| retention.retain(&key))
        });
        if let Some(retention) = retention.as_ref() {
            retention.touch();
        }
        owners.insert(owner);
        self.paged_recurrent_bytes += Self::checkpoint_bytes(&snapshots, auxiliary.as_deref());
        self.paged_recurrent_caches.insert(
            key.clone(),
            PagedRecurrentCacheEntry {
                snapshots,
                auxiliary,
                owners,
                retention,
            },
        );
        self.paged_recurrent_sequence_keys.insert(owner, key);

        let mut capacity_evictions = 0;
        while self.paged_recurrent_sequence_keys.len() > self.paged_recurrent_capacity() {
            let Some((evicted_owner, evicted_key)) =
                self.paged_recurrent_sequence_keys.shift_remove_index(0)
            else {
                break;
            };
            self.remove_paged_recurrent_owner(evicted_owner, &evicted_key);
            capacity_evictions += 1;
        }
        if capacity_evictions > 0 {
            metrics::counter!("mistralrs_prefix_cache_evictions_total")
                .increment(capacity_evictions as u64);
            metrics::counter!(
                PAGED_RECURRENT_PREFIX_OWNER_EVICTIONS_METRIC,
                "reason" => CAPACITY_EVICTION_REASON
            )
            .increment(capacity_evictions as u64);
        }
        self.publish_paged_recurrent_owner_metrics();

        debug_assert!(
            self.paged_recurrent_caches.len() <= self.paged_recurrent_sequence_keys.len()
                && self.paged_recurrent_caches.len() <= self.paged_recurrent_capacity()
        );

        if self.paged_recurrent_sequence_keys.len() == self.paged_recurrent_capacity()
            && !self.paged_recurrent_reported
        {
            self.paged_recurrent_reported = true;
            info!(
                "Recurrent prefix cache full at {} sequences, {} checkpoints, {} MB. Adjust with `--prefix-cache-n`.",
                self.paged_recurrent_sequence_keys.len(),
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
        current_owner: BlockHash,
    ) -> Option<PagedPrefixCheckpoint> {
        self.prune_revoked_paged_recurrent_entries();
        let out = self.peek_paged_recurrent_prefix(key)?;
        self.promote_paged_recurrent_prefix(key, current_owner);
        Some(out)
    }

    pub fn peek_paged_recurrent_prefix(&self, key: &[BlockHash]) -> Option<PagedPrefixCheckpoint> {
        if self.no_prefix_cache || !self.has_paged_attention || key.is_empty() {
            return None;
        }

        let entry = self.paged_recurrent_caches.get(key)?;
        if entry
            .retention
            .as_ref()
            .is_some_and(|retention| !retention.is_active())
        {
            return None;
        }
        Some(PagedPrefixCheckpoint {
            recurrent_snapshots: entry.snapshots.clone(),
            auxiliary: entry.auxiliary.clone(),
        })
    }

    pub fn promote_paged_recurrent_prefix(&mut self, key: &[BlockHash], current_owner: BlockHash) {
        self.prune_revoked_paged_recurrent_entries();
        let Some(entry) = self.paged_recurrent_caches.get(key) else {
            return;
        };
        if let Some(retention) = entry.retention.as_ref() {
            retention.touch();
        }
        let promote_owner = entry
            .owners
            .contains(&current_owner)
            .then_some(current_owner)
            .or_else(|| entry.owners.iter().copied().next());
        if let Some(promote_owner) = promote_owner {
            if let Some(key) = self
                .paged_recurrent_sequence_keys
                .shift_remove(&promote_owner)
            {
                self.paged_recurrent_sequence_keys
                    .insert(promote_owner, key);
            }
        }
    }

    pub fn has_paged_recurrent_owner(&mut self, owner: BlockHash) -> bool {
        self.prune_revoked_paged_recurrent_entries();
        self.paged_recurrent_sequence_keys.contains_key(&owner)
    }

    fn remove_paged_recurrent_owner(&mut self, owner: BlockHash, key: &[BlockHash]) {
        let remove_entry = self
            .paged_recurrent_caches
            .get_mut(key)
            .is_some_and(|entry| {
                entry.owners.remove(&owner);
                entry.owners.is_empty()
            });
        if !remove_entry {
            return;
        }
        let entry = self
            .paged_recurrent_caches
            .shift_remove(key)
            .expect("empty recurrent checkpoint entry disappeared");
        self.paged_recurrent_bytes =
            self.paged_recurrent_bytes
                .saturating_sub(Self::checkpoint_bytes(
                    &entry.snapshots,
                    entry.auxiliary.as_deref(),
                ));
    }

    pub fn get_longest_paged_recurrent_prefix(
        &mut self,
        block_hashes: &[BlockHash],
        max_blocks: usize,
    ) -> Option<(usize, PagedPrefixCheckpoint)> {
        self.prune_revoked_paged_recurrent_entries();
        let (n_blocks, checkpoint) =
            self.peek_longest_paged_recurrent_prefix(block_hashes, max_blocks)?;
        self.promote_paged_recurrent_prefix(&block_hashes[..n_blocks], *block_hashes.last()?);
        Some((n_blocks, checkpoint))
    }

    pub fn peek_longest_paged_recurrent_prefix(
        &self,
        block_hashes: &[BlockHash],
        max_blocks: usize,
    ) -> Option<(usize, PagedPrefixCheckpoint)> {
        if self.no_prefix_cache || !self.has_paged_attention {
            return None;
        }

        let max_blocks = max_blocks.min(block_hashes.len());
        let key = self
            .paged_recurrent_caches
            .iter()
            .filter(|(key, entry)| {
                key.len() <= max_blocks
                    && block_hashes.starts_with(key)
                    && entry
                        .retention
                        .as_ref()
                        .is_none_or(PrefixBlockRetentionLease::is_active)
            })
            .map(|(key, _)| key)
            .max_by_key(|key| key.len())?
            .clone();
        let n_blocks = key.len();
        self.peek_paged_recurrent_prefix(&key)
            .map(|checkpoint| (n_blocks, checkpoint))
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
    use std::{
        collections::HashSet,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    use super::{
        CacheElement, CacheKey, CachedRecurrentState, MatchingCache, PagedAuxiliaryPrefixState,
        PrefixCacheManagerV2,
    };
    use crate::{
        kv_cache::{KvCache, RecurrentStateSnapshot, RotatingCache, SingleCache},
        paged_attention::block_hash::{
            compute_block_hashes, BlockHash, MultiModalFeature, MultimodalAttentionPolicy,
            MultimodalKind,
        },
        paged_attention::block_pool::BlockPool,
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

    struct TestAuxiliaryPrefixState {
        bytes: usize,
        drops: Arc<AtomicUsize>,
    }

    impl Drop for TestAuxiliaryPrefixState {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    impl PagedAuxiliaryPrefixState for TestAuxiliaryPrefixState {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn bytes(&self) -> usize {
            self.bytes
        }
    }

    fn generation(value: u8) -> AdapterGenerationId {
        AdapterGenerationId::from_bytes([value; 32])
    }

    fn block_hashes(start: u32, len: usize) -> Vec<BlockHash> {
        let tokens = (start..start + len as u32).collect::<Vec<_>>();
        compute_block_hashes(&tokens, 1, &[], &[])
    }

    #[test]
    fn advancing_paged_recurrent_sequence_replaces_its_checkpoint() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(2, false, true);
        let hashes = block_hashes(10, 8);
        let owner = *hashes.last().unwrap();

        for n_blocks in 1..=hashes.len() {
            prefix_cacher.add_paged_recurrent_prefix(
                owner,
                hashes[..n_blocks].to_vec(),
                vec![make_recurrent_snapshot()?],
                None,
            );
        }

        assert_eq!(prefix_cacher.paged_recurrent_caches.len(), 1);
        assert_eq!(prefix_cacher.paged_recurrent_sequence_keys.len(), 1);
        assert_eq!(prefix_cacher.paged_recurrent_sequence_keys[&owner], hashes);
        assert!(prefix_cacher.paged_recurrent_caches.contains_key(&hashes));
        assert!(prefix_cacher
            .get_paged_recurrent_prefix(&hashes[..6], owner)
            .is_none());

        Ok(())
    }

    #[test]
    fn paged_recurrent_capacity_tracks_independent_sequences() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(2, false, true);
        let hashes_a = block_hashes(10, 3);
        let hashes_b = block_hashes(20, 3);
        let owner_a = *hashes_a.last().unwrap();
        let owner_b = *hashes_b.last().unwrap();

        prefix_cacher.add_paged_recurrent_prefix(
            owner_a,
            hashes_a[..1].to_vec(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        prefix_cacher.add_paged_recurrent_prefix(
            owner_b,
            hashes_b[..1].to_vec(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        prefix_cacher.add_paged_recurrent_prefix(
            owner_a,
            hashes_a.clone(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        prefix_cacher.add_paged_recurrent_prefix(
            owner_b,
            hashes_b.clone(),
            vec![make_recurrent_snapshot()?],
            None,
        );

        assert_eq!(prefix_cacher.paged_recurrent_caches.len(), 2);
        assert_eq!(
            prefix_cacher.paged_recurrent_sequence_keys[&owner_a],
            hashes_a
        );
        assert_eq!(
            prefix_cacher.paged_recurrent_sequence_keys[&owner_b],
            hashes_b
        );
        assert!(prefix_cacher
            .get_paged_recurrent_prefix(&hashes_a, owner_a)
            .is_some());
        assert!(prefix_cacher
            .get_paged_recurrent_prefix(&hashes_b, owner_b)
            .is_some());

        Ok(())
    }

    #[test]
    fn retained_blocks_follow_shared_owner_replacement_and_clear() -> candle_core::Result<()> {
        let pool = BlockPool::new(8, true, 1);
        let retention = pool.prefix_block_retention();
        let mut prefix_cacher = PrefixCacheManagerV2::new(2, false, true);
        prefix_cacher.attach_paged_block_retention(retention.clone());
        let hashes_a = compute_block_hashes(&[10, 11, 12, 13], 1, &[], &[]);
        let hashes_b = compute_block_hashes(&[10, 11, 20, 21], 1, &[], &[]);
        let hashes_c = block_hashes(30, 2);
        let owner_a = *hashes_a.last().unwrap();
        let owner_b = *hashes_b.last().unwrap();
        let owner_c = *hashes_c.last().unwrap();

        prefix_cacher.add_paged_recurrent_prefix(
            owner_a,
            hashes_a[..2].to_vec(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        prefix_cacher.add_paged_recurrent_prefix(
            owner_b,
            hashes_b[..2].to_vec(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        assert_eq!(retention.num_entries(), 1);
        assert_eq!(retention.num_hashes(), 2);

        prefix_cacher.add_paged_recurrent_prefix(
            owner_a,
            hashes_a.clone(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        assert_eq!(retention.num_entries(), 2);
        prefix_cacher.add_paged_recurrent_prefix(
            owner_c,
            hashes_c.clone(),
            vec![make_recurrent_snapshot()?],
            None,
        );

        assert!(!prefix_cacher.has_paged_recurrent_owner(owner_b));
        assert!(prefix_cacher.has_paged_recurrent_owner(owner_a));
        assert!(prefix_cacher.has_paged_recurrent_owner(owner_c));
        assert_eq!(retention.num_entries(), 2);
        assert_eq!(retention.num_hashes(), hashes_a.len() + hashes_c.len());

        prefix_cacher.evict_all_caches()?;
        assert_eq!(retention.num_entries(), 0);
        assert_eq!(retention.num_hashes(), 0);
        Ok(())
    }

    #[test]
    fn allocation_pressure_invalidates_paired_recurrent_checkpoint() -> candle_core::Result<()> {
        let mut pool = BlockPool::new(4, true, 1);
        let retention = pool.prefix_block_retention();
        let hashes = block_hashes(10, 2);
        let block_ids = pool.get_new_blocks(hashes.len()).unwrap();
        pool.cache_full_blocks(&block_ids, &hashes, 0, hashes.len(), 0);

        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, true);
        prefix_cacher.attach_paged_block_retention(retention.clone());
        let revocations = retention.revocation_monitor();
        let owner = *hashes.last().unwrap();
        prefix_cacher.add_paged_recurrent_prefix(
            owner,
            hashes.clone(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        assert!(!revocations.take_pending());
        pool.free_blocks(&block_ids.iter().rev().copied().collect::<Vec<_>>());
        assert_eq!(pool.num_retained_physical_blocks(), 2);
        assert_eq!(retention.num_entries(), 1);

        assert_eq!(pool.get_new_blocks(2).unwrap().len(), 2);
        assert_eq!(retention.num_entries(), 0);
        assert_eq!(pool.num_retained_physical_blocks(), 0);
        assert!(prefix_cacher.peek_paged_recurrent_prefix(&hashes).is_none());
        assert_eq!(prefix_cacher.paged_recurrent_caches.len(), 1);
        assert!(revocations.take_pending());
        assert!(!revocations.take_pending());
        assert_eq!(prefix_cacher.prune_revoked_paged_recurrent_entries(), 1);
        assert!(prefix_cacher.paged_recurrent_caches.is_empty());
        assert!(prefix_cacher.paged_recurrent_sequence_keys.is_empty());
        assert_eq!(prefix_cacher.paged_recurrent_bytes, 0);
        assert!(!prefix_cacher.has_paged_recurrent_owner(owner));
        Ok(())
    }

    #[test]
    fn paged_recurrent_owner_metrics_track_logical_occupancy() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(2, false, true);
        let hashes_a = compute_block_hashes(&[10, 11, 12, 13], 1, &[], &[]);
        let hashes_b = compute_block_hashes(&[10, 11, 20, 21], 1, &[], &[]);
        let hashes_c = block_hashes(30, 2);
        let owner_a = *hashes_a.last().unwrap();
        let owner_b = *hashes_b.last().unwrap();
        let owner_c = *hashes_c.last().unwrap();

        assert_eq!(prefix_cacher.paged_recurrent_owner_metric_values(), (0, 2));
        prefix_cacher.add_paged_recurrent_prefix(
            owner_a,
            hashes_a[..2].to_vec(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        prefix_cacher.add_paged_recurrent_prefix(
            owner_b,
            hashes_b[..2].to_vec(),
            vec![make_recurrent_snapshot()?],
            None,
        );

        assert_eq!(prefix_cacher.paged_recurrent_owner_metric_values(), (2, 2));
        assert_eq!(prefix_cacher.paged_recurrent_caches.len(), 1);

        prefix_cacher.add_paged_recurrent_prefix(
            owner_c,
            hashes_c,
            vec![make_recurrent_snapshot()?],
            None,
        );

        assert_eq!(prefix_cacher.paged_recurrent_owner_metric_values(), (2, 2));
        assert!(!prefix_cacher.has_paged_recurrent_owner(owner_a));
        assert!(prefix_cacher.has_paged_recurrent_owner(owner_b));
        assert!(prefix_cacher.has_paged_recurrent_owner(owner_c));

        prefix_cacher.add_paged_recurrent_prefix(
            owner_b,
            hashes_b,
            vec![make_recurrent_snapshot()?],
            None,
        );
        assert_eq!(prefix_cacher.paged_recurrent_owner_metric_values(), (2, 2));

        assert_eq!(prefix_cacher.evict_all_caches()?, 2);
        assert_eq!(prefix_cacher.paged_recurrent_owner_metric_values(), (0, 2));
        assert_eq!(
            PrefixCacheManagerV2::new(2, true, true).paged_recurrent_owner_metric_values(),
            (0, 0)
        );
        assert_eq!(
            PrefixCacheManagerV2::new(2, false, false).paged_recurrent_owner_metric_values(),
            (0, 0)
        );

        Ok(())
    }

    #[test]
    fn identical_paged_recurrent_prefixes_share_one_entry() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(2, false, true);
        let hashes_a = compute_block_hashes(&[10, 11, 12, 13, 14, 15], 1, &[], &[]);
        let hashes_b = compute_block_hashes(&[10, 11, 20, 21, 22, 23], 1, &[], &[]);
        let owner_a = *hashes_a.last().unwrap();
        let owner_b = *hashes_b.last().unwrap();

        prefix_cacher.add_paged_recurrent_prefix(
            owner_a,
            hashes_a[..2].to_vec(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        let bytes = prefix_cacher.paged_recurrent_bytes;
        prefix_cacher.add_paged_recurrent_prefix(
            owner_b,
            hashes_b[..2].to_vec(),
            vec![make_recurrent_snapshot()?],
            None,
        );

        assert_eq!(prefix_cacher.paged_recurrent_caches.len(), 1);
        assert_eq!(prefix_cacher.paged_recurrent_bytes, bytes);
        assert_eq!(
            prefix_cacher.paged_recurrent_caches[&hashes_a[..2]].owners,
            HashSet::from([owner_a, owner_b])
        );

        prefix_cacher.add_paged_recurrent_prefix(
            owner_a,
            hashes_a[..4].to_vec(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        assert_eq!(prefix_cacher.paged_recurrent_caches.len(), 2);
        assert_eq!(
            prefix_cacher.paged_recurrent_caches[&hashes_a[..2]].owners,
            HashSet::from([owner_b])
        );
        assert_eq!(
            prefix_cacher.paged_recurrent_caches[&hashes_a[..4]].owners,
            HashSet::from([owner_a])
        );

        prefix_cacher.add_paged_recurrent_prefix(
            owner_a,
            hashes_a.clone(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        assert_eq!(prefix_cacher.paged_recurrent_caches.len(), 2);
        assert_eq!(
            prefix_cacher.paged_recurrent_caches[&hashes_a[..2]].owners,
            HashSet::from([owner_b])
        );
        assert!(!prefix_cacher
            .paged_recurrent_caches
            .contains_key(&hashes_a[..4]));
        assert_eq!(
            prefix_cacher.paged_recurrent_sequence_keys[&owner_a],
            hashes_a
        );

        Ok(())
    }

    #[test]
    fn shared_checkpoint_lookup_promotes_only_the_current_owner() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(2, false, true);
        let hashes_a = compute_block_hashes(&[10, 11, 12, 13], 1, &[], &[]);
        let hashes_b = compute_block_hashes(&[10, 11, 20, 21], 1, &[], &[]);
        let hashes_c = block_hashes(30, 2);
        let owner_a = *hashes_a.last().unwrap();
        let owner_b = *hashes_b.last().unwrap();
        let owner_c = *hashes_c.last().unwrap();

        prefix_cacher.add_paged_recurrent_prefix(
            owner_a,
            hashes_a[..2].to_vec(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        prefix_cacher.add_paged_recurrent_prefix(
            owner_b,
            hashes_b[..2].to_vec(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        assert!(prefix_cacher
            .get_longest_paged_recurrent_prefix(&hashes_a, 2)
            .is_some());
        prefix_cacher.add_paged_recurrent_prefix(
            owner_c,
            hashes_c.clone(),
            vec![make_recurrent_snapshot()?],
            None,
        );

        assert!(!prefix_cacher
            .paged_recurrent_sequence_keys
            .contains_key(&owner_b));
        assert!(prefix_cacher
            .paged_recurrent_sequence_keys
            .contains_key(&owner_a));
        assert_eq!(
            prefix_cacher.paged_recurrent_caches[&hashes_a[..2]].owners,
            HashSet::from([owner_a])
        );
        assert_eq!(prefix_cacher.paged_recurrent_caches.len(), 2);
        assert_eq!(prefix_cacher.paged_recurrent_sequence_keys.len(), 2);

        prefix_cacher.evict_all_caches()?;
        assert!(prefix_cacher.paged_recurrent_caches.is_empty());
        assert!(prefix_cacher.paged_recurrent_sequence_keys.is_empty());
        assert_eq!(prefix_cacher.paged_recurrent_bytes, 0);

        Ok(())
    }

    #[test]
    fn zero_capacity_keeps_no_paged_recurrent_snapshots() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(0, false, true);
        let hashes = block_hashes(10, 2);
        let owner = *hashes.last().unwrap();
        prefix_cacher.add_paged_recurrent_prefix(
            owner,
            hashes.clone(),
            vec![make_recurrent_snapshot()?],
            None,
        );

        assert!(!prefix_cacher.accepts_paged_recurrent_prefix());
        assert!(prefix_cacher.paged_recurrent_caches.is_empty());
        assert!(prefix_cacher.paged_recurrent_sequence_keys.is_empty());
        Ok(())
    }

    #[test]
    fn auxiliary_checkpoint_bytes_and_lifetime_follow_lru_entries() -> candle_core::Result<()> {
        let mut prefix_cacher = PrefixCacheManagerV2::new(1, false, true);
        let hashes_a = block_hashes(10, 2);
        let hashes_b = block_hashes(20, 2);
        let owner_a = *hashes_a.last().unwrap();
        let owner_b = *hashes_b.last().unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let auxiliary = Arc::new(TestAuxiliaryPrefixState {
            bytes: 64,
            drops: Arc::clone(&drops),
        });

        prefix_cacher.add_paged_recurrent_prefix(
            owner_a,
            hashes_a.clone(),
            vec![make_recurrent_snapshot()?],
            Some(auxiliary.clone()),
        );
        drop(auxiliary);
        assert_eq!(prefix_cacher.paged_recurrent_bytes, 72);

        prefix_cacher.add_paged_recurrent_prefix(
            owner_a,
            hashes_a.clone(),
            vec![make_recurrent_snapshot()?],
            None,
        );
        assert_eq!(prefix_cacher.paged_recurrent_bytes, 72);
        let checkpoint = prefix_cacher
            .get_paged_recurrent_prefix(&hashes_a, owner_a)
            .expect("auxiliary checkpoint missing");
        assert!(checkpoint
            .auxiliary
            .as_deref()
            .expect("auxiliary state missing")
            .as_any()
            .is::<TestAuxiliaryPrefixState>());

        prefix_cacher.add_paged_recurrent_prefix(
            owner_b,
            hashes_b,
            vec![make_recurrent_snapshot()?],
            None,
        );
        assert_eq!(prefix_cacher.paged_recurrent_bytes, 8);
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        drop(checkpoint);
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        Ok(())
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
