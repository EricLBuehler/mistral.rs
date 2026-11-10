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
const PAGED_RECURRENT_PREFIX_CHECKPOINTS_USED_METRIC: &str =
    "mistralrs_paged_recurrent_prefix_checkpoints_used";
const PAGED_RECURRENT_PREFIX_BYTES_USED_METRIC: &str =
    "mistralrs_paged_recurrent_prefix_bytes_used";
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
    paged_block_retention: Option<PrefixBlockRetention>,
    #[cfg(any(feature = "cuda", test))]
    runtime_memory_reservation: Option<Arc<crate::resource_plan::RuntimeMemoryReservation>>,
    n_on_device: usize,
    no_prefix_cache: bool,
    has_paged_attention: bool,
}

#[cfg(any(feature = "cuda", test))]
impl Drop for PrefixCacheManagerV2 {
    fn drop(&mut self) {
        if let Some(reservation) = &self.runtime_memory_reservation {
            reservation.set_retained_prefix_snapshots(0);
        }
    }
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
            paged_block_retention: None,
            #[cfg(any(feature = "cuda", test))]
            runtime_memory_reservation: None,
            n_on_device,
            no_prefix_cache,
            has_paged_attention,
        };
        manager.publish_paged_recurrent_metrics();
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

    pub fn accepts_sequence_cache(&self) -> bool {
        !self.no_prefix_cache && !self.has_paged_attention
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

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn attach_runtime_memory_reservation(
        &mut self,
        reservation: Arc<crate::resource_plan::RuntimeMemoryReservation>,
    ) {
        self.runtime_memory_reservation = Some(reservation);
        self.publish_paged_recurrent_metrics();
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

    #[allow(clippy::cast_precision_loss)]
    fn publish_paged_recurrent_metrics(&self) {
        #[cfg(any(feature = "cuda", test))]
        self.publish_retained_prefix_snapshots(self.paged_recurrent_caches.len());
        let (used, capacity) = self.paged_recurrent_owner_metric_values();
        metrics::gauge!(PAGED_RECURRENT_PREFIX_OWNERS_USED_METRIC).set(f64::from(used));
        metrics::gauge!(PAGED_RECURRENT_PREFIX_OWNERS_CAPACITY_METRIC).set(f64::from(capacity));
        metrics::gauge!(PAGED_RECURRENT_PREFIX_CHECKPOINTS_USED_METRIC)
            .set(self.paged_recurrent_caches.len() as f64);
        metrics::gauge!(PAGED_RECURRENT_PREFIX_BYTES_USED_METRIC)
            .set(self.paged_recurrent_bytes as f64);
    }

    #[cfg(any(feature = "cuda", test))]
    fn publish_retained_prefix_snapshots(&self, count: usize) {
        if let Some(reservation) = &self.runtime_memory_reservation {
            // Multiple owners can share one physical checkpoint.
            reservation.set_retained_prefix_snapshots(count);
        }
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
        let mut removed_entries = 0;
        for key in revoked_keys {
            let Some(entry) = self.paged_recurrent_caches.shift_remove(&key) else {
                continue;
            };
            #[cfg(any(feature = "cuda", test))]
            self.publish_retained_prefix_snapshots(self.paged_recurrent_caches.len());
            removed_entries += 1;
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
        }
        if removed_entries > 0 {
            self.publish_paged_recurrent_metrics();
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
        #[cfg(any(feature = "cuda", test))]
        self.publish_retained_prefix_snapshots(0);
        self.caches.clear();
        self.paged_recurrent_caches.clear();
        self.paged_recurrent_sequence_keys.clear();
        self.paged_recurrent_bytes = 0;
        self.publish_paged_recurrent_metrics();
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
        self.publish_paged_recurrent_metrics();

        debug_assert!(
            self.paged_recurrent_caches.len() <= self.paged_recurrent_sequence_keys.len()
                && self.paged_recurrent_caches.len() <= self.paged_recurrent_capacity()
        );
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
        #[cfg(any(feature = "cuda", test))]
        self.publish_retained_prefix_snapshots(self.paged_recurrent_caches.len());
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
#[path = "prefix_cacher_tests.rs"]
mod tests;
