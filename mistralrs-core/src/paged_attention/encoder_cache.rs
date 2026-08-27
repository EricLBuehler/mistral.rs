//! Encoder output cache for multimodal models.
//!
//! Caches vision/audio encoder outputs keyed by content hash so that identical
//! media across requests (or after a prefix-cache partial hit) can skip the
//! expensive encoder pass.  Uses a simple LRU eviction strategy.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::Tensor;
use indexmap::IndexMap;

/// Modality tag that disambiguates cache keys.
///
/// Identical pixel content can produce different encoder outputs depending on
/// whether it's processed as an image or as a video frame (different patch
/// budgets, different token counts). Including the modality in the cache key
/// prevents cross-modality collisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CacheModality {
    Image,
    Video,
    Audio,
}

/// Cache key combining a content hash with a modality tag.
pub type CacheKey = (CacheModality, u64);

const ENCODER_CACHE_RESIDENT_ENTRIES_METRIC: &str = "mistralrs_encoder_cache_resident_entries";
const ENCODER_CACHE_RESIDENT_LOGICAL_BYTES_METRIC: &str =
    "mistralrs_encoder_cache_resident_logical_bytes";
const ENCODER_CACHE_EVICTIONS_METRIC: &str = "mistralrs_encoder_cache_evictions_total";
const ENCODER_CACHE_INSERT_REJECTIONS_METRIC: &str =
    "mistralrs_encoder_cache_insert_rejections_total";
const ENCODER_CACHE_INTRA_BATCH_DEDUPLICATIONS_METRIC: &str =
    "mistralrs_encoder_cache_intra_batch_deduplications_total";
const ENTRY_CAPACITY_REASON: &str = "entry_capacity";
const LOGICAL_BYTE_CAPACITY_REASON: &str = "logical_byte_capacity";
const INCOMPATIBLE_SHAPE_REASON: &str = "incompatible_shape";
const ENTRY_EXCEEDS_LOGICAL_BYTE_CAPACITY_REASON: &str = "entry_exceeds_logical_byte_capacity";
const STORAGE_COMPACTION_FAILED_REASON: &str = "storage_compaction_failed";
static PROCESS_RESIDENCY: Mutex<ProcessResidency> = Mutex::new(ProcessResidency {
    entries: 0,
    logical_bytes: 0,
});

struct ProcessResidency {
    entries: usize,
    logical_bytes: usize,
}

fn adjust_process_residency(counter: &mut usize, previous: usize, current: usize) {
    if current >= previous {
        *counter = counter
            .checked_add(current - previous)
            .expect("encoder cache process residency overflow");
    } else {
        let removed = previous - current;
        *counter = counter
            .checked_sub(removed)
            .expect("encoder cache process residency underflow");
    }
}

#[allow(clippy::cast_precision_loss)]
fn publish_process_residency_metrics(residency: &ProcessResidency) {
    metrics::gauge!(ENCODER_CACHE_RESIDENT_ENTRIES_METRIC).set(residency.entries as f64);
    metrics::gauge!(ENCODER_CACHE_RESIDENT_LOGICAL_BYTES_METRIC)
        .set(residency.logical_bytes as f64);
}

fn update_process_residency(
    previous_entries: usize,
    current_entries: usize,
    previous_logical_bytes: usize,
    current_logical_bytes: usize,
) {
    let mut residency = PROCESS_RESIDENCY
        .lock()
        .expect("encoder cache process residency lock poisoned");
    adjust_process_residency(&mut residency.entries, previous_entries, current_entries);
    adjust_process_residency(
        &mut residency.logical_bytes,
        previous_logical_bytes,
        current_logical_bytes,
    );
    publish_process_residency_metrics(&residency);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EncoderCacheCapacityPolicy {
    Entries(usize),
    LogicalBytes(usize),
}

impl EncoderCacheCapacityPolicy {
    fn eviction_reason(self) -> &'static str {
        match self {
            Self::Entries(_) => ENTRY_CAPACITY_REASON,
            Self::LogicalBytes(_) => LOGICAL_BYTE_CAPACITY_REASON,
        }
    }
}

struct EncoderCacheEntry {
    outputs: Vec<Tensor>,
    logical_bytes: usize,
}

impl EncoderCacheEntry {
    fn new(outputs: Vec<Tensor>) -> Self {
        let logical_bytes = Self::logical_bytes(&outputs);
        Self {
            outputs,
            logical_bytes,
        }
    }

    fn compact(outputs: Vec<Tensor>) -> candle_core::Result<Self> {
        let outputs = outputs
            .into_iter()
            .map(|tensor| tensor.force_contiguous().map(|tensor| tensor.detach()))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self::new(outputs))
    }

    fn logical_bytes(outputs: &[Tensor]) -> usize {
        outputs.iter().fold(0usize, |total, tensor| {
            let tensor_bytes = tensor
                .elem_count()
                .checked_mul(tensor.dtype().size_in_bytes())
                .expect("encoder cache tensor byte size overflow");
            total
                .checked_add(tensor_bytes)
                .expect("encoder cache entry byte size overflow")
        })
    }
}

/// LRU cache for encoder outputs.
///
/// Each entry stores one or more tensors (e.g. Qwen3-VL returns both main
/// embeddings and deep-stack embeddings).  Keys combine the modality with the
/// `u64` content hash computed for images/audio/video in
/// [`crate::sequence::Sequence`].
///
/// The cache is typically stored behind `Arc<Mutex<…>>` on each model struct
/// and accessed from `forward()` via interior mutability.
pub struct EncoderCacheManager {
    /// Insertion-ordered map; most-recently-used entries live at the back.
    cache: IndexMap<CacheKey, EncoderCacheEntry>,
    capacity_policy: EncoderCacheCapacityPolicy,
    cached_logical_bytes: usize,
    hits: Arc<AtomicUsize>,
    misses: Arc<AtomicUsize>,
}

impl EncoderCacheManager {
    /// Create a new encoder cache with the historical entry-count capacity.
    pub fn new(max_entries: usize) -> Self {
        Self::with_capacity_policy(EncoderCacheCapacityPolicy::Entries(max_entries))
    }

    #[cfg(test)]
    fn with_max_logical_bytes(max_bytes: usize) -> Self {
        Self::with_capacity_policy(EncoderCacheCapacityPolicy::LogicalBytes(max_bytes))
    }

    fn with_capacity_policy(capacity_policy: EncoderCacheCapacityPolicy) -> Self {
        let initial_capacity = match capacity_policy {
            EncoderCacheCapacityPolicy::Entries(max_entries) => max_entries,
            EncoderCacheCapacityPolicy::LogicalBytes(_) => 0,
        };
        let manager = Self {
            cache: IndexMap::with_capacity(initial_capacity),
            capacity_policy,
            cached_logical_bytes: 0,
            hits: Arc::new(AtomicUsize::new(0)),
            misses: Arc::new(AtomicUsize::new(0)),
        };
        {
            let residency = PROCESS_RESIDENCY
                .lock()
                .expect("encoder cache process residency lock poisoned");
            publish_process_residency_metrics(&residency);
        }
        metrics::counter!(
            ENCODER_CACHE_EVICTIONS_METRIC,
            "reason" => ENTRY_CAPACITY_REASON
        )
        .increment(0);
        metrics::counter!(
            ENCODER_CACHE_EVICTIONS_METRIC,
            "reason" => LOGICAL_BYTE_CAPACITY_REASON
        )
        .increment(0);
        metrics::counter!(
            ENCODER_CACHE_EVICTIONS_METRIC,
            "reason" => INCOMPATIBLE_SHAPE_REASON
        )
        .increment(0);
        metrics::counter!(
            ENCODER_CACHE_INSERT_REJECTIONS_METRIC,
            "reason" => ENTRY_EXCEEDS_LOGICAL_BYTE_CAPACITY_REASON
        )
        .increment(0);
        metrics::counter!(
            ENCODER_CACHE_INSERT_REJECTIONS_METRIC,
            "reason" => STORAGE_COMPACTION_FAILED_REASON
        )
        .increment(0);
        manager
    }

    pub fn set_max_logical_bytes(&mut self, max_bytes: usize) {
        assert!(max_bytes > 0, "encoder cache byte capacity must be nonzero");
        self.set_capacity_policy(EncoderCacheCapacityPolicy::LogicalBytes(max_bytes));
    }

    fn set_capacity_policy(&mut self, capacity_policy: EncoderCacheCapacityPolicy) {
        let previous_entries = self.cache.len();
        let previous_logical_bytes = self.cached_logical_bytes;
        self.capacity_policy = capacity_policy;

        let mut evicted = 0usize;
        while self.exceeds_current_capacity() {
            let (_, oldest) = self
                .cache
                .shift_remove_index(0)
                .expect("nonempty encoder cache required for capacity eviction");
            self.cached_logical_bytes = self
                .cached_logical_bytes
                .checked_sub(oldest.logical_bytes)
                .expect("encoder cache byte accounting underflow");
            evicted = evicted
                .checked_add(1)
                .expect("encoder cache eviction count overflow");
        }
        if evicted > 0 {
            metrics::counter!(
                ENCODER_CACHE_EVICTIONS_METRIC,
                "reason" => capacity_policy.eviction_reason()
            )
            .increment(u64::try_from(evicted).unwrap_or(u64::MAX));
        }
        update_process_residency(
            previous_entries,
            self.cache.len(),
            previous_logical_bytes,
            self.cached_logical_bytes,
        );
    }

    /// Number of entries currently resident in the cache.
    pub fn resident_entries(&self) -> usize {
        self.cache.len()
    }

    /// Total logical payload bytes held by all cached tensors.
    pub fn cached_logical_bytes(&self) -> usize {
        self.cached_logical_bytes
    }

    /// Return clones of the hit/miss counter Arcs (hits, misses).
    pub fn counters(&self) -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        (self.hits.clone(), self.misses.clone())
    }

    /// Look up a cached encoder output by modality + content hash.
    ///
    /// On hit the entry is moved to the back (most-recently-used position)
    /// and the tensors are cloned (cheap, Candle tensors are `Arc`-backed).
    pub fn get(&mut self, modality: CacheModality, content_hash: u64) -> Option<Vec<Tensor>> {
        self.get_validated(modality, content_hash, |_| true)
    }

    fn get_validated(
        &mut self,
        modality: CacheModality,
        content_hash: u64,
        is_valid: impl FnOnce(&[Tensor]) -> bool,
    ) -> Option<Vec<Tensor>> {
        let key = (modality, content_hash);
        let previous_entries = self.cache.len();
        let previous_logical_bytes = self.cached_logical_bytes;
        // `shift_remove` + re-insert moves the entry to the back.
        if let Some(entry) = self.cache.shift_remove(&key) {
            if !is_valid(&entry.outputs) {
                self.cached_logical_bytes = self
                    .cached_logical_bytes
                    .checked_sub(entry.logical_bytes)
                    .expect("encoder cache byte accounting underflow");
                self.misses.fetch_add(1, Ordering::Relaxed);
                metrics::counter!(
                    ENCODER_CACHE_EVICTIONS_METRIC,
                    "reason" => INCOMPATIBLE_SHAPE_REASON
                )
                .increment(1);
                update_process_residency(
                    previous_entries,
                    self.cache.len(),
                    previous_logical_bytes,
                    self.cached_logical_bytes,
                );
                return None;
            }
            let cloned = entry.outputs.clone();
            self.cache.insert(key, entry);
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(cloned)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Insert a new encoder output into the cache.
    ///
    /// If the cache is at capacity the least-recently-used (front) entry is
    /// evicted first.
    pub fn insert(&mut self, modality: CacheModality, content_hash: u64, outputs: Vec<Tensor>) {
        let key = (modality, content_hash);
        let entry_bytes = EncoderCacheEntry::logical_bytes(&outputs);
        if matches!(
            self.capacity_policy,
            EncoderCacheCapacityPolicy::LogicalBytes(max_bytes)
                if max_bytes > 0 && entry_bytes > max_bytes
        ) {
            metrics::counter!(
                ENCODER_CACHE_INSERT_REJECTIONS_METRIC,
                "reason" => ENTRY_EXCEEDS_LOGICAL_BYTE_CAPACITY_REASON
            )
            .increment(1);
            return;
        }
        let entry = match EncoderCacheEntry::compact(outputs) {
            Ok(entry) => entry,
            Err(_) => {
                metrics::counter!(
                    ENCODER_CACHE_INSERT_REJECTIONS_METRIC,
                    "reason" => STORAGE_COMPACTION_FAILED_REASON
                )
                .increment(1);
                return;
            }
        };

        let previous_entries = self.cache.len();
        let previous_logical_bytes = self.cached_logical_bytes;

        if let Some(previous) = self.cache.shift_remove(&key) {
            self.cached_logical_bytes = self
                .cached_logical_bytes
                .checked_sub(previous.logical_bytes)
                .expect("encoder cache byte accounting underflow");
        }

        let mut evicted = 0usize;
        while self.exceeds_capacity_with(entry.logical_bytes) {
            let (_, oldest) = self
                .cache
                .shift_remove_index(0)
                .expect("nonempty encoder cache required for capacity eviction");
            self.cached_logical_bytes = self
                .cached_logical_bytes
                .checked_sub(oldest.logical_bytes)
                .expect("encoder cache byte accounting underflow");
            evicted = evicted
                .checked_add(1)
                .expect("encoder cache eviction count overflow");
        }
        if evicted > 0 {
            metrics::counter!(
                ENCODER_CACHE_EVICTIONS_METRIC,
                "reason" => self.capacity_policy.eviction_reason()
            )
            .increment(u64::try_from(evicted).unwrap_or(u64::MAX));
        }

        self.cached_logical_bytes = self
            .cached_logical_bytes
            .checked_add(entry.logical_bytes)
            .expect("encoder cache byte accounting overflow");
        self.cache.insert(key, entry);
        update_process_residency(
            previous_entries,
            self.cache.len(),
            previous_logical_bytes,
            self.cached_logical_bytes,
        );
    }

    fn exceeds_capacity_with(&self, entry_bytes: usize) -> bool {
        match self.capacity_policy {
            EncoderCacheCapacityPolicy::Entries(max_entries) => {
                max_entries > 0 && self.cache.len() >= max_entries
            }
            EncoderCacheCapacityPolicy::LogicalBytes(max_bytes) => {
                max_bytes > 0
                    && self
                        .cached_logical_bytes
                        .checked_add(entry_bytes)
                        .expect("encoder cache byte accounting overflow")
                        > max_bytes
            }
        }
    }

    fn exceeds_current_capacity(&self) -> bool {
        match self.capacity_policy {
            EncoderCacheCapacityPolicy::Entries(max_entries) => {
                max_entries > 0 && self.cache.len() > max_entries
            }
            EncoderCacheCapacityPolicy::LogicalBytes(max_bytes) => {
                max_bytes > 0 && self.cached_logical_bytes > max_bytes
            }
        }
    }
}

impl Drop for EncoderCacheManager {
    fn drop(&mut self) {
        update_process_residency(self.resident_entries(), 0, self.cached_logical_bytes(), 0);
    }
}

fn record_intra_batch_deduplications(count: usize) {
    metrics::counter!(ENCODER_CACHE_INTRA_BATCH_DEDUPLICATIONS_METRIC)
        .increment(u64::try_from(count).unwrap_or(u64::MAX));
}

pub(crate) struct EncoderCacheBatchLookup {
    outputs: Vec<Option<Vec<Tensor>>>,
    miss_groups: Vec<Vec<usize>>,
}

impl EncoderCacheBatchLookup {
    pub(crate) fn lookup(
        modality: CacheModality,
        hashes: &[u64],
        cache: &Mutex<EncoderCacheManager>,
    ) -> Self {
        Self::lookup_validated(modality, hashes, cache, |_, _| true)
    }

    pub(crate) fn lookup_validated(
        modality: CacheModality,
        hashes: &[u64],
        cache: &Mutex<EncoderCacheManager>,
        mut is_valid: impl FnMut(usize, &[Tensor]) -> bool,
    ) -> Self {
        #[derive(Clone, Copy)]
        enum Resolution {
            Hit(usize),
            Miss(usize),
        }

        let mut outputs = vec![None; hashes.len()];
        let mut resolutions = IndexMap::<u64, Resolution>::new();
        let mut miss_groups = Vec::<Vec<usize>>::new();
        let mut deduplicated = 0usize;
        let mut guard = cache.lock().expect("encoder cache lock poisoned");
        for (index, &hash) in hashes.iter().enumerate() {
            if let Some(resolution) = resolutions.get(&hash) {
                match *resolution {
                    Resolution::Hit(first_idx) => outputs[index] = outputs[first_idx].clone(),
                    Resolution::Miss(group_idx) => miss_groups[group_idx].push(index),
                }
                deduplicated += 1;
                continue;
            }
            if let Some(cached) =
                guard.get_validated(modality, hash, |outputs| is_valid(index, outputs))
            {
                outputs[index] = Some(cached);
                resolutions.insert(hash, Resolution::Hit(index));
            } else {
                let group_idx = miss_groups.len();
                miss_groups.push(vec![index]);
                resolutions.insert(hash, Resolution::Miss(group_idx));
            }
        }
        drop(guard);
        record_intra_batch_deduplications(deduplicated);
        Self {
            outputs,
            miss_groups,
        }
    }

    pub(crate) fn uncached(item_count: usize) -> Self {
        Self {
            outputs: vec![None; item_count],
            miss_groups: (0..item_count).map(|index| vec![index]).collect(),
        }
    }

    pub(crate) fn miss_groups(&self) -> &[Vec<usize>] {
        &self.miss_groups
    }

    pub(crate) fn resolve_miss(&mut self, miss_idx: usize, outputs: Vec<Tensor>) {
        for &item_idx in &self.miss_groups[miss_idx] {
            self.outputs[item_idx] = Some(outputs.clone());
        }
    }

    pub(crate) fn into_outputs(self) -> candle_core::Result<Vec<Vec<Tensor>>> {
        self.outputs
            .into_iter()
            .map(|outputs| {
                outputs.ok_or_else(|| {
                    candle_core::Error::msg("encoder cache batch item is missing outputs")
                })
            })
            .collect()
    }

    fn into_optional_outputs(self) -> Vec<Option<Vec<Tensor>>> {
        self.outputs
    }
}

// ---------------------------------------------------------------------------
// Helper: cache-aware batch encoding for "Pattern A" models whose
// pixel_values have shape (N, C, H, W) with one image per dim-0 slice.
// ---------------------------------------------------------------------------

/// Encode a batch of images with per-image caching.
///
/// * `image_hashes` – one content hash per image, length **N**.
/// * `pixel_values` – stacked pixel tensor of shape `(N, C, H, W)`.
/// * `cache`        – shared encoder cache (behind `Mutex`).
/// * `encode_fn`    – called with a `(M, C, H, W)` tensor of **only** the
///   cache-miss images.  Must return `Vec<Tensor>` where each element is a
///   `(M, …)` tensor (the first element is the main embedding; extra elements
///   are auxiliary, e.g. deep-stack features).
///
/// Returns `Vec<Tensor>` in the same multi-output layout as `encode_fn`, but
/// now covering **all N** images (hits + misses reassembled in order).
pub fn cached_encode_images(
    modality: CacheModality,
    image_hashes: &[u64],
    pixel_values: &Tensor,
    cache: &Mutex<EncoderCacheManager>,
    encode_fn: impl FnOnce(&Tensor) -> candle_core::Result<Vec<Tensor>>,
) -> candle_core::Result<Vec<Tensor>> {
    let n_images = image_hashes.len();
    if n_images == 0 {
        return encode_fn(pixel_values);
    }
    debug_assert_eq!(
        n_images,
        pixel_values.dim(0)?,
        "image_hashes length must match pixel_values dim-0"
    );

    // Phase 1 – probe cache for each image.
    let mut lookup = EncoderCacheBatchLookup::lookup(modality, image_hashes, cache);

    // Fast path – all cached.
    if lookup.miss_groups().is_empty() {
        return assemble(lookup.into_optional_outputs(), n_images);
    }

    // Phase 2 – encode only the misses.
    let miss_pixels = if lookup.miss_groups().len() == n_images {
        // All misses – encode full batch without splitting.
        pixel_values.clone()
    } else {
        let slices: Vec<Tensor> = lookup
            .miss_groups()
            .iter()
            .map(|group| pixel_values.get(group[0]))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Tensor::stack(&slices, 0)?
    };

    let encoded = encode_fn(&miss_pixels)?;

    // Phase 3 – store per-image results in cache and fill `hits`.
    {
        let mut guard = cache.lock().expect("encoder cache lock poisoned");
        for batch_idx in 0..lookup.miss_groups().len() {
            let per_image: Vec<Tensor> = encoded
                .iter()
                .map(|t| t.get(batch_idx))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let first_idx = lookup.miss_groups()[batch_idx][0];
            guard.insert(modality, image_hashes[first_idx], per_image.clone());
            lookup.resolve_miss(batch_idx, per_image);
        }
    }

    assemble(lookup.into_optional_outputs(), n_images)
}

/// Re-stack per-image tensors into full-batch tensors.
fn assemble(hits: Vec<Option<Vec<Tensor>>>, n_images: usize) -> candle_core::Result<Vec<Tensor>> {
    // Determine how many output tensors per image (e.g. 1 for most, 2 for deepstack).
    let n_outputs = hits[0].as_ref().map(|v| v.len()).unwrap_or(1);

    let mut result = Vec::with_capacity(n_outputs);
    for out_idx in 0..n_outputs {
        let slices: Vec<Tensor> = (0..n_images)
            .map(|i| hits[i].as_ref().expect("all images should be resolved")[out_idx].clone())
            .collect();
        result.push(Tensor::stack(&slices, 0)?);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn dummy_tensor(val: f32) -> Tensor {
        Tensor::new(&[val], &Device::Cpu).unwrap()
    }

    fn zeros(elements: usize, dtype: DType) -> Tensor {
        Tensor::zeros((elements,), dtype, &Device::Cpu).unwrap()
    }

    #[test]
    fn test_process_residency_delta_balances() {
        let mut counter = 0;
        adjust_process_residency(&mut counter, 0, 7);
        assert_eq!(counter, 7);
        adjust_process_residency(&mut counter, 7, 3);
        assert_eq!(counter, 3);
        adjust_process_residency(&mut counter, 3, 3);
        assert_eq!(counter, 3);
    }

    // -----------------------------------------------------------------------
    // EncoderCacheManager unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_and_get() {
        let mut cache = EncoderCacheManager::new(4);
        assert_eq!(
            cache.capacity_policy,
            EncoderCacheCapacityPolicy::Entries(4)
        );
        let t = dummy_tensor(1.0);
        cache.insert(CacheModality::Image, 100, vec![t.clone()]);

        let result = cache.get(CacheModality::Image, 100);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].to_vec1::<f32>().unwrap(),
            t.to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn test_cached_logical_bytes_tracks_all_outputs_and_replacement() {
        let mut cache = EncoderCacheManager::new(4);
        cache.insert(
            CacheModality::Image,
            1,
            vec![zeros(6, DType::F32), zeros(5, DType::U8)],
        );
        cache.insert(CacheModality::Audio, 2, vec![zeros(2, DType::BF16)]);

        assert_eq!(cache.resident_entries(), 2);
        assert_eq!(cache.cached_logical_bytes(), 6 * 4 + 5 + 2 * 2);

        cache.insert(CacheModality::Image, 1, vec![zeros(3, DType::F16)]);

        assert_eq!(cache.resident_entries(), 2);
        assert_eq!(cache.cached_logical_bytes(), 3 * 2 + 2 * 2);
    }

    #[test]
    fn test_cache_compacts_tensor_views() {
        let backing =
            Tensor::from_slice(&[0f32, 1., 2., 3., 4., 5., 6., 7.], (2, 4), &Device::Cpu).unwrap();
        let view = backing.get(1).unwrap();
        assert_eq!(view.storage_and_layout().1.start_offset(), 4);

        let mut cache = EncoderCacheManager::new(4);
        cache.insert(CacheModality::Image, 1, vec![view]);

        let cached = cache.get(CacheModality::Image, 1).unwrap();
        assert_eq!(cached[0].storage_and_layout().1.start_offset(), 0);
        assert_eq!(cached[0].to_vec1::<f32>().unwrap(), vec![4., 5., 6., 7.]);
    }

    #[test]
    fn test_validated_get_discards_incompatible_cached_shape() {
        let mut cache = EncoderCacheManager::new(4);
        cache.insert(
            CacheModality::Image,
            1,
            vec![Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap()],
        );
        let (hits, misses) = cache.counters();

        let output = cache.get_validated(CacheModality::Image, 1, |outputs| {
            outputs[0].dims().first() == Some(&1)
        });

        assert!(output.is_none());
        assert_eq!(cache.resident_entries(), 0);
        assert_eq!(cache.cached_logical_bytes(), 0);
        assert_eq!(hits.load(Ordering::Relaxed), 0);
        assert_eq!(misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_logical_byte_capacity_evicts_lru_entries_until_new_entry_fits() {
        let mut cache = EncoderCacheManager::with_max_logical_bytes(16);
        cache.insert(CacheModality::Image, 1, vec![dummy_tensor(1.0)]);
        cache.insert(CacheModality::Image, 2, vec![dummy_tensor(2.0)]);
        cache.insert(CacheModality::Image, 3, vec![dummy_tensor(3.0)]);

        let _ = cache.get(CacheModality::Image, 1);
        cache.insert(CacheModality::Image, 4, vec![zeros(3, DType::F32)]);

        assert_eq!(cache.cached_logical_bytes(), 16);
        assert_eq!(cache.resident_entries(), 2);
        assert!(cache.get(CacheModality::Image, 1).is_some());
        assert!(cache.get(CacheModality::Image, 2).is_none());
        assert!(cache.get(CacheModality::Image, 3).is_none());
        assert!(cache.get(CacheModality::Image, 4).is_some());
    }

    #[test]
    fn test_reconfigure_capacity_evicts_lru_and_preserves_counters() {
        let mut cache = EncoderCacheManager::new(4);
        cache.insert(CacheModality::Image, 1, vec![zeros(4, DType::F32)]);
        cache.insert(CacheModality::Image, 2, vec![zeros(4, DType::F32)]);
        let (hits, misses) = cache.counters();
        let _ = cache.get(CacheModality::Image, 1);
        let _ = cache.get(CacheModality::Image, 3);

        cache.set_max_logical_bytes(16);

        assert_eq!(cache.resident_entries(), 1);
        assert_eq!(cache.cached_logical_bytes(), 16);
        assert!(cache.get(CacheModality::Image, 1).is_some());
        assert!(cache.get(CacheModality::Image, 2).is_none());
        assert_eq!(hits.load(Ordering::Relaxed), 2);
        assert_eq!(misses.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_oversized_entry_does_not_flush_logical_byte_limited_cache() {
        let mut cache = EncoderCacheManager::with_max_logical_bytes(8);
        cache.insert(CacheModality::Image, 1, vec![dummy_tensor(1.0)]);
        cache.insert(CacheModality::Image, 2, vec![zeros(3, DType::F32)]);

        assert_eq!(cache.cached_logical_bytes(), 4);
        assert_eq!(cache.resident_entries(), 1);
        assert!(cache.get(CacheModality::Image, 1).is_some());
        assert!(cache.get(CacheModality::Image, 2).is_none());
    }

    #[test]
    fn test_zero_capacity_limits_remain_unbounded() {
        let mut entry_cache = EncoderCacheManager::new(0);
        let mut byte_cache = EncoderCacheManager::with_max_logical_bytes(0);

        for (hash, value) in [(0, 0.0), (1, 1.0), (2, 2.0)] {
            entry_cache.insert(CacheModality::Image, hash, vec![dummy_tensor(value)]);
            byte_cache.insert(CacheModality::Image, hash, vec![dummy_tensor(value)]);
        }

        assert_eq!(entry_cache.resident_entries(), 3);
        assert_eq!(byte_cache.resident_entries(), 3);
        assert_eq!(entry_cache.cached_logical_bytes(), 12);
        assert_eq!(byte_cache.cached_logical_bytes(), 12);
    }

    #[test]
    fn test_get_miss() {
        let mut cache = EncoderCacheManager::new(4);
        assert!(cache.get(CacheModality::Image, 999).is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = EncoderCacheManager::new(3);
        cache.insert(CacheModality::Image, 1, vec![dummy_tensor(1.0)]);
        cache.insert(CacheModality::Image, 2, vec![dummy_tensor(2.0)]);
        cache.insert(CacheModality::Image, 3, vec![dummy_tensor(3.0)]);

        // Cache is full. Inserting a 4th should evict key=1 (oldest).
        cache.insert(CacheModality::Image, 4, vec![dummy_tensor(4.0)]);

        assert!(
            cache.get(CacheModality::Image, 1).is_none(),
            "key 1 should have been evicted"
        );
        assert!(cache.get(CacheModality::Image, 2).is_some());
        assert!(cache.get(CacheModality::Image, 3).is_some());
        assert!(cache.get(CacheModality::Image, 4).is_some());
    }

    #[test]
    fn test_get_bumps_lru_order() {
        let mut cache = EncoderCacheManager::new(3);
        cache.insert(CacheModality::Image, 1, vec![dummy_tensor(1.0)]);
        cache.insert(CacheModality::Image, 2, vec![dummy_tensor(2.0)]);
        cache.insert(CacheModality::Image, 3, vec![dummy_tensor(3.0)]);

        // Access key=1 to bump it to most-recently-used.
        let _ = cache.get(CacheModality::Image, 1);

        // Now key=2 is the oldest. Inserting key=4 should evict key=2.
        cache.insert(CacheModality::Image, 4, vec![dummy_tensor(4.0)]);

        assert!(
            cache.get(CacheModality::Image, 1).is_some(),
            "key 1 was accessed, should survive"
        );
        assert!(
            cache.get(CacheModality::Image, 2).is_none(),
            "key 2 should have been evicted"
        );
        assert!(cache.get(CacheModality::Image, 3).is_some());
        assert!(cache.get(CacheModality::Image, 4).is_some());
    }

    #[test]
    fn test_insert_duplicate_updates_lru() {
        let mut cache = EncoderCacheManager::new(3);
        cache.insert(CacheModality::Image, 1, vec![dummy_tensor(1.0)]);
        cache.insert(CacheModality::Image, 2, vec![dummy_tensor(2.0)]);
        cache.insert(CacheModality::Image, 3, vec![dummy_tensor(3.0)]);

        // Re-insert key=1 with new data, should bump it, not create duplicate.
        cache.insert(CacheModality::Image, 1, vec![dummy_tensor(10.0)]);

        // key=2 is now oldest.
        cache.insert(CacheModality::Image, 4, vec![dummy_tensor(4.0)]);

        assert!(
            cache.get(CacheModality::Image, 1).is_some(),
            "key 1 was re-inserted, should survive"
        );
        assert!(
            cache.get(CacheModality::Image, 2).is_none(),
            "key 2 should have been evicted"
        );

        // Verify the value was updated.
        let val = cache.get(CacheModality::Image, 1).unwrap()[0]
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(val, vec![10.0]);
    }

    #[test]
    fn test_multi_tensor_entries() {
        let mut cache = EncoderCacheManager::new(4);
        let t1 = dummy_tensor(1.0);
        let t2 = dummy_tensor(2.0);
        cache.insert(CacheModality::Image, 42, vec![t1, t2]);

        let result = cache.get(CacheModality::Image, 42).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].to_vec1::<f32>().unwrap(), vec![1.0]);
        assert_eq!(result[1].to_vec1::<f32>().unwrap(), vec![2.0]);
    }

    #[test]
    fn test_modality_isolation() {
        // Same hash under different modalities must NOT collide.
        let mut cache = EncoderCacheManager::new(4);
        cache.insert(CacheModality::Image, 42, vec![dummy_tensor(1.0)]);
        cache.insert(CacheModality::Video, 42, vec![dummy_tensor(2.0)]);
        cache.insert(CacheModality::Audio, 42, vec![dummy_tensor(3.0)]);

        assert_eq!(
            cache.get(CacheModality::Image, 42).unwrap()[0]
                .to_vec1::<f32>()
                .unwrap(),
            vec![1.0]
        );
        assert_eq!(
            cache.get(CacheModality::Video, 42).unwrap()[0]
                .to_vec1::<f32>()
                .unwrap(),
            vec![2.0]
        );
        assert_eq!(
            cache.get(CacheModality::Audio, 42).unwrap()[0]
                .to_vec1::<f32>()
                .unwrap(),
            vec![3.0]
        );
    }

    // -----------------------------------------------------------------------
    // cached_encode_images tests
    // -----------------------------------------------------------------------

    /// Build a (N, 1) pixel_values tensor for testing.
    fn make_pixels(vals: &[f32]) -> Tensor {
        Tensor::from_slice(vals, (vals.len(), 1), &Device::Cpu).unwrap()
    }

    #[test]
    fn test_cached_encode_all_miss() {
        let cache = Mutex::new(EncoderCacheManager::new(32));
        let pixels = make_pixels(&[10.0, 20.0, 30.0]);
        let hashes = [1u64, 2, 3];

        let result = cached_encode_images(CacheModality::Image, &hashes, &pixels, &cache, |pv| {
            // Identity encoder: return input as-is.
            Ok(vec![pv.clone()])
        })
        .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].dims(), &[3, 1]);
        assert_eq!(
            result[0].to_vec2::<f32>().unwrap(),
            vec![vec![10.0], vec![20.0], vec![30.0]]
        );

        // All entries should now be cached.
        let mut guard = cache.lock().unwrap();
        assert!(guard.get(CacheModality::Image, 1).is_some());
        assert!(guard.get(CacheModality::Image, 2).is_some());
        assert!(guard.get(CacheModality::Image, 3).is_some());
    }

    #[test]
    fn test_cached_encode_all_hit() {
        let cache = Mutex::new(EncoderCacheManager::new(32));

        // Pre-populate cache.
        {
            let mut guard = cache.lock().unwrap();
            guard.insert(
                CacheModality::Image,
                1,
                vec![Tensor::new(&[100.0f32], &Device::Cpu).unwrap()],
            );
            guard.insert(
                CacheModality::Image,
                2,
                vec![Tensor::new(&[200.0f32], &Device::Cpu).unwrap()],
            );
        }

        let pixels = make_pixels(&[10.0, 20.0]);
        let hashes = [1u64, 2];

        let encode_called = std::sync::atomic::AtomicBool::new(false);
        let result = cached_encode_images(CacheModality::Image, &hashes, &pixels, &cache, |pv| {
            encode_called.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(vec![pv.clone()])
        })
        .unwrap();

        assert!(
            !encode_called.load(std::sync::atomic::Ordering::SeqCst),
            "encode_fn should NOT be called when everything is cached"
        );
        // Should return the cached values, not the raw pixels.
        assert_eq!(
            result[0].to_vec2::<f32>().unwrap(),
            vec![vec![100.0], vec![200.0]]
        );
    }

    #[test]
    fn test_cached_encode_partial_hit() {
        let cache = Mutex::new(EncoderCacheManager::new(32));

        // Pre-populate only hash=2.
        {
            let mut guard = cache.lock().unwrap();
            guard.insert(
                CacheModality::Image,
                2,
                vec![Tensor::new(&[200.0f32], &Device::Cpu).unwrap()],
            );
        }

        let pixels = make_pixels(&[10.0, 20.0, 30.0]);
        let hashes = [1u64, 2, 3];

        let result = cached_encode_images(CacheModality::Image, &hashes, &pixels, &cache, |pv| {
            // Encoder doubles the value (so we can distinguish from raw pixels).
            Ok(vec![(pv * 2.0)?])
        })
        .unwrap();

        let output = result[0].to_vec2::<f32>().unwrap();
        // Image 0 (hash=1): miss, encoded = 10*2 = 20
        assert_eq!(output[0], vec![20.0]);
        // Image 1 (hash=2): hit, cached = 200
        assert_eq!(output[1], vec![200.0]);
        // Image 2 (hash=3): miss, encoded = 30*2 = 60
        assert_eq!(output[2], vec![60.0]);
    }

    #[test]
    fn test_cached_encode_deduplicates_uncached_images_within_batch() {
        let cache = Mutex::new(EncoderCacheManager::new(32));
        let pixels = make_pixels(&[10.0, 99.0, 20.0, 88.0]);
        let hashes = [1u64, 1, 2, 1];

        let result = cached_encode_images(CacheModality::Image, &hashes, &pixels, &cache, |pv| {
            assert_eq!(pv.dims(), &[2, 1]);
            assert_eq!(pv.to_vec2::<f32>()?, vec![vec![10.0], vec![20.0]]);
            Ok(vec![(pv * 2.0)?])
        })
        .unwrap();

        assert_eq!(
            result[0].to_vec2::<f32>().unwrap(),
            vec![vec![20.0], vec![20.0], vec![40.0], vec![20.0]]
        );
        let guard = cache.lock().unwrap();
        assert_eq!(guard.resident_entries(), 2);
    }

    #[test]
    fn test_cached_encode_reuses_duplicate_cached_images_within_batch() {
        let cache = Mutex::new(EncoderCacheManager::new(32));
        {
            let mut guard = cache.lock().unwrap();
            guard.insert(CacheModality::Image, 1, vec![dummy_tensor(7.0)]);
        }
        let pixels = make_pixels(&[10.0, 99.0, 88.0]);
        let hashes = [1u64, 1, 1];

        let result = cached_encode_images(CacheModality::Image, &hashes, &pixels, &cache, |_| {
            panic!("duplicate cached images should not call the encoder")
        })
        .unwrap();

        assert_eq!(
            result[0].to_vec2::<f32>().unwrap(),
            vec![vec![7.0], vec![7.0], vec![7.0]]
        );
    }

    #[test]
    fn test_cached_encode_multi_output() {
        let cache = Mutex::new(EncoderCacheManager::new(32));
        let pixels = make_pixels(&[5.0, 6.0]);
        let hashes = [10u64, 20];

        // Encoder returns two output tensors per image (e.g. main + deepstack).
        let result = cached_encode_images(CacheModality::Image, &hashes, &pixels, &cache, |pv| {
            let main = pv.clone();
            let aux = (pv * 10.0)?;
            Ok(vec![main, aux])
        })
        .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].to_vec2::<f32>().unwrap(),
            vec![vec![5.0], vec![6.0]]
        );
        assert_eq!(
            result[1].to_vec2::<f32>().unwrap(),
            vec![vec![50.0], vec![60.0]]
        );

        // Second call should be fully cached and return the same values.
        let result2 = cached_encode_images(CacheModality::Image, &hashes, &pixels, &cache, |_| {
            panic!("should not be called on full cache hit");
        })
        .unwrap();

        assert_eq!(
            result2[0].to_vec2::<f32>().unwrap(),
            vec![vec![5.0], vec![6.0]]
        );
        assert_eq!(
            result2[1].to_vec2::<f32>().unwrap(),
            vec![vec![50.0], vec![60.0]]
        );
    }
}
