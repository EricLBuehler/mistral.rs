//! Encoder output cache for multimodal models.
//!
//! Caches vision/audio encoder outputs keyed by content hash so that identical
//! media across requests (or after a prefix-cache partial hit) can skip the
//! expensive encoder pass.  Uses a simple LRU eviction strategy.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use candle_core::Tensor;
use indexmap::IndexMap;

/// LRU cache for encoder outputs.
///
/// Each entry stores one or more tensors (e.g. Qwen3-VL returns both main
/// embeddings and deep-stack embeddings).  Keys are the same `u64` content
/// hashes already computed for images/audio in [`crate::sequence::Sequence`].
///
/// The cache is typically stored behind `Arc<Mutex<…>>` on each model struct
/// and accessed from `forward()` via interior mutability.
pub struct EncoderCacheManager {
    /// Insertion-ordered map; most-recently-used entries live at the back.
    cache: IndexMap<u64, Vec<Tensor>>,
    max_entries: usize,
    hits: Arc<AtomicUsize>,
    misses: Arc<AtomicUsize>,
}

impl EncoderCacheManager {
    /// Create a new encoder cache with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: IndexMap::with_capacity(max_entries),
            max_entries,
            hits: Arc::new(AtomicUsize::new(0)),
            misses: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Return clones of the hit/miss counter Arcs (hits, misses).
    pub fn counters(&self) -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        (self.hits.clone(), self.misses.clone())
    }

    /// Look up a cached encoder output by content hash.
    ///
    /// On hit the entry is moved to the back (most-recently-used position)
    /// and the tensors are cloned (cheap — Candle tensors are `Arc`-backed).
    pub fn get(&mut self, content_hash: u64) -> Option<Vec<Tensor>> {
        // `shift_remove` + re-insert moves the entry to the back.
        if let Some(entry) = self.cache.shift_remove(&content_hash) {
            let cloned = entry.clone();
            self.cache.insert(content_hash, entry);
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
    pub fn insert(&mut self, content_hash: u64, outputs: Vec<Tensor>) {
        if self.cache.contains_key(&content_hash) {
            // Already cached (race between concurrent callers); just bump LRU.
            self.cache.shift_remove(&content_hash);
            self.cache.insert(content_hash, outputs);
            return;
        }
        if self.cache.len() >= self.max_entries && self.max_entries > 0 {
            // Evict the oldest (front) entry.
            self.cache.shift_remove_index(0);
        }
        self.cache.insert(content_hash, outputs);
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
    let mut hits: Vec<Option<Vec<Tensor>>> = vec![None; n_images];
    let mut miss_indices: Vec<usize> = Vec::new();
    {
        let mut guard = cache.lock().expect("encoder cache lock poisoned");
        for (i, &hash) in image_hashes.iter().enumerate() {
            if let Some(cached) = guard.get(hash) {
                hits[i] = Some(cached);
            } else {
                miss_indices.push(i);
            }
        }
    }

    // Fast path – all cached.
    if miss_indices.is_empty() {
        return assemble(hits, n_images);
    }

    // Phase 2 – encode only the misses.
    let miss_pixels = if miss_indices.len() == n_images {
        // All misses – encode full batch without splitting.
        pixel_values.clone()
    } else {
        let slices: Vec<Tensor> = miss_indices
            .iter()
            .map(|&i| pixel_values.get(i))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Tensor::stack(&slices, 0)?
    };

    let encoded = encode_fn(&miss_pixels)?;

    // Phase 3 – store per-image results in cache and fill `hits`.
    {
        let mut guard = cache.lock().expect("encoder cache lock poisoned");
        for (batch_idx, &orig_idx) in miss_indices.iter().enumerate() {
            let per_image: Vec<Tensor> = encoded
                .iter()
                .map(|t| t.get(batch_idx))
                .collect::<candle_core::Result<Vec<_>>>()?;
            guard.insert(image_hashes[orig_idx], per_image.clone());
            hits[orig_idx] = Some(per_image);
        }
    }

    assemble(hits, n_images)
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
    use candle_core::{Device, Tensor};

    fn dummy_tensor(val: f32) -> Tensor {
        Tensor::new(&[val], &Device::Cpu).unwrap()
    }

    // -----------------------------------------------------------------------
    // EncoderCacheManager unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_insert_and_get() {
        let mut cache = EncoderCacheManager::new(4);
        let t = dummy_tensor(1.0);
        cache.insert(100, vec![t.clone()]);

        let result = cache.get(100);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(
            result[0].to_vec1::<f32>().unwrap(),
            t.to_vec1::<f32>().unwrap()
        );
    }

    #[test]
    fn test_get_miss() {
        let mut cache = EncoderCacheManager::new(4);
        assert!(cache.get(999).is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = EncoderCacheManager::new(3);
        cache.insert(1, vec![dummy_tensor(1.0)]);
        cache.insert(2, vec![dummy_tensor(2.0)]);
        cache.insert(3, vec![dummy_tensor(3.0)]);

        // Cache is full. Inserting a 4th should evict key=1 (oldest).
        cache.insert(4, vec![dummy_tensor(4.0)]);

        assert!(cache.get(1).is_none(), "key 1 should have been evicted");
        assert!(cache.get(2).is_some());
        assert!(cache.get(3).is_some());
        assert!(cache.get(4).is_some());
    }

    #[test]
    fn test_get_bumps_lru_order() {
        let mut cache = EncoderCacheManager::new(3);
        cache.insert(1, vec![dummy_tensor(1.0)]);
        cache.insert(2, vec![dummy_tensor(2.0)]);
        cache.insert(3, vec![dummy_tensor(3.0)]);

        // Access key=1 to bump it to most-recently-used.
        let _ = cache.get(1);

        // Now key=2 is the oldest. Inserting key=4 should evict key=2.
        cache.insert(4, vec![dummy_tensor(4.0)]);

        assert!(cache.get(1).is_some(), "key 1 was accessed, should survive");
        assert!(cache.get(2).is_none(), "key 2 should have been evicted");
        assert!(cache.get(3).is_some());
        assert!(cache.get(4).is_some());
    }

    #[test]
    fn test_insert_duplicate_updates_lru() {
        let mut cache = EncoderCacheManager::new(3);
        cache.insert(1, vec![dummy_tensor(1.0)]);
        cache.insert(2, vec![dummy_tensor(2.0)]);
        cache.insert(3, vec![dummy_tensor(3.0)]);

        // Re-insert key=1 with new data — should bump it, not create duplicate.
        cache.insert(1, vec![dummy_tensor(10.0)]);

        // key=2 is now oldest.
        cache.insert(4, vec![dummy_tensor(4.0)]);

        assert!(
            cache.get(1).is_some(),
            "key 1 was re-inserted, should survive"
        );
        assert!(cache.get(2).is_none(), "key 2 should have been evicted");

        // Verify the value was updated.
        let val = cache.get(1).unwrap()[0].to_vec1::<f32>().unwrap();
        assert_eq!(val, vec![10.0]);
    }

    #[test]
    fn test_multi_tensor_entries() {
        let mut cache = EncoderCacheManager::new(4);
        let t1 = dummy_tensor(1.0);
        let t2 = dummy_tensor(2.0);
        cache.insert(42, vec![t1, t2]);

        let result = cache.get(42).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].to_vec1::<f32>().unwrap(), vec![1.0]);
        assert_eq!(result[1].to_vec1::<f32>().unwrap(), vec![2.0]);
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

        let result = cached_encode_images(&hashes, &pixels, &cache, |pv| {
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
        assert!(guard.get(1).is_some());
        assert!(guard.get(2).is_some());
        assert!(guard.get(3).is_some());
    }

    #[test]
    fn test_cached_encode_all_hit() {
        let cache = Mutex::new(EncoderCacheManager::new(32));

        // Pre-populate cache.
        {
            let mut guard = cache.lock().unwrap();
            guard.insert(1, vec![Tensor::new(&[100.0f32], &Device::Cpu).unwrap()]);
            guard.insert(2, vec![Tensor::new(&[200.0f32], &Device::Cpu).unwrap()]);
        }

        let pixels = make_pixels(&[10.0, 20.0]);
        let hashes = [1u64, 2];

        let encode_called = std::sync::atomic::AtomicBool::new(false);
        let result = cached_encode_images(&hashes, &pixels, &cache, |pv| {
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
            guard.insert(2, vec![Tensor::new(&[200.0f32], &Device::Cpu).unwrap()]);
        }

        let pixels = make_pixels(&[10.0, 20.0, 30.0]);
        let hashes = [1u64, 2, 3];

        let result = cached_encode_images(&hashes, &pixels, &cache, |pv| {
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
    fn test_cached_encode_multi_output() {
        let cache = Mutex::new(EncoderCacheManager::new(32));
        let pixels = make_pixels(&[5.0, 6.0]);
        let hashes = [10u64, 20];

        // Encoder returns two output tensors per image (e.g. main + deepstack).
        let result = cached_encode_images(&hashes, &pixels, &cache, |pv| {
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
        let result2 = cached_encode_images(&hashes, &pixels, &cache, |_| {
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
