//! Encoder output cache for multimodal models.
//!
//! Caches vision/audio encoder outputs keyed by content hash so that identical
//! media across requests (or after a prefix-cache partial hit) can skip the
//! expensive encoder pass.  Uses a simple LRU eviction strategy.

use std::sync::Mutex;

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
}

impl EncoderCacheManager {
    /// Create a new encoder cache with the given capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: IndexMap::with_capacity(max_entries),
            max_entries,
        }
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
            Some(cloned)
        } else {
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
