//! Block hashing for prefix caching.
//!
//! This module implements content-addressable hashing for KV cache blocks,
//! following vLLM's v1 approach. Each block is identified by a chain hash of:
//! - The hash of all previous blocks (parent hash)
//! - The tokens contained in the current block
//! - Optional extra keys (multimodal content hashes, LoRA names, cache salt)
//!
//! This creates a chain of hashes that uniquely identifies each block's position
//! and content in a sequence, enabling automatic prefix cache reuse.

use std::hash::{DefaultHasher, Hash, Hasher};

/// A hash that uniquely identifies a KV cache block by its content and position.
///
/// The hash incorporates the parent block's hash (chain hashing), the tokens in the
/// block, and any extra keys (e.g., multimodal content hashes).
///
/// Uses `u64` from `DefaultHasher` for speed. Can be upgraded to a cryptographic
/// hash (blake3, xxhash) later if collision resistance becomes important.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHash(u64);

impl BlockHash {
    /// Get the raw hash value.
    pub fn value(&self) -> u64 {
        self.0
    }
}

/// A block hash combined with its KV cache group ID.
///
/// In models with multiple KV cache types (e.g., full attention + sliding window),
/// the same token content in different groups maps to different physical blocks.
/// The group ID disambiguates them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockHashWithGroupId {
    pub block_hash: BlockHash,
    pub group_id: u32,
}

/// Extra keys that affect block hash computation beyond just token IDs.
#[derive(Debug, Clone, Hash)]
pub enum ExtraHashKey {
    /// Content hash of a multimodal input (image, audio, video).
    /// The identifier is a content-based hash of the raw media data.
    MultiModalHash(String),
    /// LoRA adapter name — different adapters produce different KV values.
    #[allow(dead_code)]
    LoraName(String),
    /// User-provided cache salt for per-request isolation.
    #[allow(dead_code)]
    CacheSalt(String),
}

/// Metadata about a multimodal feature's position in the token sequence.
/// Used to determine which blocks need multimodal extra hash keys.
#[derive(Debug, Clone)]
pub struct MultiModalFeature {
    /// Content hash of the media data (image pixels, audio samples, etc.).
    pub identifier: String,
    /// Token position where this feature's placeholder tokens start.
    pub offset: usize,
    /// Number of placeholder tokens this feature spans.
    pub length: usize,
}

/// The seed hash used as the parent hash for the first block in a sequence.
/// This is a fixed value (0) — consistent across all requests so that
/// identical first blocks always produce the same hash.
const NONE_HASH_SEED: u64 = 0;

/// Compute the hash for a single block given its parent hash, tokens, and extra keys.
///
/// The hash chain property ensures that two blocks with the same tokens but different
/// preceding contexts produce different hashes.
pub fn hash_block_tokens(
    parent_hash: Option<BlockHash>,
    block_tokens: &[u32],
    extra_keys: Option<&[ExtraHashKey]>,
) -> BlockHash {
    let mut hasher = DefaultHasher::new();

    // Chain: include parent hash (or seed for first block)
    match parent_hash {
        Some(parent) => parent.0.hash(&mut hasher),
        None => NONE_HASH_SEED.hash(&mut hasher),
    }

    // Hash the token content
    block_tokens.hash(&mut hasher);

    // Hash any extra keys (multimodal hashes, LoRA, salt)
    if let Some(keys) = extra_keys {
        for key in keys {
            key.hash(&mut hasher);
        }
    }

    BlockHash(hasher.finish())
}

/// Generate extra hash keys for a block based on multimodal feature overlap.
///
/// For each multimodal feature whose token range overlaps with the block's token range,
/// the feature's content identifier is included as an extra hash key. This ensures that
/// blocks containing different images/audio get different hashes, even if their placeholder
/// token IDs are identical.
///
/// Each feature is hashed individually (not as a set), so adding a new image at the end
/// of a conversation doesn't change the hashes of blocks containing earlier images.
/// Maximum number of multimodal extra hash keys per block. A pathological input
/// with many tiny multimodal features overlapping one block would otherwise produce
/// an unbounded key vector.
const MAX_MM_EXTRA_KEYS_PER_BLOCK: usize = 32;

pub fn generate_mm_extra_keys(
    block_start_token: usize,
    block_size: usize,
    mm_features: &[MultiModalFeature],
) -> Vec<ExtraHashKey> {
    let block_end_token = block_start_token + block_size;
    let mut extra_keys = Vec::new();

    for feature in mm_features {
        let feature_end = feature.offset + feature.length;
        // Check if this feature's token range overlaps with the block's range
        if feature.offset < block_end_token && feature_end > block_start_token {
            extra_keys.push(ExtraHashKey::MultiModalHash(feature.identifier.clone()));
            if extra_keys.len() >= MAX_MM_EXTRA_KEYS_PER_BLOCK {
                tracing::warn!(
                    "Block at token offset {block_start_token} has more than \
                     {MAX_MM_EXTRA_KEYS_PER_BLOCK} overlapping multimodal features; \
                     capping extra keys"
                );
                break;
            }
        }
    }

    extra_keys
}

/// Compute block hashes for all full blocks in a token sequence.
///
/// Returns a vector of `BlockHash` values, one per full block. Partial blocks
/// (the last block if the sequence length isn't a multiple of block_size) are not hashed
/// because they may still receive more tokens.
///
/// `mm_features`: multimodal features for extra key generation (pass empty slice if none).
/// `extra_keys_base`: additional extra keys applied to every block (e.g., LoRA name, cache salt).
pub fn compute_block_hashes(
    tokens: &[u32],
    block_size: usize,
    mm_features: &[MultiModalFeature],
    extra_keys_base: &[ExtraHashKey],
) -> Vec<BlockHash> {
    let num_full_blocks = tokens.len() / block_size;
    let mut hashes = Vec::with_capacity(num_full_blocks);
    let mut parent_hash = None;

    for block_idx in 0..num_full_blocks {
        let start = block_idx * block_size;
        let block_tokens = &tokens[start..start + block_size];

        // Collect extra keys: base keys + multimodal overlap keys
        let mut extra_keys = extra_keys_base.to_vec();
        let mm_keys = generate_mm_extra_keys(start, block_size, mm_features);
        extra_keys.extend(mm_keys);

        let extra = if extra_keys.is_empty() {
            None
        } else {
            Some(extra_keys.as_slice())
        };

        let hash = hash_block_tokens(parent_hash, block_tokens, extra);
        hashes.push(hash);
        parent_hash = Some(hash);
    }

    hashes
}

/// Incrementally compute block hashes for newly-full blocks.
///
/// Given existing hashes and the full token sequence, computes hashes only for
/// blocks that don't already have hashes. This is called after each decode step
/// when a new block becomes full.
pub fn compute_new_block_hashes(
    tokens: &[u32],
    block_size: usize,
    existing_hashes: &[BlockHash],
    mm_features: &[MultiModalFeature],
    extra_keys_base: &[ExtraHashKey],
) -> Vec<BlockHash> {
    let num_full_blocks = tokens.len() / block_size;
    if num_full_blocks <= existing_hashes.len() {
        return Vec::new();
    }

    let mut new_hashes = Vec::new();
    let parent_hash = existing_hashes.last().copied();
    let start_block = existing_hashes.len();

    let mut prev_hash = parent_hash;
    for block_idx in start_block..num_full_blocks {
        let start = block_idx * block_size;
        let block_tokens = &tokens[start..start + block_size];

        let mut extra_keys = extra_keys_base.to_vec();
        let mm_keys = generate_mm_extra_keys(start, block_size, mm_features);
        extra_keys.extend(mm_keys);

        let extra = if extra_keys.is_empty() {
            None
        } else {
            Some(extra_keys.as_slice())
        };

        let hash = hash_block_tokens(prev_hash, block_tokens, extra);
        new_hashes.push(hash);
        prev_hash = Some(hash);
    }

    new_hashes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_consistency() {
        let tokens = vec![1, 2, 3, 4];
        let h1 = hash_block_tokens(None, &tokens, None);
        let h2 = hash_block_tokens(None, &tokens, None);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_different_tokens_different_hash() {
        let h1 = hash_block_tokens(None, &[1, 2, 3, 4], None);
        let h2 = hash_block_tokens(None, &[1, 2, 3, 5], None);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_chain_hashing() {
        let h1 = hash_block_tokens(None, &[5, 6, 7, 8], None);
        let h2_with_parent = hash_block_tokens(Some(h1), &[9, 10, 11, 12], None);
        let h2_without_parent = hash_block_tokens(None, &[9, 10, 11, 12], None);
        // Same tokens but different parent should produce different hash
        assert_ne!(h2_with_parent, h2_without_parent);
    }

    #[test]
    fn test_extra_keys_affect_hash() {
        let tokens = vec![1, 2, 3, 4];
        let h1 = hash_block_tokens(None, &tokens, None);
        let extra = vec![ExtraHashKey::MultiModalHash("image_abc".to_string())];
        let h2 = hash_block_tokens(None, &tokens, Some(&extra));
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_different_mm_hashes_different_block_hash() {
        let tokens = vec![1, 2, 3, 4];
        let extra1 = vec![ExtraHashKey::MultiModalHash("image_1".to_string())];
        let extra2 = vec![ExtraHashKey::MultiModalHash("image_2".to_string())];
        let h1 = hash_block_tokens(None, &tokens, Some(&extra1));
        let h2 = hash_block_tokens(None, &tokens, Some(&extra2));
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_compute_block_hashes() {
        let tokens: Vec<u32> = (0..16).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);
        assert_eq!(hashes.len(), 4);

        // Verify chain property: recomputing should give same result
        let hashes2 = compute_block_hashes(&tokens, 4, &[], &[]);
        assert_eq!(hashes, hashes2);
    }

    #[test]
    fn test_compute_block_hashes_partial_block_ignored() {
        // 10 tokens with block_size=4 -> 2 full blocks, 2 leftover tokens
        let tokens: Vec<u32> = (0..10).collect();
        let hashes = compute_block_hashes(&tokens, 4, &[], &[]);
        assert_eq!(hashes.len(), 2);
    }

    #[test]
    fn test_incremental_hashing() {
        let tokens: Vec<u32> = (0..16).collect();
        let all_hashes = compute_block_hashes(&tokens, 4, &[], &[]);

        // Compute first 2 blocks, then incrementally add the rest
        let first_8: Vec<u32> = (0..8).collect();
        let initial = compute_block_hashes(&first_8, 4, &[], &[]);
        assert_eq!(initial.len(), 2);

        let new = compute_new_block_hashes(&tokens, 4, &initial, &[], &[]);
        assert_eq!(new.len(), 2);

        // Combined should match full computation
        let mut combined = initial;
        combined.extend(new);
        assert_eq!(combined, all_hashes);
    }

    #[test]
    fn test_mm_extra_keys_overlap() {
        let feature = MultiModalFeature {
            identifier: "img_hash_123".to_string(),
            offset: 2,
            length: 6,
        };

        // Block [0..4) overlaps with feature [2..8)
        let keys = generate_mm_extra_keys(0, 4, std::slice::from_ref(&feature));
        assert_eq!(keys.len(), 1);

        // Block [4..8) overlaps with feature [2..8)
        let keys = generate_mm_extra_keys(4, 4, std::slice::from_ref(&feature));
        assert_eq!(keys.len(), 1);

        // Block [8..12) does NOT overlap with feature [2..8)
        let keys = generate_mm_extra_keys(8, 4, &[feature]);
        assert_eq!(keys.len(), 0);
    }

    #[test]
    fn test_mm_extra_keys_multiple_features() {
        let features = vec![
            MultiModalFeature {
                identifier: "image_1".to_string(),
                offset: 0,
                length: 4,
            },
            MultiModalFeature {
                identifier: "image_2".to_string(),
                offset: 8,
                length: 4,
            },
        ];

        // Block [0..4) overlaps only with image_1
        let keys = generate_mm_extra_keys(0, 4, &features);
        assert_eq!(keys.len(), 1);

        // Block [4..8) overlaps with neither
        let keys = generate_mm_extra_keys(4, 4, &features);
        assert_eq!(keys.len(), 0);

        // Block [8..12) overlaps only with image_2
        let keys = generate_mm_extra_keys(8, 4, &features);
        assert_eq!(keys.len(), 1);
    }

    #[test]
    fn test_block_hash_with_group_id() {
        let hash = hash_block_tokens(None, &[1, 2, 3, 4], None);
        let g0 = BlockHashWithGroupId {
            block_hash: hash,
            group_id: 0,
        };
        let g1 = BlockHashWithGroupId {
            block_hash: hash,
            group_id: 1,
        };
        // Same hash, different group IDs should be different keys
        assert_ne!(g0, g1);
    }

    /// Validates the filter logic used by `Sequence::count_prefix_cached_mm_items`.
    /// A feature is only "fully cached" when `offset + length <= prefix_len`.
    #[test]
    fn test_mm_feature_fully_within_prefix() {
        let features = [
            MultiModalFeature {
                identifier: "img_a".to_string(),
                offset: 0,
                length: 4,
            },
            MultiModalFeature {
                identifier: "img_b".to_string(),
                offset: 6,
                length: 4, // ends at 10
            },
        ];

        let prefix_len = 8; // 2 blocks of size 4

        // Correct: offset + length <= prefix_len
        let fully_cached = features
            .iter()
            .filter(|f| f.offset + f.length <= prefix_len)
            .count();
        // img_a: 0+4=4 <= 8 → cached ✓
        // img_b: 6+4=10 > 8 → NOT cached ✓
        assert_eq!(fully_cached, 1);

        // Previously buggy: offset < prefix_len (would over-count)
        let buggy_count = features.iter().filter(|f| f.offset < prefix_len).count();
        // img_b offset=6 < 8 → would wrongly count as cached
        assert_eq!(buggy_count, 2);
        assert_ne!(
            fully_cached, buggy_count,
            "The correct filter should NOT match partially-overlapping features"
        );
    }
}
