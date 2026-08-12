//! Optional KV cache connector seam for external cache tiers.
//!
//! Default behavior is local-only prefix caching. A connector may observe
//! lookup/store/evict traffic or supply already-materialized local block IDs.

use super::block_hash::{BlockHash, BlockHashWithGroupId};

/// Hook into paged-attention prefix-cache block flow.
///
/// Implementations must keep default local behavior intact when they miss:
/// `lookup_blocks` returning `None` means "fall back to the in-process pool."
/// Returned block IDs, when `Some`, must already be valid IDs in the local
/// `BlockPool` (e.g. after the connector has hydrated external KV into them).
pub trait KvCacheConnector: Send + Sync {
    fn lookup_blocks(
        &self,
        _block_hashes: &[BlockHash],
        _group_ids: &[u32],
        _max_blocks: usize,
    ) -> Option<Vec<usize>> {
        None
    }

    fn observe_hit(
        &self,
        _block_hashes: &[BlockHash],
        _block_ids: &[usize],
        _num_computed_tokens: usize,
    ) {
    }

    fn observe_store(
        &self,
        _block_hashes: &[BlockHash],
        _block_ids: &[usize],
        _num_full_blocks: usize,
    ) {
    }

    fn observe_evict(&self, _block_hashes: &[BlockHashWithGroupId], _block_id: usize) {}
}

/// Default connector: always miss, never observe.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopKvCacheConnector;

impl KvCacheConnector for NoopKvCacheConnector {}
