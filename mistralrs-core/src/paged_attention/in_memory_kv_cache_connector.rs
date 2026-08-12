//! In-memory reference connector for the KV cache seam.
//!
//! This is an example "external" tier: it keeps a hash -> local block-id index
//! outside the block pool's own prefix map. It does not offload tensor bytes
//! (disk/S3 would still need a hydrate path before returning block IDs).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use super::block_hash::{BlockHash, BlockHashWithGroupId};
use super::kv_cache_connector::KvCacheConnector;

/// Process-local external KV index used as a reference `KvCacheConnector`.
#[derive(Debug, Default)]
pub struct InMemoryKvCacheConnector {
    blocks: Mutex<HashMap<BlockHash, usize>>,
    lookups: AtomicUsize,
    hits: AtomicUsize,
    stores: AtomicUsize,
    evicts: AtomicUsize,
}

impl InMemoryKvCacheConnector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.blocks.lock().expect("connector lock").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn lookup_count(&self) -> usize {
        self.lookups.load(Ordering::SeqCst)
    }

    pub fn hit_count(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }

    pub fn store_count(&self) -> usize {
        self.stores.load(Ordering::SeqCst)
    }

    pub fn evict_count(&self) -> usize {
        self.evicts.load(Ordering::SeqCst)
    }
}

impl KvCacheConnector for InMemoryKvCacheConnector {
    fn lookup_blocks(
        &self,
        block_hashes: &[BlockHash],
        _group_ids: &[u32],
        max_blocks: usize,
    ) -> Option<Vec<usize>> {
        self.lookups.fetch_add(1, Ordering::SeqCst);
        let guard = self.blocks.lock().expect("connector lock");
        let mut ids = Vec::new();
        for hash in block_hashes.iter().take(max_blocks) {
            match guard.get(hash) {
                Some(&id) => ids.push(id),
                None => break,
            }
        }
        if ids.is_empty() {
            None
        } else {
            Some(ids)
        }
    }

    fn observe_hit(
        &self,
        _block_hashes: &[BlockHash],
        _block_ids: &[usize],
        _num_computed_tokens: usize,
    ) {
        self.hits.fetch_add(1, Ordering::SeqCst);
    }

    fn observe_store(
        &self,
        block_hashes: &[BlockHash],
        block_ids: &[usize],
        _num_full_blocks: usize,
    ) {
        self.stores.fetch_add(1, Ordering::SeqCst);
        let mut guard = self.blocks.lock().expect("connector lock");
        for (hash, &block_id) in block_hashes.iter().zip(block_ids.iter()) {
            guard.insert(*hash, block_id);
        }
    }

    fn observe_evict(&self, block_hashes: &[BlockHashWithGroupId], block_id: usize) {
        self.evicts.fetch_add(1, Ordering::SeqCst);
        let mut guard = self.blocks.lock().expect("connector lock");
        for entry in block_hashes {
            guard.remove(&entry.block_hash);
        }
        guard.retain(|_, id| *id != block_id);
    }
}
