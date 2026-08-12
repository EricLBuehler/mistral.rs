//! Example external KV cache manager using the `KvCacheConnector` seam.
//!
//! This uses an in-memory connector as a stand-in for a future disk/S3 tier.
//! It indexes `BlockHash -> local block id` outside the block pool's own map.
//!
//! Run with:
//! `cargo run -p mistralrs --example kv_cache_connector`

use std::sync::Arc;

use anyhow::Result;
use mistralrs::{compute_block_hashes, InMemoryKvCacheConnector, KVCacheManager, KvCacheConnector};

fn main() -> Result<()> {
    let external = Arc::new(InMemoryKvCacheConnector::new());
    let mut mgr = KVCacheManager::with_connector(
        16,
        4,
        true,
        vec![0],
        Arc::clone(&external) as Arc<dyn KvCacheConnector>,
    );

    // Request A: compute and store two full blocks.
    let tokens_a: Vec<u32> = (1..=8).collect();
    let hashes_a = compute_block_hashes(&tokens_a, 4, &[], &[]);
    mgr.allocate_slots(1, 8, &[])
        .ok_or_else(|| anyhow::anyhow!("allocate request A"))?;
    mgr.cache_blocks(1, &hashes_a, 8);
    mgr.free(1);

    println!(
        "after store: external_entries={}, stores={}",
        external.len(),
        external.store_count()
    );

    // Request B: same prefix should hit through the external connector first.
    let tokens_b: Vec<u32> = (1..=12).collect();
    let hashes_b = compute_block_hashes(&tokens_b, 4, &[], &[]);
    let computed = mgr.get_computed_blocks(&hashes_b, 12);

    println!(
        "prefix hit: computed_tokens={}, block_ids={:?}, lookups={}, hits={}",
        computed.num_computed_tokens,
        computed.block_ids,
        external.lookup_count(),
        external.hit_count()
    );

    anyhow::ensure!(
        computed.num_computed_tokens == 8,
        "expected a 2-block prefix hit via the external connector"
    );
    anyhow::ensure!(external.hit_count() >= 1, "expected observe_hit");

    println!("ok: external in-memory KV connector served a prefix hit");
    Ok(())
}
