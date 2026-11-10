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

struct PendingReservationOnDrop {
    reservation: Arc<crate::resource_plan::RuntimeMemoryReservation>,
    observed: Arc<AtomicUsize>,
}

impl Drop for PendingReservationOnDrop {
    fn drop(&mut self) {
        self.observed.store(
            self.reservation
                .pending_prefix_bytes(Device::Cpu.location())
                .unwrap(),
            Ordering::Relaxed,
        );
    }
}

impl PagedAuxiliaryPrefixState for PendingReservationOnDrop {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn bytes(&self) -> usize {
        0
    }
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

#[allow(clippy::cast_possible_truncation)]
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
fn paged_recurrent_reservation_is_published_before_checkpoint_release() -> anyhow::Result<()> {
    use crate::resource_plan::{RecurrentPrefixReservation, RuntimeMemoryReservation};
    use std::collections::HashMap;

    const SNAPSHOT_BYTES: usize = 8;
    for drop_cache in [false, true] {
        let device = Device::Cpu.location();
        let reservation = RuntimeMemoryReservation::standalone(RecurrentPrefixReservation::new(
            HashMap::from([(device, SNAPSHOT_BYTES)]),
            1,
        ));
        let observed = Arc::new(AtomicUsize::new(0));
        let hashes = block_hashes(10, 2);
        let mut cacher = PrefixCacheManagerV2::new(1, false, true);
        cacher.attach_runtime_memory_reservation(reservation.clone());
        cacher.add_paged_recurrent_prefix(
            *hashes.last().unwrap(),
            hashes,
            vec![make_recurrent_snapshot()?],
            Some(Arc::new(PendingReservationOnDrop {
                reservation: reservation.clone(),
                observed: observed.clone(),
            })),
        );
        assert_eq!(reservation.pending_prefix_bytes(device)?, 0);
        if drop_cache {
            drop(cacher);
        } else {
            cacher.evict_all_caches()?;
        }
        assert_eq!(observed.load(Ordering::Relaxed), SNAPSHOT_BYTES);
    }
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
fn sequence_cache_admission_excludes_paged_attention() {
    assert!(PrefixCacheManagerV2::new(1, false, false).accepts_sequence_cache());
    assert!(!PrefixCacheManagerV2::new(1, true, false).accepts_sequence_cache());
    assert!(!PrefixCacheManagerV2::new(1, false, true).accepts_sequence_cache());
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

    let partial =
        prefix_cacher.search_for_matching_cache(&[1, 2, 3, 4, 10], None, &[], None, None, None)?;
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
