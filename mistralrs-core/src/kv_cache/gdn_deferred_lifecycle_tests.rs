#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use super::*;

const SMALL_KEY_DIM: usize = 4;
const SMALL_VALUE_DIM: usize = 8;
const TEST_MAX_SEQUENCE: usize = 32;

fn mixed_cache(device: &Device, head_dim: usize) -> Result<HybridCache> {
    let config = HybridCacheConfig {
        layer_types: vec![
            HybridLayerType::Recurrent,
            HybridLayerType::Attention,
            HybridLayerType::Recurrent,
            HybridLayerType::Recurrent,
        ],
        max_seq_len: TEST_MAX_SEQUENCE,
        recurrent: RecurrentLayerConfig {
            conv_dim: 2,
            conv_width: 3,
            state: RecurrentStateSpec::Gdn {
                heads: 6,
                key_dim: head_dim,
                value_dim: head_dim,
            },
            recurrent_dtype: Some(DType::F32),
        },
    };
    let mut overrides = HashMap::new();
    overrides.insert(
        3,
        RecurrentLayerConfig {
            conv_dim: 1,
            conv_width: 1,
            state: RecurrentStateSpec::Opaque { dims: vec![2, 3] },
            recurrent_dtype: Some(DType::F32),
        },
    );
    if device.is_cpu() {
        overrides.insert(
            2,
            RecurrentLayerConfig {
                state: RecurrentStateSpec::Gdn {
                    heads: 3,
                    key_dim: SMALL_KEY_DIM,
                    value_dim: SMALL_VALUE_DIM,
                },
                ..config.recurrent.clone()
            },
        );
    }
    HybridCache::new_with_recurrent_overrides(
        config,
        DType::BF16,
        &vec![device.clone(); 4],
        &overrides,
    )
}

fn spec(key_heads: usize, tiled: bool) -> GdnDeferredStateSpec {
    GdnDeferredStateSpec {
        num_k_heads: key_heads,
        tiled_v_heads: tiled,
        depth: crate::cuda::gdn::GDN_DEFERRED_STATE_DEPTH,
    }
}

fn pool(cache: &HybridCache, layer: usize) -> &RecurrentStatePool {
    cache.get(layer).unwrap().as_recurrent_pool().unwrap()
}

#[test]
fn deferred_reservation_supports_mixed_pools_and_per_layer_specs() -> Result<()> {
    let mut cache = mixed_cache(&Device::Cpu, SMALL_KEY_DIM)?;
    let generation = cache.recurrent_storage_generation();
    let specs = [(0, spec(2, false)), (2, spec(1, true))];
    assert!(cache.reserve_gdn_deferred_state(&specs)?);
    assert_eq!(cache.recurrent_storage_generation(), generation + 1);
    assert!(!cache.reserve_gdn_deferred_state(&specs)?);
    assert!(cache.uses_gdn_deferred_state());
    assert!(pool(&cache, 3).deferred_state().is_none());
    assert_eq!(pool(&cache, 0).deferred_state().unwrap().spec(), specs[0].1);
    assert_eq!(pool(&cache, 2).deferred_state().unwrap().spec(), specs[1].1);
    assert_eq!(
        pool(&cache, 2).deferred_state().unwrap().head_v_dim,
        SMALL_VALUE_DIM
    );
    assert!(cache
        .reserve_gdn_deferred_state(&[(0, spec(2, true))])
        .is_err());
    assert!(cache.disable_gdn_deferred_state()?);
    assert_eq!(cache.recurrent_storage_generation(), generation + 2);
    assert!(!cache.uses_gdn_deferred_state());
    assert!(!cache.disable_gdn_deferred_state()?);
    assert!(cache.reserve_gdn_deferred_state(&[(0, spec(2, false))])?);
    assert!(pool(&cache, 2).deferred_state().is_none());
    assert!(cache.reserve_gdn_deferred_state(&specs)?);
    assert_eq!(pool(&cache, 2).deferred_state().unwrap().spec(), specs[1].1);
    Ok(())
}

#[test]
fn deferred_reservation_rejects_invalid_subsets_atomically() -> Result<()> {
    let mut cache = mixed_cache(&Device::Cpu, SMALL_KEY_DIM)?;
    let generation = cache.recurrent_storage_generation();
    let valid = (0, spec(2, false));
    let invalid_specs = [
        vec![valid, (3, spec(1, false))],
        vec![valid, (1, spec(1, false))],
        vec![valid, (4, spec(1, false))],
        vec![valid, valid],
        vec![valid, (2, spec(2, false))],
        vec![
            valid,
            (
                2,
                GdnDeferredStateSpec {
                    depth: 1,
                    ..spec(1, false)
                },
            ),
        ],
    ];
    for selected in invalid_specs {
        assert!(cache.reserve_gdn_deferred_state(&selected).is_err());
        assert!(!cache.uses_gdn_deferred_state());
        assert_eq!(cache.recurrent_storage_generation(), generation);
    }
    assert!(!cache.reserve_gdn_deferred_state(&[])?);
    cache.allocate_seq(0)?;
    assert!(cache.reserve_gdn_deferred_state(&[valid]).is_err());
    assert!(!cache.uses_gdn_deferred_state());
    Ok(())
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    const HEAD_DIM: usize = 128;
    const VALUE_HEADS: usize = 6;
    const KEY_HEADS: usize = 2;

    fn install_pending(pool: &RecurrentStatePool) -> Result<()> {
        let deferred = pool.deferred_state().unwrap();
        let key = (0..deferred.key.elem_count())
            .map(|idx| ((idx / HEAD_DIM) % KEY_HEADS + 1) as f32 * 0.25)
            .collect::<Vec<_>>();
        let delta = (0..deferred.delta.elem_count())
            .map(|idx| 1.0 + ((idx / HEAD_DIM) % VALUE_HEADS) as f32 * 0.125)
            .collect::<Vec<_>>();
        deferred.key.slice_set(
            &Tensor::from_vec(key, deferred.key.shape(), pool.device())?,
            0,
            0,
        )?;
        deferred.delta.slice_set(
            &Tensor::from_vec(delta, deferred.delta.shape(), pool.device())?,
            0,
            0,
        )?;
        deferred.decay.slice_set(
            &Tensor::full(0.5f32, deferred.decay.shape(), pool.device())?,
            0,
            0,
        )?;
        deferred.pending_rows.slice_set(
            &Tensor::ones(deferred.capacity, DType::U32, pool.device())?,
            0,
            0,
        )?;
        pool.recurrent_state.slice_set(
            &Tensor::ones(pool.recurrent_state.shape(), DType::F32, pool.device())?,
            0,
            0,
        )
    }

    fn assert_materialized(state: &Tensor, tiled: bool) -> Result<()> {
        assert_eq!(state.dtype(), DType::F32);
        let actual = state
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (idx, value) in actual.into_iter().enumerate() {
            let head = (idx / (HEAD_DIM * HEAD_DIM)) % VALUE_HEADS;
            let key_head = if tiled {
                head % KEY_HEADS
            } else {
                head / (VALUE_HEADS / KEY_HEADS)
            };
            let expected = 0.5 + (key_head + 1) as f32 * 0.25 * (1.0 + head as f32 * 0.125);
            assert_eq!(value, expected, "value at {idx}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn deferred_pool_gather_snapshot_and_slot_reuse_materialize_exactly() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let mut cache = mixed_cache(&device, HEAD_DIM)?;
        cache.reserve_gdn_deferred_state(&[
            (0, spec(KEY_HEADS, false)),
            (2, spec(KEY_HEADS, true)),
        ])?;
        for sequence_id in 0..INITIAL_POOL_CAPACITY {
            assert_eq!(cache.allocate_seq(sequence_id)?, sequence_id);
        }
        for layer in [0, 2] {
            install_pending(pool(&cache, layer))?;
        }
        let opaque_before = pool(&cache, 3).recurrent_state.copy()?;
        let snapshots = cache.snapshot_recurrent_state(1, 1)?;
        assert_materialized(&snapshots[0].recurrent_state, false)?;
        assert_materialized(&snapshots[1].recurrent_state, true)?;
        for layer in [0, 2] {
            let cursors = pool(&cache, layer)
                .deferred_state()
                .unwrap()
                .pending_rows
                .to_vec1::<u32>()?;
            assert_eq!(cursors[1], 0);
            assert_eq!(cursors[0], 1);
            assert_eq!(cursors[3], 1);
        }
        cache.materialize_deferred_state_for_slots(&[
            3,
            3,
            crate::cuda::gdn::GDN_PAD_SLOT,
            INITIAL_POOL_CAPACITY as u32,
        ])?;
        let active = Tensor::from_vec(vec![0u32, 2], 2, &device)?;
        cache.set_state_indices_with_host(Some(active), Some(vec![0, 2]))?;
        cache.materialize_deferred_state_for_current_batch()?;
        for layer in [0, 2] {
            let p = pool(&cache, layer);
            let cursors = p.deferred_state().unwrap().pending_rows.to_vec1::<u32>()?;
            assert_eq!(&cursors[..4], &[0, 0, 0, 0]);
            assert_eq!(cursors[4], 1);
            assert_materialized(&p.recurrent_state.narrow(0, 0, 4)?, layer == 2)?;
        }
        let checkpoint = cache.snapshot_recurrent_checkpoint_state(4)?;
        cache.restore_recurrent_checkpoint_state(4, &checkpoint)?;
        cache.restore_recurrent_state(1, 1, &snapshots)?;
        assert!(cache.release_seq(5, 5)?);
        let reused = cache.allocate_seq(99)?;
        assert_eq!(reused, 5);
        for layer in [0, 2] {
            let p = pool(&cache, layer);
            assert_eq!(
                p.deferred_state().unwrap().pending_rows.to_vec1::<u32>()?[reused],
                0
            );
            assert!(p
                .recurrent_state
                .narrow(0, reused, 1)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .all(|v| v == 0.0));
        }
        assert_eq!(
            pool(&cache, 3)
                .recurrent_state
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            opaque_before
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?
        );
        assert!(cache.reset().is_err());
        for sequence in 0..INITIAL_POOL_CAPACITY {
            if sequence != 5 {
                assert!(cache.release_seq(sequence, sequence)?);
            }
        }
        assert!(cache.release_seq(99, reused)?);
        for layer in [0, 2] {
            install_pending(pool(&cache, layer))?;
        }
        cache.reset()?;
        for layer in [0, 2] {
            assert!(pool(&cache, layer)
                .deferred_state()
                .unwrap()
                .pending_rows
                .to_vec1::<u32>()?
                .into_iter()
                .all(|v| v == 0));
        }
        assert!(cache.disable_gdn_deferred_state()?);
        Ok(())
    }
}
