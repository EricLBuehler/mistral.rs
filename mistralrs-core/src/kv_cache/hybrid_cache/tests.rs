use std::collections::HashSet;

use super::*;

fn config(layer_types: Vec<HybridLayerType>) -> HybridCacheConfig {
    HybridCacheConfig {
        layer_types,
        max_seq_len: 32,
        recurrent: RecurrentLayerConfig {
            conv_dim: 2,
            conv_width: 3,
            state: RecurrentStateSpec::Opaque { dims: vec![2, 2] },
            recurrent_dtype: None,
        },
    }
}

fn gdn_config(layer_types: Vec<HybridLayerType>) -> HybridCacheConfig {
    HybridCacheConfig {
        layer_types,
        max_seq_len: 32,
        recurrent: RecurrentLayerConfig {
            conv_dim: 2,
            conv_width: 3,
            state: RecurrentStateSpec::Gdn {
                heads: 3,
                key_dim: 4,
                value_dim: 5,
            },
            recurrent_dtype: Some(DType::F32),
        },
    }
}

#[test]
fn requires_one_device_per_layer() {
    let error = HybridCache::new(
        config(vec![HybridLayerType::Attention, HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )
    .unwrap_err();
    assert!(error.to_string().contains("1 layer devices"));
}

#[test]
fn recurrent_snapshot_bytes_use_one_physical_row_per_layer() -> Result<()> {
    let cache = HybridCache::new(
        config(vec![
            HybridLayerType::Recurrent,
            HybridLayerType::Attention,
            HybridLayerType::Recurrent,
        ]),
        DType::F32,
        &[Device::Cpu, Device::Cpu, Device::Cpu],
    )?;

    let bytes = cache.recurrent_snapshot_bytes_by_device()?;
    assert_eq!(bytes.len(), 1);
    assert_eq!(bytes[&DeviceLocation::Cpu], 80);
    Ok(())
}

#[test]
fn recurrent_indices_are_local_to_the_layer_pool() -> Result<()> {
    let devices = vec![Device::Cpu, Device::Cpu, Device::Cpu];
    let mut cache = HybridCache::new(
        config(vec![
            HybridLayerType::Recurrent,
            HybridLayerType::Attention,
            HybridLayerType::Recurrent,
        ]),
        DType::F32,
        &devices,
    )?;
    let state_indices = Tensor::from_vec(vec![1u32, 3], (2,), &Device::Cpu)?;
    cache.set_state_indices_with_host(Some(state_indices), Some(vec![1, 3]))?;

    assert!(cache.state_indices_for_layer(1)?.is_none());
    for layer in [0, 2] {
        let indices = cache.state_indices_for_layer(layer)?.unwrap();
        let HybridLayerCache::Recurrent(pool) = cache.get(layer).unwrap() else {
            unreachable!()
        };
        assert!(indices.device().same_device(pool.device()));
        assert_eq!(indices.to_vec1::<u32>()?, vec![1, 3]);
    }
    Ok(())
}

#[test]
fn recurrent_capacity_can_be_reserved_before_admission() -> Result<()> {
    let devices = vec![Device::Cpu, Device::Cpu];
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
        DType::F32,
        &devices,
    )?;

    assert!(cache.reserve_recurrent_capacity(17)?);
    assert_eq!(cache.recurrent_capacity(), 17);
    assert!(!cache.reserve_recurrent_capacity(16)?);

    let slots = (0..17)
        .map(|sequence_id| cache.allocate_seq(sequence_id).unwrap())
        .collect::<HashSet<_>>();
    assert_eq!(slots.len(), 17);
    assert_eq!(cache.recurrent_capacity(), 17);
    Ok(())
}

#[test]
fn recurrent_slot_accounting_tracks_all_lifecycle_transitions() -> Result<()> {
    let devices = vec![Device::Cpu, Device::Cpu];
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
        DType::F32,
        &devices,
    )?;

    assert_eq!(cache.recurrent_capacity(), INITIAL_POOL_CAPACITY);
    assert_eq!(cache.recurrent_slots_used(), 0);

    let first = cache.allocate_seq(10)?;
    let second = cache.allocate_seq(20)?;
    assert_eq!(cache.recurrent_slots_used(), 2);

    assert!(cache.release_seq(10, first)?);
    assert_eq!(cache.recurrent_slots_used(), 1);
    assert!(cache.reserve_recurrent_capacity(17)?);
    assert_eq!(cache.recurrent_capacity(), 17);
    assert_eq!(cache.recurrent_slots_used(), 1);

    assert!(cache.graph_pad_slot()?.is_some());
    assert_eq!(cache.recurrent_slots_used(), 2);
    assert!(cache.release_seq(20, second)?);
    assert_eq!(cache.recurrent_slots_used(), 1);

    assert!(cache.release_graph_pad_slot()?);
    cache.reset()?;
    assert_eq!(cache.recurrent_capacity(), 17);
    assert_eq!(cache.recurrent_slots_used(), 0);
    Ok(())
}

#[test]
fn failed_slot_reset_rolls_back_every_recurrent_layer() -> Result<()> {
    let devices = vec![Device::Cpu, Device::Cpu];
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
        DType::F32,
        &devices,
    )?;
    let dirty_slot = cache.allocate_seq(1)?;
    assert_eq!(dirty_slot, 0);
    assert!(cache.release_seq(1, dirty_slot)?);
    let original_conv_state = {
        let HybridLayerCache::Recurrent(pool) = cache.get_mut(1).unwrap() else {
            unreachable!()
        };
        let original = pool.conv_state.clone();
        pool.conv_state = Tensor::zeros((INITIAL_POOL_CAPACITY, 1, 1), DType::F32, &Device::Cpu)?;
        original
    };

    let error = cache.allocate_seq(10).unwrap_err();
    assert!(error
        .to_string()
        .contains("failed to reset recurrent state slot"));
    assert_eq!(cache.recurrent_slots_used(), 0);
    for layer in [0, 1] {
        let HybridLayerCache::Recurrent(pool) = cache.get(layer).unwrap() else {
            unreachable!()
        };
        assert_eq!(pool.num_free_slots(), pool.capacity());
        assert!(!pool.is_allocated(0));
    }

    let HybridLayerCache::Recurrent(pool) = cache.get_mut(1).unwrap() else {
        unreachable!()
    };
    pool.conv_state = original_conv_state;
    assert_eq!(cache.allocate_seq(10)?, 0);
    assert_eq!(cache.recurrent_slots_used(), 1);
    Ok(())
}

#[test]
fn pristine_slot_provenance_tracks_reservation_growth_and_reset() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    assert_eq!(cache.pristine_zero_slots, vec![true; INITIAL_POOL_CAPACITY]);

    let slot = cache.reserve_seq_uninitialized(10)?;
    assert_eq!(slot, 0);
    assert!(cache.pristine_zero_slots[slot]);
    assert!(!cache.initialized_slots[slot]);
    assert!(cache.release_seq(10, slot)?);
    assert!(cache.pristine_zero_slots[slot]);

    assert_eq!(cache.allocate_seq(20)?, slot);
    assert!(!cache.pristine_zero_slots[slot]);
    assert!(cache.initialized_slots[slot]);

    let deferred_slot = cache.reserve_seq_uninitialized(30)?;
    assert_eq!(deferred_slot, 1);
    assert!(cache.pristine_zero_slots[deferred_slot]);
    cache.reset_seq(30, deferred_slot)?;
    assert!(!cache.pristine_zero_slots[deferred_slot]);
    assert!(cache.initialized_slots[deferred_slot]);

    assert!(cache.release_seq(20, slot)?);
    assert!(cache.release_seq(30, deferred_slot)?);

    assert!(cache.reserve_recurrent_capacity(17)?);
    assert!(!cache.pristine_zero_slots[slot]);
    assert!(!cache.pristine_zero_slots[deferred_slot]);
    assert!(cache.pristine_zero_slots[2..]
        .iter()
        .all(|&pristine| pristine));

    cache.reset()?;
    assert_eq!(cache.pristine_zero_slots, vec![true; 17]);
    Ok(())
}

#[test]
fn duplicate_free_cannot_duplicate_or_alias_a_slot() -> Result<()> {
    let devices = vec![Device::Cpu, Device::Cpu];
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
        DType::F32,
        &devices,
    )?;

    let slot = cache.allocate_seq(10)?;
    assert!(cache.release_seq(10, slot)?);
    assert!(!cache.release_seq(10, slot)?);
    assert_eq!(cache.recurrent_slots_used(), 0);

    let allocated = (0..cache.recurrent_capacity())
        .map(|sequence_id| cache.allocate_seq(sequence_id + 100).unwrap())
        .collect::<HashSet<_>>();
    assert_eq!(allocated.len(), cache.recurrent_capacity());
    assert!(allocated.contains(&slot));
    Ok(())
}

#[test]
fn checkpoint_lanes_expand_physical_storage_before_reservation() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;

    assert!(cache.configure_checkpoint_lanes(3)?);
    assert!(!cache.configure_checkpoint_lanes(3)?);
    assert_eq!(cache.checkpoint_lanes(), 3);
    assert_eq!(cache.recurrent_capacity(), INITIAL_POOL_CAPACITY);
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(pool.checkpoint_lanes(), 3);
    assert_eq!(pool.physical_capacity(), INITIAL_POOL_CAPACITY * 3);
    assert_eq!(pool.conv_state.dims(), &[27, 2, 3]);
    assert_eq!(pool.recurrent_state.dims(), &[27, 2, 2]);

    assert!(cache.reserve_recurrent_capacity(17)?);
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(pool.conv_state.dims(), &[51, 2, 3]);
    assert_eq!(pool.recurrent_state.dims(), &[51, 2, 2]);
    assert!(cache.configure_checkpoint_lanes(4).is_err());
    Ok(())
}

#[test]
fn transition_lanes_switch_storage_without_changing_logical_depth() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;

    assert!(cache.configure_checkpoint_lanes(8)?);
    assert_eq!(cache.checkpoint_lanes(), 8);
    assert_eq!(cache.physical_checkpoint_lanes(), 8);
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(pool.physical_capacity(), INITIAL_POOL_CAPACITY * 8);

    assert!(cache.configure_transition_lanes(8)?);
    assert!(!cache.configure_transition_lanes(8)?);
    assert_eq!(cache.checkpoint_lanes(), 8);
    assert_eq!(cache.physical_checkpoint_lanes(), 1);
    assert!(cache.uses_recurrent_transition_log());
    assert_eq!(
        cache.recurrent_speculative_storage(),
        RecurrentSpeculativeStorage::TransitionLog
    );
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(pool.checkpoint_lanes(), 1);
    assert_eq!(pool.physical_capacity(), INITIAL_POOL_CAPACITY);
    assert_eq!(pool.conv_state.dims(), &[INITIAL_POOL_CAPACITY, 2, 3]);
    assert_eq!(pool.recurrent_state.dims(), &[INITIAL_POOL_CAPACITY, 2, 2]);

    assert!(cache.configure_checkpoint_lanes(8)?);
    assert_eq!(cache.physical_checkpoint_lanes(), 8);
    assert!(!cache.uses_recurrent_transition_log());
    Ok(())
}

#[test]
fn transition_layout_growth_keeps_one_physical_row_per_sequence() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;

    assert!(cache.reserve_recurrent_transition_layout(17, 8)?);
    assert_eq!(cache.recurrent_capacity(), 17);
    assert_eq!(cache.checkpoint_lanes(), 8);
    assert_eq!(cache.physical_checkpoint_lanes(), 1);
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(pool.capacity(), 17);
    assert_eq!(pool.physical_capacity(), 17);
    assert_eq!(pool.conv_state.dims(), &[17, 2, 3]);
    assert_eq!(pool.recurrent_state.dims(), &[17, 2, 2]);

    for sequence_id in 0..17 {
        assert_eq!(cache.allocate_seq(sequence_id)?, sequence_id);
    }
    assert_eq!(cache.allocate_seq(17)?, 17);
    assert_eq!(cache.recurrent_capacity(), 34);
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(pool.checkpoint_lanes(), 1);
    assert_eq!(pool.physical_capacity(), 34);
    assert_eq!(pool.conv_state.dims(), &[34, 2, 3]);
    assert_eq!(pool.recurrent_state.dims(), &[34, 2, 2]);
    Ok(())
}

#[test]
fn pending_transition_storage_tracks_capacity_and_slot_lifecycle() -> Result<()> {
    let mut cache = HybridCache::new(
        gdn_config(vec![
            HybridLayerType::Recurrent,
            HybridLayerType::Attention,
            HybridLayerType::Recurrent,
        ]),
        DType::BF16,
        &[Device::Cpu, Device::Cpu, Device::Cpu],
    )?;
    cache.configure_transition_lanes(8)?;
    let spec = GdnPendingTransitionSpec {
        num_k_heads: 1,
        max_rows: 8,
    };
    assert!(cache.reserve_gdn_pending_transitions(spec)?);
    assert!(!cache.reserve_gdn_pending_transitions(spec)?);

    for layer_idx in [0, 2] {
        let HybridLayerCache::Recurrent(pool) = cache.get(layer_idx).unwrap() else {
            unreachable!()
        };
        let pending = pool.pending_transitions().unwrap();
        assert_eq!(pending.capacity(), INITIAL_POOL_CAPACITY);
        assert_eq!(pending.conv_input.dims(), &[9, 8, 2]);
        assert_eq!(pending.key_banks.dims(), &[2, 9, 8, 1, 4]);
        assert_eq!(pending.delta.dims(), &[9, 8, 3, 5]);
        assert_eq!(pending.decay.dims(), &[9, 8, 3]);
        assert_eq!(pending.keep_rows.dims(), &[9]);
        assert_eq!(pending.pending_epochs.dims(), &[9]);
        assert_eq!(pending.key_bank.dims(), &[9]);
        assert_eq!(pending.conv_applied_epochs.dims(), &[9, 1]);
        assert_eq!(pending.recurrent_applied_epochs.dims(), &[9, 3]);
        assert!(pending.storage_bytes()? > 0);
    }

    let slots = (0..INITIAL_POOL_CAPACITY)
        .map(|sequence_id| cache.allocate_seq(sequence_id))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(cache.recurrent_slots_for_sequences(&[2, 0]), vec![0, 2]);
    for layer_idx in [0, 2] {
        let HybridLayerCache::Recurrent(pool) = cache.get_mut(layer_idx).unwrap() else {
            unreachable!()
        };
        let pending = pool.pending_transitions.as_mut().unwrap();
        pending
            .keep_rows
            .slice_set(&Tensor::from_vec(vec![6u32; 9], (9,), &Device::Cpu)?, 0, 0)?;
        pending.pending_epochs.slice_set(
            &Tensor::from_vec(vec![17u32; 9], (9,), &Device::Cpu)?,
            0,
            0,
        )?;
        pending
            .key_bank
            .slice_set(&Tensor::from_vec(vec![1u32; 9], (9,), &Device::Cpu)?, 0, 0)?;
        pending.conv_applied_epochs.slice_set(
            &Tensor::from_vec(vec![23u32; 9], (9, 1), &Device::Cpu)?,
            0,
            0,
        )?;
        pending.recurrent_applied_epochs.slice_set(
            &Tensor::from_vec(vec![29u32; 27], (9, 3), &Device::Cpu)?,
            0,
            0,
        )?;
    }

    assert_eq!(cache.allocate_seq(INITIAL_POOL_CAPACITY)?, 9);
    assert_eq!(cache.recurrent_capacity(), 18);
    for layer_idx in [0, 2] {
        let HybridLayerCache::Recurrent(pool) = cache.get(layer_idx).unwrap() else {
            unreachable!()
        };
        let pending = pool.pending_transitions().unwrap();
        assert_eq!(pending.capacity(), 18);
        assert_eq!(pending.conv_input.dims(), &[18, 8, 2]);
        assert_eq!(pending.keep_rows.to_vec1::<u32>()?[0], 6);
        assert_eq!(pending.pending_epochs.to_vec1::<u32>()?[0], 17);
        assert_eq!(pending.key_bank.to_vec1::<u32>()?[0], 1);
        assert_eq!(
            pending
                .conv_applied_epochs
                .flatten_all()?
                .to_vec1::<u32>()?[0],
            23
        );
        assert_eq!(
            pending
                .recurrent_applied_epochs
                .flatten_all()?
                .to_vec1::<u32>()?[0],
            29
        );
        assert_eq!(pending.keep_rows.to_vec1::<u32>()?[9], 0);
        assert_eq!(pending.pending_epochs.to_vec1::<u32>()?[9], 0);
        assert_eq!(pending.key_bank.to_vec1::<u32>()?[9], 0);
        assert_eq!(
            pending
                .conv_applied_epochs
                .narrow(0, 9, 1)?
                .flatten_all()?
                .to_vec1::<u32>()?,
            vec![0]
        );
        assert_eq!(
            pending
                .recurrent_applied_epochs
                .narrow(0, 9, 1)?
                .flatten_all()?
                .to_vec1::<u32>()?,
            vec![0; 3]
        );
    }

    assert!(cache.release_seq(0, slots[0])?);
    for layer_idx in [0, 2] {
        let HybridLayerCache::Recurrent(pool) = cache.get(layer_idx).unwrap() else {
            unreachable!()
        };
        let pending = pool.pending_transitions().unwrap();
        assert_eq!(pending.keep_rows.to_vec1::<u32>()?[0], 0);
        assert_eq!(pending.pending_epochs.to_vec1::<u32>()?[0], 0);
        assert_eq!(pending.key_bank.to_vec1::<u32>()?[0], 0);
        assert_eq!(
            pending
                .conv_applied_epochs
                .narrow(0, 0, 1)?
                .flatten_all()?
                .to_vec1::<u32>()?,
            vec![0]
        );
        assert_eq!(
            pending
                .recurrent_applied_epochs
                .narrow(0, 0, 1)?
                .flatten_all()?
                .to_vec1::<u32>()?,
            vec![0; 3]
        );
    }
    Ok(())
}

#[test]
fn deferred_state_storage_tracks_capacity_and_slot_lifecycle() -> Result<()> {
    let mut cache = HybridCache::new(
        gdn_config(vec![
            HybridLayerType::Recurrent,
            HybridLayerType::Attention,
            HybridLayerType::Recurrent,
        ]),
        DType::BF16,
        &[Device::Cpu, Device::Cpu, Device::Cpu],
    )?;
    let depth = crate::cuda::gdn::GDN_DEFERRED_STATE_DEPTH;
    assert!(cache
        .reserve_gdn_deferred_state(&[(
            0,
            GdnDeferredStateSpec {
                num_k_heads: 1,
                tiled_v_heads: false,
                depth: depth - 1,
            }
        )])
        .is_err());
    let spec = GdnDeferredStateSpec {
        num_k_heads: 1,
        tiled_v_heads: false,
        depth,
    };
    assert!(cache.reserve_gdn_deferred_state(&[(0, spec), (2, spec)])?);
    assert!(!cache.reserve_gdn_deferred_state(&[(0, spec), (2, spec)])?);
    assert!(cache.uses_gdn_deferred_state());

    for layer_idx in [0, 2] {
        let HybridLayerCache::Recurrent(pool) = cache.get(layer_idx).unwrap() else {
            unreachable!()
        };
        let deferred = pool.deferred_state().unwrap();
        assert_eq!(deferred.capacity, INITIAL_POOL_CAPACITY);
        assert_eq!(deferred.key.dims(), &[9, depth, 1, 4]);
        assert_eq!(deferred.delta.dims(), &[9, depth, 3, 5]);
        assert_eq!(deferred.decay.dims(), &[9, depth, 3]);
        assert_eq!(deferred.pending_rows.dims(), &[9]);
        assert!(deferred.storage_bytes()? > 0);
    }

    let slots = (0..INITIAL_POOL_CAPACITY)
        .map(|sequence_id| cache.allocate_seq(sequence_id))
        .collect::<Result<Vec<_>>>()?;
    for layer_idx in [0, 2] {
        let HybridLayerCache::Recurrent(pool) = cache.get_mut(layer_idx).unwrap() else {
            unreachable!()
        };
        let deferred = pool.deferred_state.as_mut().unwrap();
        deferred.key = Tensor::ones(deferred.key.shape().clone(), DType::F32, &Device::Cpu)?;
        deferred.delta = Tensor::ones(deferred.delta.shape().clone(), DType::F32, &Device::Cpu)?;
        deferred.decay = Tensor::ones(deferred.decay.shape().clone(), DType::F32, &Device::Cpu)?;
        deferred.pending_rows.slice_set(
            &Tensor::from_vec(
                vec![depth as u32; INITIAL_POOL_CAPACITY],
                (INITIAL_POOL_CAPACITY,),
                &Device::Cpu,
            )?,
            0,
            0,
        )?;
    }

    assert_eq!(cache.allocate_seq(INITIAL_POOL_CAPACITY)?, 9);
    assert_eq!(cache.recurrent_capacity(), 18);
    for layer_idx in [0, 2] {
        let HybridLayerCache::Recurrent(pool) = cache.get(layer_idx).unwrap() else {
            unreachable!()
        };
        let deferred = pool.deferred_state().unwrap();
        assert_eq!(deferred.capacity, 18);
        assert_eq!(deferred.pending_rows.to_vec1::<u32>()?[0], depth as u32);
        assert_eq!(deferred.pending_rows.to_vec1::<u32>()?[9], 0);
        assert_eq!(
            deferred.key.i(0)?.flatten_all()?.to_vec1::<f32>()?,
            vec![1.0; depth * 4]
        );
    }

    // This CPU fixture checks journal storage; CUDA tests cover materialization.
    let checkpoint = RecurrentCheckpointStateSnapshot {
        states: cache
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .map(|pool| {
                Ok(RecurrentStateSnapshot {
                    conv_state: pool.conv_state.narrow(0, slots[1], 1)?.copy()?,
                    recurrent_state: pool.recurrent_state.narrow(0, slots[1], 1)?.copy()?,
                    state_layout: pool.state_layout(),
                })
            })
            .collect::<Result<Vec<_>>>()?,
        checkpoint_lanes: cache.checkpoint_lanes,
        speculative_storage: cache.speculative_storage,
        committed_lane: 0,
    };
    cache.restore_recurrent_checkpoint_state(slots[1], &checkpoint)?;
    for layer_idx in [0, 2] {
        let HybridLayerCache::Recurrent(pool) = cache.get(layer_idx).unwrap() else {
            unreachable!()
        };
        let deferred = pool.deferred_state().unwrap();
        assert_eq!(deferred.pending_rows.to_vec1::<u32>()?[1], 0);
    }

    assert!(cache.release_seq(0, slots[0])?);
    for layer_idx in [0, 2] {
        let HybridLayerCache::Recurrent(pool) = cache.get(layer_idx).unwrap() else {
            unreachable!()
        };
        let deferred = pool.deferred_state().unwrap();
        assert_eq!(deferred.pending_rows.to_vec1::<u32>()?[0], 0);
    }
    Ok(())
}

#[test]
fn deferred_state_can_only_be_disabled_before_slot_allocation() -> Result<()> {
    let spec = GdnDeferredStateSpec {
        num_k_heads: 1,
        tiled_v_heads: false,
        depth: crate::cuda::gdn::GDN_DEFERRED_STATE_DEPTH,
    };
    let mut cache = HybridCache::new(
        gdn_config(vec![HybridLayerType::Recurrent]),
        DType::BF16,
        &[Device::Cpu],
    )?;
    cache.reserve_gdn_deferred_state(&[(0, spec)])?;
    let generation = cache.recurrent_storage_generation();
    assert!(cache.disable_gdn_deferred_state()?);
    assert_eq!(cache.recurrent_storage_generation(), generation + 1);
    assert!(!cache.uses_gdn_deferred_state());
    assert!(!cache.disable_gdn_deferred_state()?);

    cache.reserve_gdn_deferred_state(&[(0, spec)])?;
    cache.allocate_seq(0)?;
    assert!(cache.disable_gdn_deferred_state().is_err());
    assert!(cache.uses_gdn_deferred_state());
    Ok(())
}

#[test]
fn recurrent_layout_reserves_capacity_and_lanes_atomically() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![
            HybridLayerType::Recurrent,
            HybridLayerType::Attention,
            HybridLayerType::Recurrent,
        ]),
        DType::F32,
        &[Device::Cpu, Device::Cpu, Device::Cpu],
    )?;
    for layer in [0, 2] {
        let HybridLayerCache::Recurrent(pool) = cache.get_mut(layer).unwrap() else {
            unreachable!()
        };
        pool.conv_state = Tensor::ones((INITIAL_POOL_CAPACITY, 2, 3), DType::F32, &Device::Cpu)?;
        pool.recurrent_state =
            Tensor::ones((INITIAL_POOL_CAPACITY, 2, 2), DType::F32, &Device::Cpu)?;
    }
    cache.set_state_indices_with_host(
        Some(Tensor::from_vec(vec![2u32], (1,), &Device::Cpu)?),
        Some(vec![2]),
    )?;

    assert!(cache.reserve_recurrent_layout(17, 3)?);

    assert_eq!(cache.recurrent_capacity(), 17);
    assert_eq!(cache.checkpoint_lanes(), 3);
    assert_eq!(cache.recurrent_storage_generation(), 1);
    assert_eq!(cache.slot_owners, vec![None; 17]);
    assert_eq!(cache.initialized_slots, vec![false; 17]);
    assert_eq!(cache.pristine_zero_slots, vec![true; 17]);
    assert_eq!(cache.last_released_sequence_owners, vec![None; 17]);
    assert_eq!(cache.committed_lanes, vec![0; 17]);
    assert!(cache.state_indices().is_none());
    assert!(cache.state_indices_host().is_none());
    assert!(cache.logical_state_indices_host().is_none());
    assert!(cache.device_state_indices.is_empty());
    for layer in [0, 2] {
        let HybridLayerCache::Recurrent(pool) = cache.get(layer).unwrap() else {
            unreachable!()
        };
        assert_eq!(pool.capacity(), 17);
        assert_eq!(pool.checkpoint_lanes(), 3);
        assert_eq!(pool.physical_capacity(), 51);
        assert_eq!(pool.conv_state.dims(), &[51, 2, 3]);
        assert_eq!(pool.recurrent_state.dims(), &[51, 2, 2]);
        assert_eq!(pool.num_free_slots(), 17);
        assert!(pool
            .conv_state
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| *value == 0.0));
        assert!(pool
            .recurrent_state
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| *value == 0.0));
    }
    assert!(cache.configure_checkpoint_lanes(4).is_err());
    assert!(cache.reserve_recurrent_layout(17, 3).is_err());
    Ok(())
}

#[test]
fn recurrent_layout_noop_locks_and_clears_batch_indices() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    cache.set_state_indices_with_host(
        Some(Tensor::from_vec(vec![2u32], (1,), &Device::Cpu)?),
        Some(vec![2]),
    )?;
    let storage_id = match cache.get(0).unwrap() {
        HybridLayerCache::Recurrent(pool) => pool.conv_state.id(),
        HybridLayerCache::Attention(_) => unreachable!(),
    };

    assert!(!cache.reserve_recurrent_layout(1, 1)?);

    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(pool.conv_state.id(), storage_id);
    assert_eq!(cache.recurrent_capacity(), INITIAL_POOL_CAPACITY);
    assert_eq!(cache.recurrent_storage_generation(), 0);
    assert!(cache.state_indices().is_none());
    assert!(cache.state_indices_host().is_none());
    assert!(cache.logical_state_indices_host().is_none());
    assert!(cache.configure_checkpoint_lanes(2).is_err());
    assert!(cache
        .reserve_recurrent_layout(INITIAL_POOL_CAPACITY, 1)
        .is_err());
    Ok(())
}

#[test]
fn recurrent_layout_failure_preserves_the_original_layout() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu, Device::Cpu],
    )?;
    let storage_ids = cache
        .caches
        .iter()
        .filter_map(HybridLayerCache::as_recurrent_pool)
        .map(|pool| pool.conv_state.id())
        .collect::<Vec<_>>();

    let error = cache.reserve_recurrent_layout(usize::MAX, 2).unwrap_err();
    assert!(error
        .to_string()
        .contains("recurrent physical capacity overflow"));
    assert_eq!(cache.recurrent_capacity(), INITIAL_POOL_CAPACITY);
    assert_eq!(cache.checkpoint_lanes(), 1);
    assert_eq!(cache.recurrent_storage_generation(), 0);
    assert_eq!(
        cache
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .map(|pool| pool.conv_state.id())
            .collect::<Vec<_>>(),
        storage_ids
    );

    assert!(cache.reserve_recurrent_layout(17, 3)?);
    assert_eq!(cache.recurrent_capacity(), 17);
    assert_eq!(cache.checkpoint_lanes(), 3);
    assert_eq!(cache.recurrent_storage_generation(), 1);
    Ok(())
}

#[test]
fn recurrent_layout_reservation_rejects_invalid_cache_state() -> Result<()> {
    let mut zero_lanes = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    assert!(zero_lanes
        .reserve_recurrent_layout(INITIAL_POOL_CAPACITY, 0)
        .unwrap_err()
        .to_string()
        .contains("must be nonzero"));

    let mut no_recurrent = HybridCache::new(
        config(vec![HybridLayerType::Attention]),
        DType::F32,
        &[Device::Cpu],
    )?;
    assert!(no_recurrent
        .reserve_recurrent_layout(INITIAL_POOL_CAPACITY, 1)
        .unwrap_err()
        .to_string()
        .contains("no recurrent state pool"));

    let mut divergent = HybridCache::new(
        config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu, Device::Cpu],
    )?;
    let HybridLayerCache::Recurrent(pool) = divergent.get_mut(1).unwrap() else {
        unreachable!()
    };
    pool.capacity -= 1;
    assert!(divergent
        .reserve_recurrent_layout(17, 2)
        .unwrap_err()
        .to_string()
        .contains("capacities diverged"));

    let mut active = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    active.allocate_seq(10)?;
    active.recurrent_storage_locked = false;
    assert!(active
        .reserve_recurrent_layout(17, 2)
        .unwrap_err()
        .to_string()
        .contains("recurrent slots are allocated"));

    let mut locked = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    locked.reserve_recurrent_capacity(INITIAL_POOL_CAPACITY)?;
    assert!(locked
        .reserve_recurrent_layout(17, 2)
        .unwrap_err()
        .to_string()
        .contains("already reserved or allocated"));
    Ok(())
}

#[test]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn checkpoint_batch_maps_logical_slots_and_commits_keep_rows() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    cache.configure_checkpoint_lanes(4)?;
    let first = cache.allocate_seq(10)?;
    let second = cache.allocate_seq(20)?;
    assert_eq!((first, second), (0, 1));

    let logical = vec![second as u32, first as u32];
    let indices = Tensor::from_vec(logical.clone(), (2,), &Device::Cpu)?;
    cache.set_state_indices_with_host(Some(indices), Some(logical.clone()))?;
    assert_eq!(cache.logical_state_indices_host(), Some(logical.as_slice()));
    assert_eq!(cache.state_indices_host(), Some([4u32, 0].as_slice()));
    assert_eq!(cache.state_indices().unwrap().to_vec1::<u32>()?, vec![4, 0]);

    assert!(cache.commit_speculative_rows(&[(0, 3), (1, 2)])?);
    assert_eq!(cache.committed_lane(second)?, 2);
    assert_eq!(cache.committed_lane(first)?, 1);
    assert_eq!(cache.state_indices_host(), Some([6u32, 1].as_slice()));
    assert_eq!(
        cache.current_recurrent_batch_mapping(),
        Some(RecurrentBatchMapping {
            logical_slots: logical,
            physical_slots: vec![6, 1],
        })
    );

    assert!(cache.commit_speculative_rows(&[(0, 0)]).is_err());
    assert!(cache.commit_speculative_rows(&[(2, 1)]).is_err());
    assert_eq!(cache.state_indices_host(), Some([6u32, 1].as_slice()));
    Ok(())
}

#[test]
fn transition_batch_mapping_is_identity_and_does_not_switch_lanes() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    cache.configure_transition_lanes(8)?;
    let first = cache.allocate_seq(10)?;
    let second = cache.allocate_seq(20)?;

    let logical = vec![second as u32, first as u32];
    let indices = Tensor::from_vec(logical.clone(), (2,), &Device::Cpu)?;
    cache.set_state_indices_with_host(Some(indices), Some(logical.clone()))?;
    assert_eq!(cache.logical_state_indices_host(), Some(logical.as_slice()));
    assert_eq!(cache.state_indices_host(), Some(logical.as_slice()));
    assert_eq!(cache.state_indices().unwrap().to_vec1::<u32>()?, logical);

    assert!(!cache.commit_speculative_rows(&[(0, 8), (1, 3)])?);
    assert_eq!(cache.committed_lane(first)?, 0);
    assert_eq!(cache.committed_lane(second)?, 0);
    assert_eq!(cache.active_physical_slot(first)?, first);
    assert_eq!(cache.active_physical_slot(second)?, second);
    assert!(cache.physical_slot(first, 1).is_err());
    Ok(())
}

#[test]
fn lane_one_declines_speculative_commit() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    let indices = Tensor::from_vec(vec![0u32], (1,), &Device::Cpu)?;
    cache.set_state_indices_with_host(Some(indices), Some(vec![0]))?;
    assert!(!cache.commit_speculative_rows(&[(0, 1)])?);
    assert_eq!(cache.state_indices_host(), Some([0u32].as_slice()));
    Ok(())
}

#[test]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn checkpoint_snapshot_uses_active_lane_and_restore_canonicalizes() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    cache.configure_checkpoint_lanes(3)?;
    let source = cache.allocate_seq(10)?;
    let destination = cache.allocate_seq(20)?;
    let indices = Tensor::from_vec(vec![source as u32], (1,), &Device::Cpu)?;
    cache.set_state_indices_with_host(Some(indices), Some(vec![source as u32]))?;
    cache.commit_speculative_rows(&[(0, 3)])?;

    let source_physical = cache.active_physical_slot(source)?;
    let destination_base = cache.physical_slot(destination, 0)?;
    let conv = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 2, 3), &Device::Cpu)?;
    let recurrent = Tensor::from_vec(vec![7f32, 8., 9., 10.], (1, 2, 2), &Device::Cpu)?;
    {
        let HybridLayerCache::Recurrent(pool) = cache.get_mut(0).unwrap() else {
            unreachable!()
        };
        let source_idx = Tensor::from_vec(vec![source_physical as u32], (1,), &Device::Cpu)?;
        pool.scatter_conv_state(&source_idx, &conv)?;
        pool.scatter_recurrent_state(&source_idx, &recurrent)?;

        let destination_indices = Tensor::from_vec(
            (destination_base..destination_base + 3)
                .map(|idx| idx as u32)
                .collect::<Vec<_>>(),
            (3,),
            &Device::Cpu,
        )?;
        pool.scatter_conv_state(
            &destination_indices,
            &Tensor::ones((3, 2, 3), DType::F32, &Device::Cpu)?,
        )?;
        pool.scatter_recurrent_state(
            &destination_indices,
            &Tensor::ones((3, 2, 2), DType::F32, &Device::Cpu)?,
        )?;
    }

    let snapshots = cache.snapshot_recurrent_state(10, source)?;
    assert_eq!(
        snapshots[0].conv_state.to_vec3::<f32>()?,
        conv.to_vec3::<f32>()?
    );
    assert_eq!(
        snapshots[0].recurrent_state.to_vec3::<f32>()?,
        recurrent.to_vec3::<f32>()?
    );

    assert!(cache.release_seq(20, destination)?);
    assert_eq!(cache.reserve_seq_uninitialized(30)?, destination);
    cache.restore_recurrent_state(30, destination, &snapshots)?;
    cache.validate_sequence_slots(&[(30, destination)])?;
    assert_eq!(cache.committed_lane(destination)?, 0);
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    let restored_indices = Tensor::from_vec(
        (destination_base..destination_base + 3)
            .map(|idx| idx as u32)
            .collect::<Vec<_>>(),
        (3,),
        &Device::Cpu,
    )?;
    let restored_conv = pool.gather_conv_state(&restored_indices)?;
    let restored_recurrent = pool.gather_recurrent_state(&restored_indices)?;
    assert_eq!(
        restored_conv.i(0)?.to_vec2::<f32>()?,
        conv.i(0)?.to_vec2::<f32>()?
    );
    assert_eq!(
        restored_recurrent.i(0)?.to_vec2::<f32>()?,
        recurrent.i(0)?.to_vec2::<f32>()?
    );
    assert_eq!(
        restored_conv
            .narrow(0, 1, 2)?
            .sum_all()?
            .to_scalar::<f32>()?,
        0.0
    );
    assert_eq!(
        restored_recurrent
            .narrow(0, 1, 2)?
            .sum_all()?
            .to_scalar::<f32>()?,
        0.0
    );
    Ok(())
}

#[test]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn restore_consumes_pristine_slot_and_reuse_resets_restored_state() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    let slot = cache.reserve_seq_uninitialized(10)?;
    assert!(cache.pristine_zero_slots[slot]);

    let snapshots = vec![RecurrentStateSnapshot {
        conv_state: Tensor::ones((1, 2, 3), DType::F32, &Device::Cpu)?,
        recurrent_state: Tensor::ones((1, 2, 2), DType::F32, &Device::Cpu)?,
        state_layout: RecurrentStateLayout::Opaque,
    }];
    cache.restore_recurrent_state(10, slot, &snapshots)?;
    assert!(!cache.pristine_zero_slots[slot]);
    assert!(cache.initialized_slots[slot]);

    let indices = Tensor::from_vec(vec![slot as u32], (1,), &Device::Cpu)?;
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(
        pool.gather_conv_state(&indices)?
            .sum_all()?
            .to_scalar::<f32>()?,
        6.0
    );
    assert_eq!(
        pool.gather_recurrent_state(&indices)?
            .sum_all()?
            .to_scalar::<f32>()?,
        4.0
    );

    assert!(cache.release_seq(10, slot)?);
    assert_eq!(cache.reserve_seq_uninitialized(20)?, slot);
    cache.reset_seq(20, slot)?;
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(
        pool.gather_conv_state(&indices)?
            .sum_all()?
            .to_scalar::<f32>()?,
        0.0
    );
    assert_eq!(
        pool.gather_recurrent_state(&indices)?
            .sum_all()?
            .to_scalar::<f32>()?,
        0.0
    );
    Ok(())
}

#[test]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn deferred_recurrent_reservation_requires_initialization_without_resetting() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    let slot = cache.allocate_seq(1)?;
    let physical_slot = cache.active_physical_slot(slot)?;
    let indices = Tensor::from_vec(vec![physical_slot as u32], (1,), &Device::Cpu)?;
    {
        let HybridLayerCache::Recurrent(pool) = cache.get_mut(0).unwrap() else {
            unreachable!()
        };
        pool.scatter_conv_state(
            &indices,
            &Tensor::ones((1, 2, 3), DType::F32, &Device::Cpu)?,
        )?;
        pool.scatter_recurrent_state(
            &indices,
            &Tensor::ones((1, 2, 2), DType::F32, &Device::Cpu)?,
        )?;
    }
    assert!(cache.release_seq(1, slot)?);
    assert_eq!(cache.reserve_seq_uninitialized(10)?, slot);

    let error = cache.validate_sequence_slots(&[(10, slot)]).unwrap_err();
    assert!(error.to_string().contains("is not initialized"));
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(
        pool.gather_conv_state(&indices)?
            .sum_all()?
            .to_scalar::<f32>()?,
        6.0
    );
    assert_eq!(
        pool.gather_recurrent_state(&indices)?
            .sum_all()?
            .to_scalar::<f32>()?,
        4.0
    );

    cache.reset_seq(10, slot)?;
    cache.validate_sequence_slots(&[(10, slot)])?;
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(
        pool.gather_conv_state(&indices)?
            .sum_all()?
            .to_scalar::<f32>()?,
        0.0
    );
    assert_eq!(
        pool.gather_recurrent_state(&indices)?
            .sum_all()?
            .to_scalar::<f32>()?,
        0.0
    );
    Ok(())
}

#[test]
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn graph_snapshot_preserves_active_checkpoint_lane() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    cache.configure_checkpoint_lanes(3)?;
    let slot = cache.allocate_seq(10)?;
    let indices = Tensor::from_vec(vec![slot as u32], (1,), &Device::Cpu)?;
    cache.set_state_indices_with_host(Some(indices), Some(vec![slot as u32]))?;
    cache.commit_speculative_rows(&[(0, 3)])?;

    let base = cache.physical_slot(slot, 0)?;
    let physical_indices = Tensor::from_vec(
        (base..base + 3).map(|idx| idx as u32).collect::<Vec<_>>(),
        (3,),
        &Device::Cpu,
    )?;
    let conv = Tensor::from_vec(
        (1..=18).map(|value| value as f32).collect::<Vec<_>>(),
        (3, 2, 3),
        &Device::Cpu,
    )?;
    let recurrent = Tensor::from_vec(
        (21..=32).map(|value| value as f32).collect::<Vec<_>>(),
        (3, 2, 2),
        &Device::Cpu,
    )?;
    {
        let HybridLayerCache::Recurrent(pool) = cache.get_mut(0).unwrap() else {
            unreachable!()
        };
        pool.scatter_conv_state(&physical_indices, &conv)?;
        pool.scatter_recurrent_state(&physical_indices, &recurrent)?;
    }

    let snapshot = cache.snapshot_recurrent_checkpoint_state(slot)?;
    cache.reset_seq(10, slot)?;
    cache.restore_recurrent_checkpoint_state(slot, &snapshot)?;

    assert_eq!(cache.committed_lane(slot)?, 2);
    assert_eq!(cache.state_indices_host(), Some([2u32].as_slice()));
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    let restored_conv = pool.gather_conv_state(&physical_indices)?;
    let restored_recurrent = pool.gather_recurrent_state(&physical_indices)?;
    assert_eq!(
        restored_conv.i(2)?.to_vec2::<f32>()?,
        conv.i(2)?.to_vec2::<f32>()?
    );
    assert_eq!(
        restored_recurrent.i(2)?.to_vec2::<f32>()?,
        recurrent.i(2)?.to_vec2::<f32>()?
    );
    assert_eq!(
        restored_conv
            .narrow(0, 0, 2)?
            .sum_all()?
            .to_scalar::<f32>()?,
        0.0
    );
    assert_eq!(
        restored_recurrent
            .narrow(0, 0, 2)?
            .sum_all()?
            .to_scalar::<f32>()?,
        0.0
    );
    Ok(())
}

#[test]
fn transition_snapshot_restore_uses_the_single_physical_row() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    cache.configure_transition_lanes(8)?;
    let slot = cache.allocate_seq(10)?;
    let indices = Tensor::from_vec(vec![slot as u32], (1,), &Device::Cpu)?;
    cache.set_state_indices_with_host(Some(indices.clone()), Some(vec![slot as u32]))?;

    let conv = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 2, 3), &Device::Cpu)?;
    let recurrent = Tensor::from_vec(vec![7f32, 8., 9., 10.], (1, 2, 2), &Device::Cpu)?;
    {
        let HybridLayerCache::Recurrent(pool) = cache.get_mut(0).unwrap() else {
            unreachable!()
        };
        pool.scatter_conv_state(&indices, &conv)?;
        pool.scatter_recurrent_state(&indices, &recurrent)?;
    }

    let snapshot = cache.snapshot_recurrent_checkpoint_state(slot)?;
    assert_eq!(snapshot.checkpoint_lanes, 8);
    assert_eq!(
        snapshot.speculative_storage,
        RecurrentSpeculativeStorage::TransitionLog
    );
    assert_eq!(snapshot.committed_lane, 0);
    let mut wrong_mode = snapshot.clone();
    wrong_mode.speculative_storage = RecurrentSpeculativeStorage::FullCheckpoints;
    assert!(cache
        .restore_recurrent_checkpoint_state(slot, &wrong_mode)
        .unwrap_err()
        .to_string()
        .contains("storage mismatch"));

    cache.reset_seq(10, slot)?;
    cache.restore_recurrent_checkpoint_state(slot, &snapshot)?;
    assert_eq!(cache.committed_lane(slot)?, 0);
    assert_eq!(cache.state_indices_host(), Some([slot as u32].as_slice()));
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(
        pool.gather_conv_state(&indices)?.to_vec3::<f32>()?,
        conv.to_vec3::<f32>()?
    );
    assert_eq!(
        pool.gather_recurrent_state(&indices)?.to_vec3::<f32>()?,
        recurrent.to_vec3::<f32>()?
    );

    cache.release_seq(10, slot)?;
    cache.reset()?;
    assert_eq!(cache.checkpoint_lanes(), 8);
    assert_eq!(cache.physical_checkpoint_lanes(), 1);
    assert!(cache.uses_recurrent_transition_log());
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(pool.physical_capacity(), INITIAL_POOL_CAPACITY);
    assert_eq!(pool.conv_state.dims(), &[INITIAL_POOL_CAPACITY, 2, 3]);
    assert_eq!(pool.recurrent_state.dims(), &[INITIAL_POOL_CAPACITY, 2, 2]);
    Ok(())
}

#[test]
fn graph_state_indices_preserve_pad_sentinel() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    cache.configure_checkpoint_lanes(3)?;
    let physical = vec![2u32, u32::MAX, 2];
    let indices = Tensor::from_vec(physical.clone(), (3,), &Device::Cpu)?;
    cache.set_state_indices_tensors(physical.clone(), vec![(Device::Cpu, indices)]);

    assert_eq!(cache.state_indices_host(), Some(physical.as_slice()));
    assert_eq!(
        cache.logical_state_indices_host(),
        Some([0u32, u32::MAX, 0].as_slice())
    );
    Ok(())
}

#[test]
fn transition_graph_indices_preserve_logical_slots_and_pad_sentinel() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    cache.configure_transition_lanes(8)?;
    let physical = vec![8u32, u32::MAX, 7];
    let indices = Tensor::from_vec(physical.clone(), (3,), &Device::Cpu)?;
    cache.set_state_indices_tensors(physical.clone(), vec![(Device::Cpu, indices)]);

    assert_eq!(cache.state_indices_host(), Some(physical.as_slice()));
    assert_eq!(
        cache.logical_state_indices_host(),
        Some(physical.as_slice())
    );

    let indices = Tensor::from_vec(physical.clone(), (3,), &Device::Cpu)?;
    cache.set_physical_state_indices_with_host(Some(indices), Some(physical.clone()));
    assert_eq!(cache.state_indices_host(), Some(physical.as_slice()));
    assert_eq!(
        cache.logical_state_indices_host(),
        Some(physical.as_slice())
    );
    Ok(())
}

#[test]
fn cpu_gdn_pool_is_explicitly_key_major() -> Result<()> {
    let cache = HybridCache::new(
        gdn_config(vec![HybridLayerType::Recurrent]),
        DType::BF16,
        &[Device::Cpu],
    )?;
    let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
        unreachable!()
    };
    assert_eq!(pool.state_layout(), RecurrentStateLayout::GdnKeyMajor);
    assert_eq!(pool.recurrent_state.dims(), &[9, 3, 4, 5]);
    Ok(())
}

#[test]
fn recurrent_snapshot_restore_validates_layout_and_count() -> Result<()> {
    let mut cache = HybridCache::new(
        gdn_config(vec![HybridLayerType::Recurrent]),
        DType::BF16,
        &[Device::Cpu],
    )?;
    let source = cache.allocate_seq(10)?;
    let destination = cache.allocate_seq(20)?;
    let snapshots = cache.snapshot_recurrent_state(10, source)?;
    let valid_snapshots = snapshots.clone();
    assert_eq!(snapshots[0].state_layout, RecurrentStateLayout::GdnKeyMajor);
    assert!(cache.restore_recurrent_state(20, destination, &[]).is_err());

    let mut wrong_layout = snapshots;
    wrong_layout[0].state_layout = RecurrentStateLayout::GdnValueMajor;
    let error = cache
        .restore_recurrent_state(20, destination, &wrong_layout)
        .unwrap_err();
    assert!(error.to_string().contains("layout mismatch"));

    assert!(cache.release_seq(20, destination)?);
    let error = cache
        .restore_recurrent_state(20, destination, &valid_snapshots)
        .unwrap_err();
    assert!(error.to_string().contains("is not allocated"));
    Ok(())
}

#[test]
fn sequence_slot_validation_rejects_missing_stale_and_duplicate_slots() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu, Device::Cpu],
    )?;
    let slot = cache.allocate_seq(10)?;

    cache.validate_sequence_slots(&[(10, slot)])?;
    let missing = cache
        .validate_sequence_slots(&[(20, slot + 1)])
        .unwrap_err();
    assert!(missing.to_string().contains("is not allocated"));
    let stale = cache.validate_sequence_slots(&[(20, slot)]).unwrap_err();
    assert!(stale.to_string().contains("expected Sequence(20)"));
    let duplicate = cache
        .validate_sequence_slots(&[(10, slot), (10, slot)])
        .unwrap_err();
    assert!(duplicate.to_string().contains("multiple sequences"));
    Ok(())
}

#[test]
fn stale_aba_release_cannot_free_a_reused_slot() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu, Device::Cpu],
    )?;
    let slot = cache.allocate_seq(10)?;
    assert!(cache.release_seq(10, slot)?);
    assert!(!cache.release_seq(10, slot)?);

    let reused = cache.allocate_seq(20)?;
    assert_eq!(reused, slot);
    let stale = cache.release_seq(10, slot).unwrap_err();
    assert!(stale.to_string().contains("owned by Sequence(20)"));
    cache.validate_sequence_slots(&[(20, slot)])?;
    assert!(cache.release_seq(20, slot)?);
    Ok(())
}

#[test]
fn graph_pad_has_distinct_ownership_and_blocks_storage_reset() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    let pad = cache.graph_pad_slot()?.unwrap();

    assert!(cache.release_seq(10, pad).is_err());
    assert!(cache.reset().is_err());
    assert!(cache.release_graph_pad_slot()?);
    assert!(!cache.release_graph_pad_slot()?);
    cache.reset()?;
    Ok(())
}

#[test]
fn recurrent_storage_generation_tracks_committed_replacements() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu],
    )?;
    assert_eq!(cache.recurrent_storage_generation(), 0);
    assert!(cache.configure_checkpoint_lanes(2)?);
    assert_eq!(cache.recurrent_storage_generation(), 1);
    assert!(cache.reserve_recurrent_capacity(17)?);
    assert_eq!(cache.recurrent_storage_generation(), 2);
    assert!(!cache.reserve_recurrent_capacity(16)?);
    assert_eq!(cache.recurrent_storage_generation(), 2);
    cache.reset()?;
    assert_eq!(cache.recurrent_storage_generation(), 3);
    Ok(())
}

#[test]
fn failed_growth_does_not_partially_replace_recurrent_storage() -> Result<()> {
    let mut cache = HybridCache::new(
        config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
        DType::F32,
        &[Device::Cpu, Device::Cpu],
    )?;
    let first_storage_id = match cache.get(0).unwrap() {
        HybridLayerCache::Recurrent(pool) => pool.conv_state.id(),
        HybridLayerCache::Attention(_) => unreachable!(),
    };
    let original_second = {
        let HybridLayerCache::Recurrent(pool) = cache.get_mut(1).unwrap() else {
            unreachable!()
        };
        std::mem::replace(
            &mut pool.conv_state,
            Tensor::zeros((INITIAL_POOL_CAPACITY, 1, 1), DType::F32, &Device::Cpu)?,
        )
    };
    let generation = cache.recurrent_storage_generation();

    assert!(cache.reserve_recurrent_capacity(17).is_err());
    assert_eq!(cache.recurrent_storage_generation(), generation);
    assert_eq!(cache.recurrent_capacity(), INITIAL_POOL_CAPACITY);
    let first = match cache.get(0).unwrap() {
        HybridLayerCache::Recurrent(pool) => pool,
        HybridLayerCache::Attention(_) => unreachable!(),
    };
    assert_eq!(first.capacity(), INITIAL_POOL_CAPACITY);
    assert_eq!(first.conv_state.id(), first_storage_id);

    let HybridLayerCache::Recurrent(pool) = cache.get_mut(1).unwrap() else {
        unreachable!()
    };
    pool.conv_state = original_second;
    Ok(())
}
