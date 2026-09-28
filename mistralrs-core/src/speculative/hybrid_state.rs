use std::collections::BTreeMap;

use candle_core::{Device, Result, Tensor};

use crate::{
    gdn::{GatedDeltaNet, GdnForwardStash, GdnLayerCache, GdnTransitionStash},
    kv_cache::{HybridCache, HybridLayerCache, RecurrentStateLayout},
    pipeline::RecurrentBatchKind,
};

/// Target activations captured for the MTP proposer: hidden states after the final norm and their
/// text RoPE or MRoPE position ids, `[b, rows, hidden]` / `[b, rows]` or `[3, b, rows]`.
#[derive(Clone)]
pub(crate) struct SpecCapture {
    pub(crate) hidden: Tensor,
    pub(crate) positions: Tensor,
    // Hidden states after each DFlash tap layer, row-aligned with `hidden`; empty unless attached
    pub(crate) taps: Vec<Tensor>,
}

/// Per-GDN-layer inputs and pre-forward states of the last multi-token decode, so a rejected tail
/// can be undone by replaying only the accepted prefix.
#[derive(Clone)]
pub(crate) struct GdnReplayStash {
    pub(crate) slots: Vec<u32>,
    pub(crate) layers: Vec<GdnLayerStash>,
}

#[derive(Clone)]
pub(crate) struct GdnLayerStash {
    pub(crate) layer_idx: usize,
    pub(crate) state_layout: RecurrentStateLayout,
    pub(crate) rollback: GdnLayerRollback,
}

#[derive(Clone)]
pub(crate) enum GdnLayerRollback {
    Replay {
        projected: GdnForwardStash,
        conv_state: Tensor,
        recurrent_state: Tensor,
    },
    Transition(GdnTransitionStash),
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct GdnReplayBatch {
    pub(crate) keep_rows: usize,
    pub(crate) batch_indices: Vec<u32>,
    pub(crate) slots: Vec<u32>,
}

struct GdnReplayIndices {
    batch_indices: Tensor,
    slots: Tensor,
}

struct GdnCommitIndices {
    keep_rows: Tensor,
    slots: Tensor,
}

fn index_select_replay_rows(source: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if source.is_contiguous() {
        source.index_select(indices, 0)
    } else {
        source.contiguous()?.index_select(indices, 0)
    }
}

pub(crate) fn should_stash_gdn_replay(
    native_speculative_commit: bool,
    store_spec_hidden: bool,
    query_len: usize,
    batch_kind: Option<RecurrentBatchKind>,
    continuation_without_cache: bool,
) -> bool {
    !native_speculative_commit
        && store_spec_hidden
        && query_len > 1
        && batch_kind == Some(RecurrentBatchKind::SpeculativeDecode)
        && continuation_without_cache
}

pub(crate) fn narrow_spec_graph_tensor(
    tensor: &Tensor,
    batch_dim: usize,
    captured_batch: usize,
    real_batch: usize,
    name: &str,
) -> Result<Tensor> {
    let tensor_batch = tensor.dim(batch_dim)?;
    if tensor_batch != captured_batch {
        candle_core::bail!(
            "speculative graph {name} has batch {tensor_batch}, expected {captured_batch}"
        );
    }
    if real_batch == captured_batch {
        Ok(tensor.clone())
    } else {
        tensor.narrow(batch_dim, 0, real_batch)
    }
}

pub(crate) fn narrow_spec_capture(capture: &mut SpecCapture, real_batch: usize) -> Result<()> {
    let captured_batch = capture.hidden.dim(0)?;
    if real_batch > captured_batch {
        candle_core::bail!(
            "speculative graph batch {real_batch} exceeds captured batch {captured_batch}"
        );
    }
    capture.hidden = narrow_spec_graph_tensor(
        &capture.hidden,
        0,
        captured_batch,
        real_batch,
        "hidden state",
    )?;
    let position_batch_dim = match capture.positions.rank() {
        2 => 0,
        3 => 1,
        rank => candle_core::bail!("unexpected speculative position rank {rank}"),
    };
    capture.positions = narrow_spec_graph_tensor(
        &capture.positions,
        position_batch_dim,
        captured_batch,
        real_batch,
        "positions",
    )?;
    for tap in &mut capture.taps {
        *tap = narrow_spec_graph_tensor(tap, 0, captured_batch, real_batch, "tap")?;
    }
    Ok(())
}

pub(crate) fn narrow_gdn_replay_stash(stash: &mut GdnReplayStash, real_batch: usize) -> Result<()> {
    let captured_batch = stash.slots.len();
    if real_batch > captured_batch {
        candle_core::bail!("GDN replay batch {real_batch} exceeds captured batch {captured_batch}");
    }
    for layer in &mut stash.layers {
        match &mut layer.rollback {
            GdnLayerRollback::Replay {
                projected,
                conv_state,
                recurrent_state,
            } => {
                projected.mixed_qkv = narrow_spec_graph_tensor(
                    &projected.mixed_qkv,
                    0,
                    captured_batch,
                    real_batch,
                    "mixed_qkv",
                )?;
                projected.convolved_qkv = narrow_spec_graph_tensor(
                    &projected.convolved_qkv,
                    0,
                    captured_batch,
                    real_batch,
                    "convolved_qkv",
                )?;
                projected.b =
                    narrow_spec_graph_tensor(&projected.b, 0, captured_batch, real_batch, "b")?;
                projected.a =
                    narrow_spec_graph_tensor(&projected.a, 0, captured_batch, real_batch, "a")?;
                *conv_state = narrow_spec_graph_tensor(
                    conv_state,
                    0,
                    captured_batch,
                    real_batch,
                    "conv_state",
                )?;
                *recurrent_state = narrow_spec_graph_tensor(
                    recurrent_state,
                    0,
                    captured_batch,
                    real_batch,
                    "recurrent_state",
                )?;
            }
            GdnLayerRollback::Transition(_) => {}
        }
    }
    stash.slots.truncate(real_batch);
    Ok(())
}

pub(crate) fn group_gdn_replay_batches(
    rows: &[(usize, usize)],
    slots: &[u32],
) -> Result<Vec<GdnReplayBatch>> {
    let mut grouped = BTreeMap::<usize, Vec<(u32, u32)>>::new();
    for &(batch_idx, keep_rows) in rows {
        let tensor_idx = u32::try_from(batch_idx).map_err(|_| {
            candle_core::Error::msg(format!("GDN replay batch row {batch_idx} exceeds u32"))
        })?;
        let slot = *slots.get(batch_idx).ok_or_else(|| {
            candle_core::Error::msg(format!("GDN replay stash has no batch row {batch_idx}"))
        })?;
        grouped
            .entry(keep_rows)
            .or_default()
            .push((tensor_idx, slot));
    }
    Ok(grouped
        .into_iter()
        .map(|(keep_rows, rows)| GdnReplayBatch {
            keep_rows,
            batch_indices: rows.iter().map(|(batch_idx, _)| *batch_idx).collect(),
            slots: rows.into_iter().map(|(_, slot)| slot).collect(),
        })
        .collect())
}

pub(crate) fn refresh_gdn_stash_slots(stash: &mut GdnReplayStash, slots: &[u32]) -> Result<()> {
    let batch_size = stash.slots.len();
    if slots.len() < batch_size {
        candle_core::bail!(
            "GDN graph state has {batch_size} rows, but the live slot table has {}",
            slots.len()
        );
    }
    stash.slots.clear();
    stash.slots.extend_from_slice(&slots[..batch_size]);
    Ok(())
}

/// Snapshot of the proposer-facing outputs of one target forward (see `SpeculativeGraphState`).
#[derive(Clone)]
pub(crate) struct SpecGraphState {
    pub(crate) spec_capture: Option<SpecCapture>,
    pub(crate) full_capture: Option<SpecCapture>,
    pub(crate) gdn_stash: Option<GdnReplayStash>,
    // Model-specific rollback tensors with the batch on dim 0
    pub(crate) aux: Vec<Tensor>,
}

impl crate::speculative::SpeculativeGraphState for SpecGraphState {
    fn tensors(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        for capture in [&self.spec_capture, &self.full_capture]
            .into_iter()
            .flatten()
        {
            out.push(capture.hidden.clone());
            out.push(capture.positions.clone());
            out.extend(capture.taps.iter().cloned());
        }
        if let Some(stash) = &self.gdn_stash {
            for layer in &stash.layers {
                match &layer.rollback {
                    GdnLayerRollback::Replay {
                        projected,
                        conv_state,
                        recurrent_state,
                    } => {
                        out.push(projected.mixed_qkv.clone());
                        out.push(projected.convolved_qkv.clone());
                        out.push(projected.b.clone());
                        out.push(projected.a.clone());
                        out.push(conv_state.clone());
                        out.push(recurrent_state.clone());
                    }
                    GdnLayerRollback::Transition(_) => {}
                }
            }
        }
        out.extend(self.aux.iter().cloned());
        out
    }

    fn with_tensors(
        &self,
        tensors: Vec<Tensor>,
    ) -> Result<Box<dyn crate::speculative::SpeculativeGraphState>> {
        let mut tensors = tensors.into_iter();
        let mut next = || {
            tensors.next().ok_or_else(|| {
                candle_core::Error::msg("speculative graph state tensor list is short")
            })
        };
        let mut state = self.clone();
        for capture in [&mut state.spec_capture, &mut state.full_capture]
            .into_iter()
            .flatten()
        {
            capture.hidden = next()?;
            capture.positions = next()?;
            for tap in capture.taps.iter_mut() {
                *tap = next()?;
            }
        }
        if let Some(stash) = state.gdn_stash.as_mut() {
            for layer in stash.layers.iter_mut() {
                match &mut layer.rollback {
                    GdnLayerRollback::Replay {
                        projected,
                        conv_state,
                        recurrent_state,
                    } => {
                        projected.mixed_qkv = next()?;
                        projected.convolved_qkv = next()?;
                        projected.b = next()?;
                        projected.a = next()?;
                        *conv_state = next()?;
                        *recurrent_state = next()?;
                    }
                    GdnLayerRollback::Transition(_) => {}
                }
            }
        }
        for aux in state.aux.iter_mut() {
            *aux = next()?;
        }
        Ok(Box::new(state))
    }

    fn for_real_batch(
        &self,
        real_batch: usize,
    ) -> Result<Box<dyn crate::speculative::SpeculativeGraphState>> {
        let mut state = self.clone();
        for capture in [&mut state.spec_capture, &mut state.full_capture]
            .into_iter()
            .flatten()
        {
            narrow_spec_capture(capture, real_batch)?;
        }
        if let Some(stash) = state.gdn_stash.as_mut() {
            narrow_gdn_replay_stash(stash, real_batch)?;
        }
        for aux in state.aux.iter_mut() {
            let captured_batch = aux.dim(0)?;
            *aux = narrow_spec_graph_tensor(aux, 0, captured_batch, real_batch, "aux state")?;
        }
        Ok(Box::new(state))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(crate) fn replay_gdn_prefixes<'a>(
    stash: &GdnReplayStash,
    rows: &[(usize, usize)],
    hybrid_cache: &mut HybridCache,
    gdn_at: impl Fn(usize) -> Option<&'a GatedDeltaNet>,
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let transition_layers = stash
        .layers
        .iter()
        .filter(|layer| matches!(layer.rollback, GdnLayerRollback::Transition(_)))
        .count();
    if transition_layers != 0 && transition_layers != stash.layers.len() {
        candle_core::bail!("GDN speculative stash mixes replay and transition layers");
    }
    if transition_layers == stash.layers.len() && !stash.layers.is_empty() {
        candle_core::bail!("GDN direct transitions must be published before replay fallback");
    }

    let devices = stash.layers.iter().fold(Vec::new(), |mut devices, layer| {
        let GdnLayerRollback::Replay { projected, .. } = &layer.rollback else {
            unreachable!("transition layers were handled above")
        };
        let device = projected.mixed_qkv.device();
        if !devices
            .iter()
            .any(|cached: &Device| cached.same_device(device))
        {
            devices.push(device.clone());
        }
        devices
    });
    let fused_commit_supported = !stash.layers.is_empty()
        && stash.layers.iter().all(|layer| match &layer.rollback {
            GdnLayerRollback::Replay { projected, .. } => projected.mixed_qkv.device().is_cuda(),
            GdnLayerRollback::Transition(_) => false,
        })
        && {
            stash.layers.iter().all(|layer| {
                let (Some(gdn), Some(HybridLayerCache::Recurrent(pool))) =
                    (gdn_at(layer.layer_idx), hybrid_cache.get(layer.layer_idx))
                else {
                    return false;
                };
                let GdnLayerRollback::Replay {
                    projected,
                    conv_state,
                    recurrent_state,
                } = &layer.rollback
                else {
                    return false;
                };
                pool.state_layout() == layer.state_layout
                    && gdn.speculative_state_commit_supported(
                        projected,
                        conv_state,
                        recurrent_state,
                        pool,
                    )
            })
        };
    if fused_commit_supported {
        let mut keep_rows_host = vec![0u32; stash.slots.len()];
        for &(batch_idx, rows) in rows {
            let keep_rows = u32::try_from(rows).map_err(|_| {
                candle_core::Error::msg(format!("GDN commit row count {rows} exceeds u32"))
            })?;
            *keep_rows_host.get_mut(batch_idx).ok_or_else(|| {
                candle_core::Error::msg(format!("GDN replay stash has no batch row {batch_idx}"))
            })? = keep_rows;
        }
        let commit_indices = devices
            .iter()
            .map(|device| {
                Ok(GdnCommitIndices {
                    keep_rows: Tensor::from_vec(
                        keep_rows_host.clone(),
                        (keep_rows_host.len(),),
                        device,
                    )?,
                    slots: Tensor::from_vec(stash.slots.clone(), (stash.slots.len(),), device)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        for layer in &stash.layers {
            let GdnLayerRollback::Replay {
                projected,
                conv_state,
                recurrent_state,
            } = &layer.rollback
            else {
                unreachable!("transition layers were handled above")
            };
            let Some(gdn) = gdn_at(layer.layer_idx) else {
                candle_core::bail!("GDN replay stash points at a non-GDN layer")
            };
            let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer.layer_idx)
            else {
                candle_core::bail!(
                    "GDN replay stash layer {} has no recurrent state pool",
                    layer.layer_idx
                );
            };
            if pool.state_layout() != layer.state_layout {
                candle_core::bail!(
                    "GDN replay state layout mismatch: stash {:?}, pool {:?}",
                    layer.state_layout,
                    pool.state_layout()
                );
            }
            let device_idx = devices
                .iter()
                .position(|device| device.same_device(projected.mixed_qkv.device()))
                .expect("stashed GDN layer device was collected above");
            let indices = &commit_indices[device_idx];
            if !gdn.commit_state_batch_from_stash_cuda(
                projected,
                conv_state,
                recurrent_state,
                &indices.keep_rows,
                &indices.slots,
                pool,
            )? {
                candle_core::bail!("CUDA GDN speculative state commit was unavailable");
            }
        }
        return Ok(());
    }

    let batches = group_gdn_replay_batches(rows, &stash.slots)?;
    let replay_indices = batches
        .iter()
        .map(|batch| {
            devices
                .iter()
                .map(|device| {
                    Ok(GdnReplayIndices {
                        batch_indices: Tensor::from_vec(
                            batch.batch_indices.clone(),
                            (batch.batch_indices.len(),),
                            device,
                        )?,
                        slots: Tensor::from_vec(batch.slots.clone(), (batch.slots.len(),), device)?,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;

    for layer in &stash.layers {
        let GdnLayerRollback::Replay {
            projected,
            conv_state,
            recurrent_state,
        } = &layer.rollback
        else {
            unreachable!("transition layers were handled above")
        };
        let Some(gdn) = gdn_at(layer.layer_idx) else {
            candle_core::bail!("GDN replay stash points at a non-GDN layer")
        };
        let Some(HybridLayerCache::Recurrent(pool)) = hybrid_cache.get_mut(layer.layer_idx) else {
            candle_core::bail!(
                "GDN replay stash layer {} has no recurrent state pool",
                layer.layer_idx
            );
        };
        if pool.state_layout() != layer.state_layout {
            candle_core::bail!(
                "GDN replay state layout mismatch: stash {:?}, pool {:?}",
                layer.state_layout,
                pool.state_layout()
            );
        }
        let device_idx = devices
            .iter()
            .position(|device| device.same_device(projected.mixed_qkv.device()))
            .expect("stashed GDN layer device was collected above");
        for (group_idx, batch) in batches.iter().enumerate() {
            let indices = &replay_indices[group_idx][device_idx];
            let mut cache = GdnLayerCache::gathered(
                index_select_replay_rows(conv_state, &indices.batch_indices)?,
                index_select_replay_rows(recurrent_state, &indices.batch_indices)?,
                layer.state_layout,
            );
            gdn.advance_state_batch_from_stash(
                projected,
                &indices.batch_indices,
                batch.keep_rows,
                &mut cache,
            )?;
            pool.scatter_conv_state(&indices.slots, &cache.conv_state)?;
            pool.scatter_recurrent_state(&indices.slots, &cache.recurrent_state)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Tensor};

    use super::{
        group_gdn_replay_batches, refresh_gdn_stash_slots, should_stash_gdn_replay,
        GdnLayerRollback, GdnLayerStash, GdnReplayBatch, GdnReplayStash, SpecCapture,
        SpecGraphState,
    };
    use crate::{
        gdn::{GdnForwardStash, GdnTransitionStash},
        kv_cache::RecurrentStateLayout,
        pipeline::RecurrentBatchKind,
        speculative::SpeculativeGraphState,
    };

    #[test]
    fn gdn_replay_batches_group_by_prefix_and_preserve_row_order() {
        let batches =
            group_gdn_replay_batches(&[(3, 4), (0, 2), (2, 4), (1, 1)], &[40, 41, 42, 43]).unwrap();
        assert_eq!(
            batches,
            vec![
                GdnReplayBatch {
                    keep_rows: 1,
                    batch_indices: vec![1],
                    slots: vec![41],
                },
                GdnReplayBatch {
                    keep_rows: 2,
                    batch_indices: vec![0],
                    slots: vec![40],
                },
                GdnReplayBatch {
                    keep_rows: 4,
                    batch_indices: vec![3, 2],
                    slots: vec![43, 42],
                },
            ]
        );
    }

    #[test]
    fn gdn_replay_batches_allow_an_all_accepted_empty_set() {
        assert!(group_gdn_replay_batches(&[], &[10, 11]).unwrap().is_empty());
    }

    #[test]
    fn gdn_graph_stash_slot_refresh_preserves_real_batch() {
        let mut stash = GdnReplayStash {
            slots: vec![1, 2, 3],
            layers: Vec::new(),
        };

        refresh_gdn_stash_slots(&mut stash, &[10, 11, 12, u32::MAX, u32::MAX]).unwrap();
        assert_eq!(stash.slots, [10, 11, 12]);
        assert!(refresh_gdn_stash_slots(&mut stash, &[20, 21]).is_err());
        assert_eq!(stash.slots, [10, 11, 12]);
    }

    #[test]
    fn gdn_replay_stash_is_only_created_for_fallback_speculative_decode() {
        assert!(should_stash_gdn_replay(
            false,
            true,
            8,
            Some(RecurrentBatchKind::SpeculativeDecode),
            true,
        ));
        for (
            native_speculative_commit,
            store_spec_hidden,
            query_len,
            batch_kind,
            continuation_without_cache,
        ) in [
            (
                true,
                true,
                8,
                Some(RecurrentBatchKind::SpeculativeDecode),
                true,
            ),
            (
                false,
                false,
                8,
                Some(RecurrentBatchKind::SpeculativeDecode),
                true,
            ),
            (
                false,
                true,
                1,
                Some(RecurrentBatchKind::SpeculativeDecode),
                true,
            ),
            (false, true, 512, Some(RecurrentBatchKind::Prefill), true),
            (false, true, 8, Some(RecurrentBatchKind::Decode), true),
            (false, true, 8, None, true),
            (
                false,
                true,
                8,
                Some(RecurrentBatchKind::SpeculativeDecode),
                false,
            ),
        ] {
            assert!(!should_stash_gdn_replay(
                native_speculative_commit,
                store_spec_hidden,
                query_len,
                batch_kind,
                continuation_without_cache,
            ));
        }
    }

    #[test]
    fn speculative_graph_state_narrows_a_bucket_to_the_live_batch() {
        let device = Device::Cpu;
        let mrope_capture = || SpecCapture {
            hidden: Tensor::zeros((16, 8, 32), DType::F32, &device).unwrap(),
            positions: Tensor::zeros((3, 16, 8), DType::U32, &device).unwrap(),
            taps: vec![Tensor::zeros((16, 8, 32), DType::F32, &device).unwrap()],
        };
        let text_capture = || SpecCapture {
            hidden: Tensor::zeros((16, 8, 32), DType::F32, &device).unwrap(),
            positions: Tensor::zeros((16, 8), DType::U32, &device).unwrap(),
            taps: vec![Tensor::zeros((16, 8, 32), DType::F32, &device).unwrap()],
        };
        let state = SpecGraphState {
            spec_capture: Some(text_capture()),
            full_capture: Some(mrope_capture()),
            gdn_stash: Some(GdnReplayStash {
                slots: (0..16).collect(),
                layers: vec![GdnLayerStash {
                    layer_idx: 2,
                    rollback: GdnLayerRollback::Replay {
                        projected: GdnForwardStash {
                            mixed_qkv: Tensor::zeros((16, 8, 24), DType::F32, &device).unwrap(),
                            convolved_qkv: Tensor::zeros((16, 8, 24), DType::F32, &device).unwrap(),
                            b: Tensor::zeros((16, 8, 4), DType::F32, &device).unwrap(),
                            a: Tensor::zeros((16, 8, 4), DType::F32, &device).unwrap(),
                        },
                        conv_state: Tensor::zeros((16, 24, 4), DType::F32, &device).unwrap(),
                        recurrent_state: Tensor::zeros((16, 2, 3, 4), DType::F32, &device).unwrap(),
                    },
                    state_layout: RecurrentStateLayout::GdnValueMajor,
                }],
            }),
            aux: Vec::new(),
        };

        let state = state.for_real_batch(9).unwrap();
        let state = state.as_any().downcast_ref::<SpecGraphState>().unwrap();

        let text_capture = state.spec_capture.as_ref().unwrap();
        assert_eq!(text_capture.hidden.dims(), &[9, 8, 32]);
        assert_eq!(text_capture.positions.dims(), &[9, 8]);
        assert_eq!(text_capture.taps[0].dims(), &[9, 8, 32]);
        let mrope_capture = state.full_capture.as_ref().unwrap();
        assert_eq!(mrope_capture.hidden.dims(), &[9, 8, 32]);
        assert_eq!(mrope_capture.positions.dims(), &[3, 9, 8]);
        assert_eq!(mrope_capture.taps[0].dims(), &[9, 8, 32]);
        let stash = state.gdn_stash.as_ref().unwrap();
        assert_eq!(stash.slots, (0..9).collect::<Vec<_>>());
        let layer = &stash.layers[0];
        let GdnLayerRollback::Replay {
            projected,
            conv_state,
            recurrent_state,
        } = &layer.rollback
        else {
            panic!("expected replay stash")
        };
        assert_eq!(projected.mixed_qkv.dims(), &[9, 8, 24]);
        assert_eq!(projected.convolved_qkv.dims(), &[9, 8, 24]);
        assert_eq!(projected.b.dims(), &[9, 8, 4]);
        assert_eq!(projected.a.dims(), &[9, 8, 4]);
        assert_eq!(conv_state.dims(), &[9, 24, 4]);
        assert_eq!(recurrent_state.dims(), &[9, 2, 3, 4]);
    }

    #[test]
    fn speculative_graph_state_narrows_direct_transition_slots() {
        let state = SpecGraphState {
            spec_capture: None,
            full_capture: None,
            gdn_stash: Some(GdnReplayStash {
                slots: (0..16).collect(),
                layers: vec![GdnLayerStash {
                    layer_idx: 2,
                    rollback: GdnLayerRollback::Transition(GdnTransitionStash),
                    state_layout: RecurrentStateLayout::GdnValueMajor,
                }],
            }),
            aux: Vec::new(),
        };

        let state = state.for_real_batch(9).unwrap();
        let state = state.as_any().downcast_ref::<SpecGraphState>().unwrap();
        let stash = state.gdn_stash.as_ref().unwrap();
        assert_eq!(stash.slots, (0..9).collect::<Vec<_>>());
        let GdnLayerRollback::Transition(_) = &stash.layers[0].rollback else {
            panic!("expected transition stash")
        };
        assert!(state.tensors().is_empty());
    }

    #[test]
    fn speculative_graph_state_rejects_a_larger_live_batch() {
        let state = SpecGraphState {
            spec_capture: None,
            full_capture: None,
            gdn_stash: Some(GdnReplayStash {
                slots: vec![10],
                layers: Vec::new(),
            }),
            aux: Vec::new(),
        };
        assert!(state.for_real_batch(2).is_err());
    }
}
