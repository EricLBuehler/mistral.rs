//! Transition-log rollback for hybrid GDN targets: verify records per-token transitions into the cache's
//! pending pool instead of mutating state, and commit publishes each sequence's accepted prefix.

use candle_core::{DType, Device, Result, Tensor};

use super::{
    hybrid_state::{GdnLayerRollback, GdnReplayStash},
    SpeculativeCommitRow,
};
use crate::{
    gdn::{GatedDeltaNet, GdnTransitionCommitConfig},
    kv_cache::{HybridCache, HybridLayerCache},
};

const GDN_PENDING_APPLY_MAX_LAYERS: usize = 32;

struct GdnPendingApplyGroup {
    device: Device,
    config: GdnTransitionCommitConfig,
    layers: Vec<usize>,
}

struct PublishGroup {
    device: Device,
    capacity: usize,
    max_rows: usize,
    layers: Vec<usize>,
}

pub(crate) fn recurrent_checkpoint_devices_supported(devices: &[Device]) -> bool {
    cfg!(feature = "cuda") && !devices.is_empty() && devices.iter().all(Device::is_cuda)
}

fn terminal_gdn_transition_slots(rows: &[SpeculativeCommitRow], slots: &[u32]) -> Result<Vec<u32>> {
    rows.iter()
        .filter(|row| row.terminal)
        .map(|row| {
            slots.get(row.batch_idx).copied().ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "GDN transition stash has no terminal batch row {}",
                    row.batch_idx
                ))
            })
        })
        .collect()
}

fn gdn_transition_keep_rows(
    rows: &[SpeculativeCommitRow],
    batch_size: usize,
    max_rows: usize,
) -> Result<Vec<u32>> {
    if rows.len() != batch_size {
        candle_core::bail!(
            "GDN transition commit has {} rows for a {batch_size}-row stash",
            rows.len()
        );
    }
    let mut keep_rows = vec![None; batch_size];
    for row in rows {
        if row.keep_rows == 0 || row.keep_rows > max_rows {
            candle_core::bail!(
                "GDN transition commit row {} keeps {}, expected 1..={max_rows}",
                row.batch_idx,
                row.keep_rows
            );
        }
        let destination = keep_rows.get_mut(row.batch_idx).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "GDN transition stash has no batch row {}",
                row.batch_idx
            ))
        })?;
        if destination.is_some() {
            candle_core::bail!(
                "GDN transition commit contains batch row {} more than once",
                row.batch_idx
            );
        }
        *destination = Some(u32::try_from(row.keep_rows).map_err(|_| {
            candle_core::Error::msg(format!(
                "GDN transition row count {} exceeds u32",
                row.keep_rows
            ))
        })?);
    }
    keep_rows
        .into_iter()
        .enumerate()
        .map(|(batch_idx, rows)| {
            rows.ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "GDN transition commit is missing batch row {batch_idx}"
                ))
            })
        })
        .collect()
}

/// A model's GDN layers keyed by their hybrid cache index.
pub(crate) struct GdnTransitionLayers<'a> {
    pub(crate) layers: Vec<(usize, &'a GatedDeltaNet)>,
    pub(crate) dtype: DType,
    pub(crate) device: &'a Device,
}

impl GdnTransitionLayers<'_> {
    fn pool(cache: &HybridCache, layer_idx: usize) -> Option<&crate::kv_cache::RecurrentStatePool> {
        match cache.get(layer_idx) {
            Some(HybridLayerCache::Recurrent(pool)) => Some(pool),
            _ => None,
        }
    }

    fn gdn(&self, layer_idx: usize) -> Option<&GatedDeltaNet> {
        self.layers
            .iter()
            .find(|(idx, _)| *idx == layer_idx)
            .map(|(_, gdn)| *gdn)
    }

    pub(crate) fn supported(&self, cache: &HybridCache) -> bool {
        if !recurrent_checkpoint_devices_supported(&cache.recurrent_devices()) {
            return false;
        }
        !self.layers.is_empty()
            && self.layers.iter().all(|&(layer_idx, gdn)| {
                Self::pool(cache, layer_idx)
                    .is_some_and(|pool| gdn.speculative_transitions_supported(pool, self.dtype))
            })
    }

    pub(crate) fn reserve(&self, cache: &mut HybridCache) -> Result<bool> {
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }
        let max_rows = cache.checkpoint_lanes();
        let mut spec = None;
        for &(layer_idx, gdn) in &self.layers {
            let Some(pool) = Self::pool(cache, layer_idx) else {
                candle_core::bail!("GDN layer {layer_idx} has no recurrent state pool");
            };
            if !gdn.speculative_transitions_supported(pool, self.dtype) {
                return Ok(false);
            }
            let layer_spec = gdn.pending_transition_spec(max_rows);
            if spec
                .replace(layer_spec)
                .is_some_and(|spec| spec != layer_spec)
            {
                candle_core::bail!("GDN transition dimensions diverge across layers");
            }
        }
        let Some(spec) = spec else {
            return Ok(false);
        };
        cache.reserve_gdn_pending_transitions(spec)
    }

    pub(crate) fn apply_pending_for_slots(
        &self,
        cache: &HybridCache,
        slots: &[u32],
    ) -> Result<bool> {
        let mut slots = slots
            .iter()
            .copied()
            .filter(|slot| *slot != crate::cuda::gdn::GDN_PAD_SLOT)
            .collect::<Vec<_>>();
        slots.sort_unstable();
        slots.dedup();
        if slots.is_empty() {
            return Ok(true);
        }
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }
        let active_slots = Tensor::from_vec(slots.clone(), (slots.len(),), self.device)?;
        self.apply_pending(cache, &active_slots, false)
    }

    pub(crate) fn apply_pending_for_current_batch(&self, cache: &HybridCache) -> Result<bool> {
        let Some(active_slots) = cache.state_indices() else {
            return Ok(false);
        };
        self.apply_pending(cache, active_slots, true)
    }

    fn apply_pending(
        &self,
        cache: &HybridCache,
        active_slots: &Tensor,
        use_cached_device_slots: bool,
    ) -> Result<bool> {
        if active_slots.elem_count() == 0 {
            return Ok(true);
        }
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }
        let mut groups = Vec::<GdnPendingApplyGroup>::new();
        for &(layer_idx, gdn) in &self.layers {
            let Some(pool) = Self::pool(cache, layer_idx) else {
                return Ok(false);
            };
            if !gdn.speculative_transitions_supported(pool, self.dtype)
                || pool.pending_transitions().is_none()
            {
                return Ok(false);
            }
            let config = gdn.transition_commit_config(pool);
            let device = pool.device();
            if let Some(group) = groups
                .iter_mut()
                .find(|group| group.config == config && group.device.same_device(device))
            {
                group.layers.push(layer_idx);
            } else {
                groups.push(GdnPendingApplyGroup {
                    device: device.clone(),
                    config,
                    layers: vec![layer_idx],
                });
            }
        }
        if groups.is_empty() {
            return Ok(false);
        }

        for group in groups {
            let active_slots = if use_cached_device_slots {
                cache
                    .state_indices_for_device(&group.device)
                    .ok_or_else(|| {
                        candle_core::Error::msg(
                            "GDN transition batch has no device-local state slots",
                        )
                    })?
            } else {
                active_slots.to_device(&group.device)?
            };
            for layer_indices in group.layers.chunks(GDN_PENDING_APPLY_MAX_LAYERS) {
                let mut layers = Vec::with_capacity(layer_indices.len());
                for &layer_idx in layer_indices {
                    let pool = Self::pool(cache, layer_idx)
                        .expect("GDN transition pool was validated above");
                    let pending = pool
                        .pending_transitions()
                        .expect("GDN pending transition pool was validated above");
                    layers.push(crate::cuda::gdn::GdnPendingTransitionApplyLayer {
                        pending_conv_input: &pending.conv_input,
                        pending_key_banks: &pending.key_banks,
                        pending_key_bank: &pending.key_bank,
                        pending_delta: &pending.delta,
                        pending_decay: &pending.decay,
                        pending_keep_rows: &pending.keep_rows,
                        pending_epochs: &pending.pending_epochs,
                        conv_applied_epochs: &pending.conv_applied_epochs,
                        recurrent_applied_epochs: &pending.recurrent_applied_epochs,
                        conv_state: &pool.conv_state,
                        recurrent_state: &pool.recurrent_state,
                    });
                }
                crate::cuda::gdn::pending_transition_apply_batched_cuda(
                    crate::cuda::gdn::GdnPendingTransitionApply {
                        layers: &layers,
                        active_slots: &active_slots,
                        num_k_heads: group.config.num_k_heads,
                        num_v_heads: group.config.num_v_heads,
                        head_k_dim: group.config.head_k_dim,
                        head_v_dim: group.config.head_v_dim,
                        conv_dim: group.config.conv_dim,
                        conv_width: group.config.conv_width,
                        tiled_v_heads: group.config.tiled_v_heads,
                        state_layout: group.config.state_layout,
                    },
                )?;
            }
        }
        Ok(true)
    }

    /// Publishes each row's accepted prefix into the pending log; `false` means the stash is not
    /// transition-based and the caller must replay instead.
    pub(crate) fn stage_prefixes(
        &self,
        cache: &HybridCache,
        stash: &GdnReplayStash,
        rows: &[SpeculativeCommitRow],
    ) -> Result<bool> {
        if rows.is_empty() {
            return Ok(true);
        }
        if stash.layers.is_empty()
            || stash
                .layers
                .iter()
                .any(|layer| !matches!(layer.rollback, GdnLayerRollback::Transition(_)))
        {
            return Ok(false);
        }
        if !cache.uses_recurrent_transition_log() {
            return Ok(false);
        }

        let max_rows = cache.checkpoint_lanes();
        let keep_rows_host = gdn_transition_keep_rows(rows, stash.slots.len(), max_rows)?;
        let mut live_slots = stash
            .slots
            .iter()
            .copied()
            .filter(|slot| *slot != crate::cuda::gdn::GDN_PAD_SLOT)
            .collect::<Vec<_>>();
        live_slots.sort_unstable();
        if live_slots.windows(2).any(|slots| slots[0] == slots[1]) {
            candle_core::bail!("GDN transition batch contains duplicate recurrent slots");
        }

        let mut groups = Vec::<PublishGroup>::new();
        for (stash_idx, layer) in stash.layers.iter().enumerate() {
            let (Some(gdn), Some(pool)) = (
                self.gdn(layer.layer_idx),
                Self::pool(cache, layer.layer_idx),
            ) else {
                return Ok(false);
            };
            let Some(pending) = pool.pending_transitions() else {
                return Ok(false);
            };
            if !gdn.speculative_transitions_supported(pool, self.dtype)
                || pool.state_layout() != layer.state_layout
                || pending.capacity() != cache.recurrent_capacity()
                || pending.spec().num_k_heads != gdn.transition_commit_config(pool).num_k_heads
                || pending.spec().max_rows != max_rows
            {
                return Ok(false);
            }
            let device = pool.device();
            if let Some(group) = groups.iter_mut().find(|group| {
                group.capacity == pending.capacity()
                    && group.max_rows == pending.spec().max_rows
                    && group.device.same_device(device)
            }) {
                group.layers.push(stash_idx);
            } else {
                groups.push(PublishGroup {
                    device: device.clone(),
                    capacity: pending.capacity(),
                    max_rows: pending.spec().max_rows,
                    layers: vec![stash_idx],
                });
            }
        }

        for group in groups {
            if live_slots
                .iter()
                .any(|slot| *slot as usize >= group.capacity)
            {
                candle_core::bail!("GDN transition slot exceeds recurrent capacity");
            }
            let keep_rows = Tensor::from_vec(
                keep_rows_host.clone(),
                (keep_rows_host.len(),),
                &group.device,
            )?;
            let slots = Tensor::from_vec(stash.slots.clone(), (stash.slots.len(),), &group.device)?;
            let mut layers = Vec::with_capacity(group.layers.len());
            for stash_idx in group.layers {
                let pool = Self::pool(cache, stash.layers[stash_idx].layer_idx)
                    .expect("transition pool was validated above");
                let pending = pool
                    .pending_transitions()
                    .expect("pending transition pool was validated above");
                layers.push(crate::cuda::gdn::GdnPendingTransitionPublishLayer {
                    pending_keep_rows: &pending.keep_rows,
                    pending_epochs: &pending.pending_epochs,
                    pending_key_bank: &pending.key_bank,
                });
            }
            crate::cuda::gdn::pending_transition_publish_batched_cuda(
                crate::cuda::gdn::GdnPendingTransitionPublish {
                    layers: &layers,
                    keep_rows: &keep_rows,
                    destination_slots: &slots,
                    max_rows: group.max_rows,
                    destination_capacity: group.capacity,
                },
            )?;
        }
        let terminal_slots = terminal_gdn_transition_slots(rows, &stash.slots)?;
        if !terminal_slots.is_empty() && !self.apply_pending_for_slots(cache, &terminal_slots)? {
            candle_core::bail!("terminal GDN recurrent transitions cannot be applied");
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::{
        gdn_transition_keep_rows, recurrent_checkpoint_devices_supported,
        terminal_gdn_transition_slots,
    };
    use crate::speculative::SpeculativeCommitRow;

    #[test]
    fn terminal_transition_rows_select_exact_slots() {
        let rows = [
            SpeculativeCommitRow {
                batch_idx: 2,
                keep_rows: 3,
                accepted_all: true,
                terminal: true,
            },
            SpeculativeCommitRow {
                batch_idx: 0,
                keep_rows: 1,
                accepted_all: false,
                terminal: false,
            },
            SpeculativeCommitRow {
                batch_idx: 1,
                keep_rows: 2,
                accepted_all: false,
                terminal: true,
            },
        ];
        assert_eq!(
            terminal_gdn_transition_slots(&rows, &[10, 11, 12]).unwrap(),
            vec![12, 11]
        );
        assert!(terminal_gdn_transition_slots(&rows, &[10, 11]).is_err());
    }

    #[test]
    fn transition_commit_rows_require_a_unique_exact_cover() {
        let row = |batch_idx, keep_rows| SpeculativeCommitRow {
            batch_idx,
            keep_rows,
            accepted_all: false,
            terminal: false,
        };
        assert_eq!(
            gdn_transition_keep_rows(&[row(2, 3), row(0, 1), row(1, 2)], 3, 8).unwrap(),
            vec![1, 2, 3]
        );
        assert!(gdn_transition_keep_rows(&[row(0, 1), row(2, 3)], 3, 8).is_err());
        assert!(gdn_transition_keep_rows(&[row(0, 1), row(0, 2), row(2, 3)], 3, 8).is_err());
        assert!(gdn_transition_keep_rows(&[row(0, 0)], 1, 8).is_err());
        assert!(gdn_transition_keep_rows(&[row(0, 9)], 1, 8).is_err());
    }

    #[test]
    fn recurrent_checkpoint_device_gate_rejects_cpu_placement() {
        assert!(!recurrent_checkpoint_devices_supported(&[Device::Cpu]));
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires CUDA"]
    fn recurrent_checkpoint_device_gate_rejects_mixed_placement() -> candle_core::Result<()> {
        let cuda = Device::new_cuda(0)?;
        assert!(recurrent_checkpoint_devices_supported(
            std::slice::from_ref(&cuda)
        ));
        assert!(!recurrent_checkpoint_devices_supported(&[
            cuda,
            Device::Cpu
        ]));
        Ok(())
    }
}
