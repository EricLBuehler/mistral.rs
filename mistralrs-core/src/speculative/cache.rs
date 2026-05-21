use std::{collections::HashMap, sync::Arc};

use candle_core::{Device, Result, Tensor};

use crate::device_map::DeviceMapper;
use crate::kv_cache::{KvCache, LayerCaches, NormalCache};
use crate::paged_attention::CacheEngine;
use crate::pipeline::text_models_inputs_processor::{
    make_flash_params, FlashParams, InputMetadata, PagedAttentionInputMetadata, PagedAttentionMeta,
};
use crate::sequence::Sequence;

use super::proposer::SpeculativeKvCache;

#[derive(Clone, Copy)]
pub struct SpeculativeCacheOutcome {
    pub keep_len: usize,
    pub accepted_all: bool,
}

pub trait SpeculativeCacheGuard {
    fn commit(&mut self) -> Result<()>;
    fn rollback_to(&mut self, keep_len: usize) -> Result<()>;
}

pub trait SpeculativeCacheAccess {
    type Guard: SpeculativeCacheGuard;

    /// Returns `Ok(None)` when the cache cannot reserve speculative slots and
    /// the caller should fall back to normal decoding for this step.
    fn begin(
        &self,
        seq_id: usize,
        base_len: usize,
        verify_len: usize,
    ) -> Result<Option<Self::Guard>>;

    fn guard_for_reserved(&self, seq_id: usize, base_len: usize, verify_len: usize) -> Self::Guard;

    fn make_verify_input_metadata(
        &self,
        verify_tokens: &[u32],
        seq_id: usize,
        base_len: usize,
        device: &Device,
        mapper: &dyn DeviceMapper,
    ) -> Result<InputMetadata>;

    fn proposer_cache(&self, sequences: &[&Sequence]) -> Result<SpeculativeKvCache<'_>>;

    fn finish_verification(
        &self,
        guard: &mut Self::Guard,
        _seq: &mut Sequence,
        keep_len: usize,
        accepted_all: bool,
    ) -> Result<()> {
        if accepted_all {
            guard.commit()
        } else {
            guard.rollback_to(keep_len)
        }
    }

    fn finish_verification_batch(
        &self,
        guards: &mut [Option<Self::Guard>],
        seqs: &mut [&mut Sequence],
        outcomes: &[Option<SpeculativeCacheOutcome>],
    ) -> Result<()> {
        if guards.len() != seqs.len() || outcomes.len() != seqs.len() {
            candle_core::bail!(
                "speculative cache batch shape mismatch: guards={}, seqs={}, outcomes={}",
                guards.len(),
                seqs.len(),
                outcomes.len()
            );
        }
        for ((guard, seq), outcome) in guards.iter_mut().zip(seqs.iter_mut()).zip(outcomes) {
            let (Some(guard), Some(outcome)) = (guard.as_mut(), outcome) else {
                continue;
            };
            self.finish_verification(guard, seq, outcome.keep_len, outcome.accepted_all)?;
        }
        Ok(())
    }

    fn can_stage_proposal(
        &self,
        _sequences: &[&Sequence],
        _base_lens: &[usize],
        _proposal_len: usize,
    ) -> bool {
        true
    }

    fn supports_staged_verification(&self) -> bool {
        true
    }
}

pub struct PagedSpeculativeCacheAccess<'a> {
    metadata: &'a PagedAttentionMeta,
    kv_cache: Vec<(Tensor, Tensor)>,
}

impl<'a> PagedSpeculativeCacheAccess<'a> {
    pub fn new(metadata: &'a PagedAttentionMeta, cache_engine: &CacheEngine) -> Self {
        Self {
            metadata,
            kv_cache: cache_engine.get_kv_cache().clone(),
        }
    }
}

pub struct PagedSpeculativeCacheGuard<'a> {
    metadata: &'a PagedAttentionMeta,
    seq_id: usize,
    reserved_len: usize,
}

impl SpeculativeCacheGuard for PagedSpeculativeCacheGuard<'_> {
    fn commit(&mut self) -> Result<()> {
        Ok(())
    }

    fn rollback_to(&mut self, keep_len: usize) -> Result<()> {
        if keep_len < self.reserved_len {
            let mut kv_mgr = crate::get_mut_arcmutex!(self.metadata.kv_cache_manager);
            kv_mgr.trim_request_to_num_tokens(self.seq_id, keep_len);
        }
        Ok(())
    }
}

impl<'a> SpeculativeCacheAccess for PagedSpeculativeCacheAccess<'a> {
    type Guard = PagedSpeculativeCacheGuard<'a>;

    fn begin(
        &self,
        seq_id: usize,
        base_len: usize,
        verify_len: usize,
    ) -> Result<Option<Self::Guard>> {
        let reserved_len = base_len + verify_len;
        let mut kv_mgr = crate::get_mut_arcmutex!(self.metadata.kv_cache_manager);
        let Some(_) = kv_mgr.allocate_slots(seq_id, reserved_len, &[]) else {
            return Ok(None);
        };
        Ok(Some(PagedSpeculativeCacheGuard {
            metadata: self.metadata,
            seq_id,
            reserved_len,
        }))
    }

    fn guard_for_reserved(
        &self,
        seq_id: usize,
        base_len: usize,
        verify_len: usize,
    ) -> PagedSpeculativeCacheGuard<'a> {
        PagedSpeculativeCacheGuard {
            metadata: self.metadata,
            seq_id,
            reserved_len: base_len + verify_len,
        }
    }

    fn make_verify_input_metadata(
        &self,
        verify_tokens: &[u32],
        seq_id: usize,
        base_len: usize,
        device: &Device,
        mapper: &dyn DeviceMapper,
    ) -> Result<InputMetadata> {
        let verify_len = verify_tokens.len();
        if verify_len == 0 {
            candle_core::bail!("speculative verification requires at least one token.");
        }

        let kv_mgr = crate::get_mut_arcmutex!(self.metadata.kv_cache_manager);
        let full_table = kv_mgr
            .get_block_ids(seq_id)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "speculative sequence {seq_id} has no paged-attention blocks"
                ))
            })?
            .to_vec();
        drop(kv_mgr);

        let mut slot_mappings = Vec::with_capacity(verify_len);
        let mut block_tables = Vec::with_capacity(verify_len);
        let mut context_lens = Vec::with_capacity(verify_len);
        let mut full_block_tables = Vec::with_capacity(verify_len);
        let mut full_context_lens = Vec::with_capacity(verify_len);

        for row in 0..verify_len {
            let token_pos = base_len + row;
            let full_context_len = token_pos + 1;
            let block_number = full_table
                .get(token_pos / self.metadata.block_size)
                .copied()
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!(
                        "speculative verification block table is too small: token_pos={token_pos}, block_size={}, table_len={}",
                        self.metadata.block_size,
                        full_table.len()
                    ))
                })?;
            let slot = block_number
                .checked_mul(self.metadata.block_size)
                .and_then(|v| v.checked_add(token_pos % self.metadata.block_size))
                .ok_or_else(|| {
                    candle_core::Error::Msg("speculative verification slot overflowed".to_string())
                })?;
            slot_mappings.push(slot as i64);

            full_block_tables.push(full_table.clone());
            full_context_lens.push(full_context_len as u32);

            if let Some(sliding_window) = self.metadata.sliding_window {
                let window_start = full_context_len.saturating_sub(sliding_window);
                let slide_idx = window_start / self.metadata.block_size;
                let block_aligned_start = slide_idx * self.metadata.block_size;
                let context_len = full_context_len.saturating_sub(block_aligned_start);
                let needed_blocks = context_len.div_ceil(self.metadata.block_size);
                let slide_end = (slide_idx + needed_blocks).min(full_table.len());
                block_tables.push(full_table.get(slide_idx..slide_end).unwrap_or(&[]).to_vec());
                context_lens.push(context_len as u32);
            } else {
                block_tables.push(full_table.clone());
                context_lens.push(full_context_len as u32);
            }
        }

        let cpu = Device::Cpu;
        let input = Tensor::from_vec(verify_tokens.to_vec(), (1, verify_len), device)?;
        let slot_mappings = Tensor::from_vec(slot_mappings, (1, verify_len), &cpu)?;

        let max_block_table_len = block_tables.iter().map(Vec::len).max().unwrap_or(1).max(1);
        let block_tables = repeated_table_tensor(&block_tables, max_block_table_len, &cpu)?;
        let context_lens = Tensor::from_vec(context_lens, (verify_len,), &cpu)?;

        let full_max_block_table_len = full_block_tables
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(1)
            .max(1);
        let full_block_tables =
            repeated_table_tensor(&full_block_tables, full_max_block_table_len, &cpu)?;
        let full_context_lens = Tensor::from_vec(full_context_lens, (verify_len,), &cpu)?;

        Ok(InputMetadata {
            input,
            positions: vec![base_len],
            context_lens: vec![(0, verify_len)],
            position_ids: vec![base_len + verify_len],
            paged_attn_meta: Some(PagedAttentionInputMetadata {
                block_tables: Some(map_to_devices(&block_tables, device, mapper)?),
                context_lens: Some(map_to_devices(&context_lens, device, mapper)?),
                slot_mappings: map_to_devices(&slot_mappings, device, mapper)?,
                max_context_len: Some(
                    context_lens
                        .to_vec1::<u32>()?
                        .into_iter()
                        .max()
                        .unwrap_or(0) as usize,
                ),
                full_block_tables: Some(map_to_devices(&full_block_tables, device, mapper)?),
                full_context_lens: Some(map_to_devices(&full_context_lens, device, mapper)?),
                full_max_context_len: Some(base_len + verify_len),
                is_first_prompt_chunk: false,
                paged_kv_indptr: None,
                paged_kv_indices: None,
                paged_kv_last_page_len: None,
                paged_kv_request_indices: None,
                paged_kv_tile_indices: None,
                paged_kv_o_indptr: None,
                paged_kv_chunk_size: None,
                num_cached_tokens: None,
                query_lens: None,
                cu_seqlens_q: None,
                cu_seqlens_kv: None,
            }),
            flash_meta: FlashParams::empty(true),
        })
    }

    fn proposer_cache(&self, _sequences: &[&Sequence]) -> Result<SpeculativeKvCache<'_>> {
        Ok(SpeculativeKvCache::Paged {
            metadata: self.metadata,
            kv_cache: &self.kv_cache,
        })
    }
}

pub struct NormalSpeculativeCacheAccess {
    cache: Arc<std::sync::Mutex<NormalCache>>,
    seq_ids: Vec<usize>,
    snapshots: HashMap<usize, NormalSpeculativeSequenceSnapshot>,
    prepared_staged: Option<NormalSpeculativeCacheState>,
    sliding_window: Option<usize>,
    max_seq_len: usize,
}

impl NormalSpeculativeCacheAccess {
    pub fn new(
        cache: Arc<std::sync::Mutex<NormalCache>>,
        seqs: &[&mut Sequence],
        prepared_staged: Option<NormalSpeculativeCacheState>,
        max_seq_len: usize,
    ) -> Result<Self> {
        let sliding_window = {
            let cache_guard = cache.lock().unwrap();
            normal_cache_sliding_window(&cache_guard)
        };
        let mut snapshots = HashMap::new();
        for seq in seqs {
            snapshots.insert(*seq.id(), snapshot_normal_sequence(seq)?);
        }
        Ok(Self {
            cache,
            seq_ids: seqs.iter().map(|seq| *seq.id()).collect(),
            snapshots,
            prepared_staged,
            sliding_window,
            max_seq_len,
        })
    }

    pub fn prepare_staged_verification(
        seqs: &mut [&mut Sequence],
        max_seq_len: usize,
    ) -> Result<Option<NormalSpeculativeCacheState>> {
        let Some(staged_len) = crate::speculative::staging::staged_batch_width(seqs) else {
            return Ok(None);
        };
        let verify_len = staged_len + 1;
        let mut expected_base_len = None;

        if !seqs.iter().all(|seq| {
            let Some(base_len) = seq.get_toks().len().checked_sub(1) else {
                return false;
            };
            if let Some(expected) = expected_base_len {
                if expected != base_len {
                    return false;
                }
            } else {
                expected_base_len = Some(base_len);
            }
            normal_sequence_cache_can_snapshot_for_append(seq, base_len, verify_len, max_seq_len)
        }) {
            for seq in seqs.iter_mut() {
                seq.clear_staged_speculative_tokens();
            }
            return Ok(None);
        }

        let mut states = HashMap::new();
        for (row_idx, seq) in seqs.iter().enumerate() {
            let base_len = seq.get_toks().len().checked_sub(1).ok_or_else(|| {
                candle_core::Error::Msg("empty staged speculative sequence".into())
            })?;
            states.insert(
                *seq.id(),
                NormalSpeculativePreparedSequence {
                    base_len,
                    verify_len,
                    row_idx,
                    batch_len: seqs.len(),
                    snapshot: snapshot_normal_sequence(seq)?,
                },
            );
        }

        Ok(Some(NormalSpeculativeCacheState { states }))
    }

    fn row_for_seq(&self, seq_id: usize) -> usize {
        self.seq_ids
            .iter()
            .position(|id| *id == seq_id)
            .unwrap_or(0)
    }
}

pub struct NormalSpeculativeCacheGuard {
    cache: Arc<std::sync::Mutex<NormalCache>>,
    seq_id: usize,
    reserved_len: usize,
    row_idx: usize,
    batch_len: usize,
    snapshot: Option<NormalSpeculativeSequenceSnapshot>,
}

#[derive(Clone)]
pub struct NormalSpeculativeCacheState {
    states: HashMap<usize, NormalSpeculativePreparedSequence>,
}

#[derive(Clone)]
struct NormalSpeculativePreparedSequence {
    base_len: usize,
    verify_len: usize,
    row_idx: usize,
    batch_len: usize,
    snapshot: NormalSpeculativeSequenceSnapshot,
}

#[derive(Clone)]
struct NormalSpeculativeSequenceSnapshot {
    layers: Vec<Option<NormalSpeculativeLayerSnapshot>>,
}

#[derive(Clone)]
enum NormalSpeculativeLayerSnapshot {
    Normal {
        max_seq_len: usize,
        current_seq_len: usize,
    },
    Rotating {
        dim: usize,
        max_seq_len: usize,
        capacity_seq_len: usize,
        current_seq_len: usize,
        retained_k: Option<Tensor>,
        retained_v: Option<Tensor>,
    },
    Shared {
        owner: usize,
    },
}

impl SpeculativeCacheGuard for NormalSpeculativeCacheGuard {
    fn commit(&mut self) -> Result<()> {
        Ok(())
    }

    fn rollback_to(&mut self, keep_len: usize) -> Result<()> {
        let _ = keep_len;
        Ok(())
    }
}

impl SpeculativeCacheAccess for NormalSpeculativeCacheAccess {
    type Guard = NormalSpeculativeCacheGuard;

    fn begin(
        &self,
        seq_id: usize,
        base_len: usize,
        verify_len: usize,
    ) -> Result<Option<Self::Guard>> {
        let Some(snapshot) = self.snapshots.get(&seq_id).cloned() else {
            return Ok(None);
        };
        if !snapshot.can_append(base_len, verify_len, self.max_seq_len) {
            return Ok(None);
        }

        Ok(Some(NormalSpeculativeCacheGuard {
            cache: Arc::clone(&self.cache),
            seq_id,
            reserved_len: base_len + verify_len,
            row_idx: self.row_for_seq(seq_id),
            batch_len: self.seq_ids.len().max(1),
            snapshot: Some(snapshot),
        }))
    }

    fn guard_for_reserved(&self, seq_id: usize, base_len: usize, verify_len: usize) -> Self::Guard {
        let prepared = self
            .prepared_staged
            .as_ref()
            .and_then(|state| state.states.get(&seq_id));
        let snapshot = prepared
            .filter(|prepared| prepared.base_len == base_len && prepared.verify_len == verify_len)
            .map(|prepared| prepared.snapshot.clone());
        let (row_idx, batch_len) = prepared
            .map(|prepared| (prepared.row_idx, prepared.batch_len))
            .unwrap_or_else(|| (self.row_for_seq(seq_id), self.seq_ids.len().max(1)));
        NormalSpeculativeCacheGuard {
            cache: Arc::clone(&self.cache),
            seq_id,
            reserved_len: base_len + verify_len,
            row_idx,
            batch_len,
            snapshot,
        }
    }

    fn make_verify_input_metadata(
        &self,
        verify_tokens: &[u32],
        _seq_id: usize,
        base_len: usize,
        device: &Device,
        mapper: &dyn DeviceMapper,
    ) -> Result<InputMetadata> {
        let verify_len = verify_tokens.len();
        if verify_len == 0 {
            candle_core::bail!("speculative verification requires at least one token.");
        }

        let input = Tensor::from_vec(verify_tokens.to_vec(), (1, verify_len), device)?;
        let flash_meta = if crate::using_flash_attn() {
            make_flash_params(
                device,
                Some(mapper),
                &[0, verify_len as u32],
                &[0, (base_len + verify_len) as u32],
                self.sliding_window,
                true,
            )
            .map_err(candle_core::Error::msg)?
        } else {
            FlashParams::empty(true)
        };

        Ok(InputMetadata {
            input,
            positions: vec![base_len],
            context_lens: vec![(0, verify_len)],
            position_ids: vec![base_len + verify_len],
            paged_attn_meta: None,
            flash_meta,
        })
    }

    fn proposer_cache(&self, sequences: &[&Sequence]) -> Result<SpeculativeKvCache<'_>> {
        let (layers, cache_lens) = build_normal_proposer_cache(sequences)?;
        Ok(SpeculativeKvCache::Normal { layers, cache_lens })
    }

    fn finish_verification(
        &self,
        guard: &mut Self::Guard,
        seq: &mut Sequence,
        keep_len: usize,
        _accepted_all: bool,
    ) -> Result<()> {
        if keep_len > guard.reserved_len {
            candle_core::bail!(
                "speculative normal-cache keep_len {keep_len} exceeds reserved length {}",
                guard.reserved_len
            );
        }
        if _accepted_all {
            return Ok(());
        }
        let Some(snapshot) = guard.snapshot.as_ref() else {
            candle_core::bail!(
                "normal-cache speculative rollback for sequence {} did not have a prepared snapshot",
                guard.seq_id
            );
        };
        let mut cache = guard.cache.lock().unwrap();
        restore_normal_cache_after_verification(
            &mut cache,
            seq,
            snapshot,
            keep_len,
            guard.row_idx,
            guard.batch_len,
        )
    }

    fn finish_verification_batch(
        &self,
        guards: &mut [Option<Self::Guard>],
        seqs: &mut [&mut Sequence],
        outcomes: &[Option<SpeculativeCacheOutcome>],
    ) -> Result<()> {
        if guards.len() != seqs.len() || outcomes.len() != seqs.len() {
            candle_core::bail!(
                "normal-cache speculative batch shape mismatch: guards={}, seqs={}, outcomes={}",
                guards.len(),
                seqs.len(),
                outcomes.len()
            );
        }
        if seqs.len() <= 1 {
            for ((guard, seq), outcome) in guards.iter_mut().zip(seqs.iter_mut()).zip(outcomes) {
                let (Some(guard), Some(outcome)) = (guard.as_mut(), outcome) else {
                    continue;
                };
                self.finish_verification(guard, seq, outcome.keep_len, outcome.accepted_all)?;
            }
            return Ok(());
        }

        let post_forward_cache = self.cache.lock().unwrap().clone();
        let mut finalized_keep_lens = Vec::new();

        for ((guard, seq), outcome) in guards.iter_mut().zip(seqs.iter_mut()).zip(outcomes) {
            let (Some(guard), Some(outcome)) = (guard.as_mut(), outcome) else {
                continue;
            };
            if outcome.keep_len > guard.reserved_len {
                candle_core::bail!(
                    "speculative normal-cache keep_len {} exceeds reserved length {}",
                    outcome.keep_len,
                    guard.reserved_len
                );
            }

            if !outcome.accepted_all {
                let Some(snapshot) = guard.snapshot.as_ref() else {
                    candle_core::bail!(
                        "normal-cache speculative rollback for sequence {} did not have a prepared snapshot",
                        guard.seq_id
                    );
                };
                restore_normal_sequence_after_verification(
                    &post_forward_cache,
                    seq,
                    snapshot,
                    outcome.keep_len,
                    guard.row_idx,
                    guard.batch_len,
                )?;
            }
            finalized_keep_lens.push(outcome.keep_len);
        }

        // Non-paged batching is length-bucketed. If verification keeps the
        // whole batch at one cache length, rebuild the live model cache so the
        // next step can use CacheInstruction::Nothing. If acceptance diverged,
        // the scheduler will split the batch by sequence length and the next
        // step will clone in from the per-sequence caches we just fixed up.
        if finalized_keep_lens.len() == seqs.len()
            && finalized_keep_lens
                .windows(2)
                .all(|pair| pair[0] == pair[1])
        {
            let mut cache = self.cache.lock().unwrap();
            rebuild_live_normal_cache_from_sequences(&mut cache, seqs)?;
        }

        Ok(())
    }

    fn supports_staged_verification(&self) -> bool {
        true
    }

    fn can_stage_proposal(
        &self,
        sequences: &[&Sequence],
        base_lens: &[usize],
        proposal_len: usize,
    ) -> bool {
        sequences.len() == base_lens.len()
            && sequences
                .iter()
                .zip(base_lens.iter())
                .all(|(seq, base_len)| {
                    normal_sequence_cache_can_snapshot_for_append(
                        seq,
                        *base_len,
                        proposal_len + 1,
                        self.max_seq_len,
                    )
                })
    }
}

fn normal_cache_sliding_window(cache: &NormalCache) -> Option<usize> {
    let mut sliding_window = None;

    for layer in &cache.0 {
        if let KvCache::Rotating { k, .. } = layer {
            sliding_window.get_or_insert(k.max_seq_len());
        }
    }

    sliding_window
}

fn build_normal_proposer_cache(
    sequences: &[&Sequence],
) -> Result<(LayerCaches, Vec<Option<Vec<usize>>>)> {
    if sequences.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let num_layers = sequences[0].normal_cache_ref().len();
    let mut layers = Vec::with_capacity(num_layers);
    let mut cache_lens = Vec::with_capacity(num_layers);

    if sequences.len() == 1 {
        for layer in sequences[0].normal_cache_ref() {
            let Some(layer) = layer.as_ref() else {
                layers.push(None);
                cache_lens.push(None);
                continue;
            };
            if matches!(layer, KvCache::Shared { .. }) {
                layers.push(None);
                cache_lens.push(None);
                continue;
            }

            let Some(k) = layer.k()? else {
                layers.push(None);
                cache_lens.push(None);
                continue;
            };
            let Some(v) = layer.v()? else {
                layers.push(None);
                cache_lens.push(None);
                continue;
            };
            let len = k.dim(2)?;
            layers.push(Some((k, v)));
            cache_lens.push(Some(vec![len]));
        }
        return Ok((layers, cache_lens));
    }

    for layer_idx in 0..num_layers {
        let mut keys = Vec::with_capacity(sequences.len());
        let mut values = Vec::with_capacity(sequences.len());
        let mut lens = Vec::with_capacity(sequences.len());
        let mut layer_is_shared = false;

        for seq in sequences {
            let Some(layer) = seq
                .normal_cache_ref()
                .get(layer_idx)
                .and_then(|x| x.as_ref())
            else {
                layer_is_shared = true;
                break;
            };
            if matches!(layer, KvCache::Shared { .. }) {
                layer_is_shared = true;
                break;
            }

            let Some(k) = layer.k()? else {
                layer_is_shared = true;
                break;
            };
            let Some(v) = layer.v()? else {
                layer_is_shared = true;
                break;
            };
            lens.push(k.dim(2)?);
            keys.push(k);
            values.push(v);
        }

        if layer_is_shared {
            layers.push(None);
            cache_lens.push(None);
            continue;
        }

        let max_len = lens.iter().copied().max().unwrap_or(0);
        let mut padded_keys = Vec::with_capacity(keys.len());
        let mut padded_values = Vec::with_capacity(values.len());
        for (k, v) in keys.into_iter().zip(values) {
            padded_keys.push(pad_cache_tensor(k, max_len)?);
            padded_values.push(pad_cache_tensor(v, max_len)?);
        }

        let key_refs = padded_keys.iter().collect::<Vec<_>>();
        let value_refs = padded_values.iter().collect::<Vec<_>>();
        layers.push(Some((
            Tensor::cat(&key_refs, 0)?,
            Tensor::cat(&value_refs, 0)?,
        )));
        cache_lens.push(Some(lens));
    }

    Ok((layers, cache_lens))
}

fn pad_cache_tensor(tensor: Tensor, max_len: usize) -> Result<Tensor> {
    let len = tensor.dim(2)?;
    if len == max_len {
        return Ok(tensor);
    }
    let mut pad_shape = tensor.dims().to_vec();
    pad_shape[2] = max_len.saturating_sub(len);
    let pad = Tensor::zeros(pad_shape, tensor.dtype(), tensor.device())?;
    Tensor::cat(&[&tensor, &pad], 2)
}

fn snapshot_normal_sequence(seq: &Sequence) -> Result<NormalSpeculativeSequenceSnapshot> {
    let mut layers = Vec::with_capacity(seq.normal_cache_ref().len());
    for layer in seq.normal_cache_ref() {
        let Some(layer) = layer else {
            layers.push(None);
            continue;
        };
        let snapshot = match layer {
            KvCache::Normal { k, .. } => NormalSpeculativeLayerSnapshot::Normal {
                max_seq_len: k.max_seq_len,
                current_seq_len: k.current_seq_len,
            },
            KvCache::Rotating { k, v } => NormalSpeculativeLayerSnapshot::Rotating {
                dim: k.dim,
                max_seq_len: k.max_seq_len,
                capacity_seq_len: k.capacity_seq_len,
                current_seq_len: k.current_seq_len,
                retained_k: k.current_data()?,
                retained_v: v.current_data()?,
            },
            KvCache::Shared { owner } => NormalSpeculativeLayerSnapshot::Shared { owner: *owner },
        };
        layers.push(Some(snapshot));
    }
    Ok(NormalSpeculativeSequenceSnapshot { layers })
}

impl NormalSpeculativeSequenceSnapshot {
    fn can_append(&self, base_len: usize, verify_len: usize, max_context_len: usize) -> bool {
        if base_len + verify_len > max_context_len {
            return false;
        }
        self.layers.iter().all(|layer| match layer {
            Some(NormalSpeculativeLayerSnapshot::Normal {
                current_seq_len,
                max_seq_len,
                ..
            }) => *current_seq_len == base_len && base_len + verify_len <= *max_seq_len,
            Some(NormalSpeculativeLayerSnapshot::Rotating {
                current_seq_len,
                max_seq_len,
                ..
            }) => *current_seq_len == base_len && verify_len <= *max_seq_len,
            Some(NormalSpeculativeLayerSnapshot::Shared { .. }) | None => true,
        })
    }
}

fn restore_normal_cache_after_verification(
    cache: &mut NormalCache,
    seq: &mut Sequence,
    snapshot: &NormalSpeculativeSequenceSnapshot,
    keep_len: usize,
    row_idx: usize,
    batch_len: usize,
) -> Result<()> {
    for layer_idx in 0..snapshot.layers.len() {
        let Some(layer_snapshot) = snapshot.layers[layer_idx].as_ref() else {
            continue;
        };

        match layer_snapshot {
            NormalSpeculativeLayerSnapshot::Normal { .. } => {
                restore_normal_layer_len(cache, seq, layer_idx, keep_len)?;
            }
            NormalSpeculativeLayerSnapshot::Rotating { .. } => {
                restore_rotating_layer_window(
                    cache,
                    seq,
                    layer_idx,
                    layer_snapshot,
                    keep_len,
                    row_idx,
                    batch_len,
                )?;
            }
            NormalSpeculativeLayerSnapshot::Shared { owner } => {
                if let Some(seq_layer) = seq.normal_cache().get_mut(layer_idx) {
                    *seq_layer = Some(KvCache::Shared { owner: *owner });
                }
                if let Some(cache_layer) = cache.0.get_mut(layer_idx) {
                    *cache_layer = KvCache::Shared { owner: *owner };
                }
            }
        }
    }
    Ok(())
}

fn restore_normal_sequence_after_verification(
    post_forward_cache: &NormalCache,
    seq: &mut Sequence,
    snapshot: &NormalSpeculativeSequenceSnapshot,
    keep_len: usize,
    row_idx: usize,
    batch_len: usize,
) -> Result<()> {
    for layer_idx in 0..snapshot.layers.len() {
        let Some(layer_snapshot) = snapshot.layers[layer_idx].as_ref() else {
            continue;
        };

        match layer_snapshot {
            NormalSpeculativeLayerSnapshot::Normal { .. } => {
                restore_normal_sequence_layer_len(seq, layer_idx, keep_len)?;
            }
            NormalSpeculativeLayerSnapshot::Rotating { .. } => {
                restore_rotating_sequence_layer_window(
                    post_forward_cache,
                    seq,
                    layer_idx,
                    layer_snapshot,
                    keep_len,
                    row_idx,
                    batch_len,
                )?;
            }
            NormalSpeculativeLayerSnapshot::Shared { owner } => {
                if let Some(seq_layer) = seq.normal_cache().get_mut(layer_idx) {
                    *seq_layer = Some(KvCache::Shared { owner: *owner });
                }
            }
        }
    }
    Ok(())
}

fn restore_normal_layer_len(
    cache: &mut NormalCache,
    seq: &mut Sequence,
    layer_idx: usize,
    keep_len: usize,
) -> Result<()> {
    restore_normal_sequence_layer_len(seq, layer_idx, keep_len)?;
    if let Some(layer) = cache.0.get_mut(layer_idx) {
        layer.set_len(keep_len)?;
    }
    Ok(())
}

fn restore_normal_sequence_layer_len(
    seq: &mut Sequence,
    layer_idx: usize,
    keep_len: usize,
) -> Result<()> {
    if let Some(layer) = seq
        .normal_cache()
        .get_mut(layer_idx)
        .and_then(Option::as_mut)
    {
        layer.set_len(keep_len)?;
    }
    Ok(())
}

fn restore_rotating_layer_window(
    cache: &mut NormalCache,
    seq: &mut Sequence,
    layer_idx: usize,
    snapshot: &NormalSpeculativeLayerSnapshot,
    keep_len: usize,
    row_idx: usize,
    batch_len: usize,
) -> Result<()> {
    let NormalSpeculativeLayerSnapshot::Rotating {
        dim,
        max_seq_len,
        capacity_seq_len,
        ..
    } = snapshot
    else {
        unreachable!()
    };
    let (restored_k, restored_v) =
        restored_rotating_window_parts(cache, layer_idx, snapshot, keep_len, row_idx, batch_len)?;

    restore_rotating_sequence_layer(
        seq,
        layer_idx,
        restored_k.clone(),
        restored_v.clone(),
        *dim,
        *max_seq_len,
        *capacity_seq_len,
        keep_len,
    )?;

    if let Some(cache_layer) = cache.0.get_mut(layer_idx) {
        *cache_layer = KvCache::Rotating {
            k: restored_rotating_cache(
                restored_k,
                *dim,
                *max_seq_len,
                *capacity_seq_len,
                keep_len,
            )?,
            v: restored_rotating_cache(
                restored_v,
                *dim,
                *max_seq_len,
                *capacity_seq_len,
                keep_len,
            )?,
        };
    }

    Ok(())
}

fn restore_rotating_sequence_layer_window(
    post_forward_cache: &NormalCache,
    seq: &mut Sequence,
    layer_idx: usize,
    snapshot: &NormalSpeculativeLayerSnapshot,
    keep_len: usize,
    row_idx: usize,
    batch_len: usize,
) -> Result<()> {
    let NormalSpeculativeLayerSnapshot::Rotating {
        dim,
        max_seq_len,
        capacity_seq_len,
        ..
    } = snapshot
    else {
        unreachable!()
    };
    let (restored_k, restored_v) = restored_rotating_window_parts(
        post_forward_cache,
        layer_idx,
        snapshot,
        keep_len,
        row_idx,
        batch_len,
    )?;
    restore_rotating_sequence_layer(
        seq,
        layer_idx,
        restored_k,
        restored_v,
        *dim,
        *max_seq_len,
        *capacity_seq_len,
        keep_len,
    )
}

fn restored_rotating_window_parts(
    post_forward_cache: &NormalCache,
    layer_idx: usize,
    snapshot: &NormalSpeculativeLayerSnapshot,
    keep_len: usize,
    row_idx: usize,
    batch_len: usize,
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    let NormalSpeculativeLayerSnapshot::Rotating {
        dim,
        max_seq_len,
        current_seq_len,
        retained_k,
        retained_v,
        ..
    } = snapshot
    else {
        unreachable!()
    };
    let accepted_len = keep_len.checked_sub(*current_seq_len).ok_or_else(|| {
        candle_core::Error::Msg("normal-cache speculative keep_len underflow".into())
    })?;

    let cache_layer = post_forward_cache.0.get(layer_idx).ok_or_else(|| {
        candle_core::Error::Msg(format!("missing normal cache layer {layer_idx}"))
    })?;
    let (appended_k, appended_v) = rotating_appended_prefix(
        cache_layer,
        *dim,
        *current_seq_len,
        *max_seq_len,
        accepted_len,
        row_idx,
        batch_len,
    )?;

    let restored_k = reconstruct_rotating_retained(
        retained_k.as_ref(),
        appended_k.as_ref(),
        *dim,
        *max_seq_len,
    )?;
    let restored_v = reconstruct_rotating_retained(
        retained_v.as_ref(),
        appended_v.as_ref(),
        *dim,
        *max_seq_len,
    )?;
    Ok((restored_k, restored_v))
}

#[allow(clippy::too_many_arguments)]
fn restore_rotating_sequence_layer(
    seq: &mut Sequence,
    layer_idx: usize,
    restored_k: Option<Tensor>,
    restored_v: Option<Tensor>,
    dim: usize,
    max_seq_len: usize,
    capacity_seq_len: usize,
    keep_len: usize,
) -> Result<()> {
    if let Some(seq_layer) = seq.normal_cache().get_mut(layer_idx) {
        *seq_layer = Some(KvCache::Rotating {
            k: restored_rotating_cache(restored_k, dim, max_seq_len, capacity_seq_len, keep_len)?,
            v: restored_rotating_cache(restored_v, dim, max_seq_len, capacity_seq_len, keep_len)?,
        });
    }
    Ok(())
}

fn rotating_appended_prefix(
    layer: &KvCache,
    dim: usize,
    base_len: usize,
    max_seq_len: usize,
    accepted_len: usize,
    row_idx: usize,
    batch_len: usize,
) -> Result<(Option<Tensor>, Option<Tensor>)> {
    if accepted_len == 0 {
        return Ok((None, None));
    }
    let KvCache::Rotating { k, v } = layer else {
        candle_core::bail!("expected rotating cache layer during speculative rollback");
    };
    let retained_len = base_len.min(max_seq_len);
    let k = k
        .last_append_result()
        .ok_or_else(|| candle_core::Error::Msg("missing rotating K append result".into()))?;
    let v = v
        .last_append_result()
        .ok_or_else(|| candle_core::Error::Msg("missing rotating V append result".into()))?;
    let k = narrow_cache_batch_row(k, row_idx, batch_len)?;
    let v = narrow_cache_batch_row(v, row_idx, batch_len)?;
    Ok((
        Some(k.narrow(dim, retained_len, accepted_len)?.contiguous()?),
        Some(v.narrow(dim, retained_len, accepted_len)?.contiguous()?),
    ))
}

fn reconstruct_rotating_retained(
    retained: Option<&Tensor>,
    appended: Option<&Tensor>,
    dim: usize,
    max_seq_len: usize,
) -> Result<Option<Tensor>> {
    let combined = match (retained, appended) {
        (Some(retained), Some(appended)) => Tensor::cat(&[retained, appended], dim)?,
        (Some(retained), None) => retained.clone(),
        (None, Some(appended)) => appended.clone(),
        (None, None) => return Ok(None),
    };
    let len = combined.dim(dim)?;
    let keep = len.min(max_seq_len);
    combined
        .narrow(dim, len - keep, keep)?
        .contiguous()
        .map(Some)
}

fn restored_rotating_cache(
    retained: Option<Tensor>,
    dim: usize,
    max_seq_len: usize,
    capacity_seq_len: usize,
    current_seq_len: usize,
) -> Result<crate::kv_cache::RotatingCache> {
    let retained_len = match retained.as_ref() {
        Some(retained) => retained.dim(dim)?,
        None => 0,
    };
    let capacity_seq_len = capacity_seq_len.max(retained_len).min(max_seq_len);
    let all_data = if let Some(retained) = retained {
        let mut shape = retained.dims().to_vec();
        shape[dim] = capacity_seq_len;
        let data = Tensor::zeros(shape, retained.dtype(), retained.device())?;
        if retained_len > 0 {
            data.slice_set(&retained, dim, 0)?;
        }
        Some(data)
    } else {
        None
    };
    Ok(crate::kv_cache::RotatingCache {
        all_data,
        dim,
        current_seq_len,
        max_seq_len,
        capacity_seq_len,
        last_append_result: None,
    })
}

fn narrow_cache_batch_row(tensor: &Tensor, row_idx: usize, batch_len: usize) -> Result<Tensor> {
    let dim0 = tensor.dim(0)?;
    if batch_len == 0 || dim0 % batch_len != 0 {
        candle_core::bail!("normal-cache batch shape mismatch: dim0={dim0}, batch_len={batch_len}");
    }
    let per_row = dim0 / batch_len;
    tensor.narrow(0, row_idx * per_row, per_row)?.contiguous()
}

fn rebuild_live_normal_cache_from_sequences(
    cache: &mut NormalCache,
    seqs: &[&mut Sequence],
) -> Result<()> {
    if seqs.is_empty() {
        return Ok(());
    }
    for layer_idx in 0..cache.0.len() {
        let Some(first_layer) = seqs[0]
            .normal_cache_ref()
            .get(layer_idx)
            .and_then(|layer| layer.as_ref())
        else {
            cache.0[layer_idx].reset();
            continue;
        };

        match first_layer {
            KvCache::Shared { owner } => {
                cache.0[layer_idx] = KvCache::Shared { owner: *owner };
            }
            KvCache::Normal {
                k: first_k,
                v: first_v,
            } => {
                let mut keys = Vec::with_capacity(seqs.len());
                let mut values = Vec::with_capacity(seqs.len());
                for seq in seqs {
                    let Some(KvCache::Normal { k, v }) = seq
                        .normal_cache_ref()
                        .get(layer_idx)
                        .and_then(|layer| layer.as_ref())
                    else {
                        candle_core::bail!(
                            "normal-cache rebuild found mismatched layer {layer_idx}"
                        );
                    };
                    keys.push(k.current_data()?.ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "normal-cache rebuild missing K data for layer {layer_idx}"
                        ))
                    })?);
                    values.push(v.current_data()?.ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "normal-cache rebuild missing V data for layer {layer_idx}"
                        ))
                    })?);
                }
                let key_refs = keys.iter().collect::<Vec<_>>();
                let value_refs = values.iter().collect::<Vec<_>>();
                let batched_k = Tensor::cat(&key_refs, 0)?.contiguous()?;
                let batched_v = Tensor::cat(&value_refs, 0)?.contiguous()?;
                let current_seq_len = first_k.current_seq_len();
                cache.0[layer_idx] = KvCache::Normal {
                    k: crate::kv_cache::SingleCache {
                        all_data: Some(batched_k),
                        dim: first_k.dim(),
                        current_seq_len,
                        max_seq_len: first_k.max_seq_len(),
                        capacity_seq_len: current_seq_len,
                    },
                    v: crate::kv_cache::SingleCache {
                        all_data: Some(batched_v),
                        dim: first_v.dim(),
                        current_seq_len,
                        max_seq_len: first_v.max_seq_len(),
                        capacity_seq_len: current_seq_len,
                    },
                };
            }
            KvCache::Rotating {
                k: first_k,
                v: first_v,
            } => {
                let mut keys = Vec::with_capacity(seqs.len());
                let mut values = Vec::with_capacity(seqs.len());
                for seq in seqs {
                    let Some(KvCache::Rotating { k, v }) = seq
                        .normal_cache_ref()
                        .get(layer_idx)
                        .and_then(|layer| layer.as_ref())
                    else {
                        candle_core::bail!(
                            "normal-cache rebuild found mismatched rotating layer {layer_idx}"
                        );
                    };
                    keys.push(k.current_data()?.ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "normal-cache rebuild missing rotating K data for layer {layer_idx}"
                        ))
                    })?);
                    values.push(v.current_data()?.ok_or_else(|| {
                        candle_core::Error::Msg(format!(
                            "normal-cache rebuild missing rotating V data for layer {layer_idx}"
                        ))
                    })?);
                }
                let key_refs = keys.iter().collect::<Vec<_>>();
                let value_refs = values.iter().collect::<Vec<_>>();
                let batched_k = Tensor::cat(&key_refs, 0)?.contiguous()?;
                let batched_v = Tensor::cat(&value_refs, 0)?.contiguous()?;
                let current_seq_len = first_k.current_seq_len();
                let retained_len = batched_k.dim(first_k.dim())?;
                cache.0[layer_idx] = KvCache::Rotating {
                    k: crate::kv_cache::RotatingCache {
                        all_data: Some(batched_k),
                        dim: first_k.dim(),
                        current_seq_len,
                        max_seq_len: first_k.max_seq_len(),
                        capacity_seq_len: retained_len,
                        last_append_result: None,
                    },
                    v: crate::kv_cache::RotatingCache {
                        all_data: Some(batched_v),
                        dim: first_v.dim(),
                        current_seq_len,
                        max_seq_len: first_v.max_seq_len(),
                        capacity_seq_len: retained_len,
                        last_append_result: None,
                    },
                };
            }
        }
    }
    Ok(())
}

fn normal_sequence_cache_can_snapshot_for_append(
    seq: &Sequence,
    base_len: usize,
    verify_len: usize,
    max_context_len: usize,
) -> bool {
    let reserved_len = base_len + verify_len;
    if reserved_len > max_context_len {
        return false;
    }
    seq.normal_cache_ref().iter().all(|layer| match layer {
        Some(KvCache::Normal { k, v }) => {
            k.current_seq_len() == base_len
                && v.current_seq_len() == base_len
                && reserved_len <= k.max_seq_len()
                && reserved_len <= v.max_seq_len()
                && k.try_set_len(base_len).is_ok()
                && v.try_set_len(base_len).is_ok()
        }
        Some(KvCache::Rotating { k, v }) => {
            k.current_seq_len() == base_len
                && v.current_seq_len() == base_len
                && verify_len <= k.max_seq_len()
                && verify_len <= v.max_seq_len()
        }
        Some(KvCache::Shared { .. }) | None => true,
    })
}

fn repeated_table_tensor(rows: &[Vec<usize>], max_len: usize, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(rows.len() * max_len);
    for row in rows {
        values.extend(row.iter().map(|x| *x as u32));
        values.extend(std::iter::repeat_n(0u32, max_len.saturating_sub(row.len())));
    }
    Tensor::from_vec(values, (rows.len(), max_len), device)
}

fn map_to_devices(
    tensor: &Tensor,
    device: &Device,
    mapper: &dyn DeviceMapper,
) -> Result<HashMap<candle_core::DeviceLocation, Tensor>> {
    let mut devices = mapper.get_unique_devices();
    if !devices
        .iter()
        .any(|dev| dev.location() == device.location())
    {
        devices.push(device.clone());
    }

    let mut map = HashMap::new();
    for dev in devices {
        map.insert(dev.location(), tensor.to_device(&dev)?);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use candle_core::{Device, Tensor};
    use tokio::sync::{mpsc::channel, Mutex};

    use crate::kv_cache::{KvCache, NormalCache, RotatingCache};
    use crate::sampler::Sampler;
    use crate::sequence::{SeqStepType, Sequence, SequenceGroup, SequenceRecognizer};

    use super::{
        restore_normal_cache_after_verification, restore_normal_sequence_after_verification,
        snapshot_normal_sequence,
    };

    fn tensor(values: &[f32]) -> candle_core::Result<Tensor> {
        Tensor::new(values.to_vec(), &Device::Cpu)?.reshape((1, 1, values.len(), 1))
    }

    fn rotating_layer(values: &[f32], window: usize) -> candle_core::Result<KvCache> {
        let mut k = RotatingCache::new(2, window, window);
        let mut v = RotatingCache::new(2, window, window);
        let src = tensor(values)?;
        k.append(&src)?;
        v.append(&src)?;
        Ok(KvCache::Rotating { k, v })
    }

    fn make_sequence(layer: KvCache) -> Sequence {
        let (tx, _rx) = channel(1);
        let sampler =
            Sampler::new(None, 0, None, None, None, None, None, 32, 1.0, 0.0, vec![]).unwrap();
        let group = Arc::new(Mutex::new(SequenceGroup::new(1, false, true, None)));
        let mut seq = Sequence::new_waiting(
            vec![0, 1, 2, 3, 4, 5, 6],
            "prompt".to_string(),
            0,
            0,
            1,
            tx,
            sampler,
            vec![],
            vec![],
            None,
            false,
            false,
            group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            None,
            false,
            vec![],
        );
        seq.normal_cache()[0] = Some(layer);
        seq
    }

    #[test]
    fn rotating_rollback_reconstructs_retained_window_after_wrap() -> candle_core::Result<()> {
        let window = 4;
        let base_layer = rotating_layer(&[0., 1., 2., 3., 4., 5.], window)?;
        let mut seq = make_sequence(base_layer.clone());
        let snapshot = snapshot_normal_sequence(&seq)?;

        let mut cache = NormalCache(vec![base_layer]);
        if let KvCache::Rotating { k, v } = &mut cache.0[0] {
            let appended = tensor(&[6., 7., 8.])?;
            k.append(&appended)?;
            v.append(&appended)?;
        }

        restore_normal_cache_after_verification(&mut cache, &mut seq, &snapshot, 8, 0, 1)?;

        let seq_values = seq.normal_cache_ref()[0]
            .as_ref()
            .unwrap()
            .k()?
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(seq_values, vec![4., 5., 6., 7.]);
        assert_eq!(
            seq.normal_cache_ref()[0]
                .as_ref()
                .unwrap()
                .current_seq_len(),
            8
        );

        let cache_values = cache.0[0].k()?.unwrap().flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(cache_values, vec![4., 5., 6., 7.]);
        assert_eq!(cache.0[0].current_seq_len(), 8);

        Ok(())
    }

    #[test]
    fn batched_rotating_rollback_uses_each_row_append_result() -> candle_core::Result<()> {
        let window = 4;
        let mut seq_a = make_sequence(rotating_layer(&[0., 1., 2., 3., 4., 5.], window)?);
        let mut seq_b = make_sequence(rotating_layer(&[10., 11., 12., 13., 14., 15.], window)?);
        let snapshot_a = snapshot_normal_sequence(&seq_a)?;
        let snapshot_b = snapshot_normal_sequence(&seq_b)?;

        let row_a = tensor(&[2., 3., 4., 5., 6., 7., 8.])?;
        let row_b = tensor(&[12., 13., 14., 15., 16., 17., 18.])?;
        let appended = Tensor::cat(&[&row_a, &row_b], 0)?;
        let live_layer = KvCache::Rotating {
            k: RotatingCache {
                all_data: Some(Tensor::zeros(
                    (2, 1, window, 1),
                    candle_core::DType::F32,
                    &Device::Cpu,
                )?),
                dim: 2,
                current_seq_len: 9,
                max_seq_len: window,
                capacity_seq_len: window,
                last_append_result: Some(appended.clone()),
            },
            v: RotatingCache {
                all_data: Some(Tensor::zeros(
                    (2, 1, window, 1),
                    candle_core::DType::F32,
                    &Device::Cpu,
                )?),
                dim: 2,
                current_seq_len: 9,
                max_seq_len: window,
                capacity_seq_len: window,
                last_append_result: Some(appended),
            },
        };
        let live_cache = NormalCache(vec![live_layer]);

        restore_normal_sequence_after_verification(&live_cache, &mut seq_a, &snapshot_a, 8, 0, 2)?;
        restore_normal_sequence_after_verification(&live_cache, &mut seq_b, &snapshot_b, 7, 1, 2)?;

        let seq_a_values = seq_a.normal_cache_ref()[0]
            .as_ref()
            .unwrap()
            .k()?
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(seq_a_values, vec![4., 5., 6., 7.]);
        assert_eq!(
            seq_a.normal_cache_ref()[0]
                .as_ref()
                .unwrap()
                .current_seq_len(),
            8
        );

        let seq_b_values = seq_b.normal_cache_ref()[0]
            .as_ref()
            .unwrap()
            .k()?
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(seq_b_values, vec![13., 14., 15., 16.]);
        assert_eq!(
            seq_b.normal_cache_ref()[0]
                .as_ref()
                .unwrap()
                .current_seq_len(),
            7
        );

        Ok(())
    }
}
