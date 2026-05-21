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
    sliding_window: Option<usize>,
}

impl NormalSpeculativeCacheAccess {
    pub fn new(cache: Arc<std::sync::Mutex<NormalCache>>, seqs: &[&mut Sequence]) -> Result<Self> {
        let sliding_window = {
            let cache_guard = cache.lock().unwrap();
            normal_cache_sliding_window(&cache_guard)
        };
        Ok(Self {
            cache,
            seq_ids: seqs.iter().map(|seq| *seq.id()).collect(),
            sliding_window,
        })
    }

    pub fn clear_unsupported_staged_tokens(seqs: &mut [&mut Sequence]) {
        let Some(staged_len) = crate::speculative::staging::staged_batch_width(seqs) else {
            return;
        };
        let verify_len = staged_len + 1;
        if seqs.iter().all(|seq| {
            let Some(base_len) = seq.get_toks().len().checked_sub(1) else {
                return false;
            };
            normal_sequence_cache_can_roll_back_after_append(seq, base_len, verify_len)
        }) {
            return;
        }
        for seq in seqs.iter_mut() {
            seq.clear_staged_speculative_tokens();
        }
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
    reserved_len: usize,
    row_idx: usize,
    batch_len: usize,
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
        let cache = self.cache.lock().unwrap();
        if !normal_cache_can_roll_back_after_append(&cache, base_len, verify_len) {
            return Ok(None);
        }
        drop(cache);

        Ok(Some(NormalSpeculativeCacheGuard {
            cache: Arc::clone(&self.cache),
            reserved_len: base_len + verify_len,
            row_idx: self.row_for_seq(seq_id),
            batch_len: self.seq_ids.len().max(1),
        }))
    }

    fn guard_for_reserved(&self, seq_id: usize, base_len: usize, verify_len: usize) -> Self::Guard {
        NormalSpeculativeCacheGuard {
            cache: Arc::clone(&self.cache),
            reserved_len: base_len + verify_len,
            row_idx: self.row_for_seq(seq_id),
            batch_len: self.seq_ids.len().max(1),
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
        let cache = guard.cache.lock().unwrap();
        copy_normal_cache_row_to_sequence(&cache, seq, guard.row_idx, guard.batch_len, keep_len)
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
                    normal_sequence_cache_can_roll_back_after_append(
                        seq,
                        *base_len,
                        proposal_len + 1,
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

fn copy_normal_cache_row_to_sequence(
    cache: &NormalCache,
    seq: &mut Sequence,
    row_idx: usize,
    batch_len: usize,
    keep_len: usize,
) -> Result<()> {
    for layer_idx in 0..cache.0.len() {
        let cache_layer = &cache.0[layer_idx];
        let output_cache = seq.normal_cache();
        if matches!(cache_layer, KvCache::Shared { .. }) {
            if let KvCache::Shared { owner } = cache_layer {
                output_cache[layer_idx] = Some(KvCache::Shared { owner: *owner });
            }
            continue;
        }

        let (row_k, row_v) = match cache_layer {
            KvCache::Normal { k, v } => {
                let Some(k_data) = k.all_data.as_ref() else {
                    continue;
                };
                let Some(v_data) = v.all_data.as_ref() else {
                    continue;
                };
                (
                    narrow_cache_batch_row(k_data, row_idx, batch_len)?,
                    narrow_cache_batch_row(v_data, row_idx, batch_len)?,
                )
            }
            KvCache::Rotating { k, v } => {
                let Some(k_data) = k.all_data.as_ref() else {
                    continue;
                };
                let Some(v_data) = v.all_data.as_ref() else {
                    continue;
                };
                (
                    narrow_cache_batch_row(k_data, row_idx, batch_len)?,
                    narrow_cache_batch_row(v_data, row_idx, batch_len)?,
                )
            }
            KvCache::Shared { .. } => unreachable!(),
        };

        match cache_layer {
            KvCache::Normal { k, v } => {
                output_cache[layer_idx] = Some(KvCache::Normal {
                    k: crate::kv_cache::SingleCache {
                        all_data: Some(row_k),
                        dim: k.dim,
                        current_seq_len: keep_len,
                        max_seq_len: k.max_seq_len,
                        capacity_seq_len: k.capacity_seq_len,
                    },
                    v: crate::kv_cache::SingleCache {
                        all_data: Some(row_v),
                        dim: v.dim,
                        current_seq_len: keep_len,
                        max_seq_len: v.max_seq_len,
                        capacity_seq_len: v.capacity_seq_len,
                    },
                });
            }
            KvCache::Rotating { k, v } => {
                output_cache[layer_idx] = Some(KvCache::Rotating {
                    k: crate::kv_cache::RotatingCache {
                        all_data: Some(row_k),
                        dim: k.dim,
                        current_seq_len: keep_len,
                        max_seq_len: k.max_seq_len,
                        capacity_seq_len: k.capacity_seq_len,
                        last_append_result: None,
                    },
                    v: crate::kv_cache::RotatingCache {
                        all_data: Some(row_v),
                        dim: v.dim,
                        current_seq_len: keep_len,
                        max_seq_len: v.max_seq_len,
                        capacity_seq_len: v.capacity_seq_len,
                        last_append_result: None,
                    },
                });
            }
            KvCache::Shared { .. } => unreachable!(),
        }
    }

    Ok(())
}

fn narrow_cache_batch_row(tensor: &Tensor, row_idx: usize, batch_len: usize) -> Result<Tensor> {
    let dim0 = tensor.dim(0)?;
    if batch_len == 0 || dim0 % batch_len != 0 {
        candle_core::bail!("normal-cache batch shape mismatch: dim0={dim0}, batch_len={batch_len}");
    }
    let per_row = dim0 / batch_len;
    tensor.narrow(0, row_idx * per_row, per_row)?.contiguous()
}

fn normal_sequence_cache_can_roll_back_after_append(
    seq: &Sequence,
    base_len: usize,
    verify_len: usize,
) -> bool {
    let reserved_len = base_len + verify_len;
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
                && k.can_roll_back_after_append(verify_len)
                && v.can_roll_back_after_append(verify_len)
        }
        Some(KvCache::Shared { .. }) | None => true,
    })
}

fn normal_cache_can_roll_back_after_append(
    cache: &NormalCache,
    base_len: usize,
    verify_len: usize,
) -> bool {
    let reserved_len = base_len + verify_len;
    cache.0.iter().all(|layer| match layer {
        KvCache::Normal { k, v } => {
            k.current_seq_len() == base_len
                && v.current_seq_len() == base_len
                && reserved_len <= k.max_seq_len()
                && reserved_len <= v.max_seq_len()
                && k.try_set_len(base_len).is_ok()
                && v.try_set_len(base_len).is_ok()
        }
        KvCache::Rotating { k, v } => {
            k.current_seq_len() == base_len
                && v.current_seq_len() == base_len
                && k.can_roll_back_after_append(verify_len)
                && v.can_roll_back_after_append(verify_len)
        }
        KvCache::Shared { .. } => true,
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
