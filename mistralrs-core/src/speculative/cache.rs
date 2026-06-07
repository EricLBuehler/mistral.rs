use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};

use crate::device_map::DeviceMapper;
use crate::paged_attention::CacheEngine;
use crate::pipeline::text_models_inputs_processor::{
    FlashParams, InputMetadata, PagedAttentionInputMetadata, PagedAttentionMeta,
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
            full_context_lens.push(usize_to_u32(full_context_len, "full context length")?);

            if let Some(sliding_window) = self.metadata.sliding_window {
                let window_start = full_context_len.saturating_sub(sliding_window);
                let slide_idx = window_start / self.metadata.block_size;
                let block_aligned_start = slide_idx * self.metadata.block_size;
                let context_len = full_context_len.saturating_sub(block_aligned_start);
                let needed_blocks = context_len.div_ceil(self.metadata.block_size);
                let slide_end = (slide_idx + needed_blocks).min(full_table.len());
                block_tables.push(full_table.get(slide_idx..slide_end).unwrap_or(&[]).to_vec());
                context_lens.push(usize_to_u32(context_len, "context length")?);
            } else {
                block_tables.push(full_table.clone());
                context_lens.push(usize_to_u32(full_context_len, "context length")?);
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

        let metadata = InputMetadata {
            input,
            positions: vec![base_len],
            context_lens: vec![(0, verify_len)],
            position_ids: vec![base_len + verify_len],
            paged_attn_meta: Some(PagedAttentionInputMetadata {
                block_tables: Some(map_to_devices(&block_tables, device, mapper)?),
                context_lens: Some(map_to_devices(&context_lens, device, mapper)?),
                block_size: Some(self.metadata.block_size),
                paged_context_lens_cpu: None,
                full_paged_context_lens_cpu: None,
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
                prompt_chunk_attention_policy:
                    crate::paged_attention::block_hash::MultimodalAttentionPolicy::Causal,
                has_noncausal_mm_context: false,
                disable_cuda_graphs: false,
                prefill_attention_heads: 1,
                prefill_key_value_heads: 1,
                prefill_head_dim: 1,
                flashinfer: None,
                rope_positions: None,
                num_cached_tokens: None,
                query_lens: None,
                cu_seqlens_q: None,
                cu_seqlens_kv: None,
            }),
            flash_meta: FlashParams::empty(true),
        };
        Ok(metadata)
    }

    fn proposer_cache(&self, _sequences: &[&Sequence]) -> Result<SpeculativeKvCache<'_>> {
        Ok(SpeculativeKvCache::Paged {
            metadata: self.metadata,
            kv_cache: &self.kv_cache,
        })
    }
}

fn repeated_table_tensor(rows: &[Vec<usize>], max_len: usize, device: &Device) -> Result<Tensor> {
    let mut values = Vec::with_capacity(rows.len() * max_len);
    for row in rows {
        for value in row {
            values.push(usize_to_u32(*value, "block table entry")?);
        }
        values.extend(std::iter::repeat_n(0u32, max_len.saturating_sub(row.len())));
    }
    Tensor::from_vec(values, (rows.len(), max_len), device)
}

fn usize_to_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| candle_core::Error::Msg(format!("{name} exceeds u32::MAX: {value}")))
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
