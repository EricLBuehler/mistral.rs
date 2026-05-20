use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};

use crate::device_map::DeviceMapper;
use crate::pipeline::text_models_inputs_processor::{
    FlashParams, InputMetadata, PagedAttentionInputMetadata, PagedAttentionMeta,
};

pub trait SpeculativeCacheGuard {
    fn commit(&mut self) -> Result<()>;
    fn rollback_to(&mut self, keep_len: usize) -> Result<()>;
}

pub trait SpeculativeCacheTransaction {
    type Guard: SpeculativeCacheGuard;

    fn begin(
        &self,
        seq_id: usize,
        base_len: usize,
        verify_len: usize,
    ) -> Result<Option<Self::Guard>>;

    fn make_verify_input_metadata(
        &self,
        verify_tokens: &[u32],
        seq_id: usize,
        base_len: usize,
        device: &Device,
        mapper: &dyn DeviceMapper,
    ) -> Result<InputMetadata>;
}

pub struct PagedSpeculativeCacheTransaction<'a> {
    metadata: &'a PagedAttentionMeta,
}

impl<'a> PagedSpeculativeCacheTransaction<'a> {
    pub fn new(metadata: &'a PagedAttentionMeta) -> Self {
        Self { metadata }
    }

    pub fn guard_for_reserved(
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

impl<'a> SpeculativeCacheTransaction for PagedSpeculativeCacheTransaction<'a> {
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
