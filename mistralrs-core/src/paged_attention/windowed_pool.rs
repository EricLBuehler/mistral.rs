use std::{
    collections::{HashMap, HashSet},
    ops::Range,
};

use candle_core::{DType, Device, IndexOp, Result, Tensor};

use super::SUPPORTED_BLOCK_SIZE;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowedKvPoolConfig {
    sequence_capacity: usize,
    layer_windows: Vec<usize>,
    max_window: usize,
    max_query_len: usize,
    page_size: usize,
    pages_per_sequence: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl WindowedKvPoolConfig {
    pub fn new(
        sequence_capacity: usize,
        layer_windows: Vec<Option<usize>>,
        max_query_len: usize,
        page_size: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        if sequence_capacity == 0 {
            candle_core::bail!("windowed KV pool sequence capacity must be nonzero");
        }
        if layer_windows.is_empty() {
            candle_core::bail!("windowed KV pool requires at least one layer");
        }
        if max_query_len == 0 {
            candle_core::bail!("windowed KV pool maximum query length must be nonzero");
        }
        if !SUPPORTED_BLOCK_SIZE.contains(&page_size) {
            candle_core::bail!(
                "windowed KV pool page size must be in {SUPPORTED_BLOCK_SIZE:?}, got {page_size}"
            );
        }
        if num_kv_heads == 0 || head_dim == 0 {
            candle_core::bail!("windowed KV pool head count and head dimension must be nonzero");
        }

        let layer_windows = layer_windows
            .into_iter()
            .enumerate()
            .map(|(layer, window)| match window {
                Some(0) => {
                    candle_core::bail!("windowed KV pool layer {layer} has a zero attention window")
                }
                Some(window) => Ok(window),
                None => {
                    candle_core::bail!("windowed KV pool cannot store full-attention layer {layer}")
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let max_window = layer_windows.iter().copied().max().unwrap_or(0);
        let retained_tokens = max_window
            .checked_sub(1)
            .and_then(|value| value.checked_add(max_query_len))
            .and_then(|value| value.checked_add(page_size - 1))
            .ok_or_else(|| candle_core::Error::msg("windowed KV pool token capacity overflow"))?;
        let pages_per_sequence = retained_tokens.div_ceil(page_size);
        let physical_blocks = sequence_capacity
            .checked_mul(pages_per_sequence)
            .ok_or_else(|| candle_core::Error::msg("windowed KV pool block count overflow"))?;
        let total_slots = physical_blocks
            .checked_mul(page_size)
            .ok_or_else(|| candle_core::Error::msg("windowed KV pool slot count overflow"))?;
        u32::try_from(physical_blocks).map_err(|_| {
            candle_core::Error::msg("windowed KV pool block count exceeds u32::MAX")
        })?;
        i64::try_from(total_slots)
            .map_err(|_| candle_core::Error::msg("windowed KV pool slot count exceeds i64::MAX"))?;
        layer_windows
            .len()
            .checked_mul(physical_blocks)
            .and_then(|value| value.checked_mul(num_kv_heads))
            .and_then(|value| value.checked_mul(page_size))
            .and_then(|value| value.checked_mul(head_dim))
            .ok_or_else(|| candle_core::Error::msg("windowed KV pool tensor size overflow"))?;

        Ok(Self {
            sequence_capacity,
            layer_windows,
            max_window,
            max_query_len,
            page_size,
            pages_per_sequence,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn sequence_capacity(&self) -> usize {
        self.sequence_capacity
    }

    pub fn num_layers(&self) -> usize {
        self.layer_windows.len()
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn pages_per_sequence(&self) -> usize {
        self.pages_per_sequence
    }

    pub fn physical_blocks(&self) -> usize {
        self.sequence_capacity * self.pages_per_sequence
    }

    #[cfg(any(
        test,
        all(feature = "cuda", feature = "flash-attn", target_family = "unix")
    ))]
    pub(crate) fn graph_max_kv_len(&self) -> usize {
        self.pages_per_sequence * self.page_size
    }

    pub fn cache_shape(&self) -> [usize; 5] {
        [
            self.num_layers(),
            self.physical_blocks(),
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
        ]
    }

    fn origin_for(&self, next_committed_pos: usize) -> usize {
        let oldest = next_committed_pos.saturating_sub(self.max_window - 1);
        oldest / self.page_size * self.page_size
    }

    fn physical_block(&self, pool_slot: usize, logical_page: usize) -> usize {
        pool_slot * self.pages_per_sequence + logical_page % self.pages_per_sequence
    }

    fn physical_slot(&self, pool_slot: usize, logical_pos: usize) -> i64 {
        let logical_page = logical_pos / self.page_size;
        let page_offset = logical_pos % self.page_size;
        let physical_block = self.physical_block(pool_slot, logical_page);
        i64::try_from(physical_block * self.page_size + page_offset)
            .expect("windowed KV slot count was validated at construction")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowedKvSequenceState {
    pub seq_id: usize,
    pub pool_slot: usize,
    pub origin_pos: usize,
    pub valid_start_pos: usize,
    pub next_committed_pos: usize,
    generation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowedKvContextWrite {
    seq_id: usize,
    pool_slot: usize,
    generation: u64,
    expected_next_committed_pos: usize,
    next_committed_pos: usize,
    origin_pos: usize,
    input_token_count: usize,
    retained_input_offset: usize,
    slot_mapping: Vec<i64>,
}

impl WindowedKvContextWrite {
    #[cfg(test)]
    pub fn origin_pos(&self) -> usize {
        self.origin_pos
    }

    pub fn retained_input_range(&self) -> Range<usize> {
        self.retained_input_offset..self.input_token_count
    }

    pub fn slot_mapping(&self) -> &[i64] {
        &self.slot_mapping
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowedKvQuery {
    pub seq_id: usize,
    pub query_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowedKvBatchRow {
    pub seq_id: usize,
    pub pool_slot: usize,
    pub origin_pos: usize,
    pub next_committed_pos: usize,
    pub query_len: usize,
    pub kv_len: usize,
    pub block_table: Vec<u32>,
    pub slot_mapping: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowedKvBatch {
    rows: Vec<WindowedKvBatchRow>,
    block_table_width: usize,
    block_tables: Vec<u32>,
    slot_mapping: Vec<i64>,
    query_lens: Vec<usize>,
    kv_lens: Vec<usize>,
    cumulative_query_lens: Vec<u32>,
    cumulative_kv_lens: Vec<u32>,
    max_query_len: usize,
    max_kv_len: usize,
}

impl WindowedKvBatch {
    #[cfg(test)]
    pub fn rows(&self) -> &[WindowedKvBatchRow] {
        &self.rows
    }

    #[cfg(test)]
    pub fn block_table_width(&self) -> usize {
        self.block_table_width
    }

    #[cfg(test)]
    pub fn block_tables(&self) -> &[u32] {
        &self.block_tables
    }

    #[cfg(test)]
    pub fn slot_mapping(&self) -> &[i64] {
        &self.slot_mapping
    }

    #[cfg(test)]
    pub fn query_lens(&self) -> &[usize] {
        &self.query_lens
    }

    #[cfg(test)]
    pub fn kv_lens(&self) -> &[usize] {
        &self.kv_lens
    }

    #[cfg(test)]
    pub fn cumulative_query_lens(&self) -> &[u32] {
        &self.cumulative_query_lens
    }

    #[cfg(test)]
    pub fn cumulative_kv_lens(&self) -> &[u32] {
        &self.cumulative_kv_lens
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    pub fn max_query_len(&self) -> usize {
        self.max_query_len
    }

    #[cfg(any(
        test,
        all(feature = "cuda", feature = "flash-attn", target_family = "unix")
    ))]
    pub fn max_kv_len(&self) -> usize {
        self.max_kv_len
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    pub(crate) fn block_table_width_for_graph(&self) -> usize {
        self.block_table_width
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    pub(crate) fn block_tables_for_graph(&self) -> &[u32] {
        &self.block_tables
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    pub(crate) fn slot_mapping_for_graph(&self) -> &[i64] {
        &self.slot_mapping
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    pub(crate) fn cumulative_query_lens_for_graph(&self) -> &[u32] {
        &self.cumulative_query_lens
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    pub(crate) fn cumulative_kv_lens_for_graph(&self) -> &[u32] {
        &self.cumulative_kv_lens
    }

    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    pub fn to_tensors(&self, device: &Device) -> Result<WindowedKvBatchTensors> {
        let batch = self.rows.len();
        Ok(WindowedKvBatchTensors {
            block_tables: Tensor::from_vec(
                self.block_tables.clone(),
                (batch, self.block_table_width),
                device,
            )?,
            slot_mapping: Tensor::from_vec(
                self.slot_mapping.clone(),
                (self.slot_mapping.len(),),
                device,
            )?,
            cumulative_query_lens: Tensor::from_vec(
                self.cumulative_query_lens.clone(),
                (batch + 1,),
                device,
            )?,
            cumulative_kv_lens: Tensor::from_vec(
                self.cumulative_kv_lens.clone(),
                (batch + 1,),
                device,
            )?,
        })
    }
}

#[derive(Debug)]
#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
pub struct WindowedKvBatchTensors {
    pub block_tables: Tensor,
    pub slot_mapping: Tensor,
    pub cumulative_query_lens: Tensor,
    pub cumulative_kv_lens: Tensor,
}

#[derive(Debug)]
pub struct WindowedKvPool {
    config: WindowedKvPoolConfig,
    key_cache: Tensor,
    value_cache: Tensor,
    free_slots: Vec<usize>,
    sequences: HashMap<usize, WindowedKvSequenceState>,
    next_generation: u64,
}

impl WindowedKvPool {
    pub fn new(config: WindowedKvPoolConfig, device: &Device) -> Result<Self> {
        let shape = config.cache_shape();
        let key_cache = Tensor::zeros(&shape, DType::BF16, device)?;
        let value_cache = Tensor::zeros(&shape, DType::BF16, device)?;
        Self::from_tensors(config, key_cache, value_cache)
    }

    pub fn from_tensors(
        config: WindowedKvPoolConfig,
        key_cache: Tensor,
        value_cache: Tensor,
    ) -> Result<Self> {
        let expected_shape = config.cache_shape();
        if key_cache.dims() != expected_shape || value_cache.dims() != expected_shape {
            candle_core::bail!(
                "windowed KV pool cache shape mismatch: expected {expected_shape:?}, got key={:?}, value={:?}",
                key_cache.dims(),
                value_cache.dims()
            );
        }
        if key_cache.dtype() != DType::BF16 || value_cache.dtype() != DType::BF16 {
            candle_core::bail!(
                "windowed KV pool caches must be BF16, got key={:?}, value={:?}",
                key_cache.dtype(),
                value_cache.dtype()
            );
        }
        if !key_cache.device().same_device(value_cache.device()) {
            candle_core::bail!("windowed KV pool K/V caches must be on the same device");
        }
        if !key_cache.is_contiguous() || !value_cache.is_contiguous() {
            candle_core::bail!("windowed KV pool K/V caches must be contiguous");
        }

        let free_slots = (0..config.sequence_capacity()).rev().collect();
        Ok(Self {
            config,
            key_cache,
            value_cache,
            free_slots,
            sequences: HashMap::new(),
            next_generation: 0,
        })
    }

    pub fn config(&self) -> &WindowedKvPoolConfig {
        &self.config
    }

    #[cfg(test)]
    pub fn dtype(&self) -> DType {
        DType::BF16
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    #[cfg(test)]
    pub fn free_capacity(&self) -> usize {
        self.free_slots.len()
    }

    pub fn sequence(&self, seq_id: usize) -> Option<WindowedKvSequenceState> {
        self.sequences.get(&seq_id).copied()
    }

    #[cfg(test)]
    pub fn acquire(&mut self, seq_id: usize) -> Result<WindowedKvSequenceState> {
        if let Some(state) = self.sequence(seq_id) {
            return Ok(state);
        }
        self.acquire_at(seq_id, 0)
    }

    pub fn acquire_at(
        &mut self,
        seq_id: usize,
        next_committed_pos: usize,
    ) -> Result<WindowedKvSequenceState> {
        if let Some(state) = self.sequence(seq_id) {
            if state.next_committed_pos != next_committed_pos {
                candle_core::bail!(
                    "windowed KV sequence {seq_id} is at position {}, expected {next_committed_pos}",
                    state.next_committed_pos
                );
            }
            return Ok(state);
        }
        let pool_slot = self.free_slots.pop().ok_or_else(|| {
            candle_core::Error::msg(format!(
                "windowed KV pool capacity {} exhausted",
                self.config.sequence_capacity()
            ))
        })?;
        let generation = self.next_generation;
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .ok_or_else(|| candle_core::Error::msg("windowed KV pool generation overflow"))?;
        let state = WindowedKvSequenceState {
            seq_id,
            pool_slot,
            origin_pos: self.config.origin_for(next_committed_pos),
            valid_start_pos: next_committed_pos,
            next_committed_pos,
            generation,
        };
        self.sequences.insert(seq_id, state);
        Ok(state)
    }

    pub fn release(&mut self, seq_id: usize) -> bool {
        let Some(state) = self.sequences.remove(&seq_id) else {
            return false;
        };
        self.free_slots.push(state.pool_slot);
        true
    }

    pub fn clear(&mut self) {
        self.sequences.clear();
        self.free_slots = (0..self.config.sequence_capacity()).rev().collect();
    }

    pub fn layer_cache(&self, layer: usize) -> Result<(Tensor, Tensor)> {
        if layer >= self.config.num_layers() {
            candle_core::bail!(
                "windowed KV pool layer index {layer} exceeds {} layers",
                self.config.num_layers()
            );
        }
        Ok((self.key_cache.i(layer)?, self.value_cache.i(layer)?))
    }

    pub fn paged_attention_layer_cache(&self, layer: usize) -> Result<(Tensor, Tensor)> {
        let (key, value) = self.layer_cache(layer)?;
        Ok((key.transpose(1, 2)?, value.transpose(1, 2)?))
    }

    pub fn plan_context_write(
        &self,
        seq_id: usize,
        input_token_count: usize,
    ) -> Result<WindowedKvContextWrite> {
        if input_token_count == 0 {
            candle_core::bail!("windowed KV context write must contain at least one token");
        }
        let state = self.sequences.get(&seq_id).ok_or_else(|| {
            candle_core::Error::msg(format!("windowed KV sequence {seq_id} is not acquired"))
        })?;
        let next_committed_pos = state
            .next_committed_pos
            .checked_add(input_token_count)
            .ok_or_else(|| candle_core::Error::msg("windowed KV context position overflow"))?;
        let origin_pos = self.config.origin_for(next_committed_pos);
        let retained_start = state.next_committed_pos.max(origin_pos);
        let retained_input_offset = retained_start - state.next_committed_pos;
        let slot_mapping = (retained_start..next_committed_pos)
            .map(|pos| self.config.physical_slot(state.pool_slot, pos))
            .collect();
        Ok(WindowedKvContextWrite {
            seq_id,
            pool_slot: state.pool_slot,
            generation: state.generation,
            expected_next_committed_pos: state.next_committed_pos,
            next_committed_pos,
            origin_pos,
            input_token_count,
            retained_input_offset,
            slot_mapping,
        })
    }

    pub fn commit_context(&mut self, write: &WindowedKvContextWrite) -> Result<()> {
        let state = self.sequences.get_mut(&write.seq_id).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "windowed KV sequence {} was released before context commit",
                write.seq_id
            ))
        })?;
        if state.pool_slot != write.pool_slot
            || state.generation != write.generation
            || state.next_committed_pos != write.expected_next_committed_pos
        {
            candle_core::bail!(
                "windowed KV context write for sequence {} is stale",
                write.seq_id
            );
        }
        state.origin_pos = write.origin_pos;
        state.valid_start_pos = state.valid_start_pos.max(write.origin_pos);
        state.next_committed_pos = write.next_committed_pos;
        Ok(())
    }

    pub fn sequence_query_ready(&self, seq_id: usize) -> bool {
        self.sequences
            .get(&seq_id)
            .is_some_and(|state| state.origin_pos >= state.valid_start_pos)
    }

    pub fn scratch_batch(&self, queries: &[WindowedKvQuery]) -> Result<WindowedKvBatch> {
        let rows = self.scratch_rows(queries)?;
        let block_table_width = rows
            .iter()
            .map(|row| row.block_table.len())
            .max()
            .unwrap_or(1)
            .max(1);
        Self::batch_from_rows(rows, block_table_width, None)
    }

    #[cfg(any(
        test,
        all(feature = "cuda", feature = "flash-attn", target_family = "unix")
    ))]
    pub(crate) fn scratch_graph_batch(
        &self,
        queries: &[WindowedKvQuery],
        batch_bucket: usize,
    ) -> Result<WindowedKvBatch> {
        let mut rows = self.scratch_rows(queries)?;
        if batch_bucket < rows.len() {
            candle_core::bail!(
                "windowed KV graph bucket {batch_bucket} is smaller than batch {}",
                rows.len()
            );
        }
        let pad_row = rows[0].clone();
        while rows.len() < batch_bucket {
            rows.push(WindowedKvBatchRow {
                slot_mapping: vec![super::_PAD_SLOT_ID; pad_row.query_len],
                ..pad_row.clone()
            });
        }
        Self::batch_from_rows(
            rows,
            self.config.pages_per_sequence(),
            Some(self.config.graph_max_kv_len()),
        )
    }

    fn scratch_rows(&self, queries: &[WindowedKvQuery]) -> Result<Vec<WindowedKvBatchRow>> {
        if queries.is_empty() {
            candle_core::bail!("windowed KV scratch batch must contain at least one sequence");
        }
        let mut seen = HashSet::with_capacity(queries.len());
        let mut rows = Vec::with_capacity(queries.len());
        for query in queries {
            if !seen.insert(query.seq_id) {
                candle_core::bail!(
                    "windowed KV scratch batch contains sequence {} more than once",
                    query.seq_id
                );
            }
            rows.push(self.scratch_row(*query)?);
        }
        Ok(rows)
    }

    fn batch_from_rows(
        rows: Vec<WindowedKvBatchRow>,
        block_table_width: usize,
        max_kv_len: Option<usize>,
    ) -> Result<WindowedKvBatch> {
        if rows
            .iter()
            .any(|row| row.block_table.len() > block_table_width)
        {
            candle_core::bail!("windowed KV block table exceeds fixed width {block_table_width}");
        }
        let mut block_tables = Vec::with_capacity(rows.len() * block_table_width);
        let mut slot_mapping = Vec::new();
        let mut query_lens = Vec::with_capacity(rows.len());
        let mut kv_lens = Vec::with_capacity(rows.len());
        let mut cumulative_query_lens = Vec::with_capacity(rows.len() + 1);
        let mut cumulative_kv_lens = Vec::with_capacity(rows.len() + 1);
        cumulative_query_lens.push(0);
        cumulative_kv_lens.push(0);

        for row in &rows {
            block_tables.extend_from_slice(&row.block_table);
            block_tables.extend(std::iter::repeat_n(
                0,
                block_table_width - row.block_table.len(),
            ));
            slot_mapping.extend_from_slice(&row.slot_mapping);
            query_lens.push(row.query_len);
            kv_lens.push(row.kv_len);
            push_cumulative_len(
                &mut cumulative_query_lens,
                row.query_len,
                "windowed KV cumulative query length",
            )?;
            push_cumulative_len(
                &mut cumulative_kv_lens,
                row.kv_len,
                "windowed KV cumulative context length",
            )?;
        }

        Ok(WindowedKvBatch {
            max_query_len: query_lens.iter().copied().max().unwrap_or(0),
            max_kv_len: max_kv_len.unwrap_or_else(|| kv_lens.iter().copied().max().unwrap_or(0)),
            rows,
            block_table_width,
            block_tables,
            slot_mapping,
            query_lens,
            kv_lens,
            cumulative_query_lens,
            cumulative_kv_lens,
        })
    }

    fn scratch_row(&self, query: WindowedKvQuery) -> Result<WindowedKvBatchRow> {
        if query.query_len == 0 || query.query_len > self.config.max_query_len {
            candle_core::bail!(
                "windowed KV scratch query length must be in 1..={}, got {}",
                self.config.max_query_len,
                query.query_len
            );
        }
        let state = self.sequences.get(&query.seq_id).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "windowed KV sequence {} is not acquired",
                query.seq_id
            ))
        })?;
        if state.origin_pos < state.valid_start_pos {
            candle_core::bail!(
                "windowed KV sequence {} is initialized from position {}, but its query context starts at {}",
                query.seq_id,
                state.valid_start_pos,
                state.origin_pos
            );
        }
        let end_pos = state
            .next_committed_pos
            .checked_add(query.query_len)
            .ok_or_else(|| candle_core::Error::msg("windowed KV scratch position overflow"))?;
        let first_page = state.origin_pos / self.config.page_size;
        let last_page = end_pos
            .checked_sub(1)
            .ok_or_else(|| candle_core::Error::msg("windowed KV scratch range is empty"))?
            / self.config.page_size;
        let block_table = (first_page..=last_page)
            .map(|logical_page| {
                u32::try_from(self.config.physical_block(state.pool_slot, logical_page)).map_err(
                    |_| candle_core::Error::msg("windowed KV block index exceeds u32::MAX"),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if block_table.len() > self.config.pages_per_sequence {
            candle_core::bail!("windowed KV scratch range exceeds its sequence page ring");
        }
        let slot_mapping = (state.next_committed_pos..end_pos)
            .map(|pos| self.config.physical_slot(state.pool_slot, pos))
            .collect();

        Ok(WindowedKvBatchRow {
            seq_id: query.seq_id,
            pool_slot: state.pool_slot,
            origin_pos: state.origin_pos,
            next_committed_pos: state.next_committed_pos,
            query_len: query.query_len,
            kv_len: end_pos - state.origin_pos,
            block_table,
            slot_mapping,
        })
    }
}

fn push_cumulative_len(values: &mut Vec<u32>, len: usize, name: &str) -> Result<()> {
    let previous = values.last().copied().unwrap_or(0);
    let len = u32::try_from(len)
        .map_err(|_| candle_core::Error::msg(format!("{name} exceeds u32::MAX")))?;
    values.push(
        previous
            .checked_add(len)
            .ok_or_else(|| candle_core::Error::msg(format!("{name} overflow")))?,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(sequence_capacity: usize) -> Result<WindowedKvPoolConfig> {
        WindowedKvPoolConfig::new(sequence_capacity, vec![Some(8), Some(6)], 3, 8, 2, 4)
    }

    fn metadata_pool(sequence_capacity: usize) -> Result<WindowedKvPool> {
        WindowedKvPool::new(config(sequence_capacity)?, &Device::Cpu)
    }

    #[test]
    fn dflash_capacity_includes_window_query_and_alignment_slack() -> Result<()> {
        let config = WindowedKvPoolConfig::new(33, vec![Some(2048); 5], 8, 32, 8, 128)?;
        assert_eq!(config.pages_per_sequence(), 66);
        assert_eq!(config.physical_blocks(), 2178);
        assert_eq!(config.cache_shape(), [5, 2178, 8, 32, 128]);
        Ok(())
    }

    #[test]
    fn circular_pages_wrap_in_logical_order() -> Result<()> {
        let mut pool = metadata_pool(1)?;
        pool.acquire(7)?;
        let write = pool.plan_context_write(7, 30)?;
        assert_eq!(write.origin_pos(), 16);
        assert_eq!(write.retained_input_range(), 16..30);
        pool.commit_context(&write)?;

        let batch = pool.scratch_batch(&[WindowedKvQuery {
            seq_id: 7,
            query_len: 3,
        }])?;
        let row = &batch.rows()[0];
        assert_eq!(row.block_table, vec![2, 0, 1]);
        assert_eq!(row.slot_mapping, vec![6, 7, 8]);
        assert_eq!(row.kv_len, 17);
        assert_eq!(row.origin_pos % pool.config().page_size(), 0);
        assert!(row.block_table.len() <= pool.config().pages_per_sequence());
        Ok(())
    }

    #[test]
    fn rejected_scratch_does_not_advance_and_acceptance_reuses_slots() -> Result<()> {
        let mut pool = metadata_pool(1)?;
        pool.acquire(11)?;
        let prefill = pool.plan_context_write(11, 10)?;
        pool.commit_context(&prefill)?;

        let proposed = pool.scratch_batch(&[WindowedKvQuery {
            seq_id: 11,
            query_len: 3,
        }])?;
        assert_eq!(pool.sequence(11).unwrap().next_committed_pos, 10);

        let retried = pool.scratch_batch(&[WindowedKvQuery {
            seq_id: 11,
            query_len: 1,
        }])?;
        assert_eq!(retried.rows()[0].kv_len, proposed.rows()[0].kv_len - 2);
        assert_eq!(
            retried.rows()[0].slot_mapping,
            proposed.rows()[0].slot_mapping[..1]
        );

        let accepted = pool.plan_context_write(11, 2)?;
        assert_eq!(
            accepted.slot_mapping(),
            &proposed.rows()[0].slot_mapping[..2]
        );
        pool.commit_context(&accepted)?;
        assert_eq!(pool.sequence(11).unwrap().next_committed_pos, 12);
        Ok(())
    }

    #[test]
    fn mixed_batch_has_independent_tables_and_lengths() -> Result<()> {
        let mut pool = metadata_pool(2)?;
        pool.acquire(1)?;
        pool.acquire(2)?;
        let first = pool.plan_context_write(1, 2)?;
        pool.commit_context(&first)?;
        let second = pool.plan_context_write(2, 11)?;
        pool.commit_context(&second)?;

        let batch = pool.scratch_batch(&[
            WindowedKvQuery {
                seq_id: 1,
                query_len: 1,
            },
            WindowedKvQuery {
                seq_id: 2,
                query_len: 3,
            },
        ])?;
        assert_eq!(batch.query_lens(), &[1, 3]);
        assert_eq!(batch.kv_lens(), &[3, 14]);
        assert_eq!(batch.cumulative_query_lens(), &[0, 1, 4]);
        assert_eq!(batch.cumulative_kv_lens(), &[0, 3, 17]);
        assert_eq!(batch.block_table_width(), 2);
        assert_eq!(batch.block_tables(), &[0, 0, 3, 4]);
        assert_eq!(batch.slot_mapping().len(), 4);
        Ok(())
    }

    #[test]
    fn graph_batch_has_fixed_tables_and_inert_pad_rows() -> Result<()> {
        let mut pool = metadata_pool(2)?;
        pool.acquire(1)?;
        let write = pool.plan_context_write(1, 11)?;
        pool.commit_context(&write)?;

        let batch = pool.scratch_graph_batch(
            &[WindowedKvQuery {
                seq_id: 1,
                query_len: 3,
            }],
            4,
        )?;
        assert_eq!(batch.rows().len(), 4);
        assert_eq!(
            batch.block_table_width(),
            pool.config().pages_per_sequence()
        );
        assert_eq!(batch.max_kv_len(), pool.config().graph_max_kv_len());
        assert_eq!(batch.cumulative_query_lens(), &[0, 3, 6, 9, 12]);
        assert_eq!(batch.cumulative_kv_lens(), &[0, 14, 28, 42, 56]);
        assert_eq!(batch.slot_mapping()[..3], [11, 12, 13]);
        assert!(batch.slot_mapping()[3..]
            .iter()
            .all(|slot| *slot == crate::paged_attention::_PAD_SLOT_ID));
        assert_eq!(
            batch.block_tables().len(),
            4 * pool.config().pages_per_sequence()
        );
        Ok(())
    }

    #[test]
    fn sequence_slots_remain_stable_until_release() -> Result<()> {
        let mut pool = metadata_pool(2)?;
        let first = pool.acquire(10)?;
        let second = pool.acquire(20)?;
        assert_eq!(pool.acquire(10)?.pool_slot, first.pool_slot);
        assert!(pool.acquire(30).is_err());

        assert!(pool.release(10));
        assert!(pool.sequence(10).is_none());
        assert_eq!(pool.sequence(20).unwrap().pool_slot, second.pool_slot);
        assert_eq!(pool.acquire(30)?.pool_slot, first.pool_slot);
        assert!(pool.release(30));
        assert!(!pool.release(30));
        pool.clear();
        assert!(pool.is_empty());
        assert_eq!(pool.free_capacity(), 2);
        Ok(())
    }

    #[test]
    fn stale_context_write_cannot_commit_after_slot_reuse() -> Result<()> {
        let mut pool = metadata_pool(1)?;
        pool.acquire(3)?;
        let stale = pool.plan_context_write(3, 1)?;
        pool.release(3);
        pool.acquire(3)?;
        assert!(pool.commit_context(&stale).is_err());
        Ok(())
    }

    #[test]
    fn nonzero_acquire_never_exposes_uninitialized_prefix_pages() -> Result<()> {
        let mut pool = metadata_pool(1)?;
        let state = pool.acquire_at(9, 10)?;
        assert_eq!(state.origin_pos, 0);
        assert_eq!(state.valid_start_pos, 10);

        let suffix = pool.plan_context_write(9, 2)?;
        pool.commit_context(&suffix)?;
        assert!(!pool.sequence_query_ready(9));
        assert!(pool
            .scratch_batch(&[WindowedKvQuery {
                seq_id: 9,
                query_len: 1,
            }])
            .is_err());

        let fill_window = pool.plan_context_write(9, 11)?;
        assert_eq!(fill_window.retained_input_range(), 4..11);
        pool.commit_context(&fill_window)?;
        let state = pool.sequence(9).unwrap();
        assert_eq!(state.origin_pos, 16);
        assert_eq!(state.valid_start_pos, 16);
        assert!(pool.sequence_query_ready(9));

        let batch = pool.scratch_batch(&[WindowedKvQuery {
            seq_id: 9,
            query_len: 1,
        }])?;
        assert_eq!(batch.rows()[0].origin_pos, 16);
        assert_eq!(batch.rows()[0].kv_len, 8);
        Ok(())
    }

    #[test]
    fn validates_bounded_layers_pages_and_query_blocks() {
        assert!(WindowedKvPoolConfig::new(1, vec![], 1, 8, 1, 1).is_err());
        assert!(WindowedKvPoolConfig::new(1, vec![None], 1, 8, 1, 1).is_err());
        assert!(WindowedKvPoolConfig::new(1, vec![Some(0)], 1, 8, 1, 1).is_err());
        assert!(WindowedKvPoolConfig::new(1, vec![Some(8)], 0, 8, 1, 1).is_err());
        assert!(WindowedKvPoolConfig::new(1, vec![Some(8)], 1, 7, 1, 1).is_err());
        assert!(WindowedKvPoolConfig::new(1, vec![Some(usize::MAX)], 2, 8, 1, 1).is_err());
    }

    #[test]
    fn cache_views_match_flashinfer_and_paged_fa_layouts() -> Result<()> {
        let pool = metadata_pool(2)?;
        assert_eq!(pool.dtype(), DType::BF16);
        assert_eq!(pool.key_cache.dims(), pool.config().cache_shape());
        let (key_hnd, value_hnd) = pool.layer_cache(1)?;
        assert_eq!(key_hnd.dims(), &[6, 2, 8, 4]);
        assert_eq!(value_hnd.dims(), key_hnd.dims());
        let (key_nhd, value_nhd) = pool.paged_attention_layer_cache(1)?;
        assert_eq!(key_nhd.dims(), &[6, 8, 2, 4]);
        assert_eq!(value_nhd.dims(), key_nhd.dims());
        Ok(())
    }
}
