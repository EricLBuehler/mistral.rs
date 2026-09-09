#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Arc};

use candle_core::{DType, Error, Result, Tensor, D};
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use super::config::Config;
use crate::attention::{Sdpa, SdpaParams};

#[derive(Clone)]
pub(crate) struct QsaSequenceSnapshot {
    raw_keys: Option<Tensor>,
    positions: Vec<u32>,
    cos: Option<Tensor>,
    sin: Option<Tensor>,
    head_dim: usize,
}

type QsaSequenceEntry = (Tensor, Vec<u32>, Option<Tensor>, Option<Tensor>);

#[allow(dead_code)]
pub(crate) struct QsaSequenceCache {
    sequences: HashMap<usize, QsaSequenceEntry>,
    head_dim: usize,
}

#[allow(dead_code)]
impl QsaSequenceCache {
    pub(crate) fn new(head_dim: usize) -> Result<Self> {
        if head_dim == 0 {
            candle_core::bail!("Qwen4Exp QSA cache head dimension must be positive");
        }
        Ok(Self {
            sequences: HashMap::new(),
            head_dim,
        })
    }

    fn validate_append(&self, raw_keys: &Tensor, positions: &[u32]) -> Result<(usize, usize)> {
        let (tokens, width) = raw_keys.dims2()?;
        if width != self.head_dim || positions.len() != tokens {
            candle_core::bail!(
                "Qwen4Exp QSA cache expected raw keys [tokens, {}] and one position per token, got {:?} and {} positions",
                self.head_dim,
                raw_keys.dims(),
                positions.len()
            );
        }
        Ok((tokens, width))
    }

    pub(crate) fn append(
        &mut self,
        sequence_id: usize,
        raw_keys: &Tensor,
        positions: &[u32],
    ) -> Result<()> {
        self.validate_append(raw_keys, positions)?;
        if let Some((cached_keys, cached_positions, cached_cos, cached_sin)) =
            self.sequences.get_mut(&sequence_id)
        {
            if cached_keys.device().location() != raw_keys.device().location()
                || cached_keys.dtype() != raw_keys.dtype()
                || cached_cos.is_some()
                || cached_sin.is_some()
            {
                candle_core::bail!(
                    "Qwen4Exp QSA cache append requires matching key dtype and device, plus position-table mode"
                );
            }
            *cached_keys = Tensor::cat(&[&*cached_keys, raw_keys], 0)?;
            cached_positions.extend_from_slice(positions);
        } else {
            self.sequences.insert(
                sequence_id,
                (raw_keys.clone(), positions.to_vec(), None, None),
            );
        }
        Ok(())
    }

    /// Append raw keys with the already-selected MRoPE tables used to rotate each token.
    pub(crate) fn append_with_position_tables(
        &mut self,
        sequence_id: usize,
        raw_keys: &Tensor,
        positions: &[u32],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<()> {
        let (tokens, _) = self.validate_append(raw_keys, positions)?;
        if cos.dims().len() != 2
            || sin.dims() != cos.dims()
            || cos.dim(0)? != tokens
            || cos.device().location() != raw_keys.device().location()
            || sin.device().location() != raw_keys.device().location()
        {
            candle_core::bail!(
                "Qwen4Exp QSA indexer position tables are incompatible with raw keys"
            );
        }
        if let Some((cached_keys, cached_positions, cached_cos, cached_sin)) =
            self.sequences.get_mut(&sequence_id)
        {
            let (Some(existing_cos), Some(existing_sin)) = (cached_cos, cached_sin) else {
                candle_core::bail!(
                    "Qwen4Exp QSA cache cannot mix scalar and sectioned MRoPE positions"
                );
            };
            if cached_keys.device().location() != raw_keys.device().location()
                || cached_keys.dtype() != raw_keys.dtype()
                || existing_cos.device().location() != cos.device().location()
                || existing_sin.device().location() != sin.device().location()
                || existing_cos.dtype() != cos.dtype()
                || existing_sin.dtype() != sin.dtype()
                || existing_cos.dim(1)? != cos.dim(1)?
            {
                candle_core::bail!("Qwen4Exp QSA cache append requires matching position-table dtype, width, and device");
            }
            *cached_keys = Tensor::cat(&[&*cached_keys, raw_keys], 0)?;
            *existing_cos = Tensor::cat(&[&*existing_cos, cos], 0)?;
            *existing_sin = Tensor::cat(&[&*existing_sin, sin], 0)?;
            cached_positions.extend_from_slice(positions);
        } else {
            self.sequences.insert(
                sequence_id,
                (
                    raw_keys.clone(),
                    positions.to_vec(),
                    Some(cos.clone()),
                    Some(sin.clone()),
                ),
            );
        }
        Ok(())
    }

    pub(crate) fn get(&self, sequence_id: usize) -> Option<(&Tensor, &[u32])> {
        self.sequences
            .get(&sequence_id)
            .map(|(keys, positions, _, _)| (keys, positions.as_slice()))
    }

    fn position_tables(&self, sequence_id: usize) -> Result<(&Tensor, &Tensor)> {
        let Some((_, _, Some(cos), Some(sin))) = self.sequences.get(&sequence_id) else {
            candle_core::bail!("Qwen4Exp QSA sequence has no sectioned MRoPE position tables");
        };
        Ok((cos, sin))
    }

    fn has_position_tables(&self, sequence_id: usize) -> bool {
        self.sequences
            .get(&sequence_id)
            .is_some_and(|(_, _, cos, sin)| cos.is_some() && sin.is_some())
    }

    pub(crate) fn validate_truncate(&self, sequence_id: usize, len: usize) -> Result<()> {
        let Some((keys, positions, cos, sin)) = self.sequences.get(&sequence_id) else {
            if len == 0 {
                return Ok(());
            }
            candle_core::bail!("Qwen4Exp QSA cannot truncate an absent sequence cache");
        };
        if keys.dim(0)? != positions.len()
            || cos
                .as_ref()
                .is_some_and(|table| table.dim(0).ok() != Some(positions.len()))
            || sin
                .as_ref()
                .is_some_and(|table| table.dim(0).ok() != Some(positions.len()))
            || cos.is_some() != sin.is_some()
        {
            candle_core::bail!(
                "Qwen4Exp QSA cache keys and position tables have different lengths"
            );
        }
        if len > positions.len() {
            candle_core::bail!(
                "Qwen4Exp QSA cannot truncate cache of length {} to {len}",
                positions.len()
            );
        }
        Ok(())
    }

    pub(crate) fn truncate(&mut self, sequence_id: usize, len: usize) -> Result<()> {
        self.validate_truncate(sequence_id, len)?;
        let Some((keys, positions, cos, sin)) = self.sequences.get_mut(&sequence_id) else {
            return Ok(());
        };
        *keys = keys.narrow(0, 0, len)?;
        if let Some(table) = cos {
            *table = table.narrow(0, 0, len)?;
        }
        if let Some(table) = sin {
            *table = table.narrow(0, 0, len)?;
        }
        positions.truncate(len);
        Ok(())
    }

    pub(crate) fn snapshot(&self, sequence_id: usize) -> QsaSequenceSnapshot {
        let entry = self.sequences.get(&sequence_id);
        QsaSequenceSnapshot {
            raw_keys: entry.map(|(keys, _, _, _)| keys.clone()),
            positions: entry
                .map(|(_, positions, _, _)| positions.clone())
                .unwrap_or_default(),
            cos: entry.and_then(|(_, _, cos, _)| cos.clone()),
            sin: entry.and_then(|(_, _, _, sin)| sin.clone()),
            head_dim: self.head_dim,
        }
    }

    pub(crate) fn validate_snapshot(&self, snapshot: &QsaSequenceSnapshot) -> Result<()> {
        if snapshot.head_dim != self.head_dim {
            candle_core::bail!("Qwen4Exp QSA snapshot is incompatible with the sequence cache");
        }
        match &snapshot.raw_keys {
            Some(keys) => {
                let (tokens, width) = keys.dims2()?;
                if width != self.head_dim || snapshot.positions.len() != tokens {
                    candle_core::bail!("Qwen4Exp QSA snapshot has an invalid key inventory");
                }
                match (&snapshot.cos, &snapshot.sin) {
                    (Some(cos), Some(sin))
                        if cos.dims().len() == 2
                            && sin.dims() == cos.dims()
                            && cos.dim(0)? == tokens => {}
                    (None, None) => {}
                    _ => candle_core::bail!("Qwen4Exp QSA snapshot has invalid position tables"),
                }
            }
            None if snapshot.positions.is_empty()
                && snapshot.cos.is_none()
                && snapshot.sin.is_none() => {}
            None => candle_core::bail!("Qwen4Exp QSA snapshot has state without raw keys"),
        }
        Ok(())
    }

    pub(crate) fn restore(
        &mut self,
        sequence_id: usize,
        snapshot: &QsaSequenceSnapshot,
    ) -> Result<()> {
        self.validate_snapshot(snapshot)?;
        if let Some(keys) = &snapshot.raw_keys {
            self.sequences.insert(
                sequence_id,
                (
                    keys.clone(),
                    snapshot.positions.clone(),
                    snapshot.cos.clone(),
                    snapshot.sin.clone(),
                ),
            );
        } else {
            self.sequences.remove(&sequence_id);
        }
        Ok(())
    }

    pub(crate) fn release(&mut self, sequence_id: usize) -> bool {
        self.sequences.remove(&sequence_id).is_some()
    }

    pub(crate) fn reset(&mut self, sequence_id: usize) {
        self.release(sequence_id);
    }

    pub(crate) fn clear(&mut self) {
        self.sequences.clear();
    }
}

#[allow(dead_code)]
pub(crate) struct QsaIndexerRotary {
    head_dim: usize,
    rotary_dim: usize,
}

#[allow(dead_code)]
impl QsaIndexerRotary {
    pub(crate) fn new(head_dim: usize, rotary_dim: usize) -> Result<Self> {
        if head_dim == 0
            || rotary_dim == 0
            || rotary_dim > head_dim
            || !rotary_dim.is_multiple_of(2)
        {
            candle_core::bail!(
                "Qwen4Exp QSA rotary dimension must be positive, even, and no larger than the head dimension"
            );
        }
        Ok(Self {
            head_dim,
            rotary_dim,
        })
    }

    fn rotate(&self, input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (rows, width) = input.dims2()?;
        let half = self.rotary_dim / 2;
        if width != self.head_dim || cos.dims() != [rows, half] || sin.dims() != [rows, half] {
            candle_core::bail!(
                "Qwen4Exp QSA rotary expected input [rows, {}] and cos/sin [rows, {half}], got {:?}, {:?}, and {:?}",
                self.head_dim,
                input.dims(),
                cos.dims(),
                sin.dims()
            );
        }
        let cos = cos.to_dtype(input.dtype())?;
        let sin = sin.to_dtype(input.dtype())?;
        let rotary = input.narrow(D::Minus1, 0, self.rotary_dim)?;
        let first = rotary.narrow(D::Minus1, 0, half)?;
        let second = rotary.narrow(D::Minus1, half, half)?;
        let rotated_first = (first.broadcast_mul(&cos)? - second.broadcast_mul(&sin)?)?;
        let rotated_second = (second.broadcast_mul(&cos)? + first.broadcast_mul(&sin)?)?;
        if self.rotary_dim == self.head_dim {
            Tensor::cat(&[&rotated_first, &rotated_second], D::Minus1)
        } else {
            let unrotated =
                input.narrow(D::Minus1, self.rotary_dim, self.head_dim - self.rotary_dim)?;
            Tensor::cat(&[&rotated_first, &rotated_second, &unrotated], D::Minus1)
        }
    }

    /// Apply precomputed MRoPE at the current query position and each block's first position.
    pub(crate) fn apply(
        &self,
        query: &Tensor,
        pooled_keys: &Tensor,
        query_position: u32,
        block_positions: &[u32],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (heads, query_width) = query.dims2()?;
        let (blocks, key_width) = pooled_keys.dims2()?;
        let half = self.rotary_dim / 2;
        if query_width != self.head_dim
            || key_width != self.head_dim
            || block_positions.len() != blocks
            || cos.dims().len() != 2
            || sin.dims() != cos.dims()
            || cos.dim(1)? != half
        {
            candle_core::bail!(
                "Qwen4Exp QSA rotary received incompatible query, key, position, or frequency shapes"
            );
        }
        let device = query.device();
        if pooled_keys.device().location() != device.location()
            || cos.device().location() != device.location()
            || sin.device().location() != device.location()
        {
            candle_core::bail!("Qwen4Exp QSA rotary inputs must use the same device");
        }
        let position_count = cos.dim(0)?;
        if query_position as usize >= position_count
            || block_positions
                .iter()
                .any(|position| *position as usize >= position_count)
        {
            candle_core::bail!("Qwen4Exp QSA rotary position is outside the frequency cache");
        }

        let query_index = Tensor::from_slice(&[query_position], 1, device)?;
        let query_cos = cos.index_select(&query_index, 0)?.repeat((heads, 1))?;
        let query_sin = sin.index_select(&query_index, 0)?.repeat((heads, 1))?;
        let block_indices = Tensor::from_slice(block_positions, blocks, device)?;
        let key_cos = cos.index_select(&block_indices, 0)?;
        let key_sin = sin.index_select(&block_indices, 0)?;
        Ok((
            self.rotate(query, &query_cos, &query_sin)?,
            self.rotate(pooled_keys, &key_cos, &key_sin)?,
        ))
    }

    /// Apply already-selected per-token MRoPE tables. This keeps 2D image positions in the
    /// indexer cache instead of reducing them to a scalar position.
    fn apply_position_tables(
        &self,
        query: &Tensor,
        pooled_keys: &Tensor,
        query_cos: &Tensor,
        query_sin: &Tensor,
        key_cos: &Tensor,
        key_sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (heads, _) = query.dims2()?;
        let (blocks, _) = pooled_keys.dims2()?;
        if query_cos.dims().len() != 2
            || query_sin.dims() != query_cos.dims()
            || key_cos.dims().len() != 2
            || key_sin.dims() != key_cos.dims()
            || query_cos.dim(0)? != 1
            || key_cos.dim(0)? != blocks
        {
            candle_core::bail!("Qwen4Exp QSA indexer received incompatible selected MRoPE tables");
        }
        Ok((
            self.rotate(
                query,
                &query_cos.repeat((heads, 1))?,
                &query_sin.repeat((heads, 1))?,
            )?,
            self.rotate(pooled_keys, key_cos, key_sin)?,
        ))
    }
}

#[allow(dead_code)]
pub(crate) struct QsaMainRotary {
    head_dim: usize,
    rotary_dim: usize,
}

#[allow(dead_code)]
impl QsaMainRotary {
    pub(crate) fn new(head_dim: usize, rotary_dim: usize) -> Result<Self> {
        if head_dim == 0
            || rotary_dim == 0
            || rotary_dim > head_dim
            || !rotary_dim.is_multiple_of(2)
        {
            candle_core::bail!(
                "Qwen4Exp QSA main rotary dimension must be positive, even, and no larger than the head dimension"
            );
        }
        Ok(Self {
            head_dim,
            rotary_dim,
        })
    }

    fn rotate(&self, input: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (_, _, tokens, width) = input.dims4()?;
        let half = self.rotary_dim / 2;
        if width != self.head_dim || cos.dims() != [tokens, half] || sin.dims() != [tokens, half] {
            candle_core::bail!(
                "Qwen4Exp QSA main rotary expected input [batch, heads, tokens, {}] and cos/sin [tokens, {half}], got {:?}, {:?}, and {:?}",
                self.head_dim,
                input.dims(),
                cos.dims(),
                sin.dims()
            );
        }
        let cos = cos.to_dtype(input.dtype())?.reshape((1, 1, tokens, half))?;
        let sin = sin.to_dtype(input.dtype())?.reshape((1, 1, tokens, half))?;
        let rotary = input.narrow(D::Minus1, 0, self.rotary_dim)?;
        let first = rotary.narrow(D::Minus1, 0, half)?;
        let second = rotary.narrow(D::Minus1, half, half)?;
        let rotated_first = (first.broadcast_mul(&cos)? - second.broadcast_mul(&sin)?)?;
        let rotated_second = (second.broadcast_mul(&cos)? + first.broadcast_mul(&sin)?)?;
        if self.rotary_dim == self.head_dim {
            Tensor::cat(&[&rotated_first, &rotated_second], D::Minus1)
        } else {
            let unrotated =
                input.narrow(D::Minus1, self.rotary_dim, self.head_dim - self.rotary_dim)?;
            Tensor::cat(&[&rotated_first, &rotated_second, &unrotated], D::Minus1)
        }
    }

    /// Rotate main normalized query and key heads at each token's own position.
    ///
    /// Rotated keys are cache-ready so gathered QSA workspaces never require re-rotation.
    pub(crate) fn apply(
        &self,
        query: &Tensor,
        key: &Tensor,
        positions: &[u32],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, _, tokens, query_width) = query.dims4()?;
        let (key_batch, _, key_tokens, key_width) = key.dims4()?;
        let half = self.rotary_dim / 2;
        if tokens == 0
            || query_width != self.head_dim
            || key_width != self.head_dim
            || key_batch != batch
            || key_tokens != tokens
            || positions.len() != tokens
            || cos.dims().len() != 2
            || sin.dims() != cos.dims()
            || cos.dim(1)? != half
        {
            candle_core::bail!(
                "Qwen4Exp QSA main rotary received incompatible query, key, position, or frequency shapes"
            );
        }
        let device = query.device();
        if key.device().location() != device.location()
            || cos.device().location() != device.location()
            || sin.device().location() != device.location()
        {
            candle_core::bail!("Qwen4Exp QSA main rotary inputs must use the same device");
        }
        let position_count = cos.dim(0)?;
        if positions
            .iter()
            .any(|position| *position as usize >= position_count)
        {
            candle_core::bail!("Qwen4Exp QSA main rotary position is outside the frequency cache");
        }

        let indices = Tensor::from_slice(positions, tokens, device)?;
        let token_cos = cos.index_select(&indices, 0)?;
        let token_sin = sin.index_select(&indices, 0)?;
        Ok((
            self.rotate(query, &token_cos, &token_sin)?,
            self.rotate(key, &token_cos, &token_sin)?,
        ))
    }

    /// Rotate main query and key heads with per-token cosine/sine tables.
    ///
    /// `cos`/`sin` are `[tokens, half]` and must already be selected at each token's own
    /// position, which serves callers that build frequency tables for a chunk instead of a
    /// full-position cache.
    pub(crate) fn apply_position_tables(
        &self,
        query: &Tensor,
        key: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        Ok((self.rotate(query, cos, sin)?, self.rotate(key, cos, sin)?))
    }
}

#[derive(Clone)]
pub(crate) struct QsaMainKvSnapshot {
    keys: Option<Tensor>,
    values: Option<Tensor>,
    kv_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
}

#[allow(dead_code)]
pub(crate) struct QsaMainKvCache {
    sequences: HashMap<usize, (Tensor, Tensor)>,
    kv_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
}

#[allow(dead_code)]
impl QsaMainKvCache {
    pub(crate) fn new(kv_heads: usize, key_head_dim: usize, value_head_dim: usize) -> Result<Self> {
        if kv_heads == 0 || key_head_dim == 0 || value_head_dim == 0 {
            candle_core::bail!("Qwen4Exp QSA main K/V cache dimensions must be positive");
        }
        Ok(Self {
            sequences: HashMap::new(),
            kv_heads,
            key_head_dim,
            value_head_dim,
        })
    }

    /// Append cache-ready rotated keys and unrotated values shaped [kv_heads, tokens, head_dim].
    pub(crate) fn append(
        &mut self,
        sequence_id: usize,
        keys: &Tensor,
        values: &Tensor,
    ) -> Result<()> {
        let (key_heads, key_tokens, key_dim) = keys.dims3()?;
        let (value_heads, value_tokens, value_dim) = values.dims3()?;
        if key_heads != self.kv_heads
            || key_dim != self.key_head_dim
            || value_heads != self.kv_heads
            || value_dim != self.value_head_dim
            || key_tokens != value_tokens
        {
            candle_core::bail!(
                "Qwen4Exp QSA main K/V cache expected keys [heads, tokens, {}] and values [heads, tokens, {}], got {:?} and {:?}",
                self.key_head_dim,
                self.value_head_dim,
                keys.dims(),
                values.dims()
            );
        }
        // Projection outputs hand over head-major views via transpose; store contiguous
        // tensors so later row gathers through `QsaKvGather` stay valid.
        let keys = keys.contiguous()?;
        let values = values.contiguous()?;
        if let Some((cached_keys, cached_values)) = self.sequences.get_mut(&sequence_id) {
            if cached_keys.device().location() != keys.device().location()
                || cached_keys.dtype() != keys.dtype()
                || cached_values.device().location() != values.device().location()
                || cached_values.dtype() != values.dtype()
            {
                candle_core::bail!(
                    "Qwen4Exp QSA main K/V cache append requires matching key and value dtype and device"
                );
            }
            let keys = Tensor::cat(&[&*cached_keys, &keys], 1)?;
            let values = Tensor::cat(&[&*cached_values, &values], 1)?;
            *cached_keys = keys;
            *cached_values = values;
        } else {
            self.sequences.insert(sequence_id, (keys, values));
        }
        Ok(())
    }

    pub(crate) fn get(&self, sequence_id: usize) -> Option<(&Tensor, &Tensor)> {
        self.sequences
            .get(&sequence_id)
            .map(|(keys, values)| (keys, values))
    }

    pub(crate) fn validate_truncate(&self, sequence_id: usize, len: usize) -> Result<()> {
        let Some((keys, values)) = self.sequences.get(&sequence_id) else {
            if len == 0 {
                return Ok(());
            }
            candle_core::bail!("Qwen4Exp QSA main K/V cache cannot truncate an absent sequence");
        };
        let key_tokens = keys.dim(1)?;
        let value_tokens = values.dim(1)?;
        if key_tokens != value_tokens {
            candle_core::bail!(
                "Qwen4Exp QSA main K/V cache keys and values have different lengths"
            );
        }
        if len > key_tokens {
            candle_core::bail!(
                "Qwen4Exp QSA main K/V cache cannot truncate a cache of length {key_tokens} to {len}"
            );
        }
        Ok(())
    }

    /// Truncate to `len` logical tokens along the token axis.
    pub(crate) fn truncate(&mut self, sequence_id: usize, len: usize) -> Result<()> {
        self.validate_truncate(sequence_id, len)?;
        let Some((keys, values)) = self.sequences.get_mut(&sequence_id) else {
            return Ok(());
        };
        *keys = keys.narrow(1, 0, len)?;
        *values = values.narrow(1, 0, len)?;
        Ok(())
    }

    pub(crate) fn snapshot(&self, sequence_id: usize) -> QsaMainKvSnapshot {
        let entry = self.sequences.get(&sequence_id);
        QsaMainKvSnapshot {
            keys: entry.map(|(keys, _)| keys.clone()),
            values: entry.map(|(_, values)| values.clone()),
            kv_heads: self.kv_heads,
            key_head_dim: self.key_head_dim,
            value_head_dim: self.value_head_dim,
        }
    }

    pub(crate) fn validate_snapshot(&self, snapshot: &QsaMainKvSnapshot) -> Result<()> {
        if snapshot.kv_heads != self.kv_heads
            || snapshot.key_head_dim != self.key_head_dim
            || snapshot.value_head_dim != self.value_head_dim
        {
            candle_core::bail!("Qwen4Exp QSA main K/V snapshot is incompatible with the cache");
        }
        match (&snapshot.keys, &snapshot.values) {
            (Some(keys), Some(values)) => {
                let (key_heads, key_tokens, key_dim) = keys.dims3()?;
                let (value_heads, value_tokens, value_dim) = values.dims3()?;
                if key_heads != self.kv_heads
                    || key_dim != self.key_head_dim
                    || value_heads != self.kv_heads
                    || value_dim != self.value_head_dim
                    || key_tokens != value_tokens
                {
                    candle_core::bail!("Qwen4Exp QSA main K/V snapshot has an invalid inventory");
                }
            }
            (None, None) => {}
            _ => {
                candle_core::bail!("Qwen4Exp QSA main K/V snapshot has only one of keys and values")
            }
        }
        Ok(())
    }

    pub(crate) fn restore(
        &mut self,
        sequence_id: usize,
        snapshot: &QsaMainKvSnapshot,
    ) -> Result<()> {
        self.validate_snapshot(snapshot)?;
        if let (Some(keys), Some(values)) = (&snapshot.keys, &snapshot.values) {
            self.sequences
                .insert(sequence_id, (keys.clone(), values.clone()));
        } else {
            self.sequences.remove(&sequence_id);
        }
        Ok(())
    }

    pub(crate) fn release(&mut self, sequence_id: usize) -> bool {
        self.sequences.remove(&sequence_id).is_some()
    }

    pub(crate) fn reset(&mut self, sequence_id: usize) {
        self.release(sequence_id);
    }

    pub(crate) fn clear(&mut self) {
        self.sequences.clear();
    }
}

#[allow(dead_code)]
pub(crate) struct QsaIndexerProjection {
    query: Arc<dyn QuantMethod>,
    key: Arc<dyn QuantMethod>,
    query_norm: Tensor,
    key_norm: Tensor,
    norm_eps: f64,
    hidden_size: usize,
    head_count: usize,
    head_dim: usize,
}

#[allow(dead_code)]
impl QsaIndexerProjection {
    pub(crate) fn new(config: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        config.validate()?;
        let query_width = config
            .indexer_n_heads
            .checked_mul(config.indexer_head_dim)
            .ok_or_else(|| Error::msg("Qwen4Exp QSA query projection width overflow"))?;
        Ok(Self {
            query: ReplicatedLayer::new(
                config.hidden_size,
                query_width,
                &config.quantization_config,
                false,
                vb.pp("q_proj"),
            )?,
            key: ReplicatedLayer::new(
                config.hidden_size,
                config.indexer_head_dim,
                &config.quantization_config,
                false,
                vb.pp("k_proj"),
            )?,
            query_norm: vb.pp("q_norm").get(config.indexer_head_dim, "weight")?,
            key_norm: vb.pp("k_norm").get(config.indexer_head_dim, "weight")?,
            norm_eps: config.rms_norm_eps,
            hidden_size: config.hidden_size,
            head_count: config.indexer_n_heads,
            head_dim: config.indexer_head_dim,
        })
    }

    fn normalize(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        if input.dim(D::Minus1)? != self.head_dim {
            candle_core::bail!(
                "Qwen4Exp QSA normalization expected width {}, got {:?}",
                self.head_dim,
                input.dims()
            );
        }
        let dtype = input.dtype();
        let input = input.to_dtype(DType::F32)?;
        let variance = input.sqr()?.mean_keepdim(D::Minus1)?;
        input
            .broadcast_div(&(variance + self.norm_eps)?.sqrt()?)?
            .broadcast_mul(&weight.to_dtype(DType::F32)?)?
            .to_dtype(dtype)
    }

    /// Project normalized queries and raw keys. Keys stay raw until after block pooling.
    pub(crate) fn project(&self, hidden: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, tokens, hidden_size) = hidden.dims3()?;
        if hidden_size != self.hidden_size {
            candle_core::bail!(
                "Qwen4Exp QSA indexer expected hidden width {}, got {hidden_size}",
                self.hidden_size
            );
        }
        let query =
            self.query
                .forward(hidden)?
                .reshape((batch, tokens, self.head_count, self.head_dim))?;
        let query = self.normalize(&query, &self.query_norm)?;
        let raw_key = self
            .key
            .forward(hidden)?
            .reshape((batch, tokens, self.head_dim))?;
        Ok((query, raw_key))
    }

    pub(crate) fn normalize_pooled_keys(&self, pooled_keys: &Tensor) -> Result<Tensor> {
        self.normalize(pooled_keys, &self.key_norm)
    }
}

#[allow(dead_code)]
pub(crate) struct QsaIndexerScorer {
    head_count: usize,
    head_dim: usize,
    compress_ratio: usize,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QsaQueryGateLayout {
    /// GGUF stores each query head immediately followed by that head's output gate.
    InterleavedPerHead,
    /// Packed projection paths may group all query values before all gate values.
    Grouped,
}

#[allow(dead_code)]
pub(crate) struct QsaMainProjection {
    query_gate: Arc<dyn QuantMethod>,
    key: Arc<dyn QuantMethod>,
    value: Arc<dyn QuantMethod>,
    query_norm: Tensor,
    key_norm: Tensor,
    norm_eps: f64,
    hidden_size: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    layout: QsaQueryGateLayout,
}

#[allow(dead_code)]
impl QsaMainProjection {
    pub(crate) fn new(
        config: &Config,
        layout: QsaQueryGateLayout,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        config.validate()?;
        let query_width = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .and_then(|width| width.checked_mul(2))
            .ok_or_else(|| Error::msg("Qwen4Exp QSA query/gate projection width overflow"))?;
        let kv_width = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| Error::msg("Qwen4Exp QSA K/V projection width overflow"))?;
        Ok(Self {
            query_gate: ReplicatedLayer::new(
                config.hidden_size,
                query_width,
                &config.quantization_config,
                false,
                vb.pp("q_proj"),
            )?,
            key: ReplicatedLayer::new(
                config.hidden_size,
                kv_width,
                &config.quantization_config,
                false,
                vb.pp("k_proj"),
            )?,
            value: ReplicatedLayer::new(
                config.hidden_size,
                kv_width,
                &config.quantization_config,
                false,
                vb.pp("v_proj"),
            )?,
            query_norm: vb.pp("q_norm").get(config.head_dim, "weight")?,
            key_norm: vb.pp("k_norm").get(config.head_dim, "weight")?,
            norm_eps: config.rms_norm_eps,
            hidden_size: config.hidden_size,
            query_heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            layout,
        })
    }

    fn split_query_gate(&self, projected: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, tokens, width) = projected.dims3()?;
        let query_width = self
            .query_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::msg("Qwen4Exp QSA query width overflow"))?;
        if width != query_width * 2 {
            candle_core::bail!(
                "Qwen4Exp QSA query/gate projection expected width {}, got {width}",
                query_width * 2
            );
        }
        match self.layout {
            QsaQueryGateLayout::InterleavedPerHead => {
                let projected =
                    projected.reshape((batch, tokens, self.query_heads, self.head_dim * 2))?;
                let query = projected.narrow(D::Minus1, 0, self.head_dim)?;
                let gate = projected
                    .narrow(D::Minus1, self.head_dim, self.head_dim)?
                    .reshape((batch, tokens, query_width))?;
                Ok((query, gate))
            }
            QsaQueryGateLayout::Grouped => Ok((
                projected.narrow(D::Minus1, 0, query_width)?.reshape((
                    batch,
                    tokens,
                    self.query_heads,
                    self.head_dim,
                ))?,
                projected.narrow(D::Minus1, query_width, query_width)?,
            )),
        }
    }

    fn normalize(&self, input: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let dtype = input.dtype();
        let input = input.to_dtype(DType::F32)?;
        let variance = input.sqr()?.mean_keepdim(D::Minus1)?;
        input
            .broadcast_div(&(variance + self.norm_eps)?.sqrt()?)?
            .broadcast_mul(&weight.to_dtype(DType::F32)?)?
            .to_dtype(dtype)
    }

    /// Project normalized Q/K, unnormalized V, and the per-head sigmoid output-gate input.
    pub(crate) fn project(&self, hidden: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (batch, tokens, hidden_size) = hidden.dims3()?;
        if hidden_size != self.hidden_size {
            candle_core::bail!(
                "Qwen4Exp QSA main projection expected hidden width {}, got {hidden_size}",
                self.hidden_size
            );
        }
        let query_gate = self.query_gate.forward(hidden)?;
        let (query, gate) = self.split_query_gate(&query_gate)?;
        let key =
            self.key
                .forward(hidden)?
                .reshape((batch, tokens, self.kv_heads, self.head_dim))?;
        let value =
            self.value
                .forward(hidden)?
                .reshape((batch, tokens, self.kv_heads, self.head_dim))?;
        Ok((
            self.normalize(&query, &self.query_norm)?.transpose(1, 2)?,
            self.normalize(&key, &self.key_norm)?.transpose(1, 2)?,
            value.transpose(1, 2)?,
            gate,
        ))
    }
}

impl QsaIndexerScorer {
    pub(crate) fn new(head_count: usize, head_dim: usize, compress_ratio: usize) -> Result<Self> {
        if head_count == 0 || head_dim == 0 || compress_ratio == 0 {
            candle_core::bail!(
                "Qwen4Exp QSA indexer requires positive head, head dimension, and compression counts"
            );
        }
        Ok(Self {
            head_count,
            head_dim,
            compress_ratio,
        })
    }

    /// Mean-pool complete raw-key blocks. The incomplete tail is selected separately.
    pub(crate) fn pool_complete_blocks(&self, raw_keys: &Tensor) -> Result<Tensor> {
        let (tokens, head_dim) = raw_keys.dims2()?;
        if head_dim != self.head_dim {
            candle_core::bail!(
                "Qwen4Exp QSA raw keys expected width {}, got {head_dim}",
                self.head_dim
            );
        }
        let complete_blocks = tokens / self.compress_ratio;
        if complete_blocks == 0 {
            return Tensor::zeros((0, self.head_dim), raw_keys.dtype(), raw_keys.device());
        }
        raw_keys
            .narrow(0, 0, complete_blocks * self.compress_ratio)?
            .reshape((complete_blocks, self.compress_ratio, self.head_dim))?
            .mean(D::Minus2)
    }

    /// Score already-normalized and position-rotated queries and pooled keys.
    pub(crate) fn score(&self, query: &Tensor, pooled_keys: &Tensor) -> Result<Tensor> {
        let (heads, query_dim) = query.dims2()?;
        let (_, key_dim) = pooled_keys.dims2()?;
        if heads != self.head_count || query_dim != self.head_dim || key_dim != self.head_dim {
            candle_core::bail!(
                "Qwen4Exp QSA scoring expected query [{}, {}] and keys [blocks, {}], got {:?} and {:?}",
                self.head_count,
                self.head_dim,
                self.head_dim,
                query.dims(),
                pooled_keys.dims()
            );
        }
        if query.device().location() != pooled_keys.device().location() {
            candle_core::bail!("Qwen4Exp QSA query and pooled keys must use the same device");
        }

        let scores = query
            .to_dtype(DType::F32)?
            .matmul(&pooled_keys.to_dtype(DType::F32)?.t()?)?
            .relu()?
            .sum(0)?;
        scores / (self.head_dim as f64).sqrt()
    }
}

#[allow(dead_code)]
pub(crate) struct QsaOutputProjection {
    output: Arc<dyn QuantMethod>,
    attention_width: usize,
    hidden_size: usize,
}

#[allow(dead_code)]
impl QsaOutputProjection {
    pub(crate) fn new(config: &Config, vb: ShardedVarBuilder) -> Result<Self> {
        config.validate()?;
        let attention_width = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| Error::msg("Qwen4Exp QSA attention width overflow"))?;
        Ok(Self {
            output: ReplicatedLayer::new(
                attention_width,
                config.hidden_size,
                &config.quantization_config,
                false,
                vb.pp("o_proj"),
            )?,
            attention_width,
            hidden_size: config.hidden_size,
        })
    }

    /// Apply the reference sigmoid output gate and project to the hidden width.
    ///
    /// `attention` is the flattened per-head attention output and `gate` the flattened
    /// per-head sigmoid input emitted by `QsaMainProjection::project`.
    pub(crate) fn forward(&self, attention: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let (batch, tokens, width) = attention.dims3()?;
        if width != self.attention_width || gate.dims3()? != (batch, tokens, self.attention_width) {
            candle_core::bail!(
                "Qwen4Exp QSA output projection expected attention and gate of width {}, got {:?} and {:?}",
                self.attention_width,
                attention.dims(),
                gate.dims()
            );
        }
        let device = attention.device();
        if gate.device().location() != device.location() {
            candle_core::bail!("Qwen4Exp QSA output projection inputs must use the same device");
        }
        let gated = attention.broadcast_mul(&candle_nn::ops::sigmoid(gate)?)?;
        self.output.forward(&gated)
    }
}

#[allow(dead_code)]
pub(crate) struct QsaBlockSelector {
    compress_ratio: usize,
    token_budget: usize,
}

#[allow(dead_code)]
pub(crate) struct QsaKvGather {
    kv_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    max_selected_tokens: usize,
}

#[allow(dead_code)]
impl QsaKvGather {
    pub(crate) fn new(
        kv_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
        token_budget: usize,
        compress_ratio: usize,
    ) -> Result<Self> {
        if kv_heads == 0
            || key_head_dim == 0
            || value_head_dim == 0
            || token_budget == 0
            || compress_ratio == 0
        {
            candle_core::bail!("Qwen4Exp QSA K/V gather dimensions must be positive");
        }
        let max_selected_tokens = token_budget
            .checked_add(compress_ratio - 1)
            .ok_or_else(|| Error::msg("Qwen4Exp QSA K/V gather budget overflow"))?;
        Ok(Self {
            kv_heads,
            key_head_dim,
            value_head_dim,
            max_selected_tokens,
        })
    }

    /// Gather selected cache rows into bounded contiguous per-query workspaces.
    pub(crate) fn gather(
        &self,
        keys: &Tensor,
        values: &Tensor,
        selections: &[Vec<u32>],
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let (key_heads, key_tokens, key_dim) = keys.dims3()?;
        let (value_heads, value_tokens, value_dim) = values.dims3()?;
        if key_heads != self.kv_heads
            || value_heads != self.kv_heads
            || key_tokens != value_tokens
            || key_dim != self.key_head_dim
            || value_dim != self.value_head_dim
        {
            candle_core::bail!(
                "Qwen4Exp QSA K/V gather expected keys [{}, tokens, {}] and values [{}, tokens, {}], got {:?} and {:?}",
                self.kv_heads,
                self.key_head_dim,
                self.kv_heads,
                self.value_head_dim,
                keys.dims(),
                values.dims()
            );
        }
        if keys.device().location() != values.device().location() || keys.dtype() != values.dtype()
        {
            candle_core::bail!("Qwen4Exp QSA K/V gather requires matching cache dtype and device");
        }
        for selected in selections {
            if selected.is_empty() || selected.len() > self.max_selected_tokens {
                candle_core::bail!(
                    "Qwen4Exp QSA K/V gather requires 1..={} selected rows, got {}",
                    self.max_selected_tokens,
                    selected.len()
                );
            }
            if selected.iter().any(|row| *row as usize >= key_tokens) {
                candle_core::bail!(
                    "Qwen4Exp QSA K/V gather row is outside the cache length {key_tokens}"
                );
            }
        }

        selections
            .iter()
            .map(|selected| {
                let indices = Tensor::from_slice(selected, selected.len(), keys.device())?;
                Ok((
                    keys.index_select(&indices, 1)?.contiguous()?,
                    values.index_select(&indices, 1)?.contiguous()?,
                ))
            })
            .collect()
    }
}

#[allow(dead_code)]
pub(crate) struct QsaSparseAttention {
    query_heads: usize,
    kv_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    sdpa_params: SdpaParams,
}

#[allow(dead_code)]
impl QsaSparseAttention {
    pub(crate) fn new(
        query_heads: usize,
        kv_heads: usize,
        key_head_dim: usize,
        value_head_dim: usize,
    ) -> Result<Self> {
        if query_heads == 0
            || kv_heads == 0
            || key_head_dim == 0
            || value_head_dim == 0
            || !query_heads.is_multiple_of(kv_heads)
        {
            candle_core::bail!(
                "Qwen4Exp sparse attention requires positive dimensions and query heads divisible by KV heads"
            );
        }
        Ok(Self {
            query_heads,
            kv_heads,
            key_head_dim,
            value_head_dim,
            sdpa_params: SdpaParams {
                n_kv_groups: query_heads / kv_heads,
                softcap: None,
                softmax_scale: 1.0 / (key_head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    /// Run independent one-query attention calls over each query's selected K/V workspace.
    pub(crate) fn forward(
        &self,
        queries: &Tensor,
        gathered: &[(Tensor, Tensor)],
    ) -> Result<Tensor> {
        let (batch, query_heads, query_tokens, query_dim) = queries.dims4()?;
        if batch != 1
            || query_heads != self.query_heads
            || query_dim != self.key_head_dim
            || gathered.len() != query_tokens
        {
            candle_core::bail!(
                "Qwen4Exp sparse attention expected queries [1, {}, tokens, {}] and one K/V workspace per token, got {:?} and {} workspaces",
                self.query_heads,
                self.key_head_dim,
                queries.dims(),
                gathered.len()
            );
        }
        for (keys, values) in gathered {
            let (key_heads, key_tokens, key_dim) = keys.dims3()?;
            let (value_heads, value_tokens, value_dim) = values.dims3()?;
            if key_heads != self.kv_heads
                || value_heads != self.kv_heads
                || key_tokens == 0
                || key_tokens != value_tokens
                || key_dim != self.key_head_dim
                || value_dim != self.value_head_dim
                || keys.dtype() != queries.dtype()
                || values.dtype() != queries.dtype()
                || keys.device().location() != queries.device().location()
                || values.device().location() != queries.device().location()
            {
                candle_core::bail!(
                    "Qwen4Exp sparse attention received an incompatible gathered K/V workspace"
                );
            }
        }

        let outputs = gathered
            .iter()
            .enumerate()
            .map(|(token, (keys, values))| {
                let query = queries.narrow(2, token, 1)?;
                Sdpa.run_attention_noflash(
                    &query,
                    &keys.unsqueeze(0)?,
                    &values.unsqueeze(0)?,
                    None,
                    &self.sdpa_params,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&outputs, 2)
    }
}

#[allow(dead_code)]
pub(crate) struct QsaIndexer {
    projection: QsaIndexerProjection,
    scorer: QsaIndexerScorer,
    rotary: QsaIndexerRotary,
    selector: QsaBlockSelector,
    cache: QsaSequenceCache,
}

#[allow(dead_code)]
impl QsaIndexer {
    pub(crate) fn new(config: &Config, rotary_dim: usize, vb: ShardedVarBuilder) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            projection: QsaIndexerProjection::new(config, vb)?,
            scorer: QsaIndexerScorer::new(
                config.indexer_n_heads,
                config.indexer_head_dim,
                config.indexer_compress_ratio,
            )?,
            rotary: QsaIndexerRotary::new(config.indexer_head_dim, rotary_dim)?,
            selector: QsaBlockSelector::new(config.indexer_compress_ratio, config.indexer_budget)?,
            cache: QsaSequenceCache::new(config.indexer_head_dim)?,
        })
    }

    /// Reset one sequence's raw indexer-key cache to the absent state.
    pub(crate) fn reset_sequence(&mut self, sequence_id: usize) {
        self.cache.reset(sequence_id);
    }

    /// Release one sequence's raw indexer-key cache entirely.
    pub(crate) fn release_sequence(&mut self, sequence_id: usize) -> bool {
        self.cache.release(sequence_id)
    }

    /// Clear every sequence's raw indexer-key cache.
    pub(crate) fn clear_sequences(&mut self) {
        self.cache.clear();
    }

    /// Whether the raw indexer-key cache still holds state for a sequence.
    pub(crate) fn has_sequence(&self, sequence_id: usize) -> bool {
        self.cache.get(sequence_id).is_some()
    }

    /// Append one sequence chunk and select visible cache rows independently for every query.
    pub(crate) fn process_chunk(
        &mut self,
        sequence_id: usize,
        hidden: &Tensor,
        positions: &[u32],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Vec<Vec<u32>>> {
        let (batch, tokens, _) = hidden.dims3()?;
        if batch != 1 || positions.len() != tokens {
            candle_core::bail!(
                "Qwen4Exp QSA indexer expects one sequence and one position per token, got batch {batch}, {tokens} tokens, and {} positions",
                positions.len()
            );
        }

        let (queries, raw_keys) = self.projection.project(hidden)?;
        let raw_keys = raw_keys.squeeze(0)?;
        let previous_len = self
            .cache
            .get(sequence_id)
            .map(|(_, positions)| positions.len())
            .unwrap_or(0);
        let snapshot = self.cache.snapshot(sequence_id);
        let result = (|| {
            self.cache.append(sequence_id, &raw_keys, positions)?;
            let (cached_keys, cached_positions) = self
                .cache
                .get(sequence_id)
                .ok_or_else(|| Error::msg("Qwen4Exp QSA cache entry disappeared after append"))?;
            let mut selections = Vec::with_capacity(tokens);

            for (token, position) in (0..tokens).zip(positions.iter().copied()) {
                let visible_len = previous_len
                    .checked_add(token + 1)
                    .ok_or_else(|| Error::msg("Qwen4Exp QSA visible length overflow"))?;
                if visible_len > u32::MAX as usize {
                    candle_core::bail!("Qwen4Exp QSA cache is too large for row indices");
                }
                let visible_keys = cached_keys.narrow(0, 0, visible_len)?;
                let complete_blocks = visible_len / self.scorer.compress_ratio;
                let visible_rows = (0..visible_len as u32).collect::<Vec<_>>();
                if complete_blocks == 0 {
                    selections.push(self.selector.select(&visible_rows, &[])?);
                    continue;
                }

                let pooled = self.scorer.pool_complete_blocks(&visible_keys)?;
                let pooled = self.projection.normalize_pooled_keys(&pooled)?;
                let block_positions = (0..complete_blocks)
                    .map(|block| cached_positions[block * self.scorer.compress_ratio])
                    .collect::<Vec<_>>();
                let query = queries
                    .narrow(0, 0, 1)?
                    .narrow(1, token, 1)?
                    .squeeze(0)?
                    .squeeze(0)?;
                let (query, pooled) =
                    self.rotary
                        .apply(&query, &pooled, position, &block_positions, cos, sin)?;
                let scores = self.scorer.score(&query, &pooled)?.to_vec1::<f32>()?;
                selections.push(self.selector.select(&visible_rows, &scores)?);
            }
            Ok(selections)
        })();

        if result.is_err() {
            self.cache.restore(sequence_id, &snapshot)?;
        }
        result
    }

    /// Append one sequence with per-token sectioned MRoPE tables and select cache rows.
    fn process_chunk_with_position_tables(
        &mut self,
        sequence_id: usize,
        hidden: &Tensor,
        positions: &[u32],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Vec<Vec<u32>>> {
        let (batch, tokens, _) = hidden.dims3()?;
        let half = self.rotary.rotary_dim / 2;
        if batch != 1
            || positions.len() != tokens
            || cos.dims() != [tokens, half]
            || sin.dims() != cos.dims()
        {
            candle_core::bail!("Qwen4Exp QSA indexer received incompatible sectioned MRoPE inputs");
        }
        let (queries, raw_keys) = self.projection.project(hidden)?;
        let raw_keys = raw_keys.squeeze(0)?;
        let previous_len = self
            .cache
            .get(sequence_id)
            .map(|(_, positions)| positions.len())
            .unwrap_or(0);
        let snapshot = self.cache.snapshot(sequence_id);
        let result = (|| {
            self.cache
                .append_with_position_tables(sequence_id, &raw_keys, positions, cos, sin)?;
            let (cached_keys, _) = self
                .cache
                .get(sequence_id)
                .ok_or_else(|| Error::msg("Qwen4Exp QSA cache entry disappeared after append"))?;
            let (cached_cos, cached_sin) = self.cache.position_tables(sequence_id)?;
            let mut selections = Vec::with_capacity(tokens);
            for token in 0..tokens {
                let visible_len = previous_len
                    .checked_add(token + 1)
                    .ok_or_else(|| Error::msg("Qwen4Exp QSA visible length overflow"))?;
                if visible_len > u32::MAX as usize {
                    candle_core::bail!("Qwen4Exp QSA cache is too large for row indices");
                }
                let complete_blocks = visible_len / self.scorer.compress_ratio;
                let visible_rows = (0..visible_len as u32).collect::<Vec<_>>();
                if complete_blocks == 0 {
                    selections.push(self.selector.select(&visible_rows, &[])?);
                    continue;
                }
                let pooled = self.projection.normalize_pooled_keys(
                    &self
                        .scorer
                        .pool_complete_blocks(&cached_keys.narrow(0, 0, visible_len)?)?,
                )?;
                let block_indices = (0..complete_blocks)
                    .map(|block| (block * self.scorer.compress_ratio) as u32)
                    .collect::<Vec<_>>();
                let block_indices =
                    Tensor::from_slice(&block_indices, complete_blocks, cached_cos.device())?;
                let query = queries
                    .narrow(0, 0, 1)?
                    .narrow(1, token, 1)?
                    .squeeze(0)?
                    .squeeze(0)?;
                let query_cos = cos.narrow(0, token, 1)?;
                let query_sin = sin.narrow(0, token, 1)?;
                let key_cos = cached_cos.index_select(&block_indices, 0)?;
                let key_sin = cached_sin.index_select(&block_indices, 0)?;
                let (query, pooled) = self.rotary.apply_position_tables(
                    &query, &pooled, &query_cos, &query_sin, &key_cos, &key_sin,
                )?;
                let scores = self.scorer.score(&query, &pooled)?.to_vec1::<f32>()?;
                selections.push(self.selector.select(&visible_rows, &scores)?);
            }
            Ok(selections)
        })();
        if result.is_err() {
            self.cache.restore(sequence_id, &snapshot)?;
        }
        result
    }

    /// Process packed chunks in input order while preserving independent sequence histories.
    pub(crate) fn process_packed_chunks(
        &mut self,
        chunks: &[(usize, &Tensor, &[u32])],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Vec<Vec<Vec<u32>>>> {
        if chunks.is_empty() {
            candle_core::bail!("Qwen4Exp packed QSA indexer requires at least one chunk");
        }
        for (sequence_id, hidden, positions) in chunks {
            let (batch, tokens, _) = hidden.dims3()?;
            if batch != 1 || positions.len() != tokens {
                candle_core::bail!(
                    "Qwen4Exp packed QSA sequence {sequence_id} expects one sequence and one position per token, got batch {batch}, {tokens} tokens, and {} positions",
                    positions.len()
                );
            }
        }

        let snapshots = chunks
            .iter()
            .map(|(sequence_id, _, _)| (*sequence_id, self.cache.snapshot(*sequence_id)))
            .collect::<HashMap<_, _>>();
        let result = (|| {
            let mut selections = Vec::with_capacity(chunks.len());
            for (sequence_id, hidden, positions) in chunks {
                selections.push(self.process_chunk(*sequence_id, hidden, positions, cos, sin)?);
            }
            Ok(selections)
        })();
        if result.is_err() {
            for (sequence_id, snapshot) in snapshots {
                self.cache.restore(sequence_id, &snapshot)?;
            }
        }
        result
    }

    pub(crate) fn cache(&self) -> &QsaSequenceCache {
        &self.cache
    }

    pub(crate) fn cache_mut(&mut self) -> &mut QsaSequenceCache {
        &mut self.cache
    }
}

/// End-to-end QSA full-attention layer for one sequence chunk or packed
/// multi-sequence chunks.
///
/// Chains the main projection, main Q/K rotation, main K/V cache append, indexer
/// selection, sparse K/V gathering, sparse attention, and sigmoid-gated output
/// projection. The main K/V cache and indexer cache update as one transaction.
#[allow(dead_code)]
#[derive(Clone)]
pub(crate) struct QsaAttentionSnapshot {
    main_kv: QsaMainKvSnapshot,
    indexer: QsaSequenceSnapshot,
}

pub(crate) struct QsaAttention {
    main: QsaMainProjection,
    rotary: QsaMainRotary,
    output: QsaOutputProjection,
    kv_cache: QsaMainKvCache,
    indexer: QsaIndexer,
    kv_gather: QsaKvGather,
    attention: QsaSparseAttention,
}

#[allow(dead_code)]
impl QsaAttention {
    /// Per-head query and key RMS gammas, exposed for ISQ residual handling.
    pub(crate) fn residual_norms(&self) -> (&Tensor, &Tensor) {
        (&self.main.query_norm, &self.main.key_norm)
    }

    pub(crate) fn new(
        config: &Config,
        rotary_dim: usize,
        layout: QsaQueryGateLayout,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        config.validate()?;
        let kv_gather = QsaKvGather::new(
            config.num_key_value_heads,
            config.head_dim,
            config.head_dim,
            config.indexer_budget,
            config.indexer_compress_ratio,
        )?;
        Ok(Self {
            main: QsaMainProjection::new(config, layout, vb.pp("attn"))?,
            rotary: QsaMainRotary::new(config.head_dim, rotary_dim)?,
            output: QsaOutputProjection::new(config, vb.pp("attn"))?,
            kv_cache: QsaMainKvCache::new(
                config.num_key_value_heads,
                config.head_dim,
                config.head_dim,
            )?,
            indexer: QsaIndexer::new(config, rotary_dim, vb.pp("indexer"))?,
            kv_gather,
            attention: QsaSparseAttention::new(
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.head_dim,
            )?,
        })
    }

    pub(crate) fn snapshot_sequence(&self, sequence_id: usize) -> QsaAttentionSnapshot {
        QsaAttentionSnapshot {
            main_kv: self.kv_cache.snapshot(sequence_id),
            indexer: self.indexer.cache().snapshot(sequence_id),
        }
    }

    pub(crate) fn validate_restore_sequence(&self, snapshot: &QsaAttentionSnapshot) -> Result<()> {
        self.kv_cache.validate_snapshot(&snapshot.main_kv)?;
        self.indexer.cache().validate_snapshot(&snapshot.indexer)?;
        match (&snapshot.main_kv.keys, &snapshot.indexer.raw_keys) {
            (Some(keys), Some(raw_keys)) if keys.dim(1)? == raw_keys.dim(0)? => Ok(()),
            (None, None) => Ok(()),
            _ => candle_core::bail!(
                "Qwen4Exp QSA snapshot main K/V and indexer caches have different sequence inventories"
            ),
        }
    }

    pub(crate) fn restore_sequence(
        &mut self,
        sequence_id: usize,
        snapshot: &QsaAttentionSnapshot,
    ) -> Result<()> {
        self.validate_restore_sequence(snapshot)?;
        let current = self.snapshot_sequence(sequence_id);
        self.kv_cache.restore(sequence_id, &snapshot.main_kv)?;
        if let Err(error) = self
            .indexer
            .cache_mut()
            .restore(sequence_id, &snapshot.indexer)
        {
            self.kv_cache.restore(sequence_id, &current.main_kv)?;
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn validate_truncate_sequence(&self, sequence_id: usize, len: usize) -> Result<()> {
        self.kv_cache.validate_truncate(sequence_id, len)?;
        self.indexer.cache().validate_truncate(sequence_id, len)?;
        match (
            self.kv_cache.get(sequence_id),
            self.indexer.cache().get(sequence_id),
        ) {
            (Some((keys, _)), Some((_, positions))) if keys.dim(1)? == positions.len() => Ok(()),
            (None, None) if len == 0 => Ok(()),
            _ => candle_core::bail!(
                "Qwen4Exp QSA main K/V and indexer caches have different sequence inventories"
            ),
        }
    }

    /// Truncate main K/V, raw indexer keys, and exact positions as one transaction.
    pub(crate) fn truncate_sequence(&mut self, sequence_id: usize, len: usize) -> Result<()> {
        self.validate_truncate_sequence(sequence_id, len)?;
        let kv_snapshot = self.kv_cache.snapshot(sequence_id);
        let indexer_snapshot = self.indexer.cache().snapshot(sequence_id);
        let result = (|| {
            self.kv_cache.truncate(sequence_id, len)?;
            self.indexer.cache_mut().truncate(sequence_id, len)
        })();
        if let Err(error) = result {
            self.kv_cache.restore(sequence_id, &kv_snapshot)?;
            self.indexer
                .cache_mut()
                .restore(sequence_id, &indexer_snapshot)?;
            return Err(error);
        }
        Ok(())
    }

    /// Reset one sequence's main K/V and raw indexer-key caches to the absent state.
    pub(crate) fn reset_sequence(&mut self, sequence_id: usize) {
        self.kv_cache.reset(sequence_id);
        self.indexer.reset_sequence(sequence_id);
    }

    /// Release one sequence's main K/V and raw indexer-key caches entirely.
    pub(crate) fn release_sequence(&mut self, sequence_id: usize) -> bool {
        self.kv_cache.release(sequence_id) | self.indexer.release_sequence(sequence_id)
    }

    /// Clear every sequence's main K/V and raw indexer-key caches.
    pub(crate) fn clear_sequences(&mut self) {
        self.kv_cache.clear();
        self.indexer.clear_sequences();
    }

    /// Whether any per-sequence cache in this layer still holds state for a sequence.
    pub(crate) fn has_cached_sequence(&self, sequence_id: usize) -> bool {
        self.kv_cache.get(sequence_id).is_some() || self.indexer.has_sequence(sequence_id)
    }

    /// Process one sequence chunk and return the attention output in packed token order.
    pub(crate) fn forward_chunk(
        &mut self,
        sequence_id: usize,
        hidden: &Tensor,
        positions: &[u32],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (batch, tokens, hidden_width) = hidden.dims3()?;
        if batch != 1 || positions.len() != tokens {
            candle_core::bail!(
                "Qwen4Exp QSA attention expects one sequence and one position per token, got batch {batch}, {tokens} tokens, and {} positions",
                positions.len()
            );
        }
        if self.main.head_dim != self.attention.value_head_dim {
            candle_core::bail!(
                "Qwen4Exp QSA attention requires matching key and value head dimensions for the per-head output gate"
            );
        }
        if hidden_width != self.main.hidden_size {
            candle_core::bail!(
                "Qwen4Exp QSA attention expected hidden width {}, got {hidden_width}",
                self.main.hidden_size
            );
        }
        // A text decode after an image prompt has scalar positions, but the cached image
        // prefix owns per-token sectioned tables. Convert the scalar rows to selected tables
        // so cache mode stays consistent across the sequence lifetime.
        if self.indexer.cache().has_position_tables(sequence_id) {
            let indices = Tensor::from_slice(positions, tokens, cos.device())?;
            return self.forward_chunk_with_position_tables(
                sequence_id,
                hidden,
                positions,
                &cos.index_select(&indices, 0)?,
                &sin.index_select(&indices, 0)?,
            );
        }

        let kv_snapshot = self.kv_cache.snapshot(sequence_id);
        let indexer_snapshot = self.indexer.cache().snapshot(sequence_id);
        let result = (|| {
            let (query, key, value, gate) = self.main.project(hidden)?;
            let (query, key) = self.rotary.apply(&query, &key, positions, cos, sin)?;
            self.kv_cache
                .append(sequence_id, &key.squeeze(0)?, &value.squeeze(0)?)?;
            let selections =
                self.indexer
                    .process_chunk(sequence_id, hidden, positions, cos, sin)?;
            let (cached_keys, cached_values) = self.kv_cache.get(sequence_id).ok_or_else(|| {
                Error::msg("Qwen4Exp QSA main K/V cache entry disappeared after append")
            })?;
            let gathered = self
                .kv_gather
                .gather(cached_keys, cached_values, &selections)?;
            let attended = self.attention.forward(&query, &gathered)?;
            let attended = attended.squeeze(0)?.transpose(0, 1)?.reshape((
                1,
                tokens,
                self.attention.query_heads * self.attention.value_head_dim,
            ))?;
            self.output.forward(&attended, &gate)
        })();

        if result.is_err() {
            self.kv_cache.restore(sequence_id, &kv_snapshot)?;
            self.indexer
                .cache_mut()
                .restore(sequence_id, &indexer_snapshot)?;
        }
        result
    }

    /// Process one sequence with already-selected section-aware MRoPE tables.
    pub(crate) fn forward_chunk_with_position_tables(
        &mut self,
        sequence_id: usize,
        hidden: &Tensor,
        positions: &[u32],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let (batch, tokens, hidden_width) = hidden.dims3()?;
        let half = self.rotary.rotary_dim / 2;
        if batch != 1
            || positions.len() != tokens
            || hidden_width != self.main.hidden_size
            || self.main.head_dim != self.attention.value_head_dim
            || cos.dims() != [tokens, half]
            || sin.dims() != cos.dims()
        {
            candle_core::bail!(
                "Qwen4Exp QSA attention received incompatible sectioned MRoPE inputs"
            );
        }
        let kv_snapshot = self.kv_cache.snapshot(sequence_id);
        let indexer_snapshot = self.indexer.cache().snapshot(sequence_id);
        let result = (|| {
            let (query, key, value, gate) = self.main.project(hidden)?;
            let (query, key) = self.rotary.apply_position_tables(&query, &key, cos, sin)?;
            self.kv_cache
                .append(sequence_id, &key.squeeze(0)?, &value.squeeze(0)?)?;
            let selections = self.indexer.process_chunk_with_position_tables(
                sequence_id,
                hidden,
                positions,
                cos,
                sin,
            )?;
            let (cached_keys, cached_values) = self.kv_cache.get(sequence_id).ok_or_else(|| {
                Error::msg("Qwen4Exp QSA main K/V cache entry disappeared after append")
            })?;
            let gathered = self
                .kv_gather
                .gather(cached_keys, cached_values, &selections)?;
            let attended = self.attention.forward(&query, &gathered)?;
            let attended = attended.squeeze(0)?.transpose(0, 1)?.reshape((
                1,
                tokens,
                self.attention.query_heads * self.attention.value_head_dim,
            ))?;
            self.output.forward(&attended, &gate)
        })();
        if result.is_err() {
            self.kv_cache.restore(sequence_id, &kv_snapshot)?;
            self.indexer
                .cache_mut()
                .restore(sequence_id, &indexer_snapshot)?;
        }
        result
    }

    /// Process packed chunks in input order while preserving independent sequence histories.
    pub(crate) fn forward_packed_chunks(
        &mut self,
        chunks: &[(usize, &Tensor, &[u32])],
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Vec<Tensor>> {
        if chunks.is_empty() {
            candle_core::bail!("Qwen4Exp packed QSA attention requires at least one chunk");
        }
        if self.main.head_dim != self.attention.value_head_dim {
            candle_core::bail!(
                "Qwen4Exp QSA attention requires matching key and value head dimensions for the per-head output gate"
            );
        }
        for (sequence_id, hidden, positions) in chunks {
            let (batch, tokens, hidden_width) = hidden.dims3()?;
            if batch != 1 || positions.len() != tokens {
                candle_core::bail!(
                    "Qwen4Exp packed QSA sequence {sequence_id} expects one sequence and one position per token, got batch {batch}, {tokens} tokens, and {} positions",
                    positions.len()
                );
            }
            if hidden_width != self.main.hidden_size {
                candle_core::bail!(
                    "Qwen4Exp packed QSA sequence {sequence_id} expected hidden width {}, got {hidden_width}",
                    self.main.hidden_size
                );
            }
        }

        // Snapshot every touched sequence once, before any mutation, so a failure
        // can restore each sequence's exact prior present or absent state.
        let mut snapshots = HashMap::new();
        for (sequence_id, _, _) in chunks {
            snapshots.entry(*sequence_id).or_insert_with(|| {
                (
                    self.kv_cache.snapshot(*sequence_id),
                    self.indexer.cache().snapshot(*sequence_id),
                )
            });
        }
        let result = (|| {
            let mut outputs = Vec::with_capacity(chunks.len());
            for (sequence_id, hidden, positions) in chunks {
                outputs.push(self.forward_chunk(*sequence_id, hidden, positions, cos, sin)?);
            }
            Ok(outputs)
        })();
        if result.is_err() {
            for (sequence_id, (kv_snapshot, indexer_snapshot)) in snapshots {
                self.kv_cache.restore(sequence_id, &kv_snapshot)?;
                self.indexer
                    .cache_mut()
                    .restore(sequence_id, &indexer_snapshot)?;
            }
        }
        result
    }

    pub(crate) fn kv_cache(&self) -> &QsaMainKvCache {
        &self.kv_cache
    }

    pub(crate) fn indexer(&self) -> &QsaIndexer {
        &self.indexer
    }
}

impl QsaBlockSelector {
    pub(crate) fn new(compress_ratio: usize, token_budget: usize) -> Result<Self> {
        if compress_ratio == 0 || token_budget == 0 || !token_budget.is_multiple_of(compress_ratio)
        {
            candle_core::bail!(
                "Qwen4Exp QSA requires a positive token budget divisible by the compression ratio"
            );
        }
        Ok(Self {
            compress_ratio,
            token_budget,
        })
    }

    pub(crate) fn select(&self, visible_tokens: &[u32], block_scores: &[f32]) -> Result<Vec<u32>> {
        let complete_blocks = visible_tokens.len() / self.compress_ratio;
        if block_scores.len() != complete_blocks {
            candle_core::bail!(
                "Qwen4Exp QSA expected {complete_blocks} block scores for {} visible tokens, got {}",
                visible_tokens.len(),
                block_scores.len()
            );
        }
        if block_scores.iter().any(|score| score.is_nan()) {
            return Err(Error::msg("Qwen4Exp QSA block scores must not contain NaN"));
        }

        let selected_blocks = complete_blocks.min(self.token_budget / self.compress_ratio);
        let mut ranked = block_scores.iter().copied().enumerate().collect::<Vec<_>>();
        // Prefer the earlier block on an exact score tie. This keeps CPU selection deterministic.
        ranked.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_index.cmp(right_index))
        });

        let tail_start = complete_blocks * self.compress_ratio;
        let output_capacity = selected_blocks
            .checked_mul(self.compress_ratio)
            .and_then(|selected| selected.checked_add(visible_tokens.len() - tail_start))
            .ok_or_else(|| Error::msg("Qwen4Exp QSA selected token count overflow"))?;
        let mut selected = Vec::with_capacity(output_capacity);
        for (block, _) in ranked.into_iter().take(selected_blocks) {
            let start = block * self.compress_ratio;
            selected.extend_from_slice(&visible_tokens[start..start + self.compress_ratio]);
        }
        selected.extend_from_slice(&visible_tokens[tail_start..]);
        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp};
    use candle_nn::Linear;
    use mistralrs_quant::{QuantMethodConfig, UnquantLinear};

    fn projection(weight: Tensor) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(weight, None)),
        )?))
    }

    fn indexer_projection() -> Result<QsaIndexerProjection> {
        let device = &Device::Cpu;
        Ok(QsaIndexerProjection {
            query: projection(Tensor::from_slice(
                &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0],
                (4, 2),
                device,
            )?)?,
            key: projection(Tensor::from_slice(
                &[2.0f32, 0.0, 0.0, 3.0],
                (2, 2),
                device,
            )?)?,
            query_norm: Tensor::from_slice(&[2.0f32, 0.5], 2, device)?,
            key_norm: Tensor::from_slice(&[0.5f32, 2.0], 2, device)?,
            norm_eps: 0.0,
            hidden_size: 2,
            head_count: 2,
            head_dim: 2,
        })
    }

    fn indexer() -> Result<QsaIndexer> {
        Ok(QsaIndexer {
            projection: indexer_projection()?,
            scorer: QsaIndexerScorer::new(2, 2, 2)?,
            rotary: QsaIndexerRotary::new(2, 2)?,
            selector: QsaBlockSelector::new(2, 2)?,
            cache: QsaSequenceCache::new(2)?,
        })
    }

    fn main_projection(layout: QsaQueryGateLayout) -> Result<QsaMainProjection> {
        let device = &Device::Cpu;
        Ok(QsaMainProjection {
            query_gate: projection(Tensor::from_slice(
                &[
                    1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0,
                    4.0,
                ],
                (8, 2),
                device,
            )?)?,
            key: projection(Tensor::from_slice(
                &[1.0f32, 0.0, 0.0, 1.0],
                (2, 2),
                device,
            )?)?,
            value: projection(Tensor::from_slice(
                &[2.0f32, 0.0, 0.0, 3.0],
                (2, 2),
                device,
            )?)?,
            query_norm: Tensor::ones(2, DType::F32, device)?,
            key_norm: Tensor::ones(2, DType::F32, device)?,
            norm_eps: 0.0,
            hidden_size: 2,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            layout,
        })
    }

    #[test]
    fn sequence_cache_appends_and_truncates_keys_with_positions() -> Result<()> {
        let device = &Device::Cpu;
        let mut cache = QsaSequenceCache::new(2)?;
        cache.append(
            7,
            &Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?,
            &[10, 11],
        )?;
        cache.append(
            7,
            &Tensor::from_slice(&[5.0f32, 6.0], (1, 2), device)?,
            &[12],
        )?;
        let (keys, positions) = cache.get(7).unwrap();
        assert_eq!(positions, &[10, 11, 12]);
        assert_eq!(
            keys.to_vec2::<f32>()?,
            vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
        );

        cache.truncate(7, 1)?;
        let (keys, positions) = cache.get(7).unwrap();
        assert_eq!(positions, &[10]);
        assert_eq!(keys.to_vec2::<f32>()?, vec![vec![1.0, 2.0]]);
        Ok(())
    }

    #[test]
    fn sequence_cache_snapshot_restore_and_release_are_isolated() -> Result<()> {
        let device = &Device::Cpu;
        let mut cache = QsaSequenceCache::new(2)?;
        let absent = cache.snapshot(3);
        cache.append(
            3,
            &Tensor::from_slice(&[1.0f32, 2.0], (1, 2), device)?,
            &[4],
        )?;
        let committed = cache.snapshot(3);
        cache.append(
            3,
            &Tensor::from_slice(&[3.0f32, 4.0], (1, 2), device)?,
            &[5],
        )?;
        cache.append(
            9,
            &Tensor::from_slice(&[9.0f32, 9.0], (1, 2), device)?,
            &[9],
        )?;

        cache.restore(3, &committed)?;
        assert_eq!(cache.get(3).unwrap().1, &[4]);
        assert_eq!(cache.get(9).unwrap().1, &[9]);
        cache.restore(3, &absent)?;
        assert!(cache.get(3).is_none());
        assert!(cache.release(9));
        assert!(!cache.release(9));

        cache.append(
            3,
            &Tensor::from_slice(&[3.0f32, 3.0], (1, 2), device)?,
            &[3],
        )?;
        cache.reset(3);
        assert!(cache.get(3).is_none());
        cache.append(
            3,
            &Tensor::from_slice(&[3.0f32, 3.0], (1, 2), device)?,
            &[3],
        )?;
        cache.clear();
        assert!(cache.get(3).is_none());
        Ok(())
    }

    #[test]
    fn sequence_cache_rejects_invalid_inventory_without_mutation() -> Result<()> {
        let device = &Device::Cpu;
        let mut cache = QsaSequenceCache::new(2)?;
        cache.append(
            1,
            &Tensor::from_slice(&[1.0f32, 2.0], (1, 2), device)?,
            &[3],
        )?;
        let error = cache
            .append(1, &Tensor::from_slice(&[3.0f32, 4.0], (1, 2), device)?, &[])
            .unwrap_err();
        assert!(error.to_string().contains("one position per token"));
        assert_eq!(cache.get(1).unwrap().1, &[3]);
        assert!(cache.truncate(1, 2).is_err());
        assert_eq!(cache.get(1).unwrap().1, &[3]);
        Ok(())
    }

    #[test]
    fn main_kv_cache_stores_contiguous_rows_for_gathering() -> Result<()> {
        let device = &Device::Cpu;
        // Projection values reach the cache as transposed head-major views.
        let values = Tensor::from_vec(
            (0..48).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 4, 12),
            device,
        )?;
        let values = values.reshape((1, 4, 2, 6))?.transpose(1, 2)?.squeeze(0)?;
        assert!(!values.is_contiguous());
        let keys = Tensor::ones((2, 4, 6), DType::F32, device)?;

        let mut cache = QsaMainKvCache::new(2, 6, 6)?;
        cache.append(7, &keys, &values)?;
        let (cached_keys, cached_values) = cache.get(7).expect("cache entry after append");
        assert!(cached_values.is_contiguous());

        let gather = QsaKvGather::new(2, 6, 6, 8, 4)?;
        let gathered = gather.gather(cached_keys, cached_values, &[vec![0, 2, 3]])?;
        assert_eq!(gathered.len(), 1);
        let (gathered_keys, gathered_values) = &gathered[0];
        assert_eq!(gathered_keys.dims(), [2, 3, 6]);
        assert_eq!(gathered_values.dims(), [2, 3, 6]);
        let gathered_rows = gathered_values.to_vec3::<f32>()?;
        assert_eq!(gathered_rows[0][0], values.i((0, 0))?.to_vec1::<f32>()?);
        assert_eq!(gathered_rows[0][1], values.i((0, 2))?.to_vec1::<f32>()?);
        assert_eq!(gathered_rows[0][2], values.i((0, 3))?.to_vec1::<f32>()?);
        Ok(())
    }

    #[test]
    fn indexer_rotary_uses_query_and_block_start_positions() -> Result<()> {
        let rotary = QsaIndexerRotary::new(2, 2)?;
        let device = &Device::Cpu;
        let query = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], (2, 2), device)?;
        let keys = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?;
        let cos = Tensor::from_slice(&[1.0f32, 0.0, -1.0], (3, 1), device)?;
        let sin = Tensor::from_slice(&[0.0f32, 1.0, 0.0], (3, 1), device)?;
        let (query, keys) = rotary.apply(&query, &keys, 1, &[0, 2], &cos, &sin)?;
        assert_eq!(
            query.to_vec2::<f32>()?,
            vec![vec![0.0, 1.0], vec![-1.0, 0.0]]
        );
        assert_eq!(
            keys.to_vec2::<f32>()?,
            vec![vec![1.0, 2.0], vec![-3.0, -4.0]]
        );

        let partial = QsaIndexerRotary::new(4, 2)?;
        let input = Tensor::from_slice(&[1.0f32, 2.0, 7.0, 8.0], (1, 4), device)?;
        let rotated = partial.rotate(
            &input,
            &Tensor::zeros((1, 1), DType::F32, device)?,
            &Tensor::ones((1, 1), DType::F32, device)?,
        )?;
        assert_eq!(rotated.to_vec2::<f32>()?, vec![vec![-2.0, 1.0, 7.0, 8.0]]);
        Ok(())
    }

    #[test]
    fn indexer_rotary_rejects_invalid_positions_and_dimensions() -> Result<()> {
        assert!(QsaIndexerRotary::new(3, 3).is_err());
        let rotary = QsaIndexerRotary::new(2, 2)?;
        let device = &Device::Cpu;
        let query = Tensor::zeros((2, 2), DType::F32, device)?;
        let keys = Tensor::zeros((1, 2), DType::F32, device)?;
        let cos = Tensor::ones((2, 1), DType::F32, device)?;
        let sin = Tensor::zeros((2, 1), DType::F32, device)?;
        assert!(rotary
            .apply(&query, &keys, 2, &[0], &cos, &sin)
            .unwrap_err()
            .to_string()
            .contains("outside the frequency cache"));
        assert!(rotary
            .apply(&query, &keys, 0, &[], &cos, &sin)
            .unwrap_err()
            .to_string()
            .contains("incompatible"));
        Ok(())
    }

    #[test]
    fn main_rotary_matches_small_interleaved_reference() -> Result<()> {
        let rotary = QsaMainRotary::new(2, 2)?;
        let device = &Device::Cpu;
        let query = Tensor::from_slice(
            &[
                1.0f32, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 0.0, 4.0, 5.0, 0.0,
            ],
            (1, 2, 3, 2),
            device,
        )?;
        let key = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 1, 3, 2), device)?;
        let cos = Tensor::from_slice(&[1.0f32, 0.0, -1.0], (3, 1), device)?;
        let sin = Tensor::from_slice(&[0.0f32, 1.0, 0.0], (3, 1), device)?;
        let (query, key) = rotary.apply(&query, &key, &[0, 1, 2], &cos, &sin)?;
        assert_eq!(
            query.flatten_all()?.to_vec1::<f32>()?,
            vec![1.0, 0.0, -1.0, 0.0, -2.0, 0.0, 3.0, 0.0, -4.0, 0.0, -5.0, 0.0]
        );
        assert_eq!(
            key.flatten_all()?.to_vec1::<f32>()?,
            vec![1.0, 2.0, -4.0, 3.0, -5.0, -6.0]
        );

        // The same position rotates the query and every KV head, including batched inputs.
        let half_root = 0.5f32.sqrt();
        let cos = Tensor::from_slice(&[half_root], (1, 1), device)?;
        let sin = Tensor::from_slice(&[half_root], (1, 1), device)?;
        let query = Tensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0], (2, 1, 1, 2), device)?;
        let key = Tensor::from_slice(&[1.0f32, 2.0, 1.0, 2.0], (2, 1, 1, 2), device)?;
        let (query, key) = rotary.apply(&query, &key, &[0], &cos, &sin)?;
        let query = query.flatten_all()?.to_vec1::<f32>()?;
        assert!((query[0] - half_root).abs() < 1e-6);
        assert!((query[1] - half_root).abs() < 1e-6);
        assert_eq!(query[0], query[2]);
        assert_eq!(query[1], query[3]);
        let key = key.flatten_all()?.to_vec1::<f32>()?;
        assert!((key[0] + half_root).abs() < 1e-6);
        assert!((key[1] - 3.0 * half_root).abs() < 1e-6);
        assert_eq!(key[0], key[2]);
        assert_eq!(key[1], key[3]);
        Ok(())
    }

    #[test]
    fn main_rotary_preserves_unrotated_trailing_dimensions() -> Result<()> {
        let rotary = QsaMainRotary::new(4, 2)?;
        let device = &Device::Cpu;
        let query = Tensor::from_slice(&[1.0f32, 2.0, 7.0, 8.0], (1, 1, 1, 4), device)?;
        let key = Tensor::from_slice(&[-1.0f32, -2.0, -7.0, -8.0], (1, 1, 1, 4), device)?;
        let cos = Tensor::from_slice(&[0.0f32], (1, 1), device)?;
        let sin = Tensor::from_slice(&[1.0f32], (1, 1), device)?;
        let (query, key) = rotary.apply(&query, &key, &[0], &cos, &sin)?;
        assert_eq!(
            query.flatten_all()?.to_vec1::<f32>()?,
            vec![-2.0, 1.0, 7.0, 8.0]
        );
        assert_eq!(
            key.flatten_all()?.to_vec1::<f32>()?,
            vec![2.0, -1.0, -7.0, -8.0]
        );
        Ok(())
    }

    #[test]
    fn main_rotary_position_tables_match_apply() -> Result<()> {
        let rotary = QsaMainRotary::new(4, 4)?;
        let device = &Device::Cpu;
        let query = Tensor::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            (1, 2, 2, 4),
            device,
        )?;
        let key = Tensor::from_slice(
            &[1.0f32, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            (1, 1, 2, 4),
            device,
        )?;
        // A four-position cache; the two tokens sit at positions 1 and 3.
        let theta = 10.0f32;
        let mut cache_cos = Vec::with_capacity(8);
        let mut cache_sin = Vec::with_capacity(8);
        for position in 0u32..4 {
            for pair in 0..2usize {
                let angle = position as f32 * theta.powf(-(2.0 * pair as f32) / 4.0);
                cache_cos.push(angle.cos());
                cache_sin.push(angle.sin());
            }
        }
        let cache_cos = Tensor::from_vec(cache_cos, (4, 2), device)?;
        let cache_sin = Tensor::from_vec(cache_sin, (4, 2), device)?;
        let mut table_cos = Vec::with_capacity(4);
        let mut table_sin = Vec::with_capacity(4);
        for position in [1u32, 3] {
            let index = Tensor::from_slice(&[position], 1, device)?;
            let selected_cos = cache_cos.index_select(&index, 0)?.to_vec2::<f32>()?;
            let selected_sin = cache_sin.index_select(&index, 0)?.to_vec2::<f32>()?;
            table_cos.extend(selected_cos[0].iter().copied());
            table_sin.extend(selected_sin[0].iter().copied());
        }
        let table_cos = Tensor::from_vec(table_cos, (2, 2), device)?;
        let table_sin = Tensor::from_vec(table_sin, (2, 2), device)?;
        let (apply_query, apply_key) =
            rotary.apply(&query, &key, &[1, 3], &cache_cos, &cache_sin)?;
        let (table_query, table_key) =
            rotary.apply_position_tables(&query, &key, &table_cos, &table_sin)?;
        assert_eq!(
            apply_query.flatten_all()?.to_vec1::<f32>()?,
            table_query.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            apply_key.flatten_all()?.to_vec1::<f32>()?,
            table_key.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn main_rotary_rejects_incompatible_inventory() -> Result<()> {
        assert!(QsaMainRotary::new(3, 3).is_err());
        assert!(QsaMainRotary::new(2, 4).is_err());
        assert!(QsaMainRotary::new(0, 2).is_err());
        let rotary = QsaMainRotary::new(2, 2)?;
        let device = &Device::Cpu;
        let query = Tensor::zeros((2, 2, 2, 2), DType::F32, device)?;
        let key = Tensor::zeros((2, 1, 2, 2), DType::F32, device)?;
        let cos = Tensor::ones((2, 1), DType::F32, device)?;
        let sin = Tensor::zeros((2, 1), DType::F32, device)?;
        assert!(rotary
            .apply(&query, &key, &[0, 1, 2], &cos, &sin)
            .unwrap_err()
            .to_string()
            .contains("incompatible"));
        let narrow_cos = Tensor::ones((2, 2), DType::F32, device)?;
        assert!(rotary
            .apply(&query, &key, &[0, 1], &narrow_cos, &sin)
            .unwrap_err()
            .to_string()
            .contains("incompatible"));
        assert!(rotary
            .apply(&query, &key, &[0, 2], &cos, &sin)
            .unwrap_err()
            .to_string()
            .contains("outside the frequency cache"));
        let short_key = Tensor::zeros((2, 1, 1, 2), DType::F32, device)?;
        assert!(rotary
            .apply(&query, &short_key, &[0, 1], &cos, &sin)
            .unwrap_err()
            .to_string()
            .contains("incompatible"));
        let wide_query = Tensor::zeros((2, 2, 2, 3), DType::F32, device)?;
        let wide_key = Tensor::zeros((2, 1, 2, 3), DType::F32, device)?;
        assert!(rotary
            .apply(&wide_query, &wide_key, &[0, 1], &cos, &sin)
            .unwrap_err()
            .to_string()
            .contains("incompatible"));
        Ok(())
    }

    #[test]
    fn main_kv_cache_appends_and_truncates_keys_and_values() -> Result<()> {
        let device = &Device::Cpu;
        let mut cache = QsaMainKvCache::new(1, 2, 2)?;
        cache.append(
            7,
            &Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (1, 2, 2), device)?,
            &Tensor::from_slice(&[-1.0f32, -2.0, -3.0, -4.0], (1, 2, 2), device)?,
        )?;
        cache.append(
            7,
            &Tensor::from_slice(&[5.0f32, 6.0], (1, 1, 2), device)?,
            &Tensor::from_slice(&[-5.0f32, -6.0], (1, 1, 2), device)?,
        )?;
        let (keys, values) = cache.get(7).unwrap();
        assert_eq!(keys.dim(1)?, 3);
        assert_eq!(values.dim(1)?, 3);
        assert_eq!(
            keys.flatten_all()?.to_vec1::<f32>()?,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            values.flatten_all()?.to_vec1::<f32>()?,
            vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        );

        cache.truncate(7, 1)?;
        let (keys, values) = cache.get(7).unwrap();
        assert_eq!(keys.dim(1)?, 1);
        assert_eq!(values.dim(1)?, 1);
        assert_eq!(keys.flatten_all()?.to_vec1::<f32>()?, vec![1.0, 2.0]);
        assert_eq!(values.flatten_all()?.to_vec1::<f32>()?, vec![-1.0, -2.0]);
        cache.truncate(7, 0)?;
        assert!(cache.get(7).is_some());
        Ok(())
    }

    #[test]
    fn main_kv_cache_snapshot_restore_and_release_are_isolated() -> Result<()> {
        let device = &Device::Cpu;
        let mut cache = QsaMainKvCache::new(1, 2, 2)?;
        let absent = cache.snapshot(3);
        cache.append(
            3,
            &Tensor::from_slice(&[1.0f32, 2.0], (1, 1, 2), device)?,
            &Tensor::from_slice(&[-1.0f32, -2.0], (1, 1, 2), device)?,
        )?;
        let committed = cache.snapshot(3);
        cache.append(
            3,
            &Tensor::from_slice(&[3.0f32, 4.0], (1, 1, 2), device)?,
            &Tensor::from_slice(&[-3.0f32, -4.0], (1, 1, 2), device)?,
        )?;
        cache.append(
            9,
            &Tensor::from_slice(&[9.0f32, 9.0], (1, 1, 2), device)?,
            &Tensor::from_slice(&[-9.0f32, -9.0], (1, 1, 2), device)?,
        )?;

        cache.restore(3, &committed)?;
        let (keys, values) = cache.get(3).unwrap();
        assert_eq!(keys.flatten_all()?.to_vec1::<f32>()?, vec![1.0, 2.0]);
        assert_eq!(values.flatten_all()?.to_vec1::<f32>()?, vec![-1.0, -2.0]);
        assert_eq!(cache.get(9).unwrap().0.dim(1)?, 1);
        cache.restore(3, &absent)?;
        assert!(cache.get(3).is_none());
        assert!(cache.release(9));
        assert!(!cache.release(9));

        cache.append(
            3,
            &Tensor::from_slice(&[3.0f32, 3.0], (1, 1, 2), device)?,
            &Tensor::from_slice(&[-3.0f32, -3.0], (1, 1, 2), device)?,
        )?;
        cache.reset(3);
        assert!(cache.get(3).is_none());
        cache.append(
            3,
            &Tensor::from_slice(&[3.0f32, 3.0], (1, 1, 2), device)?,
            &Tensor::from_slice(&[-3.0f32, -3.0], (1, 1, 2), device)?,
        )?;
        cache.clear();
        assert!(cache.get(3).is_none());
        Ok(())
    }

    #[test]
    fn main_kv_cache_rejects_invalid_inventory_without_mutation() -> Result<()> {
        assert!(QsaMainKvCache::new(0, 2, 2).is_err());
        let device = &Device::Cpu;
        let mut cache = QsaMainKvCache::new(1, 2, 2)?;
        cache.append(
            1,
            &Tensor::from_slice(&[1.0f32, 2.0], (1, 1, 2), device)?,
            &Tensor::from_slice(&[-1.0f32, -2.0], (1, 1, 2), device)?,
        )?;

        // Key and value token counts must agree.
        let error = cache
            .append(
                1,
                &Tensor::from_slice(&[3.0f32, 4.0], (1, 1, 2), device)?,
                &Tensor::from_slice(&[-3.0f32, -4.0, -3.0, -4.0], (1, 2, 2), device)?,
            )
            .unwrap_err();
        assert!(error.to_string().contains("main K/V cache"));
        // Head counts and head dimensions must match the cache.
        assert!(cache
            .append(
                1,
                &Tensor::from_slice(&[3.0f32, 4.0, 5.0, 6.0], (2, 1, 2), device)?,
                &Tensor::from_slice(&[-3.0f32, -4.0], (1, 1, 2), device)?,
            )
            .is_err());
        assert!(cache
            .append(
                1,
                &Tensor::from_slice(&[3.0f32, 4.0, 5.0], (1, 1, 3), device)?,
                &Tensor::from_slice(&[-3.0f32, -4.0], (1, 1, 2), device)?,
            )
            .is_err());
        // Appended dtypes must match the existing entry.
        assert!(cache
            .append(
                1,
                &Tensor::from_slice(&[3.0f64, 4.0], (1, 1, 2), device)?,
                &Tensor::from_slice(&[-3.0f64, -4.0], (1, 1, 2), device)?,
            )
            .is_err());

        let mut incompatible = cache.snapshot(1);
        incompatible.kv_heads = 2;
        assert!(cache.restore(1, &incompatible).is_err());
        let partial = QsaMainKvSnapshot {
            keys: Some(Tensor::from_slice(&[7.0f32], (1, 1, 1), device)?),
            values: None,
            kv_heads: 1,
            key_head_dim: 2,
            value_head_dim: 2,
        };
        assert!(cache.restore(1, &partial).is_err());

        assert!(cache.truncate(1, 2).is_err());
        assert!(cache.truncate(2, 1).is_err());

        let (keys, values) = cache.get(1).unwrap();
        assert_eq!(keys.dim(1)?, 1);
        assert_eq!(keys.flatten_all()?.to_vec1::<f32>()?, vec![1.0, 2.0]);
        assert_eq!(values.flatten_all()?.to_vec1::<f32>()?, vec![-1.0, -2.0]);
        Ok(())
    }

    #[test]
    fn indexer_projection_normalizes_queries_but_keeps_keys_raw() -> Result<()> {
        let projection = indexer_projection()?;
        let hidden = Tensor::from_slice(&[3.0f32, 4.0], (1, 1, 2), &Device::Cpu)?;
        let (query, raw_key) = projection.project(&hidden)?;
        let query = query.flatten_all()?.to_vec1::<f32>()?;
        let expected = [1.6970563f32, 0.56568545, 2.8, -0.1];
        for (actual, expected) in query.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert_eq!(raw_key.flatten_all()?.to_vec1::<f32>()?, vec![6.0, 12.0]);

        let pooled = Tensor::from_slice(&[6.0f32, 12.0], (1, 2), &Device::Cpu)?;
        let normalized = projection
            .normalize_pooled_keys(&pooled)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = [0.31622776f32, 2.529822];
        for (actual, expected) in normalized.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn indexer_projection_rejects_incompatible_hidden_width() -> Result<()> {
        let projection = indexer_projection()?;
        let hidden = Tensor::zeros((1, 2, 3), DType::F32, &Device::Cpu)?;
        assert!(projection
            .project(&hidden)
            .unwrap_err()
            .to_string()
            .contains("expected hidden width 2"));
        Ok(())
    }

    #[test]
    fn main_projection_interleaved_layout_splits_and_normalizes_qkv() -> Result<()> {
        let projection = main_projection(QsaQueryGateLayout::InterleavedPerHead)?;
        let hidden = Tensor::from_slice(&[3.0f32, 4.0], (1, 1, 2), &Device::Cpu)?;
        let (query, key, value, gate) = projection.project(&hidden)?;
        let query = query.flatten_all()?.to_vec1::<f32>()?;
        let expected_query = [0.84852815f32, 1.1313709, 0.84852815, 1.1313709];
        for (actual, expected) in query.iter().zip(expected_query) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert_eq!(
            gate.flatten_all()?.to_vec1::<f32>()?,
            vec![6.0, 8.0, 12.0, 16.0]
        );
        assert_eq!(
            key.flatten_all()?.to_vec1::<f32>()?,
            vec![0.84852815, 1.1313709]
        );
        assert_eq!(value.flatten_all()?.to_vec1::<f32>()?, vec![6.0, 12.0]);
        Ok(())
    }

    #[test]
    fn main_projection_grouped_layout_has_named_distinct_split() -> Result<()> {
        let projection = main_projection(QsaQueryGateLayout::Grouped)?;
        let hidden = Tensor::from_slice(&[3.0f32, 4.0], (1, 1, 2), &Device::Cpu)?;
        let (query, _, _, gate) = projection.project(&hidden)?;
        let query = query.flatten_all()?.to_vec1::<f32>()?;
        let expected_query = [0.84852815f32, 1.1313709, 0.84852815, 1.1313709];
        for (actual, expected) in query.iter().zip(expected_query) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        assert_eq!(
            gate.flatten_all()?.to_vec1::<f32>()?,
            vec![9.0, 12.0, 12.0, 16.0]
        );
        Ok(())
    }

    #[test]
    fn main_projection_rejects_incompatible_hidden_width() -> Result<()> {
        let projection = main_projection(QsaQueryGateLayout::InterleavedPerHead)?;
        let hidden = Tensor::zeros((1, 1, 3), DType::F32, &Device::Cpu)?;
        assert!(projection
            .project(&hidden)
            .unwrap_err()
            .to_string()
            .contains("expected hidden width 2"));
        Ok(())
    }

    #[test]
    fn output_projection_applies_reference_sigmoid_gate() -> Result<()> {
        let device = &Device::Cpu;
        let layer = QsaOutputProjection {
            output: projection(Tensor::from_slice(
                &[1.0f32, 2.0, 3.0, 4.0],
                (2, 2),
                device,
            )?)?,
            attention_width: 2,
            hidden_size: 2,
        };
        let attention = Tensor::from_slice(&[2.0f32, 2.0, 4.0, 4.0], (1, 2, 2), device)?;
        let gate = Tensor::from_slice(&[0.0f32, 4.0, 0.0, -4.0], (1, 2, 2), device)?;
        let output = layer.forward(&attention, &gate)?;

        // sigmoid(0) = 0.5, sigmoid(4) ~ 0.982014, sigmoid(-4) ~ 0.017986.
        // UnquantLinear computes x @ W^T, so each output row uses one weight row.
        let expected = [
            2.0 * 0.5 * 1.0 + 2.0 * 0.982014 * 2.0,
            2.0 * 0.5 * 3.0 + 2.0 * 0.982014 * 4.0,
            4.0 * 0.5 * 1.0 + 4.0 * 0.017986 * 2.0,
            4.0 * 0.5 * 3.0 + 4.0 * 0.017986 * 4.0,
        ];
        let output = output.flatten_all()?.to_vec1::<f32>()?;
        for (actual, expected) in output.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn output_projection_rejects_incompatible_inventory() -> Result<()> {
        let device = &Device::Cpu;
        let layer = QsaOutputProjection {
            output: projection(Tensor::from_slice(&[1.0f32, 2.0], (2, 1), device)?)?,
            attention_width: 2,
            hidden_size: 1,
        };
        let attention = Tensor::zeros((1, 2, 2), DType::F32, device)?;
        let narrow_gate = Tensor::zeros((1, 2, 1), DType::F32, device)?;
        let mismatched_batch = Tensor::zeros((2, 2, 2), DType::F32, device)?;

        assert!(layer
            .forward(&attention, &narrow_gate)
            .unwrap_err()
            .to_string()
            .contains("output projection"));
        assert!(layer
            .forward(&attention, &mismatched_batch)
            .unwrap_err()
            .to_string()
            .contains("output projection"));
        let wide_attention = Tensor::zeros((1, 2, 3), DType::F32, device)?;
        let wide_gate = Tensor::zeros((1, 2, 3), DType::F32, device)?;
        assert!(layer
            .forward(&wide_attention, &wide_gate)
            .unwrap_err()
            .to_string()
            .contains("output projection"));
        Ok(())
    }

    fn attention_layer() -> Result<QsaAttention> {
        let device = &Device::Cpu;
        let output = QsaOutputProjection {
            output: projection(Tensor::from_slice(
                &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                (2, 4),
                device,
            )?)?,
            attention_width: 4,
            hidden_size: 2,
        };
        Ok(QsaAttention {
            main: main_projection(QsaQueryGateLayout::Grouped)?,
            rotary: QsaMainRotary::new(2, 2)?,
            output,
            kv_cache: QsaMainKvCache::new(1, 2, 2)?,
            indexer: indexer()?,
            kv_gather: QsaKvGather::new(1, 2, 2, 2, 2)?,
            attention: QsaSparseAttention::new(2, 1, 2, 2)?,
        })
    }

    #[test]
    fn attention_chunk_matches_manual_component_composition() -> Result<()> {
        let device = &Device::Cpu;
        let mut layer = attention_layer()?;
        let hidden = Tensor::from_slice(&[3.0f32, 4.0, 1.0, 2.0, 5.0, 6.0], (1, 3, 2), device)?;
        let positions = [0u32, 1, 2];
        let cos = Tensor::from_slice(&[1.0f32, 0.0, -1.0], (3, 1), device)?;
        let sin = Tensor::from_slice(&[0.0f32, 1.0, 0.0], (3, 1), device)?;
        let output = layer.forward_chunk(5, &hidden, &positions, &cos, &sin)?;
        assert_eq!(output.dims(), [1, 3, 2]);
        assert_eq!(layer.kv_cache().get(5).unwrap().0.dim(1)?, 3);
        assert_eq!(layer.indexer().cache().get(5).unwrap().1.len(), 3);

        // Recompute the same chunk from fresh components and compare exactly.
        let mut manual = attention_layer()?;
        let (query, key, value, gate) = manual.main.project(&hidden)?;
        let (query, key) = manual.rotary.apply(&query, &key, &positions, &cos, &sin)?;
        manual
            .kv_cache
            .append(5, &key.squeeze(0)?, &value.squeeze(0)?)?;
        let selections = manual
            .indexer
            .process_chunk(5, &hidden, &positions, &cos, &sin)?;
        let (cached_keys, cached_values) = manual.kv_cache.get(5).unwrap();
        let gathered = manual
            .kv_gather
            .gather(cached_keys, cached_values, &selections)?;
        let attended = manual.attention.forward(&query, &gathered)?;
        let attended = attended.squeeze(0)?.transpose(0, 1)?.reshape((1, 3, 4))?;
        let expected = manual.output.forward(&attended, &gate)?;
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );

        // Decode is one more token over the same cache.
        let decode_hidden = Tensor::from_slice(&[7.0f32, 8.0], (1, 1, 2), device)?;
        let decode_cos = Tensor::from_slice(&[1.0f32, 0.0, -1.0, 0.0], (4, 1), device)?;
        let decode_sin = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 1.0], (4, 1), device)?;
        let decode = layer.forward_chunk(5, &decode_hidden, &[3], &decode_cos, &decode_sin)?;
        assert_eq!(decode.dims(), [1, 1, 2]);
        assert_eq!(layer.kv_cache().get(5).unwrap().0.dim(1)?, 4);
        assert_eq!(layer.indexer().cache().get(5).unwrap().1.len(), 4);
        Ok(())
    }

    #[test]
    fn sectioned_attention_matches_scalar_tables_and_retains_indexer_rows() -> Result<()> {
        let device = &Device::Cpu;
        let hidden = Tensor::from_slice(&[3.0f32, 4.0, 1.0, 2.0, 5.0, 6.0], (1, 3, 2), device)?;
        let positions = [0u32, 1, 2];
        let cos = Tensor::from_slice(&[1.0f32, 0.0, -1.0], (3, 1), device)?;
        let sin = Tensor::from_slice(&[0.0f32, 1.0, 0.0], (3, 1), device)?;

        let mut scalar = attention_layer()?;
        let expected = scalar.forward_chunk(5, &hidden, &positions, &cos, &sin)?;

        let mut sectioned = attention_layer()?;
        let actual =
            sectioned.forward_chunk_with_position_tables(5, &hidden, &positions, &cos, &sin)?;
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(sectioned.kv_cache().get(5).unwrap().0.dim(1)?, 3);
        assert_eq!(sectioned.indexer().cache().get(5).unwrap().1, &positions);

        // The scalar decode path selects a table row when a sectioned image prefix exists.
        let decode_hidden = Tensor::from_slice(&[7.0f32, 8.0], (1, 1, 2), device)?;
        let decode_cos = Tensor::from_slice(&[1.0f32, 0.0, -1.0, 0.0], (4, 1), device)?;
        let decode_sin = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 1.0], (4, 1), device)?;
        let decoded = sectioned.forward_chunk(5, &decode_hidden, &[3], &decode_cos, &decode_sin)?;
        assert_eq!(decoded.dims(), [1, 1, 2]);
        assert_eq!(sectioned.indexer().cache().get(5).unwrap().1, &[0, 1, 2, 3]);
        Ok(())
    }

    #[test]
    fn attention_truncates_main_and_indexer_caches_together() -> Result<()> {
        let device = &Device::Cpu;
        let mut layer = attention_layer()?;
        let hidden = Tensor::from_slice(&[3.0f32, 4.0, 1.0, 2.0, 5.0, 6.0], (1, 3, 2), device)?;
        let cos = Tensor::from_slice(&[1.0f32, 0.0, -1.0], (3, 1), device)?;
        let sin = Tensor::from_slice(&[0.0f32, 1.0, 0.0], (3, 1), device)?;
        layer.forward_chunk(5, &hidden, &[0, 1, 2], &cos, &sin)?;

        layer.truncate_sequence(5, 2)?;
        assert_eq!(layer.kv_cache().get(5).unwrap().0.dim(1)?, 2);
        assert_eq!(layer.indexer().cache().get(5).unwrap().1, &[0, 1]);

        let kv_before = layer.kv_cache().snapshot(5);
        let indexer_before = layer.indexer().cache().snapshot(5);
        assert!(layer.truncate_sequence(5, 3).is_err());
        assert_eq!(
            layer.kv_cache().get(5).unwrap().0.to_vec3::<f32>()?,
            kv_before.keys.unwrap().to_vec3::<f32>()?
        );
        assert_eq!(
            layer.indexer().cache().get(5).unwrap().1,
            indexer_before.positions
        );
        Ok(())
    }

    #[test]
    fn attention_rejects_misaligned_truncation_without_mutation() -> Result<()> {
        let device = &Device::Cpu;
        let mut layer = attention_layer()?;
        let hidden = Tensor::from_slice(&[3.0f32, 4.0, 1.0, 2.0, 5.0, 6.0], (1, 3, 2), device)?;
        let cos = Tensor::from_slice(&[1.0f32, 0.0, -1.0], (3, 1), device)?;
        let sin = Tensor::from_slice(&[0.0f32, 1.0, 0.0], (3, 1), device)?;
        layer.forward_chunk(5, &hidden, &[0, 1, 2], &cos, &sin)?;
        layer.indexer.cache_mut().truncate(5, 2)?;

        assert!(layer.truncate_sequence(5, 1).is_err());
        assert_eq!(layer.kv_cache().get(5).unwrap().0.dim(1)?, 3);
        assert_eq!(layer.indexer().cache().get(5).unwrap().1, &[0, 1]);
        Ok(())
    }

    #[test]
    fn attention_chunk_rolls_back_both_caches_on_failure() -> Result<()> {
        let device = &Device::Cpu;
        let mut layer = attention_layer()?;
        let hidden = Tensor::from_slice(&[3.0f32, 4.0, 1.0, 2.0], (1, 2, 2), device)?;
        let cos = Tensor::from_slice(&[1.0f32, 0.0], (2, 1), device)?;
        let sin = Tensor::from_slice(&[0.0f32, 1.0], (2, 1), device)?;
        assert!(layer.forward_chunk(5, &hidden, &[0, 1], &cos, &sin).is_ok());

        // Pre-seed the indexer cache with an incompatible dtype so the indexer
        // fails after the main K/V append in the next chunk.
        layer.indexer.cache_mut().restore(
            5,
            &QsaSequenceSnapshot {
                raw_keys: Some(Tensor::zeros((1, 2), DType::F16, device)?),
                positions: vec![9],
                cos: None,
                sin: None,
                head_dim: 2,
            },
        )?;

        let result = layer.forward_chunk(5, &hidden, &[2, 3], &cos, &sin);
        assert!(result.is_err());
        // Both caches are restored to their exact prior state.
        assert_eq!(layer.kv_cache().get(5).unwrap().0.dim(1)?, 2);
        let (keys, positions) = layer.indexer.cache().get(5).unwrap();
        assert_eq!(keys.dtype(), DType::F16);
        assert_eq!(positions, &[9]);
        Ok(())
    }

    #[test]
    fn packed_attention_matches_independent_interleaved_chunks() -> Result<()> {
        let device = &Device::Cpu;
        let cos = Tensor::ones((8, 1), DType::F32, device)?;
        let sin = Tensor::zeros((8, 1), DType::F32, device)?;
        let first_a = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], (1, 2, 2), device)?;
        let first_b = Tensor::from_slice(&[1.0f32, 1.0], (1, 1, 2), device)?;
        let second_a = Tensor::from_slice(&[2.0f32, 1.0], (1, 1, 2), device)?;
        let chunks = [
            (1, &first_a, &[0, 1][..]),
            (2, &first_b, &[4][..]),
            (1, &second_a, &[2][..]),
        ];

        let mut packed = attention_layer()?;
        let actual = packed.forward_packed_chunks(&chunks, &cos, &sin)?;
        assert_eq!(actual.len(), 3);
        assert_eq!(actual[0].dims(), [1, 2, 2]);
        assert_eq!(actual[1].dims(), [1, 1, 2]);
        assert_eq!(actual[2].dims(), [1, 1, 2]);

        let mut independent = attention_layer()?;
        let expected = vec![
            independent.forward_chunk(1, &first_a, &[0, 1], &cos, &sin)?,
            independent.forward_chunk(2, &first_b, &[4], &cos, &sin)?,
            independent.forward_chunk(1, &second_a, &[2], &cos, &sin)?,
        ];
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_eq!(
                actual.flatten_all()?.to_vec1::<f32>()?,
                expected.flatten_all()?.to_vec1::<f32>()?
            );
        }
        assert_eq!(packed.kv_cache().get(1).unwrap().0.dim(1)?, 3);
        assert_eq!(packed.kv_cache().get(2).unwrap().0.dim(1)?, 1);
        assert_eq!(packed.indexer().cache().get(1).unwrap().1, &[0, 1, 2]);
        assert_eq!(packed.indexer().cache().get(2).unwrap().1, &[4]);

        // Decode continues each sequence's shared history exactly like unpacked execution.
        let decode_hidden = Tensor::from_slice(&[3.0f32, 3.0], (1, 1, 2), device)?;
        let packed_decode = packed.forward_chunk(1, &decode_hidden, &[3], &cos, &sin)?;
        let independent_decode = independent.forward_chunk(1, &decode_hidden, &[3], &cos, &sin)?;
        assert_eq!(packed_decode.dims(), [1, 1, 2]);
        assert_eq!(
            packed_decode.flatten_all()?.to_vec1::<f32>()?,
            independent_decode.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(packed.kv_cache().get(1).unwrap().0.dim(1)?, 4);
        assert_eq!(packed.indexer().cache().get(1).unwrap().1, &[0, 1, 2, 3]);
        Ok(())
    }

    #[test]
    fn packed_attention_rolls_back_every_touched_sequence() -> Result<()> {
        let device = &Device::Cpu;
        let cos = Tensor::ones((4, 1), DType::F32, device)?;
        let sin = Tensor::zeros((4, 1), DType::F32, device)?;
        let mut layer = attention_layer()?;
        let existing = Tensor::from_slice(&[1.0f32, 0.0], (1, 1, 2), device)?;
        layer.forward_chunk(1, &existing, &[0], &cos, &sin)?;

        // Pre-seed sequence 2's indexer cache with an incompatible dtype so its
        // chunk fails after the main K/V append inside the packed operation.
        layer.indexer.cache_mut().restore(
            2,
            &QsaSequenceSnapshot {
                raw_keys: Some(Tensor::zeros((1, 2), DType::F16, device)?),
                positions: vec![7],
                cos: None,
                sin: None,
                head_dim: 2,
            },
        )?;

        let valid = Tensor::from_slice(&[0.0f32, 1.0], (1, 1, 2), device)?;
        let chunks = [
            (3, &valid, &[1][..]),
            (2, &valid, &[1][..]),
            (2, &valid, &[2][..]),
        ];
        let error = layer
            .forward_packed_chunks(&chunks, &cos, &sin)
            .unwrap_err();
        assert!(error.to_string().contains("matching key dtype and device"));

        // Sequence 1 keeps its exact pre-packed state.
        assert_eq!(layer.kv_cache().get(1).unwrap().0.dim(1)?, 1);
        assert_eq!(layer.indexer().cache().get(1).unwrap().1, &[0]);
        // Sequence 2 returns to its exact prior present state without the failed append.
        assert!(layer.kv_cache().get(2).is_none());
        let (keys, positions) = layer.indexer.cache().get(2).unwrap();
        assert_eq!(keys.dtype(), DType::F16);
        assert_eq!(positions, &[7]);
        // Sequence 3 returns to its exact prior absent state.
        assert!(layer.kv_cache().get(3).is_none());
        assert!(layer.indexer().cache().get(3).is_none());
        Ok(())
    }

    #[test]
    fn packed_attention_validates_all_chunks_before_mutation() -> Result<()> {
        let device = &Device::Cpu;
        let cos = Tensor::ones((2, 1), DType::F32, device)?;
        let sin = Tensor::zeros((2, 1), DType::F32, device)?;
        let mut layer = attention_layer()?;
        let valid = Tensor::zeros((1, 1, 2), DType::F32, device)?;
        let invalid_batch = Tensor::zeros((2, 1, 2), DType::F32, device)?;
        let chunks = [(1, &valid, &[0][..]), (2, &invalid_batch, &[0][..])];
        let error = layer
            .forward_packed_chunks(&chunks, &cos, &sin)
            .unwrap_err();
        assert!(error.to_string().contains("sequence 2"));
        assert!(layer.kv_cache().get(1).is_none());
        assert!(layer.indexer().cache().get(1).is_none());
        assert!(layer.kv_cache().get(2).is_none());
        assert!(layer.indexer().cache().get(2).is_none());

        let wrong_positions = [(1, &valid, &[0, 1][..])];
        let error = layer
            .forward_packed_chunks(&wrong_positions, &cos, &sin)
            .unwrap_err();
        assert!(error.to_string().contains("one position per token"));

        let wrong_width = Tensor::zeros((1, 1, 3), DType::F32, device)?;
        let chunks = [(1, &valid, &[0][..]), (2, &wrong_width, &[0][..])];
        let error = layer
            .forward_packed_chunks(&chunks, &cos, &sin)
            .unwrap_err();
        assert!(error.to_string().contains("sequence 2"));
        assert!(error.to_string().contains("hidden width 2"));
        assert!(layer.kv_cache().get(1).is_none());
        assert!(layer.indexer().cache().get(1).is_none());

        assert!(layer.forward_packed_chunks(&[], &cos, &sin).is_err());
        Ok(())
    }

    #[test]
    fn kv_gather_preserves_selected_row_order_and_head_layout() -> Result<()> {
        let device = &Device::Cpu;
        let gather = QsaKvGather::new(2, 2, 1, 4, 2)?;
        let keys = Tensor::from_slice(
            &[
                0.0f32, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 100.0, 101.0, 110.0, 111.0, 120.0,
                121.0, 130.0, 131.0,
            ],
            (2, 4, 2),
            device,
        )?;
        let values = Tensor::from_slice(
            &[0.0f32, 10.0, 20.0, 30.0, 100.0, 110.0, 120.0, 130.0],
            (2, 4, 1),
            device,
        )?;
        let gathered = gather.gather(&keys, &values, &[vec![2, 0], vec![3]])?;
        assert_eq!(gathered[0].0.dims(), &[2, 2, 2]);
        assert_eq!(
            gathered[0].0.to_vec3::<f32>()?,
            vec![
                vec![vec![20.0, 21.0], vec![0.0, 1.0]],
                vec![vec![120.0, 121.0], vec![100.0, 101.0]],
            ]
        );
        assert_eq!(
            gathered[0].1.to_vec3::<f32>()?,
            vec![vec![vec![20.0], vec![0.0]], vec![vec![120.0], vec![100.0]]]
        );
        assert_eq!(gathered[1].0.dims(), &[2, 1, 2]);
        Ok(())
    }

    #[test]
    fn kv_gather_enforces_bounded_valid_cache_rows() -> Result<()> {
        let device = &Device::Cpu;
        let gather = QsaKvGather::new(1, 2, 3, 4, 2)?;
        let keys = Tensor::zeros((1, 6, 2), DType::F32, device)?;
        let values = Tensor::zeros((1, 6, 3), DType::F32, device)?;
        assert!(gather.gather(&keys, &values, &[vec![]]).is_err());
        assert!(gather
            .gather(&keys, &values, &[vec![0, 1, 2, 3, 4, 5, 0]])
            .unwrap_err()
            .to_string()
            .contains("1..=5"));
        assert!(gather
            .gather(&keys, &values, &[vec![6]])
            .unwrap_err()
            .to_string()
            .contains("outside the cache length 6"));

        let wrong_values = Tensor::zeros((2, 6, 3), DType::F32, device)?;
        assert!(gather
            .gather(&keys, &wrong_values, &[vec![0]])
            .unwrap_err()
            .to_string()
            .contains("expected keys"));
        Ok(())
    }

    #[test]
    fn sparse_attention_matches_small_grouped_query_reference() -> Result<()> {
        let device = &Device::Cpu;
        let attention = QsaSparseAttention::new(2, 1, 2, 1)?;
        let queries = Tensor::from_slice(
            &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0],
            (1, 2, 2, 2),
            device,
        )?;
        let gathered = vec![
            (
                Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], (1, 2, 2), device)?,
                Tensor::from_slice(&[2.0f32, 6.0], (1, 2, 1), device)?,
            ),
            (
                Tensor::from_slice(&[1.0f32, 1.0, -1.0, 1.0], (1, 2, 2), device)?,
                Tensor::from_slice(&[3.0f32, 7.0], (1, 2, 1), device)?,
            ),
        ];
        let output = attention
            .forward(&queries, &gathered)?
            .reshape((2, 2))?
            .to_vec2::<f32>()?;
        let scale = 1.0f32 / 2.0f32.sqrt();
        let expected = |left: f32, right: f32, first: f32, second: f32| {
            let left = (left * scale).exp();
            let right = (right * scale).exp();
            (left * first + right * second) / (left + right)
        };
        let expected = [
            [expected(1.0, 0.0, 2.0, 6.0), expected(1.0, 1.0, 3.0, 7.0)],
            [expected(1.0, 1.0, 2.0, 6.0), expected(0.0, -2.0, 3.0, 7.0)],
        ];
        for (actual_head, expected_head) in output.iter().zip(expected) {
            for (actual, expected) in actual_head.iter().zip(expected_head) {
                assert!((actual - expected).abs() < 2e-3, "{actual} != {expected}");
            }
        }
        Ok(())
    }

    #[test]
    fn sparse_attention_rejects_incompatible_query_and_workspace_inventory() -> Result<()> {
        let device = &Device::Cpu;
        assert!(QsaSparseAttention::new(3, 2, 2, 2).is_err());
        let attention = QsaSparseAttention::new(2, 1, 2, 2)?;
        let queries = Tensor::zeros((1, 2, 2, 2), DType::F32, device)?;
        let workspace = (
            Tensor::zeros((1, 1, 2), DType::F32, device)?,
            Tensor::zeros((1, 1, 2), DType::F32, device)?,
        );
        assert!(attention
            .forward(&queries, &[workspace])
            .unwrap_err()
            .to_string()
            .contains("one K/V workspace per token"));

        let invalid_workspace = (
            Tensor::zeros((1, 1, 3), DType::F32, device)?,
            Tensor::zeros((1, 1, 2), DType::F32, device)?,
        );
        assert!(attention
            .forward(&queries.narrow(2, 0, 1)?, &[invalid_workspace],)
            .unwrap_err()
            .to_string()
            .contains("incompatible gathered K/V"));
        Ok(())
    }

    #[test]
    fn indexer_orchestration_matches_chunked_execution() -> Result<()> {
        let device = &Device::Cpu;
        let hidden = Tensor::from_slice(
            &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            (1, 4, 2),
            device,
        )?;
        let cos = Tensor::ones((8, 1), DType::F32, device)?;
        let sin = Tensor::zeros((8, 1), DType::F32, device)?;

        let mut one_shot = indexer()?;
        let expected = one_shot.process_chunk(4, &hidden, &[0, 1, 2, 3], &cos, &sin)?;

        let mut chunked = indexer()?;
        let mut actual = chunked.process_chunk(4, &hidden.narrow(1, 0, 2)?, &[0, 1], &cos, &sin)?;
        actual.extend(chunked.process_chunk(4, &hidden.narrow(1, 2, 2)?, &[2, 3], &cos, &sin)?);
        assert_eq!(actual, expected);
        assert_eq!(actual[0], vec![0]);
        assert_eq!(actual[1], vec![0, 1]);
        assert_eq!(chunked.cache().get(4).unwrap().1, &[0, 1, 2, 3]);
        Ok(())
    }

    #[test]
    fn indexer_orchestration_rolls_back_cache_on_failure() -> Result<()> {
        let device = &Device::Cpu;
        let cos = Tensor::ones((2, 1), DType::F32, device)?;
        let sin = Tensor::zeros((2, 1), DType::F32, device)?;
        let mut indexer = indexer()?;
        let first = Tensor::from_slice(&[1.0f32, 0.0], (1, 1, 2), device)?;
        indexer.process_chunk(9, &first, &[0], &cos, &sin)?;

        let invalid = Tensor::from_slice(&[0.0f32, 1.0], (1, 1, 2), device)?;
        let error = indexer
            .process_chunk(9, &invalid, &[2], &cos, &sin)
            .unwrap_err();
        assert!(error.to_string().contains("outside the frequency cache"));
        let (keys, positions) = indexer.cache().get(9).unwrap();
        assert_eq!(positions, &[0]);
        assert_eq!(keys.dims(), &[1, 2]);
        Ok(())
    }

    #[test]
    fn packed_indexer_matches_independent_interleaved_chunks() -> Result<()> {
        let device = &Device::Cpu;
        let cos = Tensor::ones((8, 1), DType::F32, device)?;
        let sin = Tensor::zeros((8, 1), DType::F32, device)?;
        let first_a = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], (1, 2, 2), device)?;
        let first_b = Tensor::from_slice(&[1.0f32, 1.0], (1, 1, 2), device)?;
        let second_a = Tensor::from_slice(&[2.0f32, 1.0], (1, 1, 2), device)?;
        let chunks = [
            (1, &first_a, &[0, 1][..]),
            (2, &first_b, &[4][..]),
            (1, &second_a, &[2][..]),
        ];

        let mut packed = indexer()?;
        let actual = packed.process_packed_chunks(&chunks, &cos, &sin)?;
        let mut independent = indexer()?;
        let expected = vec![
            independent.process_chunk(1, &first_a, &[0, 1], &cos, &sin)?,
            independent.process_chunk(2, &first_b, &[4], &cos, &sin)?,
            independent.process_chunk(1, &second_a, &[2], &cos, &sin)?,
        ];
        assert_eq!(actual, expected);
        assert_eq!(packed.cache().get(1).unwrap().1, &[0, 1, 2]);
        assert_eq!(packed.cache().get(2).unwrap().1, &[4]);
        Ok(())
    }

    #[test]
    fn packed_indexer_rolls_back_every_touched_sequence() -> Result<()> {
        let device = &Device::Cpu;
        let cos = Tensor::ones((2, 1), DType::F32, device)?;
        let sin = Tensor::zeros((2, 1), DType::F32, device)?;
        let mut indexer = indexer()?;
        let existing = Tensor::from_slice(&[1.0f32, 0.0], (1, 1, 2), device)?;
        indexer.process_chunk(1, &existing, &[0], &cos, &sin)?;

        let valid = Tensor::from_slice(&[0.0f32, 1.0], (1, 1, 2), device)?;
        let invalid = Tensor::from_slice(&[1.0f32, 1.0], (1, 1, 2), device)?;
        let chunks = [
            (1, &valid, &[1][..]),
            (2, &valid, &[0][..]),
            (1, &invalid, &[2][..]),
        ];
        let error = indexer
            .process_packed_chunks(&chunks, &cos, &sin)
            .unwrap_err();
        assert!(error.to_string().contains("outside the frequency cache"));
        assert_eq!(indexer.cache().get(1).unwrap().1, &[0]);
        assert!(indexer.cache().get(2).is_none());
        Ok(())
    }

    #[test]
    fn packed_indexer_validates_all_chunks_before_mutation() -> Result<()> {
        let device = &Device::Cpu;
        let cos = Tensor::ones((2, 1), DType::F32, device)?;
        let sin = Tensor::zeros((2, 1), DType::F32, device)?;
        let mut indexer = indexer()?;
        let valid = Tensor::zeros((1, 1, 2), DType::F32, device)?;
        let invalid = Tensor::zeros((2, 1, 2), DType::F32, device)?;
        let chunks = [(1, &valid, &[0][..]), (2, &invalid, &[0][..])];
        let error = indexer
            .process_packed_chunks(&chunks, &cos, &sin)
            .unwrap_err();
        assert!(error.to_string().contains("sequence 2"));
        assert!(indexer.cache().get(1).is_none());
        assert!(indexer.cache().get(2).is_none());
        Ok(())
    }

    #[test]
    fn indexer_pooling_and_scores_match_reference_equations() -> Result<()> {
        let device = &Device::Cpu;
        let scorer = QsaIndexerScorer::new(2, 2, 2)?;
        let raw_keys = Tensor::from_slice(
            &[1.0f32, 3.0, 3.0, 1.0, -2.0, 4.0, 2.0, 0.0, 9.0, 9.0],
            (5, 2),
            device,
        )?;
        let pooled = scorer.pool_complete_blocks(&raw_keys)?;
        assert_eq!(
            pooled.to_vec2::<f32>()?,
            vec![vec![2.0, 2.0], vec![0.0, 2.0]]
        );

        // Per-head dot products are rectified before summing, then scaled by sqrt(head_dim).
        let query = Tensor::from_slice(&[1.0f32, -1.0, 2.0, 1.0], (2, 2), device)?;
        let scores = scorer.score(&query, &pooled)?.to_vec1::<f32>()?;
        let expected = [6.0f32 / 2.0f32.sqrt(), 2.0f32 / 2.0f32.sqrt()];
        for (actual, expected) in scores.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn indexer_rejects_invalid_dimensions_without_partial_pooling() -> Result<()> {
        assert!(QsaIndexerScorer::new(0, 2, 2).is_err());
        let scorer = QsaIndexerScorer::new(2, 2, 2)?;
        let device = &Device::Cpu;
        let invalid_keys = Tensor::zeros((4, 3), DType::F32, device)?;
        assert!(scorer
            .pool_complete_blocks(&invalid_keys)
            .unwrap_err()
            .to_string()
            .contains("raw keys expected width 2"));
        let query = Tensor::zeros((1, 2), DType::F32, device)?;
        let keys = Tensor::zeros((2, 2), DType::F32, device)?;
        assert!(scorer.score(&query, &keys).is_err());
        Ok(())
    }

    #[test]
    fn histories_shorter_than_one_block_select_only_the_tail() -> Result<()> {
        let selector = QsaBlockSelector::new(4, 8)?;
        assert_eq!(selector.select(&[10, 11, 12], &[])?, vec![10, 11, 12]);
        Ok(())
    }

    #[test]
    fn selection_uses_highest_scoring_complete_blocks_and_keeps_tail() -> Result<()> {
        let selector = QsaBlockSelector::new(2, 4)?;
        let visible = [10, 11, 20, 21, 30, 31, 40];
        assert_eq!(
            selector.select(&visible, &[0.5, 3.0, 2.0])?,
            vec![20, 21, 30, 31, 40]
        );
        Ok(())
    }

    #[test]
    fn histories_below_budget_keep_every_complete_block() -> Result<()> {
        let selector = QsaBlockSelector::new(2, 8)?;
        let visible = [10, 11, 20, 21, 30];
        assert_eq!(
            selector.select(&visible, &[0.1, 9.0])?,
            vec![20, 21, 10, 11, 30]
        );
        Ok(())
    }

    #[test]
    fn exact_score_ties_prefer_the_earlier_block() -> Result<()> {
        let selector = QsaBlockSelector::new(2, 2)?;
        let visible = [10, 11, 20, 21, 30, 31];
        assert_eq!(selector.select(&visible, &[1.0, 1.0, 0.5])?, vec![10, 11]);
        Ok(())
    }

    #[test]
    fn score_inventory_and_nan_are_rejected() -> Result<()> {
        let selector = QsaBlockSelector::new(2, 2)?;
        let mismatch = selector.select(&[1, 2, 3, 4], &[1.0]).unwrap_err();
        assert!(mismatch.to_string().contains("expected 2 block scores"));
        let nan = selector.select(&[1, 2], &[f32::NAN]).unwrap_err();
        assert!(nan.to_string().contains("must not contain NaN"));
        Ok(())
    }
}
