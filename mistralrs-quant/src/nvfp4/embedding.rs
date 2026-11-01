use std::sync::Arc;

use candle_core::{DType, Result, Tensor};

use super::VALUES_PER_BYTE;
use crate::{
    embedding::{
        EmbeddingTable, MappedRows, MappedShard, RowCodec, RowScale, ScaleGrid, ScaleType,
    },
    safetensors::MmapedSafetensors,
    NVFP4_BLOCK_SIZE,
};

/// NVFP4 rows split across memory-mapped safetensors shards with one global scale.
pub struct Nvfp4Embedding {
    raw: Arc<MmapedSafetensors>,
    shards: Vec<(String, String)>,
    starts: Vec<usize>,
    rows: usize,
    columns: usize,
    global_scale: f32,
}

pub struct Nvfp4EmbeddingShard<'a> {
    pub data: &'a [u8],
    pub scales: &'a [u8],
    pub first_row: usize,
    pub rows: usize,
}

impl Nvfp4Embedding {
    pub fn new(
        raw: Arc<MmapedSafetensors>,
        shards: Vec<(String, String)>,
        columns: usize,
        global_scale_name: &str,
    ) -> Result<Self> {
        if shards.is_empty() || columns == 0 || !columns.is_multiple_of(NVFP4_BLOCK_SIZE) {
            candle_core::bail!(
                "NVFP4 embeddings require shards and a width divisible by {NVFP4_BLOCK_SIZE}"
            );
        }
        let global = raw.get(global_scale_name)?;
        if DType::try_from(global.dtype())? != DType::F32 || global.data().len() != size_of::<f32>()
        {
            candle_core::bail!(
                "NVFP4 embedding global scale {global_scale_name} must contain one F32 value"
            );
        }
        let global_scale = f32::from_le_bytes(global.data().try_into().unwrap());
        if !global_scale.is_finite() || global_scale <= 0.0 {
            candle_core::bail!("NVFP4 embedding global scale must be finite and positive");
        }
        let mut starts = Vec::with_capacity(shards.len());
        let mut rows = 0;
        for (weight, scale) in &shards {
            let data = raw.get(weight)?;
            let scales = raw.get(scale)?;
            let shape = data.shape();
            if DType::try_from(data.dtype())? != DType::U8
                || shape.len() != 2
                || shape[0] == 0
                || shape[1] != columns / VALUES_PER_BYTE
            {
                candle_core::bail!(
                    "NVFP4 embedding {weight} must be packed U8 [rows, {}]",
                    columns / VALUES_PER_BYTE
                );
            }
            if DType::try_from(scales.dtype())? != DType::F8E4M3
                || scales.shape() != [shape[0], columns / NVFP4_BLOCK_SIZE]
            {
                candle_core::bail!(
                    "NVFP4 embedding {scale} must be E4M3 [{}, {}]",
                    shape[0],
                    columns / NVFP4_BLOCK_SIZE
                );
            }
            starts.push(rows);
            rows += shape[0];
        }
        Ok(Self {
            raw,
            shards,
            starts,
            rows,
            columns,
            global_scale,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn row_bytes(&self) -> usize {
        self.columns / VALUES_PER_BYTE + self.columns / NVFP4_BLOCK_SIZE
    }

    pub fn global_scale(&self) -> f32 {
        self.global_scale
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    pub fn shard(&self, index: usize) -> Result<Nvfp4EmbeddingShard<'_>> {
        let (weight, scale) = &self.shards[index];
        let data = self.raw.get(weight)?;
        let scales = self.raw.get(scale)?;
        Ok(Nvfp4EmbeddingShard {
            data: data.data(),
            scales: scales.data(),
            first_row: self.starts[index],
            rows: data.shape()[0],
        })
    }

    pub fn codec() -> RowCodec {
        RowCodec::Fp4(ScaleGrid::per_row(ScaleType::E4m3, NVFP4_BLOCK_SIZE))
    }

    /// Rows read in place from the safetensors mapping, which the result keeps alive.
    pub fn mapped_rows(&self) -> Result<MappedRows> {
        let shards = (0..self.shard_count())
            .map(|i| {
                let shard = self.shard(i)?;
                Ok(MappedShard {
                    data: shard.data.as_ptr(),
                    scales: shard.scales.as_ptr(),
                    first_row: shard.first_row,
                    rows: shard.rows,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(unsafe { MappedRows::new(shards, self.raw.clone()) })
    }

    pub fn mapped<'a>(&self, rows: &'a MappedRows) -> EmbeddingTable<'a> {
        EmbeddingTable::mapped(Self::codec(), self.columns, rows)
            .with_row_scale(RowScale::Value(self.global_scale))
    }

    /// The table over device copies: packed `[rows, columns / 2]` and E4M3 `[rows, columns / 16]` U8 bytes.
    pub fn resident<'a>(&self, data: &'a Tensor, scales: &'a Tensor) -> EmbeddingTable<'a> {
        EmbeddingTable::new(Self::codec(), self.rows, self.columns, data)
            .with_scales(scales)
            .with_row_scale(RowScale::Value(self.global_scale))
    }
}
