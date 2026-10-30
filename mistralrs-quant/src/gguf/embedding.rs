use std::sync::Arc;

use candle_core::{DType, Result, Tensor};

use super::archive::{GgufArchive, GgufEndian};
use crate::embedding::{EmbeddingTable, MappedRows, MappedShard, RowCodec};

/// A GGUF matrix read as embedding rows straight from the archive mapping.
pub struct GgufEmbedding {
    archive: Arc<GgufArchive>,
    name: String,
    rows: usize,
    columns: usize,
    codec: RowCodec,
}

impl GgufEmbedding {
    pub fn new(archive: Arc<GgufArchive>, name: &str) -> Result<Self> {
        let info = archive.tensor_info(name)?;
        if archive.shards()[info.shard_index()].endian() != GgufEndian::Little {
            candle_core::bail!("big-endian GGUF tensor loading is not supported");
        }
        let &[rows, columns] = info.shape() else {
            candle_core::bail!("GGUF embedding {name} must be a matrix");
        };
        if columns == 0 {
            candle_core::bail!("GGUF embedding {name} must have a nonzero row width");
        }
        let codec = RowCodec::from_gguf_id(info.dtype().raw())?;
        if !columns.is_multiple_of(codec.block_columns()) {
            candle_core::bail!(
                "GGUF embedding {name} row width must be a multiple of {}",
                codec.block_columns()
            );
        }
        Ok(Self {
            archive,
            name: name.to_owned(),
            rows,
            columns,
            codec,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn codec(&self) -> RowCodec {
        self.codec
    }

    pub fn row_bytes(&self) -> usize {
        self.codec.row_bytes(self.columns)
    }

    pub fn bytes(&self) -> Result<&[u8]> {
        Ok(self.archive.tensor_data(&self.name)?.bytes())
    }

    /// Rows read in place from the archive mapping, which the result keeps alive.
    pub fn mapped_rows(&self) -> Result<MappedRows> {
        let shard = MappedShard {
            data: self.bytes()?.as_ptr(),
            scales: std::ptr::null(),
            first_row: 0,
            rows: self.rows,
        };
        Ok(unsafe { MappedRows::new(vec![shard], self.archive.clone()) })
    }

    /// Dense GGUF rows are read as whole elements, so a mapping must keep that alignment.
    pub fn mapped_gather_aligned(&self) -> Result<bool> {
        let alignment = match self.codec {
            RowCodec::Dense(dtype) => dtype.size_in_bytes(),
            _ => DType::U8.size_in_bytes(),
        };
        Ok(self.bytes()?.as_ptr().align_offset(alignment) == 0)
    }

    /// The table over a device copy of the archive bytes, `[rows, row_bytes]` U8.
    pub fn resident<'a>(&self, data: &'a Tensor) -> EmbeddingTable<'a> {
        EmbeddingTable::new(self.codec, self.rows, self.columns, data)
    }

    pub fn mapped<'a>(&self, rows: &'a MappedRows) -> EmbeddingTable<'a> {
        EmbeddingTable::mapped(self.codec, self.columns, rows)
    }
}
