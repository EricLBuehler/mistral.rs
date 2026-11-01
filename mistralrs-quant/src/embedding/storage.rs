use std::sync::Arc;

#[cfg(feature = "cuda")]
use super::device_reads_pageable_memory;
use super::{EmbeddingTable, MappedRows, MappedShard, RowCodec};
#[cfg(feature = "cuda")]
use super::{RowScale, ScaleGrid, ScaleType};
#[cfg(feature = "cuda")]
use crate::NVFP4_BLOCK_SIZE;
use crate::{nvfp4::embedding::Nvfp4Embedding, safetensors::MmapedSafetensors, GgufEmbedding};
use candle_core::{DType, Device, Result, Tensor};

pub const SYMMETRIC_GROUP: usize = 32;
const RESIDENT_GROUP: usize = SYMMETRIC_GROUP;
#[cfg(feature = "cuda")]
const RESIDENT_RESERVE_BYTES: usize = 2 << 30;
#[cfg(feature = "cuda")]
const RESIDENT_COPY_ROWS: usize = 1 << 20;
#[cfg(feature = "cuda")]
const HOST_PREFAULT_STRIDE_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug)]
pub struct WeightAllocationBudget {
    pub persistent_bytes: usize,
    pub execution_bytes: usize,
}

impl WeightAllocationBudget {
    pub fn fits(self, persistent_bytes: usize, workspace_bytes: usize) -> bool {
        persistent_bytes <= self.persistent_bytes
            && persistent_bytes
                .checked_add(workspace_bytes)
                .is_some_and(|total| total <= self.execution_bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingPacking {
    Preserve,
    Symmetric {
        preferred_bits: usize,
        fallback_bits: usize,
    },
}

impl EmbeddingPacking {
    fn bits(self, columns: usize) -> Option<(usize, usize)> {
        match self {
            Self::Symmetric {
                preferred_bits,
                fallback_bits,
            } if columns.is_multiple_of(RESIDENT_GROUP) => Some((preferred_bits, fallback_bits)),
            _ => None,
        }
    }

    pub fn planned_bytes(self, rows: usize, columns: usize, dtype: DType) -> usize {
        self.bits(columns)
            .map_or(rows * columns * dtype.size_in_bytes(), |(_, bits)| {
                resident_bytes(rows, columns, bits)
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingBacking {
    Owned,
    FileMapped,
}

#[derive(Clone, Copy, Debug)]
pub struct EmbeddingMetadata {
    pub rows: usize,
    pub columns: usize,
    pub device_bytes: usize,
    /// Fallback backing; `device_bytes` may describe a separately packed copy.
    pub host_backing: EmbeddingBacking,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingGatherMode {
    Prefill,
    Decode,
}

pub struct EmbeddingSource {
    storage: TableSource,
    columns: usize,
    packing: EmbeddingPacking,
}

impl EmbeddingSource {
    pub fn dense_shards(
        raw: Arc<MmapedSafetensors>,
        names: Vec<String>,
        columns: usize,
    ) -> Result<Self> {
        let mut starts = Vec::with_capacity(names.len());
        let mut rows = 0;
        let mut dtype = None;
        for name in &names {
            let view = raw.get(name)?;
            let shape = view.shape();
            let shard_dtype = DType::try_from(view.dtype())?;
            if shape.len() != 2
                || shape[1] != columns
                || !matches!(shard_dtype, DType::F32 | DType::F16 | DType::BF16)
            {
                candle_core::bail!(
                    "Embedding shard {name} must contain dense rows of width {columns}"
                );
            }
            if dtype.is_some_and(|dtype| dtype != shard_dtype) {
                candle_core::bail!("Embedding shards mix dtypes");
            }
            starts.push(rows);
            rows += shape[0];
            dtype = Some(shard_dtype);
        }
        let dtype =
            dtype.ok_or_else(|| candle_core::Error::msg("Embedding source has no shards"))?;
        Ok(Self {
            storage: TableSource::Shards {
                raw,
                names,
                starts,
                dtype,
            },
            columns,
            packing: EmbeddingPacking::Preserve,
        })
    }

    pub fn gguf(table: GgufEmbedding) -> Self {
        Self {
            columns: table.columns(),
            storage: TableSource::Gguf(table),
            packing: EmbeddingPacking::Preserve,
        }
    }

    pub fn nvfp4(table: Nvfp4Embedding) -> Self {
        Self {
            columns: table.columns(),
            storage: TableSource::Nvfp4(table),
            packing: EmbeddingPacking::Preserve,
        }
    }

    pub fn with_packing(mut self, packing: EmbeddingPacking) -> Result<Self> {
        if let EmbeddingPacking::Symmetric {
            preferred_bits,
            fallback_bits,
        } = packing
        {
            if !matches!(preferred_bits, 4 | 8)
                || !matches!(fallback_bits, 4 | 8)
                || fallback_bits > preferred_bits
            {
                candle_core::bail!("Symmetric embedding packing requires 4 or 8 bits, with fallback no wider than preferred");
            }
        }
        self.packing = packing;
        Ok(self)
    }

    pub fn metadata(&self) -> EmbeddingMetadata {
        let rows = match &self.storage {
            TableSource::Shards { raw, names, .. } => names
                .iter()
                .map(|name| raw.get(name).expect("validated embedding shard").shape()[0])
                .sum(),
            TableSource::Gguf(table) => table.rows(),
            TableSource::Nvfp4(table) => table.rows(),
            #[cfg(feature = "cuda")]
            TableSource::Packed { rows, .. } => *rows,
        };
        let packing = matches!(self.storage, TableSource::Shards { .. })
            .then(|| self.packing.bits(self.columns))
            .flatten();
        EmbeddingMetadata {
            rows,
            columns: self.columns,
            device_bytes: self
                .storage
                .device_bytes(self.columns, packing.map_or(0, |(_, bits)| bits)),
            host_backing: match &self.storage {
                #[cfg(feature = "cuda")]
                TableSource::Packed { .. } => EmbeddingBacking::Owned,
                _ => EmbeddingBacking::FileMapped,
            },
        }
    }
}

#[cfg(feature = "cuda")]
struct TableProgress {
    bar: indicatif::ProgressBar,
}

#[cfg(feature = "cuda")]
impl Drop for TableProgress {
    fn drop(&mut self) {
        self.bar.finish_and_clear();
    }
}

#[cfg(feature = "cuda")]
fn table_progress(silent: bool, message: &'static str, bytes: usize) -> TableProgress {
    let bar = indicatif::ProgressBar::new(bytes as u64);
    if silent {
        bar.set_draw_target(indicatif::ProgressDrawTarget::hidden());
    }
    bar.set_style(
        indicatif::ProgressStyle::default_bar()
            .template(
                "{msg}: [{elapsed_precise}] [{bar:40.blue/blue}] {bytes}/{total_bytes} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    bar.set_message(message);
    bar.tick();
    TableProgress { bar }
}

fn resident_bytes(rows: usize, head_dim: usize, bits: usize) -> usize {
    rows * (head_dim * bits / 8 + head_dim / RESIDENT_GROUP * 2)
}

#[cfg(feature = "cuda")]
fn preparation_chunk_rows(
    requested: usize,
    source_row_bytes: usize,
    packed_row_bytes: usize,
    device_workspace_budget: usize,
) -> Result<usize> {
    let device_bytes = source_row_bytes + packed_row_bytes;
    // Include source pagecache and transient host copies returned by to_vec1.
    let workspace_bytes = 2 * source_row_bytes + 3 * packed_row_bytes;
    let rows = requested
        .min(RESIDENT_RESERVE_BYTES / workspace_bytes)
        .min(device_workspace_budget / device_bytes);
    if rows == 0 {
        candle_core::bail!(
            "Embedding preparation needs {device_bytes} device bytes and {workspace_bytes} total \
             workspace bytes per row, but the device workspace budget is {device_workspace_budget} bytes"
        );
    }
    Ok(rows)
}

enum TableSource {
    /// bf16/f16 row shards in the safetensors memory map.
    Shards {
        raw: Arc<MmapedSafetensors>,
        names: Vec<String>,
        starts: Vec<usize>,
        dtype: DType,
    },
    Gguf(GgufEmbedding),
    Nvfp4(Nvfp4Embedding),
    /// Symmetric int rows followed by their f16 group scales in anonymous host RAM.
    #[cfg(feature = "cuda")]
    Packed {
        map: Arc<memmap2::Mmap>,
        rows: usize,
        data_bytes: usize,
        bits: usize,
    },
}

impl TableSource {
    fn device_bytes(&self, columns: usize, bits: usize) -> usize {
        match self {
            Self::Shards {
                raw, names, dtype, ..
            } => {
                let rows = names
                    .iter()
                    .map(|name| raw.get(name).expect("validated embedding shard").shape()[0])
                    .sum();
                if bits == 0 {
                    rows * columns * dtype.size_in_bytes()
                } else {
                    resident_bytes(rows, columns, bits)
                }
            }
            Self::Gguf(table) => table.rows() * table.row_bytes(),
            Self::Nvfp4(table) => table.rows() * table.row_bytes(),
            #[cfg(feature = "cuda")]
            Self::Packed { map, .. } => map.len(),
        }
    }

    fn codec(&self) -> RowCodec {
        match self {
            Self::Shards { dtype, .. } => RowCodec::Dense(*dtype),
            Self::Gguf(table) => table.codec(),
            Self::Nvfp4(_) => Nvfp4Embedding::codec(),
            #[cfg(feature = "cuda")]
            Self::Packed { bits, .. } => packed_codec(*bits),
        }
    }

    fn mapped_rows(&self) -> Result<MappedRows> {
        match self {
            Self::Shards {
                raw, names, starts, ..
            } => {
                let shards = names
                    .iter()
                    .zip(starts)
                    .map(|(name, &first_row)| {
                        let view = raw.get(name)?;
                        Ok(MappedShard {
                            data: view.data().as_ptr(),
                            scales: std::ptr::null(),
                            first_row,
                            rows: view.shape()[0],
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(unsafe { MappedRows::new(shards, raw.clone()) })
            }
            Self::Gguf(table) => table.mapped_rows(),
            Self::Nvfp4(table) => table.mapped_rows(),
            #[cfg(feature = "cuda")]
            Self::Packed {
                map,
                rows,
                data_bytes,
                ..
            } => {
                let shard = MappedShard {
                    data: map.as_ptr(),
                    scales: map[*data_bytes..].as_ptr(),
                    first_row: 0,
                    rows: *rows,
                };
                Ok(unsafe { MappedRows::new(vec![shard], map.clone()) })
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn device_mappable(&self) -> Result<bool> {
        match self {
            Self::Gguf(table) => table.mapped_gather_aligned(),
            _ => Ok(true),
        }
    }
}

#[cfg(feature = "cuda")]
fn packed_codec(bits: usize) -> RowCodec {
    RowCodec::SymInt {
        bits,
        grid: ScaleGrid::per_row(ScaleType::F16, RESIDENT_GROUP),
    }
}

pub struct ManagedEmbedding {
    source: TableSource,
    columns: usize,
    #[cfg(feature = "cuda")]
    packing: EmbeddingPacking,
    #[cfg(feature = "cuda")]
    silent: bool,
    /// Host rows of `source`, indexed on the device when it reads pageable memory.
    mapped: MappedRows,
    #[cfg(feature = "cuda")]
    resident: Option<ResidentTable>,
}

/// Device copy of the table so decode never faults on host pages.
#[cfg(feature = "cuda")]
struct ResidentTable {
    codec: RowCodec,
    data: Tensor,
    scales: Option<Tensor>,
    row_scale: Option<f32>,
}

#[cfg(feature = "cuda")]
impl ResidentTable {
    fn table(&self, columns: usize) -> Result<EmbeddingTable<'_>> {
        let mut table = EmbeddingTable::new(self.codec, self.data.dim(0)?, columns, &self.data);
        if let Some(scales) = &self.scales {
            table = table.with_scales(scales);
        }
        if let Some(scale) = self.row_scale {
            table = table.with_row_scale(RowScale::Value(scale));
        }
        Ok(table)
    }
}

impl ManagedEmbedding {
    #[cfg(feature = "cuda")]
    pub fn prepare(
        &mut self,
        device: &Device,
        budget: &dyn Fn(&Device) -> Result<WeightAllocationBudget>,
        allow_resident: bool,
        silent: bool,
    ) -> Result<()> {
        self.silent = silent;
        if device.is_cuda() {
            let unified = crate::cuda_device::cuda_device_info(device.as_cuda_device()?).integrated;
            if allow_resident {
                self.make_resident(device, budget(device)?)?;
            }
            if self.resident.is_none() {
                let native = matches!(self.source, TableSource::Shards { .. });
                if unified && native {
                    tracing::info!("Embedding table kept in its original file mapping");
                    return Ok(());
                }
                let host_budget = if native {
                    available_memory(&Device::Cpu)?
                } else {
                    0
                };
                self.prepare_mapped(device, RESIDENT_COPY_ROWS, host_budget)?;
            }
            return Ok(());
        }
        tracing::info!("Embedding rows are gathered from the memory-mapped table on the host");
        Ok(())
    }

    pub fn new(source: EmbeddingSource, device: &Device) -> Result<Self> {
        let EmbeddingSource {
            storage: source,
            columns,
            packing,
        } = source;
        let mapped = source.mapped_rows()?;
        #[cfg(feature = "cuda")]
        let mapped = if device.is_cuda()
            && device_reads_pageable_memory(device)?
            && source.device_mappable()?
        {
            tracing::debug!("Embedding rows can be gathered through pageable memory access");
            mapped.with_device_index(device)?
        } else {
            mapped
        };
        #[cfg(not(feature = "cuda"))]
        let _ = (device, packing);
        Ok(Self {
            source,
            columns,
            #[cfg(feature = "cuda")]
            packing,
            #[cfg(feature = "cuda")]
            silent: true,
            mapped,
            #[cfg(feature = "cuda")]
            resident: None,
        })
    }

    fn table(&self) -> EmbeddingTable<'_> {
        match &self.source {
            TableSource::Gguf(table) => table.mapped(&self.mapped),
            TableSource::Nvfp4(table) => table.mapped(&self.mapped),
            source => EmbeddingTable::mapped(source.codec(), self.columns, &self.mapped),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn reclaimable_weight_cache_bytes(&self) -> Option<usize> {
        if self.resident.is_some() || !self.mapped.has_device_index() {
            return None;
        }
        let bytes = match &self.source {
            TableSource::Gguf(table) => table.rows() * table.row_bytes(),
            TableSource::Nvfp4(table) => table.rows() * table.row_bytes(),
            TableSource::Packed { .. } | TableSource::Shards { .. } => return None,
        };
        Some(bytes + RESIDENT_RESERVE_BYTES)
    }

    #[cfg(feature = "cuda")]
    fn fallback_path(&self) -> &'static str {
        if self.mapped.has_device_index() {
            "prompt rows are staged from host storage; decode uses mapped GPU access"
        } else {
            "rows are staged from host storage"
        }
    }

    #[cfg(feature = "cuda")]
    fn make_resident(&mut self, device: &Device, budget: WeightAllocationBudget) -> Result<()> {
        if matches!(self.source, TableSource::Shards { .. })
            && self.packing != EmbeddingPacking::Preserve
            && self.packing.bits(self.columns).is_none()
        {
            tracing::warn!(
                columns = self.columns,
                "Embedding table rows are not a multiple of {RESIDENT_GROUP}, so it stays out of device memory; {}",
                self.fallback_path()
            );
            return Ok(());
        }
        let rows = self.mapped.rows();
        let budget = WeightAllocationBudget {
            execution_bytes: budget.execution_bytes.min(available_memory(device)?),
            ..budget
        };
        let candidates = match (&self.source, self.packing.bits(self.columns)) {
            (TableSource::Shards { .. }, Some((preferred, fallback))) => [preferred, fallback],
            _ => [0, 0],
        };
        let footprint = |bits: usize| self.source.device_bytes(self.columns, bits);
        let Some(bits) = candidates
            .into_iter()
            .find(|bits| budget.fits(footprint(*bits), RESIDENT_RESERVE_BYTES))
        else {
            tracing::warn!(
                persistent_budget_bytes = budget.persistent_bytes,
                execution_budget_bytes = budget.execution_bytes,
                table_bytes = footprint(candidates[1]),
                preparation_bytes = RESIDENT_RESERVE_BYTES,
                "Embedding table does not fit in device memory; {}",
                self.fallback_path()
            );
            return Ok(());
        };
        let start = std::time::Instant::now();
        let resident = match &self.source {
            TableSource::Gguf(table) => self.copy_gguf(table, rows, device)?,
            TableSource::Nvfp4(table) => self.copy_nvfp4(table, device)?,
            TableSource::Shards {
                raw,
                names,
                starts,
                dtype,
            } if bits == 0 => self.copy_dense(raw, names, starts, *dtype, device)?,
            TableSource::Shards {
                raw,
                names,
                starts,
                dtype,
            } => self.quantize_shards(QuantizeShards {
                raw,
                names,
                starts,
                dtype: *dtype,
                rows,
                bits,
                device,
            })?,
            TableSource::Packed { .. } => unreachable!("table preparation runs once"),
        };
        tracing::info!(
            size_gb = footprint(bits) >> 30,
            elapsed_s = start.elapsed().as_secs(),
            "Embedding table resident in device memory"
        );
        self.resident = Some(resident);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn copy_dense(
        &self,
        raw: &MmapedSafetensors,
        names: &[String],
        starts: &[usize],
        dtype: DType,
        device: &Device,
    ) -> Result<ResidentTable> {
        let row_bytes = self.columns * dtype.size_in_bytes();
        let copy_rows =
            preparation_chunk_rows(RESIDENT_COPY_ROWS, row_bytes, 0, RESIDENT_RESERVE_BYTES)?;
        let data = unsafe { Tensor::empty((self.mapped.rows(), self.columns), dtype, device)? };
        let progress = table_progress(
            self.silent,
            "Copying embedding table to device memory",
            self.mapped.rows() * row_bytes,
        );
        for (name, &start) in names.iter().zip(starts) {
            let view = raw.get(name)?;
            for (chunk, bytes) in view.data().chunks(copy_rows * row_bytes).enumerate() {
                prefault_source_pages(bytes);
                data.slice_set(
                    &Tensor::from_raw_buffer(
                        bytes,
                        dtype,
                        &[bytes.len() / row_bytes, self.columns],
                        device,
                    )?,
                    0,
                    start + chunk * copy_rows,
                )?;
                release_source_pages(bytes);
                progress.bar.inc(bytes.len() as u64);
            }
        }
        Ok(ResidentTable {
            codec: RowCodec::Dense(dtype),
            data,
            scales: None,
            row_scale: None,
        })
    }

    #[cfg(feature = "cuda")]
    fn copy_gguf(
        &self,
        table: &GgufEmbedding,
        rows: usize,
        device: &Device,
    ) -> Result<ResidentTable> {
        let row_bytes = table.row_bytes();
        let bytes = table.bytes()?;
        let copy_rows =
            preparation_chunk_rows(RESIDENT_COPY_ROWS, row_bytes, 0, RESIDENT_RESERVE_BYTES)?;
        let progress = table_progress(
            self.silent,
            "Copying embedding table to device memory",
            bytes.len(),
        );
        let data = unsafe { Tensor::empty((rows, row_bytes), DType::U8, device)? };
        for (chunk, src) in bytes.chunks(copy_rows * row_bytes).enumerate() {
            prefault_source_pages(src);
            data.slice_set(
                &Tensor::from_slice(src, (src.len() / row_bytes, row_bytes), device)?,
                0,
                chunk * copy_rows,
            )?;
            release_source_pages(src);
            progress.bar.inc(src.len() as u64);
        }
        progress.bar.finish_and_clear();
        Ok(ResidentTable {
            codec: table.codec(),
            data,
            scales: None,
            row_scale: None,
        })
    }

    #[cfg(feature = "cuda")]
    fn copy_nvfp4(&self, table: &Nvfp4Embedding, device: &Device) -> Result<ResidentTable> {
        let row_bytes = self.columns / 2;
        let scale_bytes = self.columns / NVFP4_BLOCK_SIZE;
        let copy_rows = preparation_chunk_rows(
            RESIDENT_COPY_ROWS,
            table.row_bytes(),
            0,
            RESIDENT_RESERVE_BYTES,
        )?;
        let progress = table_progress(
            self.silent,
            "Copying NVFP4 embedding table to device memory",
            table.rows() * table.row_bytes(),
        );
        let data = unsafe { Tensor::empty((table.rows(), row_bytes), DType::U8, device)? };
        let scales = unsafe { Tensor::empty((table.rows(), scale_bytes), DType::U8, device)? };
        for i in 0..table.shard_count() {
            let shard = table.shard(i)?;
            for first in (0..shard.rows).step_by(copy_rows) {
                let count = copy_rows.min(shard.rows - first);
                for (source, target, width) in [
                    (shard.data, &data, row_bytes),
                    (shard.scales, &scales, scale_bytes),
                ] {
                    let bytes = &source[first * width..(first + count) * width];
                    prefault_source_pages(bytes);
                    target.slice_set(
                        &Tensor::from_slice(bytes, (count, width), device)?,
                        0,
                        shard.first_row + first,
                    )?;
                    release_source_pages(bytes);
                    progress.bar.inc(bytes.len() as u64);
                }
            }
        }
        progress.bar.finish_and_clear();
        Ok(ResidentTable {
            codec: Nvfp4Embedding::codec(),
            data,
            scales: Some(scales),
            row_scale: Some(table.global_scale()),
        })
    }

    #[cfg(feature = "cuda")]
    fn quantize_shards(&self, args: QuantizeShards<'_>) -> Result<ResidentTable> {
        let QuantizeShards {
            raw,
            names,
            starts,
            dtype,
            rows,
            bits,
            device,
        } = args;
        let groups = self.columns / RESIDENT_GROUP;
        let row_bytes = self.columns * bits / 8;
        let source_row_bytes = self.columns * dtype.size_in_bytes();
        let packed_row_bytes = row_bytes + groups * DType::F16.size_in_bytes();
        let copy_rows = preparation_chunk_rows(
            RESIDENT_COPY_ROWS,
            source_row_bytes,
            packed_row_bytes,
            RESIDENT_RESERVE_BYTES,
        )?;
        let progress = table_progress(
            self.silent,
            "Quantizing embedding table into device memory",
            rows * packed_row_bytes,
        );
        let data = unsafe { Tensor::empty((rows, row_bytes), DType::U8, device)? };
        let scales = unsafe { Tensor::empty((rows, groups), DType::F16, device)? };
        for (shard, name) in names.iter().enumerate() {
            let view = raw.get(name)?;
            for (chunk, src) in view.data().chunks(copy_rows * source_row_bytes).enumerate() {
                let chunk_rows = src.len() / source_row_bytes;
                let start_row = starts[shard] + chunk * copy_rows;
                prefault_source_pages(src);
                let source =
                    Tensor::from_raw_buffer(src, dtype, &[chunk_rows, self.columns], device)?;
                super::packing::quantize(
                    &source,
                    &data.narrow(0, start_row, chunk_rows)?,
                    &scales.narrow(0, start_row, chunk_rows)?,
                    bits,
                )?;
                release_source_pages(src);
                progress.bar.inc((chunk_rows * packed_row_bytes) as u64);
            }
        }
        progress.bar.finish_and_clear();
        Ok(ResidentTable {
            codec: packed_codec(bits),
            data,
            scales: Some(scales),
            row_scale: None,
        })
    }

    #[cfg(feature = "cuda")]
    fn prepare_mapped(&mut self, device: &Device, chunk_rows: usize, budget: usize) -> Result<()> {
        if let TableSource::Nvfp4(table) = &self.source {
            if self.mapped.has_device_index() {
                let progress = table_progress(
                    self.silent,
                    "Preparing mapped NVFP4 embedding table",
                    table.rows() * table.row_bytes(),
                );
                for i in 0..table.shard_count() {
                    let shard = table.shard(i)?;
                    for bytes in [shard.data, shard.scales] {
                        let prefaulted = crate::cuda_memory::prefault_mapped_memory(bytes, device)?;
                        if !prefaulted {
                            prefault_source_pages(bytes);
                        }
                        progress.bar.inc(bytes.len() as u64);
                    }
                }
                progress.bar.finish_and_clear();
            }
            return Ok(());
        }
        if let TableSource::Gguf(table) = &self.source {
            if self.mapped.has_device_index() {
                let started = std::time::Instant::now();
                let bytes = table.bytes()?;
                let gpu_prefault = crate::cuda_memory::prefault_mapped_memory(bytes, device)?;
                if !gpu_prefault {
                    for byte in bytes.iter().step_by(HOST_PREFAULT_STRIDE_BYTES) {
                        std::hint::black_box(*byte);
                    }
                    if let Some(byte) = bytes.last() {
                        std::hint::black_box(*byte);
                    }
                }
                tracing::info!(
                    size_gb = bytes.len() >> 30,
                    elapsed_s = started.elapsed().as_secs(),
                    gpu_prefault,
                    "GGUF embedding table prefaulted in reclaimable memory-mapped storage"
                );
            }
            return Ok(());
        }
        let TableSource::Shards {
            raw,
            names,
            starts,
            dtype,
        } = &self.source
        else {
            return Ok(());
        };
        let Some((_, bits)) = self.packing.bits(self.columns) else {
            return Ok(());
        };
        let started = std::time::Instant::now();
        let rows = self.mapped.rows();
        let row_bytes = self.columns * bits / 8;
        let scale_row_bytes = self.columns / RESIDENT_GROUP * DType::F16.size_in_bytes();
        let source_row_bytes = self.columns * dtype.size_in_bytes();
        let data_bytes = rows * row_bytes;
        let packed_bytes = data_bytes + rows * scale_row_bytes;
        let required = packed_bytes + RESIDENT_RESERVE_BYTES;
        if required > budget {
            tracing::info!(
                packed_bytes,
                host_budget_bytes = budget,
                "Embedding table kept in its original file mapping because host packing exceeds the memory budget"
            );
            return Ok(());
        }
        let device_workspace_budget = available_memory(device)?;
        let chunk_rows = preparation_chunk_rows(
            chunk_rows,
            source_row_bytes,
            row_bytes + scale_row_bytes,
            device_workspace_budget,
        )?;
        let mut map = memmap2::MmapOptions::new()
            .len(packed_bytes)
            .map_anon()
            .map_err(|error| {
                candle_core::Error::msg(format!(
                    "Failed to allocate {packed_bytes} bytes of anonymous RAM for packed embedding table: {error}"
                ))
            })?;
        let progress = table_progress(
            self.silent,
            "Packing embedding table into host RAM",
            packed_bytes,
        );
        for (shard, name) in names.iter().enumerate() {
            let view = raw.get(name)?;
            for (chunk, src) in view
                .data()
                .chunks(chunk_rows * source_row_bytes)
                .enumerate()
            {
                let first = starts[shard] + chunk * chunk_rows;
                let chunk_rows = src.len() / source_row_bytes;
                prefault_source_pages(src);
                let source =
                    Tensor::from_raw_buffer(src, *dtype, &[chunk_rows, self.columns], device)?;
                let data = unsafe { Tensor::empty((chunk_rows, row_bytes), DType::U8, device)? };
                let scales = unsafe {
                    Tensor::empty(
                        (chunk_rows, self.columns / RESIDENT_GROUP),
                        DType::F16,
                        device,
                    )?
                };
                super::packing::quantize(&source, &data, &scales, bits)?;
                let data = data.flatten_all()?.to_vec1::<u8>()?;
                let scales = scales.flatten_all()?.to_vec1::<half::f16>()?;
                let data_offset = first * row_bytes;
                map[data_offset..data_offset + data.len()].copy_from_slice(&data);
                let scales = bytemuck::cast_slice::<_, u8>(&scales);
                let scale_offset = data_bytes + first * scale_row_bytes;
                map[scale_offset..scale_offset + scales.len()].copy_from_slice(scales);
                release_source_pages(src);
                progress.bar.inc((data.len() + scales.len()) as u64);
            }
        }
        progress.bar.finish_and_clear();
        let map = map.make_read_only().map_err(|error| {
            candle_core::Error::msg(format!(
                "Failed to finalize {packed_bytes}-byte packed embedding table in anonymous RAM: {error}"
            ))
        })?;
        tracing::info!(
            packed_bytes,
            "Preparing CUDA access to packed embedding table in host RAM"
        );
        let gpu_prefault = crate::cuda_memory::prefault_mapped_memory(&map, device)?;
        self.source = TableSource::Packed {
            map: Arc::new(map),
            rows,
            data_bytes,
            bits,
        };
        let mapped = self.source.mapped_rows()?;
        self.mapped = if device_reads_pageable_memory(device)? {
            mapped.with_device_index(device)?
        } else {
            mapped
        };
        tracing::info!(
            size_gb = resident_bytes(rows, self.columns, bits) >> 30,
            elapsed_s = started.elapsed().as_secs(),
            gpu_prefault,
            "Embedding table quantized into anonymous host RAM"
        );
        Ok(())
    }

    pub fn gathers_on_device(&self) -> bool {
        #[cfg(feature = "cuda")]
        if self.resident.is_some() {
            return true;
        }
        self.mapped.has_device_index()
    }

    pub fn gather(
        &self,
        rows: &Tensor,
        out: super::EmbeddingOutput,
        mode: EmbeddingGatherMode,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(resident) = &self.resident {
            return resident.table(self.columns)?.gather(rows, out);
        }
        let table = self.table();
        // CPU gathers avoid first-touch stalls from scattered GPU reads of mapped rows.
        let use_host = !table.gathers_on(rows.device()) || mode == EmbeddingGatherMode::Prefill;
        if use_host {
            let ids = rows.flatten_all()?.to_dtype(DType::I64)?.to_vec1::<i64>()?;
            let ids = ids.into_iter().map(|id| id as u64).collect::<Vec<_>>();
            let mut shape = rows.dims().to_vec();
            shape.push(self.columns);
            let gathered = if rows.device().is_cpu() {
                table.gather_host(&ids, out)?
            } else {
                table.gather_staged(&ids, out, rows.device())?
            };
            return gathered.reshape(shape);
        }
        table.gather(rows, out)
    }

    /// Host gather of `rows` into a `[rows.len(), columns]` tensor on `device`.
    pub fn gather_host(&self, rows: &[u64], device: &Device) -> Result<Tensor> {
        let dtype = match &self.source {
            TableSource::Shards { dtype, .. } => *dtype,
            _ => DType::F32,
        };
        self.table()
            .gather_host(rows, dtype.into())?
            .to_device(device)
    }
}

#[cfg(feature = "cuda")]
struct QuantizeShards<'a> {
    raw: &'a MmapedSafetensors,
    names: &'a [String],
    starts: &'a [usize],
    dtype: DType,
    rows: usize,
    bits: usize,
    device: &'a Device,
}

#[cfg(feature = "cuda")]
fn prefault_source_pages(data: &[u8]) {
    #[cfg(target_os = "linux")]
    // Faulting on the CPU avoids serial GPU faults during pageable H2D copies on unified CUDA.
    advise_source_pages(data, libc::MADV_POPULATE_READ);
    #[cfg(not(target_os = "linux"))]
    let _ = data;
}

#[cfg(feature = "cuda")]
fn release_source_pages(data: &[u8]) {
    #[cfg(target_os = "linux")]
    // Batch-unmap copied source pages; PAGEOUT forces costly per-page SMMU invalidation.
    advise_source_pages(data, libc::MADV_DONTNEED);
    #[cfg(not(target_os = "linux"))]
    let _ = data;
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn advise_source_pages(data: &[u8], advice: libc::c_int) {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return;
    }
    let page = page as usize;
    let start = (data.as_ptr() as usize).div_ceil(page) * page;
    let end = (data.as_ptr() as usize + data.len()) / page * page;
    if end > start {
        unsafe {
            libc::madvise(start as *mut libc::c_void, end - start, advice);
        }
    }
}

#[cfg(feature = "cuda")]
fn available_memory(device: &Device) -> Result<usize> {
    if let Device::Cuda(dev) = device {
        if !crate::cuda_device::cuda_device_info(dev).integrated {
            let context = dev.cuda_stream().context().clone();
            context.bind_to_thread().map_err(candle_core::Error::wrap)?;
            return context
                .mem_get_info()
                .map(|(free, _)| free)
                .map_err(candle_core::Error::wrap);
        }
    }
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    usize::try_from(system.available_memory()).map_err(candle_core::Error::wrap)
}

#[cfg(test)]
#[path = "storage_tests.rs"]
mod tests;
