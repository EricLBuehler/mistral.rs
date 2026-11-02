#![allow(clippy::cast_possible_truncation)]

use std::borrow::Cow;

use candle_core::{
    quantized::{k_quants::*, GgmlDType, QTensor},
    DType, Device, Result, Storage, Tensor,
};
use float8::F8E4M3;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "metal")]
mod metal;

#[cfg(feature = "cuda")]
mod mapped_lifetime;
#[cfg(feature = "cuda")]
mod packing;
mod storage;

pub use storage::{
    EmbeddingBacking, EmbeddingGatherMode, EmbeddingMetadata, EmbeddingPacking, EmbeddingSource,
    ManagedEmbedding, WeightAllocationBudget, SYMMETRIC_GROUP,
};

const UNIT: usize = 8;
const IQ4_BLOCK: usize = 32;
const IQ4_BLOCK_BYTES: usize = 18;
const IQ4_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
const FP4_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const SYM_INT4_OFFSET: i32 = 8;
const GGUF_IQ4_NL: u32 = 20;
const GGUF_MXFP4: u32 = 39;

/// Output of an embedding gather: row dtype plus a multiplier fused into the decode (Gemma `sqrt(hidden)`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmbeddingOutput {
    pub dtype: DType,
    pub scale: Option<f32>,
}

impl EmbeddingOutput {
    pub fn scaled(dtype: DType, scale: f64) -> Self {
        Self {
            dtype,
            scale: Some(scale as f32),
        }
    }

    fn multiplier(&self) -> f32 {
        self.scale.unwrap_or(1.0)
    }
}

impl From<DType> for EmbeddingOutput {
    fn from(dtype: DType) -> Self {
        Self { dtype, scale: None }
    }
}

// Discriminant order mirrors the GGUF codec ids in gather_rows.cu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufRowType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq4Nl,
    Mxfp4 = 18,
}

impl GgufRowType {
    fn block(self) -> (usize, usize) {
        match self {
            Self::Q4_0 | Self::Iq4Nl => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
            Self::Q8K => (256, 292),
            Self::Mxfp4 => (32, 17),
        }
    }
}

/// Scalar encodings for block scales, AFQ biases and row multipliers; discriminants mirror gather_rows.cu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleType {
    F32,
    F16,
    Bf16,
    E4m3,
    /// OCP E8M0: 0 is 2^-127 and 255 is NaN.
    E8m0,
    /// Raw `2^(e - 127)` bit pattern, as MXFP4 checkpoints decode it.
    E8m0Raw,
}

impl ScaleType {
    pub fn from_dtype(dtype: DType) -> Result<Self> {
        Ok(match dtype {
            DType::F32 => Self::F32,
            DType::F16 => Self::F16,
            DType::BF16 => Self::Bf16,
            DType::F8E4M3 => Self::E4m3,
            other => candle_core::bail!("unsupported embedding scale dtype {other:?}"),
        })
    }

    fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::E4m3 | Self::E8m0 | Self::E8m0Raw => 1,
        }
    }

    fn load(self, bytes: &[u8], index: usize) -> f32 {
        let at = index * self.bytes();
        match self {
            Self::F32 => f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()),
            Self::F16 => half::f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32(),
            Self::Bf16 => half::bf16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32(),
            Self::E4m3 => F8E4M3::from_bits(bytes[at]).to_f32(),
            Self::E8m0 => match bytes[at] {
                0 => f32::from_bits(1 << 22),
                u8::MAX => f32::NAN,
                e => f32::from_bits(u32::from(e) << 23),
            },
            Self::E8m0Raw => f32::from_bits(u32::from(bytes[at]) << 23),
        }
    }
}

/// Block scales shared by `group` consecutive columns and `rows_per_scale` consecutive rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaleGrid {
    pub ty: ScaleType,
    pub group: usize,
    pub rows_per_scale: usize,
}

impl ScaleGrid {
    pub fn per_row(ty: ScaleType, group: usize) -> Self {
        Self {
            ty,
            group,
            rows_per_scale: 1,
        }
    }

    fn row_bytes(&self, columns: usize) -> usize {
        columns.div_ceil(self.group) * self.ty.bytes()
    }
}

/// How one table row is encoded; every codec decodes rows independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowCodec {
    /// F32, F16 or BF16 rows.
    Dense(DType),
    Gguf(GgufRowType),
    /// E2M1 nibbles, first value in the low nibble (NVFP4 and MXFP4).
    Fp4(ScaleGrid),
    /// E4M3 bytes with optional block scales (MXFP8, blockwise FP8; per-tensor/channel use `RowScale`).
    Fp8(Option<ScaleGrid>),
    /// Little-endian `bits`-wide codes, `q * scale + bias` with biases laid out like the scales.
    Afq {
        bits: usize,
        grid: ScaleGrid,
    },
    /// Symmetric 4- or 8-bit codes (4-bit stored `q + 8`), one scale per group.
    SymInt {
        bits: usize,
        grid: ScaleGrid,
    },
}

impl RowCodec {
    pub fn from_ggml(dtype: GgmlDType) -> Self {
        let ty = match dtype {
            GgmlDType::F32 => return Self::Dense(DType::F32),
            GgmlDType::F16 => return Self::Dense(DType::F16),
            GgmlDType::BF16 => return Self::Dense(DType::BF16),
            GgmlDType::Q4_0 => GgufRowType::Q4_0,
            GgmlDType::Q4_1 => GgufRowType::Q4_1,
            GgmlDType::Q5_0 => GgufRowType::Q5_0,
            GgmlDType::Q5_1 => GgufRowType::Q5_1,
            GgmlDType::Q8_0 => GgufRowType::Q8_0,
            GgmlDType::Q8_1 => GgufRowType::Q8_1,
            GgmlDType::Q2K => GgufRowType::Q2K,
            GgmlDType::Q3K => GgufRowType::Q3K,
            GgmlDType::Q4K => GgufRowType::Q4K,
            GgmlDType::Q5K => GgufRowType::Q5K,
            GgmlDType::Q6K => GgufRowType::Q6K,
            GgmlDType::Q8K => GgufRowType::Q8K,
            GgmlDType::Mxfp4 => GgufRowType::Mxfp4,
        };
        Self::Gguf(ty)
    }

    /// Codec for a raw GGUF tensor type id, including types candle has no `GgmlDType` for.
    pub fn from_gguf_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => Self::Dense(DType::F32),
            1 => Self::Dense(DType::F16),
            30 => Self::Dense(DType::BF16),
            2 => Self::Gguf(GgufRowType::Q4_0),
            3 => Self::Gguf(GgufRowType::Q4_1),
            6 => Self::Gguf(GgufRowType::Q5_0),
            7 => Self::Gguf(GgufRowType::Q5_1),
            8 => Self::Gguf(GgufRowType::Q8_0),
            9 => Self::Gguf(GgufRowType::Q8_1),
            10 => Self::Gguf(GgufRowType::Q2K),
            11 => Self::Gguf(GgufRowType::Q3K),
            12 => Self::Gguf(GgufRowType::Q4K),
            13 => Self::Gguf(GgufRowType::Q5K),
            14 => Self::Gguf(GgufRowType::Q6K),
            15 => Self::Gguf(GgufRowType::Q8K),
            GGUF_IQ4_NL => Self::Gguf(GgufRowType::Iq4Nl),
            GGUF_MXFP4 => Self::Gguf(GgufRowType::Mxfp4),
            other => candle_core::bail!("unsupported GGUF embedding type {other}"),
        })
    }

    /// Columns one indivisible block of the encoding covers.
    pub fn block_columns(&self) -> usize {
        match self {
            Self::Dense(_) | Self::Fp8(_) => 1,
            Self::Gguf(ty) => ty.block().0,
            Self::Fp4(grid) | Self::Afq { grid, .. } | Self::SymInt { grid, .. } => grid.group,
        }
    }

    pub fn row_bytes(&self, columns: usize) -> usize {
        match self {
            Self::Dense(dtype) => columns * dtype.size_in_bytes(),
            Self::Gguf(ty) => {
                let (block, bytes) = ty.block();
                columns / block * bytes
            }
            Self::Fp4(_) => columns / 2,
            Self::Fp8(_) => columns,
            Self::Afq { bits, .. } | Self::SymInt { bits, .. } => columns * bits / 8,
        }
    }

    fn grid(&self) -> Option<ScaleGrid> {
        match self {
            Self::Fp4(grid) | Self::Afq { grid, .. } | Self::SymInt { grid, .. } => Some(*grid),
            Self::Fp8(grid) => *grid,
            Self::Dense(_) | Self::Gguf(_) => None,
        }
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn id(&self) -> i32 {
        match self {
            Self::Dense(DType::F32) => 0,
            Self::Dense(DType::F16) => 1,
            Self::Dense(_) => 2,
            Self::Gguf(ty) => 3 + *ty as i32,
            Self::Fp4(_) => 16,
            Self::Fp8(_) => 17,
            Self::Afq { .. } => 18,
            Self::SymInt { bits: 4, .. } => 19,
            Self::SymInt { .. } => 20,
        }
    }

    fn validate(&self, columns: usize) -> Result<()> {
        let block = self.block_columns();
        let ok = match self {
            Self::Dense(dtype) => matches!(dtype, DType::F32 | DType::F16 | DType::BF16),
            Self::Afq { bits, .. } => matches!(bits, 2 | 3 | 4 | 6 | 8),
            Self::SymInt { bits, .. } => matches!(bits, 4 | 8),
            _ => true,
        };
        if !ok
            || block == 0
            || self.grid().is_some_and(|grid| grid.group == 0)
            || !columns.is_multiple_of(block)
        {
            candle_core::bail!("{self:?} cannot encode rows of {columns} columns");
        }
        if block > 1 && !columns.is_multiple_of(UNIT) {
            candle_core::bail!("{self:?} rows must be a multiple of {UNIT} columns");
        }
        Ok(())
    }

    /// Host decode of one row; `scales`/`aux` are the row's scale and bias bytes.
    fn decode_row(&self, data: &[u8], scales: &[u8], aux: &[u8], out: &mut [f32]) {
        match *self {
            Self::Dense(dtype) => {
                let ty = ScaleType::from_dtype(dtype).unwrap();
                for (i, value) in out.iter_mut().enumerate() {
                    *value = ty.load(data, i);
                }
            }
            Self::Gguf(ty) => decode_gguf(ty, data, out),
            Self::Fp4(grid) => {
                for (i, value) in out.iter_mut().enumerate() {
                    let code = (data[i / 2] >> (4 * (i % 2))) & 15;
                    let magnitude = FP4_VALUES[usize::from(code & 7)];
                    let q = if code & 8 == 0 { magnitude } else { -magnitude };
                    *value = q * grid.ty.load(scales, i / grid.group);
                }
            }
            Self::Fp8(grid) => {
                for (i, value) in out.iter_mut().enumerate() {
                    let scale = grid.map_or(1.0, |grid| grid.ty.load(scales, i / grid.group));
                    *value = F8E4M3::from_bits(data[i]).to_f32() * scale;
                }
            }
            Self::Afq { bits, grid } => {
                let mask = (1u32 << bits) - 1;
                for (i, value) in out.iter_mut().enumerate() {
                    let bit = i * bits;
                    let lo = u32::from(data[bit / 8]);
                    let word = if bit % 8 + bits > 8 {
                        lo | (u32::from(data[bit / 8 + 1]) << 8)
                    } else {
                        lo
                    };
                    let q = (word >> (bit % 8)) & mask;
                    let g = i / grid.group;
                    *value = (q as f32).mul_add(grid.ty.load(scales, g), grid.ty.load(aux, g));
                }
            }
            Self::SymInt { bits, grid } => {
                for (i, value) in out.iter_mut().enumerate() {
                    let q = if bits == 8 {
                        i32::from(data[i] as i8)
                    } else {
                        i32::from((data[i / 2] >> (4 * (i % 2))) & 15) - SYM_INT4_OFFSET
                    };
                    *value = q as f32 * grid.ty.load(scales, i / grid.group);
                }
            }
        }
    }
}

fn dequantize_blocks<T: GgmlType>(src: &[u8], out: &mut [f32]) {
    for (bytes, dst) in src
        .chunks_exact(std::mem::size_of::<T>())
        .zip(out.chunks_exact_mut(T::BLCK_SIZE))
    {
        // Rows can start at an unaligned shard offset.
        let block = unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<T>()) };
        T::to_float(std::slice::from_ref(&block), dst);
    }
}

fn decode_gguf(ty: GgufRowType, src: &[u8], out: &mut [f32]) {
    match ty {
        GgufRowType::Q4_0 => dequantize_blocks::<BlockQ4_0>(src, out),
        GgufRowType::Q4_1 => dequantize_blocks::<BlockQ4_1>(src, out),
        GgufRowType::Q5_0 => dequantize_blocks::<BlockQ5_0>(src, out),
        GgufRowType::Q5_1 => dequantize_blocks::<BlockQ5_1>(src, out),
        GgufRowType::Q8_0 => dequantize_blocks::<BlockQ8_0>(src, out),
        GgufRowType::Q8_1 => dequantize_blocks::<BlockQ8_1>(src, out),
        GgufRowType::Q2K => dequantize_blocks::<BlockQ2K>(src, out),
        GgufRowType::Q3K => dequantize_blocks::<BlockQ3K>(src, out),
        GgufRowType::Q4K => dequantize_blocks::<BlockQ4K>(src, out),
        GgufRowType::Q5K => dequantize_blocks::<BlockQ5K>(src, out),
        GgufRowType::Q6K => dequantize_blocks::<BlockQ6K>(src, out),
        GgufRowType::Q8K => dequantize_blocks::<BlockQ8K>(src, out),
        GgufRowType::Mxfp4 => dequantize_blocks::<BlockMxfp4>(src, out),
        GgufRowType::Iq4Nl => {
            for (block, dst) in src
                .as_chunks::<IQ4_BLOCK_BYTES>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<IQ4_BLOCK>().0)
            {
                let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                for (j, &byte) in block[2..].iter().enumerate() {
                    dst[j] = scale * f32::from(IQ4_VALUES[usize::from(byte & 15)]);
                    dst[j + IQ4_BLOCK / 2] = scale * f32::from(IQ4_VALUES[usize::from(byte >> 4)]);
                }
            }
        }
    }
}

/// Multiplier applied to every value of a row on top of its block scales.
#[derive(Clone, Copy, Debug)]
pub enum RowScale<'a> {
    None,
    /// Known on the host; folded into the launch.
    Value(f32),
    /// One-element device tensor, read on the device so gathers never sync.
    Scalar(&'a Tensor),
    /// One value per table row.
    PerRow(&'a Tensor),
}

/// A row-major region holding encoded rows.
#[derive(Clone, Copy, Debug)]
pub enum TableBytes<'a> {
    Tensor(&'a Tensor),
    QTensor(&'a QTensor),
}

/// Host shard the device may dereference through pageable memory access (ATS/HMM).
#[derive(Clone, Copy, Debug)]
pub struct MappedShard {
    pub data: *const u8,
    pub scales: *const u8,
    pub first_row: usize,
    pub rows: usize,
}

/// Rows split across host shards, e.g. a memory-mapped checkpoint table.
#[derive(Debug)]
pub struct MappedRows {
    shards: Vec<MappedShard>,
    index: Option<Tensor>,
    _owner: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    #[cfg(feature = "cuda")]
    cuda_lifetime: Option<std::sync::Arc<mapped_lifetime::CudaMappedLifetime>>,
}

// Shards point into `_owner`, which is immutable and shared.
unsafe impl Send for MappedRows {}
unsafe impl Sync for MappedRows {}

impl MappedRows {
    /// # Safety
    /// Every shard must point into memory `owner` keeps mapped and unchanged, sized for its rows. Shards must
    /// be sorted by `first_row` and contiguous.
    pub unsafe fn new(
        shards: Vec<MappedShard>,
        owner: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    ) -> Self {
        Self {
            shards,
            index: None,
            _owner: owner,
            #[cfg(feature = "cuda")]
            cuda_lifetime: None,
        }
    }

    pub fn has_device_index(&self) -> bool {
        self.index.is_some()
    }

    /// Uploads the shard index so `device` can gather directly; the device must read pageable memory.
    pub fn with_device_index(mut self, device: &Device) -> Result<Self> {
        if !device.is_cuda() {
            candle_core::bail!("mapped embedding device indices require CUDA");
        }
        #[cfg(feature = "cuda")]
        if !device_reads_pageable_memory(device)? {
            candle_core::bail!(
                "mapped embedding device indices require CUDA pageable memory access"
            );
        }
        let index = self
            .shards
            .iter()
            .flat_map(|s| [s.data as i64, s.scales as i64, s.first_row as i64])
            .collect::<Vec<_>>();
        let len = index.len();
        let index = Tensor::from_vec(index, len, device)?;
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            self.cuda_lifetime = Some(std::sync::Arc::new(
                mapped_lifetime::CudaMappedLifetime::new(index.clone(), self._owner.clone()),
            ));
        }
        self.index = Some(index);
        Ok(self)
    }

    #[cfg(feature = "cuda")]
    fn cuda_read<'a>(
        &'a self,
        stream: &'a std::sync::Arc<candle_core::cuda::cudarc::driver::CudaStream>,
    ) -> Result<mapped_lifetime::MappedRead<'a>> {
        self.cuda_lifetime
            .as_ref()
            .ok_or_else(|| candle_core::Error::msg("mapped embedding rows have no CUDA index"))?
            .read(stream)
    }

    pub fn rows(&self) -> usize {
        self.shards.last().map_or(0, |s| s.first_row + s.rows)
    }

    fn shard(&self, row: usize) -> &MappedShard {
        let index = self.shards.partition_point(|s| s.first_row <= row) - 1;
        &self.shards[index]
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TableStorage<'a> {
    Device {
        data: TableBytes<'a>,
        scales: Option<&'a Tensor>,
        aux: Option<&'a Tensor>,
    },
    Mapped(&'a MappedRows),
}

/// Encoded embedding table that `gather` decodes row by row without materializing the dense table.
#[derive(Clone, Copy, Debug)]
pub struct EmbeddingTable<'a> {
    pub codec: RowCodec,
    pub rows: usize,
    pub columns: usize,
    pub storage: TableStorage<'a>,
    pub row_scale: RowScale<'a>,
}

impl<'a> EmbeddingTable<'a> {
    fn with_storage(
        codec: RowCodec,
        rows: usize,
        columns: usize,
        storage: TableStorage<'a>,
    ) -> Self {
        Self {
            codec,
            rows,
            columns,
            storage,
            row_scale: RowScale::None,
        }
    }

    fn device_storage(data: TableBytes<'a>) -> TableStorage<'a> {
        TableStorage::Device {
            data,
            scales: None,
            aux: None,
        }
    }

    pub fn new(codec: RowCodec, rows: usize, columns: usize, data: &'a Tensor) -> Self {
        Self::with_storage(
            codec,
            rows,
            columns,
            Self::device_storage(TableBytes::Tensor(data)),
        )
    }

    pub fn with_scales(mut self, scales: &'a Tensor) -> Self {
        if let TableStorage::Device { scales: s, .. } = &mut self.storage {
            *s = Some(scales);
        }
        self
    }

    pub fn with_aux(mut self, aux: &'a Tensor) -> Self {
        if let TableStorage::Device { aux: a, .. } = &mut self.storage {
            *a = Some(aux);
        }
        self
    }

    pub fn with_row_scale(mut self, row_scale: RowScale<'a>) -> Self {
        self.row_scale = row_scale;
        self
    }

    pub fn quantized(qtensor: &'a QTensor) -> Result<Self> {
        let (rows, columns) = qtensor.shape().dims2()?;
        let storage = Self::device_storage(TableBytes::QTensor(qtensor));
        Ok(Self::with_storage(
            RowCodec::from_ggml(qtensor.dtype()),
            rows,
            columns,
            storage,
        ))
    }

    pub fn mapped(codec: RowCodec, columns: usize, rows: &'a MappedRows) -> Self {
        Self::with_storage(codec, rows.rows(), columns, TableStorage::Mapped(rows))
    }

    /// Device that holds the encoded rows, or `None` for host shards.
    pub fn device(&self) -> Option<Device> {
        match self.storage {
            TableStorage::Device {
                data: TableBytes::Tensor(t),
                ..
            } => Some(t.device().clone()),
            TableStorage::Device {
                data: TableBytes::QTensor(q),
                ..
            } => Some(q.device()),
            TableStorage::Mapped(_) => None,
        }
    }

    /// Whether rows can be gathered on `device` without dequantizing the table.
    pub fn gathers_on(&self, device: &Device) -> bool {
        let cuda = cfg!(feature = "cuda") && device.is_cuda();
        let metal = cfg!(feature = "metal") && device.is_metal();
        match self.storage {
            TableStorage::Mapped(rows) => {
                device.is_cpu()
                    || cuda
                        && rows
                            .index
                            .as_ref()
                            .is_some_and(|index| index.device().same_device(device))
            }
            TableStorage::Device { .. } => {
                self.device().is_some_and(|d| d.same_device(device))
                    && (device.is_cpu() || cuda || metal)
            }
        }
    }

    /// Rows `ids` (any shape) decoded to `[..ids, columns]` on the table's device. IDs past the table give
    /// zero rows on CUDA and an error on the host.
    pub fn gather(&self, ids: &Tensor, out: EmbeddingOutput) -> Result<Tensor> {
        let mut shape = ids.dims().to_vec();
        shape.push(self.columns);
        let device = self.device().unwrap_or_else(|| ids.device().clone());
        #[cfg(feature = "cuda")]
        if device.is_cuda() && self.gathers_on(&device) {
            let ids = ids.to_device(&device)?;
            return cuda::gather(self, &ids, out)?.reshape(shape);
        }
        #[cfg(feature = "metal")]
        if device.is_metal() && matches!(self.storage, TableStorage::Device { .. }) {
            let ids = ids.to_device(&device)?;
            return metal::gather(self, &ids, out)?.reshape(shape);
        }
        if let (
            RowCodec::Dense(_),
            TableStorage::Device {
                data: TableBytes::Tensor(w),
                ..
            },
        ) = (self.codec, self.storage)
        {
            if matches!(self.row_scale, RowScale::None | RowScale::Value(_)) {
                return dense_gather(w, ids, self.host_multiplier(out), out.dtype);
            }
        }
        let host = ids.flatten_all()?.to_dtype(DType::U32)?.to_vec1::<u32>()?;
        self.gather_host(&host.into_iter().map(u64::from).collect::<Vec<_>>(), out)?
            .to_device(&device)?
            .reshape(shape)
    }

    fn host_multiplier(&self, out: EmbeddingOutput) -> f32 {
        match self.row_scale {
            RowScale::Value(v) => v * out.multiplier(),
            _ => out.multiplier(),
        }
    }

    /// Copies selected encoded host rows to `device`, then decodes them with its gather kernel.
    pub fn gather_staged(
        &self,
        ids: &[u64],
        out: EmbeddingOutput,
        device: &Device,
    ) -> Result<Tensor> {
        self.codec.validate(self.columns)?;
        if ids.is_empty() {
            return Tensor::zeros((0, self.columns), out.dtype, device);
        }
        let row_bytes = self.codec.row_bytes(self.columns);
        let grid = self.codec.grid();
        let scale_bytes = grid.map_or(0, |g| g.row_bytes(self.columns));
        let host = HostRows::new(self.storage)?;
        let mut data = Vec::with_capacity(ids.len() * row_bytes);
        let mut scales = grid.map(|_| Vec::with_capacity(ids.len() * scale_bytes));
        let mut aux = matches!(self.codec, RowCodec::Afq { .. })
            .then(|| Vec::with_capacity(ids.len() * scale_bytes));
        for &row in ids {
            let r = row as usize;
            if r >= self.rows {
                candle_core::bail!("embedding row {row} is outside {} rows", self.rows);
            }
            let (src, scale_src, aux_src) = host.row(r, self.codec, self.columns)?;
            data.extend_from_slice(src);
            if let Some(scales) = &mut scales {
                scales.extend_from_slice(scale_src);
            }
            if let Some(aux) = &mut aux {
                aux.extend_from_slice(aux_src);
            }
        }
        let data = match self.codec {
            RowCodec::Dense(dtype) => {
                Tensor::from_raw_buffer(&data, dtype, &[ids.len(), self.columns], device)?
            }
            _ => Tensor::from_vec(data, (ids.len(), row_bytes), device)?,
        };
        let upload = |bytes: Vec<u8>| Tensor::from_vec(bytes, (ids.len(), scale_bytes), device);
        let scales = scales.map(upload).transpose()?;
        let aux = aux.map(upload).transpose()?;
        let row_scales = match self.row_scale {
            RowScale::Scalar(t) => Some(t.to_device(device)?),
            RowScale::PerRow(t) => {
                let indices = Tensor::from_vec(
                    ids.iter().map(|&id| id as i64).collect::<Vec<_>>(),
                    ids.len(),
                    t.device(),
                )?;
                Some(
                    t.flatten_all()?
                        .index_select(&indices, 0)?
                        .to_device(device)?,
                )
            }
            RowScale::None | RowScale::Value(_) => None,
        };
        let mut codec = self.codec;
        match &mut codec {
            RowCodec::Fp4(grid)
            | RowCodec::Fp8(Some(grid))
            | RowCodec::Afq { grid, .. }
            | RowCodec::SymInt { grid, .. } => grid.rows_per_scale = 1,
            _ => {}
        }
        let row_scale = match (self.row_scale, row_scales.as_ref()) {
            (RowScale::Scalar(_), Some(t)) => RowScale::Scalar(t),
            (RowScale::PerRow(_), Some(t)) => RowScale::PerRow(t),
            _ => self.row_scale,
        };
        let table = EmbeddingTable {
            codec,
            rows: ids.len(),
            columns: self.columns,
            storage: TableStorage::Device {
                data: TableBytes::Tensor(&data),
                scales: scales.as_ref(),
                aux: aux.as_ref(),
            },
            row_scale,
        };
        table.gather(&Tensor::arange(0i64, ids.len() as i64, device)?, out)
    }

    /// Host decode of `ids` into a CPU `[ids.len(), columns]` tensor of `out.dtype`.
    pub fn gather_host(&self, ids: &[u64], out: EmbeddingOutput) -> Result<Tensor> {
        self.codec.validate(self.columns)?;
        let multiplier = self.host_multiplier(out);
        let row_scales = match self.row_scale {
            RowScale::Scalar(t) | RowScale::PerRow(t) => Some(
                t.flatten_all()?
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?,
            ),
            RowScale::None | RowScale::Value(_) => None,
        };
        let per_row = matches!(self.row_scale, RowScale::PerRow(_));
        let host = HostRows::new(self.storage)?;
        let mut values = vec![0f32; ids.len() * self.columns];
        for (dst, &row) in values.chunks_exact_mut(self.columns).zip(ids) {
            let r = row as usize;
            if r >= self.rows {
                candle_core::bail!("embedding row {row} is outside {} rows", self.rows);
            }
            let (data, scales, aux) = host.row(r, self.codec, self.columns)?;
            self.codec.decode_row(data, scales, aux, dst);
            let m = multiplier
                * row_scales
                    .as_ref()
                    .map_or(1.0, |v| v[if per_row { r } else { 0 }]);
            dst.iter_mut().for_each(|v| *v *= m);
        }
        Tensor::from_vec(values, (ids.len(), self.columns), &Device::Cpu)?.to_dtype(out.dtype)
    }
}

#[cfg(feature = "cuda")]
fn device_reads_pageable_memory(device: &Device) -> Result<bool> {
    use candle_core::cuda::cudarc::driver::{result, sys};
    let dev = device.as_cuda_device()?.cuda_stream().context().cu_device();
    let supported = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
        )
    }
    .map_err(candle_core::Error::wrap)?;
    Ok(supported == 1)
}

/// Applies `out` to rows a format decoded on its own.
pub(crate) fn finish_rows(rows: Tensor, out: EmbeddingOutput) -> Result<Tensor> {
    scale_rows(rows, out.multiplier(), out.dtype)
}

fn scale_row(bytes: Option<&[u8]>, row: usize, width: usize) -> &[u8] {
    bytes.map_or(&[], |b| &b[row * width..(row + 1) * width])
}

fn scale_rows(rows: Tensor, multiplier: f32, dtype: DType) -> Result<Tensor> {
    if multiplier == 1.0 {
        return rows.to_dtype(dtype);
    }
    if rows.dtype() == dtype {
        return rows.affine(f64::from(multiplier), 0.0);
    }
    rows.to_dtype(DType::F32)?
        .affine(f64::from(multiplier), 0.0)?
        .to_dtype(dtype)
}

/// Default `QuantMethod::embedding_forward`: decode rows in place, else dequantize the table.
pub fn forward<Q: crate::QuantMethod + ?Sized>(
    layer: &Q,
    ids: &Tensor,
    out: EmbeddingOutput,
) -> Result<Tensor> {
    if let Some(table) = layer.embedding_table() {
        if table
            .device()
            .is_some_and(|device| table.gathers_on(&device))
        {
            return table.gather(ids, out);
        }
    }
    dense_gather(&layer.dequantize_w()?, ids, out.multiplier(), out.dtype)
}

fn dense_gather(table: &Tensor, ids: &Tensor, multiplier: f32, dtype: DType) -> Result<Tensor> {
    let mut shape = ids.dims().to_vec();
    shape.push(table.dim(candle_core::D::Minus1)?);
    let ids = ids.to_device(table.device())?.flatten_all()?;
    scale_rows(table.index_select(&ids, 0)?, multiplier, dtype)?.reshape(shape)
}

/// Host bytes of a CPU tensor or quantized tensor, kept borrowed while decoding.
enum HostBytes<'a> {
    Tensor(
        Box<dyn std::ops::Deref<Target = Storage> + 'a>,
        usize,
        usize,
    ),
    Quantized(Cow<'a, [u8]>),
}

impl<'a> HostBytes<'a> {
    fn new(bytes: TableBytes<'a>) -> Result<Self> {
        match bytes {
            TableBytes::Tensor(t) => {
                let (storage, layout) = t.storage_and_layout();
                if !layout.is_contiguous() {
                    candle_core::bail!("embedding tables must be contiguous");
                }
                let size = t.dtype().size_in_bytes();
                Ok(Self::Tensor(
                    Box::new(storage),
                    layout.start_offset() * size,
                    t.elem_count() * size,
                ))
            }
            TableBytes::QTensor(q) => Ok(Self::Quantized(q.data()?)),
        }
    }

    fn bytes(&self) -> Result<&[u8]> {
        match self {
            Self::Quantized(bytes) => Ok(bytes),
            Self::Tensor(storage, offset, len) => {
                let Storage::Cpu(cpu) = &***storage else {
                    candle_core::bail!("host embedding decode needs CPU storage");
                };
                Ok(&cpu_bytes(cpu)?[*offset..*offset + *len])
            }
        }
    }
}

enum HostRows<'a> {
    Contiguous {
        data: HostBytes<'a>,
        scales: Option<HostBytes<'a>>,
        aux: Option<HostBytes<'a>>,
    },
    Mapped(&'a MappedRows),
}

impl<'a> HostRows<'a> {
    fn new(storage: TableStorage<'a>) -> Result<Self> {
        Ok(match storage {
            TableStorage::Device { data, scales, aux } => {
                let host = |t: Option<&'a Tensor>| {
                    t.map(|t| HostBytes::new(TableBytes::Tensor(t))).transpose()
                };
                Self::Contiguous {
                    data: HostBytes::new(data)?,
                    scales: host(scales)?,
                    aux: host(aux)?,
                }
            }
            TableStorage::Mapped(rows) => Self::Mapped(rows),
        })
    }

    fn row(&self, row: usize, codec: RowCodec, columns: usize) -> Result<(&[u8], &[u8], &[u8])> {
        let row_bytes = codec.row_bytes(columns);
        let grid = codec.grid();
        let scale_bytes = grid.map_or(0, |g| g.row_bytes(columns));
        let rows_per_scale = grid.map_or(1, |g| g.rows_per_scale);
        Ok(match self {
            Self::Contiguous { data, scales, aux } => (
                &data.bytes()?[row * row_bytes..(row + 1) * row_bytes],
                scale_row(
                    scales.as_ref().map(HostBytes::bytes).transpose()?,
                    row / rows_per_scale,
                    scale_bytes,
                ),
                scale_row(
                    aux.as_ref().map(HostBytes::bytes).transpose()?,
                    row / rows_per_scale,
                    scale_bytes,
                ),
            ),
            Self::Mapped(mapped) => {
                let shard = mapped.shard(row);
                let local = row - shard.first_row;
                let data = unsafe {
                    std::slice::from_raw_parts(shard.data.add(local * row_bytes), row_bytes)
                };
                let scales = if shard.scales.is_null() {
                    &[][..]
                } else {
                    unsafe {
                        std::slice::from_raw_parts(
                            shard.scales.add(local / rows_per_scale * scale_bytes),
                            scale_bytes,
                        )
                    }
                };
                (data, scales, &[][..])
            }
        })
    }
}

fn cpu_bytes(storage: &candle_core::CpuStorage) -> Result<&[u8]> {
    use candle_core::CpuStorage as S;
    fn raw<T>(values: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        }
    }
    Ok(match storage {
        S::U8(v) => raw(v),
        S::U32(v) => raw(v),
        S::I32(v) => raw(v),
        S::I64(v) => raw(v),
        S::BF16(v) => raw(v),
        S::F16(v) => raw(v),
        S::F32(v) => raw(v),
        S::F8E4M3(v) => raw(v),
        _ => candle_core::bail!("unsupported embedding storage dtype"),
    })
}

#[cfg(all(test, feature = "metal"))]
mod metal_tests;
#[cfg(test)]
mod tests;
