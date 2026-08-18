use std::{
    borrow::Cow,
    collections::{hash_map::Entry, HashMap, HashSet},
    fs::File,
    mem::{align_of, size_of},
    ops::Range,
    path::{Path, PathBuf},
};

use byteorder::{BigEndian, ByteOrder, LittleEndian};
use candle_core::{
    quantized::{
        gguf_file::Value,
        k_quants::{
            BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1,
            BlockQ6K, BlockQ8K, BlockQ8_0, BlockQ8_1,
        },
        GgmlDType, QStorage, QTensor,
    },
    Device, Error, Result,
};
use half::{bf16, f16};
use memmap2::{Mmap, MmapOptions};

const DEFAULT_ALIGNMENT: usize = 32;
const MAX_STRING_LENGTH: u64 = 1 << 30;
const MAX_ARRAY_ELEMENTS: u64 = 1 << 30;
const MAX_TENSOR_DIMS: u32 = 4;
const MAX_VALUE_DEPTH: usize = 64;
const SPLIT_NO: &str = "split.no";
const SPLIT_COUNT: &str = "split.count";
const SPLIT_TENSORS_COUNT: &str = "split.tensors.count";
const GENERAL_ALIGNMENT: &str = "general.alignment";
const GENERAL_TYPE: &str = "general.type";
const GENERAL_PREFIX: &str = "general.";
const SPLIT_PREFIX: &str = "split.";
const MMPROJ_TYPE: &str = "mmproj";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufEndian {
    Little,
    Big,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufVersion {
    V1,
    V2,
    V3,
}

impl GgufVersion {
    const fn length_prefix_size(self) -> usize {
        match self {
            Self::V1 => 4,
            Self::V2 | Self::V3 => 8,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GgufDType(u32);

impl GgufDType {
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    pub const fn name(self) -> &'static str {
        match self.0 {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            9 => "Q8_1",
            10 => "Q2_K",
            11 => "Q3_K",
            12 => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            15 => "Q8_K",
            16 => "IQ2_XXS",
            17 => "IQ2_XS",
            18 => "IQ3_XXS",
            19 => "IQ1_S",
            20 => "IQ4_NL",
            21 => "IQ3_S",
            22 => "IQ2_S",
            23 => "IQ4_XS",
            24 => "I8",
            25 => "I16",
            26 => "I32",
            27 => "I64",
            28 => "F64",
            29 => "IQ1_M",
            30 => "BF16",
            34 => "TQ1_0",
            35 => "TQ2_0",
            39 => "MXFP4",
            40 => "NVFP4",
            41 => "Q1_0",
            _ => "UNKNOWN",
        }
    }

    pub const fn block_size(self) -> Option<usize> {
        match self.0 {
            0 | 1 | 24..=28 | 30 => Some(1),
            2 | 3 | 6..=9 | 20 | 39 => Some(32),
            40 => Some(64),
            41 => Some(128),
            10..=19 | 21..=23 | 29 | 34 | 35 => Some(256),
            _ => None,
        }
    }

    pub const fn type_size(self) -> Option<usize> {
        match self.0 {
            0 => Some(4),
            1 => Some(2),
            2 => Some(18),
            3 => Some(20),
            6 => Some(22),
            7 => Some(24),
            8 => Some(34),
            9 => Some(36),
            10 => Some(84),
            11 => Some(110),
            12 => Some(144),
            13 => Some(176),
            14 => Some(210),
            15 => Some(292),
            16 => Some(66),
            17 => Some(74),
            18 => Some(98),
            19 => Some(50),
            20 => Some(18),
            21 => Some(110),
            22 => Some(82),
            23 => Some(136),
            24 => Some(1),
            25 | 30 => Some(2),
            26 => Some(4),
            27 | 28 => Some(8),
            29 => Some(56),
            34 => Some(54),
            35 => Some(66),
            39 => Some(17),
            40 => Some(36),
            41 => Some(18),
            _ => None,
        }
    }

    pub fn candle_dtype(self) -> Result<GgmlDType> {
        let dtype = match self.0 {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            30 => GgmlDType::BF16,
            raw => candle_core::bail!("GGUF dtype {raw} is not supported by Candle"),
        };
        Ok(dtype)
    }

    fn tensor_byte_len(self, name: &str, shape: &[usize]) -> Result<Option<usize>> {
        let (Some(block_size), Some(type_size)) = (self.block_size(), self.type_size()) else {
            return Ok(None);
        };
        let row_size = shape.last().copied().unwrap_or(1);
        if !row_size.is_multiple_of(block_size) {
            candle_core::bail!(
                "GGUF tensor `{name}` has {row_size} elements per row, not a multiple of dtype {} block size {block_size}",
                self.0
            );
        }
        let elem_count = checked_elem_count(name, shape)?;
        let blocks = elem_count
            .checked_div(block_size)
            .ok_or_else(|| Error::msg(format!("invalid block size for GGUF tensor `{name}`")))?;
        Ok(Some(blocks.checked_mul(type_size).ok_or_else(|| {
            Error::msg(format!("byte size overflow for GGUF tensor `{name}`"))
        })?))
    }
}

#[derive(Clone, Debug)]
pub struct GgufShardInfo {
    path: PathBuf,
    version: GgufVersion,
    endian: GgufEndian,
    alignment: usize,
    tensor_data_offset: usize,
    file_len: usize,
}

impl GgufShardInfo {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub const fn version(&self) -> GgufVersion {
        self.version
    }

    pub const fn endian(&self) -> GgufEndian {
        self.endian
    }

    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    pub const fn tensor_data_offset(&self) -> usize {
        self.tensor_data_offset
    }

    pub const fn file_len(&self) -> usize {
        self.file_len
    }
}

#[derive(Clone, Debug)]
pub struct GgufTensorInfo {
    name: String,
    shape: Vec<usize>,
    dtype: GgufDType,
    shard_index: usize,
    relative_offset: u64,
    data_range: Option<Range<usize>>,
    storage_range: Range<usize>,
}

impl GgufTensorInfo {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub const fn dtype(&self) -> GgufDType {
        self.dtype
    }

    pub const fn shard_index(&self) -> usize {
        self.shard_index
    }

    pub const fn relative_offset(&self) -> u64 {
        self.relative_offset
    }

    pub fn data_range(&self) -> Option<Range<usize>> {
        self.data_range.clone()
    }

    pub fn storage_range(&self) -> Range<usize> {
        self.storage_range.clone()
    }

    pub fn elem_count(&self) -> Result<usize> {
        checked_elem_count(&self.name, &self.shape)
    }

    pub fn byte_len(&self) -> Option<usize> {
        self.data_range.as_ref().map(Range::len)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GgufTensorData<'a> {
    info: &'a GgufTensorInfo,
    bytes: &'a [u8],
}

impl<'a> GgufTensorData<'a> {
    pub const fn info(self) -> &'a GgufTensorInfo {
        self.info
    }

    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }
}

impl AsRef<[u8]> for GgufTensorData<'_> {
    fn as_ref(&self) -> &[u8] {
        self.bytes
    }
}

fn parse_shards<I, P>(paths: I) -> Result<Vec<ParsedShard>>
where
    I: IntoIterator<Item = P>,
    P: AsRef<Path>,
{
    paths
        .into_iter()
        .map(|path| {
            let path = path.as_ref().to_path_buf();
            let file = File::open(&path)
                .map_err(Error::wrap)
                .map_err(|err| err.with_path(&path))?;
            // The mapping is read-only and remains owned by the archive.
            let mmap = unsafe { MmapOptions::new().map(&file) }
                .map_err(Error::wrap)
                .map_err(|err| err.with_path(&path))?;
            ParsedShard::parse(path.clone(), mmap).map_err(|err| err.with_path(&path))
        })
        .collect()
}

pub struct GgufArchive {
    mappings: Vec<Mmap>,
    shards: Vec<GgufShardInfo>,
    metadata: HashMap<String, Value>,
    tensors: HashMap<String, GgufTensorInfo>,
}

impl std::fmt::Debug for GgufArchive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufArchive")
            .field("shards", &self.shards)
            .field("metadata", &self.metadata)
            .field("tensors", &self.tensors)
            .finish_non_exhaustive()
    }
}

impl GgufArchive {
    pub fn open<I, P>(paths: I) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        Self::from_parsed(parse_shards(paths)?)
    }

    /// Opens independent GGUF components while keeping adjacent split shards in one archive.
    pub fn open_components<I, P>(paths: I) -> Result<Vec<Self>>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut parsed = parse_shards(paths)?;
        if parsed.is_empty() {
            candle_core::bail!("at least one GGUF file is required");
        }

        let mut components = Vec::new();
        while !parsed.is_empty() {
            let has_split_metadata = parsed[0].metadata.contains_key(SPLIT_NO)
                || parsed[0].metadata.contains_key(SPLIT_COUNT);
            let count = if has_split_metadata {
                metadata_usize(&parsed[0].metadata, SPLIT_COUNT)?.ok_or_else(|| {
                    Error::msg(format!(
                        "GGUF shard `{}` is missing `{SPLIT_COUNT}`",
                        parsed[0].path.display()
                    ))
                })?
            } else {
                1
            };
            if count == 0 {
                candle_core::bail!("GGUF `{SPLIT_COUNT}` must be positive");
            }
            if count > parsed.len() {
                candle_core::bail!(
                    "{} GGUF shards remain, but split metadata declares {count}",
                    parsed.len()
                );
            }
            let component = parsed.drain(..count).collect::<Vec<_>>();
            components.push(Self::from_parsed(component)?);
        }
        Ok(components)
    }

    fn from_parsed(mut parsed: Vec<ParsedShard>) -> Result<Self> {
        if parsed.is_empty() {
            candle_core::bail!("at least one GGUF file is required");
        }

        validate_and_order_splits(&mut parsed)?;
        let alignment = archive_alignment(&parsed)?;
        let declared_total = declared_tensor_count(&parsed)?;
        let mut mappings = Vec::with_capacity(parsed.len());
        let mut shards = Vec::with_capacity(parsed.len());
        let mut metadata = HashMap::new();
        let mut tensors = HashMap::new();

        for (shard_index, mut shard) in parsed.into_iter().enumerate() {
            merge_metadata(
                &mut metadata,
                std::mem::take(&mut shard.metadata),
                shard_index,
            )?;
            let (mmap, shard_info, shard_tensors) = shard.finish(shard_index, alignment)?;
            for tensor in shard_tensors {
                let name = tensor.name.clone();
                match tensors.entry(name.clone()) {
                    Entry::Vacant(entry) => {
                        entry.insert(tensor);
                    }
                    Entry::Occupied(_) => {
                        candle_core::bail!("GGUF tensor `{name}` is duplicated across shards")
                    }
                }
            }
            mappings.push(mmap);
            shards.push(shard_info);
        }

        if let Some(expected) = declared_total {
            if tensors.len() != expected {
                candle_core::bail!(
                    "GGUF split metadata declares {expected} tensors, but {} were cataloged",
                    tensors.len()
                );
            }
        }

        Ok(Self {
            mappings,
            shards,
            metadata,
            tensors,
        })
    }

    pub fn open_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::open([path])
    }

    pub fn merge_component(&mut self, component: Self) -> Result<()> {
        self.merge_components([component])
    }

    pub fn merge_components<I>(&mut self, components: I) -> Result<()>
    where
        I: IntoIterator<Item = Self>,
    {
        let components = components.into_iter().collect::<Vec<_>>();
        let mut metadata = self.metadata.clone();
        let mut tensor_names = self.tensors.keys().cloned().collect::<HashSet<_>>();

        for component in &components {
            validate_component_type(component)?;
            for (key, value) in &component.metadata {
                if key.starts_with(GENERAL_PREFIX) || key.starts_with(SPLIT_PREFIX) {
                    continue;
                }
                match metadata.entry(key.clone()) {
                    Entry::Vacant(entry) => {
                        entry.insert(value.clone());
                    }
                    Entry::Occupied(entry) => {
                        if !values_equal(entry.get(), value) {
                            candle_core::bail!(
                                "GGUF component metadata key `{}` conflicts with the main archive or another component",
                                entry.key()
                            );
                        }
                    }
                }
            }
            for name in component.tensors.keys() {
                if !tensor_names.insert(name.clone()) {
                    candle_core::bail!("GGUF tensor `{name}` is duplicated across components");
                }
            }
        }

        self.metadata = metadata;
        for component in components {
            let shard_offset = self.shards.len();
            let Self {
                mappings,
                shards,
                tensors,
                ..
            } = component;
            for (name, mut tensor) in tensors {
                tensor.shard_index += shard_offset;
                self.tensors.insert(name, tensor);
            }
            self.mappings.extend(mappings);
            self.shards.extend(shards);
        }
        Ok(())
    }

    pub fn shards(&self) -> &[GgufShardInfo] {
        &self.shards
    }

    pub fn metadata(&self) -> &HashMap<String, Value> {
        &self.metadata
    }

    pub fn metadata_value(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    pub fn tensors(&self) -> &HashMap<String, GgufTensorInfo> {
        &self.tensors
    }

    pub fn tensor_info(&self, name: &str) -> Result<&GgufTensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::msg(format!("cannot find GGUF tensor `{name}`")))
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn tensor_data(&self, name: &str) -> Result<GgufTensorData<'_>> {
        let info = self.tensor_info(name)?;
        let range = info.data_range.as_ref().ok_or_else(|| {
            Error::msg(format!(
                "cannot determine the exact byte length of GGUF tensor `{name}` with dtype {}",
                info.dtype.raw()
            ))
        })?;
        Ok(GgufTensorData {
            info,
            bytes: &self.mappings[info.shard_index][range.clone()],
        })
    }

    pub fn tensor_storage_data(&self, name: &str) -> Result<GgufTensorData<'_>> {
        let info = self.tensor_info(name)?;
        Ok(GgufTensorData {
            info,
            bytes: &self.mappings[info.shard_index][info.storage_range.clone()],
        })
    }

    pub fn load_qtensor(&self, name: &str, device: &Device) -> Result<QTensor> {
        let data = self.tensor_data(name)?;
        let info = data.info();
        if self.shards[info.shard_index].endian != GgufEndian::Little {
            candle_core::bail!("big-endian GGUF tensor loading is not supported");
        }
        qtensor_from_gguf_data(
            info.dtype.candle_dtype()?,
            data.bytes(),
            info.shape.clone(),
            device,
        )
    }
}

pub(super) fn qtensor_from_gguf_data(
    dtype: GgmlDType,
    data: &[u8],
    dims: Vec<usize>,
    device: &Device,
) -> Result<QTensor> {
    let elem_count = dims.iter().try_fold(1usize, |count, dim| {
        count
            .checked_mul(*dim)
            .ok_or_else(|| Error::msg("GGUF tensor element count overflow"))
    })?;
    let block_size = dtype.block_size();
    if !elem_count.is_multiple_of(block_size) {
        candle_core::bail!(
            "GGUF tensor has {elem_count} elements, not divisible by {dtype:?} block size {block_size}"
        );
    }
    let expected = (elem_count / block_size)
        .checked_mul(dtype.type_size())
        .ok_or_else(|| Error::msg("GGUF tensor byte length overflow"))?;
    if data.len() != expected {
        candle_core::bail!(
            "GGUF {dtype:?} tensor contains {} bytes, expected {expected}",
            data.len()
        );
    }
    let storage = if data.as_ptr().align_offset(ggml_dtype_alignment(dtype)) == 0 {
        QStorage::from_data(Cow::Borrowed(data), device, dtype)?
    } else {
        let mut aligned = vec![0u128; data.len().div_ceil(size_of::<u128>())];
        let bytes = bytemuck::cast_slice_mut(&mut aligned);
        bytes[..data.len()].copy_from_slice(data);
        QStorage::from_data(Cow::Borrowed(&bytes[..data.len()]), device, dtype)?
    };
    QTensor::new(storage, dims)
}

const fn ggml_dtype_alignment(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::F32 => align_of::<f32>(),
        GgmlDType::F16 => align_of::<f16>(),
        GgmlDType::Q4_0 => align_of::<BlockQ4_0>(),
        GgmlDType::Q4_1 => align_of::<BlockQ4_1>(),
        GgmlDType::Q5_0 => align_of::<BlockQ5_0>(),
        GgmlDType::Q5_1 => align_of::<BlockQ5_1>(),
        GgmlDType::Q8_0 => align_of::<BlockQ8_0>(),
        GgmlDType::Q8_1 => align_of::<BlockQ8_1>(),
        GgmlDType::Q2K => align_of::<BlockQ2K>(),
        GgmlDType::Q3K => align_of::<BlockQ3K>(),
        GgmlDType::Q4K => align_of::<BlockQ4K>(),
        GgmlDType::Q5K => align_of::<BlockQ5K>(),
        GgmlDType::Q6K => align_of::<BlockQ6K>(),
        GgmlDType::Q8K => align_of::<BlockQ8K>(),
        GgmlDType::BF16 => align_of::<bf16>(),
    }
}

#[derive(Debug)]
struct RawTensorInfo {
    name: String,
    shape: Vec<usize>,
    dtype: GgufDType,
    offset: u64,
}

struct ParsedShard {
    path: PathBuf,
    mmap: Mmap,
    version: GgufVersion,
    endian: GgufEndian,
    metadata: HashMap<String, Value>,
    tensors: Vec<RawTensorInfo>,
    header_end: usize,
}

impl ParsedShard {
    fn parse(path: PathBuf, mmap: Mmap) -> Result<Self> {
        let (endian, initial_offset) = parse_magic(&mmap)?;
        let mut reader = SliceReader::new(&mmap, initial_offset, endian);
        let version = match reader.read_u32()? {
            1 => GgufVersion::V1,
            2 => GgufVersion::V2,
            3 => GgufVersion::V3,
            version => candle_core::bail!("unsupported GGUF version {version}"),
        };
        let tensor_count = reader.read_length(version)?;
        let metadata_count = reader.read_length(version)?;
        if tensor_count > MAX_ARRAY_ELEMENTS {
            candle_core::bail!(
                "GGUF tensor count {tensor_count} exceeds maximum {MAX_ARRAY_ELEMENTS}"
            );
        }
        if metadata_count > MAX_ARRAY_ELEMENTS {
            candle_core::bail!(
                "GGUF metadata count {metadata_count} exceeds maximum {MAX_ARRAY_ELEMENTS}"
            );
        }
        validate_header_minimum(reader.remaining(), version, tensor_count, metadata_count)?;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let key = reader.read_string(version)?;
            let value_type = ValueType::from_raw(reader.read_u32()?)?;
            let value = reader.read_value(value_type, version, 0)?;
            if metadata.insert(key.clone(), value).is_some() {
                candle_core::bail!("GGUF metadata key `{key}` is duplicated");
            }
        }

        let mut tensors = Vec::new();
        let mut names = HashMap::new();
        for tensor_index in 0..tensor_count {
            let name = reader.read_string(version)?;
            if names.insert(name.clone(), tensor_index).is_some() {
                candle_core::bail!("GGUF tensor `{name}` is duplicated");
            }
            let n_dims = reader.read_u32()?;
            if n_dims > MAX_TENSOR_DIMS {
                candle_core::bail!(
                    "GGUF tensor `{name}` has {n_dims} dimensions, maximum is {MAX_TENSOR_DIMS}"
                );
            }
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let dim = match version {
                    GgufVersion::V1 => u64::from(reader.read_u32()?),
                    GgufVersion::V2 | GgufVersion::V3 => reader.read_u64()?,
                };
                shape.push(usize::try_from(dim).map_err(Error::wrap)?);
            }
            shape.reverse();
            checked_elem_count(&name, &shape)?;
            let dtype = GgufDType::new(reader.read_u32()?);
            let offset = reader.read_u64()?;
            tensors.push(RawTensorInfo {
                name,
                shape,
                dtype,
                offset,
            });
        }

        let header_end = reader.position();
        Ok(Self {
            path,
            mmap,
            version,
            endian,
            metadata,
            tensors,
            header_end,
        })
    }

    fn finish(
        self,
        shard_index: usize,
        alignment: usize,
    ) -> Result<(Mmap, GgufShardInfo, Vec<GgufTensorInfo>)> {
        let tensor_data_offset = align_up(self.header_end, alignment)?;
        if !self.tensors.is_empty() && tensor_data_offset > self.mmap.len() {
            candle_core::bail!(
                "GGUF tensor data begins at byte {tensor_data_offset}, beyond file length {}",
                self.mmap.len()
            );
        }

        let mut tensors = Vec::with_capacity(self.tensors.len());
        for (index, raw) in self.tensors.iter().enumerate() {
            let offset = usize::try_from(raw.offset).map_err(Error::wrap)?;
            if !offset.is_multiple_of(alignment) {
                candle_core::bail!(
                    "GGUF tensor `{}` offset {offset} is not aligned to {alignment} bytes",
                    raw.name
                );
            }
            if index == 0 && offset != 0 {
                candle_core::bail!(
                    "first GGUF tensor `{}` starts at relative offset {offset}, expected 0",
                    raw.name
                );
            }
            let next_offset = self
                .tensors
                .get(index + 1)
                .map(|next| usize::try_from(next.offset).map_err(Error::wrap))
                .transpose()?;
            if let Some(next_offset) = next_offset {
                if next_offset <= offset {
                    candle_core::bail!(
                        "GGUF tensor `{}` offset {offset} is not before the next tensor offset {next_offset}",
                        raw.name
                    );
                }
            }

            let absolute_start = tensor_data_offset.checked_add(offset).ok_or_else(|| {
                Error::msg(format!("offset overflow for GGUF tensor `{}`", raw.name))
            })?;
            let available_end = match next_offset {
                Some(next) => tensor_data_offset.checked_add(next).ok_or_else(|| {
                    Error::msg(format!("offset overflow for GGUF tensor `{}`", raw.name))
                })?,
                None => self.mmap.len(),
            };
            if absolute_start > available_end || available_end > self.mmap.len() {
                candle_core::bail!(
                    "GGUF tensor `{}` range begins at {absolute_start} with storage ending at {available_end}, outside file length {}",
                    raw.name,
                    self.mmap.len()
                );
            }

            let exact_len = raw.dtype.tensor_byte_len(&raw.name, &raw.shape)?;
            let (data_range, storage_end) = match exact_len {
                Some(exact_len) => {
                    let exact_end = absolute_start.checked_add(exact_len).ok_or_else(|| {
                        Error::msg(format!(
                            "byte range overflow for GGUF tensor `{}`",
                            raw.name
                        ))
                    })?;
                    if exact_end > available_end {
                        candle_core::bail!(
                            "GGUF tensor `{}` needs {exact_len} bytes at byte {absolute_start}, but its storage ends at {available_end}",
                            raw.name
                        );
                    }
                    let padded_len = align_up(exact_len, alignment)?;
                    let expected_next = offset.checked_add(padded_len).ok_or_else(|| {
                        Error::msg(format!(
                            "padded size overflow for GGUF tensor `{}`",
                            raw.name
                        ))
                    })?;
                    if let Some(next_offset) = next_offset {
                        if next_offset != expected_next {
                            candle_core::bail!(
                                "GGUF tensor `{}` is followed by offset {next_offset}, expected {expected_next}",
                                raw.name
                            );
                        }
                    }
                    let storage_end = absolute_start.checked_add(padded_len).ok_or_else(|| {
                        Error::msg(format!(
                            "storage range overflow for GGUF tensor `{}`",
                            raw.name
                        ))
                    })?;
                    if storage_end > self.mmap.len() {
                        candle_core::bail!(
                            "padded storage for GGUF tensor `{}` ends at byte {storage_end}, beyond file length {}",
                            raw.name,
                            self.mmap.len()
                        );
                    }
                    (Some(absolute_start..exact_end), storage_end)
                }
                None => (None, available_end),
            };

            tensors.push(GgufTensorInfo {
                name: raw.name.clone(),
                shape: raw.shape.clone(),
                dtype: raw.dtype,
                shard_index,
                relative_offset: raw.offset,
                data_range,
                storage_range: absolute_start..storage_end,
            });
        }

        let info = GgufShardInfo {
            path: self.path,
            version: self.version,
            endian: self.endian,
            alignment,
            tensor_data_offset,
            file_len: self.mmap.len(),
        };
        Ok((self.mmap, info, tensors))
    }
}

struct SliceReader<'a> {
    data: &'a [u8],
    position: usize,
    endian: GgufEndian,
}

impl<'a> SliceReader<'a> {
    const fn new(data: &'a [u8], position: usize, endian: GgufEndian) -> Self {
        Self {
            data,
            position,
            endian,
        }
    }

    const fn position(&self) -> usize {
        self.position
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    fn read_exact(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| Error::msg("GGUF read offset overflow"))?;
        let bytes = self.data.get(self.position..end).ok_or_else(|| {
            Error::msg(format!(
                "GGUF needs {len} bytes at offset {}, only {} remain",
                self.position,
                self.remaining()
            ))
        })?;
        self.position = end;
        Ok(bytes)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_exact(2)?;
        Ok(match self.endian {
            GgufEndian::Little => LittleEndian::read_u16(bytes),
            GgufEndian::Big => BigEndian::read_u16(bytes),
        })
    }

    fn read_i16(&mut self) -> Result<i16> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_exact(4)?;
        Ok(match self.endian {
            GgufEndian::Little => LittleEndian::read_u32(bytes),
            GgufEndian::Big => BigEndian::read_u32(bytes),
        })
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_exact(8)?;
        Ok(match self.endian {
            GgufEndian::Little => LittleEndian::read_u64(bytes),
            GgufEndian::Big => BigEndian::read_u64(bytes),
        })
    }

    fn read_i64(&mut self) -> Result<i64> {
        Ok(self.read_u64()? as i64)
    }

    fn read_length(&mut self, version: GgufVersion) -> Result<u64> {
        match version {
            GgufVersion::V1 => Ok(u64::from(self.read_u32()?)),
            GgufVersion::V2 | GgufVersion::V3 => self.read_u64(),
        }
    }

    fn read_string(&mut self, version: GgufVersion) -> Result<String> {
        let len = self.read_length(version)?;
        if len > MAX_STRING_LENGTH {
            candle_core::bail!("GGUF string length {len} exceeds maximum {MAX_STRING_LENGTH}");
        }
        let len = usize::try_from(len).map_err(Error::wrap)?;
        let mut bytes = self.read_exact(len)?;
        while bytes.last() == Some(&0) {
            bytes = &bytes[..bytes.len() - 1];
        }
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }

    fn read_value(
        &mut self,
        value_type: ValueType,
        version: GgufVersion,
        depth: usize,
    ) -> Result<Value> {
        if depth > MAX_VALUE_DEPTH {
            candle_core::bail!("GGUF value nesting exceeds maximum depth {MAX_VALUE_DEPTH}");
        }
        let value = match value_type {
            ValueType::U8 => Value::U8(self.read_u8()?),
            ValueType::I8 => Value::I8(self.read_i8()?),
            ValueType::U16 => Value::U16(self.read_u16()?),
            ValueType::I16 => Value::I16(self.read_i16()?),
            ValueType::U32 => Value::U32(self.read_u32()?),
            ValueType::I32 => Value::I32(self.read_i32()?),
            ValueType::U64 => Value::U64(self.read_u64()?),
            ValueType::I64 => Value::I64(self.read_i64()?),
            ValueType::F32 => Value::F32(f32::from_bits(self.read_u32()?)),
            ValueType::F64 => Value::F64(f64::from_bits(self.read_u64()?)),
            ValueType::Bool => match self.read_u8()? {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                value => candle_core::bail!("invalid GGUF boolean value {value}"),
            },
            ValueType::String => Value::String(self.read_string(version)?),
            ValueType::Array => {
                let element_type = ValueType::from_raw(self.read_u32()?)?;
                let len = self.read_length(version)?;
                if len > MAX_ARRAY_ELEMENTS {
                    candle_core::bail!(
                        "GGUF array length {len} exceeds maximum {MAX_ARRAY_ELEMENTS}"
                    );
                }
                let minimum = u128::from(len)
                    .checked_mul(element_type.min_disk_size(version) as u128)
                    .ok_or_else(|| Error::msg("GGUF array minimum size overflow"))?;
                if minimum > self.remaining() as u128 {
                    candle_core::bail!(
                        "GGUF array of {len} values needs at least {minimum} bytes, only {} remain",
                        self.remaining()
                    );
                }
                let mut values = Vec::new();
                for _ in 0..len {
                    values.push(self.read_value(element_type, version, depth + 1)?);
                }
                Value::Array(values)
            }
        };
        Ok(value)
    }
}

#[derive(Clone, Copy)]
enum ValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl ValueType {
    fn from_raw(raw: u32) -> Result<Self> {
        let value_type = match raw {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            raw => candle_core::bail!("unknown GGUF value type {raw}"),
        };
        Ok(value_type)
    }

    const fn min_disk_size(self, version: GgufVersion) -> usize {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String => version.length_prefix_size(),
            Self::Array => 4 + version.length_prefix_size(),
        }
    }
}

fn parse_magic(data: &[u8]) -> Result<(GgufEndian, usize)> {
    let magic = data
        .get(..4)
        .ok_or_else(|| Error::msg("GGUF file is shorter than its magic"))?;
    match magic {
        b"GGUF" => Ok((GgufEndian::Little, 4)),
        b"FUGG" => Ok((GgufEndian::Big, 4)),
        _ => candle_core::bail!("invalid GGUF magic {magic:02x?}"),
    }
}

fn validate_header_minimum(
    remaining: usize,
    version: GgufVersion,
    tensor_count: u64,
    metadata_count: u64,
) -> Result<()> {
    let prefix = version.length_prefix_size() as u128;
    let metadata_minimum = u128::from(metadata_count)
        .checked_mul(prefix + 4 + 1)
        .ok_or_else(|| Error::msg("GGUF metadata minimum size overflow"))?;
    let tensor_minimum = u128::from(tensor_count)
        .checked_mul(prefix + 4 + 4 + 8)
        .ok_or_else(|| Error::msg("GGUF tensor-info minimum size overflow"))?;
    let minimum = metadata_minimum
        .checked_add(tensor_minimum)
        .ok_or_else(|| Error::msg("GGUF header minimum size overflow"))?;
    if minimum > remaining as u128 {
        candle_core::bail!(
            "GGUF header declares {tensor_count} tensors and {metadata_count} metadata entries needing at least {minimum} bytes, only {remaining} remain"
        );
    }
    Ok(())
}

fn validate_and_order_splits(shards: &mut [ParsedShard]) -> Result<()> {
    let has_split_metadata = shards.iter().any(|shard| {
        shard.metadata.contains_key(SPLIT_NO) || shard.metadata.contains_key(SPLIT_COUNT)
    });
    if !has_split_metadata {
        if shards.len() != 1 {
            candle_core::bail!(
                "{} GGUF files were supplied without split metadata",
                shards.len()
            );
        }
        return Ok(());
    }

    let mut expected_count = None;
    let mut indices = HashMap::new();
    for shard in shards.iter() {
        let index = metadata_usize(&shard.metadata, SPLIT_NO)?.ok_or_else(|| {
            Error::msg(format!(
                "GGUF shard `{}` is missing `{SPLIT_NO}`",
                shard.path.display()
            ))
        })?;
        let count = metadata_usize(&shard.metadata, SPLIT_COUNT)?.ok_or_else(|| {
            Error::msg(format!(
                "GGUF shard `{}` is missing `{SPLIT_COUNT}`",
                shard.path.display()
            ))
        })?;
        if count == 0 {
            candle_core::bail!("GGUF `{SPLIT_COUNT}` must be positive");
        }
        if index >= count {
            candle_core::bail!("GGUF split index {index} is outside split count {count}");
        }
        if let Some(expected) = expected_count {
            if count != expected {
                candle_core::bail!("GGUF shards disagree on split count: {expected} and {count}");
            }
        } else {
            expected_count = Some(count);
        }
        if indices.insert(index, shard.path.clone()).is_some() {
            candle_core::bail!("GGUF split index {index} is duplicated");
        }
    }

    let expected_count = expected_count.unwrap_or(1);
    if shards.len() != expected_count {
        candle_core::bail!(
            "{} GGUF shards were supplied, but split metadata declares {expected_count}",
            shards.len()
        );
    }
    shards.sort_by_key(|shard| metadata_usize(&shard.metadata, SPLIT_NO).ok().flatten());
    for (expected, shard) in shards.iter().enumerate() {
        let actual = metadata_usize(&shard.metadata, SPLIT_NO)?.unwrap();
        if actual != expected {
            candle_core::bail!("missing GGUF split index {expected}");
        }
    }
    Ok(())
}

fn archive_alignment(shards: &[ParsedShard]) -> Result<usize> {
    let primary =
        metadata_usize(&shards[0].metadata, GENERAL_ALIGNMENT)?.unwrap_or(DEFAULT_ALIGNMENT);
    if primary == 0 || !primary.is_power_of_two() {
        candle_core::bail!("GGUF alignment {primary} is not a nonzero power of two");
    }
    for shard in shards.iter().skip(1) {
        if let Some(alignment) = metadata_usize(&shard.metadata, GENERAL_ALIGNMENT)? {
            if alignment != primary {
                candle_core::bail!("GGUF shards disagree on alignment: {primary} and {alignment}");
            }
        }
    }
    Ok(primary)
}

fn declared_tensor_count(shards: &[ParsedShard]) -> Result<Option<usize>> {
    let mut declared = None;
    for shard in shards {
        let Some(count) = metadata_usize(&shard.metadata, SPLIT_TENSORS_COUNT)? else {
            continue;
        };
        if let Some(expected) = declared {
            if count != expected {
                candle_core::bail!(
                    "GGUF shards disagree on declared tensor count: {expected} and {count}"
                );
            }
        } else {
            declared = Some(count);
        }
    }
    if shards.len() > 1 && declared.is_none() {
        candle_core::bail!("split GGUF is missing `{SPLIT_TENSORS_COUNT}`");
    }
    Ok(declared)
}

fn merge_metadata(
    merged: &mut HashMap<String, Value>,
    shard: HashMap<String, Value>,
    shard_index: usize,
) -> Result<()> {
    for (key, value) in shard {
        if key == SPLIT_NO && shard_index != 0 {
            continue;
        }
        match merged.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(value);
            }
            Entry::Occupied(entry) => {
                if !values_equal(entry.get(), &value) {
                    candle_core::bail!("GGUF shards disagree on metadata key `{}`", entry.key());
                }
            }
        }
    }
    Ok(())
}

fn values_equal(lhs: &Value, rhs: &Value) -> bool {
    match (lhs, rhs) {
        (Value::U8(lhs), Value::U8(rhs)) => lhs == rhs,
        (Value::I8(lhs), Value::I8(rhs)) => lhs == rhs,
        (Value::U16(lhs), Value::U16(rhs)) => lhs == rhs,
        (Value::I16(lhs), Value::I16(rhs)) => lhs == rhs,
        (Value::U32(lhs), Value::U32(rhs)) => lhs == rhs,
        (Value::I32(lhs), Value::I32(rhs)) => lhs == rhs,
        (Value::U64(lhs), Value::U64(rhs)) => lhs == rhs,
        (Value::I64(lhs), Value::I64(rhs)) => lhs == rhs,
        (Value::F32(lhs), Value::F32(rhs)) => lhs.to_bits() == rhs.to_bits(),
        (Value::F64(lhs), Value::F64(rhs)) => lhs.to_bits() == rhs.to_bits(),
        (Value::Bool(lhs), Value::Bool(rhs)) => lhs == rhs,
        (Value::String(lhs), Value::String(rhs)) => lhs == rhs,
        (Value::Array(lhs), Value::Array(rhs)) => {
            lhs.len() == rhs.len()
                && lhs
                    .iter()
                    .zip(rhs.iter())
                    .all(|(lhs, rhs)| values_equal(lhs, rhs))
        }
        _ => false,
    }
}

fn validate_component_type(component: &GgufArchive) -> Result<()> {
    match component.metadata_value(GENERAL_TYPE) {
        Some(Value::String(value)) if value == MMPROJ_TYPE => Ok(()),
        Some(Value::String(value)) => {
            candle_core::bail!(
                "GGUF component `{GENERAL_TYPE}` must be `{MMPROJ_TYPE}`, got `{value}`"
            )
        }
        Some(_) => candle_core::bail!("GGUF component `{GENERAL_TYPE}` must be a string"),
        None => candle_core::bail!("GGUF component is missing `{GENERAL_TYPE}`"),
    }
}

fn metadata_usize(metadata: &HashMap<String, Value>, key: &str) -> Result<Option<usize>> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    let value = match value {
        Value::U8(value) => u64::from(*value),
        Value::U16(value) => u64::from(*value),
        Value::U32(value) => u64::from(*value),
        Value::U64(value) => *value,
        Value::I8(value) if *value >= 0 => *value as u64,
        Value::I16(value) if *value >= 0 => *value as u64,
        Value::I32(value) if *value >= 0 => *value as u64,
        Value::I64(value) if *value >= 0 => *value as u64,
        _ => candle_core::bail!("GGUF metadata `{key}` must be a nonnegative integer"),
    };
    Ok(Some(usize::try_from(value).map_err(Error::wrap)?))
}

fn checked_elem_count(name: &str, shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(*dimension)
            .ok_or_else(|| Error::msg(format!("element count overflow for GGUF tensor `{name}`")))
    })
}

fn align_up(value: usize, alignment: usize) -> Result<usize> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| Error::msg("GGUF alignment overflow"))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Write, sync::Arc};

    use byteorder::{LittleEndian, WriteBytesExt};
    use candle_core::quantized::GgmlDType;
    use tempfile::NamedTempFile;

    use super::*;

    struct TestTensor {
        name: &'static str,
        shape: Vec<u64>,
        dtype: u32,
        data: Vec<u8>,
        offset: Option<u64>,
    }

    fn write_string(writer: &mut impl Write, value: &str) {
        writer
            .write_u64::<LittleEndian>(value.len() as u64)
            .unwrap();
        writer.write_all(value.as_bytes()).unwrap();
    }

    fn value_type(value: &Value) -> u32 {
        match value {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::String(_) => 8,
            Value::Array(_) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }

    fn write_value(writer: &mut impl Write, value: &Value) {
        match value {
            Value::U8(value) => writer.write_u8(*value).unwrap(),
            Value::I8(value) => writer.write_i8(*value).unwrap(),
            Value::U16(value) => writer.write_u16::<LittleEndian>(*value).unwrap(),
            Value::I16(value) => writer.write_i16::<LittleEndian>(*value).unwrap(),
            Value::U32(value) => writer.write_u32::<LittleEndian>(*value).unwrap(),
            Value::I32(value) => writer.write_i32::<LittleEndian>(*value).unwrap(),
            Value::U64(value) => writer.write_u64::<LittleEndian>(*value).unwrap(),
            Value::I64(value) => writer.write_i64::<LittleEndian>(*value).unwrap(),
            Value::F32(value) => writer.write_f32::<LittleEndian>(*value).unwrap(),
            Value::F64(value) => writer.write_f64::<LittleEndian>(*value).unwrap(),
            Value::Bool(value) => writer.write_u8(u8::from(*value)).unwrap(),
            Value::String(value) => write_string(writer, value),
            Value::Array(values) => {
                let element_type = values.first().map(value_type).unwrap_or(4);
                writer.write_u32::<LittleEndian>(element_type).unwrap();
                writer
                    .write_u64::<LittleEndian>(values.len() as u64)
                    .unwrap();
                for value in values {
                    assert_eq!(value_type(value), element_type);
                    write_value(writer, value);
                }
            }
        }
    }

    fn write_test_gguf(
        metadata: &[(&str, Value)],
        tensors: &[TestTensor],
        alignment: usize,
    ) -> NamedTempFile {
        let mut offsets = Vec::new();
        let mut next_offset = 0usize;
        for tensor in tensors {
            let offset = tensor.offset.unwrap_or(next_offset as u64);
            offsets.push(offset);
            next_offset = usize::try_from(offset).unwrap() + tensor.data.len();
            next_offset = next_offset.div_ceil(alignment) * alignment;
        }

        let mut bytes = Vec::new();
        bytes.write_all(b"GGUF").unwrap();
        bytes.write_u32::<LittleEndian>(3).unwrap();
        bytes
            .write_u64::<LittleEndian>(tensors.len() as u64)
            .unwrap();
        bytes
            .write_u64::<LittleEndian>(metadata.len() as u64)
            .unwrap();
        for (key, value) in metadata {
            write_string(&mut bytes, key);
            bytes.write_u32::<LittleEndian>(value_type(value)).unwrap();
            write_value(&mut bytes, value);
        }
        for (tensor, offset) in tensors.iter().zip(offsets.iter()) {
            write_string(&mut bytes, tensor.name);
            bytes
                .write_u32::<LittleEndian>(tensor.shape.len() as u32)
                .unwrap();
            for dimension in tensor.shape.iter().rev() {
                bytes.write_u64::<LittleEndian>(*dimension).unwrap();
            }
            bytes.write_u32::<LittleEndian>(tensor.dtype).unwrap();
            bytes.write_u64::<LittleEndian>(*offset).unwrap();
        }
        let data_offset = bytes.len().div_ceil(alignment) * alignment;
        bytes.resize(data_offset, 0);
        for (tensor, offset) in tensors.iter().zip(offsets) {
            let start = data_offset + usize::try_from(offset).unwrap();
            bytes.resize(start, 0);
            bytes.extend_from_slice(&tensor.data);
            let padded = bytes.len().div_ceil(alignment) * alignment;
            bytes.resize(padded, 0);
        }

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        file
    }

    fn base_metadata() -> Vec<(&'static str, Value)> {
        vec![
            ("general.architecture", Value::String("qwen3".to_string())),
            (GENERAL_ALIGNMENT, Value::U32(DEFAULT_ALIGNMENT as u32)),
            (
                "tokenizer.ggml.tokens",
                Value::Array(vec![
                    Value::String("hello".to_string()),
                    Value::String("world".to_string()),
                ]),
            ),
        ]
    }

    #[test]
    fn supported_dtype_catalog_matches_candle() {
        for (raw, dtype) in [
            (0, GgmlDType::F32),
            (1, GgmlDType::F16),
            (2, GgmlDType::Q4_0),
            (3, GgmlDType::Q4_1),
            (6, GgmlDType::Q5_0),
            (7, GgmlDType::Q5_1),
            (8, GgmlDType::Q8_0),
            (9, GgmlDType::Q8_1),
            (10, GgmlDType::Q2K),
            (11, GgmlDType::Q3K),
            (12, GgmlDType::Q4K),
            (13, GgmlDType::Q5K),
            (14, GgmlDType::Q6K),
            (15, GgmlDType::Q8K),
            (30, GgmlDType::BF16),
        ] {
            let catalog = GgufDType::new(raw);
            assert_eq!(catalog.block_size(), Some(dtype.block_size()), "{dtype:?}");
            assert_eq!(catalog.type_size(), Some(dtype.type_size()), "{dtype:?}");
        }
    }

    #[test]
    fn stages_unaligned_quantized_data() -> Result<()> {
        let mut bytes = vec![0u8; GgmlDType::Q4_0.type_size() + 1];
        let aligned = ggml_dtype_alignment(GgmlDType::Q4_0);
        let start = usize::from(bytes.as_ptr().align_offset(aligned) == 0);
        let data = &mut bytes[start..start + GgmlDType::Q4_0.type_size()];
        assert_ne!(data.as_ptr().align_offset(aligned), 0);

        let tensor = qtensor_from_gguf_data(GgmlDType::Q4_0, data, vec![32], &Device::Cpu)?;
        assert_eq!(tensor.dtype(), GgmlDType::Q4_0);
        assert_eq!(tensor.dequantize(&Device::Cpu)?.dims(), [32]);
        Ok(())
    }

    #[test]
    fn catalogs_tensors_and_loads_supported_qtensor() -> Result<()> {
        let q4_data = vec![7; 36];
        let q8_1_data = vec![3; 36];
        let q8_k_data = vec![5; 292];
        let f32_data = (0..8)
            .flat_map(|value| (value as f32).to_le_bytes())
            .collect::<Vec<_>>();
        let file = write_test_gguf(
            &base_metadata(),
            &[
                TestTensor {
                    name: "blk.0.attn_q.weight",
                    shape: vec![2, 32],
                    dtype: 2,
                    data: q4_data.clone(),
                    offset: None,
                },
                TestTensor {
                    name: "blk.0.attn_k.weight",
                    shape: vec![1, 32],
                    dtype: 9,
                    data: q8_1_data.clone(),
                    offset: None,
                },
                TestTensor {
                    name: "blk.0.attn_v.weight",
                    shape: vec![1, 256],
                    dtype: 15,
                    data: q8_k_data.clone(),
                    offset: None,
                },
                TestTensor {
                    name: "output_norm.weight",
                    shape: vec![8],
                    dtype: 0,
                    data: f32_data.clone(),
                    offset: None,
                },
            ],
            DEFAULT_ALIGNMENT,
        );

        let archive = GgufArchive::open_file(file.path())?;
        assert_eq!(archive.shards().len(), 1);
        assert_eq!(
            archive
                .metadata_value("general.architecture")
                .unwrap()
                .to_string()?,
            "qwen3"
        );
        let q4 = archive.tensor_info("blk.0.attn_q.weight")?;
        assert_eq!(q4.shape(), [2, 32]);
        assert_eq!(q4.dtype().raw(), 2);
        assert_eq!(q4.byte_len(), Some(36));
        assert_eq!(archive.tensor_data(q4.name())?.bytes(), q4_data);
        assert_eq!(
            q4.data_range().unwrap().start,
            archive.shards()[0].tensor_data_offset()
        );
        let q8_1 = archive.tensor_info("blk.0.attn_k.weight")?;
        assert_eq!(q8_1.byte_len(), Some(36));
        assert_eq!(
            archive.load_qtensor(q8_1.name(), &Device::Cpu)?.dtype(),
            GgmlDType::Q8_1
        );
        assert_eq!(archive.tensor_data(q8_1.name())?.bytes(), q8_1_data);
        let q8_k = archive.tensor_info("blk.0.attn_v.weight")?;
        assert_eq!(q8_k.byte_len(), Some(292));
        assert_eq!(
            archive.load_qtensor(q8_k.name(), &Device::Cpu)?.dtype(),
            GgmlDType::Q8K
        );
        assert_eq!(archive.tensor_data(q8_k.name())?.bytes(), q8_k_data);

        let dense = archive.load_qtensor("output_norm.weight", &Device::Cpu)?;
        assert_eq!(dense.dtype(), GgmlDType::F32);
        assert_eq!(dense.shape().dims(), [8]);
        assert_eq!(archive.tensor_data("output_norm.weight")?.bytes(), f32_data);
        Ok(())
    }

    #[test]
    fn retains_iq_and_future_dtype_codes() -> Result<()> {
        let file = write_test_gguf(
            &base_metadata(),
            &[
                TestTensor {
                    name: "iq.weight",
                    shape: vec![256],
                    dtype: 16,
                    data: vec![1; 66],
                    offset: None,
                },
                TestTensor {
                    name: "future.weight",
                    shape: vec![32],
                    dtype: 999,
                    data: vec![2; 17],
                    offset: None,
                },
            ],
            DEFAULT_ALIGNMENT,
        );
        let archive = GgufArchive::open_file(file.path())?;

        let iq = archive.tensor_info("iq.weight")?;
        assert_eq!(iq.dtype().raw(), 16);
        assert_eq!(iq.byte_len(), Some(66));
        assert_eq!(archive.tensor_data(iq.name())?.bytes().len(), 66);
        assert!(archive.load_qtensor(iq.name(), &Device::Cpu).is_err());

        let future = archive.tensor_info("future.weight")?;
        assert_eq!(future.dtype().raw(), 999);
        assert_eq!(future.byte_len(), None);
        assert!(archive.tensor_data(future.name()).is_err());
        assert_eq!(
            archive.tensor_storage_data(future.name())?.bytes().len(),
            DEFAULT_ALIGNMENT
        );
        Ok(())
    }

    #[test]
    fn orders_and_validates_split_shards() -> Result<()> {
        let first = write_test_gguf(
            &[
                ("general.architecture", Value::String("qwen3".to_string())),
                (GENERAL_ALIGNMENT, Value::U32(DEFAULT_ALIGNMENT as u32)),
                (SPLIT_NO, Value::U16(0)),
                (SPLIT_COUNT, Value::U16(2)),
                (SPLIT_TENSORS_COUNT, Value::I32(2)),
            ],
            &[TestTensor {
                name: "first.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![0; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let second = write_test_gguf(
            &[
                (SPLIT_NO, Value::U16(1)),
                (SPLIT_COUNT, Value::U16(2)),
                (SPLIT_TENSORS_COUNT, Value::I32(2)),
            ],
            &[TestTensor {
                name: "second.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![1; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );

        let archive = GgufArchive::open([second.path(), first.path()])?;
        assert_eq!(archive.shards()[0].path(), first.path());
        assert_eq!(archive.shards()[1].path(), second.path());
        assert!(archive.contains_tensor("first.weight"));
        assert!(archive.contains_tensor("second.weight"));
        assert_eq!(
            archive
                .metadata_value("general.architecture")
                .unwrap()
                .to_string()?,
            "qwen3"
        );
        Ok(())
    }

    #[test]
    fn merges_split_mmproj_component() -> Result<()> {
        let main_file = write_test_gguf(
            &[
                ("general.architecture", Value::String("qwen3".to_string())),
                (GENERAL_TYPE, Value::String("model".to_string())),
                ("general.name", Value::String("text".to_string())),
                ("shared.value", Value::U32(7)),
            ],
            &[TestTensor {
                name: "token_embd.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![1; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let component_first = write_test_gguf(
            &[
                ("general.architecture", Value::String("clip".to_string())),
                (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
                ("general.name", Value::String("vision".to_string())),
                (SPLIT_NO, Value::U16(0)),
                (SPLIT_COUNT, Value::U16(2)),
                (SPLIT_TENSORS_COUNT, Value::U32(2)),
                (
                    "clip.projector_type",
                    Value::String("qwen3vl_merger".to_string()),
                ),
                ("shared.value", Value::U32(7)),
            ],
            &[TestTensor {
                name: "v.patch_embd.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![2; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let component_second = write_test_gguf(
            &[
                (SPLIT_NO, Value::U16(1)),
                (SPLIT_COUNT, Value::U16(2)),
                (SPLIT_TENSORS_COUNT, Value::U32(2)),
            ],
            &[TestTensor {
                name: "mm.0.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![3; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );

        let mut archive = GgufArchive::open_file(main_file.path())?;
        let component = GgufArchive::open([component_second.path(), component_first.path()])?;
        archive.merge_component(component)?;

        assert_eq!(archive.shards().len(), 3);
        assert_eq!(archive.shards()[0].path(), main_file.path());
        assert_eq!(archive.shards()[1].path(), component_first.path());
        assert_eq!(archive.shards()[2].path(), component_second.path());
        assert_eq!(archive.tensor_info("token_embd.weight")?.shard_index(), 0);
        assert_eq!(archive.tensor_info("v.patch_embd.weight")?.shard_index(), 1);
        assert_eq!(archive.tensor_info("mm.0.weight")?.shard_index(), 2);
        assert_eq!(
            archive.tensor_data("v.patch_embd.weight")?.bytes(),
            vec![2; 32]
        );
        assert_eq!(archive.tensor_data("mm.0.weight")?.bytes(), vec![3; 32]);
        assert_eq!(
            archive
                .metadata_value("general.architecture")
                .unwrap()
                .to_string()?,
            "qwen3"
        );
        assert_eq!(
            archive.metadata_value(GENERAL_TYPE).unwrap().to_string()?,
            "model"
        );
        assert_eq!(
            archive
                .metadata_value("general.name")
                .unwrap()
                .to_string()?,
            "text"
        );
        assert_eq!(
            archive
                .metadata_value("clip.projector_type")
                .unwrap()
                .to_string()?,
            "qwen3vl_merger"
        );
        assert!(archive.metadata_value(SPLIT_COUNT).is_none());
        Ok(())
    }

    #[test]
    fn merges_multiple_mmproj_components() -> Result<()> {
        let main_file = write_test_gguf(&base_metadata(), &[], DEFAULT_ALIGNMENT);
        let vision_file = write_test_gguf(
            &[
                ("general.architecture", Value::String("clip".to_string())),
                (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
                ("clip.has_vision_encoder", Value::Bool(true)),
            ],
            &[TestTensor {
                name: "v.patch_embd.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![1; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let audio_file = write_test_gguf(
            &[
                ("general.architecture", Value::String("clip".to_string())),
                (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
                ("clip.has_audio_encoder", Value::Bool(true)),
            ],
            &[TestTensor {
                name: "a.position_embd.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![2; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );

        let mut archive = GgufArchive::open_file(main_file.path())?;
        archive.merge_components([
            GgufArchive::open_file(vision_file.path())?,
            GgufArchive::open_file(audio_file.path())?,
        ])?;

        assert_eq!(archive.tensor_info("v.patch_embd.weight")?.shard_index(), 1);
        assert_eq!(
            archive.tensor_info("a.position_embd.weight")?.shard_index(),
            2
        );
        assert!(matches!(
            archive.metadata_value("clip.has_vision_encoder"),
            Some(Value::Bool(true))
        ));
        assert!(matches!(
            archive.metadata_value("clip.has_audio_encoder"),
            Some(Value::Bool(true))
        ));
        Ok(())
    }

    #[test]
    fn opens_split_and_independent_mmproj_components() -> Result<()> {
        let main_file = write_test_gguf(&base_metadata(), &[], DEFAULT_ALIGNMENT);
        let vision_first = write_test_gguf(
            &[
                ("general.architecture", Value::String("clip".to_string())),
                (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
                (SPLIT_NO, Value::U16(0)),
                (SPLIT_COUNT, Value::U16(2)),
                (SPLIT_TENSORS_COUNT, Value::U32(2)),
            ],
            &[TestTensor {
                name: "v.patch_embd.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![1; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let vision_second = write_test_gguf(
            &[
                (SPLIT_NO, Value::U16(1)),
                (SPLIT_COUNT, Value::U16(2)),
                (SPLIT_TENSORS_COUNT, Value::U32(2)),
            ],
            &[TestTensor {
                name: "mm.0.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![2; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let audio = write_test_gguf(
            &[
                ("general.architecture", Value::String("clip".to_string())),
                (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
            ],
            &[TestTensor {
                name: "a.position_embd.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![3; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );

        let components = GgufArchive::open_components([
            vision_second.path(),
            vision_first.path(),
            audio.path(),
        ])?;

        assert_eq!(components.len(), 2);
        assert_eq!(components[0].shards().len(), 2);
        assert!(components[0].contains_tensor("v.patch_embd.weight"));
        assert!(components[0].contains_tensor("mm.0.weight"));
        assert_eq!(components[1].shards().len(), 1);
        assert!(components[1].contains_tensor("a.position_embd.weight"));

        let mut archive = GgufArchive::open_file(main_file.path())?;
        archive.merge_components(components)?;
        assert_eq!(archive.shards().len(), 4);
        assert_eq!(archive.tensor_info("v.patch_embd.weight")?.shard_index(), 1);
        assert_eq!(archive.tensor_info("mm.0.weight")?.shard_index(), 2);
        assert_eq!(
            archive.tensor_info("a.position_embd.weight")?.shard_index(),
            3
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_or_duplicate_mmproj_components() {
        let main_file = write_test_gguf(
            &base_metadata(),
            &[TestTensor {
                name: "duplicate.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![0; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let wrong_type_file = write_test_gguf(
            &[(GENERAL_TYPE, Value::String("model".to_string()))],
            &[],
            DEFAULT_ALIGNMENT,
        );
        let duplicate_file = write_test_gguf(
            &[(GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string()))],
            &[TestTensor {
                name: "duplicate.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![1; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );

        let mut archive = GgufArchive::open_file(main_file.path()).unwrap();
        let error = archive
            .merge_component(GgufArchive::open_file(wrong_type_file.path()).unwrap())
            .unwrap_err();
        assert!(error.to_string().contains("must be `mmproj`"));
        let error = archive
            .merge_component(GgufArchive::open_file(duplicate_file.path()).unwrap())
            .unwrap_err();
        assert!(error.to_string().contains("duplicated across components"));
        assert_eq!(archive.shards().len(), 1);
    }

    #[test]
    fn rejects_conflicting_component_metadata_atomically() {
        let main_file = write_test_gguf(&base_metadata(), &[], DEFAULT_ALIGNMENT);
        let first_file = write_test_gguf(
            &[
                (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
                (
                    "clip.projector_type",
                    Value::String("qwen3vl_merger".to_string()),
                ),
            ],
            &[TestTensor {
                name: "first.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![1; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let second_file = write_test_gguf(
            &[
                (GENERAL_TYPE, Value::String(MMPROJ_TYPE.to_string())),
                ("clip.projector_type", Value::String("gemma3".to_string())),
            ],
            &[TestTensor {
                name: "second.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![2; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );

        let mut archive = GgufArchive::open_file(main_file.path()).unwrap();
        let error = archive
            .merge_components([
                GgufArchive::open_file(first_file.path()).unwrap(),
                GgufArchive::open_file(second_file.path()).unwrap(),
            ])
            .unwrap_err();
        assert!(error.to_string().contains("component metadata key"));
        assert_eq!(archive.shards().len(), 1);
        assert!(!archive.contains_tensor("first.weight"));
        assert!(!archive.contains_tensor("second.weight"));
        assert!(archive.metadata_value("clip.projector_type").is_none());
    }

    #[test]
    fn rejects_duplicate_split_tensor_names() {
        let first = write_test_gguf(
            &[
                (SPLIT_NO, Value::U16(0)),
                (SPLIT_COUNT, Value::U16(2)),
                (SPLIT_TENSORS_COUNT, Value::I32(2)),
            ],
            &[TestTensor {
                name: "duplicate.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![0; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let second = write_test_gguf(
            &[
                (SPLIT_NO, Value::U16(1)),
                (SPLIT_COUNT, Value::U16(2)),
                (SPLIT_TENSORS_COUNT, Value::I32(2)),
            ],
            &[TestTensor {
                name: "duplicate.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![1; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let error = GgufArchive::open([first.path(), second.path()]).unwrap_err();
        assert!(error.to_string().contains("duplicated across shards"));
    }

    #[test]
    fn rejects_incomplete_splits_and_wrong_total_count() {
        let first = write_test_gguf(
            &[
                (SPLIT_NO, Value::U16(0)),
                (SPLIT_COUNT, Value::U16(2)),
                (SPLIT_TENSORS_COUNT, Value::I32(1)),
            ],
            &[TestTensor {
                name: "first.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![0; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let error = GgufArchive::open_file(first.path()).unwrap_err();
        assert!(error
            .to_string()
            .contains("were supplied, but split metadata declares 2"));

        let first = write_test_gguf(
            &[
                (SPLIT_NO, Value::U16(0)),
                (SPLIT_COUNT, Value::U16(2)),
                (SPLIT_TENSORS_COUNT, Value::I32(3)),
            ],
            &[TestTensor {
                name: "first.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![0; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let second = write_test_gguf(
            &[
                (SPLIT_NO, Value::U16(1)),
                (SPLIT_COUNT, Value::U16(2)),
                (SPLIT_TENSORS_COUNT, Value::I32(3)),
            ],
            &[TestTensor {
                name: "second.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![0; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let error = GgufArchive::open([first.path(), second.path()]).unwrap_err();
        assert!(error.to_string().contains("declares 3 tensors"));
    }

    #[test]
    fn rejects_truncated_storage_and_unaligned_offsets() {
        let file = write_test_gguf(
            &base_metadata(),
            &[TestTensor {
                name: "truncated.weight",
                shape: vec![8],
                dtype: 0,
                data: vec![0; 32],
                offset: None,
            }],
            DEFAULT_ALIGNMENT,
        );
        let shortened_len = fs::metadata(file.path()).unwrap().len() - 1;
        file.as_file().set_len(shortened_len).unwrap();
        let error = GgufArchive::open_file(file.path()).unwrap_err();
        assert!(error.to_string().contains("storage ends"));

        let file = write_test_gguf(
            &base_metadata(),
            &[
                TestTensor {
                    name: "aligned.weight",
                    shape: vec![32],
                    dtype: 999,
                    data: vec![0; 17],
                    offset: Some(0),
                },
                TestTensor {
                    name: "unaligned.weight",
                    shape: vec![8],
                    dtype: 0,
                    data: vec![0; 32],
                    offset: Some(33),
                },
            ],
            DEFAULT_ALIGNMENT,
        );
        let error = GgufArchive::open_file(file.path()).unwrap_err();
        assert!(error.to_string().contains("not aligned"));
    }

    #[test]
    fn archive_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GgufArchive>();

        let file = write_test_gguf(&base_metadata(), &[], DEFAULT_ALIGNMENT);
        let archive = Arc::new(GgufArchive::open_file(file.path()).unwrap());
        let copy = archive.clone();
        std::thread::spawn(move || {
            assert_eq!(copy.shards().len(), 1);
        })
        .join()
        .unwrap();
    }
}
