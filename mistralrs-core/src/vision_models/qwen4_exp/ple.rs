#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use mistralrs_quant::{
    safetensors::MmapedSafetensors, GgufArchive, QuantMethod, ReplicatedLayer, ShardedVarBuilder,
};

use super::config::{PleConfig, TextConfig};
use super::hyper::grouped_rms_norm;

const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;
const LAYER_SEED_PRIME: u64 = 10007;
const GATE_MAGNITUDE_FLOOR: f64 = 1e-6;
const TABLE_SHARD_PREFIX: &str = "shard_";
const GGUF_TABLE_NAME: &str = "per_layer_token_embd.weight";
const GGML_TYPE_IQ4_NL: u32 = 20;
const IQ4_NL_BLOCK: usize = 32;
const IQ4_NL_BLOCK_BYTES: usize = 18;
const KVALUES_IQ4_NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];
#[cfg(feature = "cuda")]
const RESIDENT_GROUP: usize = 32;
#[cfg(feature = "cuda")]
const RESIDENT_RESERVE_BYTES: usize = 2 << 30;
#[cfg(feature = "cuda")]
const RESIDENT_BITS: [usize; 2] = [8, 4];
#[cfg(feature = "cuda")]
const PLANNED_RESIDENT_BITS: usize = 4;
#[cfg(feature = "cuda")]
const RESIDENT_COPY_ROWS: usize = 1 << 20;
#[cfg(feature = "cuda")]
const HOST_PREFAULT_STRIDE_BYTES: usize = 4096;

#[cfg(feature = "cuda")]
fn resident_bytes(rows: usize, head_dim: usize, bits: usize) -> usize {
    rows * (head_dim * bits / 8 + head_dim / RESIDENT_GROUP * 2)
}

/// Device bytes of the resident n-gram table at the precision the device map plans for.
#[cfg(feature = "cuda")]
pub(crate) fn planned_resident_table_bytes(cfg: &TextConfig, ple: &PleConfig) -> usize {
    let rows = PleHash::new(cfg, ple, 0).vocab_sizes.iter().sum::<u64>() as usize;
    resident_bytes(rows, ple.head_dim(), PLANNED_RESIDENT_BITS)
}

fn splitmix64(value: u64) -> u64 {
    let mut value = value.wrapping_add(SPLITMIX_GAMMA);
    value = (value ^ (value >> 30)).wrapping_mul(SPLITMIX_M1);
    value = (value ^ (value >> 27)).wrapping_mul(SPLITMIX_M2);
    value ^ (value >> 31)
}

fn is_prime(value: u64) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3u64;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

fn nth_prime_after(start: u64, count: usize) -> u64 {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

/// Hash constants of one PLE module; matches `Qwen4ExpTextNGramEmbedding`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PleHash {
    pub multipliers: Vec<u64>,
    pub vocab_sizes: Vec<u64>,
    pub offsets: Vec<u64>,
    pub heads_per_ngram: usize,
    pub eos: u32,
}

impl PleHash {
    pub(super) fn new(cfg: &TextConfig, ple: &PleConfig, ple_layer_index: u64) -> Self {
        if let Some(hash) = &cfg.ple_hash {
            return Self {
                multipliers: hash.layer_multipliers.clone(),
                vocab_sizes: hash.head_vocab_sizes.clone(),
                offsets: hash.head_offsets.clone(),
                heads_per_ngram: ple.heads_per_ngram,
                eos: ple.eos_token_id,
            };
        }
        let max_long = (1u64 << 63) - 1;
        let multiplier_max = max_long / (cfg.vocab_size as u64).max(1);
        let half_bound = (multiplier_max / 2).max(1);
        let base_seed = cfg
            .seed
            .wrapping_add(LAYER_SEED_PRIME.wrapping_mul(ple_layer_index));
        let multipliers = (0..ple.ngram_size as u64)
            .map(|index| {
                let value = base_seed.wrapping_add(SPLITMIX_GAMMA.wrapping_mul(index + 1));
                2 * (splitmix64(value) % half_bound) + 1
            })
            .collect();
        let num_heads = ple.num_heads();
        let mut vocab_sizes = Vec::with_capacity(num_heads);
        let mut offsets = Vec::with_capacity(num_heads);
        let mut total = 0u64;
        for head in 0..num_heads {
            let global = ple_layer_index as usize * num_heads + head;
            let size = nth_prime_after(cfg.ngram_vocab_size_base - 1, global + 1);
            vocab_sizes.push(size);
            offsets.push(total);
            total += size;
        }
        Self {
            multipliers,
            vocab_sizes,
            offsets,
            heads_per_ngram: ple.heads_per_ngram,
            eos: ple.eos_token_id,
        }
    }

    pub(super) fn num_heads(&self) -> usize {
        self.vocab_sizes.len()
    }

    /// Table rows of `tokens[local]`; `history` precedes `tokens`, oldest first, `None` for no token.
    pub(super) fn rows(
        &self,
        tokens: &[u32],
        local: usize,
        history: &[Option<u32>],
        out: &mut [u64],
    ) {
        let ngram = self.multipliers.len();
        let mut ctx = vec![self.eos as u64; ngram];
        ctx[0] = tokens[local] as u64;
        let mut cut = false;
        for (s, slot) in ctx.iter_mut().enumerate().skip(1) {
            let tok = if local >= s {
                Some(tokens[local - s])
            } else {
                let back = s - local;
                history.len().checked_sub(back).and_then(|idx| history[idx])
            };
            cut = cut || tok.is_none_or(|tok| tok == self.eos);
            *slot = if cut {
                self.eos as u64
            } else {
                tok.unwrap() as u64
            };
        }
        for n in 2..=ngram {
            let mixed = ctx[..n]
                .iter()
                .zip(&self.multipliers)
                .fold(0u64, |mixed, (tok, mult)| mixed ^ tok.wrapping_mul(*mult));
            let base = (n - 2) * self.heads_per_ngram;
            for g in 0..self.heads_per_ngram {
                let h = base + g;
                out[h] = mixed % self.vocab_sizes[h] + self.offsets[h];
            }
        }
    }

    #[cfg(feature = "cuda")]
    fn ffi(&self) -> Result<crate::cuda::qwen4_exp::Q4PleHashParams> {
        use crate::cuda::qwen4_exp::{PLE_MAX_HEADS, PLE_MAX_NGRAM};
        if self.multipliers.len() > PLE_MAX_NGRAM || self.num_heads() > PLE_MAX_HEADS {
            candle_core::bail!("PLE n-gram configuration exceeds the CUDA kernel limits");
        }
        let mut params = crate::cuda::qwen4_exp::Q4PleHashParams {
            multipliers: [0; PLE_MAX_NGRAM],
            vocab_sizes: [1; PLE_MAX_HEADS],
            offsets: [0; PLE_MAX_HEADS],
            ngram_size: self.multipliers.len() as i32,
            heads_per_ngram: self.heads_per_ngram as i32,
            num_heads: self.num_heads() as i32,
            eos: self.eos,
        };
        params.multipliers[..self.multipliers.len()].copy_from_slice(&self.multipliers);
        params.vocab_sizes[..self.num_heads()].copy_from_slice(&self.vocab_sizes);
        params.offsets[..self.num_heads()].copy_from_slice(&self.offsets);
        Ok(params)
    }
}

/// Where the ~100 GB hashed n-gram table lives; it is never loaded as a weight tensor.
enum TableSource {
    /// bf16/f16 row shards in the safetensors memory map.
    Shards {
        raw: Arc<MmapedSafetensors>,
        names: Vec<String>,
        starts: Vec<usize>,
        dtype: DType,
    },
    /// One tensor in the GGUF memory map; `load_gguf` admits only IQ4_NL, which llama.cpp emits.
    Gguf {
        archive: Arc<GgufArchive>,
        #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
        rows: usize,
    },
    #[cfg(feature = "cuda")]
    Packed {
        map: memmap2::Mmap,
        rows: usize,
        data_bytes: usize,
    },
}

#[cfg(feature = "cuda")]
enum DeviceTableIndex {
    Shards { pointers: Tensor, starts: Tensor },
    Packed,
}

struct PleTable {
    source: TableSource,
    head_dim: usize,
    #[cfg(feature = "cuda")]
    device_index: Option<DeviceTableIndex>,
    #[cfg(feature = "cuda")]
    resident: Option<ResidentTable>,
}

/// Device copy of the table in a gather-kernel format so decode never faults on host pages.
#[cfg(feature = "cuda")]
struct ResidentTable {
    data: Tensor,
    scales: Option<Tensor>,
    format: crate::cuda::qwen4_exp::PleTableFormat,
}

fn iq4_nl_row_bytes(head_dim: usize) -> usize {
    head_dim / IQ4_NL_BLOCK * IQ4_NL_BLOCK_BYTES
}

/// ggml `dequantize_row_iq4_nl` for one row.
fn dequantize_iq4_nl_row(src: &[u8], out: &mut [f32]) {
    let (blocks, _) = src.as_chunks::<IQ4_NL_BLOCK_BYTES>();
    let (dsts, _) = out.as_chunks_mut::<IQ4_NL_BLOCK>();
    for (block, dst) in blocks.iter().zip(dsts) {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let (lo, hi) = dst.split_at_mut(IQ4_NL_BLOCK / 2);
        for (j, &byte) in block[2..].iter().enumerate() {
            lo[j] = d * f32::from(KVALUES_IQ4_NL[usize::from(byte & 0xf)]);
            hi[j] = d * f32::from(KVALUES_IQ4_NL[usize::from(byte >> 4)]);
        }
    }
}

impl PleTable {
    #[cfg(feature = "cuda")]
    fn reclaimable_weight_cache_bytes(&self) -> Option<usize> {
        if self.resident.is_some() || !matches!(self.device_index, Some(DeviceTableIndex::Packed)) {
            return None;
        }
        let bytes = match &self.source {
            TableSource::Gguf { rows, .. } => rows * iq4_nl_row_bytes(self.head_dim),
            TableSource::Packed { map, .. } => map.len(),
            TableSource::Shards { .. } => return None,
        };
        Some(bytes + RESIDENT_RESERVE_BYTES)
    }

    fn load(vb: &ShardedVarBuilder, head_dim: usize, device: &Device) -> Result<Self> {
        if let Some(archive) = vb.raw_gguf() {
            return Self::load_gguf(archive, head_dim, device);
        }
        let raw = vb.raw_safetensors().ok_or_else(|| {
            candle_core::Error::msg(
                "Qwen4-Exp PLE requires memory-mapped safetensors or GGUF weights (unset MISTRALRS_NO_MMAP)",
            )
        })?;
        let prefix = vb.prefix();
        let mut names = Vec::new();
        let mut starts = Vec::new();
        let mut rows = 0usize;
        let mut dtype = None;
        loop {
            let name = format!("{prefix}.{TABLE_SHARD_PREFIX}{}.weight", names.len());
            let Ok(view) = raw.get(&name) else {
                break;
            };
            let shape = view.shape();
            if shape.len() != 2 || shape[1] != head_dim {
                candle_core::bail!(
                    "PLE table shard {name} has shape {shape:?}, expected [_, {head_dim}]"
                );
            }
            let shard_dtype: DType = view.dtype().try_into()?;
            if !matches!(shard_dtype, DType::BF16 | DType::F16) {
                candle_core::bail!("PLE table shard {name} has unsupported dtype {shard_dtype:?}");
            }
            if dtype.is_some_and(|dtype| dtype != shard_dtype) {
                candle_core::bail!("PLE table shards mix dtypes");
            }
            dtype = Some(shard_dtype);
            starts.push(rows);
            rows += shape[0];
            names.push(name);
        }
        let dtype = dtype.ok_or_else(|| {
            candle_core::Error::msg(format!(
                "Qwen4-Exp PLE table `{prefix}.{TABLE_SHARD_PREFIX}N.weight` not found"
            ))
        })?;
        tracing::debug!(
            shards = names.len(),
            rows,
            "Qwen4-Exp PLE n-gram table is memory-mapped"
        );
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut table = Self {
            source: TableSource::Shards {
                raw,
                names,
                starts,
                dtype,
            },
            head_dim,
            #[cfg(feature = "cuda")]
            device_index: None,
            #[cfg(feature = "cuda")]
            resident: None,
        };
        #[cfg(feature = "cuda")]
        if device.is_cuda() && device_reads_pageable_memory(device)? {
            let TableSource::Shards {
                raw, names, starts, ..
            } = &table.source
            else {
                unreachable!()
            };
            let ptrs = names
                .iter()
                .map(|name| Ok(raw.get(name)?.data().as_ptr() as i64))
                .collect::<Result<Vec<_>>>()?;
            let starts = starts.iter().map(|s| *s as i64).collect::<Vec<_>>();
            let n = ptrs.len();
            table.device_index = Some(DeviceTableIndex::Shards {
                pointers: Tensor::from_vec(ptrs, n, device)?,
                starts: Tensor::from_vec(starts, n, device)?,
            });
            tracing::debug!("Qwen4-Exp PLE rows can be gathered through pageable memory access");
        }
        #[cfg(not(feature = "cuda"))]
        let _ = device;
        Ok(table)
    }

    fn load_gguf(archive: Arc<GgufArchive>, head_dim: usize, device: &Device) -> Result<Self> {
        let info = archive.tensor_info(GGUF_TABLE_NAME)?;
        if info.dtype().raw() != GGML_TYPE_IQ4_NL {
            candle_core::bail!(
                "Qwen4-Exp GGUF n-gram table `{GGUF_TABLE_NAME}` is {}, only IQ4_NL is supported",
                info.dtype().name()
            );
        }
        let rows = match info.shape() {
            &[rows, dim] if dim == head_dim && head_dim.is_multiple_of(IQ4_NL_BLOCK) => rows,
            shape => candle_core::bail!(
                "Qwen4-Exp GGUF n-gram table has shape {shape:?}, expected [_, {head_dim}]"
            ),
        };
        tracing::debug!(
            rows,
            "Qwen4-Exp PLE n-gram table is memory-mapped from GGUF"
        );
        #[cfg(not(feature = "cuda"))]
        let _ = device;
        Ok(Self {
            source: TableSource::Gguf { archive, rows },
            head_dim,
            #[cfg(feature = "cuda")]
            device_index: (device.is_cuda() && device_reads_pageable_memory(device)?)
                .then_some(DeviceTableIndex::Packed),
            #[cfg(feature = "cuda")]
            resident: None,
        })
    }

    /// Element dtype of the rows `device_index` points at.
    #[cfg(feature = "cuda")]
    fn shard_dtype(&self) -> DType {
        match &self.source {
            TableSource::Shards { dtype, .. } => *dtype,
            TableSource::Gguf { .. } => DType::U8,
            TableSource::Packed { .. } => DType::U8,
        }
    }

    #[cfg(feature = "cuda")]
    fn total_rows(&self) -> Result<usize> {
        match &self.source {
            TableSource::Shards {
                raw, names, starts, ..
            } => {
                let last = names.len() - 1;
                Ok(starts[last] + raw.get(&names[last])?.shape()[0])
            }
            TableSource::Gguf { rows, .. } => Ok(*rows),
            TableSource::Packed { rows, .. } => Ok(*rows),
        }
    }

    #[cfg(feature = "cuda")]
    fn fallback_path(&self) -> &'static str {
        if self.device_index.is_some() {
            if self.head_dim.is_multiple_of(RESIDENT_GROUP) {
                "prompt rows are gathered on the CPU; decode uses mapped GPU access"
            } else {
                "the GPU gathers rows from mapped host storage"
            }
        } else {
            "rows are gathered on the host"
        }
    }

    /// Move the whole table into device memory when it fits beside a working reserve: GGUF IQ4_NL rows
    /// are copied as is, safetensors rows are quantized to 8 or 4 bits.
    #[cfg(feature = "cuda")]
    fn make_resident(&mut self, device: &Device, budget: usize) -> Result<()> {
        if !self.head_dim.is_multiple_of(RESIDENT_GROUP) {
            tracing::warn!(
                head_dim = self.head_dim,
                "Qwen4-Exp PLE table rows are not a multiple of {RESIDENT_GROUP}, so it stays out of device memory; {}",
                self.fallback_path()
            );
            return Ok(());
        }
        let rows = self.total_rows()?;
        let available = crate::MemoryUsage.query(device)?.available().min(budget);
        let footprint = |bits: usize| resident_bytes(rows, self.head_dim, bits);
        let Some(bits) = RESIDENT_BITS
            .into_iter()
            .filter(|bits| {
                *bits == PLANNED_RESIDENT_BITS || matches!(self.source, TableSource::Shards { .. })
            })
            .find(|bits| footprint(*bits) + RESIDENT_RESERVE_BYTES <= available)
        else {
            tracing::warn!(
                available_gb = available >> 30,
                needed_gb = (footprint(PLANNED_RESIDENT_BITS) + RESIDENT_RESERVE_BYTES) >> 30,
                "Qwen4-Exp PLE table does not fit in device memory; {}",
                self.fallback_path()
            );
            return Ok(());
        };
        let start = std::time::Instant::now();
        let resident = match &self.source {
            TableSource::Gguf { archive, .. } => self.copy_iq4_nl(archive, rows, device)?,
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
            format = ?resident.format,
            size_gb = footprint(bits) >> 30,
            elapsed_s = start.elapsed().as_secs(),
            "Qwen4-Exp PLE table resident in device memory"
        );
        self.resident = Some(resident);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn copy_iq4_nl(
        &self,
        archive: &GgufArchive,
        rows: usize,
        device: &Device,
    ) -> Result<ResidentTable> {
        let row_bytes = iq4_nl_row_bytes(self.head_dim);
        let bytes = archive.tensor_data(GGUF_TABLE_NAME)?.bytes();
        let data = unsafe { Tensor::empty((rows, row_bytes), DType::U8, device)? };
        for (chunk, src) in bytes.chunks(RESIDENT_COPY_ROWS * row_bytes).enumerate() {
            prefault_source_pages(src);
            data.slice_set(
                &Tensor::from_slice(src, (src.len() / row_bytes, row_bytes), device)?,
                0,
                chunk * RESIDENT_COPY_ROWS,
            )?;
            release_source_pages(src);
        }
        Ok(ResidentTable {
            data,
            scales: None,
            format: crate::cuda::qwen4_exp::PleTableFormat::Iq4Nl,
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
        let groups = self.head_dim / RESIDENT_GROUP;
        let row_bytes = self.head_dim * bits / 8;
        let data = unsafe { Tensor::empty((rows, row_bytes), DType::U8, device)? };
        let scales = unsafe { Tensor::empty((rows, groups), DType::F16, device)? };
        let source_row_bytes = self.head_dim * dtype.size_in_bytes();
        for (shard, name) in names.iter().enumerate() {
            let view = raw.get(name)?;
            for (chunk, src) in view
                .data()
                .chunks(RESIDENT_COPY_ROWS * source_row_bytes)
                .enumerate()
            {
                let chunk_rows = src.len() / source_row_bytes;
                let start_row = starts[shard] + chunk * RESIDENT_COPY_ROWS;
                prefault_source_pages(src);
                let source =
                    Tensor::from_raw_buffer(src, dtype, &[chunk_rows, self.head_dim], device)?;
                crate::cuda::qwen4_exp::ple_quantize_table(
                    &source,
                    &data.narrow(0, start_row, chunk_rows)?,
                    &scales.narrow(0, start_row, chunk_rows)?,
                    bits,
                )?;
                release_source_pages(src);
            }
        }
        Ok(ResidentTable {
            data,
            scales: Some(scales),
            format: if bits == 8 {
                crate::cuda::qwen4_exp::PleTableFormat::Q8
            } else {
                crate::cuda::qwen4_exp::PleTableFormat::Q4
            },
        })
    }

    #[cfg(feature = "cuda")]
    fn prepare_mapped(&mut self, device: &Device, chunk_rows: usize) -> Result<()> {
        use std::io::{Seek, SeekFrom, Write};

        if let TableSource::Gguf { archive, .. } = &self.source {
            if matches!(self.device_index, Some(DeviceTableIndex::Packed)) {
                let started = std::time::Instant::now();
                let bytes = archive.tensor_data(GGUF_TABLE_NAME)?.bytes();
                for byte in bytes.iter().step_by(HOST_PREFAULT_STRIDE_BYTES) {
                    std::hint::black_box(*byte);
                }
                if let Some(byte) = bytes.last() {
                    std::hint::black_box(*byte);
                }
                tracing::info!(
                    size_gb = bytes.len() >> 30,
                    elapsed_s = started.elapsed().as_secs(),
                    "Qwen4-Exp GGUF PLE table prefaulted in reclaimable memory-mapped storage"
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
        if !self.head_dim.is_multiple_of(RESIDENT_GROUP) {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let rows = self.total_rows()?;
        let row_bytes = self.head_dim * PLANNED_RESIDENT_BITS / 8;
        let scale_row_bytes = self.head_dim / RESIDENT_GROUP * DType::F16.size_in_bytes();
        let source_row_bytes = self.head_dim * dtype.size_in_bytes();
        let data_bytes = rows * row_bytes;
        // /tmp can be tmpfs; the packed table needs reclaimable file-backed pages.
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .join("mistralrs")
            .join("ple");
        std::fs::create_dir_all(&cache_dir).map_err(candle_core::Error::wrap)?;
        let mut file = tempfile::tempfile_in(cache_dir).map_err(candle_core::Error::wrap)?;
        file.set_len((data_bytes + rows * scale_row_bytes) as u64)
            .map_err(candle_core::Error::wrap)?;
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
                    Tensor::from_raw_buffer(src, *dtype, &[chunk_rows, self.head_dim], device)?;
                let data = unsafe { Tensor::empty((chunk_rows, row_bytes), DType::U8, device)? };
                let scales = unsafe {
                    Tensor::empty(
                        (chunk_rows, self.head_dim / RESIDENT_GROUP),
                        DType::F16,
                        device,
                    )?
                };
                crate::cuda::qwen4_exp::ple_quantize_table(
                    &source,
                    &data,
                    &scales,
                    PLANNED_RESIDENT_BITS,
                )?;
                let data = data.flatten_all()?.to_vec1::<u8>()?;
                let scales = scales.flatten_all()?.to_vec1::<half::f16>()?;
                file.seek(SeekFrom::Start((first * row_bytes) as u64))
                    .and_then(|_| file.write_all(&data))
                    .map_err(candle_core::Error::wrap)?;
                file.seek(SeekFrom::Start(
                    (data_bytes + first * scale_row_bytes) as u64,
                ))
                .and_then(|_| file.write_all(bytemuck::cast_slice(&scales)))
                .and_then(|_| file.sync_data())
                .map_err(candle_core::Error::wrap)?;
                release_source_pages(src);
                evict_file_cache(&file, first * row_bytes, data.len());
                evict_file_cache(
                    &file,
                    data_bytes + first * scale_row_bytes,
                    scales.len() * 2,
                );
            }
        }
        let map = unsafe { memmap2::MmapOptions::new().populate().map(&file) }
            .map_err(candle_core::Error::wrap)?;
        self.source = TableSource::Packed {
            map,
            rows,
            data_bytes,
        };
        self.device_index =
            device_reads_pageable_memory(device)?.then_some(DeviceTableIndex::Packed);
        tracing::info!(
            size_gb = resident_bytes(rows, self.head_dim, PLANNED_RESIDENT_BITS) >> 30,
            elapsed_s = started.elapsed().as_secs(),
            "Qwen4-Exp PLE table quantized into reclaimable memory-mapped storage"
        );
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn gather_mapped(
        &self,
        rows: &Tensor,
        layout: &crate::cuda::qwen4_exp::TokenLayout,
    ) -> Result<Tensor> {
        if layout.kv_lens.is_some() {
            // CPU gathers avoid first-touch stalls from scattered GPU reads of the mapped table.
            let (tokens, heads) = rows.dims2()?;
            let host_rows = rows
                .flatten_all()?
                .to_vec1::<i64>()?
                .into_iter()
                .map(|row| row as u64)
                .collect::<Vec<_>>();
            return self
                .gather_host(&host_rows, &Device::Cpu)?
                .to_dtype(DType::BF16)?
                .reshape((tokens, heads * self.head_dim))?
                .to_device(rows.device());
        }
        let (data, scales, format) = match &self.source {
            TableSource::Gguf { archive, .. } => (
                archive.tensor_data(GGUF_TABLE_NAME)?.bytes(),
                None,
                crate::cuda::qwen4_exp::PleTableFormat::Iq4Nl,
            ),
            TableSource::Packed {
                map, data_bytes, ..
            } => (
                &map[..*data_bytes],
                Some(&map[*data_bytes..]),
                crate::cuda::qwen4_exp::PleTableFormat::Q4,
            ),
            TableSource::Shards { .. } => unreachable!("packed device table index"),
        };
        crate::cuda::qwen4_exp::ple_gather_quant_mapped(rows, data, scales, self.head_dim, format)
    }

    /// Host gather of `rows` into a `[rows.len(), head_dim]` tensor on `device`.
    fn gather_host(&self, rows: &[u64], device: &Device) -> Result<Tensor> {
        let outside = |row: u64| {
            candle_core::Error::msg(format!("PLE row {row} is outside the n-gram table"))
        };
        match &self.source {
            TableSource::Shards {
                raw,
                names,
                starts,
                dtype,
            } => {
                let row_bytes = self.head_dim * dtype.size_in_bytes();
                let mut data = vec![0u8; rows.len() * row_bytes];
                for (dst, &row) in data.chunks_exact_mut(row_bytes).zip(rows) {
                    let idx = row as usize;
                    let shard = starts.partition_point(|start| *start <= idx) - 1;
                    let view = raw.get(&names[shard])?;
                    let offset = (idx - starts[shard]) * row_bytes;
                    let src = view
                        .data()
                        .get(offset..offset + row_bytes)
                        .ok_or_else(|| outside(row))?;
                    dst.copy_from_slice(src);
                }
                Tensor::from_raw_buffer(&data, *dtype, &[rows.len(), self.head_dim], &Device::Cpu)?
                    .to_device(device)
            }
            TableSource::Gguf { archive, .. } => {
                let row_bytes = iq4_nl_row_bytes(self.head_dim);
                let bytes = archive.tensor_data(GGUF_TABLE_NAME)?.bytes();
                let mut out = vec![0f32; rows.len() * self.head_dim];
                for (dst, &row) in out.chunks_exact_mut(self.head_dim).zip(rows) {
                    let offset = row as usize * row_bytes;
                    let src = bytes
                        .get(offset..offset + row_bytes)
                        .ok_or_else(|| outside(row))?;
                    dequantize_iq4_nl_row(src, dst);
                }
                Tensor::from_vec(out, (rows.len(), self.head_dim), device)
            }
            #[cfg(feature = "cuda")]
            TableSource::Packed {
                map, data_bytes, ..
            } => {
                let row_bytes = self.head_dim / 2;
                let scale_row_bytes = self.head_dim / RESIDENT_GROUP * 2;
                let mut out = vec![0f32; rows.len() * self.head_dim];
                for (dst, &row) in out.chunks_exact_mut(self.head_dim).zip(rows) {
                    let offset = row as usize * row_bytes;
                    let src = map
                        .get(offset..offset + row_bytes)
                        .ok_or_else(|| outside(row))?;
                    let scale_offset = data_bytes + row as usize * scale_row_bytes;
                    let scales = map
                        .get(scale_offset..scale_offset + scale_row_bytes)
                        .ok_or_else(|| outside(row))?;
                    for (i, value) in dst.iter_mut().enumerate() {
                        let scale = 2 * (i / RESIDENT_GROUP);
                        let scale =
                            half::f16::from_le_bytes([scales[scale], scales[scale + 1]]).to_f32();
                        let q = (src[i / 2] >> (4 * (i % 2))) & 15;
                        *value = (i32::from(q) - 8) as f32 * scale;
                    }
                }
                Tensor::from_vec(out, (rows.len(), self.head_dim), device)
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn evict_file_cache(file: &std::fs::File, offset: usize, len: usize) {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                offset as libc::off_t,
                len as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, offset, len);
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
fn device_reads_pageable_memory(device: &Device) -> Result<bool> {
    use candle_core::cuda::cudarc::driver::{result, sys};
    let ordinal = device.as_cuda_device()?.cuda_stream().context().ordinal();
    let dev = result::device::get(ordinal as i32).map_err(candle_core::Error::wrap)?;
    let supported = unsafe {
        result::device::get_attribute(
            dev,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
        )
    }
    .map_err(candle_core::Error::wrap)?;
    Ok(supported == 1)
}

/// PLE state before a multi-token speculative verify plus that step's inputs, so a rejected tail can be
/// dropped by rebuilding the state from the accepted prefix.
#[derive(Clone)]
pub(super) struct PleStash {
    pub slots: Vec<u32>,
    // [b, hc * hidden, state_len]
    pub conv: Tensor,
    // [b, context] token history as `id + 1`
    pub history: Tensor,
    // [b, q, hc * hidden]
    pub conv_inputs: Tensor,
    // [b, q] as `id + 1`
    pub tokens: Tensor,
}

impl PleStash {
    pub(super) fn into_tensors(self) -> Vec<Tensor> {
        vec![self.conv, self.history, self.conv_inputs, self.tokens]
    }

    pub(super) fn from_tensors(tensors: &[Tensor], slots: Vec<u32>) -> Result<Self> {
        let [conv, history, conv_inputs, tokens] = tensors else {
            candle_core::bail!("PLE stash expects 4 tensors, got {}", tensors.len());
        };
        Ok(Self {
            slots,
            conv: conv.clone(),
            history: history.clone(),
            conv_inputs: conv_inputs.clone(),
            tokens: tokens.clone(),
        })
    }

    /// Rewrites each `(batch_idx, keep_rows)` sequence's state to the pre-verify state advanced by its kept rows.
    pub(super) fn rollback(
        &self,
        rows: &[(usize, usize)],
        conv_pool: &Tensor,
        history_pool: &Tensor,
    ) -> Result<()> {
        let state_len = self.conv.dim(2)?;
        let context = self.history.dim(1)?;
        for &(batch_idx, keep) in rows {
            let slot = *self.slots.get(batch_idx).ok_or_else(|| {
                candle_core::Error::msg(format!("PLE stash has no batch row {batch_idx}"))
            })? as usize;
            let conv = Tensor::cat(
                &[
                    self.conv.i(batch_idx)?.t()?,
                    self.conv_inputs
                        .i(batch_idx)?
                        .narrow(0, 0, keep)?
                        .to_dtype(self.conv.dtype())?,
                ],
                0,
            )?;
            let conv = conv
                .narrow(0, conv.dim(0)? - state_len, state_len)?
                .t()?
                .unsqueeze(0)?
                .contiguous()?;
            conv_pool.slice_set(&conv, 0, slot)?;
            let history = Tensor::cat(
                &[
                    self.history.i(batch_idx)?,
                    self.tokens.i(batch_idx)?.narrow(0, 0, keep)?,
                ],
                0,
            )?;
            let history = history
                .narrow(0, history.dim(0)? - context, context)?
                .unsqueeze(0)?
                .contiguous()?;
            history_pool.slice_set(&history, 0, slot)?;
        }
        Ok(())
    }
}

/// Hybrid-cache PLE rows: conv history `[slots, hc * hidden, state_len]`, token history as `id + 1` f32.
pub(super) struct PleState<'a> {
    pub conv: &'a Tensor,
    pub history: &'a Tensor,
    pub slots: &'a Tensor,
}

/// Where each logical sequence sits on the flattened token axis.
pub(super) struct PleBatch<'a> {
    pub seqs: &'a [(usize, usize)],
    #[cfg(feature = "cuda")]
    pub layout: Option<&'a crate::cuda::qwen4_exp::TokenLayout>,
}

/// Hashed n-gram embedding injected into every hyper-connection stream (HF `Qwen4ExpTextPLELayer`).
pub(super) struct PleLayer {
    table: PleTable,
    #[cfg(feature = "cuda")]
    pub(super) allow_resident_table: bool,
    hash: PleHash,
    #[cfg(feature = "cuda")]
    hash_ffi: crate::cuda::qwen4_exp::Q4PleHashParams,
    key_proj: Arc<dyn QuantMethod>,
    value_proj: Arc<dyn QuantMethod>,
    norm_key: Tensor,
    norm_query: Tensor,
    norm_conv: Tensor,
    // [hc * hidden, kernel]
    conv_weight: Tensor,
    cfg: PleConfig,
    hc: usize,
    hidden: usize,
    eps: f64,
}

impl PleLayer {
    pub(super) fn load(cfg: &TextConfig, ple: &PleConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let hc_hidden = cfg.hc_hidden_size();
        let device = vb.device().clone();
        let hash = PleHash::new(cfg, ple, 0);
        let table = PleTable::load(
            &vb.pp("ple_embedding").pp("ngram_embedding"),
            ple.head_dim(),
            &device,
        )?;
        if let TableSource::Shards { raw, .. } = &table.source {
            verify_hash_buffers(raw, &vb.pp("ple_embedding").prefix(), &hash)?;
        }
        let norm = |name: &str| -> Result<Tensor> {
            vb.pp(name).get(hc_hidden, "weight")?.to_dtype(DType::F32)? + 1.0
        };
        let conv_weight = vb
            .pp("conv1d")
            .get((hc_hidden, 1, ple.conv_kernel_size), "weight")?
            .squeeze(1)?
            .contiguous()?;
        Ok(Self {
            #[cfg(feature = "cuda")]
            hash_ffi: hash.ffi()?,
            table,
            #[cfg(feature = "cuda")]
            allow_resident_table: true,
            hash,
            key_proj: ReplicatedLayer::new(
                ple.embed_dim,
                hc_hidden,
                &cfg.quantization_config,
                false,
                vb.pp("key_proj"),
            )?,
            value_proj: ReplicatedLayer::new(
                ple.embed_dim,
                cfg.hidden_size,
                &cfg.quantization_config,
                false,
                vb.pp("value_proj"),
            )?,
            norm_key: norm("norm_key")?,
            norm_query: norm("norm_query")?,
            norm_conv: norm("norm_conv")?,
            conv_weight,
            cfg: ple.clone(),
            hc: cfg.hc_count,
            hidden: cfg.hidden_size,
            eps: cfg.rms_norm_eps,
        })
    }

    #[cfg(feature = "cuda")]
    pub(super) fn gathers_on_device(&self) -> bool {
        self.table.resident.is_some() || self.table.device_index.is_some()
    }

    #[cfg(feature = "cuda")]
    pub(super) fn reclaimable_weight_cache_bytes(&self) -> Option<(Device, usize)> {
        self.table
            .reclaimable_weight_cache_bytes()
            .map(|bytes| (self.conv_weight.device().clone(), bytes))
    }

    #[cfg(feature = "cuda")]
    pub(super) fn prepare_table(
        &mut self,
        budget: &dyn Fn(&Device) -> Result<usize>,
    ) -> Result<()> {
        let device = self.conv_weight.device();
        if device.is_cuda() {
            if self.allow_resident_table {
                self.table.make_resident(device, budget(device)?)?;
            }
            if self.table.resident.is_none() {
                self.table.prepare_mapped(device, RESIDENT_COPY_ROWS)?;
            }
            return Ok(());
        }
        tracing::info!("Qwen4-Exp PLE rows are gathered from the memory-mapped table on the host");
        Ok(())
    }

    /// `hidden_states + ple(hidden_states, tokens)`, advancing the per-sequence state. `conv_inputs_out`
    /// receives the `[tokens, hc * hidden]` rows appended to the conv history.
    pub(super) fn forward(
        &self,
        hidden_states: &Tensor,
        tokens: &Tensor,
        state: &PleState<'_>,
        batch: &PleBatch<'_>,
        conv_inputs_out: Option<&mut Option<Tensor>>,
    ) -> Result<Tensor> {
        let dims = hidden_states.dims().to_vec();
        let width = dims[dims.len() - 1];
        let flat = hidden_states.reshape(((), width))?;
        let n_tokens = flat.dim(0)?;
        #[cfg(feature = "cuda")]
        if let (true, Some(layout)) = (self.gathers_on_device(), batch.layout) {
            let rows = crate::cuda::qwen4_exp::ple_hash(
                tokens,
                layout,
                state.history,
                state.slots,
                &self.hash_ffi,
            )?;
            let emb = match (&self.table.resident, &self.table.device_index) {
                (Some(resident), _) => crate::cuda::qwen4_exp::ple_gather_quant(
                    &rows,
                    &resident.data,
                    resident.scales.as_ref(),
                    self.table.head_dim,
                    resident.format,
                )?,
                (None, Some(DeviceTableIndex::Shards { pointers, starts })) => {
                    crate::cuda::qwen4_exp::ple_gather(
                        &rows,
                        pointers,
                        starts,
                        self.table.head_dim,
                        self.table.shard_dtype(),
                    )?
                }
                (None, Some(DeviceTableIndex::Packed)) => {
                    self.table.gather_mapped(&rows, layout)?
                }
                (None, None) => unreachable!("checked by gathers_on_device"),
            }
            .to_dtype(flat.dtype())?;
            let (gated, normed) = self.gate(&emb, &flat)?;
            if let Some(out) = conv_inputs_out {
                *out = Some(normed.clone());
            }
            let out = crate::cuda::qwen4_exp::ple_conv(crate::cuda::qwen4_exp::PleConvArgs {
                normed: &normed,
                gated: &gated,
                residual: &flat,
                weight: &self.conv_weight,
                state_pool: state.conv,
                slots: state.slots,
                layout,
                dilation: self.cfg.conv_dilation(),
            })?;
            return out.reshape(dims);
        }
        let emb = self.host_embeddings(tokens, state, batch.seqs, n_tokens)?;
        let (gated, normed) = self.gate(&emb.to_dtype(flat.dtype())?, &flat)?;
        if let Some(out) = conv_inputs_out {
            *out = Some(normed.clone());
        }
        let out = self.conv_host(&flat, &gated, &normed, state, batch.seqs)?;
        out.reshape(dims)
    }

    /// `(gated_value, norm_conv(gated_value))`, both `[tokens, hc * hidden]`.
    fn gate(&self, emb: &Tensor, flat: &Tensor) -> Result<(Tensor, Tensor)> {
        let key = self.key_proj.forward(emb)?;
        let value = self.value_proj.forward(emb)?;
        #[cfg(feature = "cuda")]
        if flat.device().is_cuda() {
            return crate::cuda::qwen4_exp::ple_gate(
                &key,
                flat,
                &value,
                crate::cuda::qwen4_exp::PleGateWeights {
                    key: &self.norm_key,
                    query: &self.norm_query,
                    conv: &self.norm_conv,
                },
                self.hc,
                self.eps,
            );
        }
        let n = flat.dim(0)?;
        let key = grouped_rms_norm(&key, &self.norm_key, self.hc, self.eps)?
            .to_dtype(DType::F32)?
            .reshape((n, self.hc, self.hidden))?;
        let query = grouped_rms_norm(flat, &self.norm_query, self.hc, self.eps)?
            .to_dtype(DType::F32)?
            .reshape((n, self.hc, self.hidden))?;
        let gate = ((key * query)?.sum_keepdim(D::Minus1)? / (self.hidden as f64).sqrt())?;
        let magnitude = gate.abs()?.maximum(GATE_MAGNITUDE_FLOOR)?.sqrt()?;
        let gate = candle_nn::ops::sigmoid(&(magnitude * gate.sign()?)?)?;
        let gated = gate
            .broadcast_mul(&value.to_dtype(DType::F32)?.unsqueeze(1)?)?
            .reshape((n, self.hc * self.hidden))?
            .to_dtype(flat.dtype())?;
        let normed = grouped_rms_norm(&gated, &self.norm_conv, self.hc, self.eps)?;
        Ok((gated, normed))
    }

    fn host_embeddings(
        &self,
        tokens: &Tensor,
        state: &PleState<'_>,
        seqs: &[(usize, usize)],
        n_tokens: usize,
    ) -> Result<Tensor> {
        let tokens = tokens
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        let slots = state.slots.to_dtype(DType::U32)?.to_vec1::<u32>()?;
        let history = state.history.to_device(&Device::Cpu)?;
        let heads = self.hash.num_heads();
        let context = self.cfg.context_len();
        let mut rows = vec![self.hash.offsets[0]; n_tokens * heads];
        let mut new_history = Vec::with_capacity(seqs.len());
        for (seq, &(start, len)) in seqs.iter().enumerate() {
            let stored = history.i(slots[seq] as usize)?.to_vec1::<f32>()?;
            let mut hist = stored
                .iter()
                .map(|v| (*v > 0.0).then(|| *v as u32 - 1))
                .collect::<Vec<_>>();
            let seq_tokens = &tokens[start..start + len];
            for local in 0..len {
                let t = start + local;
                self.hash.rows(
                    seq_tokens,
                    local,
                    &hist,
                    &mut rows[t * heads..(t + 1) * heads],
                );
            }
            hist.extend(seq_tokens.iter().map(|t| Some(*t)));
            let tail = hist[hist.len() - context..]
                .iter()
                .map(|t| t.map_or(0.0, |t| t as f32 + 1.0))
                .collect::<Vec<_>>();
            new_history.push(tail);
        }
        for (seq, tail) in new_history.into_iter().enumerate() {
            let row = Tensor::from_vec(tail, (1, context), state.history.device())?;
            state.history.slice_set(&row, 0, slots[seq] as usize)?;
        }
        let emb = self.table.gather_host(&rows, state.conv.device())?;
        emb.reshape((n_tokens, heads * self.table.head_dim))
    }

    fn conv_host(
        &self,
        flat: &Tensor,
        gated: &Tensor,
        normed: &Tensor,
        state: &PleState<'_>,
        seqs: &[(usize, usize)],
    ) -> Result<Tensor> {
        let slots = state.slots.to_dtype(DType::U32)?.to_vec1::<u32>()?;
        let kernel = self.cfg.conv_kernel_size;
        let dilation = self.cfg.conv_dilation();
        let state_len = self.cfg.conv_state_len();
        let weight = self.conv_weight.to_dtype(DType::F32)?;
        let mut conv_out = Vec::with_capacity(seqs.len());
        for (seq, &(start, len)) in seqs.iter().enumerate() {
            let slot = slots[seq] as usize;
            let history = state.conv.i(slot)?.to_dtype(DType::F32)?.t()?;
            let x = normed.narrow(0, start, len)?.to_dtype(DType::F32)?;
            let padded = Tensor::cat(&[&history, &x], 0)?;
            let mut acc = Tensor::zeros(x.shape(), DType::F32, x.device())?;
            for k in 0..kernel {
                let offset = state_len - (kernel - 1 - k) * dilation;
                let tap = padded.narrow(0, offset, len)?;
                acc = (acc + tap.broadcast_mul(&weight.i((.., k))?.unsqueeze(0)?)?)?;
            }
            conv_out.push(candle_nn::ops::silu(&acc)?);
            let new_state = padded
                .narrow(0, padded.dim(0)? - state_len, state_len)?
                .t()?
                .to_dtype(state.conv.dtype())?
                .unsqueeze(0)?
                .contiguous()?;
            state.conv.slice_set(&new_state, 0, slot)?;
        }
        let conv = Tensor::cat(&conv_out, 0)?;
        if conv.dim(0)? != flat.dim(0)? {
            candle_core::bail!("Qwen4-Exp host PLE requires unpadded, contiguous sequences");
        }
        ((flat.to_dtype(DType::F32)? + gated.to_dtype(DType::F32)?)? + conv)?.to_dtype(flat.dtype())
    }
}

fn verify_hash_buffers(raw: &MmapedSafetensors, prefix: &str, hash: &PleHash) -> Result<()> {
    let read = |name: &str| -> Result<Option<Vec<u64>>> {
        let Ok(view) = raw.get(&format!("{prefix}.{name}")) else {
            return Ok(None);
        };
        if view.dtype() != safetensors::Dtype::I64 {
            candle_core::bail!("PLE buffer {name} must be int64");
        }
        Ok(Some(
            view.data()
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| i64::from_le_bytes(*b) as u64)
                .collect(),
        ))
    };
    for (name, expected) in [
        ("layer_multipliers", &hash.multipliers),
        ("ngram_heads_vocab_sizes", &hash.vocab_sizes),
        ("ngram_heads_offsets", &hash.offsets),
    ] {
        if let Some(stored) = read(name)? {
            if &stored != expected {
                candle_core::bail!(
                    "Qwen4-Exp PLE `{name}` in the checkpoint ({stored:?}) does not match the config ({expected:?})"
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision_models::qwen4_exp::config::tests::flash_next_text_config;
    #[cfg(feature = "cuda")]
    const GGUF_TEST_ALIGNMENT: usize = 32;

    #[cfg(feature = "cuda")]
    fn mapped_bytes(bytes: &[u8]) -> Result<memmap2::Mmap> {
        use std::io::Write;
        let mut file = tempfile::tempfile().map_err(candle_core::Error::wrap)?;
        file.write_all(bytes).map_err(candle_core::Error::wrap)?;
        unsafe { memmap2::MmapOptions::new().map(&file) }.map_err(candle_core::Error::wrap)
    }

    #[cfg(feature = "cuda")]
    fn mapped_iq4_table(
        data: &[u8],
        rows: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<PleTable> {
        use std::io::Write;
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(GGUF_TABLE_NAME.len() as u64).to_le_bytes());
        bytes.extend_from_slice(GGUF_TABLE_NAME.as_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&(head_dim as u64).to_le_bytes());
        bytes.extend_from_slice(&(rows as u64).to_le_bytes());
        bytes.extend_from_slice(&GGML_TYPE_IQ4_NL.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.resize(
            bytes.len().div_ceil(GGUF_TEST_ALIGNMENT) * GGUF_TEST_ALIGNMENT,
            0,
        );
        bytes.extend_from_slice(data);
        bytes.resize(
            bytes.len().div_ceil(GGUF_TEST_ALIGNMENT) * GGUF_TEST_ALIGNMENT,
            0,
        );
        let mut file = tempfile::NamedTempFile::new().map_err(candle_core::Error::wrap)?;
        file.write_all(&bytes).map_err(candle_core::Error::wrap)?;
        let archive = Arc::new(GgufArchive::open_file(file.path())?);
        PleTable::load_gguf(archive, head_dim, device)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn resident_quantization_matches_host() -> Result<()> {
        const HEAD_DIM: usize = 160;
        const ROWS: usize = 37;
        const CHUNK_ROWS: usize = 11;
        const GUARD_BYTE: u8 = 0xa5;
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            for bits in RESIDENT_BITS {
                let qmax = if bits == 8 { 127.0 } else { 7.0 };
                let row_bytes = HEAD_DIM * bits / 8;
                let groups = HEAD_DIM / RESIDENT_GROUP;
                let values = (0..ROWS * HEAD_DIM)
                    .map(|i| match i / RESIDENT_GROUP {
                        0 => 0.0,
                        1 => match i % RESIDENT_GROUP {
                            0 => qmax,
                            1 => -qmax,
                            j => (j % 8) as f32 / 2.0 - 1.5,
                        },
                        2 => (i as f32 - 80.0) * 1e-9,
                        _ => {
                            let scale = [1e-5, 1e-2, 1.0, 128.0][i / RESIDENT_GROUP % 4];
                            ((splitmix64(i as u64) >> 40) as f32 / (1u64 << 24) as f32 - 0.5)
                                * scale
                        }
                    })
                    .collect::<Vec<_>>();
                let source =
                    Tensor::from_vec(values, (ROWS, HEAD_DIM), &Device::Cpu)?.to_dtype(dtype)?;
                let values = source
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let source = source.to_device(&device)?;
                let data = Tensor::from_vec(
                    vec![GUARD_BYTE; (ROWS + 2) * row_bytes],
                    (ROWS + 2, row_bytes),
                    &device,
                )?;
                let scales = Tensor::zeros((ROWS + 2, groups), DType::F16, &device)?;
                for first in (0..ROWS).step_by(CHUNK_ROWS) {
                    let count = CHUNK_ROWS.min(ROWS - first);
                    crate::cuda::qwen4_exp::ple_quantize_table(
                        &source.narrow(0, first, count)?,
                        &data.narrow(0, first + 1, count)?,
                        &scales.narrow(0, first + 1, count)?,
                        bits,
                    )?;
                }
                let got = data.flatten_all()?.to_vec1::<u8>()?;
                let got_scales = scales.flatten_all()?.to_vec1::<half::f16>()?;
                assert!(got[..row_bytes].iter().all(|v| *v == GUARD_BYTE));
                assert!(got[(ROWS + 1) * row_bytes..]
                    .iter()
                    .all(|v| *v == GUARD_BYTE));
                assert!(got_scales[..groups].iter().all(|v| *v == half::f16::ZERO));
                assert!(got_scales[(ROWS + 1) * groups..]
                    .iter()
                    .all(|v| *v == half::f16::ZERO));
                let mut expected = vec![0u8; ROWS * row_bytes];
                for (g, values) in values.chunks_exact(RESIDENT_GROUP).enumerate() {
                    let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                    let scale = half::f16::from_f32(amax / qmax);
                    assert_eq!(got_scales[groups + g], scale, "{dtype:?} Q{bits} group {g}");
                    let inv = if scale.to_f32() > 0.0 {
                        1.0 / scale.to_f32()
                    } else {
                        0.0
                    };
                    for (j, value) in values.iter().enumerate() {
                        let q = (value * inv).round().clamp(-qmax, qmax) as i32;
                        let i = g * RESIDENT_GROUP + j;
                        if bits == 8 {
                            expected[i] = q as i8 as u8;
                        } else {
                            expected[i / 2] |= ((q + 8) as u8) << (4 * (i % 2));
                        }
                    }
                }
                assert_eq!(
                    &got[row_bytes..(ROWS + 1) * row_bytes],
                    &expected,
                    "{dtype:?} Q{bits}"
                );
                if device_reads_pageable_memory(&device)? {
                    let mapped_data = mapped_bytes(&got)?;
                    let mapped_scales = mapped_bytes(bytemuck::cast_slice(&got_scales))?;
                    let rows = Tensor::from_slice(&[0i64, 36, 5, 5, 17, 1], (2, 3), &device)?;
                    let format = if bits == 8 {
                        crate::cuda::qwen4_exp::PleTableFormat::Q8
                    } else {
                        crate::cuda::qwen4_exp::PleTableFormat::Q4
                    };
                    let expected = crate::cuda::qwen4_exp::ple_gather_quant(
                        &rows,
                        &data.narrow(0, 1, ROWS)?,
                        Some(&scales.narrow(0, 1, ROWS)?),
                        HEAD_DIM,
                        format,
                    )?
                    .flatten_all()?
                    .to_vec1::<half::bf16>()?;
                    let actual = crate::cuda::qwen4_exp::ple_gather_quant_mapped(
                        &rows,
                        &mapped_data[row_bytes..(ROWS + 1) * row_bytes],
                        Some(&mapped_scales[groups * 2..(ROWS + 1) * groups * 2]),
                        HEAD_DIM,
                        format,
                    )?
                    .flatten_all()?
                    .to_vec1::<half::bf16>()?;
                    assert_eq!(actual, expected, "mapped {dtype:?} Q{bits}");
                }
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn mapped_quantization_preserves_shard_and_chunk_offsets() -> Result<()> {
        const HEAD_DIM: usize = 160;
        const ROWS: usize = 37;
        const FIRST_SHARD_ROWS: usize = 19;
        const CHUNK_ROWS: usize = 11;
        let device = Device::new_cuda(0)?;
        for dtype in [DType::F16, DType::BF16] {
            let source = Tensor::from_vec(
                (0..ROWS * HEAD_DIM)
                    .map(|i| (i as f32 * 0.3).sin())
                    .collect::<Vec<_>>(),
                (ROWS, HEAD_DIM),
                &Device::Cpu,
            )?
            .to_dtype(dtype)?;
            let names = vec!["shard_0.weight".to_string(), "shard_1.weight".to_string()];
            let starts = vec![0, FIRST_SHARD_ROWS];
            let file = tempfile::NamedTempFile::new().map_err(candle_core::Error::wrap)?;
            candle_core::safetensors::save(
                &std::collections::HashMap::from([
                    (
                        names[0].clone(),
                        source.narrow(0, 0, FIRST_SHARD_ROWS)?.force_contiguous()?,
                    ),
                    (
                        names[1].clone(),
                        source
                            .narrow(0, FIRST_SHARD_ROWS, ROWS - FIRST_SHARD_ROWS)?
                            .force_contiguous()?,
                    ),
                ]),
                file.path(),
            )?;
            let raw = Arc::new(unsafe { MmapedSafetensors::new(file.path())? });
            let mut table = PleTable {
                source: TableSource::Shards {
                    raw,
                    names,
                    starts,
                    dtype,
                },
                head_dim: HEAD_DIM,
                device_index: None,
                resident: None,
            };
            table.make_resident(&device, 0)?;
            assert!(table.resident.is_none());
            table.prepare_mapped(&device, CHUNK_ROWS)?;
            let TableSource::Packed {
                map, data_bytes, ..
            } = &table.source
            else {
                unreachable!()
            };
            let row_bytes = HEAD_DIM / 2;
            let groups = HEAD_DIM / RESIDENT_GROUP;
            let data = Tensor::zeros((ROWS, row_bytes), DType::U8, &device)?;
            let scales = Tensor::zeros((ROWS, groups), DType::F16, &device)?;
            crate::cuda::qwen4_exp::ple_quantize_table(
                &source.to_device(&device)?,
                &data,
                &scales,
                PLANNED_RESIDENT_BITS,
            )?;
            assert_eq!(&map[..*data_bytes], data.flatten_all()?.to_vec1::<u8>()?);
            assert_eq!(
                &map[*data_bytes..],
                bytemuck::cast_slice::<_, u8>(&scales.flatten_all()?.to_vec1::<half::f16>()?)
            );
            let lookups = [0u64, 36, 18, 19, 10, 11];
            let host = table
                .gather_host(&lookups, &Device::Cpu)?
                .to_dtype(DType::BF16)?
                .flatten_all()?
                .to_vec1::<half::bf16>()?;
            if device_reads_pageable_memory(&device)? {
                let rows = Tensor::from_slice(&lookups.map(|row| row as i64), (2, 3), &device)?;
                for layout in [
                    crate::cuda::qwen4_exp::TokenLayout::rectangular(2, 1),
                    crate::cuda::qwen4_exp::TokenLayout::from_host(&[(0, 2)], &[2], 2, &device)?,
                ] {
                    let actual = table.gather_mapped(&rows, &layout)?;
                    assert!(actual.device().same_device(&device));
                    assert_eq!(actual.dims(), &[2, 3 * HEAD_DIM]);
                    assert_eq!(actual.flatten_all()?.to_vec1::<half::bf16>()?, host);
                }
                assert_eq!(
                    table.reclaimable_weight_cache_bytes(),
                    Some(
                        resident_bytes(ROWS, HEAD_DIM, PLANNED_RESIDENT_BITS)
                            + RESIDENT_RESERVE_BYTES
                    )
                );
            }
            table.resident = Some(ResidentTable {
                data,
                scales: Some(scales),
                format: crate::cuda::qwen4_exp::PleTableFormat::Q4,
            });
            assert_eq!(table.reclaimable_weight_cache_bytes(), None);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn iq4_nl_gather_matches_host_dequant() -> Result<()> {
        const HEAD_DIM: usize = 160;
        const ROWS: usize = 37;
        let device = Device::new_cuda(0)?;
        let row_bytes = iq4_nl_row_bytes(HEAD_DIM);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut table = vec![0u8; ROWS * row_bytes];
        for block in table.as_chunks_mut::<IQ4_NL_BLOCK_BYTES>().0 {
            for byte in block.iter_mut() {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *byte = state as u8;
            }
            let scale = half::f16::from_f32(((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5);
            block[..2].copy_from_slice(&scale.to_le_bytes());
        }
        let lookups: Vec<i64> = vec![0, 36, 5, 5, 17, 1, 30, 22];
        let data = Tensor::from_slice(&table, (ROWS, row_bytes), &device)?;
        let rows = Tensor::from_slice(&lookups, (2, 4), &device)?;
        let got = crate::cuda::qwen4_exp::ple_gather_quant(
            &rows,
            &data,
            None,
            HEAD_DIM,
            crate::cuda::qwen4_exp::PleTableFormat::Iq4Nl,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        let mut expected = vec![0f32; lookups.len() * HEAD_DIM];
        for (dst, &row) in expected
            .as_chunks_mut::<HEAD_DIM>()
            .0
            .iter_mut()
            .zip(&lookups)
        {
            let row = row as usize;
            dequantize_iq4_nl_row(&table[row * row_bytes..(row + 1) * row_bytes], dst);
        }
        for (g, e) in got.iter().zip(&expected) {
            assert_eq!(*g, half::bf16::from_f32(*e).to_f32());
        }
        if device_reads_pageable_memory(&device)? {
            let map = mapped_bytes(&table)?;
            let actual = crate::cuda::qwen4_exp::ple_gather_quant_mapped(
                &rows,
                &map,
                None,
                HEAD_DIM,
                crate::cuda::qwen4_exp::PleTableFormat::Iq4Nl,
            )?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            assert_eq!(actual, got);
            let table = mapped_iq4_table(&table, ROWS, HEAD_DIM, &device)?;
            for layout in [
                crate::cuda::qwen4_exp::TokenLayout::rectangular(2, 1),
                crate::cuda::qwen4_exp::TokenLayout::from_host(&[(0, 2)], &[2], 2, &device)?,
            ] {
                let actual = table.gather_mapped(&rows, &layout)?;
                assert!(actual.device().same_device(&device));
                assert_eq!(actual.dims(), &[2, 4 * HEAD_DIM]);
                assert_eq!(
                    actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?,
                    got
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn planned_table_matches_checkpoint_rows() {
        // The checkpoint pads the hashed rows up to a multiple of its shard count
        const CHECKPOINT_ROWS: usize = 320_001_536;
        const CHECKPOINT_SHARDS: usize = 128;
        let cfg = flash_next_text_config(4);
        let ple = cfg.ple().unwrap().unwrap();
        let row_bytes = ple.head_dim() / 2 + ple.head_dim() / RESIDENT_GROUP * 2;
        let rows = planned_resident_table_bytes(&cfg, &ple) / row_bytes;
        assert_eq!(
            rows.div_ceil(CHECKPOINT_SHARDS) * CHECKPOINT_SHARDS,
            CHECKPOINT_ROWS
        );
    }

    #[test]
    fn hashed_rows_match_reference() {
        let cfg = flash_next_text_config(4);
        let ple = cfg.ple().unwrap().unwrap();
        let hash = PleHash::new(&cfg, &ple, 0);
        assert_eq!(
            hash.multipliers,
            [23703573157769, 20109073645365, 8052911324071]
        );
        assert_eq!(&hash.vocab_sizes[..3], &[20000003, 20000023, 20000033]);
        assert_eq!(&hash.offsets[..3], &[0, 20000003, 40000026]);

        // Rows 0, 1, 7, 8 and 15 per token from `Qwen4ExpTextNGramEmbedding` with no history
        let tokens = [9707, 11, 1879, 248044, 553, 1024, 7, 248044, 248044, 42];
        let expected = [
            [16410909, 39682429, 152932897, 169641436, 300984276],
            [18158303, 36390029, 159566246, 175132467, 307699687],
            [6380558, 26411572, 148942556, 164226950, 312121804],
            [18827606, 27700912, 142052741, 163793416, 318943450],
            [12851972, 31952205, 159255184, 173644045, 312013235],
            [12082042, 20450345, 149364883, 165982112, 316678718],
            [17604418, 20085892, 154993589, 175674761, 313648512],
            [10204458, 27984170, 158727320, 162586317, 310194357],
            [9663979, 26558231, 151022725, 170054832, 300986548],
            [4064167, 21428206, 156147803, 175176707, 314347972],
        ];
        let mut rows = vec![0u64; hash.num_heads()];
        for (local, expected) in expected.iter().enumerate() {
            hash.rows(&tokens, local, &[None, None], &mut rows);
            let picked = [rows[0], rows[1], rows[7], rows[8], rows[15]];
            assert_eq!(&picked, expected, "token {local}");
        }

        // A chunk boundary: the history carries the context the reference sees in-sequence
        let mut split = vec![0u64; hash.num_heads()];
        hash.rows(&tokens[5..], 0, &[Some(248044), Some(553)], &mut split);
        hash.rows(&tokens, 5, &[None, None], &mut rows);
        assert_eq!(split, rows);
    }
}
