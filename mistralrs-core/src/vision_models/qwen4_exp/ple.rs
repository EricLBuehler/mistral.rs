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
}

struct PleTable {
    source: TableSource,
    head_dim: usize,
    #[cfg(feature = "cuda")]
    device_index: Option<(Tensor, Tensor)>,
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
    fn load(vb: &ShardedVarBuilder, head_dim: usize, device: &Device) -> Result<Self> {
        if let Some(archive) = vb.raw_gguf() {
            return Self::load_gguf(archive, head_dim);
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
            table.device_index = Some((
                Tensor::from_vec(ptrs, n, device)?,
                Tensor::from_vec(starts, n, device)?,
            ));
            tracing::debug!("Qwen4-Exp PLE rows can be gathered through pageable memory access");
        }
        #[cfg(not(feature = "cuda"))]
        let _ = device;
        Ok(table)
    }

    fn load_gguf(archive: Arc<GgufArchive>, head_dim: usize) -> Result<Self> {
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
        Ok(Self {
            source: TableSource::Gguf { archive, rows },
            head_dim,
            #[cfg(feature = "cuda")]
            device_index: None,
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
        }
    }

    #[cfg(feature = "cuda")]
    fn fallback_path(&self) -> &'static str {
        if self.device_index.is_some() {
            "the GPU reads rows from the memory map (slow)"
        } else {
            "rows are gathered on the host every step (slow)"
        }
    }

    /// Move the whole table into device memory when it fits beside a working reserve: GGUF IQ4_NL rows
    /// are copied as is, safetensors rows are quantized to 8 or 4 bits.
    #[cfg(feature = "cuda")]
    fn make_resident(&mut self, device: &Device) -> Result<()> {
        if !self.head_dim.is_multiple_of(RESIDENT_GROUP) {
            tracing::warn!(
                head_dim = self.head_dim,
                "Qwen4-Exp PLE table rows are not a multiple of {RESIDENT_GROUP}, so it stays out of device memory; {}",
                self.fallback_path()
            );
            return Ok(());
        }
        let rows = self.total_rows()?;
        let available = crate::MemoryUsage.query(device)?.available();
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
            data.slice_set(
                &Tensor::from_slice(src, (src.len() / row_bytes, row_bytes), device)?,
                0,
                chunk * RESIDENT_COPY_ROWS,
            )?;
            evict_page_cache(src);
        }
        Ok(ResidentTable {
            data,
            scales: None,
            format: crate::cuda::qwen4_exp::PleTableFormat::Iq4Nl,
        })
    }

    #[cfg(feature = "cuda")]
    fn quantize_shards(&self, args: QuantizeShards<'_>) -> Result<ResidentTable> {
        use rayon::prelude::*;
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
        let head_dim = self.head_dim;
        let decode = match dtype {
            DType::F16 => |b: [u8; 2]| half::f16::from_le_bytes(b).to_f32(),
            _ => |b: [u8; 2]| half::bf16::from_le_bytes(b).to_f32(),
        };
        for (shard, name) in names.iter().enumerate() {
            let view = raw.get(name)?;
            let shard_rows = view.shape()[0];
            let src = view.data();
            let mut q = vec![0u8; shard_rows * row_bytes];
            let mut sc = vec![half::f16::ZERO; shard_rows * groups];
            q.par_chunks_mut(row_bytes)
                .zip(sc.par_chunks_mut(groups))
                .enumerate()
                .for_each(|(row, (q_row, sc_row))| {
                    let base = row * head_dim * 2;
                    let value = |i: usize| decode([src[base + 2 * i], src[base + 2 * i + 1]]);
                    let q_max = if bits == 8 { 127.0 } else { 7.0 };
                    for (g, scale) in sc_row.iter_mut().enumerate() {
                        let range = g * RESIDENT_GROUP..(g + 1) * RESIDENT_GROUP;
                        let amax = range.clone().map(|i| value(i).abs()).fold(0f32, f32::max);
                        let s = half::f16::from_f32(amax / q_max);
                        *scale = s;
                        let inv = if s.to_f32() > 0.0 {
                            1.0 / s.to_f32()
                        } else {
                            0.0
                        };
                        for i in range {
                            let v = (value(i) * inv).round().clamp(-q_max, q_max) as i32;
                            if bits == 8 {
                                q_row[i] = v as i8 as u8;
                            } else {
                                q_row[i / 2] |= ((v + 8) as u8) << (4 * (i % 2));
                            }
                        }
                    }
                });
            let start_row = starts[shard];
            data.slice_set(
                &Tensor::from_vec(q, (shard_rows, row_bytes), device)?,
                0,
                start_row,
            )?;
            scales.slice_set(
                &Tensor::from_vec(sc, (shard_rows, groups), device)?,
                0,
                start_row,
            )?;
            evict_page_cache(src);
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
        }
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

/// Drop the page cache behind a mapped range; on unified memory the allocator can't reclaim it.
#[cfg(feature = "cuda")]
fn evict_page_cache(data: &[u8]) {
    #[cfg(target_os = "linux")]
    {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(1) as usize;
        let start = data.as_ptr() as usize / page * page;
        let len = data.as_ptr() as usize + data.len() - start;
        unsafe {
            libc::madvise(start as *mut libc::c_void, len, libc::MADV_PAGEOUT);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = data;
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

    /// Quantize the n-gram table into device memory if it fits; call once the rest of the model is loaded.
    pub(super) fn make_table_resident(&mut self, device: &Device) -> Result<()> {
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            return self.table.make_resident(device);
        }
        let _ = device;
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
                (None, Some((ptrs, starts))) => crate::cuda::qwen4_exp::ple_gather(
                    &rows,
                    ptrs,
                    starts,
                    self.table.head_dim,
                    self.table.shard_dtype(),
                )?,
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
