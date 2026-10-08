use candle_core::{DType, MetalStorage, Result, Shape, Storage, Tensor};
use candle_metal_kernels::metal::{Buffer, ConstantValues, Value};
use objc2_metal::MTLSize;

use crate::metal::kernels::Kernels;

const MAX_HEAD_DIM: usize = 4096;
const MATRIX_VALUE_DIM: usize = 512;
const MATRIX_QUERY_TILE: usize = 8;
const WIDE_QUERY_TILE: usize = 32;
const DIRECT_QUERY_TILE: usize = 8;
const WIDE_NATIVE_HEAD_DIM: usize = 256;
const WIDE_FLOAT_HEAD_DIM: usize = 128;
const MIN_TYPED_HEAD_DIM: usize = 64;
const SIMD_WIDTH: usize = 32;
const MATRIX_THREADS: usize = 128;
const DIRECT_THREADS: usize = 256;
const ROW_FIELDS: usize = 9;
const RANGE_FIELDS: usize = 2;
const METADATA_CHUNK_WORDS: usize = 256;
const INLINE_METADATA_BYTES: usize = 4096;
const ALLOCATION_GRANULE: usize = 16384;
const ABSENT_DTYPE: u64 = u64::MAX;
const SPECIALIZATION_CONSTANT_BASE: usize = 100;
const SOFTCAP_CONSTANT: usize = 119;
const MULTIMODAL_CONSTANT: usize = 120;
const BLOCK_SHIFT_CONSTANT: usize = 121;
const NON_POWER_OF_TWO_SHIFT: u16 = u64::BITS as u16;

#[repr(C)]
#[derive(Default)]
struct PrefillParams {
    query_strides: [u64; 4],
    key_strides: [u64; 5],
    value_strides: [u64; 4],
    table_strides: [u64; 2],
    mask_strides: [u64; 4],
    mask_shape: [u64; 4],
    heads: u64,
    kv_heads: u64,
    key_dim: u64,
    value_dim: u64,
    block_size: u64,
    packed_x: u64,
    query_axis: u64,
    hnd: u64,
    packed_queries: u64,
    causal: u64,
    window_left: u64,
    query_dtype: u64,
    cache_dtype: u64,
    mask_dtype: u64,
    alibi_dtype: u64,
    sink_dtype: u64,
    packed_mask: u64,
    batch: u64,
    blocks: u64,
    has_scales: u64,
    alibi_stride: u64,
    sink_stride: u64,
    window_right: u64,
    attention_chunk_size: u64,
}

impl PrefillParams {
    fn constants(&self, softcap: bool, ranges: bool) -> Result<ConstantValues> {
        let geometry = [
            self.heads,
            self.kv_heads,
            self.key_dim,
            self.value_dim,
            self.block_size,
            self.packed_x,
            self.hnd,
            self.packed_queries,
            self.causal,
            self.window_left,
            self.query_dtype,
            self.cache_dtype,
            self.mask_dtype,
            self.alibi_dtype,
            self.sink_dtype,
            self.packed_mask,
            self.has_scales,
            self.window_right,
            self.attention_chunk_size,
        ];
        let mut constants = geometry
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                Ok((
                    SPECIALIZATION_CONSTANT_BASE + index,
                    Value::USize(usize::try_from(value).map_err(candle_core::Error::wrap)?),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        constants.push((SOFTCAP_CONSTANT, Value::Bool(softcap)));
        constants.push((MULTIMODAL_CONSTANT, Value::Bool(ranges)));
        let block_shift = if self.block_size.is_power_of_two() {
            self.block_size.trailing_zeros() as u16
        } else {
            NON_POWER_OF_TWO_SHIFT
        };
        constants.push((BLOCK_SHIFT_CONSTANT, Value::U16(block_shift)));
        Ok(ConstantValues::new(constants))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MetalPagedPrefillShape {
    pub num_seqs: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub max_query_len: usize,
    pub total_query_tokens: usize,
    pub max_context_len: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub block_size: usize,
    pub num_multimodal_ranges: usize,
    pub query_dtype: DType,
    pub cache_dtype: DType,
    pub packed_queries: bool,
    pub hnd: bool,
}

impl MetalPagedPrefillShape {
    fn storage_rows(self) -> Result<usize> {
        if self.packed_queries {
            Ok(self.total_query_tokens)
        } else {
            checked_product(&[self.num_seqs, self.max_query_len])
        }
    }

    pub fn query_copy_workspace_bytes(self) -> Result<usize> {
        rounded_allocation(checked_product(&[
            self.storage_rows()?,
            self.num_heads,
            self.key_head_dim,
            self.query_dtype.size_in_bytes(),
        ])?)
    }

    pub fn output_copy_workspace_bytes(self) -> Result<usize> {
        rounded_allocation(checked_product(&[
            self.storage_rows()?,
            self.num_heads,
            self.value_head_dim,
            self.query_dtype.size_in_bytes(),
        ])?)
    }

    pub fn supports(self) -> bool {
        self.num_seqs > 0
            && self.num_heads > 0
            && self.num_kv_heads > 0
            && self.num_heads.is_multiple_of(self.num_kv_heads)
            && self.max_query_len > 0
            && self.total_query_tokens >= self.max_query_len
            && self.max_context_len > 0
            && (1..=MAX_HEAD_DIM).contains(&self.key_head_dim)
            && (1..=MAX_HEAD_DIM).contains(&self.value_head_dim)
            && self.block_size > 0
            && matches!(self.query_dtype, DType::F16 | DType::BF16 | DType::F32)
            && matches!(
                self.cache_dtype,
                DType::F16 | DType::BF16 | DType::F32 | DType::F8E4M3
            )
    }

    /// Includes output and reusable metadata backing; caller-owned masks and cache writes are excluded.
    pub fn workspace_bytes(self) -> Result<usize> {
        if !self.supports() {
            candle_core::bail!("unsupported Metal paged prefill shape {self:?}");
        }
        let rows = if self.packed_queries {
            self.total_query_tokens
        } else {
            checked_product(&[self.num_seqs, self.max_query_len])?
        };
        let output = checked_product(&[
            rows,
            self.num_heads,
            self.value_head_dim,
            self.query_dtype.size_in_bytes(),
        ])?;
        let metadata = self
            .num_seqs
            .checked_mul(ROW_FIELDS)
            .and_then(|rows| {
                self.num_multimodal_ranges
                    .checked_mul(RANGE_FIELDS)
                    .and_then(|ranges| rows.checked_add(ranges))
            })
            .and_then(|words| words.checked_mul(size_of::<u64>()))
            .ok_or_else(|| candle_core::Error::msg("paged prefill metadata size overflow"))?;
        rounded_allocation(output)?
            .checked_add(rounded_allocation(metadata)?)
            .ok_or_else(|| candle_core::Error::msg("paged prefill workspace size overflow"))
    }
}

pub struct MetalPagedPrefillRequest<'a> {
    pub query: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub block_tables: &'a Tensor,
    pub query_lens: &'a [usize],
    pub context_lens: &'a [usize],
    pub query_start_positions: &'a [usize],
    pub key_start_positions: &'a [usize],
    pub packed_queries: bool,
    pub causal: bool,
    pub sliding_window: Option<usize>,
    pub window_right: Option<usize>,
    pub attention_chunk_size: Option<usize>,
    pub multimodal_ranges: Option<&'a [Vec<(usize, usize)>]>,
    pub additive_mask: Option<&'a Tensor>,
    pub packed_mask: bool,
    pub mask_key_offsets: Option<&'a [usize]>,
    pub alibi_slopes: Option<&'a Tensor>,
    pub sinks: Option<&'a Tensor>,
    pub k_scale: Option<&'a Tensor>,
    pub v_scale: Option<&'a Tensor>,
    pub scale: f32,
    pub softcap: Option<f32>,
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
        .ok_or_else(|| candle_core::Error::msg("paged prefill size overflow"))
}

fn rounded_allocation(bytes: usize) -> Result<usize> {
    bytes
        .checked_next_power_of_two()
        .map(|bytes| bytes.max(ALLOCATION_GRANULE))
        .ok_or_else(|| candle_core::Error::msg("paged prefill allocation size overflow"))
}

fn dtype_code(dtype: DType) -> Result<u64> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        DType::F8E4M3 => Ok(3),
        _ => candle_core::bail!("unsupported paged prefill dtype {dtype:?}"),
    }
}

fn binding(tensor: &Tensor, query: &Tensor) -> Result<(Buffer, usize)> {
    if !tensor.device().same_device(query.device()) {
        candle_core::bail!("paged prefill tensors must share one Metal device");
    }
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Metal(storage) = &*storage else {
        candle_core::bail!("paged prefill requires Metal tensors");
    };
    Ok((
        storage.buffer().clone(),
        layout.start_offset() * tensor.dtype().size_in_bytes(),
    ))
}

fn optional_binding(tensor: Option<&Tensor>, query: &Tensor) -> Result<(Buffer, usize)> {
    binding(tensor.unwrap_or(query), query)
}

fn feature_code(tensor: Option<&Tensor>, heads: usize) -> Result<(u64, u64)> {
    let Some(tensor) = tensor else {
        return Ok((ABSENT_DTYPE, 0));
    };
    if tensor.dims() != [heads] || !matches!(tensor.dtype(), DType::F16 | DType::BF16 | DType::F32)
    {
        candle_core::bail!(
            "paged prefill head features require a floating vector of length {heads}"
        );
    }
    Ok((dtype_code(tensor.dtype())?, tensor.stride()[0] as u64))
}

/// Explicit mask key offsets override local or packed key origins; packed query indices remain concatenated.
pub fn metal_paged_prefill(request: MetalPagedPrefillRequest<'_>) -> Result<Tensor> {
    let r = request;
    let (query_batch, heads, query_axis, key_dim) = r.query.dims4()?;
    let batch = r.query_lens.len();
    if batch == 0
        || r.context_lens.len() != batch
        || r.query_start_positions.len() != batch
        || r.key_start_positions.len() != batch
        || r.query_lens.contains(&0)
        || r.context_lens.contains(&0)
        || r.multimodal_ranges
            .is_some_and(|ranges| ranges.len() != batch)
    {
        candle_core::bail!("invalid paged prefill sequence metadata");
    }
    let total_q = r
        .query_lens
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b))
        .ok_or_else(|| candle_core::Error::msg("paged prefill query count overflow"))?;
    let max_q = *r.query_lens.iter().max().unwrap();
    let max_k = *r.context_lens.iter().max().unwrap();
    if (r.packed_queries && (query_batch != 1 || query_axis != total_q))
        || (!r.packed_queries && (query_batch != batch || query_axis != max_q))
    {
        candle_core::bail!("paged prefill query shape does not match its lengths");
    }
    let (blocks, kv_heads, block_size, cache_key_dim, x, hnd) = match r.key_cache.dims() {
        &[blocks, heads, block, dim] => (blocks, heads, block, dim, 1, true),
        &[blocks, heads, dim, block, x] => {
            (blocks, heads, block, checked_product(&[dim, x])?, x, false)
        }
        dims => candle_core::bail!("unsupported paged prefill key cache shape {dims:?}"),
    };
    let (value_blocks, value_heads, value_dim, value_block) = match r.value_cache.dims4()? {
        (blocks, heads, block, dim) if hnd => (blocks, heads, dim, block),
        (blocks, heads, dim, block) => (blocks, heads, dim, block),
    };
    if blocks != value_blocks
        || kv_heads != value_heads
        || block_size != value_block
        || cache_key_dim != key_dim
        || r.key_cache.dtype() != r.value_cache.dtype()
    {
        candle_core::bail!("paged prefill query and cache geometry do not match");
    }
    if r.mask_key_offsets
        .is_some_and(|offsets| offsets.len() != batch)
    {
        candle_core::bail!("paged prefill mask offsets must match the sequence count");
    }
    let mask_lengths = r
        .context_lens
        .iter()
        .enumerate()
        .map(|(row, len)| {
            len.checked_add(r.mask_key_offsets.map_or(0, |offsets| offsets[row]))
                .ok_or_else(|| candle_core::Error::msg("paged prefill mask extent overflow"))
        })
        .collect::<Result<Vec<_>>>()?;
    let total_mask_k = mask_lengths
        .iter()
        .try_fold(0usize, |a, b| a.checked_add(*b))
        .ok_or_else(|| candle_core::Error::msg("paged prefill mask extent overflow"))?;
    let max_mask_k = *mask_lengths.iter().max().unwrap();
    let num_ranges = r
        .multimodal_ranges
        .map_or(0, |ranges| ranges.iter().map(Vec::len).sum());
    let shape = MetalPagedPrefillShape {
        num_seqs: batch,
        num_heads: heads,
        num_kv_heads: kv_heads,
        max_query_len: max_q,
        total_query_tokens: total_q,
        max_context_len: max_k,
        key_head_dim: key_dim,
        value_head_dim: value_dim,
        block_size,
        num_multimodal_ranges: num_ranges,
        query_dtype: r.query.dtype(),
        cache_dtype: r.key_cache.dtype(),
        packed_queries: r.packed_queries,
        hnd,
    };
    shape.workspace_bytes()?;
    let (table_batch, table_blocks) = r.block_tables.dims2()?;
    if table_batch != batch
        || !matches!(r.block_tables.dtype(), DType::U32 | DType::I32)
        || table_blocks < max_k.div_ceil(block_size)
        || blocks == 0
    {
        candle_core::bail!("invalid paged prefill block table");
    }
    if r.sliding_window == Some(0)
        || r.window_right == Some(0)
        || r.attention_chunk_size == Some(0)
        || !r.scale.is_finite()
        || r.softcap.is_some_and(|cap| !cap.is_finite() || cap <= 0.0)
    {
        candle_core::bail!("invalid paged prefill scale, softcap, or window");
    }
    super::validate_kv_cache_scales(r.key_cache.dtype(), r.k_scale, r.v_scale, "paged prefill")?;
    let mut metadata = vec![0u64; batch * ROW_FIELDS];
    let mut q_offset = 0;
    let mut k_offset = 0;
    for row in 0..batch {
        let query_end = r.query_start_positions[row].checked_add(r.query_lens[row]);
        let key_end = r.key_start_positions[row].checked_add(r.context_lens[row]);
        if query_end.is_none()
            || key_end.is_none()
            || r.window_right
                .is_some_and(|right| query_end.and_then(|end| end.checked_add(right)).is_none())
        {
            candle_core::bail!("paged prefill absolute position overflow");
        }
        let ranges = r
            .multimodal_ranges
            .map_or(&[][..], |ranges| ranges[row].as_slice());
        let range_offset = metadata.len();
        for &(start, end) in ranges {
            if end <= start {
                candle_core::bail!("invalid paged prefill multimodal range");
            }
            metadata.extend_from_slice(&[start as u64, end as u64]);
        }
        metadata[row * ROW_FIELDS..(row + 1) * ROW_FIELDS].copy_from_slice(&[
            r.query_lens[row] as u64,
            r.context_lens[row] as u64,
            r.query_start_positions[row] as u64,
            r.key_start_positions[row] as u64,
            q_offset as u64,
            k_offset as u64,
            range_offset as u64,
            ranges.len() as u64,
            r.mask_key_offsets
                .map_or(if r.packed_mask { k_offset } else { 0 }, |offsets| {
                    offsets[row]
                }) as u64,
        ]);
        q_offset += r.query_lens[row];
        k_offset += r.context_lens[row];
    }
    let mut p = PrefillParams::default();
    for (target, stride) in p.query_strides.iter_mut().zip(r.query.stride()) {
        *target = *stride as u64;
    }
    for (target, stride) in p.key_strides.iter_mut().zip(r.key_cache.stride()) {
        *target = *stride as u64;
    }
    for (target, stride) in p.value_strides.iter_mut().zip(r.value_cache.stride()) {
        *target = *stride as u64;
    }
    p.table_strides[0] = r.block_tables.stride()[0] as u64;
    p.table_strides[1] = r.block_tables.stride()[1] as u64;
    p.mask_dtype = ABSENT_DTYPE;
    if let Some(mask) = r.additive_mask {
        if !(2..=4).contains(&mask.rank())
            || !matches!(mask.dtype(), DType::F16 | DType::BF16 | DType::F32)
        {
            candle_core::bail!(
                "paged prefill additive masks must be rank 2, 3, or 4 floating tensors"
            );
        }
        let expected = if r.packed_mask {
            [
                1,
                heads,
                total_q,
                if r.mask_key_offsets.is_some() {
                    max_mask_k
                } else {
                    total_mask_k
                },
            ]
        } else {
            [batch, heads, max_q, max_mask_k]
        };
        let mut dims = [1usize; 4];
        let mut strides = [0usize; 4];
        dims[4 - mask.rank()..].copy_from_slice(mask.dims());
        strides[4 - mask.rank()..].copy_from_slice(mask.stride());
        for dim in 0..4 {
            if dims[dim] != 1
                && (if dim == 3 {
                    dims[dim] < expected[dim]
                } else {
                    dims[dim] != expected[dim]
                })
            {
                candle_core::bail!(
                    "paged prefill mask {:?} cannot broadcast to {expected:?}",
                    mask.dims()
                );
            }
            p.mask_strides[dim] = strides[dim] as u64;
            p.mask_shape[dim] = dims[dim] as u64;
        }
        p.mask_dtype = dtype_code(mask.dtype())?;
    }
    let (alibi_dtype, alibi_stride) = feature_code(r.alibi_slopes, heads)?;
    let (sink_dtype, sink_stride) = feature_code(r.sinks, heads)?;
    p.heads = heads as u64;
    p.kv_heads = kv_heads as u64;
    p.key_dim = key_dim as u64;
    p.value_dim = value_dim as u64;
    p.block_size = block_size as u64;
    p.packed_x = x as u64;
    p.query_axis = query_axis as u64;
    p.hnd = u64::from(hnd);
    p.packed_queries = u64::from(r.packed_queries);
    p.causal = u64::from(r.causal);
    p.window_left = r.sliding_window.unwrap_or(0) as u64;
    p.query_dtype = dtype_code(r.query.dtype())?;
    p.cache_dtype = dtype_code(r.key_cache.dtype())?;
    p.alibi_dtype = alibi_dtype;
    p.sink_dtype = sink_dtype;
    p.packed_mask = u64::from(r.packed_mask);
    p.batch = batch as u64;
    p.blocks = blocks as u64;
    p.has_scales = u64::from(r.k_scale.is_some());
    p.alibi_stride = alibi_stride;
    p.sink_stride = sink_stride;
    p.window_right = r.window_right.unwrap_or(0) as u64;
    p.attention_chunk_size = r.attention_chunk_size.unwrap_or(0) as u64;
    let q = binding(r.query, r.query)?;
    let k = binding(r.key_cache, r.query)?;
    let v = binding(r.value_cache, r.query)?;
    let table = binding(r.block_tables, r.query)?;
    let mask = optional_binding(r.additive_mask, r.query)?;
    let alibi = optional_binding(r.alibi_slopes, r.query)?;
    let sinks = optional_binding(r.sinks, r.query)?;
    let k_scale = optional_binding(r.k_scale, r.query)?;
    let v_scale = optional_binding(r.v_scale, r.query)?;
    let device = r.query.device().as_metal_device()?;
    let out_shape = Shape::from((query_batch, heads, query_axis, value_dim));
    let out = device.new_buffer(
        out_shape.elem_count(),
        r.query.dtype(),
        "paged-prefill-output",
    )?;
    let kernels = Kernels::new();
    let meta = if size_of_val(metadata.as_slice()) > INLINE_METADATA_BYTES {
        let meta = device.new_buffer(metadata.len() * 2, DType::U32, "paged-prefill-metadata")?;
        let copy_pipeline = kernels
            .load_pipeline(device.device(), "paged_prefill_metadata_copy".into())
            .map_err(candle_core::Error::wrap)?;
        for (chunk_index, chunk) in metadata.chunks(METADATA_CHUNK_WORDS).enumerate() {
            let guard = device.command_encoder()?;
            let encoder = guard.as_ref();
            encoder.set_compute_pipeline_state(&copy_pipeline);
            encoder.set_bytes_directly(0, size_of_val(chunk), chunk.as_ptr().cast());
            encoder.set_output_buffer(
                1,
                Some(&meta),
                chunk_index * METADATA_CHUNK_WORDS * size_of::<u64>(),
            );
            encoder.set_bytes(2, &(chunk.len() as u64));
            encoder.dispatch_threads(
                MTLSize {
                    width: chunk.len(),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: SIMD_WIDTH,
                    height: 1,
                    depth: 1,
                },
            );
        }
        Some(meta)
    } else {
        None
    };
    let matrix = value_dim <= MATRIX_VALUE_DIM;
    let typed = matrix && key_dim <= MATRIX_VALUE_DIM;
    let native_dtype = r.query.dtype() == r.key_cache.dtype()
        && matches!(r.query.dtype(), DType::F16 | DType::BF16)
        && r.k_scale.is_none();
    let max_dim = key_dim.max(value_dim);
    let wide = max_q > 1
        && max_dim
            <= if native_dtype {
                WIDE_NATIVE_HEAD_DIM
            } else {
                WIDE_FLOAT_HEAD_DIM
            };
    let direct = native_dtype && max_dim > WIDE_NATIVE_HEAD_DIM && max_dim <= MATRIX_VALUE_DIM;
    let operand_dtype = if native_dtype {
        if r.query.dtype() == DType::F16 {
            "f16"
        } else {
            "bf16"
        }
    } else {
        "f32"
    };
    let bucket = max_dim.next_power_of_two().max(MIN_TYPED_HEAD_DIM);
    let name = if wide {
        format!("paged_prefill_q32_{operand_dtype}_{bucket}")
    } else if direct {
        format!("paged_prefill_direct_{operand_dtype}_{bucket}")
    } else if typed {
        format!("paged_prefill_typed_{operand_dtype}_{bucket}")
    } else if matrix {
        "paged_prefill_matrix".into()
    } else {
        "paged_prefill_vector".into()
    };
    let query_tile = if wide {
        WIDE_QUERY_TILE
    } else if direct {
        DIRECT_QUERY_TILE
    } else if matrix {
        MATRIX_QUERY_TILE
    } else {
        1
    };
    let pipeline = kernels
        .load_pipeline_with_constants(
            device.device(),
            name.clone(),
            Some(p.constants(r.softcap.is_some(), shape.num_multimodal_ranges > 0)?),
        )
        .map_err(candle_core::Error::wrap)?;
    let guard = device.command_encoder()?;
    let encoder = guard.as_ref();
    encoder.set_label(&name);
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_bytes(0, &p);
    encoder.set_bytes(1, &[r.scale, r.softcap.unwrap_or(0.0)]);
    for (index, (buffer, offset)) in [q, k, v, table, mask, alibi, sinks, k_scale, v_scale]
        .iter()
        .enumerate()
    {
        encoder.set_input_buffer(index + 2, Some(buffer), *offset);
    }
    if let Some(meta) = &meta {
        encoder.set_input_buffer(11, Some(meta), 0);
    } else {
        encoder.set_bytes_directly(
            11,
            size_of_val(metadata.as_slice()),
            metadata.as_ptr().cast(),
        );
    }
    encoder.set_output_buffer(12, Some(&out), 0);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: max_q.div_ceil(query_tile),
            height: heads,
            depth: batch,
        },
        MTLSize {
            width: if direct {
                DIRECT_THREADS
            } else if matrix {
                MATRIX_THREADS
            } else {
                SIMD_WIDTH
            },
            height: 1,
            depth: 1,
        },
    );
    drop(guard);
    Ok(Tensor::from((
        Storage::Metal(MetalStorage::new(
            out,
            device.clone(),
            out_shape.elem_count(),
            r.query.dtype(),
        )),
        out_shape,
    )))
}
