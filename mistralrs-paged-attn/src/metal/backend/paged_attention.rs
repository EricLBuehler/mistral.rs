use std::sync::OnceLock;

use candle_core::{
    backend::BackendStorage, CpuStorage, DType, Layout, MetalStorage, Result, Shape, Storage,
    Tensor,
};

use crate::metal::{
    backend::{validate_kv_cache_scales, CacheShape},
    kernels::{self, PagedAttentionDType},
};

pub const METAL_PAGED_CACHE_VECTOR_BYTES: usize = 16;
const SPECIALIZED_PACKED_X: usize = METAL_PAGED_CACHE_VECTOR_BYTES / 2;
const GENERIC_PARTITION_SIZE: usize = 512;
const GENERIC_V1_HEAD_THRESHOLD: usize = 512;
const METAL_PAGED_KERNEL_ENV: &str = "MISTRALRS_METAL_PAGED_KERNEL";

#[derive(Clone, Copy, PartialEq, Eq)]
enum MetalPagedKernel {
    Auto,
    Grouped,
    Legacy,
}

fn metal_paged_kernel() -> MetalPagedKernel {
    static KERNEL: OnceLock<MetalPagedKernel> = OnceLock::new();
    *KERNEL.get_or_init(
        || match std::env::var(METAL_PAGED_KERNEL_ENV).ok().as_deref() {
            None | Some("auto") => MetalPagedKernel::Auto,
            Some("grouped") => MetalPagedKernel::Grouped,
            Some("legacy") => MetalPagedKernel::Legacy,
            Some(value) => panic!("invalid {METAL_PAGED_KERNEL_ENV}={value}"),
        },
    )
}

#[derive(Clone, Copy, Debug)]
pub struct MetalPagedAttentionWorkspace {
    pub num_seqs: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_size: usize,
    pub block_size: usize,
    pub max_context_len: usize,
    pub sliding_window: Option<usize>,
    pub dtype: DType,
    pub cache_dtype: DType,
    pub hnd: bool,
    pub packed_x: usize,
    pub has_alibi: bool,
    pub has_sinks: bool,
    pub has_scales: bool,
    pub softcapping: f32,
}

impl MetalPagedAttentionWorkspace {
    fn specialized_variant(self, kernel: MetalPagedKernel) -> Option<kernels::PagedDecodeVariant> {
        let gqa = self.num_heads.checked_div(self.num_kv_heads)?;
        let specialized = kernel != MetalPagedKernel::Legacy
            && matches!(self.dtype, DType::BF16 | DType::F16)
            && self.cache_dtype == self.dtype
            && !self.has_scales
            && !self.has_alibi
            && !self.has_sinks
            && self.softcapping == 1.0
            && self.block_size == kernels::PAGED_DECODE_BLOCK_SIZE
            && (self.hnd || self.packed_x == SPECIALIZED_PACKED_X)
            && (kernels::PAGED_DECODE_HEAD_SIZES.contains(&self.head_size)
                || (self.hnd
                    && self.head_size == kernels::PAGED_DECODE_LARGE_HEAD_SIZE
                    && gqa <= kernels::PAGED_DECODE_LARGE_HEAD_MAX_GQA))
            && self.num_heads.is_multiple_of(self.num_kv_heads)
            && (1..=kernels::PAGED_DECODE_MAX_GQA).contains(&gqa);
        specialized.then(|| {
            if kernel == MetalPagedKernel::Grouped {
                kernels::paged_decode_grouped_variant(
                    self.max_context_len,
                    self.num_seqs,
                    self.num_kv_heads,
                )
            } else if self.hnd {
                kernels::paged_decode_hnd_variant(
                    self.max_context_len,
                    self.num_seqs,
                    self.num_heads,
                    self.num_kv_heads,
                    self.sliding_window,
                    self.head_size,
                )
            } else {
                kernels::paged_decode_variant(
                    self.max_context_len,
                    self.num_seqs,
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_size,
                )
            }
        })
    }

    fn use_v1(self) -> bool {
        (self.max_context_len.div_ceil(GENERIC_PARTITION_SIZE) == 1
            || self.num_seqs.saturating_mul(self.num_heads) > GENERIC_V1_HEAD_THRESHOLD)
            && GENERIC_PARTITION_SIZE.is_multiple_of(self.block_size)
    }

    pub fn bytes(self) -> Result<usize> {
        self.bytes_for_kernel(metal_paged_kernel())
    }

    /// Monotonic workspace bound for every batch and context up to this shape.
    pub fn capacity_bytes(self) -> Result<usize> {
        self.capacity_bytes_for_kernel(metal_paged_kernel())
    }

    fn capacity_bytes_for_kernel(self, kernel: MetalPagedKernel) -> Result<usize> {
        let mut bytes = self
            .bytes_for_kernel(kernel)?
            .max(self.bytes_for_partitions(
                self.max_context_len.div_ceil(GENERIC_PARTITION_SIZE),
                self.dtype,
            )?);
        if self.specialized_variant(kernel).is_some() {
            bytes = bytes.max(
                self.bytes_for_partitions(
                    self.max_context_len
                        .div_ceil(kernels::PAGED_DECODE_SHORT_PARTITION),
                    self.dtype,
                )?,
            );
            let split = Self {
                num_seqs: 1,
                max_context_len: kernels::PAGED_DECODE_SPLIT_MIN_CONTEXT,
                ..self
            };
            if self.max_context_len >= split.max_context_len
                && matches!(
                    split.specialized_variant(kernel),
                    Some(kernels::PagedDecodeVariant::Split)
                )
            {
                bytes = bytes.max(split.bytes_for_kernel(kernel)?);
            }
        }
        Ok(bytes)
    }

    fn bytes_for_kernel(self, kernel: MetalPagedKernel) -> Result<usize> {
        if self.num_seqs == 0
            || self.num_heads == 0
            || self.num_kv_heads == 0
            || self.head_size == 0
            || self.block_size == 0
            || self.max_context_len == 0
        {
            candle_core::bail!("invalid Metal paged attention workspace shape");
        }
        metal_buffer_bytes(&[self.num_seqs, self.num_heads, self.head_size], self.dtype)?;
        let (partitions, partial_dtype) = match self.specialized_variant(kernel) {
            Some(variant) => (
                variant.num_partitions(self.max_context_len),
                if matches!(variant, kernels::PagedDecodeVariant::Split) {
                    DType::F32
                } else {
                    self.dtype
                },
            ),
            None if self.use_v1() => (1, self.dtype),
            None => (
                self.max_context_len.div_ceil(GENERIC_PARTITION_SIZE),
                self.dtype,
            ),
        };
        self.bytes_for_partitions(partitions, partial_dtype)
    }

    fn bytes_for_partitions(self, partitions: usize, partial_dtype: DType) -> Result<usize> {
        let heads = self
            .num_seqs
            .checked_mul(self.num_heads)
            .ok_or_else(|| candle_core::Error::msg("Metal attention workspace overflow"))?;
        let output = metal_buffer_bytes(&[heads, self.head_size], self.dtype)?;
        let mut bytes = output;
        if partitions > 1 {
            for buffer in [
                metal_buffer_bytes(&[heads, partitions], DType::F32)?,
                metal_buffer_bytes(&[heads, partitions], DType::F32)?,
                metal_buffer_bytes(&[heads, partitions, self.head_size], partial_dtype)?,
            ] {
                bytes = bytes
                    .checked_add(buffer)
                    .ok_or_else(|| candle_core::Error::msg("Metal attention workspace overflow"))?;
            }
        }
        if self.has_sinks {
            bytes = bytes
                .checked_add(metal_buffer_bytes(&[self.num_heads], DType::F32)?)
                .ok_or_else(|| candle_core::Error::msg("Metal attention workspace overflow"))?;
        }
        Ok(bytes)
    }
}

fn metal_buffer_bytes(dimensions: &[usize], dtype: DType) -> Result<usize> {
    dimensions
        .iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dim| bytes.checked_mul(*dim))
        .and_then(usize::checked_next_power_of_two)
        .ok_or_else(|| candle_core::Error::msg("Metal attention buffer size overflow"))
}

struct PagedAttention {
    softmax_scale: f32,
    softcapping: f32,

    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    alibi_slopes: Option<Tensor>,
    max_context_len: usize,
    sliding_window: Option<usize>,

    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    sinks: Option<Tensor>,
}

impl candle_core::CustomOp1 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for paged-attention")
    }

    fn metal_fwd(&self, q: &MetalStorage, q_l: &Layout) -> Result<(MetalStorage, Shape)> {
        validate_kv_cache_scales(
            self.key_cache.dtype(),
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
            "paged_attention",
        )?;
        let ty = match q.dtype() {
            DType::F16 => PagedAttentionDType::F16,
            DType::BF16 => PagedAttentionDType::BF16,
            DType::F32 => PagedAttentionDType::F32,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        };
        let cache_ty = match self.key_cache.dtype() {
            DType::F16 => PagedAttentionDType::F16,
            DType::BF16 => PagedAttentionDType::BF16,
            DType::F32 => PagedAttentionDType::F32,
            DType::F8E4M3 => PagedAttentionDType::F8E4M3,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        };

        let dev = q.device();
        let out_shape = q_l.shape().clone();

        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Metal(kc) => kc,
            _ => candle_core::bail!("key_cache must be a metal tensor"),
        };

        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Metal(vc) => vc,
            _ => candle_core::bail!("value_cache must be a metal tensor"),
        };

        let (bt, bt_l) = self.block_tables.storage_and_layout();
        let bt = match &*bt {
            Storage::Metal(bt) => bt,
            _ => candle_core::bail!("block_tables must be a metal tensor"),
        };

        let (cl, cl_l) = self.context_lens.storage_and_layout();
        let cl = match &*cl {
            Storage::Metal(cl) => cl,
            _ => candle_core::bail!("context_lens must be a metal tensor"),
        };

        let q_rank = q_l.stride().len();

        if q_rank != 3 {
            candle_core::bail!(
                "paged-attention expects `q` tensor to be of rank 3 \
                (q: {q_l:?})"
            )
        }

        let alibi_storage_and_offset = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            let (alibi_s, alibi_s_l) = alibi_slopes.storage_and_layout();
            let alibi_s = match &*alibi_s {
                Storage::Metal(alibi_s) => alibi_s,
                _ => candle_core::bail!("context_lens must be a metal tensor"),
            };
            Some((
                alibi_s.clone(),
                alibi_s_l.start_offset() * alibi_s.dtype().size_in_bytes(),
            ))
        } else {
            None
        };

        let (num_seqs, num_heads, head_size) = q_l.shape().dims3()?;
        if !(head_size == 64
            || head_size == 80
            || head_size == 96
            || head_size == 112
            || head_size == 128
            || head_size == 192
            || head_size == 256
            || head_size == 512)
        {
            candle_core::bail!("`head_size` must be one of 64, 80, 96, 112, 128, 192, 256 or 512");
        }

        let (num_seqs_bt, max_num_blocks_per_seq) = bt_l.shape().dims2()?;

        if num_seqs_bt != num_seqs {
            candle_core::bail!(
                "shape mismatch block_tables {:?}, expected {:?}",
                bt_l.shape(),
                (num_seqs, max_num_blocks_per_seq)
            )
        }

        let cache = CacheShape::new(&self.key_cache, &self.value_cache)?;
        let num_kv_heads = cache.num_heads;
        let block_size = cache.block_size;
        if cache.head_size != head_size || num_heads % num_kv_heads != 0 {
            candle_core::bail!("query and cache head dimensions do not match");
        }
        if self.sliding_window.is_some() && !cache.hnd {
            candle_core::bail!("Metal sliding paged attention requires an HND cache");
        }
        let sliding_window = self.sliding_window.unwrap_or(0) as i32;

        if (num_seqs) != cl_l.shape().dims1()? {
            candle_core::bail!(
                "shape mismatch context_lens {:?}, expected {:?}",
                cl_l.shape(),
                (num_seqs)
            )
        }

        let k_v_scale = if let (Some(k_scale), Some(v_scale)) = (&self.k_scale, &self.v_scale) {
            if k_scale.elem_count() != 1 || v_scale.elem_count() != 1 {
                candle_core::bail!("k_scale and v_scale must be scalars");
            }
            if k_scale.dtype() != DType::F32 || v_scale.dtype() != DType::F32 {
                candle_core::bail!("k_scale and v_scale must be f32");
            }

            let (k_scale, _) = k_scale.storage_and_layout();
            let k_scale = match &*k_scale {
                Storage::Metal(k_scale) => k_scale,
                _ => candle_core::bail!("k_scale must be a metal tensor"),
            };

            let (v_scale, _) = v_scale.storage_and_layout();
            let v_scale = match &*v_scale {
                Storage::Metal(v_scale) => v_scale,
                _ => candle_core::bail!("v_scale must be a metal tensor"),
            };

            Some((k_scale.buffer().clone(), v_scale.buffer().clone()))
        } else {
            None
        };

        let sinks_storage_and_offset = if let Some(sinks) = self.sinks.as_ref() {
            let (s, s_l) = sinks.storage_and_layout();
            let s = match &*s {
                Storage::Metal(s) => s,
                _ => candle_core::bail!("sinks must be a metal tensor"),
            };
            Some((
                s.buffer().clone(),
                s_l.start_offset() * s.dtype().size_in_bytes(),
            ))
        } else {
            None
        };

        let q_stride = q_l.stride()[0];
        let kv_block_stride = kc_l.stride()[0];
        let kv_head_stride = kc_l.stride()[1];

        let workload = MetalPagedAttentionWorkspace {
            num_seqs,
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            max_context_len: self.max_context_len,
            sliding_window: self.sliding_window,
            dtype: q.dtype(),
            cache_dtype: self.key_cache.dtype(),
            hnd: cache.hnd,
            packed_x: cache.x,
            has_alibi: alibi_storage_and_offset.is_some(),
            has_sinks: sinks_storage_and_offset.is_some(),
            has_scales: k_v_scale.is_some(),
            softcapping: self.softcapping,
        };
        let max_num_partitions = self.max_context_len.div_ceil(GENERIC_PARTITION_SIZE);
        let use_v1 = workload.use_v1();

        let elem_count = out_shape.elem_count();

        let encoder = dev.command_encoder()?;
        encoder.set_label("paged-attention");

        let out = dev.new_buffer(elem_count, q.dtype(), "paged-attention-out")?;

        if let Some(variant) = workload.specialized_variant(metal_paged_kernel()) {
            let num_partitions = variant.num_partitions(self.max_context_len);
            let partials_len = num_seqs * num_heads * num_partitions;
            let partial_dtype = if matches!(variant, kernels::PagedDecodeVariant::Split) {
                DType::F32
            } else {
                q.dtype()
            };
            let scratch = (num_partitions > 1)
                .then(|| -> Result<_> {
                    Ok((
                        dev.new_buffer(partials_len, DType::F32, "paged-decode-expsums")?,
                        dev.new_buffer(partials_len, DType::F32, "paged-decode-maxlogits")?,
                        dev.new_buffer(
                            partials_len * head_size,
                            partial_dtype,
                            "paged-decode-tmpout",
                        )?,
                    ))
                })
                .transpose()?;
            let dtype_size = |dtype: DType| dtype.size_in_bytes();
            kernels::call_paged_decode_gqa(
                dev.device(),
                &encoder,
                &kernels::Kernels::new(),
                kernels::PagedDecodeArgs {
                    ty,
                    hnd: cache.hnd,
                    sliding_window,
                    q: (q.buffer(), q_l.start_offset() * dtype_size(q.dtype())),
                    k_cache: (kc.buffer(), kc_l.start_offset() * dtype_size(kc.dtype())),
                    v_cache: (vc.buffer(), vc_l.start_offset() * dtype_size(vc.dtype())),
                    block_tables: (bt.buffer(), bt_l.start_offset() * dtype_size(bt.dtype())),
                    context_lens: (cl.buffer(), cl_l.start_offset() * dtype_size(cl.dtype())),
                    output: &out,
                    partials: scratch.as_ref().map(|(exp_sums, max_logits, tmp_out)| {
                        kernels::PagedDecodePartials {
                            exp_sums,
                            max_logits,
                            tmp_out,
                        }
                    }),
                    scale: self.softmax_scale,
                    variant,
                    max_context_len: self.max_context_len as i32,
                    num_seqs: num_seqs as i32,
                    num_heads: num_heads as i32,
                    num_kv_heads: num_kv_heads as i32,
                    head_size: head_size as i32,
                    max_num_blocks_per_seq: max_num_blocks_per_seq as i32,
                    q_stride: q_stride as i32,
                    kv_block_stride: kv_block_stride as i32,
                    kv_head_stride: kv_head_stride as i32,
                },
            )
            .map_err(candle_core::Error::wrap)?;
        } else if use_v1 {
            kernels::call_paged_attention_v1(
                dev.device(),
                &encoder,
                &kernels::Kernels::new(),
                ty,
                cache_ty,
                q.buffer(),
                q_l.start_offset() * q.dtype().size_in_bytes(),
                kc.buffer(),
                kc_l.start_offset() * kc.dtype().size_in_bytes(),
                vc.buffer(),
                vc_l.start_offset() * vc.dtype().size_in_bytes(),
                bt.buffer(),
                bt_l.start_offset() * bt.dtype().size_in_bytes(),
                cl.buffer(),
                cl_l.start_offset() * cl.dtype().size_in_bytes(),
                k_v_scale,
                alibi_storage_and_offset,
                &out,
                num_kv_heads as i32,
                self.softmax_scale,
                self.softcapping,
                block_size as i32,
                self.max_context_len as i32,
                num_seqs as i32,
                num_heads as i32,
                head_size as i32,
                max_num_blocks_per_seq as i32,
                q_stride as i32,
                kv_block_stride as i32,
                kv_head_stride as i32,
                sinks_storage_and_offset
                    .as_ref()
                    .map(|(b, o)| (b as &_, *o)),
                cache.hnd,
                sliding_window,
            )
            .map_err(candle_core::Error::wrap)?;
        } else {
            let tmp_out_shape = Shape::from((num_seqs, num_heads, max_num_partitions, head_size));
            let exp_sums_shape = Shape::from((num_seqs, num_heads, max_num_partitions));
            let tmp_out = dev.new_buffer(
                tmp_out_shape.elem_count(),
                q.dtype(),
                "paged-attention-tmpout",
            )?;
            let exp_sums = dev.new_buffer(
                exp_sums_shape.elem_count(),
                DType::F32,
                "paged-attention-expsums",
            )?;
            let max_logits = dev.new_buffer(
                exp_sums_shape.elem_count(),
                DType::F32,
                "paged-attention-maxlogits",
            )?;

            kernels::call_paged_attention_v2(
                dev.device(),
                &encoder,
                &kernels::Kernels::new(),
                ty,
                cache_ty,
                &exp_sums,
                &max_logits,
                q.buffer(),
                q_l.start_offset() * q.dtype().size_in_bytes(),
                kc.buffer(),
                kc_l.start_offset() * kc.dtype().size_in_bytes(),
                vc.buffer(),
                vc_l.start_offset() * vc.dtype().size_in_bytes(),
                bt.buffer(),
                bt_l.start_offset() * bt.dtype().size_in_bytes(),
                cl.buffer(),
                cl_l.start_offset() * cl.dtype().size_in_bytes(),
                k_v_scale,
                alibi_storage_and_offset,
                &tmp_out,
                &out,
                num_kv_heads as i32,
                self.softmax_scale,
                self.softcapping,
                block_size as i32,
                self.max_context_len as i32,
                num_seqs as i32,
                num_heads as i32,
                head_size as i32,
                max_num_blocks_per_seq as i32,
                q_stride as i32,
                kv_block_stride as i32,
                kv_head_stride as i32,
                sinks_storage_and_offset
                    .as_ref()
                    .map(|(b, o)| (b as &_, *o)),
                cache.hnd,
                sliding_window,
            )
            .map_err(candle_core::Error::wrap)?;
        }

        let newstorage =
            candle_core::MetalStorage::new(out, q.device().clone(), elem_count, q.dtype());
        Ok((newstorage, out_shape))
    }
}

/// PagedAttention layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors key_cache and value_cache
/// with fewer heads than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(num_sequences, num_heads_q, head_size)`.
/// * `key_cache` - HND `(blocks, kv_heads, block_size, head_size)` or legacy packed rank-5 cache.
/// * `value_cache` - HND shape matching K, or legacy `(blocks, kv_heads, head_size, block_size)`.
/// * `block_tables` - Padded table associating blocks to each sequence of shape `(num_sequences, max_context_len // block_size)`
/// * `context_lens` - Tensor associating lengths to each sequence of shape `(num_sequences)`
/// * `max_context_len` - Max of `context_len`
/// * `softmax_scale` - scaling factor
/// * `softcapping`- Softcapping value as in Gemma 2. Using 1.0 means do nothing.
/// * `alibi_slopes`- Optional alibi slopes, `(num_heads_q)`.
///
/// The resulting tensor has dimensions `(num_sequences, num_heads_q, head_size)`.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    paged_attention_with_window(
        q,
        k_scale,
        v_scale,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        alibi_slopes,
        max_context_len,
        softmax_scale,
        softcapping,
        sinks,
        None,
    )
}

/// Decode from an HND cache with an optional exact trailing token window.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_with_window(
    q: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
    sinks: Option<&Tensor>,
    sliding_window: Option<usize>,
) -> Result<Tensor> {
    if sliding_window == Some(0) {
        candle_core::bail!("sliding_window must be positive");
    }
    let op = PagedAttention {
        softmax_scale,
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        block_tables: block_tables.clone(),
        context_lens: context_lens.clone(),
        max_context_len,
        sliding_window,
        softcapping,
        alibi_slopes: alibi_slopes.cloned(),
        k_scale: k_scale.cloned(),
        v_scale: v_scale.cloned(),
        sinks: sinks
            .map(|s| s.to_dtype(candle_core::DType::F32))
            .transpose()?,
    };
    q.apply_op1(op)
}

/// Insert key and values at the provided slot mapping inside the key value paged cache
///
/// # Arguments
///
/// * `key` - Key tensor of shape `(num_tokens, num_heads, head_size)`.
/// * `value` - Value tensor of shape `(num_tokens, num_heads, head_size)`.
/// * `key_cache` - HND `(blocks, heads, block_size, head_size)` or legacy packed rank-5 cache.
/// * `value_cache` - HND shape matching K, or legacy `(blocks, heads, head_size, block_size)`.
/// * `slot_mapping` - Mapping associating a slot to each token of shape `(num_tokens)`.
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    validate_kv_cache_scales(key_cache.dtype(), k_scale, v_scale, "reshape_and_cache")?;
    let kv_ty = match key.dtype() {
        DType::F16 => PagedAttentionDType::F16,
        DType::BF16 => PagedAttentionDType::BF16,
        DType::F32 => PagedAttentionDType::F32,
        dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
    };
    let cache_ty = match key_cache.dtype() {
        DType::F16 => PagedAttentionDType::F16,
        DType::BF16 => PagedAttentionDType::BF16,
        DType::F32 => PagedAttentionDType::F32,
        DType::F8E4M3 => PagedAttentionDType::F8E4M3,
        dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
    };

    let (k, k_l) = key.storage_and_layout();
    let k = match &*k {
        Storage::Metal(k) => k,
        _ => candle_core::bail!("key must be a metal tensor"),
    };

    let (v, v_l) = value.storage_and_layout();
    let v = match &*v {
        Storage::Metal(v) => v,
        _ => candle_core::bail!("value must be a metal tensor"),
    };

    let (kc, kc_l) = key_cache.storage_and_layout();
    let kc = match &*kc {
        Storage::Metal(kc) => kc,
        _ => candle_core::bail!("key_cache must be a metal tensor"),
    };

    let (vc, vc_l) = value_cache.storage_and_layout();
    let vc = match &*vc {
        Storage::Metal(vc) => vc,
        _ => candle_core::bail!("value_cache must be a metal tensor"),
    };

    let (s, s_l) = slot_mapping.storage_and_layout();
    let s = match &*s {
        Storage::Metal(s) => s,
        _ => candle_core::bail!("slot_mapping must be a metal tensor"),
    };

    let k_v_scale = if let (Some(k_scale), Some(v_scale)) = (k_scale, v_scale) {
        if k_scale.elem_count() != 1 || v_scale.elem_count() != 1 {
            candle_core::bail!("k_scale and v_scale must be scalars");
        }
        if k_scale.dtype() != DType::F32 || v_scale.dtype() != DType::F32 {
            candle_core::bail!("k_scale and v_scale must be f32");
        }

        let (k_scale, _) = k_scale.storage_and_layout();
        let k_scale = match &*k_scale {
            Storage::Metal(k_scale) => k_scale,
            _ => candle_core::bail!("k_scale must be a metal tensor"),
        };

        let (v_scale, _) = v_scale.storage_and_layout();
        let v_scale = match &*v_scale {
            Storage::Metal(v_scale) => v_scale,
            _ => candle_core::bail!("v_scale must be a metal tensor"),
        };

        Some((k_scale.buffer().clone(), v_scale.buffer().clone()))
    } else {
        None
    };

    let k_rank = k_l.stride().len();
    let v_rank = v_l.stride().len();

    if k_rank != 3 || v_rank != 3 {
        candle_core::bail!(
            "paged-attention expects input tensors of rank 3 (k: {k_l:?}, v: {v_l:?})"
        )
    }

    let (num_tokens, num_heads, head_size) = k_l.shape().dims3()?;
    if (num_tokens, num_heads, head_size) != v_l.shape().dims3()? {
        candle_core::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
    }

    let cache = CacheShape::new(key_cache, value_cache)?;
    if cache.num_heads != num_heads || cache.head_size != head_size {
        candle_core::bail!("input and cache head dimensions do not match");
    }

    if (num_tokens) != s_l.shape().dims1()? {
        candle_core::bail!(
            "shape mismatch slot_mapping {:?}, expected {:?}",
            s_l.shape(),
            (num_tokens)
        )
    }

    let key_stride = k_l.stride()[0] as i32;
    let value_stride = v_l.stride()[0] as i32;

    let dev = key.device().as_metal_device()?;

    let encoder = dev.command_encoder()?;
    encoder.set_label("reshape-and-cache");

    kernels::call_reshape_and_cache(
        dev.device(),
        &encoder,
        &kernels::Kernels::new(),
        kv_ty,
        cache_ty,
        k.buffer(),
        k_l.start_offset() * key.dtype().size_in_bytes(),
        v.buffer(),
        v_l.start_offset() * value.dtype().size_in_bytes(),
        kc.buffer(),
        kc_l.start_offset() * key_cache.dtype().size_in_bytes(),
        vc.buffer(),
        vc_l.start_offset() * value_cache.dtype().size_in_bytes(),
        s.buffer(),
        s_l.start_offset() * slot_mapping.dtype().size_in_bytes(),
        k_v_scale,
        num_tokens as i32,
        num_heads as i32,
        head_size as i32,
        cache.block_size as i32,
        cache.x as i32,
        key_stride,
        value_stride,
        cache.hnd,
    )
    .map_err(candle_core::Error::wrap)?;

    Ok(())
}

#[cfg(test)]
mod workspace_tests {
    use super::*;

    fn workload() -> MetalPagedAttentionWorkspace {
        MetalPagedAttentionWorkspace {
            num_seqs: 1,
            num_heads: 8,
            num_kv_heads: 2,
            head_size: 512,
            block_size: 32,
            max_context_len: 1024,
            sliding_window: None,
            dtype: DType::BF16,
            cache_dtype: DType::BF16,
            hnd: true,
            packed_x: 1,
            has_alibi: false,
            has_sinks: false,
            has_scales: false,
            softcapping: 1.0,
        }
    }

    #[test]
    fn split_accounts_for_f32_partials_and_each_backing_buffer() {
        let input = workload();
        assert_eq!(
            input.bytes_for_kernel(MetalPagedKernel::Auto).unwrap(),
            8 * 512 * 2 + 2 * 8 * 32 * 4 + 8 * 32 * 512 * 4
        );
        assert_eq!(
            MetalPagedAttentionWorkspace {
                max_context_len: 256,
                ..input
            }
            .bytes_for_kernel(MetalPagedKernel::Auto)
            .unwrap(),
            8 * 512 * 2
        );
    }

    #[test]
    fn feature_fallback_accounts_for_generic_partitions() {
        let input = MetalPagedAttentionWorkspace {
            cache_dtype: DType::F8E4M3,
            has_scales: true,
            max_context_len: 1025,
            ..workload()
        };
        assert_eq!(
            input.bytes_for_kernel(MetalPagedKernel::Auto).unwrap(),
            8192 + 2 * 128 + 32768
        );
        assert_eq!(
            input.bytes_for_kernel(MetalPagedKernel::Auto).unwrap(),
            input.bytes_for_kernel(MetalPagedKernel::Legacy).unwrap()
        );
    }

    #[test]
    fn noncanonical_legacy_packing_uses_the_same_generic_fallback_as_dispatch() {
        let canonical = MetalPagedAttentionWorkspace {
            num_heads: 16,
            num_kv_heads: 8,
            head_size: 128,
            hnd: false,
            packed_x: SPECIALIZED_PACKED_X,
            max_context_len: 1024,
            ..workload()
        };
        let noncanonical = MetalPagedAttentionWorkspace {
            packed_x: SPECIALIZED_PACKED_X / 2,
            ..canonical
        };
        assert!(canonical
            .specialized_variant(MetalPagedKernel::Auto)
            .is_some());
        assert!(noncanonical
            .specialized_variant(MetalPagedKernel::Auto)
            .is_none());
        assert_eq!(
            noncanonical
                .bytes_for_kernel(MetalPagedKernel::Auto)
                .unwrap(),
            noncanonical
                .bytes_for_kernel(MetalPagedKernel::Legacy)
                .unwrap()
        );
        assert_ne!(
            noncanonical
                .bytes_for_kernel(MetalPagedKernel::Auto)
                .unwrap(),
            canonical.bytes_for_kernel(MetalPagedKernel::Auto).unwrap()
        );
    }

    #[test]
    fn rejects_workspace_overflow() {
        assert!(MetalPagedAttentionWorkspace {
            num_seqs: usize::MAX,
            ..workload()
        }
        .bytes()
        .is_err());
        assert!(MetalPagedAttentionWorkspace {
            num_kv_heads: 0,
            ..workload()
        }
        .bytes()
        .is_err());
    }

    #[test]
    fn capacity_bound_covers_dispatch_switches_and_smaller_batches() {
        let contexts = [
            1, 128, 256, 288, 512, 513, 767, 768, 1024, 2048, 4096, 4097, 8192,
        ];
        let batches = [1, 2, 4, 8, 16, 32, 64];
        for kernel in [
            MetalPagedKernel::Auto,
            MetalPagedKernel::Grouped,
            MetalPagedKernel::Legacy,
        ] {
            for shape in [
                workload(),
                MetalPagedAttentionWorkspace {
                    num_seqs: 8,
                    ..workload()
                },
                MetalPagedAttentionWorkspace {
                    num_seqs: 64,
                    num_heads: 16,
                    num_kv_heads: 8,
                    head_size: 128,
                    ..workload()
                },
                MetalPagedAttentionWorkspace {
                    num_seqs: 64,
                    cache_dtype: DType::F8E4M3,
                    has_scales: true,
                    ..workload()
                },
                MetalPagedAttentionWorkspace {
                    sliding_window: Some(512),
                    ..workload()
                },
            ] {
                let mut previous = 0;
                for capacity in contexts {
                    let bound = MetalPagedAttentionWorkspace {
                        max_context_len: capacity,
                        ..shape
                    }
                    .capacity_bytes_for_kernel(kernel)
                    .unwrap();
                    assert!(bound >= previous, "capacity {capacity} reduced the bound");
                    previous = bound;
                    for batch in batches.into_iter().filter(|batch| *batch <= shape.num_seqs) {
                        for context in contexts.into_iter().filter(|context| *context <= capacity) {
                            let exact = MetalPagedAttentionWorkspace {
                                num_seqs: batch,
                                max_context_len: context,
                                ..shape
                            }
                            .bytes_for_kernel(kernel)
                            .unwrap();
                            assert!(
                                bound >= exact,
                                "batch {batch}, context {context} exceeds capacity {capacity}"
                            );
                        }
                    }
                }
            }
        }
    }
}
