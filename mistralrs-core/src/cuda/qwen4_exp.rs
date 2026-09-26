//! Launch wrappers for the Qwen4-Exp CUDA kernels in `qwen4_exp.cu`.
#![allow(clippy::cast_possible_truncation)]

use candle_core::{
    cuda_backend::cudarc::driver::DevicePtr, DType, Device, Result, Storage, Tensor,
};

pub(crate) const PLE_MAX_NGRAM: usize = 8;
pub(crate) const PLE_MAX_HEADS: usize = 64;
pub(crate) const QSA_HEAD_DIM: usize = 128;
pub(crate) const QSA_MAX_INDEX_HEADS: usize = 8;
pub(crate) const ATTN_HEAD_DIM: usize = 256;
const ATTN_TARGET_CTAS: usize = 256;
const ATTN_MIN_ITEMS_PER_SPLIT: usize = 64;
const ATTN_MAX_SPLITS: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Q4Tokens {
    pub tok_seq: u64,
    pub tok_local: u64,
    pub seq_start: u64,
    pub seq_len: u64,
    pub n_tokens: i32,
    pub n_seqs: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Q4Paged {
    pub block_tables: u64,
    pub kv_lens: u64,
    pub max_blocks_per_seq: i32,
    pub block_size: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Q4PleHashParams {
    pub multipliers: [u64; PLE_MAX_NGRAM],
    pub vocab_sizes: [u64; PLE_MAX_HEADS],
    pub offsets: [u64; PLE_MAX_HEADS],
    pub ngram_size: i32,
    pub heads_per_ngram: i32,
    pub num_heads: i32,
    pub eos: u32,
}

mod ffi {
    use super::{Q4Paged, Q4PleHashParams, Q4Tokens};
    use core::ffi::c_void;

    extern "C" {
        pub(super) fn qwen4_hc_norm(
            x: *const c_void,
            weight: *const f32,
            out: *mut c_void,
            rows: i32,
            hc: i32,
            hidden: i32,
            eps: f32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn qwen4_hc_mix(
            xn: *const c_void,
            gate: *const c_void,
            out: *mut c_void,
            tokens: i32,
            hc: i32,
            hidden: i32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn qwen4_hc_combine(
            res: *const c_void,
            block_out: *const c_void,
            inject: *const c_void,
            res_out: *mut c_void,
            next_weight: *const f32,
            xn_out: *mut c_void,
            rows: i32,
            hc: i32,
            hidden: i32,
            eps: f32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn qwen4_ple_hash(
            tokens: *const u32,
            layout: Q4Tokens,
            hist_pool: *const f32,
            slots: *const u32,
            hist_len: i32,
            params: *const Q4PleHashParams,
            rows_out: *mut i64,
            stream: i64,
        );
        pub(super) fn qwen4_ple_hist_update(
            tokens: *const u32,
            layout: Q4Tokens,
            hist_pool: *mut f32,
            slots: *const u32,
            hist_len: i32,
            stream: i64,
        );
        pub(super) fn qwen4_ple_gather(
            rows: *const i64,
            n_lookups: i32,
            shard_ptrs: *const u64,
            shard_starts: *const i64,
            n_shards: i32,
            head_dim: i32,
            out: *mut c_void,
            stream: i64,
        );
        pub(super) fn qwen4_ple_gather_quant(
            rows: *const i64,
            n_lookups: i32,
            data: *const u8,
            scales: *const c_void,
            head_dim: i32,
            bits: i32,
            out: *mut c_void,
            stream: i64,
        );
        pub(super) fn qwen4_ple_gate(
            key: *const c_void,
            hidden_states: *const c_void,
            value: *const c_void,
            w_key: *const f32,
            w_query: *const f32,
            w_conv: *const f32,
            gated_out: *mut c_void,
            normed_out: *mut c_void,
            tokens: i32,
            hc: i32,
            hidden: i32,
            eps: f32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn qwen4_ple_conv(
            normed: *const c_void,
            gated: *const c_void,
            residual: *const c_void,
            weight: *const c_void,
            state_pool: *mut c_void,
            slots: *const u32,
            out: *mut c_void,
            layout: Q4Tokens,
            channels: i32,
            kernel_size: i32,
            dilation: i32,
            state_len: i32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn qwen4_qsa_aux_write(
            raw_key: *const c_void,
            cos: *const c_void,
            sin: *const c_void,
            slot_mapping: *const i64,
            aux: *mut c_void,
            n_tokens: i32,
            key_dim: i32,
            half_rot: i32,
            aux_dim: i32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn qwen4_qsa_finalize(
            aux: *mut c_void,
            layout: Q4Tokens,
            paged: Q4Paged,
            norm_weight: *const f32,
            ratio: i32,
            half_rot: i32,
            aux_dim: i32,
            eps: f32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn qwen4_qsa_select(
            q: *const c_void,
            aux: *const c_void,
            layout: Q4Tokens,
            paged: Q4Paged,
            scores: *mut f32,
            score_stride: i32,
            n_heads: i32,
            ratio: i32,
            topk: i32,
            aux_dim: i32,
            selected: *mut i32,
            n_selected: *mut i32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn qwen4_qsa_attention(
            q: *const c_void,
            k_cache: *const c_void,
            v_cache: *const c_void,
            layout: Q4Tokens,
            paged: Q4Paged,
            selected: *const i32,
            n_selected: *const i32,
            topk: i32,
            ratio: i32,
            n_q_heads: i32,
            n_kv_heads: i32,
            splits: i32,
            items_per_split: i32,
            scale: f32,
            flashinfer_layout: i32,
            out: *mut c_void,
            part_acc: *mut f32,
            part_ml: *mut f32,
            dtype: i32,
            stream: i64,
        );
    }
}

/// Device address of a tensor's first element. The tensor must stay alive while it is used.
pub(crate) fn dev_ptr(t: &Tensor) -> Result<u64> {
    let (storage, layout) = t.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("expected a CUDA tensor");
    };
    macro_rules! ptr {
        ($ty:ty) => {{
            let slice = storage.as_cuda_slice::<$ty>()?;
            slice
                .slice(layout.start_offset()..)
                .device_ptr(slice.stream())
                .0
        }};
    }
    Ok(match t.dtype() {
        DType::BF16 => ptr!(half::bf16),
        DType::F16 => ptr!(half::f16),
        DType::F32 => ptr!(f32),
        DType::U32 => ptr!(u32),
        DType::I32 => ptr!(i32),
        DType::I64 => ptr!(i64),
        DType::U8 => ptr!(u8),
        DType::F8E4M3 => ptr!(float8::F8E4M3),
        other => candle_core::bail!("unsupported dtype {other:?} for a raw device pointer"),
    })
}

fn opt_ptr(t: Option<&Tensor>) -> Result<u64> {
    t.map_or(Ok(0), dev_ptr)
}

fn stream(device: &Device) -> Result<i64> {
    Ok(device.as_cuda_device()?.cuda_stream().cu_stream() as i64)
}

fn dtype_code(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle_core::bail!("Qwen4-Exp CUDA kernels do not support {other:?}"),
    }
}

fn half_dtype_code(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        other => candle_core::bail!("Qwen4-Exp QSA kernels require f16/bf16, got {other:?}"),
    }
}

fn empty(shape: &[usize], dtype: DType, device: &Device) -> Result<Tensor> {
    // Safety: every element is written by the kernel before it is read.
    unsafe { Tensor::empty(shape, dtype, device) }
}

/// Device-side token layout; `None` fields mean decode, where token `t` is sequence `t`'s new token.
#[derive(Clone)]
pub(crate) struct TokenLayout {
    pub tok_seq: Option<Tensor>,
    pub tok_local: Option<Tensor>,
    pub seq_start: Option<Tensor>,
    pub seq_len: Option<Tensor>,
    /// Kv length after this step per sequence; `None` means the paged metadata's own lengths.
    pub kv_lens: Option<Tensor>,
    pub n_tokens: usize,
    pub n_seqs: usize,
}

impl TokenLayout {
    pub(crate) fn decode(n_seqs: usize) -> Self {
        Self {
            tok_seq: None,
            tok_local: None,
            seq_start: None,
            seq_len: None,
            kv_lens: None,
            n_tokens: n_seqs,
            n_seqs,
        }
    }

    /// `seqs[i]` is `(start, len)` on the flattened token axis, `kv_lens[i]` its kv length after this step.
    pub(crate) fn from_host(
        seqs: &[(usize, usize)],
        kv_lens: &[usize],
        n_tokens: usize,
        device: &Device,
    ) -> Result<Self> {
        let mut tok_seq = vec![-1i32; n_tokens];
        let mut tok_local = vec![0i32; n_tokens];
        for (seq, &(start, len)) in seqs.iter().enumerate() {
            for local in 0..len {
                tok_seq[start + local] = seq as i32;
                tok_local[start + local] = local as i32;
            }
        }
        let starts = seqs.iter().map(|(s, _)| *s as i32).collect::<Vec<_>>();
        let lens = seqs.iter().map(|(_, l)| *l as i32).collect::<Vec<_>>();
        Ok(Self {
            tok_seq: Some(Tensor::from_vec(tok_seq, n_tokens, device)?),
            tok_local: Some(Tensor::from_vec(tok_local, n_tokens, device)?),
            seq_start: Some(Tensor::from_vec(starts, seqs.len(), device)?),
            seq_len: Some(Tensor::from_vec(lens, seqs.len(), device)?),
            kv_lens: Some(Tensor::from_vec(
                kv_lens.iter().map(|l| *l as u32).collect::<Vec<_>>(),
                kv_lens.len(),
                device,
            )?),
            n_tokens,
            n_seqs: seqs.len(),
        })
    }

    fn ffi(&self) -> Result<Q4Tokens> {
        Ok(Q4Tokens {
            tok_seq: opt_ptr(self.tok_seq.as_ref())?,
            tok_local: opt_ptr(self.tok_local.as_ref())?,
            seq_start: opt_ptr(self.seq_start.as_ref())?,
            seq_len: opt_ptr(self.seq_len.as_ref())?,
            n_tokens: self.n_tokens as i32,
            n_seqs: self.n_seqs as i32,
        })
    }
}

/// 32-bit `[seqs, max_blocks]` block tables and `[seqs]` kv lengths after this step.
pub(crate) struct PagedView<'a> {
    pub block_tables: &'a Tensor,
    pub kv_lens: &'a Tensor,
    pub block_size: usize,
}

impl PagedView<'_> {
    fn ffi(&self) -> Result<Q4Paged> {
        Ok(Q4Paged {
            block_tables: dev_ptr(self.block_tables)?,
            kv_lens: dev_ptr(self.kv_lens)?,
            max_blocks_per_seq: self.block_tables.dim(1)? as i32,
            block_size: self.block_size as i32,
        })
    }
}

/// Grouped RMSNorm over each of the `hc` streams of `x: [tokens, hc * hidden]`.
pub(crate) fn hc_norm(x: &Tensor, weight: &Tensor, hc: usize, eps: f64) -> Result<Tensor> {
    let x = x.contiguous()?;
    let width = x.dim(candle_core::D::Minus1)?;
    let rows = x.elem_count() / (width / hc);
    let out = empty(x.dims(), x.dtype(), x.device())?;
    unsafe {
        ffi::qwen4_hc_norm(
            dev_ptr(&x)? as *const _,
            dev_ptr(weight)? as *const f32,
            dev_ptr(&out)? as *mut _,
            rows as i32,
            hc as i32,
            (width / hc) as i32,
            eps as f32,
            dtype_code(x.dtype())?,
            stream(x.device())?,
        );
    }
    Ok(out)
}

/// Mean over streams of `xn * sigmoid(gate)`: `[.., hc * hidden] -> [.., hidden]`.
pub(crate) fn hc_mix(xn: &Tensor, gate: &Tensor, hc: usize) -> Result<Tensor> {
    let xn = xn.contiguous()?;
    let gate = gate.contiguous()?;
    let width = xn.dim(candle_core::D::Minus1)?;
    let hidden = width / hc;
    let tokens = xn.elem_count() / width;
    let mut shape = xn.dims().to_vec();
    *shape.last_mut().expect("rank >= 1") = hidden;
    let out = empty(&shape, xn.dtype(), xn.device())?;
    unsafe {
        ffi::qwen4_hc_mix(
            dev_ptr(&xn)? as *const _,
            dev_ptr(&gate)? as *const _,
            dev_ptr(&out)? as *mut _,
            tokens as i32,
            hc as i32,
            hidden as i32,
            dtype_code(xn.dtype())?,
            stream(xn.device())?,
        );
    }
    Ok(out)
}

/// `res + block_out * 2 * sigmoid(inject / hc)` per stream, plus the next mixer's grouped norm.
pub(crate) fn hc_combine(
    res: &Tensor,
    block_out: &Tensor,
    inject: &Tensor,
    next_norm: Option<&Tensor>,
    hc: usize,
    eps: f64,
) -> Result<(Tensor, Option<Tensor>)> {
    let res = res.contiguous()?;
    let block_out = block_out.contiguous()?;
    let inject = inject.contiguous()?;
    let width = res.dim(candle_core::D::Minus1)?;
    let hidden = width / hc;
    let rows = res.elem_count() / hidden;
    let out = empty(res.dims(), res.dtype(), res.device())?;
    let xn = next_norm
        .map(|_| empty(res.dims(), res.dtype(), res.device()))
        .transpose()?;
    unsafe {
        ffi::qwen4_hc_combine(
            dev_ptr(&res)? as *const _,
            dev_ptr(&block_out)? as *const _,
            dev_ptr(&inject)? as *const _,
            dev_ptr(&out)? as *mut _,
            opt_ptr(next_norm)? as *const f32,
            opt_ptr(xn.as_ref())? as *mut _,
            rows as i32,
            hc as i32,
            hidden as i32,
            eps as f32,
            dtype_code(res.dtype())?,
            stream(res.device())?,
        );
    }
    Ok((out, xn))
}

/// `[tokens, heads]` table rows from each token's n-gram context; also advances `hist_pool`.
pub(crate) fn ple_hash(
    tokens: &Tensor,
    layout: &TokenLayout,
    hist_pool: &Tensor,
    slots: &Tensor,
    params: &Q4PleHashParams,
) -> Result<Tensor> {
    let tokens = tokens.flatten_all()?.contiguous()?;
    let device = tokens.device();
    let hist_len = hist_pool.dim(1)?;
    let rows = empty(
        &[layout.n_tokens, params.num_heads as usize],
        DType::I64,
        device,
    )?;
    let ffi_layout = layout.ffi()?;
    unsafe {
        ffi::qwen4_ple_hash(
            dev_ptr(&tokens)? as *const u32,
            ffi_layout,
            dev_ptr(hist_pool)? as *const f32,
            dev_ptr(slots)? as *const u32,
            hist_len as i32,
            params as *const Q4PleHashParams,
            dev_ptr(&rows)? as *mut i64,
            stream(device)?,
        );
        ffi::qwen4_ple_hist_update(
            dev_ptr(&tokens)? as *const u32,
            ffi_layout,
            dev_ptr(hist_pool)? as *mut f32,
            dev_ptr(slots)? as *const u32,
            hist_len as i32,
            stream(device)?,
        );
    }
    Ok(rows)
}

/// Gather 16-bit rows from shard base addresses the device can dereference (ATS/HMM).
pub(crate) fn ple_gather(
    rows: &Tensor,
    shard_ptrs: &Tensor,
    shard_starts: &Tensor,
    head_dim: usize,
    dtype: DType,
) -> Result<Tensor> {
    let (tokens, heads) = rows.dims2()?;
    let out = empty(&[tokens, heads * head_dim], dtype, rows.device())?;
    unsafe {
        ffi::qwen4_ple_gather(
            dev_ptr(rows)? as *const i64,
            (tokens * heads) as i32,
            dev_ptr(shard_ptrs)? as *const u64,
            dev_ptr(shard_starts)? as *const i64,
            shard_ptrs.dim(0)? as i32,
            head_dim as i32,
            dev_ptr(&out)? as *mut _,
            stream(rows.device())?,
        );
    }
    Ok(out)
}

/// Gather and dequantize rows from the device-resident table into `[tokens, heads * head_dim]` bf16.
pub(crate) fn ple_gather_quant(
    rows: &Tensor,
    data: &Tensor,
    scales: &Tensor,
    head_dim: usize,
    bits: usize,
) -> Result<Tensor> {
    let (tokens, heads) = rows.dims2()?;
    let out = empty(&[tokens, heads * head_dim], DType::BF16, rows.device())?;
    unsafe {
        ffi::qwen4_ple_gather_quant(
            dev_ptr(rows)? as *const i64,
            (tokens * heads) as i32,
            dev_ptr(data)? as *const u8,
            dev_ptr(scales)? as *const _,
            head_dim as i32,
            bits as i32,
            dev_ptr(&out)? as *mut _,
            stream(rows.device())?,
        );
    }
    Ok(out)
}

pub(crate) struct PleGateWeights<'a> {
    pub key: &'a Tensor,
    pub query: &'a Tensor,
    pub conv: &'a Tensor,
}

/// Returns `(gated, gated_normed)`, both `[tokens, hc * hidden]`.
pub(crate) fn ple_gate(
    key: &Tensor,
    hidden_states: &Tensor,
    value: &Tensor,
    weights: PleGateWeights<'_>,
    hc: usize,
    eps: f64,
) -> Result<(Tensor, Tensor)> {
    let key = key.contiguous()?;
    let hidden_states = hidden_states.contiguous()?;
    let value = value.contiguous()?;
    let hidden = value.dim(candle_core::D::Minus1)?;
    let tokens = value.elem_count() / hidden;
    let gated = empty(key.dims(), key.dtype(), key.device())?;
    let normed = empty(key.dims(), key.dtype(), key.device())?;
    unsafe {
        ffi::qwen4_ple_gate(
            dev_ptr(&key)? as *const _,
            dev_ptr(&hidden_states)? as *const _,
            dev_ptr(&value)? as *const _,
            dev_ptr(weights.key)? as *const f32,
            dev_ptr(weights.query)? as *const f32,
            dev_ptr(weights.conv)? as *const f32,
            dev_ptr(&gated)? as *mut _,
            dev_ptr(&normed)? as *mut _,
            tokens as i32,
            hc as i32,
            hidden as i32,
            eps as f32,
            dtype_code(key.dtype())?,
            stream(key.device())?,
        );
    }
    Ok((gated, normed))
}

pub(crate) struct PleConvArgs<'a> {
    pub normed: &'a Tensor,
    pub gated: &'a Tensor,
    pub residual: &'a Tensor,
    pub weight: &'a Tensor,
    pub state_pool: &'a Tensor,
    pub slots: &'a Tensor,
    pub layout: &'a TokenLayout,
    pub dilation: usize,
}

/// `residual + gated + silu(conv(normed))`, advancing the pooled conv state in place.
pub(crate) fn ple_conv(args: PleConvArgs<'_>) -> Result<Tensor> {
    let normed = args.normed.contiguous()?;
    let gated = args.gated.contiguous()?;
    let residual = args.residual.contiguous()?;
    let (channels, kernel_size) = args.weight.dims2()?;
    let state_len = args.state_pool.dim(2)?;
    let out = empty(residual.dims(), residual.dtype(), residual.device())?;
    unsafe {
        ffi::qwen4_ple_conv(
            dev_ptr(&normed)? as *const _,
            dev_ptr(&gated)? as *const _,
            dev_ptr(&residual)? as *const _,
            dev_ptr(args.weight)? as *const _,
            dev_ptr(args.state_pool)? as *mut _,
            dev_ptr(args.slots)? as *const u32,
            dev_ptr(&out)? as *mut _,
            args.layout.ffi()?,
            channels as i32,
            kernel_size as i32,
            args.dilation as i32,
            state_len as i32,
            dtype_code(residual.dtype())?,
            stream(residual.device())?,
        );
    }
    Ok(out)
}

/// Store `[raw key | cos | sin]` of every token at its slot of the aux cache.
pub(crate) fn qsa_aux_write(
    raw_key: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    slot_mapping: &Tensor,
    aux: &Tensor,
) -> Result<()> {
    let raw_key = raw_key.contiguous()?;
    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    if slot_mapping.dtype() != DType::I64 {
        candle_core::bail!(
            "QSA expects i64 slot mappings, got {:?}",
            slot_mapping.dtype()
        );
    }
    let slot_mapping = slot_mapping.flatten_all()?;
    let n_tokens = slot_mapping.dim(0)?;
    let key_dim = raw_key.elem_count() / n_tokens;
    let half_rot = cos.elem_count() / n_tokens;
    unsafe {
        ffi::qwen4_qsa_aux_write(
            dev_ptr(&raw_key)? as *const _,
            dev_ptr(&cos)? as *const _,
            dev_ptr(&sin)? as *const _,
            dev_ptr(&slot_mapping)? as *const i64,
            dev_ptr(aux)? as *mut _,
            n_tokens as i32,
            key_dim as i32,
            half_rot as i32,
            aux.dim(candle_core::D::Minus1)? as i32,
            half_dtype_code(aux.dtype())?,
            stream(aux.device())?,
        );
    }
    Ok(())
}

pub(crate) struct QsaShape {
    pub ratio: usize,
    pub topk: usize,
    pub half_rot: usize,
    pub n_index_heads: usize,
}

/// Pool, normalize and rotate the key of every compressed block this step completes.
pub(crate) fn qsa_finalize(
    aux: &Tensor,
    layout: &TokenLayout,
    paged: &PagedView<'_>,
    norm_weight: &Tensor,
    shape: &QsaShape,
    eps: f64,
) -> Result<()> {
    unsafe {
        ffi::qwen4_qsa_finalize(
            dev_ptr(aux)? as *mut _,
            layout.ffi()?,
            paged.ffi()?,
            dev_ptr(norm_weight)? as *const f32,
            shape.ratio as i32,
            shape.half_rot as i32,
            aux.dim(candle_core::D::Minus1)? as i32,
            eps as f32,
            half_dtype_code(aux.dtype())?,
            stream(aux.device())?,
        );
    }
    Ok(())
}

/// Per-query top-k blocks as `(selected [tokens, topk], n_selected [tokens])`, both i32.
pub(crate) fn qsa_select(
    q: &Tensor,
    aux: &Tensor,
    layout: &TokenLayout,
    paged: &PagedView<'_>,
    shape: &QsaShape,
    max_blocks: usize,
) -> Result<(Tensor, Tensor)> {
    if shape.n_index_heads > QSA_MAX_INDEX_HEADS || q.dim(candle_core::D::Minus1)? != QSA_HEAD_DIM {
        candle_core::bail!(
            "QSA selection supports at most {QSA_MAX_INDEX_HEADS} indexer heads of dim {QSA_HEAD_DIM}"
        );
    }
    let q = q.contiguous()?;
    let device = q.device();
    let n_tokens = layout.n_tokens;
    let score_stride = max_blocks.max(shape.topk + 1);
    let scores = empty(&[n_tokens, score_stride], DType::F32, device)?;
    let selected = empty(&[n_tokens, shape.topk], DType::I32, device)?;
    let n_selected = empty(&[n_tokens], DType::I32, device)?;
    unsafe {
        ffi::qwen4_qsa_select(
            dev_ptr(&q)? as *const _,
            dev_ptr(aux)? as *const _,
            layout.ffi()?,
            paged.ffi()?,
            dev_ptr(&scores)? as *mut f32,
            score_stride as i32,
            shape.n_index_heads as i32,
            shape.ratio as i32,
            shape.topk as i32,
            aux.dim(candle_core::D::Minus1)? as i32,
            dev_ptr(&selected)? as *mut i32,
            dev_ptr(&n_selected)? as *mut i32,
            half_dtype_code(aux.dtype())?,
            stream(device)?,
        );
    }
    Ok((selected, n_selected))
}

pub(crate) struct QsaAttentionArgs<'a> {
    pub q: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub layout: &'a TokenLayout,
    pub paged: &'a PagedView<'a>,
    pub selected: &'a Tensor,
    pub n_selected: &'a Tensor,
    pub shape: &'a QsaShape,
    pub n_kv_heads: usize,
    pub scale: f32,
}

/// Attention of `q: [tokens, q_heads, 256]` over each query's selected blocks plus its tail.
pub(crate) fn qsa_attention(args: QsaAttentionArgs<'_>) -> Result<Tensor> {
    let q = args.q.contiguous()?;
    let (n_tokens, n_q_heads, head_dim) = q.dims3()?;
    if head_dim != ATTN_HEAD_DIM {
        candle_core::bail!("QSA attention requires head dim {ATTN_HEAD_DIM}, got {head_dim}");
    }
    let device = q.device();
    let flashinfer_layout = args.key_cache.rank() == 4;
    let max_items = args.shape.topk * args.shape.ratio + args.shape.ratio - 1;
    let ctas = n_tokens * args.n_kv_heads;
    let splits = ATTN_TARGET_CTAS
        .div_ceil(ctas.max(1))
        .min(max_items.div_ceil(ATTN_MIN_ITEMS_PER_SPLIT))
        .clamp(1, ATTN_MAX_SPLITS);
    let items_per_split = max_items.div_ceil(splits);
    let out = empty(q.dims(), q.dtype(), device)?;
    let (part_acc, part_ml) = if splits > 1 {
        let rows = n_tokens * n_q_heads * splits;
        (
            Some(empty(&[rows, ATTN_HEAD_DIM], DType::F32, device)?),
            Some(empty(&[rows, 2], DType::F32, device)?),
        )
    } else {
        (None, None)
    };
    unsafe {
        ffi::qwen4_qsa_attention(
            dev_ptr(&q)? as *const _,
            dev_ptr(args.key_cache)? as *const _,
            dev_ptr(args.value_cache)? as *const _,
            args.layout.ffi()?,
            args.paged.ffi()?,
            dev_ptr(args.selected)? as *const i32,
            dev_ptr(args.n_selected)? as *const i32,
            args.shape.topk as i32,
            args.shape.ratio as i32,
            n_q_heads as i32,
            args.n_kv_heads as i32,
            splits as i32,
            items_per_split as i32,
            args.scale,
            i32::from(flashinfer_layout),
            dev_ptr(&out)? as *mut _,
            opt_ptr(part_acc.as_ref())? as *mut f32,
            opt_ptr(part_ml.as_ref())? as *mut f32,
            half_dtype_code(q.dtype())?,
            stream(device)?,
        );
    }
    Ok(out)
}
