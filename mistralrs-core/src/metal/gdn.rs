//! Metal GDN kernels for Gated Delta Net recurrence, causal conv1d, and fused gating.
//!
//! Mirrors the CUDA implementations in `cuda/gdn.rs`.

#![allow(clippy::cast_possible_truncation)]

#[cfg(feature = "metal")]
use candle_core::backend::BackendStorage;
#[cfg(feature = "metal")]
use candle_core::{DType, Device, Result, Storage, Tensor};

#[cfg(feature = "metal")]
use candle_metal_kernels::metal::{
    Buffer, ComputeCommandEncoder, ComputePipeline, Device as MetalRawDevice,
};

#[cfg(feature = "metal")]
use objc2_metal::MTLSize;

#[cfg(feature = "metal")]
use std::collections::HashMap;

#[cfg(feature = "metal")]
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "metal")]
use super::kernels::{device_key, RecurrentLibrary};

#[cfg(feature = "metal")]
type Pipelines = HashMap<u64, HashMap<String, ComputePipeline>>;

#[cfg(feature = "metal")]
static GDN_PIPELINES: OnceLock<RwLock<Pipelines>> = OnceLock::new();

#[cfg(feature = "metal")]
const SIMD_WIDTH: usize = 32;
#[cfg(feature = "metal")]
const SIMDGROUPS_PER_THREADGROUP: usize = 4;
#[cfg(feature = "metal")]
const PREFILL_VALUE_COLUMNS: usize = 4;
#[cfg(feature = "metal")]
const MIN_GROUPED_THREADGROUPS: usize = 256;
#[cfg(feature = "metal")]
const MIN_GROUPED_VALUE_DIM: usize = 64;

#[cfg(feature = "metal")]
fn load_pipeline(device: &MetalRawDevice, name: &str) -> Result<ComputePipeline> {
    let pipelines_lock = GDN_PIPELINES.get_or_init(|| RwLock::new(Pipelines::new()));
    let key = device_key(device);

    {
        let pipelines = pipelines_lock
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline cache: {e}")))?;
        if let Some(pipeline) = pipelines
            .get(&key)
            .and_then(|pipelines| pipelines.get(name))
        {
            return Ok(pipeline.clone());
        }
    }

    let mut pipelines = pipelines_lock.write().map_err(|e| {
        candle_core::Error::Msg(format!("Failed to lock pipeline cache for write: {e}"))
    })?;
    let pipelines = pipelines.entry(key).or_default();
    if let Some(pipeline) = pipelines.get(name) {
        return Ok(pipeline.clone());
    }
    let lib = RecurrentLibrary::Gdn.load(device)?;
    let func = lib.get_function(name, None).map_err(|e| {
        candle_core::Error::Msg(format!("Failed to load Metal function '{name}': {e}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create pipeline for '{name}': {e}"))
        })?;

    pipelines.insert(name.to_owned(), pipeline.clone());
    Ok(pipeline)
}

#[cfg(feature = "metal")]
pub(crate) fn warmup_kernels(
    device: &MetalRawDevice,
    dtype: DType,
    head_k_dim: usize,
) -> Result<()> {
    let type_name = match dtype {
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        DType::F32 => "float",
        dtype => candle_core::bail!("unsupported Metal GDN activation dtype {dtype:?}"),
    };
    if matches!(dtype, DType::F16 | DType::BF16) {
        let suffix = if dtype == DType::BF16 {
            "bfloat16_t"
        } else {
            "half"
        };
        for name in [
            "causal_conv1d_full",
            "save_conv_state",
            "causal_conv1d_full_token_major",
            "save_conv_state_token_major",
            "causal_conv1d_update",
            "causal_conv1d_update_pooled",
            "fused_gdn_gating",
        ] {
            load_pipeline(device, &format!("{name}_{suffix}"))?;
        }
    }
    load_pipeline(device, &format!("gdn_rmsnorm_gated_{type_name}"))?;
    if matches!(head_k_dim, 64 | 128) {
        for name in [
            format!("gdn_prefill_from_convolved_{type_name}_{head_k_dim}"),
            format!("gdn_prefill_from_convolved_grouped_{type_name}_{head_k_dim}_{PREFILL_VALUE_COLUMNS}"),
            format!("gated_delta_rule_{head_k_dim}_64"),
            format!("gated_delta_rule_pooled_decode_{head_k_dim}_64"),
            format!("chunked_gated_delta_rule_32_{head_k_dim}_64"),
        ] {
            load_pipeline(device, &name)?;
        }
    } else {
        load_pipeline(device, "gated_delta_rule_fallback")?;
    }
    Ok(())
}

// Helper to extract Metal buffer and byte offset from a tensor
#[cfg(feature = "metal")]
fn metal_buffer_and_offset(tensor: &Tensor) -> Result<(Buffer, usize)> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Metal(m) => {
            let offset = layout.start_offset() * m.dtype().size_in_bytes();
            Ok((m.buffer().clone(), offset))
        }
        _ => candle_core::bail!("Expected Metal tensor"),
    }
}

// ============================================================================
// Public API: gated_delta_rule_recurrence
// ============================================================================

/// Gated delta rule recurrence on Metal.
///
/// q, k: [BH, S, K]  v: [BH, S, V]  g, beta: [BH, S]
/// state: [BH, K, V] (mutated in-place)
/// Returns: output [BH, S, V]
#[cfg(feature = "metal")]
pub fn gated_delta_rule_recurrence_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;

    let (bh, seq_len, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;

    let Device::Metal(dev) = q.device() else {
        candle_core::bail!("gated_delta_rule_recurrence_metal: expected Metal device");
    };

    let kernel_name = match k_dim {
        128 => "gated_delta_rule_128_64",
        64 => "gated_delta_rule_64_64",
        _ => "gated_delta_rule_fallback",
    };
    let bv = 64usize;

    let pipeline = load_pipeline(dev.device(), kernel_name)?;

    // Allocate output
    let output = Tensor::zeros((bh, seq_len, v_dim), DType::F32, q.device())?;

    let (q_buf, q_off) = metal_buffer_and_offset(&q)?;
    let (k_buf, k_off) = metal_buffer_and_offset(&k)?;
    let (v_buf, v_off) = metal_buffer_and_offset(&v)?;
    let (g_buf, g_off) = metal_buffer_and_offset(&g)?;
    let (beta_buf, beta_off) = metal_buffer_and_offset(&beta)?;
    let (state_buf, state_off) = metal_buffer_and_offset(state)?;
    let (out_buf, out_off) = metal_buffer_and_offset(&output)?;

    let encoder = dev.command_encoder()?;
    encoder.set_label("gdn-recurrence");
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(&q_buf), q_off);
    encoder.set_input_buffer(1, Some(&k_buf), k_off);
    encoder.set_input_buffer(2, Some(&v_buf), v_off);
    encoder.set_input_buffer(3, Some(&g_buf), g_off);
    encoder.set_input_buffer(4, Some(&beta_buf), beta_off);
    encoder.set_output_buffer(5, Some(&state_buf), state_off);
    encoder.set_output_buffer(6, Some(&out_buf), out_off);

    let seq_len_i32 = seq_len as i32;
    let v_dim_i32 = v_dim as i32;

    if kernel_name == "gated_delta_rule_fallback" {
        let k_dim_i32 = k_dim as i32;
        encoder.set_bytes(7, &seq_len_i32);
        encoder.set_bytes(8, &k_dim_i32);
        encoder.set_bytes(9, &v_dim_i32);
        // Dynamic threadgroup memory: 2 * k_dim floats
        encoder.set_threadgroup_memory_length(0, 2 * k_dim * std::mem::size_of::<f32>());
    } else {
        encoder.set_bytes(7, &seq_len_i32);
        encoder.set_bytes(8, &v_dim_i32);
    }

    let grid_x = v_dim.div_ceil(bv);
    let thread_groups = MTLSize {
        width: grid_x,
        height: bh,
        depth: 1,
    };
    let threads_per_group = MTLSize {
        width: bv,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    Ok(output)
}

#[cfg(not(feature = "metal"))]
#[allow(dead_code)]
pub fn gated_delta_rule_recurrence_metal(
    _q: &candle_core::Tensor,
    _k: &candle_core::Tensor,
    _v: &candle_core::Tensor,
    _g: &candle_core::Tensor,
    _beta: &candle_core::Tensor,
    _state: &mut candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("gated_delta_rule_recurrence_metal requires the metal feature")
}

// ============================================================================
// Public API: chunked_gated_delta_rule_recurrence (prefill optimization)
// ============================================================================

/// Chunked gated delta rule recurrence on Metal (prefill optimization).
///
/// Processes prefill tokens in 32-token chunks instead of one at a time.
/// Same interface as `gated_delta_rule_recurrence_metal`.
///
/// q, k: [BH, S, K]  v: [BH, S, V]  g, beta: [BH, S]
/// state: [BH, K, V] (mutated in-place)
/// Returns: output [BH, S, V]
#[cfg(feature = "metal")]
pub fn chunked_gated_delta_rule_recurrence_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;

    let (bh, seq_len, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;

    let Device::Metal(dev) = q.device() else {
        candle_core::bail!("chunked_gated_delta_rule_recurrence_metal: expected Metal device");
    };

    // BT=32 for all Metal variants (fits 32KB threadgroup memory)
    let kernel_name = match k_dim {
        128 => "chunked_gated_delta_rule_32_128_64",
        64 => "chunked_gated_delta_rule_32_64_64",
        _ => {
            // Fallback to sequential kernel for unsupported k_dim
            return gated_delta_rule_recurrence_metal(&q, &k, &v, &g, &beta, state);
        }
    };
    let bv = 64usize;

    let pipeline = load_pipeline(dev.device(), kernel_name)?;

    // Allocate output
    let output = Tensor::zeros((bh, seq_len, v_dim), DType::F32, q.device())?;

    let (q_buf, q_off) = metal_buffer_and_offset(&q)?;
    let (k_buf, k_off) = metal_buffer_and_offset(&k)?;
    let (v_buf, v_off) = metal_buffer_and_offset(&v)?;
    let (g_buf, g_off) = metal_buffer_and_offset(&g)?;
    let (beta_buf, beta_off) = metal_buffer_and_offset(&beta)?;
    let (state_buf, state_off) = metal_buffer_and_offset(state)?;
    let (out_buf, out_off) = metal_buffer_and_offset(&output)?;

    let encoder = dev.command_encoder()?;
    encoder.set_label("gdn-chunked-recurrence");
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(&q_buf), q_off);
    encoder.set_input_buffer(1, Some(&k_buf), k_off);
    encoder.set_input_buffer(2, Some(&v_buf), v_off);
    encoder.set_input_buffer(3, Some(&g_buf), g_off);
    encoder.set_input_buffer(4, Some(&beta_buf), beta_off);
    encoder.set_output_buffer(5, Some(&state_buf), state_off);
    encoder.set_output_buffer(6, Some(&out_buf), out_off);

    let seq_len_i32 = seq_len as i32;
    let v_dim_i32 = v_dim as i32;
    encoder.set_bytes(7, &seq_len_i32);
    encoder.set_bytes(8, &v_dim_i32);

    let grid_x = v_dim.div_ceil(bv);
    let thread_groups = MTLSize {
        width: grid_x,
        height: bh,
        depth: 1,
    };
    let threads_per_group = MTLSize {
        width: bv,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    Ok(output)
}

#[cfg(not(feature = "metal"))]
#[allow(dead_code)]
pub fn chunked_gated_delta_rule_recurrence_metal(
    _q: &candle_core::Tensor,
    _k: &candle_core::Tensor,
    _v: &candle_core::Tensor,
    _g: &candle_core::Tensor,
    _beta: &candle_core::Tensor,
    _state: &mut candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("chunked_gated_delta_rule_recurrence_metal requires the metal feature")
}

// ============================================================================
// Public API: causal_conv1d
// ============================================================================

/// Single-token decode against the recurrent pool: row `slots[b]` is updated in place.
///
/// q, k: [B*H, K]  v: [B*H, V]  g, beta: [B*H]  state_pool: [P, H, K, V] f32  slots: [B] u32
/// Returns: output [B*H, V] f32
#[cfg(feature = "metal")]
pub fn gated_delta_rule_pooled_decode_metal(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state_pool: &Tensor,
    slots: &Tensor,
) -> Result<Tensor> {
    const BV: usize = 64;
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let g = g.contiguous()?;
    let beta = beta.contiguous()?;
    let (bh, k_dim) = q.dims2()?;
    let v_dim = v.dim(1)?;
    let num_heads = state_pool.dim(1)?;
    let Device::Metal(dev) = q.device() else {
        candle_core::bail!("gated_delta_rule_pooled_decode_metal: expected Metal device");
    };
    let kernel_name = match k_dim {
        128 => "gated_delta_rule_pooled_decode_128_64",
        64 => "gated_delta_rule_pooled_decode_64_64",
        _ => candle_core::bail!("gated_delta_rule_pooled_decode_metal: unsupported k_dim {k_dim}"),
    };
    let pipeline = load_pipeline(dev.device(), kernel_name)?;
    let output = Tensor::zeros((bh, v_dim), DType::F32, q.device())?;

    let (q_buf, q_off) = metal_buffer_and_offset(&q)?;
    let (k_buf, k_off) = metal_buffer_and_offset(&k)?;
    let (v_buf, v_off) = metal_buffer_and_offset(&v)?;
    let (g_buf, g_off) = metal_buffer_and_offset(&g)?;
    let (beta_buf, beta_off) = metal_buffer_and_offset(&beta)?;
    let (state_buf, state_off) = metal_buffer_and_offset(state_pool)?;
    let (out_buf, out_off) = metal_buffer_and_offset(&output)?;
    let (slots_buf, slots_off) = metal_buffer_and_offset(slots)?;

    let encoder = dev.command_encoder()?;
    encoder.set_label("gdn-pooled-decode");
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_input_buffer(0, Some(&q_buf), q_off);
    encoder.set_input_buffer(1, Some(&k_buf), k_off);
    encoder.set_input_buffer(2, Some(&v_buf), v_off);
    encoder.set_input_buffer(3, Some(&g_buf), g_off);
    encoder.set_input_buffer(4, Some(&beta_buf), beta_off);
    encoder.set_output_buffer(5, Some(&state_buf), state_off);
    encoder.set_output_buffer(6, Some(&out_buf), out_off);
    encoder.set_bytes(7, &(v_dim as i32));
    encoder.set_input_buffer(8, Some(&slots_buf), slots_off);
    encoder.set_bytes(9, &(num_heads as i32));
    encoder.dispatch_thread_groups(
        MTLSize {
            width: v_dim.div_ceil(BV),
            height: bh,
            depth: 1,
        },
        MTLSize {
            width: BV,
            height: 1,
            depth: 1,
        },
    );
    Ok(output)
}

#[cfg(feature = "metal")]
#[repr(C)]
struct GdnPrefillParams {
    seq_len: i32,
    num_k_heads: i32,
    num_v_heads: i32,
    head_v_dim: i32,
    tiled_v_heads: i32,
    use_slots: i32,
    b_stride: [i32; 3],
    a_stride: [i32; 3],
}

#[cfg(feature = "metal")]
pub struct RecurrenceFromConvolved<'a> {
    /// [B, T, 2 * key_dim + value_dim]
    pub mixed_qkv: &'a Tensor,
    /// [B, T, Hv], may be strided
    pub b: &'a Tensor,
    /// [B, T, Hv], may be strided
    pub a: &'a Tensor,
    /// [Hv], f32
    pub a_log: &'a Tensor,
    /// [Hv], f32
    pub dt_bias: &'a Tensor,
    /// [rows, Hv, K, V], f32, updated in place: the pool when `slots` is set, else the batch's gathered rows
    pub state: &'a Tensor,
    pub slots: Option<&'a Tensor>,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub tiled_v_heads: bool,
}

/// Fused GDN recurrence returning [B, T, Hv, V] in the projection dtype.
#[cfg(feature = "metal")]
pub fn gdn_recurrence_from_convolved_metal(args: RecurrenceFromConvolved<'_>) -> Result<Tensor> {
    let (batch_size, seq_len, _) = args.mixed_qkv.dims3()?;
    let (_, num_v_heads, _, head_v_dim) = args.state.dims4()?;
    let columns = prefill_value_columns(batch_size, seq_len, num_v_heads, head_v_dim);
    gdn_recurrence_from_convolved_metal_with_columns(args, columns)
}

#[cfg(feature = "metal")]
fn prefill_value_columns(batch: usize, tokens: usize, heads: usize, value_dim: usize) -> usize {
    let threadgroups =
        batch * heads * value_dim.div_ceil(PREFILL_VALUE_COLUMNS * SIMDGROUPS_PER_THREADGROUP);
    if tokens > 1 && value_dim >= MIN_GROUPED_VALUE_DIM && threadgroups >= MIN_GROUPED_THREADGROUPS
    {
        PREFILL_VALUE_COLUMNS
    } else {
        1
    }
}

#[cfg(feature = "metal")]
fn gdn_recurrence_from_convolved_metal_with_columns(
    args: RecurrenceFromConvolved<'_>,
    columns: usize,
) -> Result<Tensor> {
    let mixed = args.mixed_qkv.contiguous()?;
    let (batch_size, seq_len, _) = mixed.dims3()?;
    let (_, num_v_heads, _, head_v_dim) = args.state.dims4()?;
    let type_name = match mixed.dtype() {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        dtype => {
            candle_core::bail!("gdn_recurrence_from_convolved_metal: unsupported dtype {dtype:?}")
        }
    };
    if args.b.dtype() != mixed.dtype() || args.a.dtype() != mixed.dtype() {
        candle_core::bail!(
            "gdn_recurrence_from_convolved_metal: b/a must match the projection dtype"
        );
    }
    let Device::Metal(dev) = mixed.device() else {
        candle_core::bail!("gdn_recurrence_from_convolved_metal: expected Metal device");
    };
    let name = match columns {
        1 => format!("gdn_prefill_from_convolved_{type_name}_{}", args.head_k_dim),
        PREFILL_VALUE_COLUMNS => format!(
            "gdn_prefill_from_convolved_grouped_{type_name}_{}_{columns}",
            args.head_k_dim
        ),
        _ => candle_core::bail!("unsupported Metal GDN value columns {columns}"),
    };
    let pipeline = load_pipeline(dev.device(), &name)?;
    // Every element is written by the kernel.
    let output = unsafe {
        Tensor::empty(
            (batch_size, seq_len, num_v_heads, head_v_dim),
            mixed.dtype(),
            mixed.device(),
        )?
    };
    let to_i32 = |v: usize| i32::try_from(v).map_err(candle_core::Error::wrap);
    let strides3 = |t: &Tensor| -> Result<[i32; 3]> {
        let st = t.stride();
        Ok([to_i32(st[0])?, to_i32(st[1])?, to_i32(st[2])?])
    };
    let params = GdnPrefillParams {
        seq_len: to_i32(seq_len)?,
        num_k_heads: to_i32(args.num_k_heads)?,
        num_v_heads: to_i32(num_v_heads)?,
        head_v_dim: to_i32(head_v_dim)?,
        tiled_v_heads: i32::from(args.tiled_v_heads),
        use_slots: i32::from(args.slots.is_some()),
        b_stride: strides3(args.b)?,
        a_stride: strides3(args.a)?,
    };

    let (mixed_buf, mixed_off) = metal_buffer_and_offset(&mixed)?;
    let (b_buf, b_off) = metal_buffer_and_offset(args.b)?;
    let (a_buf, a_off) = metal_buffer_and_offset(args.a)?;
    let (a_log_buf, a_log_off) = metal_buffer_and_offset(args.a_log)?;
    let (dt_buf, dt_off) = metal_buffer_and_offset(args.dt_bias)?;
    let (state_buf, state_off) = metal_buffer_and_offset(args.state)?;
    let (out_buf, out_off) = metal_buffer_and_offset(&output)?;
    let (slots_buf, slots_off) = match args.slots {
        Some(slots) => metal_buffer_and_offset(slots)?,
        None => (a_log_buf.clone(), a_log_off),
    };

    let encoder = dev.command_encoder()?;
    encoder.set_label("gdn-recurrence-from-convolved");
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_input_buffer(0, Some(&mixed_buf), mixed_off);
    encoder.set_input_buffer(1, Some(&b_buf), b_off);
    encoder.set_input_buffer(2, Some(&a_buf), a_off);
    encoder.set_input_buffer(3, Some(&a_log_buf), a_log_off);
    encoder.set_input_buffer(4, Some(&dt_buf), dt_off);
    encoder.set_output_buffer(5, Some(&state_buf), state_off);
    encoder.set_output_buffer(6, Some(&out_buf), out_off);
    encoder.set_input_buffer(7, Some(&slots_buf), slots_off);
    encoder.set_bytes(8, &params);
    encoder.dispatch_threads(
        MTLSize {
            width: SIMD_WIDTH,
            height: head_v_dim.div_ceil(columns),
            depth: batch_size * num_v_heads,
        },
        MTLSize {
            width: SIMD_WIDTH,
            height: SIMDGROUPS_PER_THREADGROUP,
            depth: 1,
        },
    );
    Ok(output)
}

/// Gated RMSNorm over the last dim: x * rsqrt(mean(x^2) + eps) * weight * silu(gate).
#[cfg(feature = "metal")]
pub fn rmsnorm_gated_metal(x: &Tensor, gate: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let x = x.contiguous()?;
    let gate = gate.reshape(x.shape())?.contiguous()?;
    let weight = weight.to_dtype(x.dtype())?.contiguous()?;
    let dim = x.dim(candle_core::D::Minus1)?;
    let rows = x.elem_count() / dim;
    let type_name = match x.dtype() {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "bfloat",
        dtype => candle_core::bail!("rmsnorm_gated_metal: unsupported dtype {dtype:?}"),
    };
    let Device::Metal(dev) = x.device() else {
        candle_core::bail!("rmsnorm_gated_metal: expected Metal device");
    };
    let pipeline = load_pipeline(dev.device(), &format!("gdn_rmsnorm_gated_{type_name}"))?;
    // Every element is written by the kernel.
    let output = unsafe { Tensor::empty(x.shape(), x.dtype(), x.device())? };

    let (x_buf, x_off) = metal_buffer_and_offset(&x)?;
    let (g_buf, g_off) = metal_buffer_and_offset(&gate)?;
    let (w_buf, w_off) = metal_buffer_and_offset(&weight)?;
    let (out_buf, out_off) = metal_buffer_and_offset(&output)?;

    let encoder = dev.command_encoder()?;
    encoder.set_label("gdn-rmsnorm-gated");
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_input_buffer(0, Some(&x_buf), x_off);
    encoder.set_input_buffer(1, Some(&g_buf), g_off);
    encoder.set_input_buffer(2, Some(&w_buf), w_off);
    encoder.set_output_buffer(3, Some(&out_buf), out_off);
    encoder.set_bytes(4, &(dim as i32));
    encoder.set_bytes(5, &(eps as f32));
    encoder.dispatch_thread_groups(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(output)
}

/// Single-token conv update against the conv pool: row `slots[b]` is shifted in place.
///
/// x: [B, conv_dim, 1]  weight: [conv_dim, kernel_size]  conv_pool: [P, conv_dim, kernel_size]
/// Returns: output [B, conv_dim, 1]
#[cfg(feature = "metal")]
pub fn causal_conv1d_update_pooled_metal(
    x: &Tensor,
    weight: &Tensor,
    conv_pool: &Tensor,
    slots: &Tensor,
    kernel_size: usize,
) -> Result<Tensor> {
    const THREADS: usize = 256;
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let type_suffix = match x.dtype() {
        DType::F16 => "half",
        DType::BF16 => "bfloat16_t",
        dtype => {
            candle_core::bail!("causal_conv1d_update_pooled_metal: unsupported dtype {dtype:?}")
        }
    };
    let Device::Metal(dev) = x.device() else {
        candle_core::bail!("causal_conv1d_update_pooled_metal: expected Metal device");
    };
    let (batch_size, conv_dim, _) = x.dims3()?;
    let pipeline = load_pipeline(
        dev.device(),
        &format!("causal_conv1d_update_pooled_{type_suffix}"),
    )?;
    let output = Tensor::zeros((batch_size, conv_dim, 1), x.dtype(), x.device())?;

    let (x_buf, x_off) = metal_buffer_and_offset(&x)?;
    let (w_buf, w_off) = metal_buffer_and_offset(&weight)?;
    let (cs_buf, cs_off) = metal_buffer_and_offset(conv_pool)?;
    let (out_buf, out_off) = metal_buffer_and_offset(&output)?;
    let (slots_buf, slots_off) = metal_buffer_and_offset(slots)?;

    let encoder = dev.command_encoder()?;
    encoder.set_label("gdn-conv-update-pooled");
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_input_buffer(0, Some(&x_buf), x_off);
    encoder.set_input_buffer(1, Some(&w_buf), w_off);
    encoder.set_output_buffer(2, Some(&cs_buf), cs_off);
    encoder.set_output_buffer(3, Some(&out_buf), out_off);
    encoder.set_bytes(4, &(batch_size as i32));
    encoder.set_bytes(5, &(conv_dim as i32));
    encoder.set_bytes(6, &(kernel_size as i32));
    encoder.set_input_buffer(7, Some(&slots_buf), slots_off);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: conv_dim.div_ceil(THREADS),
            height: batch_size,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(output)
}

/// Causal conv1d on Metal (both update and full paths).
///
/// x (update): [B, conv_dim, 1] or (full): [B, conv_dim, S]
/// weight: [conv_dim, kernel_size]
/// conv_state: [B, conv_dim, kernel_size]
/// Returns: (output, new_conv_state)
#[cfg(feature = "metal")]
pub fn causal_conv1d_metal(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    is_update: bool,
    kernel_size: usize,
) -> Result<(Tensor, Tensor)> {
    let mode = if is_update {
        ConvMode::Update
    } else {
        ConvMode::PrefillChannelMajor
    };
    causal_conv1d_metal_impl(x, weight, conv_state, mode, kernel_size)
}

#[cfg(feature = "metal")]
enum ConvMode {
    Update,
    PrefillChannelMajor,
    PrefillTokenMajor,
}

/// Prefill with token-major activations and channel-major convolution state.
#[cfg(feature = "metal")]
pub fn causal_conv1d_prefill_metal(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
) -> Result<(Tensor, Tensor)> {
    causal_conv1d_metal_impl(
        x,
        weight,
        conv_state,
        ConvMode::PrefillTokenMajor,
        kernel_size,
    )
}

#[cfg(feature = "metal")]
fn causal_conv1d_metal_impl(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    mode: ConvMode,
    kernel_size: usize,
) -> Result<(Tensor, Tensor)> {
    let x = x.contiguous()?;
    let weight = weight.contiguous()?;
    let conv_state = conv_state.contiguous()?;

    let dtype = x.dtype();
    let type_suffix = match dtype {
        DType::F16 => "half",
        DType::BF16 => "bfloat16_t",
        _ => candle_core::bail!(
            "causal_conv1d_metal: unsupported dtype {dtype:?}, expected F16 or BF16"
        ),
    };

    let Device::Metal(dev) = x.device() else {
        candle_core::bail!("causal_conv1d_metal: expected Metal device");
    };

    if matches!(mode, ConvMode::Update) {
        let batch_size = x.dim(0)?;
        let conv_dim = x.dim(1)?;

        let kernel_name = format!("causal_conv1d_update_{type_suffix}");
        let pipeline = load_pipeline(dev.device(), &kernel_name)?;

        let output = Tensor::zeros((batch_size, conv_dim, 1), dtype, x.device())?;
        let new_conv_state = conv_state.clone();

        let (x_buf, x_off) = metal_buffer_and_offset(&x)?;
        let (w_buf, w_off) = metal_buffer_and_offset(&weight)?;
        let (cs_buf, cs_off) = metal_buffer_and_offset(&new_conv_state)?;
        let (out_buf, out_off) = metal_buffer_and_offset(&output)?;

        let encoder = dev.command_encoder()?;
        encoder.set_label("gdn-conv-update");
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_input_buffer(0, Some(&x_buf), x_off);
        encoder.set_input_buffer(1, Some(&w_buf), w_off);
        encoder.set_output_buffer(2, Some(&cs_buf), cs_off);
        encoder.set_output_buffer(3, Some(&out_buf), out_off);

        let bs = batch_size as i32;
        let cd = conv_dim as i32;
        let ks = kernel_size as i32;
        encoder.set_bytes(4, &bs);
        encoder.set_bytes(5, &cd);
        encoder.set_bytes(6, &ks);

        let thread_groups = MTLSize {
            width: conv_dim.div_ceil(256),
            height: batch_size,
            depth: 1,
        };
        let threads_per_group = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);

        Ok((output, new_conv_state))
    } else {
        let batch_size = x.dim(0)?;
        let token_major = matches!(mode, ConvMode::PrefillTokenMajor);
        let conv_dim = x.dim(if token_major { 2 } else { 1 })?;
        let seq_len = x.dim(if token_major { 1 } else { 2 })?;
        let layout_suffix = if token_major { "_token_major" } else { "" };

        let conv_name = format!("causal_conv1d_full{layout_suffix}_{type_suffix}");
        let conv_pipeline = load_pipeline(dev.device(), &conv_name)?;

        // Both kernels write every element of their respective outputs.
        let output = unsafe { Tensor::empty(x.shape(), dtype, x.device())? };

        let (x_buf, x_off) = metal_buffer_and_offset(&x)?;
        let (w_buf, w_off) = metal_buffer_and_offset(&weight)?;
        let (prior_buf, prior_off) = metal_buffer_and_offset(&conv_state)?;
        let (out_buf, out_off) = metal_buffer_and_offset(&output)?;

        {
            let encoder = dev.command_encoder()?;
            encoder.set_label("gdn-conv");
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&conv_pipeline);

            encoder.set_input_buffer(0, Some(&x_buf), x_off);
            encoder.set_input_buffer(1, Some(&w_buf), w_off);
            encoder.set_input_buffer(2, Some(&prior_buf), prior_off);
            encoder.set_output_buffer(3, Some(&out_buf), out_off);

            let bs = batch_size as i32;
            let cd = conv_dim as i32;
            let sl = seq_len as i32;
            let ks = kernel_size as i32;
            encoder.set_bytes(4, &bs);
            encoder.set_bytes(5, &cd);
            encoder.set_bytes(6, &sl);
            encoder.set_bytes(7, &ks);

            let thread_groups = MTLSize {
                width: conv_dim.div_ceil(256),
                height: seq_len,
                depth: batch_size,
            };
            let threads_per_group = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(thread_groups, threads_per_group);
        }

        let save_name = format!("save_conv_state{layout_suffix}_{type_suffix}");
        let save_pipeline = load_pipeline(dev.device(), &save_name)?;

        let new_conv_state =
            unsafe { Tensor::empty((batch_size, conv_dim, kernel_size), dtype, x.device())? };
        let (cs_buf, cs_off) = metal_buffer_and_offset(&new_conv_state)?;

        {
            let encoder = dev.command_encoder()?;
            encoder.set_label("gdn-conv-save-state");
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&save_pipeline);

            encoder.set_input_buffer(0, Some(&x_buf), x_off);
            encoder.set_input_buffer(1, Some(&prior_buf), prior_off);
            encoder.set_output_buffer(2, Some(&cs_buf), cs_off);

            let bs = batch_size as i32;
            let cd = conv_dim as i32;
            let sl = seq_len as i32;
            let ks = kernel_size as i32;
            encoder.set_bytes(3, &bs);
            encoder.set_bytes(4, &cd);
            encoder.set_bytes(5, &sl);
            encoder.set_bytes(6, &ks);

            let thread_groups = MTLSize {
                width: conv_dim.div_ceil(256),
                height: batch_size,
                depth: 1,
            };
            let threads_per_group = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            encoder.dispatch_thread_groups(thread_groups, threads_per_group);
        }

        Ok((output, new_conv_state))
    }
}

#[cfg(not(feature = "metal"))]
#[allow(dead_code)]
pub fn causal_conv1d_metal(
    _x: &candle_core::Tensor,
    _weight: &candle_core::Tensor,
    _conv_state: &candle_core::Tensor,
    _is_update: bool,
    _kernel_size: usize,
) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
    candle_core::bail!("causal_conv1d_metal requires the metal feature")
}

// ============================================================================
// Public API: fused_gdn_gating
// ============================================================================

/// Fused GDN gating on Metal.
///
/// b, a: [total_elements] (f16/bf16)
/// a_log, dt_bias: [num_heads] (f32)
/// Returns: (beta, g) in same dtype as input
#[cfg(feature = "metal")]
pub fn fused_gdn_gating_metal(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let b = b.contiguous()?;
    let a = a.contiguous()?;
    let a_log = a_log.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;

    let dtype = b.dtype();
    let type_suffix = match dtype {
        DType::F16 => "half",
        DType::BF16 => "bfloat16_t",
        _ => candle_core::bail!(
            "fused_gdn_gating_metal: unsupported dtype {dtype:?}, expected F16 or BF16"
        ),
    };

    let total_elements = b.elem_count();
    let num_heads = a_log.elem_count();

    let Device::Metal(dev) = b.device() else {
        candle_core::bail!("fused_gdn_gating_metal: expected Metal device");
    };

    let kernel_name = format!("fused_gdn_gating_{type_suffix}");
    let pipeline = load_pipeline(dev.device(), &kernel_name)?;

    let beta_out = Tensor::zeros(b.shape(), dtype, b.device())?;
    let g_out = Tensor::zeros(b.shape(), dtype, b.device())?;

    let (b_buf, b_off) = metal_buffer_and_offset(&b)?;
    let (a_buf, a_off) = metal_buffer_and_offset(&a)?;
    let (alog_buf, alog_off) = metal_buffer_and_offset(&a_log)?;
    let (dtb_buf, dtb_off) = metal_buffer_and_offset(&dt_bias)?;
    let (beta_buf, beta_off) = metal_buffer_and_offset(&beta_out)?;
    let (g_buf, g_off) = metal_buffer_and_offset(&g_out)?;

    let encoder = dev.command_encoder()?;
    encoder.set_label("gdn-gating");
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(&b_buf), b_off);
    encoder.set_input_buffer(1, Some(&a_buf), a_off);
    encoder.set_input_buffer(2, Some(&alog_buf), alog_off);
    encoder.set_input_buffer(3, Some(&dtb_buf), dtb_off);
    encoder.set_output_buffer(4, Some(&beta_buf), beta_off);
    encoder.set_output_buffer(5, Some(&g_buf), g_off);

    let total = total_elements as i32;
    let heads = num_heads as i32;
    encoder.set_bytes(6, &total);
    encoder.set_bytes(7, &heads);

    let thread_groups = MTLSize {
        width: total_elements.div_ceil(256),
        height: 1,
        depth: 1,
    };
    let threads_per_group = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    Ok((beta_out, g_out))
}

#[cfg(not(feature = "metal"))]
#[allow(dead_code)]
pub fn fused_gdn_gating_metal(
    _b: &candle_core::Tensor,
    _a: &candle_core::Tensor,
    _a_log: &candle_core::Tensor,
    _dt_bias: &candle_core::Tensor,
) -> candle_core::Result<(candle_core::Tensor, candle_core::Tensor)> {
    candle_core::bail!("fused_gdn_gating_metal requires the metal feature")
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;

    const TEST_BATCH: usize = 2;
    const TEST_KEY_HEADS: usize = 2;
    const TEST_VALUE_HEADS: usize = 4;
    const TEST_POOL_ROWS: usize = 5;
    const TEST_SLOTS: [u32; TEST_BATCH] = [3, 1];
    const TEST_STATE_TOLERANCE: f32 = 2e-6;

    fn patterned_tensor(shape: impl Into<candle_core::Shape>, seed: usize) -> Result<Tensor> {
        let shape = shape.into();
        let data = (0..shape.elem_count())
            .map(|index| {
                (f32::from(u8::try_from((index * 13 + seed) % 53).unwrap()) - 26.0) * 0.013
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(data, shape, &Device::Cpu)
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, tag: &str) {
        assert_eq!(actual.len(), expected.len(), "{tag}");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
                "{tag}: element {index}, actual={actual}, expected={expected}"
            );
        }
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn grouped_prefill_preserves_outputs_state_and_pool_rows() -> Result<()> {
        let device = Device::new_metal(0)?;
        for (seq_index, seq_len) in [1, 7, 53, 129].into_iter().enumerate() {
            for head_k_dim in [64, 128] {
                for (dtype, head_v_dim, pooled, tolerance) in [
                    (DType::F32, 7, false, 2e-6),
                    (DType::F16, 65, true, 1e-4),
                    (DType::BF16, 128, true, 1e-3),
                ] {
                    let conv_dim = 2 * TEST_KEY_HEADS * head_k_dim + TEST_VALUE_HEADS * head_v_dim;
                    let mixed = patterned_tensor((TEST_BATCH, seq_len, conv_dim), 1)?
                        .to_dtype(dtype)?
                        .to_device(&device)?;
                    let gates = patterned_tensor((TEST_BATCH, seq_len, TEST_VALUE_HEADS * 2), 2)?
                        .to_dtype(dtype)?
                        .to_device(&device)?;
                    let b = gates.narrow(2, 0, TEST_VALUE_HEADS)?;
                    let a = gates.narrow(2, TEST_VALUE_HEADS, TEST_VALUE_HEADS)?;
                    let a_log = patterned_tensor(TEST_VALUE_HEADS, 3)?.to_device(&device)?;
                    let dt_bias = patterned_tensor(TEST_VALUE_HEADS, 4)?.to_device(&device)?;
                    let rows = if pooled { TEST_POOL_ROWS } else { TEST_BATCH };
                    let initial_state =
                        patterned_tensor((rows, TEST_VALUE_HEADS, head_k_dim, head_v_dim), 5)?;
                    let slots = pooled
                        .then(|| Tensor::new(&TEST_SLOTS, &device))
                        .transpose()?;
                    let tiled_v_heads = seq_index % 2 == 0;
                    let run = |columns| -> Result<(Vec<f32>, Vec<f32>)> {
                        let state = initial_state.to_device(&device)?;
                        let output = gdn_recurrence_from_convolved_metal_with_columns(
                            RecurrenceFromConvolved {
                                mixed_qkv: &mixed,
                                b: &b,
                                a: &a,
                                a_log: &a_log,
                                dt_bias: &dt_bias,
                                state: &state,
                                slots: slots.as_ref(),
                                num_k_heads: TEST_KEY_HEADS,
                                head_k_dim,
                                tiled_v_heads,
                            },
                            columns,
                        )?;
                        Ok((
                            output.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?,
                            state.flatten_all()?.to_vec1()?,
                        ))
                    };
                    let (expected_output, expected_state) = run(1)?;
                    let initial_state = initial_state.flatten_all()?.to_vec1::<f32>()?;
                    let row_size = TEST_VALUE_HEADS * head_k_dim * head_v_dim;
                    let (output, state) = run(PREFILL_VALUE_COLUMNS)?;
                    let tag = format!(
                        "{dtype:?} T={seq_len} K={head_k_dim} V={head_v_dim} \
                         tiled={tiled_v_heads} pooled={pooled}"
                    );
                    assert_close(&output, &expected_output, tolerance, &tag);
                    assert_close(&state, &expected_state, TEST_STATE_TOLERANCE, &tag);
                    if pooled {
                        for row in 0..TEST_POOL_ROWS {
                            if TEST_SLOTS.contains(&(row as u32)) {
                                continue;
                            }
                            let range = row * row_size..(row + 1) * row_size;
                            assert_eq!(state[range.clone()], initial_state[range], "{tag}");
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn grouped_prefill_requires_enough_parallel_work() {
        assert_eq!(prefill_value_columns(1, 1, 48, 128), 1);
        assert_eq!(prefill_value_columns(1, 53, 48, 128), PREFILL_VALUE_COLUMNS);
        assert_eq!(
            prefill_value_columns(8, 512, 48, 128),
            PREFILL_VALUE_COLUMNS
        );
        assert_eq!(prefill_value_columns(1, 512, 8, 64), 1);
        assert_eq!(prefill_value_columns(8, 512, 8, 64), PREFILL_VALUE_COLUMNS);
    }

    #[test]
    #[ignore = "requires a Metal device"]
    fn warmup_covers_recurrence_variants_and_reuses_pipelines() -> Result<()> {
        let device = Device::new_metal(0)?;
        let raw = device.as_metal_device()?.device();
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for head_k_dim in [64, 128, 256] {
                warmup_kernels(raw, dtype, head_k_dim)?;
            }
        }
        let first = load_pipeline(raw, "gdn_prefill_from_convolved_bfloat_128")?;
        let second = load_pipeline(raw, "gdn_prefill_from_convolved_bfloat_128")?;
        assert!(std::ptr::eq(first.as_ref(), second.as_ref()));
        Ok(())
    }
}
