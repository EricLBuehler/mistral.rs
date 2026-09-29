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
    Buffer, ComputeCommandEncoder, ComputePipeline, Device as MetalRawDevice, Library,
};

#[cfg(feature = "metal")]
use objc2_metal::{MTLCompileOptions, MTLLanguageVersion, MTLMathMode, MTLSize};

#[cfg(feature = "metal")]
use std::collections::HashMap;

#[cfg(feature = "metal")]
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "metal")]
static GDN_LIBRARY: OnceLock<Library> = OnceLock::new();

#[cfg(feature = "metal")]
type Pipelines = HashMap<String, ComputePipeline>;

#[cfg(feature = "metal")]
static GDN_PIPELINES: OnceLock<RwLock<Pipelines>> = OnceLock::new();

#[cfg(feature = "metal")]
const GDN_METAL_SOURCE: &str = include_str!("kernels/gdn.metal");

#[cfg(feature = "metal")]
fn load_gdn_library(device: &MetalRawDevice) -> Result<Library> {
    if let Some(lib) = GDN_LIBRARY.get() {
        return Ok(lib.clone());
    }
    let compile_options = {
        let opts = MTLCompileOptions::new();
        opts.setLanguageVersion(MTLLanguageVersion::Version3_1);
        opts.setMathMode(MTLMathMode::Fast);
        opts
    };
    let lib = device
        .new_library_with_source(GDN_METAL_SOURCE, Some(&compile_options))
        .map_err(|e| {
            candle_core::Error::Msg(format!("Failed to compile GDN Metal kernels: {e}"))
        })?;
    Ok(GDN_LIBRARY.get_or_init(|| lib).clone())
}

#[cfg(feature = "metal")]
fn load_pipeline(device: &MetalRawDevice, name: &str) -> Result<ComputePipeline> {
    let pipelines_lock = GDN_PIPELINES.get_or_init(|| RwLock::new(Pipelines::new()));

    // Check read lock first
    {
        let pipelines = pipelines_lock
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline cache: {e}")))?;
        if let Some(pipeline) = pipelines.get(name) {
            return Ok(pipeline.clone());
        }
    }

    // Not found, compile and insert
    let lib = load_gdn_library(device)?;
    let func = lib.get_function(name, None).map_err(|e| {
        candle_core::Error::Msg(format!("Failed to load Metal function '{name}': {e}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create pipeline for '{name}': {e}"))
        })?;

    let mut pipelines = pipelines_lock.write().map_err(|e| {
        candle_core::Error::Msg(format!("Failed to lock pipeline cache for write: {e}"))
    })?;
    pipelines.insert(name.to_string(), pipeline.clone());
    Ok(pipeline)
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

/// GDN recurrence over T >= 1 tokens from the convolved projection (norms, gates and state update in one
/// kernel, one simdgroup per value column). Returns y: [B, T, Hv, V] in the projection dtype.
#[cfg(feature = "metal")]
pub fn gdn_recurrence_from_convolved_metal(args: RecurrenceFromConvolved<'_>) -> Result<Tensor> {
    const SIMD_WIDTH: usize = 32;
    const COLUMNS_PER_THREADGROUP: usize = 4;
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
    let pipeline = load_pipeline(
        dev.device(),
        &format!("gdn_prefill_from_convolved_{type_name}_{}", args.head_k_dim),
    )?;
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
            height: head_v_dim,
            depth: batch_size * num_v_heads,
        },
        MTLSize {
            width: SIMD_WIDTH,
            height: COLUMNS_PER_THREADGROUP,
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

    if is_update {
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
        let conv_dim = x.dim(1)?;
        let seq_len = x.dim(2)?;

        // Full convolution kernel
        let conv_name = format!("causal_conv1d_full_{type_suffix}");
        let conv_pipeline = load_pipeline(dev.device(), &conv_name)?;

        let output = Tensor::zeros((batch_size, conv_dim, seq_len), dtype, x.device())?;

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

        // Save conv state kernel
        let save_name = format!("save_conv_state_{type_suffix}");
        let save_pipeline = load_pipeline(dev.device(), &save_name)?;

        let new_conv_state = Tensor::zeros((batch_size, conv_dim, kernel_size), dtype, x.device())?;
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
