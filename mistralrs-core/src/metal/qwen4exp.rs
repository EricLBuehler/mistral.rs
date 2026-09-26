//! Metal kernels for Qwen4Exp-specific operations (Completion Phase 7).
//!
//! Mirrors the CUDA/Metal GDN kernel structure: one lazily compiled Metal
//! library with cached compute pipelines, plus a non-metal stub so callers can
//! cfg-gate the fast paths.

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
use objc2_metal::{MTLCompileOptions, MTLLanguageVersion, MTLSize};

#[cfg(feature = "metal")]
use std::collections::HashMap;

#[cfg(feature = "metal")]
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "metal")]
static QWEN4EXP_LIBRARY: OnceLock<Library> = OnceLock::new();

#[cfg(feature = "metal")]
type Pipelines = HashMap<String, ComputePipeline>;

#[cfg(feature = "metal")]
static QWEN4EXP_PIPELINES: OnceLock<RwLock<Pipelines>> = OnceLock::new();

#[cfg(feature = "metal")]
const QWEN4EXP_METAL_SOURCE: &str = include_str!("kernels/qwen4exp.metal");

#[cfg(feature = "metal")]
fn load_qwen4exp_library(device: &MetalRawDevice) -> Result<Library> {
    if let Some(lib) = QWEN4EXP_LIBRARY.get() {
        return Ok(lib.clone());
    }
    let compile_options = {
        let opts = MTLCompileOptions::new();
        opts.setLanguageVersion(MTLLanguageVersion::Version3_1);
        opts
    };
    let lib = device
        .new_library_with_source(QWEN4EXP_METAL_SOURCE, Some(&compile_options))
        .map_err(|e| {
            candle_core::Error::Msg(format!("Failed to compile Qwen4Exp Metal kernels: {e}"))
        })?;
    Ok(QWEN4EXP_LIBRARY.get_or_init(|| lib).clone())
}

#[cfg(feature = "metal")]
fn load_pipeline(device: &MetalRawDevice, name: &str) -> Result<ComputePipeline> {
    let pipelines_lock = QWEN4EXP_PIPELINES.get_or_init(|| RwLock::new(Pipelines::new()));

    {
        let pipelines = pipelines_lock
            .read()
            .map_err(|e| candle_core::Error::Msg(format!("Failed to lock pipeline cache: {e}")))?;
        if let Some(pipeline) = pipelines.get(name) {
            return Ok(pipeline.clone());
        }
    }

    let lib = load_qwen4exp_library(device)?;
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
// Public API: hc_rmsnorm_flatten
// ============================================================================

/// Fused Qwen4Exp hyper-connection mixer prefix on Metal: per-stream RMS
/// normalization over the grouped residual `[rows, streams, hidden]` plus the
/// flattened gamma multiply, emitting the projection input in the flattened
/// `[rows, streams * hidden]` layout with the activation dtype preserved.
///
/// `residual` and `weight` must be contiguous Metal tensors sharing one dtype
/// (F32, F16, or BF16); RMS statistics are computed in F32 exactly like the
/// composed Candle path.
#[cfg(feature = "metal")]
pub fn hc_rmsnorm_flatten_metal(residual: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;

    let dtype = residual.dtype();
    let type_suffix = match dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "bfloat16_t",
        _ => candle_core::bail!(
            "hc_rmsnorm_flatten_metal: unsupported dtype {dtype:?}, expected F32, F16, or BF16"
        ),
    };

    let dims = residual.dims();
    if dims.len() < 2 {
        candle_core::bail!("hc_rmsnorm_flatten_metal: expected rank >= 2 residual, got {dims:?}");
    }
    let hidden = dims[dims.len() - 1];
    let streams = dims[dims.len() - 2];
    let wide = streams
        .checked_mul(hidden)
        .ok_or_else(|| candle_core::Error::msg("hc_rmsnorm_flatten_metal: width overflow"))?;
    if wide == 0 || weight.elem_count() != wide {
        candle_core::bail!("hc_rmsnorm_flatten_metal: incompatible residual/weight shapes");
    }

    let Device::Metal(dev) = residual.device() else {
        candle_core::bail!("hc_rmsnorm_flatten_metal: expected Metal device");
    };

    let kernel_name = format!("hc_rmsnorm_flatten_{type_suffix}");
    let pipeline = load_pipeline(dev.device(), &kernel_name)?;

    // The kernel writes flat elements; the projection input expects the grouped
    // residual merged on the last two dims — [.., streams * hidden] — matching the
    // composed path's flattened output shape.
    let mut flat_shape = dims[..dims.len() - 2].to_vec();
    flat_shape.push(wide);
    let output = Tensor::zeros(flat_shape, dtype, residual.device())?;

    let (x_buf, x_off) = metal_buffer_and_offset(&residual)?;
    let (w_buf, w_off) = metal_buffer_and_offset(&weight)?;
    let (o_buf, o_off) = metal_buffer_and_offset(&output)?;

    let encoder = dev.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(&x_buf), x_off);
    encoder.set_input_buffer(1, Some(&w_buf), w_off);
    encoder.set_output_buffer(2, Some(&o_buf), o_off);

    let hidden_u32 = hidden as u32;
    let eps_f32 = eps as f32;
    let streams_u32 = streams as u32;
    encoder.set_bytes(3, &hidden_u32);
    encoder.set_bytes(4, &eps_f32);
    encoder.set_bytes(5, &streams_u32);

    let groups = residual.elem_count() / hidden;
    let thread_groups = MTLSize {
        width: groups,
        height: 1,
        depth: 1,
    };
    let threads_per_group = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    Ok(output)
}

/// Stub for non-metal builds.
#[cfg(not(feature = "metal"))]
#[allow(dead_code)]
pub fn hc_rmsnorm_flatten_metal(
    _residual: &candle_core::Tensor,
    _weight: &candle_core::Tensor,
    _eps: f64,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("hc_rmsnorm_flatten_metal requires the metal feature")
}

// ============================================================================
// Public API: ple_conv1d
// ============================================================================

/// Fused Qwen4Exp PLE causal dilated depthwise convolution on Metal over the
/// concatenated per-sequence state `[history_tokens + tokens, channels]` and the
/// `[kernel_size, channels]` tap kernel, emitting the SiLU-activated
/// `[tokens, channels]` output with the activation dtype preserved.
///
/// `state_and_input` and `kernel` must be contiguous Metal tensors sharing one
/// dtype (F32, F16, or BF16); taps accumulate in F32 exactly like the composed
/// Candle path's tap order (tap 0 is the oldest visible row,
/// `lookback = (kernel_size - 1 - tap) * dilation`), and the caller must satisfy
/// `(kernel_size - 1) * dilation <= history_tokens` so every read stays in bounds.
#[cfg(feature = "metal")]
pub fn ple_conv1d_metal(
    state_and_input: &Tensor,
    kernel: &Tensor,
    history_tokens: usize,
    kernel_size: usize,
    dilation: usize,
) -> Result<Tensor> {
    let state_and_input = state_and_input.contiguous()?;
    let kernel = kernel.contiguous()?;

    let dtype = state_and_input.dtype();
    let type_suffix = match dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::BF16 => "bfloat16_t",
        _ => candle_core::bail!(
            "ple_conv1d_metal: unsupported dtype {dtype:?}, expected F32, F16, or BF16"
        ),
    };
    if kernel.dtype() != dtype {
        candle_core::bail!("ple_conv1d_metal: state and kernel dtypes differ");
    }

    let (rows, channels) = state_and_input.dims2()?;
    if channels == 0 {
        candle_core::bail!("ple_conv1d_metal: channels must be non-zero");
    }
    if rows < history_tokens {
        candle_core::bail!(
            "ple_conv1d_metal: concatenated state has {rows} rows, expected at least {history_tokens} history rows"
        );
    }
    let tokens = rows - history_tokens;
    if kernel_size == 0 || dilation == 0 {
        candle_core::bail!("ple_conv1d_metal: kernel size and dilation must be non-zero");
    }
    if kernel.dims() != [kernel_size, channels] {
        candle_core::bail!(
            "ple_conv1d_metal: kernel shape {:?} does not match [{kernel_size}, {channels}]",
            kernel.dims()
        );
    }
    let coverage = (kernel_size - 1)
        .checked_mul(dilation)
        .ok_or_else(|| candle_core::Error::msg("ple_conv1d_metal: kernel coverage overflow"))?;
    if coverage > history_tokens {
        candle_core::bail!(
            "ple_conv1d_metal: kernel coverage ({coverage} rows) exceeds the {history_tokens}-row history"
        );
    }
    if tokens == 0 {
        return Tensor::zeros((0, channels), dtype, state_and_input.device());
    }

    let Device::Metal(dev) = state_and_input.device() else {
        candle_core::bail!("ple_conv1d_metal: expected Metal device");
    };

    let kernel_name = format!("ple_conv1d_{type_suffix}");
    let pipeline = load_pipeline(dev.device(), &kernel_name)?;

    let output = Tensor::zeros((tokens, channels), dtype, state_and_input.device())?;

    let (s_buf, s_off) = metal_buffer_and_offset(&state_and_input)?;
    let (k_buf, k_off) = metal_buffer_and_offset(&kernel)?;
    let (o_buf, o_off) = metal_buffer_and_offset(&output)?;

    let encoder = dev.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(&s_buf), s_off);
    encoder.set_input_buffer(1, Some(&k_buf), k_off);
    encoder.set_output_buffer(2, Some(&o_buf), o_off);

    let channels_u32 = channels as u32;
    let history_u32 = history_tokens as u32;
    let kernel_size_u32 = kernel_size as u32;
    let dilation_u32 = dilation as u32;
    encoder.set_bytes(3, &channels_u32);
    encoder.set_bytes(4, &history_u32);
    encoder.set_bytes(5, &kernel_size_u32);
    encoder.set_bytes(6, &dilation_u32);

    let thread_groups = MTLSize {
        width: tokens,
        height: 1,
        depth: 1,
    };
    let threads_per_group = MTLSize {
        width: 256,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    Ok(output)
}

/// Stub for non-metal builds.
#[cfg(not(feature = "metal"))]
#[allow(dead_code)]
pub fn ple_conv1d_metal(
    _state_and_input: &candle_core::Tensor,
    _kernel: &candle_core::Tensor,
    _history_tokens: usize,
    _kernel_size: usize,
    _dilation: usize,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("ple_conv1d_metal requires the metal feature")
}

// ============================================================================
// Public API: qsa_topk_indices
// ============================================================================

/// Maximum QSA block-score columns the Metal top-k kernel supports. The kernel
/// ranks one score row in a fixed 4096-entry (16 KB) threadgroup scratch, which
/// is within the Apple GPU 32 KB threadgroup limit; wider histories fall back
/// to the host selector. Zero disables the device path entirely.
#[cfg(feature = "metal")]
pub const QSA_TOPK_MAX_COLUMNS: usize = 4096;

/// Device-side QSA top-k is unavailable without the metal feature.
#[cfg(not(feature = "metal"))]
pub const QSA_TOPK_MAX_COLUMNS: usize = 0;

/// Deterministic QSA indexer block top-k on Metal: ranks each `[rows, columns]`
/// F32 score row by the exact host `QsaBlockSelector::select` order — score
/// descending under `f32::total_cmp` bit semantics, earlier block index on an
/// exact tie — and returns the first `k` block indices per row as U32
/// `[rows, k]`. Non-finite scores are ordered like `f32::total_cmp` instead of
/// being rejected; the host selector keeps strict NaN rejection. Rows wider
/// than [`QSA_TOPK_MAX_COLUMNS`] are rejected so callers fall back to the host
/// selector.
#[cfg(feature = "metal")]
pub fn qsa_topk_indices_metal(scores: &Tensor, k: usize) -> Result<Tensor> {
    let scores = scores.contiguous()?;
    if scores.dtype() != DType::F32 {
        candle_core::bail!(
            "qsa_topk_indices_metal: scores must be F32, got {:?}",
            scores.dtype()
        );
    }
    let dims = scores.dims();
    if dims.len() != 2 {
        candle_core::bail!("qsa_topk_indices_metal: expected [rows, columns] scores, got {dims:?}");
    }
    let (rows, ncols) = (dims[0], dims[1]);
    if ncols == 0 {
        candle_core::bail!("qsa_topk_indices_metal: score rows must have at least one column");
    }
    if k == 0 || k > ncols {
        candle_core::bail!("qsa_topk_indices_metal: k {k} is outside 1..={ncols}");
    }
    if ncols > QSA_TOPK_MAX_COLUMNS {
        candle_core::bail!(
            "qsa_topk_indices_metal: score rows with {ncols} columns exceed the supported maximum of {QSA_TOPK_MAX_COLUMNS}; use the host selector"
        );
    }
    let Device::Metal(dev) = scores.device() else {
        candle_core::bail!("qsa_topk_indices_metal: expected Metal device");
    };

    let pipeline = load_pipeline(dev.device(), "qsa_topk_indices_kernel")?;
    let output = Tensor::zeros((rows, k), DType::U32, scores.device())?;

    let (s_buf, s_off) = metal_buffer_and_offset(&scores)?;
    let (o_buf, o_off) = metal_buffer_and_offset(&output)?;

    let encoder = dev.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_input_buffer(0, Some(&s_buf), s_off);
    encoder.set_output_buffer(1, Some(&o_buf), o_off);

    let ncols_u32 = ncols as u32;
    let k_u32 = k as u32;
    let ncols_pad = ncols.next_power_of_two() as u32;
    encoder.set_bytes(2, &ncols_u32);
    encoder.set_bytes(3, &k_u32);
    encoder.set_bytes(4, &ncols_pad);

    let thread_groups = MTLSize {
        width: rows,
        height: 1,
        depth: 1,
    };
    // One thread per column up to the threadgroup limit; wider rows stride.
    let threads_per_group = MTLSize {
        width: (ncols_pad as usize).min(1024),
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    Ok(output)
}

/// Stub for non-metal builds.
#[cfg(not(feature = "metal"))]
#[allow(dead_code)]
pub fn qsa_topk_indices_metal(
    _scores: &candle_core::Tensor,
    _k: usize,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("qsa_topk_indices_metal requires the metal feature")
}
