/// Metal GDN kernels for Gated Delta Net recurrence, causal conv1d, and fused gating.
///
/// Mirrors the CUDA implementations in `cuda/gdn.rs`.

#[cfg(feature = "metal")]
use candle_core::backend::BackendStorage;
#[cfg(feature = "metal")]
use candle_core::{DType, Device, Result, Storage, Tensor};

#[cfg(feature = "metal")]
use candle_metal_kernels::metal::{
    Buffer, ComputeCommandEncoder, ComputePipeline, Device as MetalRawDevice, Library,
};

#[cfg(feature = "metal")]
use objc2_metal::{MTLCompileOptions, MTLMathMode, MTLSize};

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
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(&q_buf), q_off);
    encoder.set_buffer(1, Some(&k_buf), k_off);
    encoder.set_buffer(2, Some(&v_buf), v_off);
    encoder.set_buffer(3, Some(&g_buf), g_off);
    encoder.set_buffer(4, Some(&beta_buf), beta_off);
    encoder.set_buffer(5, Some(&state_buf), state_off);
    encoder.set_buffer(6, Some(&out_buf), out_off);

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

    let grid_x = (v_dim + bv - 1) / bv;
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
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(&q_buf), q_off);
    encoder.set_buffer(1, Some(&k_buf), k_off);
    encoder.set_buffer(2, Some(&v_buf), v_off);
    encoder.set_buffer(3, Some(&g_buf), g_off);
    encoder.set_buffer(4, Some(&beta_buf), beta_off);
    encoder.set_buffer(5, Some(&state_buf), state_off);
    encoder.set_buffer(6, Some(&out_buf), out_off);

    let seq_len_i32 = seq_len as i32;
    let v_dim_i32 = v_dim as i32;
    encoder.set_bytes(7, &seq_len_i32);
    encoder.set_bytes(8, &v_dim_i32);

    let grid_x = (v_dim + bv - 1) / bv;
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
        DType::BF16 => "bfloat",
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
        let encoder: &ComputeCommandEncoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);

        encoder.set_buffer(0, Some(&x_buf), x_off);
        encoder.set_buffer(1, Some(&w_buf), w_off);
        encoder.set_buffer(2, Some(&cs_buf), cs_off);
        encoder.set_buffer(3, Some(&out_buf), out_off);

        let bs = batch_size as i32;
        let cd = conv_dim as i32;
        let ks = kernel_size as i32;
        encoder.set_bytes(4, &bs);
        encoder.set_bytes(5, &cd);
        encoder.set_bytes(6, &ks);

        let thread_groups = MTLSize {
            width: (conv_dim + 255) / 256,
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
        let (out_buf, out_off) = metal_buffer_and_offset(&output)?;

        {
            let encoder = dev.command_encoder()?;
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&conv_pipeline);

            encoder.set_buffer(0, Some(&x_buf), x_off);
            encoder.set_buffer(1, Some(&w_buf), w_off);
            encoder.set_buffer(2, Some(&out_buf), out_off);

            let bs = batch_size as i32;
            let cd = conv_dim as i32;
            let sl = seq_len as i32;
            let ks = kernel_size as i32;
            encoder.set_bytes(3, &bs);
            encoder.set_bytes(4, &cd);
            encoder.set_bytes(5, &sl);
            encoder.set_bytes(6, &ks);

            let thread_groups = MTLSize {
                width: (conv_dim + 255) / 256,
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
            let encoder: &ComputeCommandEncoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&save_pipeline);

            encoder.set_buffer(0, Some(&x_buf), x_off);
            encoder.set_buffer(1, Some(&cs_buf), cs_off);

            let bs = batch_size as i32;
            let cd = conv_dim as i32;
            let sl = seq_len as i32;
            let ks = kernel_size as i32;
            encoder.set_bytes(2, &bs);
            encoder.set_bytes(3, &cd);
            encoder.set_bytes(4, &sl);
            encoder.set_bytes(5, &ks);

            let thread_groups = MTLSize {
                width: (conv_dim + 255) / 256,
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
        DType::BF16 => "bfloat",
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
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(&b_buf), b_off);
    encoder.set_buffer(1, Some(&a_buf), a_off);
    encoder.set_buffer(2, Some(&alog_buf), alog_off);
    encoder.set_buffer(3, Some(&dtb_buf), dtb_off);
    encoder.set_buffer(4, Some(&beta_buf), beta_off);
    encoder.set_buffer(5, Some(&g_buf), g_off);

    let total = total_elements as i32;
    let heads = num_heads as i32;
    encoder.set_bytes(6, &total);
    encoder.set_bytes(7, &heads);

    let thread_groups = MTLSize {
        width: (total_elements + 255) / 256,
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
