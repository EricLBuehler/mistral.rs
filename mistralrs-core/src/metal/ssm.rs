/// Metal selective scan kernel for Mamba SSM.
///
/// Replaces the per-timestep Rust loop in `MambaLayer::forward_full` with a
/// single GPU kernel dispatch on Metal devices.

#[cfg(feature = "metal")]
use candle_core::{DType, Device, Result, Storage, Tensor};

#[cfg(feature = "metal")]
use candle_metal_kernels::metal::{
    Buffer, ComputeCommandEncoder, ComputePipeline, Device as MetalRawDevice, Library,
};

#[cfg(feature = "metal")]
use objc2_metal::{MTLCompileOptions, MTLDevice, MTLMathMode, MTLSize};

#[cfg(feature = "metal")]
use std::collections::HashMap;

#[cfg(feature = "metal")]
use std::sync::{OnceLock, RwLock};

#[cfg(feature = "metal")]
static SSM_LIBRARY: OnceLock<Library> = OnceLock::new();

#[cfg(feature = "metal")]
type Pipelines = HashMap<String, ComputePipeline>;

#[cfg(feature = "metal")]
static SSM_PIPELINES: OnceLock<RwLock<Pipelines>> = OnceLock::new();

#[cfg(feature = "metal")]
const SSM_METAL_SOURCE: &str = include_str!("kernels/ssm.metal");

#[cfg(feature = "metal")]
fn load_ssm_library(device: &MetalRawDevice) -> Result<Library> {
    if let Some(lib) = SSM_LIBRARY.get() {
        return Ok(lib.clone());
    }
    let compile_options = {
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Fast);
        opts
    };
    let lib = device
        .new_library_with_source(SSM_METAL_SOURCE, Some(&compile_options))
        .map_err(|e| {
            candle_core::Error::Msg(format!("Failed to compile SSM Metal kernels: {e}"))
        })?;
    Ok(SSM_LIBRARY.get_or_init(|| lib).clone())
}

#[cfg(feature = "metal")]
fn load_pipeline(device: &MetalRawDevice, name: &str) -> Result<ComputePipeline> {
    let pipelines_lock = SSM_PIPELINES.get_or_init(|| RwLock::new(Pipelines::new()));

    {
        let pipelines = pipelines_lock.read().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to lock SSM pipeline cache: {e}"))
        })?;
        if let Some(pipeline) = pipelines.get(name) {
            return Ok(pipeline.clone());
        }
    }

    let lib = load_ssm_library(device)?;
    let func = lib.get_function(name, None).map_err(|e| {
        candle_core::Error::Msg(format!("Failed to load SSM Metal function '{name}': {e}"))
    })?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create SSM pipeline for '{name}': {e}"))
        })?;

    let mut pipelines = pipelines_lock.write().map_err(|e| {
        candle_core::Error::Msg(format!("Failed to lock SSM pipeline cache for write: {e}"))
    })?;
    pipelines.insert(name.to_string(), pipeline.clone());
    Ok(pipeline)
}

#[cfg(feature = "metal")]
fn metal_buffer_and_offset(tensor: &Tensor) -> Result<(&Buffer, usize)> {
    let (storage, layout) = tensor.storage_and_layout();
    match &*storage {
        Storage::Metal(m) => {
            let offset = layout.start_offset() * m.dtype().size_in_bytes();
            Ok((m.buffer(), offset))
        }
        _ => candle_core::bail!("Expected Metal tensor"),
    }
}

/// Selective scan on Metal.
///
/// Inputs:
/// - `x`:       (batch, seq_len, n_heads, head_dim) - input after conv1d+SiLU, f32
/// - `dt`:      (batch, seq_len, n_heads) - timestep (pre-bias, pre-softplus), f32
/// - `a`:       (n_heads,) - negative exp of A_log, f32
/// - `b`:       (batch, seq_len, n_heads, d_state) - input projection, f32
/// - `c`:       (batch, seq_len, n_heads, d_state) - output projection, f32
/// - `d`:       (n_heads,) - skip connection weight, f32
/// - `dt_bias`: (n_heads,) - dt bias, f32
/// - `state`:   (batch, n_heads, head_dim, d_state) - SSM state (mutated in-place), f32
///
/// Returns:
/// - `y`:       (batch, seq_len, n_heads, head_dim) - output, f32
#[cfg(feature = "metal")]
pub fn selective_scan_metal(
    x: &Tensor,
    dt: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: &Tensor,
    dt_bias: &Tensor,
    state: &mut Tensor,
    dt_min: f32,
    dt_max: f32,
) -> Result<Tensor> {
    let x = x.contiguous()?;
    let dt = dt.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let c = c.contiguous()?;
    let d = d.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;

    let (batch_size, seq_len, n_heads, head_dim) = x.dims4()?;
    let d_state = b.dims4()?.3;

    // Reshape to flat layout expected by kernel
    let x_flat = x.reshape((batch_size, seq_len, n_heads * head_dim))?;
    let b_flat = b.reshape((batch_size, seq_len, n_heads * d_state))?;
    let c_flat = c.reshape((batch_size, seq_len, n_heads * d_state))?;

    let Device::Metal(dev) = x_flat.device() else {
        candle_core::bail!("selective_scan_metal: expected Metal device");
    };

    // Select kernel based on c_factor = ceil(d_state / 32)
    let c_factor = (d_state + 31) / 32;
    let kernel_name = match c_factor {
        1 => "ssm_scan_c1",
        2 => "ssm_scan_c2",
        3 | 4 => "ssm_scan_c4",
        _ => "ssm_scan_c8",
    };

    let pipeline = load_pipeline(dev.device(), kernel_name)?;

    // Allocate output
    let y = Tensor::zeros(
        (batch_size, seq_len, n_heads * head_dim),
        DType::F32,
        x_flat.device(),
    )?;

    let (x_buf, x_off) = metal_buffer_and_offset(&x_flat)?;
    let (dt_buf, dt_off) = metal_buffer_and_offset(&dt)?;
    let (a_buf, a_off) = metal_buffer_and_offset(&a)?;
    let (b_buf, b_off) = metal_buffer_and_offset(&b_flat)?;
    let (c_buf, c_off) = metal_buffer_and_offset(&c_flat)?;
    let (d_buf, d_off) = metal_buffer_and_offset(&d)?;
    let (dtb_buf, dtb_off) = metal_buffer_and_offset(&dt_bias)?;
    let (st_buf, st_off) = metal_buffer_and_offset(state)?;
    let (y_buf, y_off) = metal_buffer_and_offset(&y)?;

    let encoder = dev.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(x_buf), x_off);
    encoder.set_buffer(1, Some(dt_buf), dt_off);
    encoder.set_buffer(2, Some(a_buf), a_off);
    encoder.set_buffer(3, Some(b_buf), b_off);
    encoder.set_buffer(4, Some(c_buf), c_off);
    encoder.set_buffer(5, Some(d_buf), d_off);
    encoder.set_buffer(6, Some(dtb_buf), dtb_off);
    encoder.set_buffer(7, Some(st_buf), st_off);
    encoder.set_buffer(8, Some(y_buf), y_off);

    let n_heads_i32 = n_heads as i32;
    let head_dim_i32 = head_dim as i32;
    let d_state_i32 = d_state as i32;
    let seq_len_i32 = seq_len as i32;
    encoder.set_bytes(9, &n_heads_i32);
    encoder.set_bytes(10, &head_dim_i32);
    encoder.set_bytes(11, &d_state_i32);
    encoder.set_bytes(12, &seq_len_i32);
    encoder.set_bytes(13, &dt_min);
    encoder.set_bytes(14, &dt_max);

    let n_warps = n_heads * head_dim;
    let thread_groups = MTLSize {
        width: n_warps,
        height: batch_size,
        depth: 1,
    };
    let threads_per_group = MTLSize {
        width: 32, // SIMD_SIZE
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    let y = y.reshape((batch_size, seq_len, n_heads, head_dim))?;
    Ok(y)
}

#[cfg(not(feature = "metal"))]
#[allow(dead_code)]
pub fn selective_scan_metal(
    _x: &candle_core::Tensor,
    _dt: &candle_core::Tensor,
    _a: &candle_core::Tensor,
    _b: &candle_core::Tensor,
    _c: &candle_core::Tensor,
    _d: &candle_core::Tensor,
    _dt_bias: &candle_core::Tensor,
    _state: &mut candle_core::Tensor,
    _dt_min: f32,
    _dt_max: f32,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("selective_scan_metal requires the metal feature")
}
