//! Metal MoE forward — dual-path dispatch:
//! 1. Custom tiled kernel (moe_mm_q4k_routed) for prefill (many tokens)
//! 2. Candle's kernel_mul_mv_id for decode (single token)
//!
//! The tiled kernel uses device-buffer routing (no threadgroup limit)
//! and simdgroup_multiply_accumulate for full GPU utilization.

use candle_core::{
    backend::BackendStorage,
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Storage, Tensor,
};
use std::sync::Arc;

use candle_metal_kernels::{
    metal::{Buffer, ComputeCommandEncoder, ComputePipeline, Device as MetalRawDevice, Library},
    set_params, Kernels, Source,
};
use objc2_metal::{MTLCompileOptions, MTLMathMode, MTLResourceUsage, MTLSize};
use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

// ── Custom kernel loading ──

static MOE_LIBRARY: OnceLock<Library> = OnceLock::new();
static MOE_PIPELINES: OnceLock<RwLock<HashMap<String, ComputePipeline>>> = OnceLock::new();

const MOE_METAL_SOURCE: &str = include_str!("../metal_kernels/indexed_moe.metal");

fn load_moe_library(device: &MetalRawDevice) -> Result<Library> {
    if let Some(lib) = MOE_LIBRARY.get() {
        return Ok(lib.clone());
    }
    let opts = MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Fast);
    let lib = device
        .new_library_with_source(MOE_METAL_SOURCE, Some(&opts))
        .map_err(|e| candle_core::Error::Msg(format!("MoE Metal compile: {e}")))?;
    Ok(MOE_LIBRARY.get_or_init(|| lib).clone())
}

fn load_moe_pipeline(device: &MetalRawDevice, name: &str) -> Result<ComputePipeline> {
    let lock = MOE_PIPELINES.get_or_init(|| RwLock::new(HashMap::new()));
    {
        let cache = lock.read().map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        if let Some(p) = cache.get(name) {
            return Ok(p.clone());
        }
    }
    let lib = load_moe_library(device)?;
    let func = lib.get_function(name, None).map_err(|e| {
        candle_core::Error::Msg(format!("MoE function '{name}': {e}"))
    })?;
    let pipeline = device.new_compute_pipeline_state_with_function(&func).map_err(|e| {
        candle_core::Error::Msg(format!("MoE pipeline '{name}': {e}"))
    })?;
    let mut cache = lock.write().map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
    cache.insert(name.to_string(), pipeline.clone());
    Ok(pipeline)
}

// ── Helpers ──

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

fn qtensor_metal_buffer(qtensor: &QTensor) -> Result<(Buffer, GgmlDType)> {
    let ms = qtensor.metal_storage()?;
    Ok((ms.buffer().clone(), qtensor.dtype()))
}

fn divide(m: usize, n: usize) -> usize { (m + n - 1) / n }

// ── Public API ──

pub fn metal_indexed_moe_forward(
    qmatmul: &QMatMul,
    x: &Tensor,
    ids: &Tensor,
    _dequant_cache: &std::sync::Mutex<Option<Tensor>>,
) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => dispatch_quantized_moe(qtensor, x, ids),
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => dispatch_unquantized_moe(t, x, ids),
    }
}

/// Fused gate+up+SiLU dispatch: silu(gate_proj @ x) * (up_proj @ x) in one Metal kernel.
/// Returns the activated result ready for down_proj.
pub fn metal_fused_gate_up_swiglu(
    gate_qt: &Arc<QTensor>,
    up_qt: &Arc<QTensor>,
    x: &Tensor,
    ids: &Tensor,
) -> Result<Tensor> {
    let Device::Metal(dev) = x.device() else {
        candle_core::bail!("expected Metal device for fused MoE");
    };

    let (gate_buf, gate_dtype) = qtensor_metal_buffer(gate_qt)?;
    let (up_buf, up_dtype) = qtensor_metal_buffer(up_qt)?;

    if gate_dtype != GgmlDType::Q4K || up_dtype != GgmlDType::Q4K {
        candle_core::bail!("fused_gate_up_swiglu only supports Q4K, got {gate_dtype:?}/{up_dtype:?}");
    }

    let (n_experts, n_out, n_in) = match gate_qt.shape().dims() {
        &[e, o, i] => (e, o, i),
        d => candle_core::bail!("Expected 3D gate weights, got {d:?}"),
    };

    let (batch, topk, input_dim1, x_flat) = match x.dims() {
        &[b, s, 1, 1, h] => { let (_, _, k) = ids.dims3()?; (b*s, k, 1usize, x.reshape((b*s, h))?) }
        &[b, s, k, h] if k > 1 => (b*s, k, k, x.reshape((b*s*k, h))?),
        &[n, 1, h] => { let (_, k) = ids.dims2()?; (n, k, 1, x.reshape((n, h))?) }
        d => candle_core::bail!("unsupported input shape for fused MoE: {d:?}"),
    };

    let x_flat = x_flat.contiguous()?.to_dtype(DType::F32)?;
    let flat_ids = ids.reshape((batch, topk))?.to_dtype(DType::U32)?.contiguous()?;
    let output = Tensor::zeros((batch * topk, n_out), DType::F32, x_flat.device())?;

    let (x_buf, x_off) = metal_buffer_and_offset(&x_flat)?;
    let (id_buf, id_off) = metal_buffer_and_offset(&flat_ids)?;
    let (out_buf, _out_off) = metal_buffer_and_offset(&output)?;

    let block_size = gate_dtype.block_size();
    let type_size = gate_dtype.type_size();
    let blocks_per_row = n_in / block_size;
    let nb01 = (blocks_per_row * type_size) as u64;
    let nb02 = (n_out as u64) * nb01;

    candle_metal_kernels::call_fused_moe_swiglu(
        dev.device(),
        &dev.command_encoder()?,
        dev.kernels(),
        candle_metal_kernels::GgmlDType::Q4K,
        &gate_buf,
        &up_buf,
        &x_buf,
        x_off,
        &id_buf,
        id_off,
        &out_buf,
        n_out,
        n_in,
        batch,
        topk,
        input_dim1,
        nb01,
        nb02,
    ).map_err(|e| candle_core::Error::Msg(format!("fused_moe_swiglu dispatch: {e}")))?;

    reshape_output(&output, x.dims(), ids)
}

// ── Quantized MoE dispatch ──

fn dispatch_quantized_moe(qtensor: &Arc<QTensor>, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let Device::Metal(dev) = x.device() else {
        candle_core::bail!("expected Metal device");
    };

    let (w_buf, ggml_dtype) = qtensor_metal_buffer(qtensor)?;
    let (n_experts, n_out, n_in) = match qtensor.shape().dims() {
        &[e, o, i] => (e, o, i),
        d => candle_core::bail!("Expected 3D weights, got {d:?}"),
    };

    let (batch, topk, input_dim1, x_flat) = match x.dims() {
        &[b, s, 1, 1, h] => { let (_, _, k) = ids.dims3()?; (b*s, k, 1usize, x.reshape((b*s, h))?) }
        &[b, s, k, h] if k > 1 => (b*s, k, k, x.reshape((b*s*k, h))?),
        &[n, 1, h] => { let (_, k) = ids.dims2()?; (n, k, 1, x.reshape((n, h))?) }
        d => candle_core::bail!("unsupported input {d:?}"),
    };

    let total_pairs = batch * topk;

    // For single-token decode, use mv_id (fast for matvec)
    // For prefill (many tokens), use custom tiled kernel
    // Tiled kernel for prefill (simdgroup matmul), mv_id for decode (matvec)
    // Tiled kernel wins for long sequences (routing overhead amortized).
    // For short sequences (< 32 tokens), mv_id is faster.
    let use_tiled = batch > 32 && ggml_dtype == GgmlDType::Q4K;

    if use_tiled {
        dispatch_tiled_moe(dev, &w_buf, ggml_dtype, n_experts, n_out, n_in, &x_flat, ids, batch, topk, input_dim1, x.dims())
    } else {
        dispatch_mv_id_moe(dev, &w_buf, ggml_dtype, n_experts, n_out, n_in, &x_flat, ids, batch, topk, input_dim1, x.dims())
    }
}

/// Custom tiled kernel with device-buffer routing
fn dispatch_tiled_moe(
    dev: &candle_core::MetalDevice,
    w_buf: &Buffer,
    ggml_dtype: GgmlDType,
    n_experts: usize, n_out: usize, n_in: usize,
    x_flat: &Tensor, ids: &Tensor,
    batch: usize, topk: usize, input_dim1: usize,
    orig_dims: &[usize],
) -> Result<Tensor> {
    let total_pairs = batch * topk;
    let flat_ids = ids.reshape((total_pairs,))?.to_dtype(DType::U32)?;
    let idx_vec: Vec<u32> = flat_ids.to_vec1()?;

    // Build routing table
    let mut expert_pairs: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n_experts]; // (tok_idx, pair_idx)
    for (pair_idx, &eid) in idx_vec.iter().enumerate() {
        // tok_idx is always the token index (not the flat pair index)
        // The kernel uses nb12*id[1] + nb11*(id[0]%ne11) for input addressing
        let tok_idx = pair_idx / topk;
        expert_pairs[eid as usize].push((tok_idx as u32, pair_idx as u32));
    }

    // Build flat routing arrays
    // rowids = ushort2(expert_slot, token_idx) matching kernel's expected format
    let mut route_counts = vec![0u32; n_experts];
    let mut route_offsets = vec![0u32; n_experts];
    let mut rowids_packed: Vec<u32> = Vec::with_capacity(total_pairs); // ushort2 packed as u32
    let mut max_tokens_per_expert = 0usize;

    let mut offset = 0u32;
    for (eid, pairs) in expert_pairs.iter().enumerate() {
        route_counts[eid] = pairs.len() as u32;
        route_offsets[eid] = offset;
        max_tokens_per_expert = max_tokens_per_expert.max(pairs.len());
        for &(tok_idx, pair_idx) in pairs {
            // ushort2 packed as u32: low 16 bits = slot, high 16 bits = token_idx
            let slot = (pair_idx as usize % topk) as u16;
            let packed = (slot as u32) | ((tok_idx as u32 & 0xFFFF) << 16);
            rowids_packed.push(packed);
        }
        offset += pairs.len() as u32;
    }

    // Create Metal tensors for routing
    let device = x_flat.device();
    let counts_t = Tensor::new(route_counts, device)?;
    let offsets_t = Tensor::new(route_offsets, device)?;
    // rowids packed as u32 — kernel reads as ushort2 (same memory layout)
    let rowids_t = Tensor::new(rowids_packed, device)?;

    let x_flat = x_flat.contiguous()?.to_dtype(DType::F32)?;
    let output = Tensor::zeros((total_pairs, n_out), DType::F32, device)?;

    let (x_buf, x_off) = metal_buffer_and_offset(&x_flat)?;
    let (out_buf, out_off) = metal_buffer_and_offset(&output)?;
    let (cnt_buf, cnt_off) = metal_buffer_and_offset(&counts_t)?;
    let (rid_buf, rid_off) = metal_buffer_and_offset(&rowids_t)?;
    let (off_buf, off_off) = metal_buffer_and_offset(&offsets_t)?;

    // Kernel scalar params
    let block_size = ggml_dtype.block_size();
    let type_size = ggml_dtype.type_size();
    let blocks_per_row = n_in / block_size;
    let nb01 = (blocks_per_row * type_size) as u64;

    let ne00 = n_in as i64;
    let ne0 = n_out as i64;
    let ne11 = input_dim1 as i64;  // 1 for broadcast, topk for per-slot
    let nb10 = 4u64;               // f32 = 4 bytes
    let nb11 = (n_in * 4) as u64;
    let nb12 = (input_dim1 as u64) * nb11;
    let ne0ne1 = (n_out * topk) as i64;  // output stride: ne0 * topk

    let pipeline = load_moe_pipeline(dev.device(), "moe_mm_q4k_routed")?;

    let encoder = dev.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(w_buf), 0);
    encoder.set_buffer(1, Some(&x_buf), x_off);
    encoder.set_buffer(2, Some(&out_buf), out_off);
    encoder.set_buffer(3, Some(&cnt_buf), cnt_off);
    encoder.set_buffer(4, Some(&rid_buf), rid_off);  // ushort2 rowids
    encoder.set_buffer(5, Some(&off_buf), off_off);
    encoder.set_bytes(6, &ne00);
    encoder.set_bytes(7, &ne0);
    encoder.set_bytes(8, &nb01);
    encoder.set_bytes(9, &ne11);
    encoder.set_bytes(10, &nb10);
    encoder.set_bytes(11, &nb11);
    encoder.set_bytes(12, &nb12);
    encoder.set_bytes(13, &ne0ne1);

    let grid = MTLSize {
        width: divide(max_tokens_per_expert, 32),
        height: divide(n_out, 64),
        depth: n_experts,
    };
    let threads = MTLSize { width: 128, height: 1, depth: 1 };

    encoder.use_resource(w_buf, MTLResourceUsage::Read);
    encoder.use_resource(&x_buf, MTLResourceUsage::Read);
    encoder.use_resource(&out_buf, MTLResourceUsage::Write);
    encoder.use_resource(&cnt_buf, MTLResourceUsage::Read);
    encoder.use_resource(&rid_buf, MTLResourceUsage::Read);
    encoder.use_resource(&off_buf, MTLResourceUsage::Read);

    encoder.set_threadgroup_memory_length(0, 8192);
    encoder.dispatch_thread_groups(grid, threads);

    reshape_output(&output, orig_dims, ids)
}

/// Candle's kernel_mul_mv_id — good for single-token decode
fn dispatch_mv_id_moe(
    dev: &candle_core::MetalDevice,
    w_buf: &Buffer,
    ggml_dtype: GgmlDType,
    _n_experts: usize, n_out: usize, n_in: usize,
    x_flat: &Tensor, ids: &Tensor,
    batch: usize, topk: usize, input_dim1: usize,
    orig_dims: &[usize],
) -> Result<Tensor> {
    let flat_ids = ids.reshape((batch, topk))?.to_dtype(DType::U32)?;
    let x_flat = x_flat.contiguous()?.to_dtype(DType::F32)?;
    let flat_ids = flat_ids.contiguous()?;
    let output = Tensor::zeros((batch * topk, n_out), DType::F32, x_flat.device())?;

    let (x_buf, x_off) = metal_buffer_and_offset(&x_flat)?;
    let (id_buf, id_off) = metal_buffer_and_offset(&flat_ids)?;
    let (out_buf, out_off) = metal_buffer_and_offset(&output)?;

    let block_size = ggml_dtype.block_size();
    let type_size = ggml_dtype.type_size();
    let blocks_per_row = n_in / block_size;
    let nb01 = (blocks_per_row * type_size) as u64;
    let nb02 = (n_out as u64) * nb01;

    let nei0 = topk as i64;
    let nei1 = batch as i64;
    let nbi1 = (topk * 4) as u64;
    let ne00 = n_in as i64;
    let ne01 = n_out as i64;
    let ne02 = 1i64;
    let nb00 = type_size as u64;
    let ne10 = n_in as i64;
    let ne11 = input_dim1 as i64;
    let ne12 = 1i64;
    let ne13 = 1i64;
    let nb10 = 4u64;
    let nb11 = (n_in * 4) as u64;
    let nb12 = (input_dim1 as u64) * nb11;
    let ne0 = n_out as i64;
    let ne1 = topk as i64;
    let nb1 = (n_out * 4) as u64;

    let (nth0, nth1, align) = match ggml_dtype {
        GgmlDType::Q4K => (32usize, 2usize, 4usize), // 2 simdgroups × 32 threads = 64
        GgmlDType::Q2K => (2, 32, 4),
        GgmlDType::Q3K | GgmlDType::Q5K => (2, 32, 4),
        GgmlDType::Q6K => (2, 32, 2),
        GgmlDType::Q8_0 => (8, 8, 8),
        _ => (32, 1, 8),
    };
    let kernel_name = match ggml_dtype {
        GgmlDType::Q4K => "kernel_mul_mv_id_q4_K_f32",
        GgmlDType::Q2K => "kernel_mul_mv_id_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_id_q3_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_id_q6_K_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_id_q8_0_f32",
        dt => candle_core::bail!("Unsupported dtype: {dt:?}"),
    };

    let tg = MTLSize { width: divide(n_out, align), height: 1, depth: batch * topk };
    let tpg = MTLSize { width: nth0, height: nth1, depth: 1 };

    let pipeline = dev.kernels()
        .load_pipeline(dev.device(), Source::Quantized, kernel_name)
        .map_err(|e| candle_core::Error::Msg(format!("mv_id load: {e}")))?;

    let encoder = dev.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (
        (w_buf, 0usize), (&x_buf, x_off), (&out_buf, out_off), (&id_buf, id_off),
        nei0, nei1, nbi1,
        ne00, ne01, ne02, nb00, nb01, nb02,
        ne10, ne11, ne12, ne13, nb10, nb11, nb12,
        ne0, ne1, nb1
    ));
    encoder.use_resource(w_buf, MTLResourceUsage::Read);
    encoder.use_resource(&x_buf, MTLResourceUsage::Read);
    encoder.use_resource(&id_buf, MTLResourceUsage::Read);
    encoder.use_resource(&out_buf, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(tg, tpg);

    reshape_output(&output, orig_dims, ids)
}

fn reshape_output(output: &Tensor, orig_dims: &[usize], ids: &Tensor) -> Result<Tensor> {
    match orig_dims {
        &[b, s, 1, 1, _] => { let (_, _, k) = ids.dims3()?; output.reshape((b, s, k, output.dim(1)?)) }
        &[b, s, k, _] if k > 1 => output.reshape((b, s, k, output.dim(1)?)),
        &[n, 1, _] => { let (_, k) = ids.dims2()?; output.reshape((n, k, output.dim(1)?)) }
        _ => Ok(output.clone()),
    }
}

fn dispatch_unquantized_moe(weights: &Tensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let (ne, of, _) = weights.dims3()?;
    let (batch, topk, x_flat) = match x.dims() {
        &[b, s, 1, 1, h] => { let (_, _, k) = ids.dims3()?; (b*s, k, x.reshape((b*s, h))?) }
        &[b, s, k, h] if k > 1 => (b*s, k, x.reshape((b*s*k, h))?),
        &[n, 1, h] => { let (_, k) = ids.dims2()?; (n, k, x.reshape((n, h))?) }
        d => candle_core::bail!("unsupported {d:?}"),
    };
    let fi = ids.reshape((batch*topk,))?;
    let iv: Vec<u32> = fi.to_vec1()?;
    let mut et: Vec<Vec<usize>> = vec![Vec::new(); ne];
    for (pi, &eid) in iv.iter().enumerate() { et[eid as usize].push(pi); }
    let dev = x.device();
    let dt = x.dtype();
    let mut out = Tensor::zeros((batch*topk, of), dt, dev)?;
    for (eid, pairs) in et.iter().enumerate() {
        if pairs.is_empty() { continue; }
        let ei = Tensor::new(&[eid as u32], dev)?;
        let ew = weights.index_select(&ei, 0)?.squeeze(0)?;
        let ti: Vec<u32> = pairs.iter().map(|&p| (p/topk) as u32).collect();
        let tok = x_flat.index_select(&Tensor::new(ti, dev)?, 0)?;
        let r = tok.matmul(&ew.t()?)?;
        let pi = Tensor::new(pairs.iter().map(|&i| i as u32).collect::<Vec<_>>(), dev)?;
        out = out.index_add(&pi, &r, 0)?;
    }
    reshape_output(&out, x.dims(), ids)
}
