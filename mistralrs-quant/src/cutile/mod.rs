//! cuTile MoE backend: faithful port of vLLM's fused MoE (unquantized BF16).
//! `moe_align` + `gelu_tanh_and_mul` are direct CUDA ports (kernels/moe/*.cu);
//! the grouped GEMM is a cuTile kernel (moe.rs). See the kernel module for the
//! op-by-op translation notes.

mod ffi;
pub mod interop;
pub mod moe;

use candle_core::cuda::cudarc::driver::CudaSlice;
use candle_core::{CudaDevice, DType, Result, Storage, Tensor};
use cuda_async::device_buffer::DevicePointer;
use cuda_async::device_operation::DeviceOp;
use cuda_core::sys::CUdeviceptr;
use cutile::tile_kernel::TileKernel;
use half::bf16;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};
use candle_core::{Device, DeviceLocation};

/// Launch tile config from vLLM `get_default_config` (bf16 branch, E used only
/// for GROUP_SIZE_M). Computed once from the original token count and reused for
/// both GEMMs; `bm` is the `moe_align` block size.
#[derive(Clone, Copy)]
pub struct MoeTileConfig {
    pub bm: i32,
    pub bn: i32,
    pub bk: i32,
    pub group_m: i32,
}

// Token counts to warm per registered shape: decode batch sizes plus common prefill lengths.
// Each distinct m yields a distinct compiled kernel key (shape/scalar divisibility hints), so
// this is the set of M for which the runtime forward will hit a warm cache.
const MOE_WARMUP_TOKENS: &[usize] = &[
    1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
];
static MOE_WARMED_LOCATIONS: OnceLock<Mutex<HashSet<DeviceLocation>>> = OnceLock::new();

#[derive(Clone)]
struct MoeWarmupEntry {
    gate_up_w: Tensor,
    down_w: Tensor,
    num_experts: usize,
    top_k: usize,
    hidden: usize,
    inter: usize,
}

static MOE_SHAPES: OnceLock<Mutex<Vec<MoeWarmupEntry>>> = OnceLock::new();

/// Register a model's MoE weights so warmup compiles the exact kernel keys hit at inference.
/// Deduped by (hidden, inter, num_experts, top_k); weights are Arc clones, not copies.
pub fn register_moe_shape(gate_up_w: Tensor, down_w: Tensor, num_experts: usize, top_k: usize) {
    let (Ok(hidden), Ok(inter)) = (gate_up_w.dim(1), down_w.dim(1)) else {
        return;
    };
    let mut shapes = MOE_SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if shapes.iter().any(|e| {
        e.hidden == hidden && e.inter == inter && e.num_experts == num_experts && e.top_k == top_k
    }) {
        return;
    }
    shapes.push(MoeWarmupEntry {
        gate_up_w,
        down_w,
        num_experts,
        top_k,
        hidden,
        inter,
    });
}

pub const fn get_default_config(m: usize, num_experts: usize) -> MoeTileConfig {
    let bm = if m <= 32 {
        16
    } else if m <= 96 {
        32
    } else if m <= 512 {
        64
    } else {
        128
    };
    let bn = if m <= 64 { 64 } else { 128 };
    let bk = if m <= 64 { 128 } else { 64 };
    let num_experts = if num_experts == 0 { 1 } else { num_experts };
    let group_m = if m / num_experts > 128 { 16 } else { 1 };
    MoeTileConfig {
        bm,
        bn,
        bk,
        group_m,
    }
}

pub fn warmup_moe_kernels(device: &Device) -> Result<()> {
    let Device::Cuda(dev) = device else {
        return Ok(());
    };
    let location = device.location();
    {
        let mut warmed = MOE_WARMED_LOCATIONS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap();
        if !warmed.insert(location) {
            return Ok(());
        }
    }

    if let Err(err) = warmup_moe_kernels_uncached(dev) {
        let mut warmed = MOE_WARMED_LOCATIONS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap();
        warmed.remove(&location);
        return Err(err);
    }

    device.synchronize()
}

fn warmup_moe_kernels_uncached(dev: &CudaDevice) -> Result<()> {
    let entries: Vec<MoeWarmupEntry> = MOE_SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .clone();
    for entry in &entries {
        for &m in MOE_WARMUP_TOKENS {
            if let Err(err) = warmup_shape(dev, entry, m) {
                tracing::warn!(
                    "cuTile MoE warmup failed (hidden={} inter={} m={m}): {err}",
                    entry.hidden,
                    entry.inter
                );
            }
        }
    }
    Ok(())
}

// Replays both fused-MoE GEMMs (gate_up then down) with the real weights and dummy activations,
// so the compiled kernel keys match the runtime forward at this token count.
fn warmup_shape(dev: &CudaDevice, entry: &MoeWarmupEntry, m: usize) -> Result<()> {
    let device = Device::Cuda(dev.clone());
    let topk = entry.top_k;
    let num_experts = entry.num_experts;
    let num_valid = m * topk;
    let cfg = get_default_config(m, num_experts);

    let topk_ids_host = vec![0u32; num_valid];
    let mut topk_ids = unsafe { dev.alloc::<u32>(num_valid)? };
    dev.memcpy_htod(&topk_ids_host, &mut topk_ids)?;
    let (sids, eids, ntpp, em) = moe_align(&topk_ids, m, num_experts, topk, cfg.bm, dev)?;

    let a1 = Tensor::zeros((m, entry.hidden), DType::BF16, &device)?;
    let _ = cutile_grouped_gemm(
        &a1,
        &entry.gate_up_w,
        &sids,
        &eids,
        &ntpp,
        None,
        em,
        num_valid,
        topk,
        false,
        cfg,
        dev,
    )?;

    let a2 = Tensor::zeros((num_valid, entry.inter), DType::BF16, &device)?;
    let tw_host = vec![0f32; num_valid];
    let mut tw = unsafe { dev.alloc::<f32>(num_valid)? };
    dev.memcpy_htod(&tw_host, &mut tw)?;
    let _ = cutile_grouped_gemm(
        &a2,
        &entry.down_w,
        &sids,
        &eids,
        &ntpp,
        Some(&tw),
        em,
        num_valid,
        1,
        true,
        cfg,
        dev,
    )?;
    Ok(())
}

/// vLLM `max_num_tokens_padded` (EM): padded length of `sorted_token_ids`.
pub fn moe_align_em(
    num_tokens: usize,
    topk: usize,
    num_experts: usize,
    block_size: usize,
) -> usize {
    let numel = num_tokens * topk;
    let em = numel + num_experts * (block_size - 1);
    if numel < num_experts {
        (numel * block_size).min(em)
    } else {
        em
    }
}

/// vLLM `moe_align_block_size`. Returns (sorted_token_ids[EM], expert_ids[nblocks],
/// num_tokens_post_pad[1], EM). topk_ids must be a contiguous u32 slice of
/// [num_tokens*topk] expert ids (reinterpreted as i32; ids are < 2^31).
pub fn moe_align(
    topk_ids_u32: &CudaSlice<u32>,
    num_tokens: usize,
    num_experts: usize,
    topk: usize,
    block_size: i32,
    dev: &CudaDevice,
) -> Result<(CudaSlice<i32>, CudaSlice<i32>, CudaSlice<i32>, usize)> {
    let numel = num_tokens * topk;
    let bs = block_size as usize;
    let em = moe_align_em(num_tokens, topk, num_experts, bs);
    let nblocks = em.div_ceil(bs);

    let mut sids = unsafe { dev.alloc::<i32>(em)? };
    let mut eids = unsafe { dev.alloc::<i32>(nblocks)? };
    let mut ntpp = unsafe { dev.alloc::<i32>(1)? };
    let mut cumsum = unsafe { dev.alloc::<i32>(num_experts + 1)? };

    let stream = dev.cuda_stream();
    let cu_stream = stream.cu_stream();
    let (tk_ptr, _tk_guard) = slice_ptr_on_stream(topk_ids_u32, 0, &stream);
    let (s_ptr, s_guard) = slice_ptr_mut_on_stream(&mut sids, 0, &stream);
    let (e_ptr, e_guard) = slice_ptr_mut_on_stream(&mut eids, 0, &stream);
    let (n_ptr, n_guard) = slice_ptr_mut_on_stream(&mut ntpp, 0, &stream);
    let (c_ptr, c_guard) = slice_ptr_mut_on_stream(&mut cumsum, 0, &stream);
    unsafe {
        ffi::launch_moe_align(
            tk_ptr as *const i32,
            s_ptr as *mut i32,
            e_ptr as *mut i32,
            n_ptr as *mut i32,
            c_ptr as *mut i32,
            num_experts as i32,
            block_size,
            numel as i32,
            em as i32,
            cu_stream,
        );
    }
    drop((s_guard, e_guard, n_guard, c_guard));
    Ok((sids, eids, ntpp, em))
}

/// vLLM `gelu_tanh_and_mul`: input [num_tokens, 2*d] -> [num_tokens, d] bf16.
pub fn gelu_tanh_and_mul(input: &Tensor, d: usize, dev: &CudaDevice) -> Result<Tensor> {
    let (num_tokens, two_d) = input.dims2()?;
    assert_eq!(two_d, 2 * d, "gelu_tanh_and_mul expects last dim == 2*d");
    assert_eq!(input.dtype(), DType::BF16, "cutile gelu path is bf16-only");

    let mut out = unsafe { dev.alloc::<bf16>(num_tokens * d)? };
    let stream = dev.cuda_stream();
    let cu_stream = stream.cu_stream();

    let (in_storage, in_layout) = input.storage_and_layout();
    let in_slice = match &*in_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
        _ => candle_core::bail!("input must be cuda"),
    };
    let (in_ptr, _in_guard) = slice_ptr_on_stream(in_slice, in_layout.start_offset(), &stream);
    let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
    unsafe {
        ffi::launch_gelu_tanh_and_mul_bf16(
            out_ptr as *mut core::ffi::c_void,
            in_ptr as *const core::ffi::c_void,
            num_tokens as i32,
            d as i32,
            cu_stream,
        );
    }
    drop(out_guard);

    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((Storage::Cuda(storage), (num_tokens, d))))
}

pub fn moe_sum_bf16(
    input: &Tensor,
    num_tokens: usize,
    topk: usize,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let (total_assignments, hidden) = input.dims2()?;
    assert_eq!(
        total_assignments,
        num_tokens * topk,
        "moe_sum_bf16 input rows mismatch"
    );
    assert_eq!(input.dtype(), DType::BF16, "moe_sum_bf16 is bf16-only");

    let mut out = unsafe { dev.alloc::<bf16>(num_tokens * hidden)? };
    let stream = dev.cuda_stream();
    let cu_stream = stream.cu_stream();

    let (in_storage, in_layout) = input.storage_and_layout();
    let in_slice = match &*in_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
        _ => candle_core::bail!("input must be cuda"),
    };
    let (in_ptr, _in_guard) = slice_ptr_on_stream(in_slice, in_layout.start_offset(), &stream);
    let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
    unsafe {
        ffi::launch_moe_sum_bf16(
            out_ptr as *mut core::ffi::c_void,
            in_ptr as *const core::ffi::c_void,
            num_tokens as i32,
            hidden as i32,
            topk as i32,
            cu_stream,
        );
    }
    drop(out_guard);

    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((Storage::Cuda(storage), (num_tokens, hidden))))
}

/// One faithful `fused_moe_kernel` launch. `a` [num_a_rows, K] bf16, `b` [E, K, N]
/// bf16 (contiguous, stacked in/out layout). Returns C [num_valid_tokens, N] bf16.
#[allow(clippy::too_many_arguments)]
pub fn cutile_grouped_gemm(
    a: &Tensor,
    b: &Tensor,
    sorted_token_ids: &CudaSlice<i32>,
    expert_ids: &CudaSlice<i32>,
    num_tokens_post_pad: &CudaSlice<i32>,
    topk_weights: Option<&CudaSlice<f32>>,
    em: usize,
    num_valid_tokens: usize,
    top_k: usize,
    mul_routed_weight: bool,
    cfg: MoeTileConfig,
    dev: &CudaDevice,
) -> Result<Tensor> {
    assert_eq!(a.dtype(), DType::BF16, "cutile gemm is bf16-only");
    assert_eq!(b.dtype(), DType::BF16, "cutile gemm is bf16-only");
    let (_e, k_size, n_size) = b.dims3()?;
    assert_eq!(a.dim(1)?, k_size, "A K and B K mismatch");

    let mut out = unsafe { dev.alloc::<bf16>(num_valid_tokens * n_size)? };
    let stream = dev.cuda_stream();

    let (a_storage, a_layout) = a.storage_and_layout();
    let a_slice = match &*a_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
        _ => candle_core::bail!("a must be cuda"),
    };
    let (b_storage, b_layout) = b.storage_and_layout();
    let b_slice = match &*b_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
        _ => candle_core::bail!("b must be cuda"),
    };

    let (a_addr, _a_guard) = slice_ptr_on_stream(a_slice, a_layout.start_offset(), &stream);
    let (b_addr, _b_guard) = slice_ptr_on_stream(b_slice, b_layout.start_offset(), &stream);
    let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
    let (sids_addr, _sids_guard) = slice_ptr_on_stream(sorted_token_ids, 0, &stream);
    let (eids_addr, _eids_guard) = slice_ptr_on_stream(expert_ids, 0, &stream);
    let (ntpp_addr, _ntpp_guard) = slice_ptr_on_stream(num_tokens_post_pad, 0, &stream);
    let tw_guard;
    let tw_addr = match topk_weights {
        Some(tw) => {
            let (addr, guard) = slice_ptr_on_stream(tw, 0, &stream);
            tw_guard = Some(guard);
            addr
        }
        None => {
            tw_guard = None;
            0
        }
    };

    let num_pid_m = em.div_ceil(cfg.bm as usize);
    let num_pid_n = n_size.div_ceil(cfg.bn as usize);
    let grid_x = (num_pid_m * num_pid_n) as u32;

    let generics = vec![
        cfg.bm.to_string(),
        cfg.bn.to_string(),
        cfg.bk.to_string(),
        cfg.group_m.to_string(),
        (top_k as i32).to_string(),
        (if mul_routed_weight { 1 } else { 0 }).to_string(),
    ];

    let ctx = interop::execution_context(dev);
    let launcher = unsafe {
        moe::fused_moe::fused_moe_kernel(
            DevicePointer::<bf16>::from_cu_deviceptr(out_addr as CUdeviceptr),
            DevicePointer::<bf16>::from_cu_deviceptr(a_addr as CUdeviceptr),
            DevicePointer::<bf16>::from_cu_deviceptr(b_addr as CUdeviceptr),
            DevicePointer::<i32>::from_cu_deviceptr(sids_addr as CUdeviceptr),
            DevicePointer::<i32>::from_cu_deviceptr(eids_addr as CUdeviceptr),
            DevicePointer::<i32>::from_cu_deviceptr(ntpp_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(tw_addr as CUdeviceptr),
            n_size as i32,
            k_size as i32,
            em as i32,
            num_valid_tokens as i32,
        )
    }
    .generics(generics)
    .grid((grid_x, 1, 1));

    unsafe { launcher.execute(&ctx) }
        .map_err(|e| candle_core::Error::Msg(format!("cutile fused_moe launch: {e:?}")))?;
    drop((out_guard, tw_guard));

    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((
        Storage::Cuda(storage),
        (num_valid_tokens, n_size),
    )))
}
