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

use crate::utils::slice_ptr;

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

pub fn get_default_config(m: usize, num_experts: usize) -> MoeTileConfig {
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
    let group_m = if m / num_experts.max(1) > 128 { 16 } else { 1 };
    MoeTileConfig {
        bm,
        bn,
        bk,
        group_m,
    }
}

/// vLLM `max_num_tokens_padded` (EM): padded length of `sorted_token_ids`.
pub fn moe_align_em(num_tokens: usize, topk: usize, num_experts: usize, block_size: usize) -> usize {
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

    let sids = unsafe { dev.alloc::<i32>(em)? };
    let eids = unsafe { dev.alloc::<i32>(nblocks)? };
    let ntpp = unsafe { dev.alloc::<i32>(1)? };
    let cumsum = unsafe { dev.alloc::<i32>(num_experts + 1)? };

    let cu_stream = dev.cuda_stream().cu_stream();
    let (tk_ptr, _) = slice_ptr(topk_ids_u32, 0);
    let (s_ptr, _) = slice_ptr(&sids, 0);
    let (e_ptr, _) = slice_ptr(&eids, 0);
    let (n_ptr, _) = slice_ptr(&ntpp, 0);
    let (c_ptr, _) = slice_ptr(&cumsum, 0);
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
    Ok((sids, eids, ntpp, em))
}

/// vLLM `gelu_tanh_and_mul`: input [num_tokens, 2*d] -> [num_tokens, d] bf16.
pub fn gelu_tanh_and_mul(input: &Tensor, d: usize, dev: &CudaDevice) -> Result<Tensor> {
    let (num_tokens, two_d) = input.dims2()?;
    assert_eq!(two_d, 2 * d, "gelu_tanh_and_mul expects last dim == 2*d");
    assert_eq!(input.dtype(), DType::BF16, "cutile gelu path is bf16-only");

    let out = unsafe { dev.alloc::<bf16>(num_tokens * d)? };
    let cu_stream = dev.cuda_stream().cu_stream();

    let (in_storage, in_layout) = input.storage_and_layout();
    let in_slice = match &*in_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
        _ => candle_core::bail!("input must be cuda"),
    };
    let (in_ptr, _) = slice_ptr(in_slice, in_layout.start_offset());
    let (out_ptr, _) = slice_ptr(&out, 0);
    unsafe {
        ffi::launch_gelu_tanh_and_mul_bf16(
            out_ptr as *mut core::ffi::c_void,
            in_ptr as *const core::ffi::c_void,
            num_tokens as i32,
            d as i32,
            cu_stream,
        );
    }

    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((Storage::Cuda(storage), (num_tokens, d))))
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

    let out = unsafe { dev.alloc::<bf16>(num_valid_tokens * n_size)? };

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

    let (a_addr, _) = slice_ptr(a_slice, a_layout.start_offset());
    let (b_addr, _) = slice_ptr(b_slice, b_layout.start_offset());
    let (out_addr, _) = slice_ptr(&out, 0);
    let (sids_addr, _) = slice_ptr(sorted_token_ids, 0);
    let (eids_addr, _) = slice_ptr(expert_ids, 0);
    let (ntpp_addr, _) = slice_ptr(num_tokens_post_pad, 0);
    // GEMM1 (mul=0) never reads topk_weights but the launcher still needs a
    // valid pointer arg; a 1-elem alloc stands in.
    let tw_fallback;
    let tw_addr = match topk_weights {
        Some(tw) => slice_ptr(tw, 0).0,
        None => {
            tw_fallback = unsafe { dev.alloc::<f32>(1)? };
            slice_ptr(&tw_fallback, 0).0
        }
    };

    let num_pid_m = em.div_ceil(cfg.bm as usize);
    let num_pid_n = n_size.div_ceil(cfg.bn as usize);
    let grid_x = (num_pid_m * num_pid_n) as u32;

    let generics = vec![
        "bf16".to_string(),
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
            core::mem::size_of::<bf16>() as i64,
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

    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((Storage::Cuda(storage), (num_valid_tokens, n_size))))
}
