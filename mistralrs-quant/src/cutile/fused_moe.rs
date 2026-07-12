//! The fused MoE grouped-GEMM cuTile kernel (bf16), its host-side launch (`cutile_grouped_gemm`), and its JIT warmup.
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

use candle_core::cuda::cudarc::driver::CudaSlice;
use candle_core::{CudaDevice, DType, Device, Result, Storage, Tensor};
use cuda_async::device_buffer::DevicePointer;
use cuda_async::device_operation::DeviceOp;
use cuda_core::sys::CUdeviceptr;
use cutile::tile_kernel::TileKernel;
use half::bf16;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::moe::cuda::{moe_align, moe_align_em};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

use super::warmup::CutileKernel;
use super::{context, get_default_config, MoeTileConfig};

#[cutile::module]
pub mod fused_moe {
    use cutile::core::*;

    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2,),
        )
    )]
    pub unsafe fn fused_moe_kernel<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const GROUP_M: i32,
        const TOP_K: i32,             // model top-k for gate_up, 1 for down
        const MUL_ROUTED_WEIGHT: i32, // 0 / 1
    >(
        out_ptr: *mut bf16,                   // C: [num_valid_tokens, N], stride (N, 1)
        a_ptr: *mut bf16,                     // A: [num_a_rows, K], stride (K, 1)
        b_ptr: *mut bf16,                     // B: [E, N, K], stride (N*K, K, 1)
        sorted_token_ids_ptr: *mut i32,       // [EM]
        expert_ids_ptr: *mut i32,             // [num_pid_m]
        num_tokens_post_padded_ptr: *mut i32, // scalar
        topk_weights_ptr: *mut f32,           // [num_valid_tokens]
        n_size: i32,
        k_size: i32,
        em: i32,
        num_valid_tokens: i32,
    ) {
        let pid: i32 = get_tile_block_id().0;
        let num_pid_m: i32 = ceil_div(em, BM);
        let num_pid_n: i32 = ceil_div(n_size, BN);
        let num_pid_in_group: i32 = GROUP_M * num_pid_n;
        let group_id: i32 = pid / num_pid_in_group;
        let first_pid_m: i32 = group_id * GROUP_M;
        let group_size_m: i32 = {
            let rem = num_pid_m - first_pid_m;
            if rem < GROUP_M {
                rem
            } else {
                GROUP_M
            }
        };
        let pid_m: i32 = first_pid_m + ((pid % num_pid_in_group) % group_size_m);
        let pid_n: i32 = (pid % num_pid_in_group) / group_size_m;

        // num_tokens_post_padded (scalar load)
        let ntpp_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(num_tokens_post_padded_ptr);
        let ntpp_p1: PointerTile<*mut i32, { [1] }> = ntpp_p0.reshape(const_shape![1]);
        let (ntpp_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            ntpp_p1,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let ntpp_s: Tile<i32, { [] }> = ntpp_t.reshape(const_shape![]);
        let ntpp: i32 = tile_to_scalar(ntpp_s);

        if pid_m * BM < ntpp {
            // offs_token = sorted_token_ids[pid_m*BM + arange(BM)]   (contiguous, sentinel fill)
            let iota_m: Tile<i32, { [BM] }> = iota(const_shape![BM]);
            let base_m: Tile<i32, { [BM] }> = broadcast_scalar(pid_m * BM, const_shape![BM]);
            let offs_token_id: Tile<i32, { [BM] }> = iota_m + base_m;
            let em_t: Tile<i32, { [BM] }> = broadcast_scalar(em, const_shape![BM]);
            let id_inb: Tile<bool, { [BM] }> = lt_tile(offs_token_id, em_t);

            let sids_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(sorted_token_ids_ptr);
            let sids_p1: PointerTile<*mut i32, { [1] }> = sids_p0.reshape(const_shape![1]);
            let sids_p2: PointerTile<*mut i32, { [BM] }> = sids_p1.broadcast(const_shape![BM]);
            let sids_ptrs: PointerTile<*mut i32, { [BM] }> = sids_p2.offset_tile(offs_token_id);
            let (offs_token, _): (Tile<i32, { [BM] }>, Token) = load_ptr_tko(
                sids_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(id_inb),
                Some(num_valid_tokens),
                None,
                Latency::<0>,
            );
            let nvt_t: Tile<i32, { [BM] }> = broadcast_scalar(num_valid_tokens, const_shape![BM]);
            let token_mask: Tile<bool, { [BM] }> = lt_tile(offs_token, nvt_t);

            // off_experts = expert_ids[pid_m]   (scalar)
            let eid_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(expert_ids_ptr);
            let eid_p1: PointerTile<*mut i32, { [1] }> = eid_p0.reshape(const_shape![1]);
            let pid_m_t: Tile<i32, { [1] }> = broadcast_scalar(pid_m, const_shape![1]);
            let eid_p2: PointerTile<*mut i32, { [1] }> = eid_p1.offset_tile(pid_m_t);
            let (eid_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
                eid_p2,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let eid_s: Tile<i32, { [] }> = eid_t.reshape(const_shape![]);
            let off_experts: i32 = tile_to_scalar(eid_s);

            let iota_n: Tile<i32, { [BN] }> = iota(const_shape![BN]);
            let base_n: Tile<i32, { [BN] }> = broadcast_scalar(pid_n * BN, const_shape![BN]);
            let offs_cn: Tile<i32, { [BN] }> = iota_n + base_n;
            let n_t_bn: Tile<i32, { [BN] }> = broadcast_scalar(n_size, const_shape![BN]);
            let cn_inb: Tile<bool, { [BN] }> = lt_tile(offs_cn, n_t_bn);

            let ot_col: Tile<i32, { [BM, 1] }> = offs_token.reshape(const_shape![BM, 1]);
            let ot_2d: Tile<i32, { [BM, BN] }> = ot_col.broadcast(const_shape![BM, BN]);
            let n_2d: Tile<i32, { [BM, BN] }> = broadcast_scalar(n_size, const_shape![BM, BN]);
            let ot_n: Tile<i32, { [BM, BN] }> = muli(ot_2d, n_2d, overflow::NoSignedWrap);
            let cn_row: Tile<i32, { [1, BN] }> = offs_cn.reshape(const_shape![1, BN]);
            let cn_2d: Tile<i32, { [BM, BN] }> = cn_row.broadcast(const_shape![BM, BN]);
            let c_off: Tile<i32, { [BM, BN] }> = ot_n + cn_2d;
            let c_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
            let c_base1: PointerTile<*mut bf16, { [1, 1] }> = c_base0.reshape(const_shape![1, 1]);
            let c_base2: PointerTile<*mut bf16, { [BM, BN] }> =
                c_base1.broadcast(const_shape![BM, BN]);
            let c_ptrs: PointerTile<*mut bf16, { [BM, BN] }> = c_base2.offset_tile(c_off);

            let tm_col: Tile<bool, { [BM, 1] }> = token_mask.reshape(const_shape![BM, 1]);
            let tm_2d: Tile<bool, { [BM, BN] }> = tm_col.broadcast(const_shape![BM, BN]);
            let cnb_row: Tile<bool, { [1, BN] }> = cn_inb.reshape(const_shape![1, BN]);
            let cnb_2d: Tile<bool, { [BM, BN] }> = cnb_row.broadcast(const_shape![BM, BN]);
            let c_mask: Tile<bool, { [BM, BN] }> = tm_2d & cnb_2d;

            if off_experts == -1 {
                let zeros: Tile<bf16, { [BM, BN] }> = constant(bf16::ZERO, const_shape![BM, BN]);
                store_ptr_tko(
                    c_ptrs,
                    zeros,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(c_mask),
                    None,
                    Latency::<0>,
                );
            } else {
                // offs_bn = offs_cn % N bounds B reads on the last N-tile; manual modulo (cuTile remi won't serialize).
                let n_tile: Tile<i32, { [BN] }> = broadcast_scalar(n_size, const_shape![BN]);
                let q_bn: Tile<i32, { [BN] }> = offs_cn / n_tile;
                let qn_bn: Tile<i32, { [BN] }> = muli(q_bn, n_tile, overflow::NoSignedWrap);
                let offs_bn: Tile<i32, { [BN] }> = subi(offs_cn, qn_bn, overflow::NoSignedWrap);
                let top_k_t: Tile<i32, { [BM] }> = broadcast_scalar(TOP_K, const_shape![BM]);
                let a_row: Tile<i32, { [BM] }> = offs_token / top_k_t;
                // Clamp padding rows (token_mask false) to row 0 so gathered A stays in-bounds; discarded at the C store.
                let zero_row: Tile<i32, { [BM] }> = broadcast_scalar(0i32, const_shape![BM]);
                let safe_row: Tile<i32, { [BM] }> = select(token_mask, a_row, zero_row);
                let k_t_bm: Tile<i32, { [BM] }> = broadcast_scalar(k_size, const_shape![BM]);
                let a_row_off: Tile<i32, { [BM] }> = muli(safe_row, k_t_bm, overflow::NoSignedWrap);
                let be: i32 = off_experts * (k_size * n_size);
                let a_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(a_ptr);
                let a_base1: PointerTile<*mut bf16, { [1, 1] }> =
                    a_base0.reshape(const_shape![1, 1]);
                let a_base2: PointerTile<*mut bf16, { [BM, BK] }> =
                    a_base1.broadcast(const_shape![BM, BK]);
                let b_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(b_ptr);
                let b_base1: PointerTile<*mut bf16, { [1, 1] }> =
                    b_base0.reshape(const_shape![1, 1]);
                let b_base2: PointerTile<*mut bf16, { [BK, BN] }> =
                    b_base1.broadcast(const_shape![BK, BN]);

                let iota_k: Tile<i32, { [BK] }> = iota(const_shape![BK]);
                let ar_col: Tile<i32, { [BM, 1] }> = a_row_off.reshape(const_shape![BM, 1]);
                let ar_2d: Tile<i32, { [BM, BK] }> = ar_col.broadcast(const_shape![BM, BK]);
                let ok_row: Tile<i32, { [1, BK] }> = iota_k.reshape(const_shape![1, BK]);
                let ok_2d_a: Tile<i32, { [BM, BK] }> = ok_row.broadcast(const_shape![BM, BK]);
                let a_off: Tile<i32, { [BM, BK] }> = ar_2d + ok_2d_a;
                let mut a_ptrs: PointerTile<*mut bf16, { [BM, BK] }> = a_base2.offset_tile(a_off);

                // ENK [E, N, K]: B[e,n,k] = be + n*k_size + k (K contiguous, N stride k_size); natural [out,in] weight.
                let be_2d: Tile<i32, { [BK, BN] }> = broadcast_scalar(be, const_shape![BK, BN]);
                let ok_col: Tile<i32, { [BK, 1] }> = iota_k.reshape(const_shape![BK, 1]);
                let ok_2d_b: Tile<i32, { [BK, BN] }> = ok_col.broadcast(const_shape![BK, BN]);
                let obn_row: Tile<i32, { [1, BN] }> = offs_bn.reshape(const_shape![1, BN]);
                let obn_2d: Tile<i32, { [BK, BN] }> = obn_row.broadcast(const_shape![BK, BN]);
                let k_2d_b: Tile<i32, { [BK, BN] }> =
                    broadcast_scalar(k_size, const_shape![BK, BN]);
                let obn_k: Tile<i32, { [BK, BN] }> = muli(obn_2d, k_2d_b, overflow::NoSignedWrap);
                let b_off_a: Tile<i32, { [BK, BN] }> = be_2d + obn_k;
                let b_off: Tile<i32, { [BK, BN] }> = b_off_a + ok_2d_b;
                let mut b_ptrs: PointerTile<*mut bf16, { [BK, BN] }> = b_base2.offset_tile(b_off);
                let a_step: Tile<i32, { [BM, BK] }> = broadcast_scalar(BK, const_shape![BM, BK]);
                let b_step: Tile<i32, { [BK, BN] }> = broadcast_scalar(BK, const_shape![BK, BN]);

                let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                let kt: i32 = ceil_div(k_size, BK);
                for kk in 0i32..kt {
                    let base_k: Tile<i32, { [BK] }> = broadcast_scalar(kk * BK, const_shape![BK]);
                    let offs_k: Tile<i32, { [BK] }> = iota_k + base_k;
                    let k_size_t: Tile<i32, { [BK] }> = broadcast_scalar(k_size, const_shape![BK]);
                    let k_mask_1d: Tile<bool, { [BK] }> = lt_tile(offs_k, k_size_t);
                    let a_mask_row: Tile<bool, { [1, BK] }> =
                        k_mask_1d.reshape(const_shape![1, BK]);
                    let a_mask: Tile<bool, { [BM, BK] }> =
                        a_mask_row.broadcast(const_shape![BM, BK]);
                    let b_mask_col: Tile<bool, { [BK, 1] }> =
                        k_mask_1d.reshape(const_shape![BK, 1]);
                    let b_k_mask: Tile<bool, { [BK, BN] }> =
                        b_mask_col.broadcast(const_shape![BK, BN]);
                    let b_n_mask_row: Tile<bool, { [1, BN] }> = cn_inb.reshape(const_shape![1, BN]);
                    let b_n_mask: Tile<bool, { [BK, BN] }> =
                        b_n_mask_row.broadcast(const_shape![BK, BN]);
                    let b_mask: Tile<bool, { [BK, BN] }> = b_k_mask & b_n_mask;
                    let (a_load, _): (Tile<bf16, { [BM, BK] }>, Token) = load_ptr_tko(
                        a_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(a_mask),
                        None,
                        None,
                        Latency::<0>,
                    );
                    let (b_load, _): (Tile<bf16, { [BK, BN] }>, Token) = load_ptr_tko(
                        b_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(b_mask),
                        None,
                        None,
                        Latency::<0>,
                    );
                    let a_zero: Tile<bf16, { [BM, BK] }> =
                        constant(bf16::ZERO, const_shape![BM, BK]);
                    let b_zero: Tile<bf16, { [BK, BN] }> =
                        constant(bf16::ZERO, const_shape![BK, BN]);
                    let a_raw: Tile<bf16, { [BM, BK] }> = select(a_mask, a_load, a_zero);
                    let b_tile: Tile<bf16, { [BK, BN] }> = select(b_mask, b_load, b_zero);

                    acc = mmaf(a_raw, b_tile, acc);
                    a_ptrs = a_ptrs.offset_tile(a_step);
                    b_ptrs = b_ptrs.offset_tile(b_step);
                }

                if MUL_ROUTED_WEIGHT != 0 {
                    let w_p0: PointerTile<*mut f32, { [] }> = pointer_to_tile(topk_weights_ptr);
                    let w_p1: PointerTile<*mut f32, { [1] }> = w_p0.reshape(const_shape![1]);
                    let w_p2: PointerTile<*mut f32, { [BM] }> = w_p1.broadcast(const_shape![BM]);
                    let w_ptrs: PointerTile<*mut f32, { [BM] }> = w_p2.offset_tile(offs_token);
                    let (moe_w, _): (Tile<f32, { [BM] }>, Token) = load_ptr_tko(
                        w_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(token_mask),
                        Some(0.0f32),
                        None,
                        Latency::<0>,
                    );
                    let moe_w_col: Tile<f32, { [BM, 1] }> = moe_w.reshape(const_shape![BM, 1]);
                    let moe_w_2d: Tile<f32, { [BM, BN] }> =
                        moe_w_col.broadcast(const_shape![BM, BN]);
                    acc = acc * moe_w_2d;
                }

                let acc_bf: Tile<bf16, { [BM, BN] }> = convert_tile(acc);
                store_ptr_tko(
                    c_ptrs,
                    acc_bf,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(c_mask),
                    None,
                    Latency::<0>,
                );
            }
        }
    }
}

/// One `fused_moe_kernel` launch: `a` [num_a_rows,K] bf16, `b` [E,N,K] bf16 -> C [num_valid_tokens,N] bf16.
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
    let (_e, n_size, k_size) = b.dims3()?;
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

    let ctx = context::execution_context(dev);
    let launcher = unsafe {
        fused_moe::fused_moe_kernel(
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
    let (Ok(hidden), Ok(inter)) = (gate_up_w.dim(2), down_w.dim(2)) else {
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

fn div_hint_class(x: i32) -> i32 {
    1i32 << (x as u32).trailing_zeros().min(4)
}

// Returns one representative token count per distinct kernel the forward can ever launch, so warmup
// can compile all of them up front. cuTile JIT-compiles per cache key into a thread-local cache with
// no cross-thread or on-disk reuse, so the first launch of an un-warmed key is a latency spike
// mid-inference. The point of this function is that the set of keys is CLOSED and small, so we can
// enumerate it exactly from (top_k, E) rather than guessing a list of token lengths.
//
// This kernel takes raw pointers + i32 scalars (no shaped tensors), so as the token count m varies
// the cache key moves only via (a) the generics from get_default_config(m, E) and (b) the scalar
// DivHints of `em` and `num_valid = top_k*m`. n_size/k_size are fixed model dims and raw pointers
// carry only a stable alignment hint, so neither adds an m-dependent axis.
//
// Two facts make the key set finite. The generics step at fixed m thresholds (bm at 32/96/512, bn/bk
// at 64, group_m flips 1->16 once integer m/E > 128, i.e. m >= 129*E). And DivHint(x) is the largest
// power-of-2 divisor clamped to 16, so it is decided purely by x mod 16 (the clamp makes higher
// divisibility irrelevant), hence both scalar hints are periodic in m with period <= 16. So within
// any one generics interval, 16 consecutive m sweep every (DivHint(em), DivHint(num_valid)) pair the
// interval can produce. Probing <= 16 m per interval and deduping the full key signature therefore
// yields a provably complete, minimal cover: every token count of any size maps to a warmed kernel.
//
// Full specialization is kept rather than collapsed to one kernel (CompileOptions::max_divisibility(1)),
// which would also drop the pointer-alignment vectorization.
fn warmup_token_counts(entry: &MoeWarmupEntry) -> Vec<usize> {
    let e = entry.num_experts.max(1);
    let k = entry.top_k;
    // m values where the generics can change; 129*e - 1 is the last m with group_m=1.
    let mut bps = vec![32usize, 64, 96, 512, 129 * e - 1];
    bps.sort_unstable();
    bps.dedup();

    // One <=16-wide window per interval: 16 consecutive m cover all residues mod 16, so the window
    // hits every DivHint class the (constant-generics) interval can produce.
    let mut probes = Vec::new();
    let mut lo = 1usize;
    for &bp in &bps {
        for m in lo..=bp.min(lo + 15) {
            probes.push(m);
        }
        lo = bp + 1;
    }
    // Tail above the largest breakpoint: the group_m=16 regime, unbounded in m but one kernel.
    probes.extend(lo..=lo + 15);

    let mut seen = HashSet::new();
    let mut reps = Vec::new();
    for m in probes {
        let cfg = get_default_config(m, e);
        let em = moe_align_em(m, k, e, cfg.bm as usize);
        // sig mirrors the cuTile cache key for this launch, so distinct sigs == distinct kernels.
        let sig = (
            cfg.bm,
            cfg.bn,
            cfg.bk,
            cfg.group_m,
            div_hint_class(em as i32),
            div_hint_class((k * m) as i32),
        );
        if seen.insert(sig) {
            reps.push(m);
        }
    }
    reps
}

fn warmup_moe_kernels_uncached(dev: &CudaDevice) -> Result<()> {
    let entries: Vec<MoeWarmupEntry> = MOE_SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .clone();
    if entries.is_empty() {
        return Ok(());
    }
    let plan: Vec<(MoeWarmupEntry, Vec<usize>)> = entries
        .into_iter()
        .map(|e| {
            let ms = warmup_token_counts(&e);
            (e, ms)
        })
        .collect();
    let total: usize = plan.iter().map(|(_, ms)| ms.len()).sum();
    tracing::info!("Warming {total} cuTile MoE kernels.");
    let bar = ProgressBar::new(total as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} warming cuTile MoE kernels ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    for (entry, ms) in &plan {
        for &m in ms {
            if let Err(err) = warmup_shape(dev, entry, m) {
                tracing::warn!(
                    "cuTile MoE warmup failed (hidden={} inter={} m={m}): {err}",
                    entry.hidden,
                    entry.inter
                );
            }
            bar.inc(1);
        }
    }
    bar.finish_and_clear();
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

/// The one cuTile kernel today; the warmup driver in `warmup` keeps it in its registry.
pub struct FusedMoeKernel;
pub static FUSED_MOE: FusedMoeKernel = FusedMoeKernel;

impl CutileKernel for FusedMoeKernel {
    fn warm(&self, dev: &CudaDevice) -> Result<()> {
        warmup_moe_kernels_uncached(dev)
    }
}
