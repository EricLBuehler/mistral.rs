//! Blockwise FP8 W8A8 fused MoE on cuTile: the grouped GEMM with 128x128 weight scales and 1x128
//! activation scales, the full expert forward around it, and its JIT warmup.
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

use candle_core::cuda::cudarc::driver::CudaSlice;
use candle_core::{CudaDevice, DType, Device, Result, Storage, Tensor};
use cutile::core::f8e4m3fn;
use cutile::cuda_async::device_buffer::DevicePointer;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tile_kernel::TileKernel;
use float8::F8E4M3;
use half::bf16;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::blockwise_fp8::mma::quantize_activation_padded;
use crate::moe::cuda::{moe_align, moe_sum_bf16};
use crate::utils::{fused_split_glu_quantized_bf16, slice_ptr_mut_on_stream, slice_ptr_on_stream};
use crate::GluActivationType;

use super::fused_moe::{warmup_token_counts_for, MoeAlign};
use super::split_k::reduce_split_k;
use super::tune::{
    cutile_error, tune, Bucket, Prepared, Space, TuneMode, TuneRequest, TuneRouting, TunedTable,
    TUNE_WEIGHT_SETS,
};
use super::warmup::CutileKernel;
use super::{catch_cutile_panic, context, MoeShapeKey, MoeTileConfig};

/// Scale group along K and N; the kernel steps K one group at a time so every partial product is
/// scaled by a single (row, group) pair before it joins the accumulator.
pub const FP8_MOE_GROUP: usize = 128;

#[cutile::module]
pub mod fused_moe_fp8 {
    use cutile::core::*;
    use cutile::cutile_compiler;

    // Same routing as the bf16 kernel: one tile block per (aligned token block, N tile). A rows are
    // gathered per token, B is the expert's [N, K] slab, and each BK = 128 step is scaled by the
    // token's activation scale and the expert's weight scale for that (N, K) block.
    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2,),
        )
    )]
    pub unsafe fn fused_moe_fp8_kernel<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const GROUP_M: i32,
        const TOP_K: i32,
        const MUL_ROUTED_WEIGHT: i32,
        const SPLIT_K: i32, // K ranges per output tile; partials go to partial_ptr when > 1
        const LATENCY: i32, // operand-load pipelining hint, 0 = compiler default
    >(
        out_ptr: *mut bf16,                   // C: [num_valid_tokens, N]
        partial_ptr: *mut f32,                // [SPLIT_K, num_valid_tokens, N]
        a_ptr: *mut f8e4m3fn,                 // A: [rows, K]
        a_scale_ptr: *mut f32,                // [K / BK, a_scale_stride], group-major
        b_ptr: *mut f8e4m3fn,                 // B: [E, N, K]
        b_scale_ptr: *mut f32,                // [E, N / 128, K / 128]
        sorted_token_ids_ptr: *mut i32,       // [EM]
        expert_ids_ptr: *mut i32,             // [num_pid_m]
        num_tokens_post_padded_ptr: *mut i32, // scalar
        topk_weights_ptr: *mut f32,           // [num_valid_tokens]
        n_size: i32,
        k_size: i32,
        em: i32,
        num_valid_tokens: i32,
        a_scale_stride: i32,
    ) {
        let pid_all: i32 = get_tile_block_id().0;
        let split: i32 = pid_all % SPLIT_K;
        let pid: i32 = pid_all / SPLIT_K;
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
            let c_mask: Tile<bool, { [BM, BN] }> = tm_col.broadcast(const_shape![BM, BN]);
            let p_base0: PointerTile<*mut f32, { [] }> = pointer_to_tile(partial_ptr);
            let p_base1: PointerTile<*mut f32, { [1, 1] }> = p_base0.reshape(const_shape![1, 1]);
            let p_base2: PointerTile<*mut f32, { [BM, BN] }> =
                p_base1.broadcast(const_shape![BM, BN]);
            let slice_off: Tile<i32, { [BM, BN] }> =
                broadcast_scalar(split * (num_valid_tokens * n_size), const_shape![BM, BN]);
            let p_ptrs: PointerTile<*mut f32, { [BM, BN] }> =
                p_base2.offset_tile(c_off + slice_off);
            if off_experts == -1 {
                if SPLIT_K > 1 {
                    let zeros: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                    store_ptr_tko(
                        p_ptrs,
                        zeros,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(c_mask),
                        None,
                        Latency::<0>,
                    );
                } else {
                    let zeros: Tile<bf16, { [BM, BN] }> =
                        constant(bf16::ZERO, const_shape![BM, BN]);
                    store_ptr_tko(
                        c_ptrs,
                        zeros,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(c_mask),
                        None,
                        Latency::<0>,
                    );
                }
            } else {
                let top_k_t: Tile<i32, { [BM] }> = broadcast_scalar(TOP_K, const_shape![BM]);
                let a_row: Tile<i32, { [BM] }> = offs_token / top_k_t;
                // padding rows read row 0 so the gathers stay in bounds; the C store drops them
                let zero_row: Tile<i32, { [BM] }> = broadcast_scalar(0i32, const_shape![BM]);
                let safe_row: Tile<i32, { [BM] }> = select(token_mask, a_row, zero_row);
                let k_t_bm: Tile<i32, { [BM] }> = broadcast_scalar(k_size, const_shape![BM]);
                let a_row_off: Tile<i32, { [BM] }> = muli(safe_row, k_t_bm, overflow::NoSignedWrap);
                let be: i32 = off_experts * (k_size * n_size);
                let a_base0: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(a_ptr);
                let a_base1: PointerTile<*mut f8e4m3fn, { [1, 1] }> =
                    a_base0.reshape(const_shape![1, 1]);
                let a_base2: PointerTile<*mut f8e4m3fn, { [BM, BK] }> =
                    a_base1.broadcast(const_shape![BM, BK]);
                let b_base0: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(b_ptr);
                let b_base1: PointerTile<*mut f8e4m3fn, { [1, 1] }> =
                    b_base0.reshape(const_shape![1, 1]);
                let b_base2: PointerTile<*mut f8e4m3fn, { [BK, BN] }> =
                    b_base1.broadcast(const_shape![BK, BN]);
                let iota_k: Tile<i32, { [BK] }> = iota(const_shape![BK]);
                let ar_col: Tile<i32, { [BM, 1] }> = a_row_off.reshape(const_shape![BM, 1]);
                let ar_2d: Tile<i32, { [BM, BK] }> = ar_col.broadcast(const_shape![BM, BK]);
                let ok_row: Tile<i32, { [1, BK] }> = iota_k.reshape(const_shape![1, BK]);
                let ok_2d_a: Tile<i32, { [BM, BK] }> = ok_row.broadcast(const_shape![BM, BK]);
                let a_off: Tile<i32, { [BM, BK] }> = ar_2d + ok_2d_a;
                let mut a_ptrs: PointerTile<*mut f8e4m3fn, { [BM, BK] }> =
                    a_base2.offset_tile(a_off);
                // ENK [E, N, K]: B[e, n, k] = be + n * k_size + k
                let be_2d: Tile<i32, { [BK, BN] }> = broadcast_scalar(be, const_shape![BK, BN]);
                let ok_col: Tile<i32, { [BK, 1] }> = iota_k.reshape(const_shape![BK, 1]);
                let ok_2d_b: Tile<i32, { [BK, BN] }> = ok_col.broadcast(const_shape![BK, BN]);
                let obn_row: Tile<i32, { [1, BN] }> = offs_cn.reshape(const_shape![1, BN]);
                let obn_2d: Tile<i32, { [BK, BN] }> = obn_row.broadcast(const_shape![BK, BN]);
                let k_2d_b: Tile<i32, { [BK, BN] }> =
                    broadcast_scalar(k_size, const_shape![BK, BN]);
                let obn_k: Tile<i32, { [BK, BN] }> = muli(obn_2d, k_2d_b, overflow::NoSignedWrap);
                let b_off_a: Tile<i32, { [BK, BN] }> = be_2d + obn_k;
                let b_off: Tile<i32, { [BK, BN] }> = b_off_a + ok_2d_b;
                let mut b_ptrs: PointerTile<*mut f8e4m3fn, { [BK, BN] }> =
                    b_base2.offset_tile(b_off);
                let a_step: Tile<i32, { [BM, BK] }> = broadcast_scalar(BK, const_shape![BM, BK]);
                let b_step: Tile<i32, { [BK, BN] }> = broadcast_scalar(BK, const_shape![BK, BN]);

                // activation scales: [K / BK, stride] group-major, one column per token row
                let xs_p0: PointerTile<*mut f32, { [] }> = pointer_to_tile(a_scale_ptr);
                let xs_p1: PointerTile<*mut f32, { [1] }> = xs_p0.reshape(const_shape![1]);
                let xs_p2: PointerTile<*mut f32, { [BM] }> = xs_p1.broadcast(const_shape![BM]);
                let xs_step: Tile<i32, { [BM] }> =
                    broadcast_scalar(a_scale_stride, const_shape![BM]);
                let mut xs_off: Tile<i32, { [BM] }> = safe_row;
                // weight scales: [E, N / 128, K / 128], one value per 128-column block of this tile
                let k_groups: i32 = k_size / 128;
                let n_groups: i32 = n_size / 128;
                let blk_bn: Tile<i32, { [BN] }> = broadcast_scalar(128i32, const_shape![BN]);
                let cn_group: Tile<i32, { [BN] }> = offs_cn / blk_bn;
                let kg_bn: Tile<i32, { [BN] }> = broadcast_scalar(k_groups, const_shape![BN]);
                let ws_row: Tile<i32, { [BN] }> = muli(cn_group, kg_bn, overflow::NoSignedWrap);
                let ws_base: Tile<i32, { [BN] }> =
                    broadcast_scalar(off_experts * (n_groups * k_groups), const_shape![BN]);
                let mut ws_off: Tile<i32, { [BN] }> = ws_base + ws_row;
                let ws_p0: PointerTile<*mut f32, { [] }> = pointer_to_tile(b_scale_ptr);
                let ws_p1: PointerTile<*mut f32, { [1] }> = ws_p0.reshape(const_shape![1]);
                let ws_p2: PointerTile<*mut f32, { [BN] }> = ws_p1.broadcast(const_shape![BN]);
                let ws_step: Tile<i32, { [BN] }> = broadcast_scalar(1i32, const_shape![BN]);
                let zero_bm: Tile<f32, { [BM] }> = constant(0.0f32, const_shape![BM]);
                let zero_acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);

                let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                let kt: i32 = k_size / BK / SPLIT_K;
                let k_begin: i32 = split * kt;
                let a_skip: Tile<i32, { [BM, BK] }> =
                    broadcast_scalar(k_begin * BK, const_shape![BM, BK]);
                let b_skip: Tile<i32, { [BK, BN] }> =
                    broadcast_scalar(k_begin * BK, const_shape![BK, BN]);
                let xs_skip: Tile<i32, { [BM] }> =
                    broadcast_scalar(k_begin * a_scale_stride, const_shape![BM]);
                let ws_skip: Tile<i32, { [BN] }> = broadcast_scalar(k_begin, const_shape![BN]);
                a_ptrs = a_ptrs.offset_tile(a_skip);
                b_ptrs = b_ptrs.offset_tile(b_skip);
                xs_off = xs_off + xs_skip;
                ws_off = ws_off + ws_skip;
                for _kk in 0i32..kt {
                    // the latency hint is a type-level literal, so the generic picks among fixed values
                    let a_tile: Tile<f8e4m3fn, { [BM, BK] }> = if LATENCY >= 4 {
                        let (t, _): (Tile<f8e4m3fn, { [BM, BK] }>, Token) = load_ptr_tko(
                            a_ptrs,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<4>,
                        );
                        t
                    } else if LATENCY >= 2 {
                        let (t, _): (Tile<f8e4m3fn, { [BM, BK] }>, Token) = load_ptr_tko(
                            a_ptrs,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<2>,
                        );
                        t
                    } else if LATENCY == 1 {
                        let (t, _): (Tile<f8e4m3fn, { [BM, BK] }>, Token) = load_ptr_tko(
                            a_ptrs,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<1>,
                        );
                        t
                    } else {
                        let (t, _): (Tile<f8e4m3fn, { [BM, BK] }>, Token) = load_ptr_tko(
                            a_ptrs,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        t
                    };
                    let b_tile: Tile<f8e4m3fn, { [BK, BN] }> = if LATENCY >= 4 {
                        let (t, _): (Tile<f8e4m3fn, { [BK, BN] }>, Token) = load_ptr_tko(
                            b_ptrs,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<4>,
                        );
                        t
                    } else if LATENCY >= 2 {
                        let (t, _): (Tile<f8e4m3fn, { [BK, BN] }>, Token) = load_ptr_tko(
                            b_ptrs,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<2>,
                        );
                        t
                    } else if LATENCY == 1 {
                        let (t, _): (Tile<f8e4m3fn, { [BK, BN] }>, Token) = load_ptr_tko(
                            b_ptrs,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<1>,
                        );
                        t
                    } else {
                        let (t, _): (Tile<f8e4m3fn, { [BK, BN] }>, Token) = load_ptr_tko(
                            b_ptrs,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        t
                    };
                    let part: Tile<f32, { [BM, BN] }> = mmaf(a_tile, b_tile, zero_acc);
                    let xs_ptrs: PointerTile<*mut f32, { [BM] }> = xs_p2.offset_tile(xs_off);
                    let (sx_load, _): (Tile<f32, { [BM] }>, Token) = load_ptr_tko(
                        xs_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(token_mask),
                        Some(0.0f32),
                        None,
                        Latency::<0>,
                    );
                    let sx: Tile<f32, { [BM] }> = select(token_mask, sx_load, zero_bm);
                    let ws_ptrs: PointerTile<*mut f32, { [BN] }> = ws_p2.offset_tile(ws_off);
                    let (sw, _): (Tile<f32, { [BN] }>, Token) = load_ptr_tko(
                        ws_ptrs,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let sx_2d: Tile<f32, { [BM, BN] }> = sx
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, BN]);
                    let sw_2d: Tile<f32, { [BM, BN] }> = sw
                        .reshape(const_shape![1, BN])
                        .broadcast(const_shape![BM, BN]);
                    let scaled: Tile<f32, { [BM, BN] }> = part * sx_2d * sw_2d;
                    acc = acc + scaled;
                    a_ptrs = a_ptrs.offset_tile(a_step);
                    b_ptrs = b_ptrs.offset_tile(b_step);
                    xs_off = xs_off + xs_step;
                    ws_off = ws_off + ws_step;
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
                    let moe_w_2d: Tile<f32, { [BM, BN] }> = moe_w
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, BN]);
                    acc = acc * moe_w_2d;
                }
                if SPLIT_K > 1 {
                    store_ptr_tko(
                        p_ptrs,
                        acc,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(c_mask),
                        None,
                        Latency::<0>,
                    );
                } else {
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
}

/// Stacked blockwise FP8 expert weights in ENK layout with their [E, N/128, K/128] scales.
#[derive(Clone)]
pub struct CutileFp8MoeWeights {
    pub gate_up: Tensor,
    pub gate_up_scales: Tensor,
    pub down: Tensor,
    pub down_scales: Tensor,
    pub num_experts: usize,
    pub top_k: usize,
}

impl CutileFp8MoeWeights {
    pub fn new(
        gate_up: Tensor,
        gate_up_scales: Tensor,
        down: Tensor,
        down_scales: Tensor,
        num_experts: usize,
        top_k: usize,
    ) -> Result<Self> {
        let (e1, two_inter, hidden) = gate_up.dims3()?;
        let (e2, hidden2, inter) = down.dims3()?;
        if e1 != num_experts
            || e2 != num_experts
            || hidden != hidden2
            || two_inter != 2 * inter
            || !hidden.is_multiple_of(FP8_MOE_GROUP)
            || !inter.is_multiple_of(FP8_MOE_GROUP)
            || top_k == 0
        {
            candle_core::bail!(
                "cuTile FP8 MoE needs [E, 2I, H] and [E, H, I] experts with H and I multiples of {FP8_MOE_GROUP}"
            )
        }
        if gate_up.dtype() != DType::F8E4M3
            || down.dtype() != DType::F8E4M3
            || gate_up_scales.dtype() != DType::F32
            || down_scales.dtype() != DType::F32
        {
            candle_core::bail!("cuTile FP8 MoE needs E4M3 experts with F32 scales")
        }
        let groups = |n: usize, k: usize| (num_experts, n / FP8_MOE_GROUP, k / FP8_MOE_GROUP);
        if gate_up_scales.dims3()? != groups(two_inter, hidden)
            || down_scales.dims3()? != groups(hidden, inter)
        {
            candle_core::bail!("cuTile FP8 MoE scale shapes do not match the experts")
        }
        for tensor in [&gate_up, &gate_up_scales, &down, &down_scales] {
            if !tensor.is_contiguous() {
                candle_core::bail!("cuTile FP8 MoE needs contiguous experts and scales")
            }
        }
        Ok(Self {
            gate_up,
            gate_up_scales,
            down,
            down_scales,
            num_experts,
            top_k,
        })
    }

    pub fn hidden(&self) -> usize {
        self.down.dim(1).expect("validated at construction")
    }

    pub fn inter(&self) -> usize {
        self.down.dim(2).expect("validated at construction")
    }

    pub fn shape_key(&self) -> MoeShapeKey {
        MoeShapeKey {
            hidden: self.hidden(),
            inter: self.inter(),
            num_experts: self.num_experts,
            top_k: self.top_k,
        }
    }
}

/// Whether this device and expert shape can run the cuTile FP8 MoE.
pub fn fp8_moe_supported(dev: &CudaDevice, hidden: usize, inter: usize) -> bool {
    super::jit_available(dev)
        && hidden.is_multiple_of(FP8_MOE_GROUP)
        && inter.is_multiple_of(FP8_MOE_GROUP)
}

const TUNE_KERNEL: &str = "fused_moe_fp8";
/// Token counts where the launch policy may change; the warmup key enumeration probes around them.
const DECODE_TOKENS: usize = 96;
const CHUNK_TOKENS: usize = 512;
/// Decode is timed at a serving-sized batch, prefill at a full chunk of a long prompt.
const DECODE_PROBE_TOKENS: usize = 32;
const PREFILL_PROBE_TOKENS: usize = 4096;
static TUNED: TunedTable<MoeShapeKey, MoeTileConfig> = TunedTable::new();

// Measured on GB10 (see the tile sweep test): 64x64 tiles with a 16-row group swizzle run the
// 4k-token regime 1.5x faster than 128x128; below ~96 tokens the kernel is HBM-bound on expert
// slabs, where thin 16x64 tiles with a small swizzle are never slower and 8% faster when a batch
// of similar prompts piles 32 rows onto the same experts.
fn fp8_policy(m: usize) -> MoeTileConfig {
    let small = m <= DECODE_TOKENS;
    MoeTileConfig::tiles(
        if small { 16 } else { 64 },
        64,
        FP8_MOE_GROUP as i32,
        if small { 8 } else { 16 },
    )
}

fn fp8_config(m: usize, shape: MoeShapeKey) -> MoeTileConfig {
    TUNED.get(shape, m).unwrap_or_else(|| fp8_policy(m))
}

fn fp8_buckets() -> [Bucket; 3] {
    [
        Bucket {
            upper: DECODE_TOKENS,
            probe: DECODE_PROBE_TOKENS,
        },
        Bucket {
            upper: CHUNK_TOKENS,
            probe: CHUNK_TOKENS,
        },
        Bucket {
            upper: usize::MAX,
            probe: PREFILL_PROBE_TOKENS,
        },
    ]
}

/// The search space for one bucket: tile shapes as a grid, then the knobs one axis at a time;
/// split-K only where both GEMMs' K groups divide.
fn fp8_space(bucket: Bucket, shape: MoeShapeKey) -> Space {
    let tiles: &[[i64; 3]] = if bucket.upper <= DECODE_TOKENS {
        &[
            [16, 64, 8],
            [16, 64, 16],
            [16, 128, 1],
            [16, 128, 8],
            [32, 64, 8],
        ]
    } else {
        &[
            [64, 64, 16],
            [64, 64, 8],
            [64, 64, 32],
            [64, 128, 8],
            [128, 64, 8],
            [32, 64, 8],
        ]
    };
    let k_groups = [shape.hidden, shape.inter].map(|k| (k / FP8_MOE_GROUP) as i64);
    Space::new()
        .joint(["bm", "bn", "group_m"], tiles.iter().copied())
        .axis("latency", [0, 2, 4])
        .axis("warps", [0, 4, 8])
        .axis("occupancy", [0, 2])
        .axis("split_k", [1, 2, 4])
        .constrain(move |c| {
            c.int("split_k")
                .is_some_and(|s| k_groups.iter().all(|g| g % s == 0))
        })
        .policy(fp8_policy(bucket.probe).to_config())
}

/// Synthetic operands for one probe token count, built once so every candidate sees the same
/// inputs and the gate compares like with like.
struct Fp8Operands {
    a1: Tensor,
    a1_scales: Tensor,
    a2: Tensor,
    a2_scales: Tensor,
    tw: Arc<CudaSlice<f32>>,
    topk_ids: Arc<CudaSlice<u32>>,
    num_valid: usize,
    align: HashMap<i32, Arc<MoeAlign>>,
}

/// Prepares timed launch sets for the tuner, rotating through the registered weight sets.
pub(super) struct Fp8Tuner {
    dev: CudaDevice,
    sets: Vec<CutileFp8MoeWeights>,
    routing: Option<TuneRouting>,
    zero_inputs: bool,
    operands: HashMap<usize, Fp8Operands>,
}

impl Fp8Tuner {
    pub(super) fn new(dev: &CudaDevice, sets: &[CutileFp8MoeWeights]) -> Self {
        Self {
            dev: dev.clone(),
            sets: sets.to_vec(),
            routing: None,
            zero_inputs: false,
            operands: HashMap::new(),
        }
    }

    /// Fixes the routing instead of deriving it from the token count (the sweep test).
    #[cfg(test)]
    pub(super) fn with_routing(mut self, routing: TuneRouting) -> Self {
        self.routing = Some(routing);
        self
    }

    /// All-zero activations, which let the tensor cores clock higher than real data does (the sweep test).
    #[cfg(test)]
    pub(super) fn with_zero_inputs(mut self) -> Self {
        self.zero_inputs = true;
        self
    }

    fn operands(&mut self, m: usize) -> Result<&mut Fp8Operands> {
        if !self.operands.contains_key(&m) {
            let device = Device::Cuda(self.dev.clone());
            let first = &self.sets[0];
            let (hidden, inter) = (first.hidden(), first.inter());
            let num_valid = m * first.top_k;
            let routing = self.routing.unwrap_or_else(|| TuneRouting::for_tokens(m));
            let ids = routing.expert_ids(m, first.top_k, first.num_experts);
            let mut topk_ids = unsafe { self.dev.alloc::<u32>(num_valid)? };
            self.dev.memcpy_htod(&ids, &mut topk_ids)?;
            let mut tw = unsafe { self.dev.alloc::<f32>(num_valid)? };
            self.dev.memcpy_htod(&vec![0.5f32; num_valid], &mut tw)?;
            let zero_inputs = self.zero_inputs;
            let input = |shape: (usize, usize)| -> Result<Tensor> {
                if zero_inputs {
                    Tensor::zeros(shape, DType::BF16, &device)
                } else {
                    Tensor::rand(-1f32, 1f32, shape, &device)?.to_dtype(DType::BF16)
                }
            };
            let token_rows = m.div_ceil(FP8_MOE_GROUP) * FP8_MOE_GROUP;
            let (a1, a1_scales) = quantize_activation_padded(&input((m, hidden))?, token_rows)?;
            let valid_rows = num_valid.div_ceil(FP8_MOE_GROUP) * FP8_MOE_GROUP;
            let (a2, a2_scales) =
                quantize_activation_padded(&input((num_valid, inter))?, valid_rows)?;
            self.operands.insert(
                m,
                Fp8Operands {
                    a1,
                    a1_scales,
                    a2,
                    a2_scales,
                    tw: Arc::new(tw),
                    topk_ids: Arc::new(topk_ids),
                    num_valid,
                    align: HashMap::new(),
                },
            );
        }
        Ok(self.operands.get_mut(&m).expect("inserted above"))
    }

    pub(super) fn prepare(&mut self, m: usize, cfg: MoeTileConfig) -> Result<Prepared> {
        let dev = self.dev.clone();
        let sets = self.sets.clone();
        let (experts, top_k) = (sets[0].num_experts, sets[0].top_k);
        let ops = self.operands(m)?;
        if !ops.align.contains_key(&cfg.bm) {
            let align = MoeAlign::build(&dev, &ops.topk_ids, m, experts, top_k, cfg.bm)?;
            ops.align.insert(cfg.bm, Arc::new(align));
        }
        let align = ops.align[&cfg.bm].clone();
        let (a1, a1_scales) = (ops.a1.clone(), ops.a1_scales.clone());
        let (a2, a2_scales) = (ops.a2.clone(), ops.a2_scales.clone());
        let (tw, num_valid) = (ops.tw.clone(), ops.num_valid);
        let launch = move |w: &CutileFp8MoeWeights, compile_only: bool| -> Result<Tensor> {
            grouped_gemm_fp8(
                &a1,
                &a1_scales,
                &w.gate_up,
                &w.gate_up_scales,
                &align.sids,
                &align.eids,
                &align.ntpp,
                None,
                align.em,
                num_valid,
                top_k,
                false,
                cfg,
                &dev,
                compile_only,
            )?;
            grouped_gemm_fp8(
                &a2,
                &a2_scales,
                &w.down,
                &w.down_scales,
                &align.sids,
                &align.eids,
                &align.ntpp,
                Some(&tw),
                align.em,
                num_valid,
                1,
                true,
                cfg,
                &dev,
                compile_only,
            )
        };
        launch(&sets[0], true)?;
        let sample = launch(&sets[0], false)?;
        let mut next = 0usize;
        let run = Box::new(move |_: &Arc<cutile::cuda_core::Stream>| {
            let w = &sets[next % sets.len()];
            next += 1;
            launch(w, false).map(|_| ()).map_err(cutile_error)
        });
        Ok(Prepared { run, sample })
    }
}

/// Routing tensors as contiguous device slices starting at offset zero, as the kernels expect.
fn routing_slice<T: candle_core::cuda::cudarc::driver::DeviceRepr + candle_core::WithDType>(
    tensor: &Tensor,
) -> Result<Tensor> {
    let tensor = tensor.to_dtype(T::DTYPE)?.flatten_all()?.contiguous()?;
    let offset = tensor.layout().start_offset();
    if offset == 0 {
        Ok(tensor)
    } else {
        tensor.copy()
    }
}

/// `x` [T, H] BF16, `topk_ids` U32 [T, top_k], `topk_weights` F32 [T, top_k] -> [T, H] BF16.
pub fn cutile_fused_moe_fp8(
    x: &Tensor,
    weights: &CutileFp8MoeWeights,
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    activation: GluActivationType,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let (num_tokens, hidden) = x.dims2()?;
    if hidden != weights.hidden() || x.dtype() != DType::BF16 {
        candle_core::bail!("cuTile FP8 MoE input must be BF16 [tokens, hidden]")
    }
    let top_k = weights.top_k;
    let inter = weights.inter();
    let num_valid = num_tokens * top_k;
    if topk_ids.dims2()? != (num_tokens, top_k) || topk_weights.dims2()? != (num_tokens, top_k) {
        candle_core::bail!("cuTile FP8 MoE routing shapes do not match the input")
    }
    let cfg = fp8_config(num_tokens, weights.shape_key());
    let topk_ids = routing_slice::<u32>(topk_ids)?;
    let (ids_storage, _) = topk_ids.storage_and_layout();
    let Storage::Cuda(ids_cuda) = &*ids_storage else {
        candle_core::bail!("cuTile FP8 MoE routing tensors must be CUDA tensors")
    };
    let (sids, eids, ntpp, em) = moe_align(
        ids_cuda.as_cuda_slice::<u32>()?,
        num_tokens,
        weights.num_experts,
        top_k,
        cfg.bm,
        dev,
    )?;
    let topk_weights = routing_slice::<f32>(topk_weights)?;
    let (tw_storage, _) = topk_weights.storage_and_layout();
    let Storage::Cuda(tw_cuda) = &*tw_storage else {
        candle_core::bail!("cuTile FP8 MoE routing tensors must be CUDA tensors")
    };
    let tw_slice = tw_cuda.as_cuda_slice::<f32>()?;

    let token_rows = num_tokens.div_ceil(FP8_MOE_GROUP) * FP8_MOE_GROUP;
    let (a1, a1_scales) = quantize_activation_padded(&x.contiguous()?, token_rows)?;
    let gate_up = grouped_gemm_fp8(
        &a1,
        &a1_scales,
        &weights.gate_up,
        &weights.gate_up_scales,
        &sids,
        &eids,
        &ntpp,
        None,
        em,
        num_valid,
        top_k,
        false,
        cfg,
        dev,
        false,
    )?;
    let valid_rows = num_valid.div_ceil(FP8_MOE_GROUP) * FP8_MOE_GROUP;
    let (a2, a2_scales) =
        fused_split_glu_quantized_bf16(&gate_up, inter, FP8_MOE_GROUP, valid_rows, activation)?;
    let down = grouped_gemm_fp8(
        &a2,
        &a2_scales,
        &weights.down,
        &weights.down_scales,
        &sids,
        &eids,
        &ntpp,
        Some(tw_slice),
        em,
        num_valid,
        1,
        true,
        cfg,
        dev,
        false,
    )?;
    moe_sum_bf16(&down, num_tokens, top_k, dev)
}

/// `a` [rows, K] E4M3 whose storage spans `a_scales.dim(1)` rows, `a_scales` [K/128, rows_padded],
/// `b` [E, N, K] E4M3, `b_scales` [E, N/128, K/128] -> [num_valid_tokens, N] BF16.
fn grouped_gemm_fp8(
    a: &Tensor,
    a_scales: &Tensor,
    b: &Tensor,
    b_scales: &Tensor,
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
    compile_only: bool,
) -> Result<Tensor> {
    let (_e, n_size, k_size) = b.dims3()?;
    let (a_rows, a_k) = a.dims2()?;
    let (scale_groups, scale_stride) = a_scales.dims2()?;
    if a.dtype() != DType::F8E4M3 || b.dtype() != DType::F8E4M3 {
        candle_core::bail!("cuTile FP8 MoE GEMM needs E4M3 operands")
    }
    if a_k != k_size
        || !k_size.is_multiple_of(FP8_MOE_GROUP)
        || !n_size.is_multiple_of(cfg.bn as usize)
        || scale_groups != k_size / FP8_MOE_GROUP
        || scale_stride < a_rows
        || cfg.bk as usize != FP8_MOE_GROUP
        || !a.is_contiguous()
    {
        candle_core::bail!(
            "cuTile FP8 MoE GEMM got unsupported shapes rows={a_rows} k={k_size} n={n_size}"
        )
    }
    let mut out = unsafe { dev.alloc::<bf16>(num_valid_tokens * n_size)? };
    let stream = dev.cuda_stream();
    let (a_storage, a_layout) = a.storage_and_layout();
    let (as_storage, as_layout) = a_scales.storage_and_layout();
    let (b_storage, b_layout) = b.storage_and_layout();
    let (bs_storage, bs_layout) = b_scales.storage_and_layout();
    let (
        Storage::Cuda(a_cuda),
        Storage::Cuda(as_cuda),
        Storage::Cuda(b_cuda),
        Storage::Cuda(bs_cuda),
    ) = (&*a_storage, &*as_storage, &*b_storage, &*bs_storage)
    else {
        candle_core::bail!("cuTile FP8 MoE GEMM operands must be CUDA tensors")
    };
    let a_slice = a_cuda.as_cuda_slice::<F8E4M3>()?;
    if a_layout.start_offset() + scale_stride * k_size > a_slice.len() {
        candle_core::bail!("cuTile FP8 MoE GEMM activation storage is short of its padded rows")
    }
    let (a_addr, _a_guard) = slice_ptr_on_stream(a_slice, a_layout.start_offset(), &stream);
    let (as_addr, _as_guard) = slice_ptr_on_stream(
        as_cuda.as_cuda_slice::<f32>()?,
        as_layout.start_offset(),
        &stream,
    );
    let (b_addr, _b_guard) = slice_ptr_on_stream(
        b_cuda.as_cuda_slice::<F8E4M3>()?,
        b_layout.start_offset(),
        &stream,
    );
    let (bs_addr, _bs_guard) = slice_ptr_on_stream(
        bs_cuda.as_cuda_slice::<f32>()?,
        bs_layout.start_offset(),
        &stream,
    );
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
    let splits = cfg.split_k.max(1) as usize;
    if !(k_size / cfg.bk as usize).is_multiple_of(splits) {
        candle_core::bail!(
            "cuTile FP8 MoE split_k {splits} does not divide K={k_size} in {} groups",
            cfg.bk
        )
    }
    let numel = num_valid_tokens * n_size;
    let mut partial = if splits > 1 {
        Some(unsafe { dev.alloc::<f32>(splits * numel)? })
    } else {
        None
    };
    let partial_guard;
    let partial_addr = match partial.as_mut() {
        Some(buf) => {
            let (addr, guard) = slice_ptr_mut_on_stream(buf, 0, &stream);
            partial_guard = Some(guard);
            addr
        }
        None => {
            partial_guard = None;
            0
        }
    };
    let num_pid_m = em.div_ceil(cfg.bm as usize);
    let num_pid_n = n_size / cfg.bn as usize;
    let grid_x = (num_pid_m * num_pid_n * splits) as u32;
    let generics = vec![
        cfg.bm.to_string(),
        cfg.bn.to_string(),
        cfg.bk.to_string(),
        cfg.group_m.to_string(),
        (top_k as i32).to_string(),
        (if mul_routed_weight { 1 } else { 0 }).to_string(),
        (splits as i32).to_string(),
        cfg.latency.to_string(),
    ];
    let cutile_stream = context::stream(dev);
    let launcher = unsafe {
        fused_moe_fp8::fused_moe_fp8_kernel(
            DevicePointer::<bf16>::from_cu_deviceptr(out_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(partial_addr as CUdeviceptr),
            DevicePointer::<f8e4m3fn>::from_cu_deviceptr(a_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(as_addr as CUdeviceptr),
            DevicePointer::<f8e4m3fn>::from_cu_deviceptr(b_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(bs_addr as CUdeviceptr),
            DevicePointer::<i32>::from_cu_deviceptr(sids_addr as CUdeviceptr),
            DevicePointer::<i32>::from_cu_deviceptr(eids_addr as CUdeviceptr),
            DevicePointer::<i32>::from_cu_deviceptr(ntpp_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(tw_addr as CUdeviceptr),
            n_size as i32,
            k_size as i32,
            em as i32,
            num_valid_tokens as i32,
            scale_stride as i32,
        )
    }
    .generics(generics)
    .grid((grid_x, 1, 1))
    .compile_options(cfg.compile_options());
    if compile_only {
        catch_cutile_panic("fused FP8 MoE kernel compile", || {
            launcher.compile_on(&cutile_stream).map_err(|e| {
                candle_core::Error::Msg(format!("cutile fused_moe_fp8 compile: {e:?}"))
            })
        })?;
    } else {
        catch_cutile_panic("fused FP8 MoE kernel execute", || unsafe {
            launcher
                .async_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile fused_moe_fp8 launch: {e:?}")))
        })?;
    }
    drop((out_guard, tw_guard, partial_guard));
    if let Some(partial) = &partial {
        reduce_split_k(partial, &mut out, splits as i32, numel, dev, compile_only)?;
    }
    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((
        Storage::Cuda(storage),
        (num_valid_tokens, n_size),
    )))
}

static FP8_MOE_SHAPES: OnceLock<Mutex<Vec<Vec<CutileFp8MoeWeights>>>> = OnceLock::new();

/// Register a model's FP8 experts so warmup tunes and compiles the kernel keys the forward will
/// launch. Up to `TUNE_WEIGHT_SETS` layers of the same shape are kept as weight handles.
pub fn register_moe_fp8_shape(weights: &CutileFp8MoeWeights) {
    let mut shapes = FP8_MOE_SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    let key = weights.shape_key();
    if let Some(sets) = shapes.iter_mut().find(|sets| sets[0].shape_key() == key) {
        if sets.len() < TUNE_WEIGHT_SETS {
            sets.push(weights.clone());
        }
        return;
    }
    shapes.push(vec![weights.clone()]);
}

fn warmup_shape(dev: &CudaDevice, weights: &CutileFp8MoeWeights, m: usize) -> Result<()> {
    let device = Device::Cuda(dev.clone());
    let top_k = weights.top_k;
    let num_valid = m * top_k;
    let cfg = fp8_config(m, weights.shape_key());
    let mut topk_ids = unsafe { dev.alloc::<u32>(num_valid)? };
    dev.memcpy_htod(&vec![0u32; num_valid], &mut topk_ids)?;
    let (sids, eids, ntpp, em) = moe_align(&topk_ids, m, weights.num_experts, top_k, cfg.bm, dev)?;
    let token_rows = m.div_ceil(FP8_MOE_GROUP) * FP8_MOE_GROUP;
    let a1 = Tensor::zeros((token_rows, weights.hidden()), DType::F8E4M3, &device)?;
    let a1_scales = Tensor::zeros(
        (weights.hidden() / FP8_MOE_GROUP, token_rows),
        DType::F32,
        &device,
    )?;
    grouped_gemm_fp8(
        &a1,
        &a1_scales,
        &weights.gate_up,
        &weights.gate_up_scales,
        &sids,
        &eids,
        &ntpp,
        None,
        em,
        num_valid,
        top_k,
        false,
        cfg,
        dev,
        true,
    )?;
    let valid_rows = num_valid.div_ceil(FP8_MOE_GROUP) * FP8_MOE_GROUP;
    let a2 = Tensor::zeros((valid_rows, weights.inter()), DType::F8E4M3, &device)?;
    let a2_scales = Tensor::zeros(
        (weights.inter() / FP8_MOE_GROUP, valid_rows),
        DType::F32,
        &device,
    )?;
    let mut tw = unsafe { dev.alloc::<f32>(num_valid)? };
    dev.memcpy_htod(&vec![0f32; num_valid], &mut tw)?;
    grouped_gemm_fp8(
        &a2,
        &a2_scales,
        &weights.down,
        &weights.down_scales,
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
        true,
    )?;
    Ok(())
}

pub struct FusedMoeFp8Kernel;
pub static FUSED_MOE_FP8: FusedMoeFp8Kernel = FusedMoeFp8Kernel;

impl CutileKernel for FusedMoeFp8Kernel {
    fn warm(&self, dev: &CudaDevice) -> Result<()> {
        let shapes: Vec<Vec<CutileFp8MoeWeights>> = FP8_MOE_SHAPES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap()
            .clone();
        if shapes.is_empty() {
            return Ok(());
        }
        let mode = TuneMode::from_env();
        let buckets = fp8_buckets();
        for sets in &shapes {
            let key = sets[0].shape_key();
            let request = TuneRequest {
                kernel: TUNE_KERNEL,
                source_hash: fused_moe_fp8::_SOURCE_HASH,
                shape: key.to_string(),
                buckets: &buckets,
                space: &|bucket| fp8_space(bucket, key),
            };
            let mut tuner = Fp8Tuner::new(dev, sets);
            let tuned = tune(dev, mode, &request, |m, config| {
                let cfg = MoeTileConfig::from_config(config)
                    .ok_or_else(|| candle_core::Error::Msg("config outside the space".into()))?;
                tuner.prepare(m, cfg)
            });
            TUNED.set(key, &tuned, MoeTileConfig::from_config);
        }
        let plan: Vec<(CutileFp8MoeWeights, Vec<usize>)> = shapes
            .into_iter()
            .map(|sets| {
                let w = sets[0].clone();
                let key = w.shape_key();
                let ms = warmup_token_counts_for(w.num_experts, w.top_k, |m| fp8_config(m, key));
                (w, ms)
            })
            .collect();
        let total: usize = plan.iter().map(|(_, ms)| ms.len()).sum();
        tracing::info!("Warming {total} cuTile FP8 MoE kernels.");
        for (weights, ms) in &plan {
            for &m in ms {
                if let Err(err) = warmup_shape(dev, weights, m) {
                    tracing::warn!(
                        "cuTile FP8 MoE warmup failed (hidden={} inter={} m={m}): {err}",
                        weights.hidden(),
                        weights.inter()
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(all(test, has_blockwise_fp8_kernels))]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::{cutile_fused_moe_fp8, CutileFp8MoeWeights, FP8_MOE_GROUP};
    use super::{Bucket, MoeTileConfig};
    use crate::blockwise_fp8::ops;
    use crate::cutile::tune::{Source, Tuned};
    use crate::GluActivationType;

    fn patterned(len: usize, seed: usize, amplitude: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index * 7919 + seed * 104729) % 2001) as f32 / 1000.0 - 1.0)
            .map(|value| value * amplitude + offset)
            .collect()
    }

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    // Times the two grouped GEMMs on real expert shapes per tile config; set CUTILE_MOE_TUNE to
    // "tokens,hidden,inter,experts,topk" to override the Qwen3-30B-A3B defaults, CUTILE_MOE_TUNE_HOT
    // to route every token to the same experts, CUTILE_MOE_TUNE_LAYERS to rotate that many distinct
    // weight sets so decode-sized runs read cold from HBM, and CUTILE_MOE_TUNE_ZEROS for all-zero
    // activations.
    #[test]
    #[ignore = "benchmark; requires a CUDA device with cuTile support"]
    fn cutile_fused_moe_fp8_tile_sweep() -> Result<()> {
        use super::{Fp8Tuner, MoeTileConfig, TuneRouting};
        use crate::cutile::tune::bench_ms;

        let spec: Vec<usize> = std::env::var("CUTILE_MOE_TUNE")
            .ok()
            .map(|v| v.split(',').map(|x| x.parse().unwrap()).collect())
            .unwrap_or_else(|| vec![4096, 2048, 768, 128, 8]);
        let (tokens, hidden, inter, experts, top_k) = (spec[0], spec[1], spec[2], spec[3], spec[4]);
        let routing = if std::env::var("CUTILE_MOE_TUNE_HOT").is_ok() {
            TuneRouting::Hot
        } else {
            TuneRouting::Spread
        };
        let layers: usize = std::env::var("CUTILE_MOE_TUNE_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
            .max(1);
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let sets = (0..layers)
            .map(|_| {
                let gate_up = Tensor::zeros((experts, 2 * inter, hidden), DType::F8E4M3, &dev)?;
                let gate_up_scales = Tensor::ones(
                    (experts, 2 * inter / FP8_MOE_GROUP, hidden / FP8_MOE_GROUP),
                    DType::F32,
                    &dev,
                )?;
                let down = Tensor::zeros((experts, hidden, inter), DType::F8E4M3, &dev)?;
                let down_scales = Tensor::ones(
                    (experts, hidden / FP8_MOE_GROUP, inter / FP8_MOE_GROUP),
                    DType::F32,
                    &dev,
                )?;
                CutileFp8MoeWeights::new(gate_up, gate_up_scales, down, down_scales, experts, top_k)
            })
            .collect::<Result<Vec<_>>>()?;
        let flops = 2.0 * (tokens * top_k) as f64 * (hidden as f64) * (3 * inter) as f64;
        let mut tuner = Fp8Tuner::new(cuda, &sets).with_routing(routing);
        if std::env::var("CUTILE_MOE_TUNE_ZEROS").is_ok() {
            tuner = tuner.with_zero_inputs();
        }
        for (bm, bn, group_m) in [
            (128, 128, 1),
            (128, 64, 8),
            (64, 128, 8),
            (64, 64, 8),
            (64, 64, 16),
            (64, 64, 32),
            (32, 64, 8),
            (32, 128, 1),
            (32, 128, 8),
            (32, 128, 16),
            (16, 64, 8),
            (16, 64, 16),
            (16, 128, 1),
            (16, 128, 8),
            (16, 128, 16),
            (16, 128, 32),
        ] {
            let cfg = MoeTileConfig::tiles(bm, bn, FP8_MOE_GROUP as i32, group_m);
            let ms = bench_ms(cuda, tuner.prepare(tokens, cfg)?)?;
            eprintln!(
                "bm={bm} bn={bn} group_m={group_m}: {ms:.3} ms per layer, {:.1} TFLOPS",
                flops / ms / 1e9
            );
        }
        // knob sweep on the policy tiles for this token count
        let base = super::fp8_policy(tokens);
        let k_groups = [hidden, inter].map(|k| k / FP8_MOE_GROUP);
        let mut knobs: Vec<MoeTileConfig> = Vec::new();
        knobs.extend([0, 2, 4].map(|latency| MoeTileConfig { latency, ..base }));
        knobs.extend([0, 4, 8].map(|warps| MoeTileConfig { warps, ..base }));
        knobs.extend([0, 2].map(|occupancy| MoeTileConfig { occupancy, ..base }));
        knobs.extend(
            [1, 2, 4]
                .iter()
                .filter(|&&split_k| k_groups.iter().all(|g| g % split_k as usize == 0))
                .map(|&split_k| MoeTileConfig { split_k, ..base }),
        );
        for cfg in knobs {
            match tuner
                .prepare(tokens, cfg)
                .and_then(|prepared| bench_ms(cuda, prepared))
            {
                Ok(ms) => eprintln!(
                    "{cfg}: {ms:.3} ms per layer, {:.1} TFLOPS",
                    flops / ms / 1e9
                ),
                Err(err) => eprintln!("{cfg}: failed: {err}"),
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_fused_moe_fp8_matches_dequantized_reference() -> Result<()> {
        const E: usize = 4;
        const TOP_K: usize = 2;
        const H: usize = 256;
        const I: usize = 256;
        const T: usize = 37;
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let block = vec![FP8_MOE_GROUP, FP8_MOE_GROUP];
        let mut gate_up = Vec::new();
        let mut gate_up_scales = Vec::new();
        let mut down = Vec::new();
        let mut down_scales = Vec::new();
        let mut gate_up_ref = Vec::new();
        let mut down_ref = Vec::new();
        for e in 0..E {
            let gu = Tensor::from_vec(patterned(2 * I * H, 11 + e, 1.5, 0.05), (2 * I, H), &dev)?
                .to_dtype(DType::BF16)?;
            let (q, s) = ops::fp8_blockwise_quantize(&gu, block.clone())?;
            gate_up_ref.push(
                ops::fp8_blockwise_dequantize(&q, &s, block.clone(), DType::F32)?
                    .to_device(&Device::Cpu)?,
            );
            gate_up.push(q);
            gate_up_scales.push(s);
            let dn = Tensor::from_vec(patterned(H * I, 31 + e, 1.2, -0.03), (H, I), &dev)?
                .to_dtype(DType::BF16)?;
            let (q, s) = ops::fp8_blockwise_quantize(&dn, block.clone())?;
            down_ref.push(
                ops::fp8_blockwise_dequantize(&q, &s, block.clone(), DType::F32)?
                    .to_device(&Device::Cpu)?,
            );
            down.push(q);
            down_scales.push(s);
        }
        let weights = CutileFp8MoeWeights::new(
            Tensor::stack(&gate_up, 0)?,
            Tensor::stack(&gate_up_scales, 0)?,
            Tensor::stack(&down, 0)?,
            Tensor::stack(&down_scales, 0)?,
            E,
            TOP_K,
        )?;
        let x_host = patterned(T * H, 5, 2.0, 0.1);
        let x = Tensor::from_vec(x_host.clone(), (T, H), &dev)?.to_dtype(DType::BF16)?;
        let ids_host: Vec<u32> = (0..T * TOP_K)
            .map(|i| ((i * 7 + i / TOP_K) % E) as u32)
            .collect();
        let w_host: Vec<f32> = (0..T * TOP_K)
            .map(|i| 0.25 + (i % 3) as f32 * 0.3)
            .collect();
        let topk_ids = Tensor::from_vec(ids_host.clone(), (T, TOP_K), &dev)?;
        let topk_weights = Tensor::from_vec(w_host.clone(), (T, TOP_K), &dev)?;
        let run = || -> Result<Vec<Vec<f32>>> {
            cutile_fused_moe_fp8(
                &x,
                &weights,
                &topk_ids,
                &topk_weights,
                GluActivationType::Silu,
                cuda,
            )?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()
        };
        let policy = run()?;
        // force split-K with a pipelining hint through the tuned table, as the tuner would
        let forced = MoeTileConfig {
            split_k: 2,
            latency: 2,
            ..super::fp8_policy(T)
        };
        super::TUNED.set(
            weights.shape_key(),
            &[Tuned {
                bucket: Bucket {
                    upper: usize::MAX,
                    probe: T,
                },
                config: forced.to_config(),
                source: Source::Measured,
                ms: 0.0,
                policy_ms: 0.0,
            }],
            MoeTileConfig::from_config,
        );
        let split = run()?;
        super::TUNED.set(weights.shape_key(), &[], MoeTileConfig::from_config);
        let outs = [("policy", policy), ("split_k=2 latency=2", split)];

        let x_bf: Vec<f32> = Tensor::from_vec(x_host, (T, H), &Device::Cpu)?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let gate_up_ref: Vec<Vec<Vec<f32>>> = gate_up_ref
            .iter()
            .map(|t| t.to_vec2::<f32>())
            .collect::<Result<_>>()?;
        let down_ref: Vec<Vec<Vec<f32>>> = down_ref
            .iter()
            .map(|t| t.to_vec2::<f32>())
            .collect::<Result<_>>()?;
        for (label, out) in &outs {
            let mut max_err = 0f32;
            let mut max_ref = 0f32;
            for t in 0..T {
                let xt = &x_bf[t * H..(t + 1) * H];
                let mut acc = vec![0f32; H];
                for j in 0..TOP_K {
                    let e = ids_host[t * TOP_K + j] as usize;
                    let w = w_host[t * TOP_K + j];
                    let mut h = vec![0f32; I];
                    for i in 0..I {
                        let g: f32 = (0..H).map(|k| gate_up_ref[e][i][k] * xt[k]).sum();
                        let u: f32 = (0..H).map(|k| gate_up_ref[e][I + i][k] * xt[k]).sum();
                        h[i] = silu(g) * u;
                    }
                    for n in 0..H {
                        let d: f32 = (0..I).map(|i| down_ref[e][n][i] * h[i]).sum();
                        acc[n] += w * d;
                    }
                }
                for n in 0..H {
                    max_err = max_err.max((out[t][n] - acc[n]).abs());
                    max_ref = max_ref.max(acc[n].abs());
                }
            }
            assert!(
                max_err <= 0.05 * max_ref.max(1.0e-2),
                "fused FP8 MoE error {max_err} vs reference {max_ref} ({label})"
            );
        }
        Ok(())
    }
}
