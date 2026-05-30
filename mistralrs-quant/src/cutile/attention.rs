#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

use candle_core::backend::BackendStorage;
use candle_core::cuda::cudarc::driver::CudaSlice;
use candle_core::{
    CudaDevice, CudaStorage, DType, Device, DeviceLocation, Result, Storage, Tensor,
};
use cuda_async::device_buffer::DevicePointer;
use cuda_async::device_operation::DeviceOp;
use cuda_core::sys::CUdeviceptr;
use cutile::tile_kernel::TileKernel;
use half::bf16;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

use super::{context, device_supported};

const HEAD_SIZE_512: usize = 512;
const DECODE_TILE_SIZE: usize = 16;
const PREFILL_TILE_SIZE: usize = 32;
const NUM_DECODE_SEGMENTS: usize = 16;
const BLOCK_M: usize = 16;

static WARMED_ATTENTION_Q_GROUPS: OnceLock<Mutex<HashSet<(DeviceLocation, usize)>>> =
    OnceLock::new();
static REGISTERED_ATTENTION_Q_GROUPS: OnceLock<Mutex<HashSet<usize>>> = OnceLock::new();

pub fn cutile_paged_attention_supported(
    device: &candle_core::Device,
    dtype: DType,
    head_size: usize,
) -> bool {
    matches!(device, candle_core::Device::Cuda(dev) if device_supported(dev))
        && dtype == DType::BF16
        && head_size == HEAD_SIZE_512
}

pub fn register_cutile_attention_q_group(q_group: usize) {
    if q_group == 0 || q_group > BLOCK_M || BLOCK_M % q_group != 0 {
        return;
    }
    REGISTERED_ATTENTION_Q_GROUPS
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap()
        .insert(q_group);
}

#[cutile::module]
mod unified_attention {
    use cutile::core::*;

    fn load_scalar_i32(ptr: *mut u32, idx: i32) -> i32 {
        let p0: PointerTile<*mut u32, { [] }> = pointer_to_tile(ptr);
        let p1: PointerTile<*mut u32, { [1] }> = p0.reshape(const_shape![1]);
        let i: Tile<i32, { [1] }> = broadcast_scalar(idx, const_shape![1]);
        let p2: PointerTile<*mut u32, { [1] }> = p1.offset_tile(i);
        let (v_u, _): (Tile<u32, { [1] }>, Token) = load_ptr_tko(
            p2,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let v_i: Tile<i32, { [1] }> = bitcast(v_u);
        tile_to_scalar(v_i.reshape(const_shape![]))
    }

    fn query_scale() -> f32 {
        let two: Tile<f32, { [] }> = constant(2.0f32, const_shape![]);
        tile_to_scalar(log(two))
    }

    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_80 = (max_divisibility = 1,),
            sm_100 = (max_divisibility = 1,),
            sm_120 = (num_cta_in_cga = 1, max_divisibility = 1,),
        )
    )]
    pub unsafe fn paged_prefill_bf16<
        const HEAD_SIZE: i32,
        const BLOCK_M: i32,
        const TILE_SIZE: i32,
        const NUM_Q_PER_KV: i32,
        const BLOCK_Q: i32,
    >(
        out_ptr: *mut bf16,
        query_ptr: *mut bf16,
        key_cache_ptr: *mut bf16,
        value_cache_ptr: *mut bf16,
        block_tables_ptr: *mut u32,
        context_lens_ptr: *mut u32,
        q_indptr_ptr: *mut i32,
        softmax_scale: f32,
        num_q_heads: i32,
        num_seqs: i32,
        block_table_stride: i32,
        q_stride_token: i32,
        q_stride_head: i32,
        out_stride_token: i32,
        out_stride_head: i32,
        k_stride_block: i32,
        k_stride_head: i32,
        k_stride_token: i32,
        k_stride_dim: i32,
        v_stride_block: i32,
        v_stride_head: i32,
        v_stride_token: i32,
        v_stride_dim: i32,
        block_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_block_global_idx = pid.0;
        let kv_head_idx = pid.1;

        let mut left: i32 = 0;
        let mut right: i32 = num_seqs;
        while left < right {
            let mid = (left + right) / 2;
            let mid_q: i32 = {
                let p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(q_indptr_ptr);
                let p1: PointerTile<*mut i32, { [1] }> = p0.reshape(const_shape![1]);
                let mi: Tile<i32, { [1] }> = broadcast_scalar(mid, const_shape![1]);
                let p2: PointerTile<*mut i32, { [1] }> = p1.offset_tile(mi);
                let (v, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
                    p2,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    None,
                    None,
                    None,
                    Latency::<0>,
                );
                tile_to_scalar(v.reshape(const_shape![]))
            };
            let mid_val = mid_q / BLOCK_Q + mid;
            if mid_val <= q_block_global_idx {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        let seq_idx = left - 1;
        if seq_idx < 0 {
            return;
        }

        let cur_start: i32 = {
            let p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(q_indptr_ptr);
            let p1: PointerTile<*mut i32, { [1] }> = p0.reshape(const_shape![1]);
            let si: Tile<i32, { [1] }> = broadcast_scalar(seq_idx, const_shape![1]);
            let p2: PointerTile<*mut i32, { [1] }> = p1.offset_tile(si);
            let (v, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
                p2,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            tile_to_scalar(v.reshape(const_shape![]))
        };
        let cur_stop: i32 = {
            let p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(q_indptr_ptr);
            let p1: PointerTile<*mut i32, { [1] }> = p0.reshape(const_shape![1]);
            let si: Tile<i32, { [1] }> = broadcast_scalar(seq_idx + 1, const_shape![1]);
            let p2: PointerTile<*mut i32, { [1] }> = p1.offset_tile(si);
            let (v, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
                p2,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            tile_to_scalar(v.reshape(const_shape![]))
        };
        let cur_query_len = cur_stop - cur_start;
        let q_block_start_idx = cur_start / BLOCK_Q + seq_idx;
        let q_block_local_idx = q_block_global_idx - q_block_start_idx;
        if q_block_local_idx * BLOCK_Q >= cur_query_len {
            return;
        }

        let seq_len = load_scalar_i32(context_lens_ptr, seq_idx);
        let context_len = seq_len - cur_query_len;
        let max_seq_prefix_len = min(
            context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) / NUM_Q_PER_KV + 1,
            seq_len,
        );
        let loop_hi = ceil_div(max_seq_prefix_len, TILE_SIZE);
        let log2 = query_scale();
        let qk_scale = softmax_scale / log2;

        let offs_m: Tile<i32, { [BLOCK_M] }> = iota(const_shape![BLOCK_M]);
        let offs_d: Tile<i32, { [HEAD_SIZE] }> = iota(const_shape![HEAD_SIZE]);
        let query_pos: Tile<i32, { [BLOCK_M] }> =
            broadcast_scalar(q_block_local_idx * BLOCK_Q, const_shape![BLOCK_M])
                + offs_m / broadcast_scalar(NUM_Q_PER_KV, const_shape![BLOCK_M]);
        let query_head: Tile<i32, { [BLOCK_M] }> =
            broadcast_scalar(kv_head_idx * NUM_Q_PER_KV, const_shape![BLOCK_M]) + offs_m
                - (offs_m / broadcast_scalar(NUM_Q_PER_KV, const_shape![BLOCK_M]))
                    * broadcast_scalar(NUM_Q_PER_KV, const_shape![BLOCK_M]);
        let q_len_t: Tile<i32, { [BLOCK_M] }> =
            broadcast_scalar(cur_query_len, const_shape![BLOCK_M]);
        let q_head_t: Tile<i32, { [BLOCK_M] }> =
            broadcast_scalar(num_q_heads, const_shape![BLOCK_M]);
        let query_mask: Tile<bool, { [BLOCK_M] }> =
            lt_tile(query_pos, q_len_t) & lt_tile(query_head, q_head_t);

        let q_token: Tile<i32, { [BLOCK_M] }> =
            broadcast_scalar(cur_start, const_shape![BLOCK_M]) + query_pos;
        let q_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(query_ptr);
        let q_base1: PointerTile<*mut bf16, { [1, 1] }> = q_base0.reshape(const_shape![1, 1]);
        let q_base2: PointerTile<*mut bf16, { [BLOCK_M, HEAD_SIZE] }> =
            q_base1.broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
        let q_off: Tile<i32, { [BLOCK_M, HEAD_SIZE] }> = q_token
            .reshape(const_shape![BLOCK_M, 1])
            .broadcast(const_shape![BLOCK_M, HEAD_SIZE])
            * broadcast_scalar(q_stride_token, const_shape![BLOCK_M, HEAD_SIZE])
            + query_head
                .reshape(const_shape![BLOCK_M, 1])
                .broadcast(const_shape![BLOCK_M, HEAD_SIZE])
                * broadcast_scalar(q_stride_head, const_shape![BLOCK_M, HEAD_SIZE])
            + offs_d
                .reshape(const_shape![1, HEAD_SIZE])
                .broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
        let q_ptrs: PointerTile<*mut bf16, { [BLOCK_M, HEAD_SIZE] }> = q_base2.offset_tile(q_off);
        let q_mask2: Tile<bool, { [BLOCK_M, HEAD_SIZE] }> = query_mask
            .reshape(const_shape![BLOCK_M, 1])
            .broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
        let (q_load, _): (Tile<bf16, { [BLOCK_M, HEAD_SIZE] }>, Token) = load_ptr_tko(
            q_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(q_mask2),
            None,
            None,
            Latency::<0>,
        );
        let q_zero_f32: Tile<f32, { [BLOCK_M, HEAD_SIZE] }> =
            constant(0.0f32, const_shape![BLOCK_M, HEAD_SIZE]);
        let q_zero: Tile<bf16, { [BLOCK_M, HEAD_SIZE] }> = convert_tile(q_zero_f32);
        let q_load: Tile<bf16, { [BLOCK_M, HEAD_SIZE] }> = select(q_mask2, q_load, q_zero);

        let mut m_i: Tile<f32, { [BLOCK_M, 1] }> =
            constant(f32::NEG_INFINITY, const_shape![BLOCK_M, 1]);
        let mut l_i: Tile<f32, { [BLOCK_M, 1] }> = constant(0.0f32, const_shape![BLOCK_M, 1]);
        let mut acc: Tile<f32, { [BLOCK_M, HEAD_SIZE] }> =
            constant(0.0f32, const_shape![BLOCK_M, HEAD_SIZE]);

        let offs_t: Tile<i32, { [TILE_SIZE] }> = iota(const_shape![TILE_SIZE]);
        for j in 0i32..loop_hi {
            let seq_offset: Tile<i32, { [TILE_SIZE] }> =
                broadcast_scalar(j * TILE_SIZE, const_shape![TILE_SIZE]) + offs_t;
            let bt_idx: Tile<i32, { [TILE_SIZE] }> =
                broadcast_scalar(seq_idx, const_shape![TILE_SIZE])
                    * broadcast_scalar(block_table_stride, const_shape![TILE_SIZE])
                    + seq_offset / broadcast_scalar(block_size, const_shape![TILE_SIZE]);
            let bt_base0: PointerTile<*mut u32, { [] }> = pointer_to_tile(block_tables_ptr);
            let bt_base1: PointerTile<*mut u32, { [1] }> = bt_base0.reshape(const_shape![1]);
            let bt_base2: PointerTile<*mut u32, { [TILE_SIZE] }> =
                bt_base1.broadcast(const_shape![TILE_SIZE]);
            let bt_ptrs: PointerTile<*mut u32, { [TILE_SIZE] }> = bt_base2.offset_tile(bt_idx);
            let (phys_u, _): (Tile<u32, { [TILE_SIZE] }>, Token) = load_ptr_tko(
                bt_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let physical_block: Tile<i32, { [TILE_SIZE] }> = bitcast(phys_u);
            let block_offset: Tile<i32, { [TILE_SIZE] }> = seq_offset
                - (seq_offset / broadcast_scalar(block_size, const_shape![TILE_SIZE]))
                    * broadcast_scalar(block_size, const_shape![TILE_SIZE]);

            let k_off: Tile<i32, { [HEAD_SIZE, TILE_SIZE] }> = physical_block
                .reshape(const_shape![1, TILE_SIZE])
                .broadcast(const_shape![HEAD_SIZE, TILE_SIZE])
                * broadcast_scalar(k_stride_block, const_shape![HEAD_SIZE, TILE_SIZE])
                + broadcast_scalar(kv_head_idx, const_shape![HEAD_SIZE, TILE_SIZE])
                    * broadcast_scalar(k_stride_head, const_shape![HEAD_SIZE, TILE_SIZE])
                + block_offset
                    .reshape(const_shape![1, TILE_SIZE])
                    .broadcast(const_shape![HEAD_SIZE, TILE_SIZE])
                    * broadcast_scalar(k_stride_token, const_shape![HEAD_SIZE, TILE_SIZE])
                + offs_d
                    .reshape(const_shape![HEAD_SIZE, 1])
                    .broadcast(const_shape![HEAD_SIZE, TILE_SIZE])
                    * broadcast_scalar(k_stride_dim, const_shape![HEAD_SIZE, TILE_SIZE]);
            let v_off: Tile<i32, { [TILE_SIZE, HEAD_SIZE] }> = physical_block
                .reshape(const_shape![TILE_SIZE, 1])
                .broadcast(const_shape![TILE_SIZE, HEAD_SIZE])
                * broadcast_scalar(v_stride_block, const_shape![TILE_SIZE, HEAD_SIZE])
                + broadcast_scalar(kv_head_idx, const_shape![TILE_SIZE, HEAD_SIZE])
                    * broadcast_scalar(v_stride_head, const_shape![TILE_SIZE, HEAD_SIZE])
                + block_offset
                    .reshape(const_shape![TILE_SIZE, 1])
                    .broadcast(const_shape![TILE_SIZE, HEAD_SIZE])
                    * broadcast_scalar(v_stride_token, const_shape![TILE_SIZE, HEAD_SIZE])
                + offs_d
                    .reshape(const_shape![1, HEAD_SIZE])
                    .broadcast(const_shape![TILE_SIZE, HEAD_SIZE])
                    * broadcast_scalar(v_stride_dim, const_shape![TILE_SIZE, HEAD_SIZE]);
            let k_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(key_cache_ptr);
            let k_base1: PointerTile<*mut bf16, { [1, 1] }> = k_base0.reshape(const_shape![1, 1]);
            let k_base2: PointerTile<*mut bf16, { [HEAD_SIZE, TILE_SIZE] }> =
                k_base1.broadcast(const_shape![HEAD_SIZE, TILE_SIZE]);
            let v_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(value_cache_ptr);
            let v_base1: PointerTile<*mut bf16, { [1, 1] }> = v_base0.reshape(const_shape![1, 1]);
            let v_base2: PointerTile<*mut bf16, { [TILE_SIZE, HEAD_SIZE] }> =
                v_base1.broadcast(const_shape![TILE_SIZE, HEAD_SIZE]);

            let tile_valid_1d: Tile<bool, { [TILE_SIZE] }> = lt_tile(
                seq_offset,
                broadcast_scalar(max_seq_prefix_len, const_shape![TILE_SIZE]),
            );
            let k_mask: Tile<bool, { [HEAD_SIZE, TILE_SIZE] }> = tile_valid_1d
                .reshape(const_shape![1, TILE_SIZE])
                .broadcast(const_shape![HEAD_SIZE, TILE_SIZE]);
            let v_mask: Tile<bool, { [TILE_SIZE, HEAD_SIZE] }> = tile_valid_1d
                .reshape(const_shape![TILE_SIZE, 1])
                .broadcast(const_shape![TILE_SIZE, HEAD_SIZE]);
            let k_ptrs: PointerTile<*mut bf16, { [HEAD_SIZE, TILE_SIZE] }> =
                k_base2.offset_tile(k_off);
            let v_ptrs: PointerTile<*mut bf16, { [TILE_SIZE, HEAD_SIZE] }> =
                v_base2.offset_tile(v_off);
            let (k_load, _): (Tile<bf16, { [HEAD_SIZE, TILE_SIZE] }>, Token) = load_ptr_tko(
                k_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(k_mask),
                None,
                None,
                Latency::<0>,
            );
            let (v_load, _): (Tile<bf16, { [TILE_SIZE, HEAD_SIZE] }>, Token) = load_ptr_tko(
                v_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(v_mask),
                None,
                None,
                Latency::<0>,
            );
            let k_zero_f32: Tile<f32, { [HEAD_SIZE, TILE_SIZE] }> =
                constant(0.0f32, const_shape![HEAD_SIZE, TILE_SIZE]);
            let v_zero_f32: Tile<f32, { [TILE_SIZE, HEAD_SIZE] }> =
                constant(0.0f32, const_shape![TILE_SIZE, HEAD_SIZE]);
            let k_zero: Tile<bf16, { [HEAD_SIZE, TILE_SIZE] }> = convert_tile(k_zero_f32);
            let v_zero: Tile<bf16, { [TILE_SIZE, HEAD_SIZE] }> = convert_tile(v_zero_f32);
            let k_load: Tile<bf16, { [HEAD_SIZE, TILE_SIZE] }> = select(k_mask, k_load, k_zero);
            let v_load: Tile<bf16, { [TILE_SIZE, HEAD_SIZE] }> = select(v_mask, v_load, v_zero);

            let mut scores: Tile<f32, { [BLOCK_M, TILE_SIZE] }> =
                constant(0.0f32, const_shape![BLOCK_M, TILE_SIZE]);
            scores = mmaf(q_load, k_load, scores)
                * broadcast_scalar(qk_scale, const_shape![BLOCK_M, TILE_SIZE]);
            let query_abs: Tile<i32, { [BLOCK_M] }> =
                broadcast_scalar(context_len, const_shape![BLOCK_M]) + query_pos;
            let causal_mask: Tile<bool, { [BLOCK_M, TILE_SIZE] }> = le_tile(
                seq_offset
                    .reshape(const_shape![1, TILE_SIZE])
                    .broadcast(const_shape![BLOCK_M, TILE_SIZE]),
                query_abs
                    .reshape(const_shape![BLOCK_M, 1])
                    .broadcast(const_shape![BLOCK_M, TILE_SIZE]),
            );
            let row_mask: Tile<bool, { [BLOCK_M, TILE_SIZE] }> = query_mask
                .reshape(const_shape![BLOCK_M, 1])
                .broadcast(const_shape![BLOCK_M, TILE_SIZE]);
            let col_mask: Tile<bool, { [BLOCK_M, TILE_SIZE] }> = tile_valid_1d
                .reshape(const_shape![1, TILE_SIZE])
                .broadcast(const_shape![BLOCK_M, TILE_SIZE]);
            let neg_inf: Tile<f32, { [BLOCK_M, TILE_SIZE] }> =
                constant(f32::NEG_INFINITY, const_shape![BLOCK_M, TILE_SIZE]);
            let scores = select(row_mask & col_mask & causal_mask, scores, neg_inf);
            let qk_max: Tile<f32, { [BLOCK_M] }> = reduce_max(scores, 1);
            let qk_max: Tile<f32, { [BLOCK_M, 1] }> = qk_max.reshape(const_shape![BLOCK_M, 1]);
            let m_ij: Tile<f32, { [BLOCK_M, 1] }> = max_tile(m_i, qk_max);
            let shifted = scores - m_ij.broadcast(const_shape![BLOCK_M, TILE_SIZE]);
            let p: Tile<f32, { [BLOCK_M, TILE_SIZE] }> = exp2(shifted, ftz::Disabled);
            let l_ij: Tile<f32, { [BLOCK_M] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BLOCK_M, 1] }> = l_ij.reshape(const_shape![BLOCK_M, 1]);
            let alpha: Tile<f32, { [BLOCK_M, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
            let p_bf16: Tile<bf16, { [BLOCK_M, TILE_SIZE] }> = convert_tile(p);
            acc = mmaf(p_bf16, v_load, acc);
            m_i = m_ij;
        }

        let acc = true_div(acc, l_i.broadcast(const_shape![BLOCK_M, HEAD_SIZE]));
        let out_bf16: Tile<bf16, { [BLOCK_M, HEAD_SIZE] }> = convert_tile(acc);
        let out_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
        let out_base1: PointerTile<*mut bf16, { [1, 1] }> = out_base0.reshape(const_shape![1, 1]);
        let out_base2: PointerTile<*mut bf16, { [BLOCK_M, HEAD_SIZE] }> =
            out_base1.broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
        let out_off: Tile<i32, { [BLOCK_M, HEAD_SIZE] }> = q_token
            .reshape(const_shape![BLOCK_M, 1])
            .broadcast(const_shape![BLOCK_M, HEAD_SIZE])
            * broadcast_scalar(out_stride_token, const_shape![BLOCK_M, HEAD_SIZE])
            + query_head
                .reshape(const_shape![BLOCK_M, 1])
                .broadcast(const_shape![BLOCK_M, HEAD_SIZE])
                * broadcast_scalar(out_stride_head, const_shape![BLOCK_M, HEAD_SIZE])
            + offs_d
                .reshape(const_shape![1, HEAD_SIZE])
                .broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
        let out_ptrs: PointerTile<*mut bf16, { [BLOCK_M, HEAD_SIZE] }> =
            out_base2.offset_tile(out_off);
        store_ptr_tko(
            out_ptrs,
            out_bf16,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(q_mask2),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_80 = (max_divisibility = 1,),
            sm_100 = (max_divisibility = 1,),
            sm_120 = (num_cta_in_cga = 1, max_divisibility = 1,),
        )
    )]
    pub unsafe fn paged_decode_segment_bf16<
        const HEAD_SIZE: i32,
        const BLOCK_M: i32,
        const TILE_SIZE: i32,
        const NUM_Q_PER_KV: i32,
        const NUM_SEGMENTS: i32,
    >(
        segm_out_ptr: *mut f32,
        segm_max_ptr: *mut f32,
        segm_sum_ptr: *mut f32,
        query_ptr: *mut bf16,
        key_cache_ptr: *mut bf16,
        value_cache_ptr: *mut bf16,
        block_tables_ptr: *mut u32,
        context_lens_ptr: *mut u32,
        softmax_scale: f32,
        num_q_heads: i32,
        block_table_stride: i32,
        q_stride_token: i32,
        q_stride_head: i32,
        k_stride_block: i32,
        k_stride_head: i32,
        k_stride_token: i32,
        k_stride_dim: i32,
        v_stride_block: i32,
        v_stride_head: i32,
        v_stride_token: i32,
        v_stride_dim: i32,
        block_size: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let kv_head_idx = pid.1;
        let seg_idx = pid.2;
        let seq_len = load_scalar_i32(context_lens_ptr, seq_idx);
        let tiles_per_segment = ceil_div(seq_len, NUM_SEGMENTS * TILE_SIZE);
        let loop_lo = seg_idx * tiles_per_segment;
        let loop_hi = min(
            (seg_idx + 1) * tiles_per_segment,
            ceil_div(seq_len, TILE_SIZE),
        );
        if loop_lo * TILE_SIZE >= seq_len {
            return;
        }

        let log2 = query_scale();
        let qk_scale = softmax_scale / log2;
        let offs_m: Tile<i32, { [BLOCK_M] }> = iota(const_shape![BLOCK_M]);
        let offs_d: Tile<i32, { [HEAD_SIZE] }> = iota(const_shape![HEAD_SIZE]);
        let query_pos: Tile<i32, { [BLOCK_M] }> =
            offs_m / broadcast_scalar(NUM_Q_PER_KV, const_shape![BLOCK_M]);
        let query_head: Tile<i32, { [BLOCK_M] }> =
            broadcast_scalar(kv_head_idx * NUM_Q_PER_KV, const_shape![BLOCK_M]) + offs_m
                - query_pos * broadcast_scalar(NUM_Q_PER_KV, const_shape![BLOCK_M]);
        let zero_m: Tile<i32, { [BLOCK_M] }> = broadcast_scalar(0, const_shape![BLOCK_M]);
        let query_mask: Tile<bool, { [BLOCK_M] }> = eq_tile(query_pos, zero_m)
            & lt_tile(
                query_head,
                broadcast_scalar(num_q_heads, const_shape![BLOCK_M]),
            );

        let q_token: Tile<i32, { [BLOCK_M] }> = broadcast_scalar(seq_idx, const_shape![BLOCK_M]);
        let q_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(query_ptr);
        let q_base1: PointerTile<*mut bf16, { [1, 1] }> = q_base0.reshape(const_shape![1, 1]);
        let q_base2: PointerTile<*mut bf16, { [BLOCK_M, HEAD_SIZE] }> =
            q_base1.broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
        let q_off: Tile<i32, { [BLOCK_M, HEAD_SIZE] }> = q_token
            .reshape(const_shape![BLOCK_M, 1])
            .broadcast(const_shape![BLOCK_M, HEAD_SIZE])
            * broadcast_scalar(q_stride_token, const_shape![BLOCK_M, HEAD_SIZE])
            + query_head
                .reshape(const_shape![BLOCK_M, 1])
                .broadcast(const_shape![BLOCK_M, HEAD_SIZE])
                * broadcast_scalar(q_stride_head, const_shape![BLOCK_M, HEAD_SIZE])
            + offs_d
                .reshape(const_shape![1, HEAD_SIZE])
                .broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
        let q_ptrs: PointerTile<*mut bf16, { [BLOCK_M, HEAD_SIZE] }> = q_base2.offset_tile(q_off);
        let q_mask2: Tile<bool, { [BLOCK_M, HEAD_SIZE] }> = query_mask
            .reshape(const_shape![BLOCK_M, 1])
            .broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
        let (q_load, _): (Tile<bf16, { [BLOCK_M, HEAD_SIZE] }>, Token) = load_ptr_tko(
            q_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(q_mask2),
            None,
            None,
            Latency::<0>,
        );
        let q_zero_f32: Tile<f32, { [BLOCK_M, HEAD_SIZE] }> =
            constant(0.0f32, const_shape![BLOCK_M, HEAD_SIZE]);
        let q_zero: Tile<bf16, { [BLOCK_M, HEAD_SIZE] }> = convert_tile(q_zero_f32);
        let q_load: Tile<bf16, { [BLOCK_M, HEAD_SIZE] }> = select(q_mask2, q_load, q_zero);

        let mut m_i: Tile<f32, { [BLOCK_M, 1] }> =
            constant(f32::NEG_INFINITY, const_shape![BLOCK_M, 1]);
        let mut l_i: Tile<f32, { [BLOCK_M, 1] }> = constant(0.0f32, const_shape![BLOCK_M, 1]);
        let mut acc: Tile<f32, { [BLOCK_M, HEAD_SIZE] }> =
            constant(0.0f32, const_shape![BLOCK_M, HEAD_SIZE]);
        let offs_t: Tile<i32, { [TILE_SIZE] }> = iota(const_shape![TILE_SIZE]);

        for j in loop_lo..loop_hi {
            let seq_offset: Tile<i32, { [TILE_SIZE] }> =
                broadcast_scalar(j * TILE_SIZE, const_shape![TILE_SIZE]) + offs_t;
            let bt_idx: Tile<i32, { [TILE_SIZE] }> =
                broadcast_scalar(seq_idx, const_shape![TILE_SIZE])
                    * broadcast_scalar(block_table_stride, const_shape![TILE_SIZE])
                    + seq_offset / broadcast_scalar(block_size, const_shape![TILE_SIZE]);
            let bt_base0: PointerTile<*mut u32, { [] }> = pointer_to_tile(block_tables_ptr);
            let bt_base1: PointerTile<*mut u32, { [1] }> = bt_base0.reshape(const_shape![1]);
            let bt_base2: PointerTile<*mut u32, { [TILE_SIZE] }> =
                bt_base1.broadcast(const_shape![TILE_SIZE]);
            let bt_ptrs: PointerTile<*mut u32, { [TILE_SIZE] }> = bt_base2.offset_tile(bt_idx);
            let (phys_u, _): (Tile<u32, { [TILE_SIZE] }>, Token) = load_ptr_tko(
                bt_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let physical_block: Tile<i32, { [TILE_SIZE] }> = bitcast(phys_u);
            let block_offset: Tile<i32, { [TILE_SIZE] }> = seq_offset
                - (seq_offset / broadcast_scalar(block_size, const_shape![TILE_SIZE]))
                    * broadcast_scalar(block_size, const_shape![TILE_SIZE]);
            let k_off: Tile<i32, { [HEAD_SIZE, TILE_SIZE] }> = physical_block
                .reshape(const_shape![1, TILE_SIZE])
                .broadcast(const_shape![HEAD_SIZE, TILE_SIZE])
                * broadcast_scalar(k_stride_block, const_shape![HEAD_SIZE, TILE_SIZE])
                + broadcast_scalar(kv_head_idx, const_shape![HEAD_SIZE, TILE_SIZE])
                    * broadcast_scalar(k_stride_head, const_shape![HEAD_SIZE, TILE_SIZE])
                + block_offset
                    .reshape(const_shape![1, TILE_SIZE])
                    .broadcast(const_shape![HEAD_SIZE, TILE_SIZE])
                    * broadcast_scalar(k_stride_token, const_shape![HEAD_SIZE, TILE_SIZE])
                + offs_d
                    .reshape(const_shape![HEAD_SIZE, 1])
                    .broadcast(const_shape![HEAD_SIZE, TILE_SIZE])
                    * broadcast_scalar(k_stride_dim, const_shape![HEAD_SIZE, TILE_SIZE]);
            let v_off: Tile<i32, { [TILE_SIZE, HEAD_SIZE] }> = physical_block
                .reshape(const_shape![TILE_SIZE, 1])
                .broadcast(const_shape![TILE_SIZE, HEAD_SIZE])
                * broadcast_scalar(v_stride_block, const_shape![TILE_SIZE, HEAD_SIZE])
                + broadcast_scalar(kv_head_idx, const_shape![TILE_SIZE, HEAD_SIZE])
                    * broadcast_scalar(v_stride_head, const_shape![TILE_SIZE, HEAD_SIZE])
                + block_offset
                    .reshape(const_shape![TILE_SIZE, 1])
                    .broadcast(const_shape![TILE_SIZE, HEAD_SIZE])
                    * broadcast_scalar(v_stride_token, const_shape![TILE_SIZE, HEAD_SIZE])
                + offs_d
                    .reshape(const_shape![1, HEAD_SIZE])
                    .broadcast(const_shape![TILE_SIZE, HEAD_SIZE])
                    * broadcast_scalar(v_stride_dim, const_shape![TILE_SIZE, HEAD_SIZE]);
            let k_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(key_cache_ptr);
            let k_base1: PointerTile<*mut bf16, { [1, 1] }> = k_base0.reshape(const_shape![1, 1]);
            let k_base2: PointerTile<*mut bf16, { [HEAD_SIZE, TILE_SIZE] }> =
                k_base1.broadcast(const_shape![HEAD_SIZE, TILE_SIZE]);
            let v_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(value_cache_ptr);
            let v_base1: PointerTile<*mut bf16, { [1, 1] }> = v_base0.reshape(const_shape![1, 1]);
            let v_base2: PointerTile<*mut bf16, { [TILE_SIZE, HEAD_SIZE] }> =
                v_base1.broadcast(const_shape![TILE_SIZE, HEAD_SIZE]);
            let tile_valid_1d: Tile<bool, { [TILE_SIZE] }> = lt_tile(
                seq_offset,
                broadcast_scalar(seq_len, const_shape![TILE_SIZE]),
            );
            let k_mask: Tile<bool, { [HEAD_SIZE, TILE_SIZE] }> = tile_valid_1d
                .reshape(const_shape![1, TILE_SIZE])
                .broadcast(const_shape![HEAD_SIZE, TILE_SIZE]);
            let v_mask: Tile<bool, { [TILE_SIZE, HEAD_SIZE] }> = tile_valid_1d
                .reshape(const_shape![TILE_SIZE, 1])
                .broadcast(const_shape![TILE_SIZE, HEAD_SIZE]);
            let k_ptrs: PointerTile<*mut bf16, { [HEAD_SIZE, TILE_SIZE] }> =
                k_base2.offset_tile(k_off);
            let v_ptrs: PointerTile<*mut bf16, { [TILE_SIZE, HEAD_SIZE] }> =
                v_base2.offset_tile(v_off);
            let (k_load, _): (Tile<bf16, { [HEAD_SIZE, TILE_SIZE] }>, Token) = load_ptr_tko(
                k_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(k_mask),
                None,
                None,
                Latency::<0>,
            );
            let (v_load, _): (Tile<bf16, { [TILE_SIZE, HEAD_SIZE] }>, Token) = load_ptr_tko(
                v_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(v_mask),
                None,
                None,
                Latency::<0>,
            );
            let k_zero_f32: Tile<f32, { [HEAD_SIZE, TILE_SIZE] }> =
                constant(0.0f32, const_shape![HEAD_SIZE, TILE_SIZE]);
            let v_zero_f32: Tile<f32, { [TILE_SIZE, HEAD_SIZE] }> =
                constant(0.0f32, const_shape![TILE_SIZE, HEAD_SIZE]);
            let k_zero: Tile<bf16, { [HEAD_SIZE, TILE_SIZE] }> = convert_tile(k_zero_f32);
            let v_zero: Tile<bf16, { [TILE_SIZE, HEAD_SIZE] }> = convert_tile(v_zero_f32);
            let k_load: Tile<bf16, { [HEAD_SIZE, TILE_SIZE] }> = select(k_mask, k_load, k_zero);
            let v_load: Tile<bf16, { [TILE_SIZE, HEAD_SIZE] }> = select(v_mask, v_load, v_zero);
            let mut scores: Tile<f32, { [BLOCK_M, TILE_SIZE] }> =
                constant(0.0f32, const_shape![BLOCK_M, TILE_SIZE]);
            scores = mmaf(q_load, k_load, scores)
                * broadcast_scalar(qk_scale, const_shape![BLOCK_M, TILE_SIZE]);
            let row_mask: Tile<bool, { [BLOCK_M, TILE_SIZE] }> = query_mask
                .reshape(const_shape![BLOCK_M, 1])
                .broadcast(const_shape![BLOCK_M, TILE_SIZE]);
            let col_mask: Tile<bool, { [BLOCK_M, TILE_SIZE] }> = tile_valid_1d
                .reshape(const_shape![1, TILE_SIZE])
                .broadcast(const_shape![BLOCK_M, TILE_SIZE]);
            let neg_inf: Tile<f32, { [BLOCK_M, TILE_SIZE] }> =
                constant(f32::NEG_INFINITY, const_shape![BLOCK_M, TILE_SIZE]);
            let scores = select(row_mask & col_mask, scores, neg_inf);
            let qk_max: Tile<f32, { [BLOCK_M] }> = reduce_max(scores, 1);
            let qk_max: Tile<f32, { [BLOCK_M, 1] }> = qk_max.reshape(const_shape![BLOCK_M, 1]);
            let m_ij: Tile<f32, { [BLOCK_M, 1] }> = max_tile(m_i, qk_max);
            let p: Tile<f32, { [BLOCK_M, TILE_SIZE] }> = exp2(
                scores - m_ij.broadcast(const_shape![BLOCK_M, TILE_SIZE]),
                ftz::Disabled,
            );
            let l_ij: Tile<f32, { [BLOCK_M] }> = reduce_sum(p, 1);
            let l_ij: Tile<f32, { [BLOCK_M, 1] }> = l_ij.reshape(const_shape![BLOCK_M, 1]);
            let alpha: Tile<f32, { [BLOCK_M, 1] }> = exp2(m_i - m_ij, ftz::Disabled);
            l_i = l_i * alpha + l_ij;
            acc = acc * alpha.broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
            let p_bf16: Tile<bf16, { [BLOCK_M, TILE_SIZE] }> = convert_tile(p);
            acc = mmaf(p_bf16, v_load, acc);
            m_i = m_ij;
        }

        let seg_base0: PointerTile<*mut f32, { [] }> = pointer_to_tile(segm_out_ptr);
        let seg_base1: PointerTile<*mut f32, { [1, 1] }> = seg_base0.reshape(const_shape![1, 1]);
        let seg_base2: PointerTile<*mut f32, { [BLOCK_M, HEAD_SIZE] }> =
            seg_base1.broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
        let seg_off: Tile<i32, { [BLOCK_M, HEAD_SIZE] }> = q_token
            .reshape(const_shape![BLOCK_M, 1])
            .broadcast(const_shape![BLOCK_M, HEAD_SIZE])
            * broadcast_scalar(
                num_q_heads * NUM_SEGMENTS * HEAD_SIZE,
                const_shape![BLOCK_M, HEAD_SIZE],
            )
            + query_head
                .reshape(const_shape![BLOCK_M, 1])
                .broadcast(const_shape![BLOCK_M, HEAD_SIZE])
                * broadcast_scalar(NUM_SEGMENTS * HEAD_SIZE, const_shape![BLOCK_M, HEAD_SIZE])
            + broadcast_scalar(seg_idx * HEAD_SIZE, const_shape![BLOCK_M, HEAD_SIZE])
            + offs_d
                .reshape(const_shape![1, HEAD_SIZE])
                .broadcast(const_shape![BLOCK_M, HEAD_SIZE]);
        let seg_ptrs: PointerTile<*mut f32, { [BLOCK_M, HEAD_SIZE] }> =
            seg_base2.offset_tile(seg_off);
        store_ptr_tko(
            seg_ptrs,
            acc,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(q_mask2),
            None,
            Latency::<0>,
        );

        let scalar_base: Tile<i32, { [BLOCK_M] }> = q_token
            * broadcast_scalar(num_q_heads * NUM_SEGMENTS, const_shape![BLOCK_M])
            + query_head * broadcast_scalar(NUM_SEGMENTS, const_shape![BLOCK_M])
            + broadcast_scalar(seg_idx, const_shape![BLOCK_M]);
        let max_base0: PointerTile<*mut f32, { [] }> = pointer_to_tile(segm_max_ptr);
        let max_base1: PointerTile<*mut f32, { [1] }> = max_base0.reshape(const_shape![1]);
        let max_base2: PointerTile<*mut f32, { [BLOCK_M] }> =
            max_base1.broadcast(const_shape![BLOCK_M]);
        let sum_base0: PointerTile<*mut f32, { [] }> = pointer_to_tile(segm_sum_ptr);
        let sum_base1: PointerTile<*mut f32, { [1] }> = sum_base0.reshape(const_shape![1]);
        let sum_base2: PointerTile<*mut f32, { [BLOCK_M] }> =
            sum_base1.broadcast(const_shape![BLOCK_M]);
        let max_ptrs: PointerTile<*mut f32, { [BLOCK_M] }> = max_base2.offset_tile(scalar_base);
        let sum_ptrs: PointerTile<*mut f32, { [BLOCK_M] }> = sum_base2.offset_tile(scalar_base);
        store_ptr_tko(
            max_ptrs,
            m_i.reshape(const_shape![BLOCK_M]),
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(query_mask),
            None,
            Latency::<0>,
        );
        store_ptr_tko(
            sum_ptrs,
            l_i.reshape(const_shape![BLOCK_M]),
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(query_mask),
            None,
            Latency::<0>,
        );
    }

    #[cutile::entry(
        unchecked_accesses = true,
        optimization_hints = (
            sm_80 = (max_divisibility = 1,),
            sm_100 = (max_divisibility = 1,),
            sm_120 = (num_cta_in_cga = 1, max_divisibility = 1,),
        )
    )]
    pub unsafe fn reduce_segments_bf16<const HEAD_SIZE: i32, const NUM_SEGMENTS: i32>(
        out_ptr: *mut bf16,
        segm_out_ptr: *mut f32,
        segm_max_ptr: *mut f32,
        segm_sum_ptr: *mut f32,
        context_lens_ptr: *mut u32,
        num_q_heads: i32,
        out_stride_token: i32,
        out_stride_head: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let token_idx = pid.0;
        let q_head_idx = pid.1;
        let seq_len = load_scalar_i32(context_lens_ptr, token_idx);
        let tiles_per_segment = ceil_div(seq_len, NUM_SEGMENTS * 16);
        let active_segments = ceil_div(seq_len, tiles_per_segment * 16);
        let offs_s: Tile<i32, { [NUM_SEGMENTS] }> = iota(const_shape![NUM_SEGMENTS]);
        let seg_mask: Tile<bool, { [NUM_SEGMENTS] }> = lt_tile(
            offs_s,
            broadcast_scalar(active_segments, const_shape![NUM_SEGMENTS]),
        );

        let scalar_base: Tile<i32, { [NUM_SEGMENTS] }> =
            broadcast_scalar(
                token_idx * num_q_heads * NUM_SEGMENTS,
                const_shape![NUM_SEGMENTS],
            ) + broadcast_scalar(q_head_idx * NUM_SEGMENTS, const_shape![NUM_SEGMENTS])
                + offs_s;
        let max_base0: PointerTile<*mut f32, { [] }> = pointer_to_tile(segm_max_ptr);
        let max_base1: PointerTile<*mut f32, { [1] }> = max_base0.reshape(const_shape![1]);
        let max_base2: PointerTile<*mut f32, { [NUM_SEGMENTS] }> =
            max_base1.broadcast(const_shape![NUM_SEGMENTS]);
        let max_ptrs: PointerTile<*mut f32, { [NUM_SEGMENTS] }> =
            max_base2.offset_tile(scalar_base);
        let (seg_max, _): (Tile<f32, { [NUM_SEGMENTS] }>, Token) = load_ptr_tko(
            max_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(seg_mask),
            None,
            None,
            Latency::<0>,
        );
        let seg_neg_inf: Tile<f32, { [NUM_SEGMENTS] }> =
            constant(f32::NEG_INFINITY, const_shape![NUM_SEGMENTS]);
        let seg_max: Tile<f32, { [NUM_SEGMENTS] }> = select(seg_mask, seg_max, seg_neg_inf);
        let overall_max_s: Tile<f32, { [] }> = reduce_max(seg_max, 0);
        let overall_max: Tile<f32, { [1] }> = overall_max_s.reshape(const_shape![1]);
        let sum_base0: PointerTile<*mut f32, { [] }> = pointer_to_tile(segm_sum_ptr);
        let sum_base1: PointerTile<*mut f32, { [1] }> = sum_base0.reshape(const_shape![1]);
        let sum_base2: PointerTile<*mut f32, { [NUM_SEGMENTS] }> =
            sum_base1.broadcast(const_shape![NUM_SEGMENTS]);
        let sum_ptrs: PointerTile<*mut f32, { [NUM_SEGMENTS] }> =
            sum_base2.offset_tile(scalar_base);
        let (seg_sum, _): (Tile<f32, { [NUM_SEGMENTS] }>, Token) = load_ptr_tko(
            sum_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(seg_mask),
            None,
            None,
            Latency::<0>,
        );
        let seg_zero: Tile<f32, { [NUM_SEGMENTS] }> = constant(0.0f32, const_shape![NUM_SEGMENTS]);
        let seg_sum: Tile<f32, { [NUM_SEGMENTS] }> = select(seg_mask, seg_sum, seg_zero);
        let exp_weights: Tile<f32, { [NUM_SEGMENTS] }> = seg_sum
            * exp2(
                seg_max - overall_max.broadcast(const_shape![NUM_SEGMENTS]),
                ftz::Disabled,
            );
        let denom_s: Tile<f32, { [] }> = reduce_sum(exp_weights, 0);
        let denom: Tile<f32, { [1] }> = denom_s.reshape(const_shape![1]);
        let acc_weights: Tile<f32, { [NUM_SEGMENTS] }> = exp2(
            seg_max - overall_max.broadcast(const_shape![NUM_SEGMENTS]),
            ftz::Disabled,
        );

        let offs_d: Tile<i32, { [512] }> = iota(const_shape![512]);
        let seg_off: Tile<i32, { [NUM_SEGMENTS, 512] }> = broadcast_scalar(
            token_idx * num_q_heads * NUM_SEGMENTS * 512,
            const_shape![NUM_SEGMENTS, 512],
        ) + broadcast_scalar(
            q_head_idx * NUM_SEGMENTS * 512,
            const_shape![NUM_SEGMENTS, 512],
        ) + offs_s
            .reshape(const_shape![NUM_SEGMENTS, 1])
            .broadcast(const_shape![NUM_SEGMENTS, 512])
            * broadcast_scalar(512, const_shape![NUM_SEGMENTS, 512])
            + offs_d
                .reshape(const_shape![1, 512])
                .broadcast(const_shape![NUM_SEGMENTS, 512]);
        let seg_base0: PointerTile<*mut f32, { [] }> = pointer_to_tile(segm_out_ptr);
        let seg_base1: PointerTile<*mut f32, { [1, 1] }> = seg_base0.reshape(const_shape![1, 1]);
        let seg_base2: PointerTile<*mut f32, { [NUM_SEGMENTS, 512] }> =
            seg_base1.broadcast(const_shape![NUM_SEGMENTS, 512]);
        let seg_ptrs: PointerTile<*mut f32, { [NUM_SEGMENTS, 512] }> =
            seg_base2.offset_tile(seg_off);
        let seg_mask2: Tile<bool, { [NUM_SEGMENTS, 512] }> = seg_mask
            .reshape(const_shape![NUM_SEGMENTS, 1])
            .broadcast(const_shape![NUM_SEGMENTS, 512]);
        let (partials, _): (Tile<f32, { [NUM_SEGMENTS, 512] }>, Token) = load_ptr_tko(
            seg_ptrs,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(seg_mask2),
            None,
            None,
            Latency::<0>,
        );
        let partial_zero: Tile<f32, { [NUM_SEGMENTS, 512] }> =
            constant(0.0f32, const_shape![NUM_SEGMENTS, 512]);
        let partials: Tile<f32, { [NUM_SEGMENTS, 512] }> =
            select(seg_mask2, partials, partial_zero);
        let w2: Tile<f32, { [NUM_SEGMENTS, 512] }> = acc_weights
            .reshape(const_shape![NUM_SEGMENTS, 1])
            .broadcast(const_shape![NUM_SEGMENTS, 512]);
        let acc_sum: Tile<f32, { [512] }> = reduce_sum(partials * w2, 0);
        let denom_h: Tile<f32, { [512] }> = denom.broadcast(const_shape![512]);
        let out_f32: Tile<f32, { [512] }> = true_div(acc_sum, denom_h);
        let out_bf16: Tile<bf16, { [512] }> = convert_tile(out_f32);
        let out_off: Tile<i32, { [512] }> = broadcast_scalar(
            token_idx * out_stride_token + q_head_idx * out_stride_head,
            const_shape![512],
        ) + offs_d;
        let out_base0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
        let out_base1: PointerTile<*mut bf16, { [1] }> = out_base0.reshape(const_shape![1]);
        let out_base2: PointerTile<*mut bf16, { [512] }> = out_base1.broadcast(const_shape![512]);
        let out_ptrs: PointerTile<*mut bf16, { [512] }> = out_base2.offset_tile(out_off);
        store_ptr_tko(
            out_ptrs,
            out_bf16,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            Latency::<0>,
        );
    }
}

fn cuda_storage_layout(tensor: &Tensor) -> Result<(CudaStorage, candle_core::Layout)> {
    let (storage, layout) = tensor.storage_and_layout();
    let storage = match &*storage {
        Storage::Cuda(storage) => storage.try_clone(layout)?,
        _ => candle_core::bail!("expected cuda tensor"),
    };
    Ok((storage, layout.clone()))
}

fn launch_prefill(
    out: &mut CudaSlice<bf16>,
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    q_indptr: &Tensor,
    softmax_scale: f32,
    num_q_heads: usize,
    num_kv_heads: usize,
    total_q_blocks: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (q_storage, q_l) = cuda_storage_layout(query)?;
    let (kc_storage, kc_l) = cuda_storage_layout(key_cache)?;
    let (vc_storage, vc_l) = cuda_storage_layout(value_cache)?;
    let (bt_storage, bt_l) = cuda_storage_layout(block_tables)?;
    let (cl_storage, cl_l) = cuda_storage_layout(context_lens)?;
    let (qi_storage, qi_l) = cuda_storage_layout(q_indptr)?;
    let q_s = &q_storage;
    let kc_s = &kc_storage;
    let vc_s = &vc_storage;
    let bt_s = &bt_storage;
    let cl_s = &cl_storage;
    let qi_s = &qi_storage;
    let stream = dev.cuda_stream();

    let q = q_s.as_cuda_slice::<bf16>()?;
    let kc = kc_s.as_cuda_slice::<bf16>()?;
    let vc = vc_s.as_cuda_slice::<bf16>()?;
    let bt = bt_s.as_cuda_slice::<u32>()?;
    let cl = cl_s.as_cuda_slice::<u32>()?;
    let qi = qi_s.as_cuda_slice::<i32>()?;
    let (q_addr, _q_guard) = slice_ptr_on_stream(q, q_l.start_offset(), &stream);
    let (kc_addr, _kc_guard) = slice_ptr_on_stream(kc, kc_l.start_offset(), &stream);
    let (vc_addr, _vc_guard) = slice_ptr_on_stream(vc, vc_l.start_offset(), &stream);
    let (bt_addr, _bt_guard) = slice_ptr_on_stream(bt, bt_l.start_offset(), &stream);
    let (cl_addr, _cl_guard) = slice_ptr_on_stream(cl, cl_l.start_offset(), &stream);
    let (qi_addr, _qi_guard) = slice_ptr_on_stream(qi, qi_l.start_offset(), &stream);
    let (out_addr, out_guard) = slice_ptr_mut_on_stream(out, 0, &stream);

    let (_, _, block_size, _) = key_cache.dims4()?;
    let q_group = num_q_heads / num_kv_heads;
    let block_q = BLOCK_M / q_group;
    let generics = vec![
        HEAD_SIZE_512.to_string(),
        BLOCK_M.to_string(),
        PREFILL_TILE_SIZE.to_string(),
        q_group.to_string(),
        block_q.to_string(),
    ];
    let ctx = context::execution_context(dev);
    let launcher = unsafe {
        unified_attention::paged_prefill_bf16(
            DevicePointer::<bf16>::from_cu_deviceptr(out_addr as CUdeviceptr),
            DevicePointer::<bf16>::from_cu_deviceptr(q_addr as CUdeviceptr),
            DevicePointer::<bf16>::from_cu_deviceptr(kc_addr as CUdeviceptr),
            DevicePointer::<bf16>::from_cu_deviceptr(vc_addr as CUdeviceptr),
            DevicePointer::<u32>::from_cu_deviceptr(bt_addr as CUdeviceptr),
            DevicePointer::<u32>::from_cu_deviceptr(cl_addr as CUdeviceptr),
            DevicePointer::<i32>::from_cu_deviceptr(qi_addr as CUdeviceptr),
            softmax_scale,
            num_q_heads as i32,
            context_lens.dims1()? as i32,
            bt_l.stride()[0] as i32,
            q_l.stride()[0] as i32,
            q_l.stride()[1] as i32,
            (num_q_heads * HEAD_SIZE_512) as i32,
            HEAD_SIZE_512 as i32,
            kc_l.stride()[0] as i32,
            kc_l.stride()[1] as i32,
            kc_l.stride()[2] as i32,
            kc_l.stride()[3] as i32,
            vc_l.stride()[0] as i32,
            vc_l.stride()[1] as i32,
            vc_l.stride()[2] as i32,
            vc_l.stride()[3] as i32,
            block_size as i32,
        )
    }
    .generics(generics)
    .grid((total_q_blocks as u32, num_kv_heads as u32, 1));
    unsafe { launcher.execute(&ctx) }
        .map_err(|err| candle_core::Error::Msg(format!("cutile paged prefill launch: {err:?}")))?;
    drop(out_guard);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_decode_segments(
    segm_out: &mut CudaSlice<f32>,
    segm_max: &mut CudaSlice<f32>,
    segm_sum: &mut CudaSlice<f32>,
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    softmax_scale: f32,
    num_q_heads: usize,
    num_kv_heads: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (q_storage, q_l) = cuda_storage_layout(query)?;
    let (kc_storage, kc_l) = cuda_storage_layout(key_cache)?;
    let (vc_storage, vc_l) = cuda_storage_layout(value_cache)?;
    let (bt_storage, bt_l) = cuda_storage_layout(block_tables)?;
    let (cl_storage, cl_l) = cuda_storage_layout(context_lens)?;
    let q_s = &q_storage;
    let kc_s = &kc_storage;
    let vc_s = &vc_storage;
    let bt_s = &bt_storage;
    let cl_s = &cl_storage;
    let stream = dev.cuda_stream();
    let q = q_s.as_cuda_slice::<bf16>()?;
    let kc = kc_s.as_cuda_slice::<bf16>()?;
    let vc = vc_s.as_cuda_slice::<bf16>()?;
    let bt = bt_s.as_cuda_slice::<u32>()?;
    let cl = cl_s.as_cuda_slice::<u32>()?;
    let (q_addr, _q_guard) = slice_ptr_on_stream(q, q_l.start_offset(), &stream);
    let (kc_addr, _kc_guard) = slice_ptr_on_stream(kc, kc_l.start_offset(), &stream);
    let (vc_addr, _vc_guard) = slice_ptr_on_stream(vc, vc_l.start_offset(), &stream);
    let (bt_addr, _bt_guard) = slice_ptr_on_stream(bt, bt_l.start_offset(), &stream);
    let (cl_addr, _cl_guard) = slice_ptr_on_stream(cl, cl_l.start_offset(), &stream);
    let (segm_out_addr, segm_out_guard) = slice_ptr_mut_on_stream(segm_out, 0, &stream);
    let (segm_max_addr, segm_max_guard) = slice_ptr_mut_on_stream(segm_max, 0, &stream);
    let (segm_sum_addr, segm_sum_guard) = slice_ptr_mut_on_stream(segm_sum, 0, &stream);
    let (_, _, block_size, _) = key_cache.dims4()?;
    let q_group = num_q_heads / num_kv_heads;
    let generics = vec![
        HEAD_SIZE_512.to_string(),
        BLOCK_M.to_string(),
        DECODE_TILE_SIZE.to_string(),
        q_group.to_string(),
        NUM_DECODE_SEGMENTS.to_string(),
    ];
    let ctx = context::execution_context(dev);
    let batch = query.dim(0)?;
    let launcher = unsafe {
        unified_attention::paged_decode_segment_bf16(
            DevicePointer::<f32>::from_cu_deviceptr(segm_out_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(segm_max_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(segm_sum_addr as CUdeviceptr),
            DevicePointer::<bf16>::from_cu_deviceptr(q_addr as CUdeviceptr),
            DevicePointer::<bf16>::from_cu_deviceptr(kc_addr as CUdeviceptr),
            DevicePointer::<bf16>::from_cu_deviceptr(vc_addr as CUdeviceptr),
            DevicePointer::<u32>::from_cu_deviceptr(bt_addr as CUdeviceptr),
            DevicePointer::<u32>::from_cu_deviceptr(cl_addr as CUdeviceptr),
            softmax_scale,
            num_q_heads as i32,
            bt_l.stride()[0] as i32,
            q_l.stride()[0] as i32,
            q_l.stride()[1] as i32,
            kc_l.stride()[0] as i32,
            kc_l.stride()[1] as i32,
            kc_l.stride()[2] as i32,
            kc_l.stride()[3] as i32,
            vc_l.stride()[0] as i32,
            vc_l.stride()[1] as i32,
            vc_l.stride()[2] as i32,
            vc_l.stride()[3] as i32,
            block_size as i32,
        )
    }
    .generics(generics)
    .grid((
        batch as u32,
        num_kv_heads as u32,
        NUM_DECODE_SEGMENTS as u32,
    ));
    unsafe { launcher.execute(&ctx) }.map_err(|err| {
        candle_core::Error::Msg(format!("cutile paged decode segment launch: {err:?}"))
    })?;
    drop((segm_out_guard, segm_max_guard, segm_sum_guard));
    Ok(())
}

fn launch_reduce(
    out: &mut CudaSlice<bf16>,
    segm_out: &CudaSlice<f32>,
    segm_max: &CudaSlice<f32>,
    segm_sum: &CudaSlice<f32>,
    context_lens: &Tensor,
    num_q_heads: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let (cl_storage, cl_l) = cuda_storage_layout(context_lens)?;
    let cl_s = &cl_storage;
    let stream = dev.cuda_stream();
    let cl = cl_s.as_cuda_slice::<u32>()?;
    let (cl_addr, _cl_guard) = slice_ptr_on_stream(cl, cl_l.start_offset(), &stream);
    let (segm_out_addr, _segm_out_guard) = slice_ptr_on_stream(segm_out, 0, &stream);
    let (segm_max_addr, _segm_max_guard) = slice_ptr_on_stream(segm_max, 0, &stream);
    let (segm_sum_addr, _segm_sum_guard) = slice_ptr_on_stream(segm_sum, 0, &stream);
    let (out_addr, out_guard) = slice_ptr_mut_on_stream(out, 0, &stream);
    let generics = vec![HEAD_SIZE_512.to_string(), NUM_DECODE_SEGMENTS.to_string()];
    let ctx = context::execution_context(dev);
    let batch = context_lens.dims1()?;
    let launcher = unsafe {
        unified_attention::reduce_segments_bf16(
            DevicePointer::<bf16>::from_cu_deviceptr(out_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(segm_out_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(segm_max_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(segm_sum_addr as CUdeviceptr),
            DevicePointer::<u32>::from_cu_deviceptr(cl_addr as CUdeviceptr),
            num_q_heads as i32,
            (num_q_heads * HEAD_SIZE_512) as i32,
            HEAD_SIZE_512 as i32,
        )
    }
    .generics(generics)
    .grid((batch as u32, num_q_heads as u32, 1));
    unsafe { launcher.execute(&ctx) }.map_err(|err| {
        candle_core::Error::Msg(format!("cutile reduce segments launch: {err:?}"))
    })?;
    drop(out_guard);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn cutile_paged_attention_prefill(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    q_indptr: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let (total_q, num_q_heads, head_size) = query.dims3()?;
    let (_, num_kv_heads, block_size, cache_head_size) = key_cache.dims4()?;
    if query.dtype() != DType::BF16
        || key_cache.dtype() != DType::BF16
        || value_cache.dtype() != DType::BF16
        || head_size != HEAD_SIZE_512
        || cache_head_size != head_size
        || value_cache.dims4()? != key_cache.dims4()?
        || block_tables.dtype() != DType::U32
        || context_lens.dtype() != DType::U32
        || q_indptr.dtype() != DType::I32
        || block_size % PREFILL_TILE_SIZE != 0
        || num_q_heads % num_kv_heads != 0
        || num_q_heads / num_kv_heads > BLOCK_M
        || BLOCK_M % (num_q_heads / num_kv_heads) != 0
    {
        candle_core::bail!("cutile paged prefill unsupported shape");
    }
    let dev = match query.device() {
        candle_core::Device::Cuda(dev) if device_supported(dev) => dev,
        _ => candle_core::bail!("cutile paged prefill requires a supported cuda device"),
    };
    let mut out = unsafe { dev.alloc::<bf16>(total_q * num_q_heads * head_size)? };
    let total_q_blocks =
        total_q / (BLOCK_M / (num_q_heads / num_kv_heads)) + context_lens.dims1()?;
    launch_prefill(
        &mut out,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        q_indptr,
        softmax_scale,
        num_q_heads,
        num_kv_heads,
        total_q_blocks,
        dev,
    )?;
    let storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((
        Storage::Cuda(storage),
        (total_q, num_q_heads, head_size),
    )))
}

pub fn cutile_paged_attention_decode(
    query: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let (batch, num_q_heads, head_size) = query.dims3()?;
    let (_, num_kv_heads, block_size, cache_head_size) = key_cache.dims4()?;
    if query.dtype() != DType::BF16
        || key_cache.dtype() != DType::BF16
        || value_cache.dtype() != DType::BF16
        || head_size != HEAD_SIZE_512
        || cache_head_size != head_size
        || value_cache.dims4()? != key_cache.dims4()?
        || block_tables.dtype() != DType::U32
        || context_lens.dtype() != DType::U32
        || block_size % DECODE_TILE_SIZE != 0
        || num_q_heads % num_kv_heads != 0
        || num_q_heads / num_kv_heads > BLOCK_M
        || BLOCK_M % (num_q_heads / num_kv_heads) != 0
    {
        candle_core::bail!("cutile paged decode unsupported shape");
    }
    let dev = match query.device() {
        candle_core::Device::Cuda(dev) if device_supported(dev) => dev,
        _ => candle_core::bail!("cutile paged decode requires a supported cuda device"),
    };
    let mut out = unsafe { dev.alloc::<bf16>(batch * num_q_heads * head_size)? };
    let mut segm_out =
        unsafe { dev.alloc::<f32>(batch * num_q_heads * NUM_DECODE_SEGMENTS * head_size)? };
    let mut segm_max = unsafe { dev.alloc::<f32>(batch * num_q_heads * NUM_DECODE_SEGMENTS)? };
    let mut segm_sum = unsafe { dev.alloc::<f32>(batch * num_q_heads * NUM_DECODE_SEGMENTS)? };
    launch_decode_segments(
        &mut segm_out,
        &mut segm_max,
        &mut segm_sum,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        softmax_scale,
        num_q_heads,
        num_kv_heads,
        dev,
    )?;
    launch_reduce(
        &mut out,
        &segm_out,
        &segm_max,
        &segm_sum,
        context_lens,
        num_q_heads,
        dev,
    )?;
    let storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((
        Storage::Cuda(storage),
        (batch, num_q_heads, head_size),
    )))
}

pub fn warmup_cutile_attention_kernels(device: &Device) -> Result<()> {
    let Device::Cuda(dev) = device else {
        return Ok(());
    };
    if !device_supported(dev) {
        return Ok(());
    }
    let location = device.location();
    let q_groups = registered_attention_q_groups();
    if q_groups.is_empty() {
        return Ok(());
    }
    let mut q_groups_to_warm = Vec::new();
    {
        let mut warmed = WARMED_ATTENTION_Q_GROUPS
            .get_or_init(|| Mutex::new(HashSet::new()))
            .lock()
            .unwrap();
        for q_group in q_groups {
            if warmed.insert((location, q_group)) {
                q_groups_to_warm.push(q_group);
            }
        }
    }
    if q_groups_to_warm.is_empty() {
        return Ok(());
    }
    for q_group in &q_groups_to_warm {
        if let Err(err) = warmup_attention_q_group(device, *q_group) {
            let mut warmed = WARMED_ATTENTION_Q_GROUPS
                .get_or_init(|| Mutex::new(HashSet::new()))
                .lock()
                .unwrap();
            for q_group in q_groups_to_warm {
                warmed.remove(&(location, q_group));
            }
            return Err(err);
        }
    }
    device.synchronize()
}

fn registered_attention_q_groups() -> Vec<usize> {
    let mut groups = REGISTERED_ATTENTION_Q_GROUPS
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap()
        .iter()
        .copied()
        .collect::<Vec<_>>();
    groups.sort_unstable();
    groups
}

fn warmup_attention_q_group(device: &Device, q_group: usize) -> Result<()> {
    let block_size = 32usize;
    let key_cache = Tensor::zeros(
        (1usize, 1usize, block_size, HEAD_SIZE_512),
        DType::BF16,
        device,
    )?;
    let value_cache = Tensor::zeros(
        (1usize, 1usize, block_size, HEAD_SIZE_512),
        DType::BF16,
        device,
    )?;
    let block_tables = Tensor::from_vec(vec![0u32], (1usize, 1usize), device)?;
    let context_lens = Tensor::from_vec(vec![block_size as u32], (1usize,), device)?;
    let query_decode = Tensor::zeros((1usize, q_group, HEAD_SIZE_512), DType::BF16, device)?;
    let _ = cutile_paged_attention_decode(
        &query_decode,
        &key_cache,
        &value_cache,
        &block_tables,
        &context_lens,
        1.0 / (HEAD_SIZE_512 as f32).sqrt(),
    )?;
    Ok(())
}
