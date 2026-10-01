//! Qwen4-Exp QSA prefill kernels on tensor cores: indexer block scoring and sparse attention over
//! each query's selected blocks.
#![allow(clippy::too_many_arguments, clippy::missing_safety_doc)]

use candle_core::{CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor};
use cutile::core::bf16 as tile_bf16;
use cutile::cuda_async::device_buffer::DevicePointer;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tile_kernel::TileKernel;
use half::bf16;

use super::warmup::CutileKernel;
use super::{catch_cutile_panic, context, jit_available};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

pub const QSA_SCORE_TOKENS: usize = 16;
pub const QSA_INDEX_HEADS: usize = 4;
pub const QSA_INDEX_DIM: usize = 128;
const QSA_SCORE_BLOCKS: usize = 64;
pub const QSA_ATTN_HEAD_DIM: usize = 256;
const QSA_ATTN_HEAD_ROWS: usize = 16;
const QSA_ATTN_ITEMS: usize = 64;

// Scalars that follow the context length travel as f32: cuTile keys its JIT cache on the
// divisibility of every i32 scalar.
#[cutile::module]
mod kernels {
    use cutile::core::*;

    // scores[t, j] = scale * sum_h relu(q[t, h] . block_key[j]) for tokens that select fewer blocks than
    // they have. Grid (score tiles, block tiles); a score tile is [first token, sequence, tokens, first
    // position] of up to BT consecutive tokens of one sequence.
    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn qsa_score<const BT: i32, const NH: i32, const R: i32, const D: i32, const BN: i32>(
        scores_ptr: *mut f32,
        q_ptr: *mut bf16,
        block_keys_ptr: *mut bf16,
        tables_ptr: *mut i32,
        tiles_ptr: *mut i32,
        score_stride_f: f32,
        max_blocks_f: f32,
        ratio: i32,
        block_size: i32,
        block_keys_dim: i32,
        topk: i32,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let tile: i32 = pid.0;
        let j0: i32 = pid.1 * BN;
        let stride_t: Tile<f32, { [1] }> = broadcast_scalar(score_stride_f, const_shape![1]);
        let stride_i: Tile<i32, { [1] }> = convert_tile(stride_t);
        let stride_s: Tile<i32, { [] }> = stride_i.reshape(const_shape![]);
        let score_stride: i32 = tile_to_scalar(stride_s);
        let mb_t: Tile<f32, { [1] }> = broadcast_scalar(max_blocks_f, const_shape![1]);
        let mb_i: Tile<i32, { [1] }> = convert_tile(mb_t);
        let mb_s: Tile<i32, { [] }> = mb_i.reshape(const_shape![]);
        let max_blocks: i32 = tile_to_scalar(mb_s);

        let d_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(tiles_ptr);
        let d_p1: PointerTile<*mut i32, { [1] }> = d_p0.reshape(const_shape![1]);
        let d0: Tile<i32, { [1] }> = broadcast_scalar(tile * 4, const_shape![1]);
        let d0_ptr: PointerTile<*mut i32, { [1] }> = d_p1.offset_tile(d0);
        let (t0_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            d0_ptr,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let d1: Tile<i32, { [1] }> = broadcast_scalar(tile * 4 + 1, const_shape![1]);
        let d1_ptr: PointerTile<*mut i32, { [1] }> = d_p1.offset_tile(d1);
        let (seq_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            d1_ptr,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let d2: Tile<i32, { [1] }> = broadcast_scalar(tile * 4 + 2, const_shape![1]);
        let d2_ptr: PointerTile<*mut i32, { [1] }> = d_p1.offset_tile(d2);
        let (nt_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            d2_ptr,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let d3: Tile<i32, { [1] }> = broadcast_scalar(tile * 4 + 3, const_shape![1]);
        let d3_ptr: PointerTile<*mut i32, { [1] }> = d_p1.offset_tile(d3);
        let (p0_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            d3_ptr,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let t0_s: Tile<i32, { [] }> = t0_t.reshape(const_shape![]);
        let t0: i32 = tile_to_scalar(t0_s);
        let seq_s: Tile<i32, { [] }> = seq_t.reshape(const_shape![]);
        let seq: i32 = tile_to_scalar(seq_s);
        let nt_s: Tile<i32, { [] }> = nt_t.reshape(const_shape![]);
        let n_tok: i32 = tile_to_scalar(nt_s);
        let p0_s: Tile<i32, { [] }> = p0_t.reshape(const_shape![]);
        let pos0: i32 = tile_to_scalar(p0_s);
        let nb_last: i32 = (pos0 + n_tok) / ratio;

        if nb_last > topk && j0 < nb_last {
            let iota_d: Tile<i32, { [D] }> = iota(const_shape![D]);
            let iota_r: Tile<i32, { [R] }> = iota(const_shape![R]);
            let nh_r: Tile<i32, { [R] }> = broadcast_scalar(NH, const_shape![R]);
            let tok_r: Tile<i32, { [R] }> = iota_r / nh_r;
            let ntok_r: Tile<i32, { [R] }> = broadcast_scalar(n_tok, const_shape![R]);
            let row_ok: Tile<bool, { [R] }> = lt_tile(tok_r, ntok_r);
            let qrow0: Tile<i32, { [R] }> = broadcast_scalar(t0 * NH, const_shape![R]);
            let dim_r: Tile<i32, { [R] }> = broadcast_scalar(D, const_shape![R]);
            let q_row: Tile<i32, { [R] }> = (qrow0 + iota_r) * dim_r;
            let q_off: Tile<i32, { [R, D] }> = q_row
                .reshape(const_shape![R, 1])
                .broadcast(const_shape![R, D])
                + iota_d
                    .reshape(const_shape![1, D])
                    .broadcast(const_shape![R, D]);
            let q_mask: Tile<bool, { [R, D] }> = row_ok
                .reshape(const_shape![R, 1])
                .broadcast(const_shape![R, D]);
            let q_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(q_ptr);
            let q_p1: PointerTile<*mut bf16, { [1, 1] }> = q_p0.reshape(const_shape![1, 1]);
            let q_p2: PointerTile<*mut bf16, { [R, D] }> = q_p1.broadcast(const_shape![R, D]);
            let q_ptrs: PointerTile<*mut bf16, { [R, D] }> = q_p2.offset_tile(q_off);
            let (q_raw, _): (Tile<bf16, { [R, D] }>, Token) = load_ptr_tko(
                q_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(q_mask),
                None,
                None,
                Latency::<0>,
            );
            let zero_rd: Tile<bf16, { [R, D] }> = constant(bf16::ZERO, const_shape![R, D]);
            let q_tile: Tile<bf16, { [R, D] }> = select(q_mask, q_raw, zero_rd);

            let iota_n: Tile<i32, { [BN] }> = iota(const_shape![BN]);
            let j: Tile<i32, { [BN] }> = broadcast_scalar(j0, const_shape![BN]) + iota_n;
            let nb_n: Tile<i32, { [BN] }> = broadcast_scalar(nb_last, const_shape![BN]);
            let key_ok: Tile<bool, { [BN] }> = lt_tile(j, nb_n);
            let zero_n: Tile<i32, { [BN] }> = constant(0i32, const_shape![BN]);
            let ratio_n: Tile<i32, { [BN] }> = broadcast_scalar(ratio, const_shape![BN]);
            let bs_n: Tile<i32, { [BN] }> = broadcast_scalar(block_size, const_shape![BN]);
            let key_pos: Tile<i32, { [BN] }> = select(key_ok, j * ratio_n, zero_n);
            let page: Tile<i32, { [BN] }> = key_pos / bs_n;
            let table_row: Tile<i32, { [BN] }> =
                broadcast_scalar(seq * max_blocks, const_shape![BN]);
            let tb_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(tables_ptr);
            let tb_p1: PointerTile<*mut i32, { [1] }> = tb_p0.reshape(const_shape![1]);
            let tb_p2: PointerTile<*mut i32, { [BN] }> = tb_p1.broadcast(const_shape![BN]);
            let tb_idx: Tile<i32, { [BN] }> = table_row + page;
            let tb_ptrs: PointerTile<*mut i32, { [BN] }> = tb_p2.offset_tile(tb_idx);
            let (blk_raw, _): (Tile<i32, { [BN] }>, Token) = load_ptr_tko(
                tb_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(key_ok),
                None,
                None,
                Latency::<0>,
            );
            let blk: Tile<i32, { [BN] }> = select(key_ok, blk_raw, zero_n);
            let slot: Tile<i32, { [BN] }> = blk * bs_n + key_pos - page * bs_n;
            let block_keys_n: Tile<i32, { [BN] }> =
                broadcast_scalar(block_keys_dim, const_shape![BN]);
            let k_off: Tile<i32, { [BN, D] }> = (slot / ratio_n * block_keys_n)
                .reshape(const_shape![BN, 1])
                .broadcast(const_shape![BN, D])
                + iota_d
                    .reshape(const_shape![1, D])
                    .broadcast(const_shape![BN, D]);
            let k_mask: Tile<bool, { [BN, D] }> = key_ok
                .reshape(const_shape![BN, 1])
                .broadcast(const_shape![BN, D]);
            let a_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(block_keys_ptr);
            let a_p1: PointerTile<*mut bf16, { [1, 1] }> = a_p0.reshape(const_shape![1, 1]);
            let a_p2: PointerTile<*mut bf16, { [BN, D] }> = a_p1.broadcast(const_shape![BN, D]);
            let k_ptrs: PointerTile<*mut bf16, { [BN, D] }> = a_p2.offset_tile(k_off);
            let (k_raw, _): (Tile<bf16, { [BN, D] }>, Token) = load_ptr_tko(
                k_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(k_mask),
                None,
                None,
                Latency::<0>,
            );
            let zero_keys: Tile<bf16, { [BN, D] }> = constant(bf16::ZERO, const_shape![BN, D]);
            let k_tile: Tile<bf16, { [BN, D] }> = select(k_mask, k_raw, zero_keys);
            let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
                dims: &[1i32, 0i32],
            };
            let k_t: Tile<bf16, { [D, BN] }> = permute(k_tile, transpose);
            let zero_rn: Tile<f32, { [R, BN] }> = constant(0.0f32, const_shape![R, BN]);
            let dots: Tile<f32, { [R, BN] }> = mmaf(q_tile, k_t, zero_rn);
            let relu: Tile<f32, { [R, BN] }> = max_tile(dots, zero_rn);
            let per_head: Tile<f32, { [BT, NH, BN] }> = relu.reshape(const_shape![BT, NH, BN]);
            let head_sum: Tile<f32, { [BT, BN] }> = reduce_sum(per_head, 1i32);
            // reductions come back with concrete shapes; reshaping restores the generic ones
            let summed: Tile<f32, { [BT, BN] }> = head_sum.reshape(const_shape![BT, BN]);
            let scale_tn: Tile<f32, { [BT, BN] }> = broadcast_scalar(scale, const_shape![BT, BN]);
            let score: Tile<f32, { [BT, BN] }> = summed * scale_tn;

            let iota_t: Tile<i32, { [BT] }> = iota(const_shape![BT]);
            let ntok_t: Tile<i32, { [BT] }> = broadcast_scalar(n_tok, const_shape![BT]);
            let tok_ok: Tile<bool, { [BT] }> = lt_tile(iota_t, ntok_t);
            let ratio_t: Tile<i32, { [BT] }> = broadcast_scalar(ratio, const_shape![BT]);
            let one_t: Tile<i32, { [BT] }> = broadcast_scalar(pos0 + 1, const_shape![BT]);
            let nb_t: Tile<i32, { [BT] }> = (iota_t + one_t) / ratio_t;
            let topk_t: Tile<i32, { [BT] }> = broadcast_scalar(topk, const_shape![BT]);
            let scored_t: Tile<bool, { [BT] }> = tok_ok & gt_tile(nb_t, topk_t);
            let nb_tn: Tile<i32, { [BT, BN] }> = nb_t
                .reshape(const_shape![BT, 1])
                .broadcast(const_shape![BT, BN]);
            let j_tn: Tile<i32, { [BT, BN] }> = j
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BT, BN]);
            let store_mask: Tile<bool, { [BT, BN] }> = scored_t
                .reshape(const_shape![BT, 1])
                .broadcast(const_shape![BT, BN])
                & lt_tile(j_tn, nb_tn);
            let row0_t: Tile<i32, { [BT] }> = broadcast_scalar(t0, const_shape![BT]);
            let stride_tt: Tile<i32, { [BT] }> = broadcast_scalar(score_stride, const_shape![BT]);
            let s_off: Tile<i32, { [BT, BN] }> = ((row0_t + iota_t) * stride_tt)
                .reshape(const_shape![BT, 1])
                .broadcast(const_shape![BT, BN])
                + j_tn;
            let s_p0: PointerTile<*mut f32, { [] }> = pointer_to_tile(scores_ptr);
            let s_p1: PointerTile<*mut f32, { [1, 1] }> = s_p0.reshape(const_shape![1, 1]);
            let s_p2: PointerTile<*mut f32, { [BT, BN] }> = s_p1.broadcast(const_shape![BT, BN]);
            let s_ptrs: PointerTile<*mut f32, { [BT, BN] }> = s_p2.offset_tile(s_off);
            store_ptr_tko(
                s_ptrs,
                score,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(store_mask),
                None,
                Latency::<0>,
            );
        }
    }

    // Attention of one token's query heads sharing kv head g over its selected blocks plus the
    // incomplete tail. Grid (tokens, kv heads); `tokens` rows are [sequence (negative = padding),
    // position]; K and V are [blocks, kv heads, block size, D] (FlashInfer HND).
    #[cutile::entry(unchecked_accesses = false)]
    unsafe fn qsa_attention<const HP: i32, const D: i32, const BK: i32>(
        out_ptr: *mut bf16,
        q_ptr: *mut bf16,
        k_ptr: *mut bf16,
        v_ptr: *mut bf16,
        tables_ptr: *mut i32,
        selected_ptr: *mut i32,
        n_selected_ptr: *mut i32,
        tokens_ptr: *mut i32,
        max_blocks_f: f32,
        topk: i32,
        ratio: i32,
        block_size: i32,
        group: i32,
        n_q_heads: i32,
        n_kv_heads: i32,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let t: i32 = pid.0;
        let g: i32 = pid.1;
        let mb_t: Tile<f32, { [1] }> = broadcast_scalar(max_blocks_f, const_shape![1]);
        let mb_i: Tile<i32, { [1] }> = convert_tile(mb_t);
        let mb_s: Tile<i32, { [] }> = mb_i.reshape(const_shape![]);
        let max_blocks: i32 = tile_to_scalar(mb_s);

        let tk_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(tokens_ptr);
        let tk_p1: PointerTile<*mut i32, { [1] }> = tk_p0.reshape(const_shape![1]);
        let o_seq: Tile<i32, { [1] }> = broadcast_scalar(t * 2, const_shape![1]);
        let o_seq_ptr: PointerTile<*mut i32, { [1] }> = tk_p1.offset_tile(o_seq);
        let (seq_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            o_seq_ptr,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let o_pos: Tile<i32, { [1] }> = broadcast_scalar(t * 2 + 1, const_shape![1]);
        let o_pos_ptr: PointerTile<*mut i32, { [1] }> = tk_p1.offset_tile(o_pos);
        let (pos_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            o_pos_ptr,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let seq_s: Tile<i32, { [] }> = seq_t.reshape(const_shape![]);
        let seq: i32 = tile_to_scalar(seq_s);
        let pos_s: Tile<i32, { [] }> = pos_t.reshape(const_shape![]);
        let pos: i32 = tile_to_scalar(pos_s);

        let iota_h: Tile<i32, { [HP] }> = iota(const_shape![HP]);
        let iota_d: Tile<i32, { [D] }> = iota(const_shape![D]);
        let group_h: Tile<i32, { [HP] }> = broadcast_scalar(group, const_shape![HP]);
        let head_ok: Tile<bool, { [HP] }> = lt_tile(iota_h, group_h);
        let head_mask: Tile<bool, { [HP, D] }> = head_ok
            .reshape(const_shape![HP, 1])
            .broadcast(const_shape![HP, D]);
        let head0: Tile<i32, { [HP] }> =
            broadcast_scalar(t * n_q_heads + g * group, const_shape![HP]);
        let dim_h: Tile<i32, { [HP] }> = broadcast_scalar(D, const_shape![HP]);
        let hd_off: Tile<i32, { [HP, D] }> = ((head0 + iota_h) * dim_h)
            .reshape(const_shape![HP, 1])
            .broadcast(const_shape![HP, D])
            + iota_d
                .reshape(const_shape![1, D])
                .broadcast(const_shape![HP, D]);
        let o_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
        let o_p1: PointerTile<*mut bf16, { [1, 1] }> = o_p0.reshape(const_shape![1, 1]);
        let o_p2: PointerTile<*mut bf16, { [HP, D] }> = o_p1.broadcast(const_shape![HP, D]);
        let o_ptrs: PointerTile<*mut bf16, { [HP, D] }> = o_p2.offset_tile(hd_off);
        let zero_hd: Tile<f32, { [HP, D] }> = constant(0.0f32, const_shape![HP, D]);

        if seq >= 0 {
            let q_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(q_ptr);
            let q_p1: PointerTile<*mut bf16, { [1, 1] }> = q_p0.reshape(const_shape![1, 1]);
            let q_p2: PointerTile<*mut bf16, { [HP, D] }> = q_p1.broadcast(const_shape![HP, D]);
            let q_ptrs: PointerTile<*mut bf16, { [HP, D] }> = q_p2.offset_tile(hd_off);
            let (q_raw, _): (Tile<bf16, { [HP, D] }>, Token) = load_ptr_tko(
                q_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(head_mask),
                None,
                None,
                Latency::<0>,
            );
            let zero_q: Tile<bf16, { [HP, D] }> = constant(bf16::ZERO, const_shape![HP, D]);
            let q_tile: Tile<bf16, { [HP, D] }> = select(head_mask, q_raw, zero_q);

            let ns_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(n_selected_ptr);
            let ns_p1: PointerTile<*mut i32, { [1] }> = ns_p0.reshape(const_shape![1]);
            let ns_off: Tile<i32, { [1] }> = broadcast_scalar(t, const_shape![1]);
            let ns_ptr: PointerTile<*mut i32, { [1] }> = ns_p1.offset_tile(ns_off);
            let (ns_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
                ns_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let ns_s: Tile<i32, { [] }> = ns_t.reshape(const_shape![]);
            let n_sel: i32 = tile_to_scalar(ns_s);
            let nb: i32 = (pos + 1) / ratio;
            let n_block_items: i32 = n_sel * ratio;
            let tail_start: i32 = nb * ratio;
            let n_items: i32 = n_block_items + pos + 1 - tail_start;
            let n_chunks: i32 = (n_items + BK - 1) / BK;

            let iota_k: Tile<i32, { [BK] }> = iota(const_shape![BK]);
            let zero_k: Tile<i32, { [BK] }> = constant(0i32, const_shape![BK]);
            let ratio_k: Tile<i32, { [BK] }> = broadcast_scalar(ratio, const_shape![BK]);
            let bs_k: Tile<i32, { [BK] }> = broadcast_scalar(block_size, const_shape![BK]);
            let items_k: Tile<i32, { [BK] }> = broadcast_scalar(n_items, const_shape![BK]);
            let blocks_k: Tile<i32, { [BK] }> = broadcast_scalar(n_block_items, const_shape![BK]);
            let sel_row: Tile<i32, { [BK] }> = broadcast_scalar(t * topk, const_shape![BK]);
            let tail_k: Tile<i32, { [BK] }> =
                broadcast_scalar(tail_start - n_block_items, const_shape![BK]);
            let table_row: Tile<i32, { [BK] }> =
                broadcast_scalar(seq * max_blocks, const_shape![BK]);
            let kvh_k: Tile<i32, { [BK] }> = broadcast_scalar(n_kv_heads, const_shape![BK]);
            let g_k: Tile<i32, { [BK] }> = broadcast_scalar(g, const_shape![BK]);
            let dim_k: Tile<i32, { [BK] }> = broadcast_scalar(D, const_shape![BK]);
            let col_kd: Tile<i32, { [BK, D] }> = iota_d
                .reshape(const_shape![1, D])
                .broadcast(const_shape![BK, D]);
            let sel_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(selected_ptr);
            let sel_p1: PointerTile<*mut i32, { [1] }> = sel_p0.reshape(const_shape![1]);
            let tb_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(tables_ptr);
            let tb_p1: PointerTile<*mut i32, { [1] }> = tb_p0.reshape(const_shape![1]);
            let k_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(k_ptr);
            let k_p1: PointerTile<*mut bf16, { [1, 1] }> = k_p0.reshape(const_shape![1, 1]);
            let v_p0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(v_ptr);
            let v_p1: PointerTile<*mut bf16, { [1, 1] }> = v_p0.reshape(const_shape![1, 1]);
            let zero_kd: Tile<bf16, { [BK, D] }> = constant(bf16::ZERO, const_shape![BK, D]);
            let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
                dims: &[1i32, 0i32],
            };
            let scale_hk: Tile<f32, { [HP, BK] }> = broadcast_scalar(scale, const_shape![HP, BK]);
            // finite mask so masked softmax weights vanish without inf - inf
            let masked_hk: Tile<f32, { [HP, BK] }> = constant(-1.0e30f32, const_shape![HP, BK]);
            let zero_hk: Tile<f32, { [HP, BK] }> = constant(0.0f32, const_shape![HP, BK]);

            let mut m: Tile<f32, { [HP] }> = constant(-1.0e30f32, const_shape![HP]);
            let mut l: Tile<f32, { [HP] }> = constant(0.0f32, const_shape![HP]);
            let mut acc: Tile<f32, { [HP, D] }> = zero_hd;
            for chunk in 0i32..n_chunks {
                let it: Tile<i32, { [BK] }> =
                    broadcast_scalar(chunk * BK, const_shape![BK]) + iota_k;
                let valid: Tile<bool, { [BK] }> = lt_tile(it, items_k);
                let is_block: Tile<bool, { [BK] }> = valid & lt_tile(it, blocks_k);
                let blk_item: Tile<i32, { [BK] }> = select(is_block, it / ratio_k, zero_k);
                let sel_p2: PointerTile<*mut i32, { [BK] }> = sel_p1.broadcast(const_shape![BK]);
                let sel_idx: Tile<i32, { [BK] }> = sel_row + blk_item;
                let sel_ptrs: PointerTile<*mut i32, { [BK] }> = sel_p2.offset_tile(sel_idx);
                let (sel_raw, _): (Tile<i32, { [BK] }>, Token) = load_ptr_tko(
                    sel_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(is_block),
                    None,
                    None,
                    Latency::<0>,
                );
                let sel: Tile<i32, { [BK] }> = select(is_block, sel_raw, zero_k);
                let p_block: Tile<i32, { [BK] }> = sel * ratio_k + it - blk_item * ratio_k;
                let p_tail: Tile<i32, { [BK] }> = it + tail_k;
                let p_any: Tile<i32, { [BK] }> = select(is_block, p_block, p_tail);
                let p: Tile<i32, { [BK] }> = select(valid, p_any, zero_k);
                let page: Tile<i32, { [BK] }> = p / bs_k;
                let tb_p2: PointerTile<*mut i32, { [BK] }> = tb_p1.broadcast(const_shape![BK]);
                let tb_idx: Tile<i32, { [BK] }> = table_row + page;
                let tb_ptrs: PointerTile<*mut i32, { [BK] }> = tb_p2.offset_tile(tb_idx);
                let (blk_raw, _): (Tile<i32, { [BK] }>, Token) = load_ptr_tko(
                    tb_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(valid),
                    None,
                    None,
                    Latency::<0>,
                );
                let blk: Tile<i32, { [BK] }> = select(valid, blk_raw, zero_k);
                let off: Tile<i32, { [BK] }> = p - page * bs_k;
                let row: Tile<i32, { [BK] }> = ((blk * kvh_k + g_k) * bs_k + off) * dim_k;
                let kv_off: Tile<i32, { [BK, D] }> = row
                    .reshape(const_shape![BK, 1])
                    .broadcast(const_shape![BK, D])
                    + col_kd;
                let kv_mask: Tile<bool, { [BK, D] }> = valid
                    .reshape(const_shape![BK, 1])
                    .broadcast(const_shape![BK, D]);
                let k_p2: PointerTile<*mut bf16, { [BK, D] }> = k_p1.broadcast(const_shape![BK, D]);
                let k_ptrs: PointerTile<*mut bf16, { [BK, D] }> = k_p2.offset_tile(kv_off);
                let v_p2: PointerTile<*mut bf16, { [BK, D] }> = v_p1.broadcast(const_shape![BK, D]);
                let v_ptrs: PointerTile<*mut bf16, { [BK, D] }> = v_p2.offset_tile(kv_off);
                let (k_raw, _): (Tile<bf16, { [BK, D] }>, Token) = load_ptr_tko(
                    k_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(kv_mask),
                    None,
                    None,
                    Latency::<0>,
                );
                let (v_raw, _): (Tile<bf16, { [BK, D] }>, Token) = load_ptr_tko(
                    v_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(kv_mask),
                    None,
                    None,
                    Latency::<0>,
                );
                let k_tile: Tile<bf16, { [BK, D] }> = select(kv_mask, k_raw, zero_kd);
                let v_tile: Tile<bf16, { [BK, D] }> = select(kv_mask, v_raw, zero_kd);
                let k_t: Tile<bf16, { [D, BK] }> = permute(k_tile, transpose);
                let dots: Tile<f32, { [HP, BK] }> = mmaf(q_tile, k_t, zero_hk);
                let col_ok: Tile<bool, { [HP, BK] }> = valid
                    .reshape(const_shape![1, BK])
                    .broadcast(const_shape![HP, BK]);
                let scaled: Tile<f32, { [HP, BK] }> = dots * scale_hk;
                let s: Tile<f32, { [HP, BK] }> = select(col_ok, scaled, masked_hk);
                let chunk_max_raw: Tile<f32, { [HP] }> = reduce_max(s, 1i32);
                let chunk_max: Tile<f32, { [HP] }> = chunk_max_raw.reshape(const_shape![HP]);
                let m_new: Tile<f32, { [HP] }> = max_tile(m, chunk_max);
                let corr: Tile<f32, { [HP] }> = exp(m - m_new);
                let m_hk: Tile<f32, { [HP, BK] }> = m_new
                    .reshape(const_shape![HP, 1])
                    .broadcast(const_shape![HP, BK]);
                let shifted: Tile<f32, { [HP, BK] }> = exp(s - m_hk);
                let probs: Tile<f32, { [HP, BK] }> = select(col_ok, shifted, zero_hk);
                let row_sum_raw: Tile<f32, { [HP] }> = reduce_sum(probs, 1i32);
                let row_sum: Tile<f32, { [HP] }> = row_sum_raw.reshape(const_shape![HP]);
                let l_new: Tile<f32, { [HP] }> = l * corr + row_sum;
                let corr_hd: Tile<f32, { [HP, D] }> = corr
                    .reshape(const_shape![HP, 1])
                    .broadcast(const_shape![HP, D]);
                let probs_b: Tile<bf16, { [HP, BK] }> = convert_tile(probs);
                let acc_scaled: Tile<f32, { [HP, D] }> = acc * corr_hd;
                let acc_new: Tile<f32, { [HP, D] }> = mmaf(probs_b, v_tile, acc_scaled);
                m = m_new;
                l = l_new;
                acc = acc_new;
            }
            let zero_h: Tile<f32, { [HP] }> = constant(0.0f32, const_shape![HP]);
            let one_h: Tile<f32, { [HP] }> = constant(1.0f32, const_shape![HP]);
            let has_mass: Tile<bool, { [HP] }> = gt_tile(l, zero_h);
            let inv: Tile<f32, { [HP] }> = select(has_mass, one_h / l, zero_h);
            let out: Tile<f32, { [HP, D] }> = acc
                * inv
                    .reshape(const_shape![HP, 1])
                    .broadcast(const_shape![HP, D]);
            let out_b: Tile<bf16, { [HP, D] }> = convert_tile(out);
            store_ptr_tko(
                o_ptrs,
                out_b,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(head_mask),
                None,
                Latency::<0>,
            );
        } else {
            let zero_b: Tile<bf16, { [HP, D] }> = convert_tile(zero_hd);
            store_ptr_tko(
                o_ptrs,
                zero_b,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(head_mask),
                None,
                Latency::<0>,
            );
        }
    }
}

/// Paged view of one QSA layer: block tables `[seqs, max_blocks_per_seq]` (I32 or U32).
pub struct QsaPaged<'a> {
    pub block_tables: &'a Tensor,
    pub max_blocks_per_seq: usize,
    pub block_size: usize,
}

/// Prefill scoring from BF16 queries and pooled block keys, using per-sequence I32 score tiles.
pub struct QsaScoreArgs<'a> {
    pub q: &'a Tensor,
    pub block_keys: &'a Tensor,
    pub paged: &'a QsaPaged<'a>,
    pub tiles: &'a Tensor,
    pub scores: &'a Tensor,
    pub max_blocks: usize,
    pub ratio: usize,
    pub topk: usize,
}

/// Sparse attention for prefill. `q: [tokens, q_heads, 256]` BF16, caches `[blocks, kv_heads,
/// block_size, 256]` BF16, `selected: [tokens, topk]` and `n_selected: [tokens]` I32, `tokens:
/// [tokens, 2]` I32 (sequence or -1, kv position).
pub struct QsaAttentionArgs<'a> {
    pub q: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub paged: &'a QsaPaged<'a>,
    pub selected: &'a Tensor,
    pub n_selected: &'a Tensor,
    pub tokens: &'a Tensor,
    pub n_kv_heads: usize,
    pub ratio: usize,
    pub scale: f32,
}

fn device_addr<'a>(
    cuda: &'a CudaStorage,
    dtype: DType,
    start: usize,
    stream: &'a candle_core::cuda::cudarc::driver::CudaStream,
) -> Result<(u64, candle_core::cuda::cudarc::driver::SyncOnDrop<'a>)> {
    Ok(match dtype {
        DType::BF16 => slice_ptr_on_stream(cuda.as_cuda_slice::<bf16>()?, start, stream),
        DType::F32 => slice_ptr_on_stream(cuda.as_cuda_slice::<f32>()?, start, stream),
        DType::I32 => slice_ptr_on_stream(cuda.as_cuda_slice::<i32>()?, start, stream),
        DType::U32 => slice_ptr_on_stream(cuda.as_cuda_slice::<u32>()?, start, stream),
        other => candle_core::bail!("cuTile QSA does not take {other:?} operands"),
    })
}

// Keeps the storage borrow alive for as long as the address is used
macro_rules! cuda_addr {
    ($tensor:expr, $stream:expr, $held:ident, $addr:ident, $guard:ident) => {
        let $held = $tensor.storage_and_layout();
        let Storage::Cuda(cuda) = &*$held.0 else {
            candle_core::bail!("cuTile QSA operands must be CUDA tensors")
        };
        let ($addr, $guard) = device_addr(cuda, $tensor.dtype(), $held.1.start_offset(), $stream)?;
    };
}

fn require(tensor: &Tensor, dtype: DType, name: &str) -> Result<()> {
    if tensor.dtype() != dtype || !tensor.is_contiguous() {
        candle_core::bail!("cuTile QSA `{name}` must be contiguous {dtype:?}");
    }
    Ok(())
}

unsafe fn ptr<T: cutile::DType>(addr: u64) -> DevicePointer<T> {
    DevicePointer::<T>::from_cu_deviceptr(addr as CUdeviceptr)
}

/// Writes block scores into `args.scores: [tokens, score_stride]` F32; entries no top-k reads stay unset.
pub fn cutile_qsa_score(args: &QsaScoreArgs<'_>, dev: &CudaDevice) -> Result<()> {
    score_launch(args, dev, false)
}

fn score_launch(args: &QsaScoreArgs<'_>, dev: &CudaDevice, compile_only: bool) -> Result<()> {
    require(args.q, DType::BF16, "q")?;
    require(args.block_keys, DType::BF16, "block_keys")?;
    require(args.tiles, DType::I32, "tiles")?;
    require(args.scores, DType::F32, "scores")?;
    let (_, heads, dim) = args.q.dims3()?;
    if heads != QSA_INDEX_HEADS || dim != QSA_INDEX_DIM {
        candle_core::bail!("cuTile QSA scoring needs {QSA_INDEX_HEADS} heads of {QSA_INDEX_DIM}");
    }
    let n_tiles = args.tiles.dim(0)?;
    let block_tiles = args.max_blocks.div_ceil(QSA_SCORE_BLOCKS);
    if n_tiles == 0 || block_tiles == 0 {
        return Ok(());
    }
    let score_stride = args.scores.dim(1)?;
    let stream = dev.cuda_stream();
    cuda_addr!(args.q, &stream, q_held, q_addr, _q_guard);
    cuda_addr!(
        args.block_keys,
        &stream,
        block_keys_held,
        block_keys_addr,
        _block_keys_guard
    );
    cuda_addr!(
        args.paged.block_tables,
        &stream,
        tb_held,
        tb_addr,
        _tb_guard
    );
    cuda_addr!(args.tiles, &stream, ti_held, ti_addr, _ti_guard);
    cuda_addr!(args.scores, &stream, sc_held, sc_addr, _sc_guard);
    let block_keys_dim = args.block_keys.dim(1)?;
    let rows = QSA_SCORE_TOKENS * QSA_INDEX_HEADS;
    let generics = vec![
        QSA_SCORE_TOKENS.to_string(),
        QSA_INDEX_HEADS.to_string(),
        rows.to_string(),
        QSA_INDEX_DIM.to_string(),
        QSA_SCORE_BLOCKS.to_string(),
    ];
    let launch = unsafe {
        kernels::qsa_score(
            ptr::<f32>(sc_addr),
            ptr::<tile_bf16>(q_addr),
            ptr::<tile_bf16>(block_keys_addr),
            ptr::<i32>(tb_addr),
            ptr::<i32>(ti_addr),
            score_stride as f32,
            args.paged.max_blocks_per_seq as f32,
            args.ratio as i32,
            args.paged.block_size as i32,
            block_keys_dim as i32,
            args.topk as i32,
            (QSA_INDEX_DIM as f32).sqrt().recip(),
        )
    }
    .generics(generics)
    .grid((n_tiles as u32, block_tiles as u32, 1));
    let cutile_stream = context::stream(dev);
    catch_cutile_panic("QSA score launch", || unsafe {
        if compile_only {
            launch
                .compile_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile qsa score compile: {e:?}")))?;
        } else {
            launch
                .async_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile qsa score launch: {e:?}")))?;
        }
        Ok(())
    })?;
    Ok(())
}

/// Returns the attention output `[tokens, q_heads, 256]` BF16.
pub fn cutile_qsa_attention(args: &QsaAttentionArgs<'_>, dev: &CudaDevice) -> Result<Tensor> {
    attention_launch(args, dev, false)
}

fn attention_launch(
    args: &QsaAttentionArgs<'_>,
    dev: &CudaDevice,
    compile_only: bool,
) -> Result<Tensor> {
    require(args.q, DType::BF16, "q")?;
    require(args.key_cache, DType::BF16, "key cache")?;
    require(args.value_cache, DType::BF16, "value cache")?;
    require(args.selected, DType::I32, "selected")?;
    require(args.n_selected, DType::I32, "n_selected")?;
    require(args.tokens, DType::I32, "tokens")?;
    let (n_tokens, n_q_heads, dim) = args.q.dims3()?;
    let group = n_q_heads / args.n_kv_heads.max(1);
    if dim != QSA_ATTN_HEAD_DIM
        || group > QSA_ATTN_HEAD_ROWS
        || n_q_heads != group * args.n_kv_heads
        || args.key_cache.rank() != 4
    {
        candle_core::bail!("cuTile QSA attention got unsupported shapes");
    }
    let topk = args.selected.dim(1)?;
    let stream = dev.cuda_stream();
    let mut out = unsafe { dev.alloc::<bf16>(n_tokens * n_q_heads * dim)? };
    if n_tokens > 0 {
        cuda_addr!(args.q, &stream, q_held, q_addr, _q_guard);
        cuda_addr!(args.key_cache, &stream, k_held, k_addr, _k_guard);
        cuda_addr!(args.value_cache, &stream, v_held, v_addr, _v_guard);
        cuda_addr!(
            args.paged.block_tables,
            &stream,
            tb_held,
            tb_addr,
            _tb_guard
        );
        cuda_addr!(args.selected, &stream, sel_held, sel_addr, _sel_guard);
        cuda_addr!(args.n_selected, &stream, ns_held, ns_addr, _ns_guard);
        cuda_addr!(args.tokens, &stream, tk_held, tk_addr, _tk_guard);
        let (o_addr, _o_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
        let generics = vec![
            QSA_ATTN_HEAD_ROWS.to_string(),
            QSA_ATTN_HEAD_DIM.to_string(),
            QSA_ATTN_ITEMS.to_string(),
        ];
        let launch = unsafe {
            kernels::qsa_attention(
                ptr::<tile_bf16>(o_addr),
                ptr::<tile_bf16>(q_addr),
                ptr::<tile_bf16>(k_addr),
                ptr::<tile_bf16>(v_addr),
                ptr::<i32>(tb_addr),
                ptr::<i32>(sel_addr),
                ptr::<i32>(ns_addr),
                ptr::<i32>(tk_addr),
                args.paged.max_blocks_per_seq as f32,
                topk as i32,
                args.ratio as i32,
                args.paged.block_size as i32,
                group as i32,
                n_q_heads as i32,
                args.n_kv_heads as i32,
                args.scale,
            )
        }
        .generics(generics)
        .grid((n_tokens as u32, args.n_kv_heads as u32, 1));
        let cutile_stream = context::stream(dev);
        catch_cutile_panic("QSA attention launch", || unsafe {
            if compile_only {
                launch.compile_on(&cutile_stream).map_err(|e| {
                    candle_core::Error::Msg(format!("cutile qsa attention compile: {e:?}"))
                })?;
            } else {
                launch.async_on(&cutile_stream).map_err(|e| {
                    candle_core::Error::Msg(format!("cutile qsa attention launch: {e:?}"))
                })?;
            }
            Ok(())
        })?;
    }
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from_dims(&[n_tokens, n_q_heads, dim]),
    )))
}

/// The integer scalars a model's QSA layers launch with; cuTile compiles per divisibility class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QsaWarmShape {
    pub ratio: usize,
    pub topk: usize,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub block_size: usize,
}

static QSA_WARM_SHAPES: std::sync::Mutex<Vec<QsaWarmShape>> = std::sync::Mutex::new(Vec::new());

/// Ask the startup warmup to compile the QSA kernels for `shape`.
pub fn register_qsa_shape(shape: QsaWarmShape) {
    let mut shapes = QSA_WARM_SHAPES.lock().unwrap();
    if !shapes.contains(&shape) {
        shapes.push(shape);
        super::warmup::mark_dirty();
    }
}

pub struct QsaKernel;

pub(super) static QSA: QsaKernel = QsaKernel;

impl CutileKernel for QsaKernel {
    fn warm(&self, dev: &CudaDevice) -> Result<()> {
        let shapes = QSA_WARM_SHAPES.lock().unwrap().clone();
        if shapes.is_empty() || !jit_available(dev) {
            return Ok(());
        }
        let device = Device::Cuda(dev.clone());
        tracing::info!("Warming cuTile QSA kernels.");
        for shape in shapes {
            let tables = Tensor::zeros((1, 1), DType::I32, &device)?;
            let paged = QsaPaged {
                block_tables: &tables,
                max_blocks_per_seq: 1,
                block_size: shape.block_size,
            };
            let scores = Tensor::zeros((1, shape.topk + 1), DType::F32, &device)?;
            score_launch(
                &QsaScoreArgs {
                    q: &Tensor::zeros((1, QSA_INDEX_HEADS, QSA_INDEX_DIM), DType::BF16, &device)?,
                    block_keys: &Tensor::zeros((1, QSA_INDEX_DIM), DType::BF16, &device)?,
                    paged: &paged,
                    tiles: &Tensor::zeros((1, 4), DType::I32, &device)?,
                    scores: &scores,
                    max_blocks: 1,
                    ratio: shape.ratio,
                    topk: shape.topk,
                },
                dev,
                true,
            )?;
            let cache = Tensor::zeros(
                (1, shape.n_kv_heads, shape.block_size, QSA_ATTN_HEAD_DIM),
                DType::BF16,
                &device,
            )?;
            attention_launch(
                &QsaAttentionArgs {
                    q: &Tensor::zeros(
                        (1, shape.n_q_heads, QSA_ATTN_HEAD_DIM),
                        DType::BF16,
                        &device,
                    )?,
                    key_cache: &cache,
                    value_cache: &cache,
                    paged: &paged,
                    selected: &Tensor::zeros((1, shape.topk), DType::I32, &device)?,
                    n_selected: &Tensor::zeros(1, DType::I32, &device)?,
                    tokens: &Tensor::zeros((1, 2), DType::I32, &device)?,
                    n_kv_heads: shape.n_kv_heads,
                    ratio: shape.ratio,
                    scale: 1.0,
                },
                dev,
                true,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};
    use half::bf16;

    use super::*;

    const BLOCK_SIZE: usize = 32;
    const RATIO: usize = 4;
    const TOPK: usize = 8;
    const KV_HEADS: usize = 2;
    const Q_HEADS: usize = 24;
    const POOL_BLOCKS: usize = 48;
    const MAX_BLOCKS: usize = 12;
    // (tokens, kv length after the step) per sequence; the short ones select every block or none
    const SEQS: [(usize, usize); 4] = [(37, 300), (21, 150), (4, 6), (3, 3)];

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32) / (1u64 << 24) as f32 - 0.5
        }
        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n)
                .map(|_| bf16::from_f32(self.next()).to_f32())
                .collect()
        }
    }

    struct Fixture {
        tables: Vec<i32>,
        // (sequence, position) per flattened token, plus one padding token
        tokens: Vec<(i32, i32)>,
        tiles: Vec<i32>,
    }

    fn fixture() -> Fixture {
        // shuffled, non-identity pages so every lookup goes through the table
        let mut tables = vec![0i32; SEQS.len() * MAX_BLOCKS];
        for (i, entry) in tables.iter_mut().enumerate() {
            *entry = ((i * 7 + 3) % POOL_BLOCKS) as i32;
        }
        let mut tokens = Vec::new();
        let mut tiles = Vec::new();
        for (seq, &(len, kv_len)) in SEQS.iter().enumerate() {
            let first = tokens.len();
            for local in 0..len {
                tokens.push((seq as i32, (kv_len - len + local) as i32));
            }
            for start in (0..len).step_by(QSA_SCORE_TOKENS) {
                let n = QSA_SCORE_TOKENS.min(len - start);
                tiles.extend([
                    (first + start) as i32,
                    seq as i32,
                    n as i32,
                    (kv_len - len + start) as i32,
                ]);
            }
        }
        tokens.push((-1, 0));
        Fixture {
            tables,
            tokens,
            tiles,
        }
    }

    fn slot(tables: &[i32], seq: usize, pos: usize) -> usize {
        tables[seq * MAX_BLOCKS + pos / BLOCK_SIZE] as usize * BLOCK_SIZE + pos % BLOCK_SIZE
    }

    fn to_bf16(data: &[f32], shape: &[usize], dev: &Device) -> Result<Tensor> {
        Tensor::from_vec(
            data.iter().map(|x| bf16::from_f32(*x)).collect::<Vec<_>>(),
            shape,
            dev,
        )
    }

    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_qsa_score_matches_reference() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let fx = fixture();
        let mut rng = Lcg(7);
        let n_tokens = fx.tokens.len();
        let q = rng.vec(n_tokens * QSA_INDEX_HEADS * QSA_INDEX_DIM);
        let block_keys = rng.vec(POOL_BLOCKS * BLOCK_SIZE / RATIO * QSA_INDEX_DIM);
        let max_blocks = SEQS.iter().map(|(_, kv)| kv / RATIO).max().unwrap();
        let stride = max_blocks + 1;
        let scores = Tensor::full(f32::NAN, (n_tokens, stride), &dev)?;
        let tables = Tensor::from_vec(fx.tables.clone(), (SEQS.len(), MAX_BLOCKS), &dev)?;
        let paged = QsaPaged {
            block_tables: &tables,
            max_blocks_per_seq: MAX_BLOCKS,
            block_size: BLOCK_SIZE,
        };
        let n_tiles = fx.tiles.len() / 4;
        cutile_qsa_score(
            &QsaScoreArgs {
                q: &to_bf16(&q, &[n_tokens, QSA_INDEX_HEADS, QSA_INDEX_DIM], &dev)?,
                block_keys: &to_bf16(
                    &block_keys,
                    &[POOL_BLOCKS * BLOCK_SIZE / RATIO, QSA_INDEX_DIM],
                    &dev,
                )?,
                paged: &paged,
                tiles: &Tensor::from_vec(fx.tiles.clone(), (n_tiles, 4), &dev)?,
                scores: &scores,
                max_blocks,
                ratio: RATIO,
                topk: TOPK,
            },
            cuda,
        )?;
        let got = scores.to_vec2::<f32>()?;
        let mut checked = 0;
        for (t, &(seq, pos)) in fx.tokens.iter().enumerate() {
            if seq < 0 {
                continue;
            }
            let nb = (pos as usize + 1) / RATIO;
            if nb <= TOPK {
                continue;
            }
            for j in 0..nb {
                let key = &block_keys
                    [slot(&fx.tables, seq as usize, j * RATIO) / RATIO * QSA_INDEX_DIM..];
                let expected: f32 = (0..QSA_INDEX_HEADS)
                    .map(|h| {
                        let qh = &q[(t * QSA_INDEX_HEADS + h) * QSA_INDEX_DIM..];
                        (0..QSA_INDEX_DIM)
                            .map(|d| qh[d] * key[d])
                            .sum::<f32>()
                            .max(0.0)
                    })
                    .sum::<f32>()
                    / (QSA_INDEX_DIM as f32).sqrt();
                let diff = (got[t][j] - expected).abs();
                assert!(
                    diff <= 1e-3 * expected.abs().max(1.0),
                    "token {t} block {j}: {} vs {expected}",
                    got[t][j]
                );
                checked += 1;
            }
        }
        assert!(checked > 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_qsa_attention_matches_reference() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let fx = fixture();
        let mut rng = Lcg(19);
        let n_tokens = fx.tokens.len();
        let d = QSA_ATTN_HEAD_DIM;
        let q = rng.vec(n_tokens * Q_HEADS * d);
        let cache_len = POOL_BLOCKS * KV_HEADS * BLOCK_SIZE * d;
        let k = rng.vec(cache_len);
        let v = rng.vec(cache_len);
        let mut selected = vec![0i32; n_tokens * TOPK];
        let mut n_selected = vec![0i32; n_tokens];
        for (t, &(seq, pos)) in fx.tokens.iter().enumerate() {
            if seq < 0 {
                continue;
            }
            let nb = (pos as usize + 1) / RATIO;
            let n = nb.min(TOPK);
            n_selected[t] = n as i32;
            for (i, entry) in selected[t * TOPK..t * TOPK + n].iter_mut().enumerate() {
                *entry = ((i * 5 + t) % nb) as i32;
            }
        }
        let tables = Tensor::from_vec(fx.tables.clone(), (SEQS.len(), MAX_BLOCKS), &dev)?;
        let paged = QsaPaged {
            block_tables: &tables,
            max_blocks_per_seq: MAX_BLOCKS,
            block_size: BLOCK_SIZE,
        };
        let tokens = fx
            .tokens
            .iter()
            .flat_map(|(s, p)| [*s, *p])
            .collect::<Vec<_>>();
        let scale = (d as f32).sqrt().recip();
        let cache_shape = [POOL_BLOCKS, KV_HEADS, BLOCK_SIZE, d];
        let out = cutile_qsa_attention(
            &QsaAttentionArgs {
                q: &to_bf16(&q, &[n_tokens, Q_HEADS, d], &dev)?,
                key_cache: &to_bf16(&k, &cache_shape, &dev)?,
                value_cache: &to_bf16(&v, &cache_shape, &dev)?,
                paged: &paged,
                selected: &Tensor::from_vec(selected.clone(), (n_tokens, TOPK), &dev)?,
                n_selected: &Tensor::from_vec(n_selected.clone(), n_tokens, &dev)?,
                tokens: &Tensor::from_vec(tokens, (n_tokens, 2), &dev)?,
                n_kv_heads: KV_HEADS,
                ratio: RATIO,
                scale,
            },
            cuda,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        let group = Q_HEADS / KV_HEADS;
        let mut max_err = 0f32;
        for (t, &(seq, pos)) in fx.tokens.iter().enumerate() {
            for h in 0..Q_HEADS {
                let row = &out[(t * Q_HEADS + h) * d..(t * Q_HEADS + h + 1) * d];
                if seq < 0 {
                    assert!(row.iter().all(|x| *x == 0.0), "padding token must be zero");
                    continue;
                }
                let (seq, pos) = (seq as usize, pos as usize);
                let nb = (pos + 1) / RATIO;
                let mut items = Vec::new();
                for &b in &selected[t * TOPK..t * TOPK + n_selected[t] as usize] {
                    items.extend((0..RATIO).map(|r| b as usize * RATIO + r));
                }
                items.extend(nb * RATIO..=pos);
                let g = h / group;
                let qh = &q[(t * Q_HEADS + h) * d..(t * Q_HEADS + h + 1) * d];
                let base = |p: usize| {
                    (slot(&fx.tables, seq, p) / BLOCK_SIZE * KV_HEADS + g) * BLOCK_SIZE * d
                        + slot(&fx.tables, seq, p) % BLOCK_SIZE * d
                };
                let logits = items
                    .iter()
                    .map(|&p| (0..d).map(|i| qh[i] * k[base(p) + i]).sum::<f32>() * scale)
                    .collect::<Vec<_>>();
                let m = logits.iter().copied().fold(f32::MIN, f32::max);
                let w = logits.iter().map(|x| (x - m).exp()).collect::<Vec<_>>();
                let total: f32 = w.iter().sum();
                for i in 0..d {
                    let expected: f32 = items
                        .iter()
                        .zip(&w)
                        .map(|(&p, wi)| wi * v[base(p) + i])
                        .sum::<f32>()
                        / total;
                    max_err = max_err.max((row[i] - expected).abs());
                }
            }
        }
        assert!(max_err < 2e-3, "max abs error {max_err}");
        Ok(())
    }
}
