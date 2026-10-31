use candle_core::{
    cuda::cudarc::driver::{DeviceRepr, SyncOnDrop},
    CudaDevice, CudaStorage, DType, Device, Result, Storage, Tensor,
};
use cutile::{
    cuda_async::{device_buffer::DevicePointer, device_operation::DeviceOp},
    tile_kernel::TileKernel,
};
use half::{bf16, f16};

use super::warmup::WarmupProgress;
use super::{
    catch_cutile_panic, context, device_multiprocessor_count, jit_available, QsaPaged, QsaWarmShape,
};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

const ITEM_TILE: usize = 64;
const HEAD_TILE: usize = 16;
const PROBABILITY_COMPONENTS: usize = 3;
const SUPPORTED_HEAD_DIMS: &[usize] = &[128, 256];
const WARMUP_DTYPES: &[DType] = &[DType::BF16, DType::F16];
const MAX_INDEXED_ELEMENTS: usize = i32::MAX as usize;
const MAX_EXACT_FLOAT_INTEGER: usize = 1 << 24;
const POINTER_ALIGNMENT: usize = 16;
const MAX_GRID_SECONDARY_DIM: usize = u16::MAX as usize;

#[cutile::module]
mod kernels {
    use cutile::core::*;
    #[cutile::entry(unchecked_accesses = false)]
    pub unsafe fn qsa_attention_split<
        T: ElementType,
        const HP: i32,
        const D: i32,
        const BK: i32,
        const PP: i32,
    >(
        out_ptr: *mut f32,
        stats_ptr: *mut f32,
        q_ptr: *mut T,
        k_ptr: *mut T,
        v_ptr: *mut T,
        tables_ptr: *mut i32,
        selected_ptr: *mut i32,
        n_selected_ptr: *mut i32,
        kv_lens_ptr: *mut i32,
        max_blocks_f: f32,
        topk: i32,
        ratio: i32,
        block_size: i32,
        group: i32,
        n_q_heads: i32,
        n_kv_heads: i32,
        q_len_f: f32,
        splits_f: f32,
        items_per_split_f: f32,
        selection_enabled_f: f32,
        scale: f32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let t: i32 = pid.0;
        let g: i32 = pid.1;
        // Exact f32 integers avoid new divisibility specializations during graph capture.
        let mb_t: Tile<f32, { [1] }> = broadcast_scalar(max_blocks_f, const_shape![1]);
        let mb_i: Tile<i32, { [1] }> = convert_tile(mb_t);
        let mb_s: Tile<i32, { [] }> = mb_i.reshape(const_shape![]);
        let max_blocks: i32 = tile_to_scalar(mb_s);

        let q_len_tile: Tile<f32, { [1] }> = broadcast_scalar(q_len_f, const_shape![1]);
        let q_len_int: Tile<i32, { [1] }> = convert_tile(q_len_tile);
        let q_len_scalar: Tile<i32, { [] }> = q_len_int.reshape(const_shape![]);
        let q_len: i32 = tile_to_scalar(q_len_scalar);
        let splits_tile: Tile<f32, { [1] }> = broadcast_scalar(splits_f, const_shape![1]);
        let splits_int: Tile<i32, { [1] }> = convert_tile(splits_tile);
        let splits_scalar: Tile<i32, { [] }> = splits_int.reshape(const_shape![]);
        let splits: i32 = tile_to_scalar(splits_scalar);
        let items_per_split_tile: Tile<f32, { [1] }> =
            broadcast_scalar(items_per_split_f, const_shape![1]);
        let items_per_split_int: Tile<i32, { [1] }> = convert_tile(items_per_split_tile);
        let items_per_split_scalar: Tile<i32, { [] }> = items_per_split_int.reshape(const_shape![]);
        let items_per_split: i32 = tile_to_scalar(items_per_split_scalar);
        let selection_enabled_tile: Tile<f32, { [1] }> =
            broadcast_scalar(selection_enabled_f, const_shape![1]);
        let selection_enabled_int: Tile<i32, { [1] }> = convert_tile(selection_enabled_tile);
        let selection_enabled_scalar: Tile<i32, { [] }> =
            selection_enabled_int.reshape(const_shape![]);
        let selection_enabled: i32 = tile_to_scalar(selection_enabled_scalar);
        let seq: i32 = t / q_len;
        let split: i32 = pid.2;
        let kl_p0: PointerTile<*mut i32, { [] }> = pointer_to_tile(kv_lens_ptr);
        let kl_p1: PointerTile<*mut i32, { [1] }> = kl_p0.reshape(const_shape![1]);
        let kl_offset: Tile<i32, { [1] }> = broadcast_scalar(seq, const_shape![1]);
        let kl_ptr: PointerTile<*mut i32, { [1] }> = kl_p1.offset_tile(kl_offset);
        let (kv_t, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            kl_ptr,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let kv_s: Tile<i32, { [] }> = kv_t.reshape(const_shape![]);
        let kv_len: i32 = tile_to_scalar(kv_s);
        let pos: i32 = kv_len - q_len + t % q_len;

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
        let o_p0: PointerTile<*mut f32, { [] }> = pointer_to_tile(out_ptr);
        let o_p1: PointerTile<*mut f32, { [1, 1] }> = o_p0.reshape(const_shape![1, 1]);
        let o_p2: PointerTile<*mut f32, { [HP, D] }> = o_p1.broadcast(const_shape![HP, D]);
        let partial_rows: Tile<i32, { [HP] }> = (head0 + iota_h)
            * broadcast_scalar(splits, const_shape![HP])
            + broadcast_scalar(split, const_shape![HP]);
        let partial_off: Tile<i32, { [HP, D] }> = (partial_rows * dim_h)
            .reshape(const_shape![HP, 1])
            .broadcast(const_shape![HP, D])
            + iota_d
                .reshape(const_shape![1, D])
                .broadcast(const_shape![HP, D]);
        let o_ptrs: PointerTile<*mut f32, { [HP, D] }> = o_p2.offset_tile(partial_off);
        let st_p0: PointerTile<*mut f32, { [] }> = pointer_to_tile(stats_ptr);
        let st_p1: PointerTile<*mut f32, { [1] }> = st_p0.reshape(const_shape![1]);
        let st_p2: PointerTile<*mut f32, { [HP] }> = st_p1.broadcast(const_shape![HP]);
        let st_m_offset: Tile<i32, { [HP] }> =
            partial_rows * broadcast_scalar(2i32, const_shape![HP]);
        let st_l_offset: Tile<i32, { [HP] }> =
            st_m_offset + broadcast_scalar(1i32, const_shape![HP]);
        let st_m: PointerTile<*mut f32, { [HP] }> = st_p2.offset_tile(st_m_offset);
        let st_l: PointerTile<*mut f32, { [HP] }> = st_p2.offset_tile(st_l_offset);
        let zero_hd: Tile<f32, { [HP, D] }> = constant(0.0f32, const_shape![HP, D]);

        if pos >= 0 {
            let q_p0: PointerTile<*mut T, { [] }> = pointer_to_tile(q_ptr);
            let q_p1: PointerTile<*mut T, { [1, 1] }> = q_p0.reshape(const_shape![1, 1]);
            let q_p2: PointerTile<*mut T, { [HP, D] }> = q_p1.broadcast(const_shape![HP, D]);
            let q_ptrs: PointerTile<*mut T, { [HP, D] }> = q_p2.offset_tile(hd_off);
            let (q_raw, _): (Tile<T, { [HP, D] }>, Token) = load_ptr_tko(
                q_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(head_mask),
                None,
                None,
                Latency::<0>,
            );
            let zero_q: Tile<T, { [HP, D] }> = constant(T::ZERO, const_shape![HP, D]);
            let q_tile: Tile<T, { [HP, D] }> = select(head_mask, q_raw, zero_q);

            let nb: i32 = (pos + 1) / ratio;
            let mut n_block_items: i32 = 0;
            let mut tail_start: i32 = 0;
            if selection_enabled != 0 {
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
                n_block_items = n_sel * ratio;
                tail_start = nb * ratio;
            }
            let n_items: i32 = n_block_items + pos + 1 - tail_start;
            let n_chunks: i32 = (items_per_split + BK - 1) / BK;
            let split_end: i32 = (split + 1) * items_per_split;

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
            let k_p0: PointerTile<*mut T, { [] }> = pointer_to_tile(k_ptr);
            let k_p1: PointerTile<*mut T, { [1, 1] }> = k_p0.reshape(const_shape![1, 1]);
            let v_p0: PointerTile<*mut T, { [] }> = pointer_to_tile(v_ptr);
            let v_p1: PointerTile<*mut T, { [1, 1] }> = v_p0.reshape(const_shape![1, 1]);
            let zero_kd: Tile<T, { [BK, D] }> = constant(T::ZERO, const_shape![BK, D]);
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
                    broadcast_scalar(split * items_per_split + chunk * BK, const_shape![BK])
                        + iota_k;
                let valid: Tile<bool, { [BK] }> = lt_tile(it, items_k)
                    & lt_tile(it, broadcast_scalar(split_end, const_shape![BK]));
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
                let k_p2: PointerTile<*mut T, { [BK, D] }> = k_p1.broadcast(const_shape![BK, D]);
                let k_ptrs: PointerTile<*mut T, { [BK, D] }> = k_p2.offset_tile(kv_off);
                let v_p2: PointerTile<*mut T, { [BK, D] }> = v_p1.broadcast(const_shape![BK, D]);
                let v_ptrs: PointerTile<*mut T, { [BK, D] }> = v_p2.offset_tile(kv_off);
                let (k_raw, _): (Tile<T, { [BK, D] }>, Token) = load_ptr_tko(
                    k_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(kv_mask),
                    None,
                    None,
                    Latency::<0>,
                );
                let (v_raw, _): (Tile<T, { [BK, D] }>, Token) = load_ptr_tko(
                    v_ptrs,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(kv_mask),
                    None,
                    None,
                    Latency::<0>,
                );
                let k_tile: Tile<T, { [BK, D] }> = select(kv_mask, k_raw, zero_kd);
                let v_tile: Tile<T, { [BK, D] }> = select(kv_mask, v_raw, zero_kd);
                let k_t: Tile<T, { [D, BK] }> = permute(k_tile, transpose);
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
                // Residual components avoid a single half-precision rounding of softmax probabilities.
                let probs_b: Tile<T, { [HP, BK] }> = convert_tile(probs);
                let acc_scaled: Tile<f32, { [HP, D] }> = acc * corr_hd;
                let mut acc_new: Tile<f32, { [HP, D] }> = mmaf(probs_b, v_tile, acc_scaled);
                if PP > 1 {
                    let probs_hi: Tile<f32, { [HP, BK] }> = convert_tile(probs_b);
                    let rem: Tile<f32, { [HP, BK] }> = probs - probs_hi;
                    let probs_lo: Tile<T, { [HP, BK] }> = convert_tile(rem);
                    acc_new = mmaf(probs_lo, v_tile, acc_new);
                    if PP > 2 {
                        let lo_f: Tile<f32, { [HP, BK] }> = convert_tile(probs_lo);
                        let probs_tail: Tile<T, { [HP, BK] }> = convert_tile(rem - lo_f);
                        acc_new = mmaf(probs_tail, v_tile, acc_new);
                    }
                }
                m = m_new;
                l = l_new;
                acc = acc_new;
            }
            store_ptr_tko(
                o_ptrs,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(head_mask),
                None,
                Latency::<0>,
            );
            store_ptr_tko(
                st_m,
                m,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(head_ok),
                None,
                Latency::<0>,
            );
            store_ptr_tko(
                st_l,
                l,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(head_ok),
                None,
                Latency::<0>,
            );
        } else {
            let zero_h: Tile<f32, { [HP] }> = constant(0.0f32, const_shape![HP]);
            store_ptr_tko(
                o_ptrs,
                zero_hd,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(head_mask),
                None,
                Latency::<0>,
            );
            store_ptr_tko(
                st_m,
                zero_h,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(head_ok),
                None,
                Latency::<0>,
            );
            store_ptr_tko(
                st_l,
                zero_h,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(head_ok),
                None,
                Latency::<0>,
            );
        }
    }
}

pub struct QsaDecodeArgs<'a> {
    pub q: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub paged: &'a QsaPaged<'a>,
    pub kv_lens: &'a Tensor,
    pub selection: Option<(&'a Tensor, &'a Tensor)>,
    pub q_len: usize,
    pub n_kv_heads: usize,
    pub ratio: usize,
    pub topk: usize,
    pub scale: f32,
}

struct DecodeShape {
    tokens: usize,
    heads: usize,
    head_dim: usize,
    items_per_split: usize,
    splits: usize,
}

fn indexed_product(dimensions: &[usize]) -> Option<usize> {
    dimensions.iter().try_fold(1usize, |product, dimension| {
        product
            .checked_mul(*dimension)
            .filter(|product| *product <= MAX_INDEXED_ELEMENTS)
    })
}

fn aligned_operand(tensor: &Tensor) -> bool {
    tensor.is_contiguous()
        && tensor.elem_count() <= MAX_INDEXED_ELEMENTS
        && tensor
            .layout()
            .start_offset()
            .is_multiple_of(POINTER_ALIGNMENT / tensor.dtype().size_in_bytes())
}

fn decode_shape(args: &QsaDecodeArgs<'_>, dev: &CudaDevice) -> Result<Option<DecodeShape>> {
    let (tokens, heads, head_dim) = args.q.dims3()?;
    let dtype = args.q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16)
        || !SUPPORTED_HEAD_DIMS.contains(&head_dim)
        || args.n_kv_heads == 0
        || args.n_kv_heads > MAX_GRID_SECONDARY_DIM
        || heads == 0
        || !heads.is_multiple_of(args.n_kv_heads)
        || heads / args.n_kv_heads > HEAD_TILE
        || args.q_len == 0
        || !tokens.is_multiple_of(args.q_len)
        || args.ratio == 0
        || args.ratio > MAX_INDEXED_ELEMENTS
        || args.topk > MAX_INDEXED_ELEMENTS
        || args.q_len > MAX_EXACT_FLOAT_INTEGER
        || args.paged.max_blocks_per_seq > MAX_EXACT_FLOAT_INTEGER
        || args.paged.block_size == 0
        || args.key_cache.rank() != 4
        || args.key_cache.dims().get(1..)
            != Some(&[args.n_kv_heads, args.paged.block_size, head_dim][..])
        || args.key_cache.dtype() != dtype
        || args.value_cache.shape() != args.key_cache.shape()
        || args.value_cache.dtype() != dtype
        || !matches!(args.kv_lens.dtype(), DType::I32 | DType::U32)
        || !matches!(args.paged.block_tables.dtype(), DType::I32 | DType::U32)
        || !jit_available(dev)
    {
        return Ok(None);
    }
    if args.kv_lens.elem_count() != tokens / args.q_len
        || args.paged.block_tables.dims() != [tokens / args.q_len, args.paged.max_blocks_per_seq]
    {
        return Ok(None);
    }
    let operands = [
        args.q,
        args.key_cache,
        args.value_cache,
        args.kv_lens,
        args.paged.block_tables,
    ];
    if operands.iter().any(|tensor| {
        !aligned_operand(tensor) || tensor.device().location() != args.q.device().location()
    }) {
        return Ok(None);
    }
    if let Some((selected, counts)) = args.selection {
        if selected.dims() != [tokens, args.topk] || counts.dims() != [tokens] {
            return Ok(None);
        }
        if [selected, counts].iter().any(|tensor| {
            tensor.dtype() != DType::I32
                || !aligned_operand(tensor)
                || tensor.device().location() != args.q.device().location()
        }) {
            return Ok(None);
        }
    }
    let Some(max_context) =
        indexed_product(&[args.paged.max_blocks_per_seq, args.paged.block_size])
    else {
        return Ok(None);
    };
    let max_items = if args.selection.is_some() {
        args.topk
            .checked_mul(args.ratio)
            .and_then(|items| items.checked_add(args.ratio - 1))
    } else {
        Some(max_context)
    };
    let Some(max_items) = max_items
        .filter(|items| *items > 0 && *items <= MAX_INDEXED_ELEMENTS.saturating_sub(max_context))
    else {
        return Ok(None);
    };
    let Some(blocks) = indexed_product(&[tokens, args.n_kv_heads]).filter(|blocks| *blocks > 0)
    else {
        return Ok(None);
    };
    let target_splits = device_multiprocessor_count(dev).div_ceil(blocks);
    let items_per_split = max_items.div_ceil(target_splits).div_ceil(ITEM_TILE) * ITEM_TILE;
    let splits = max_items.div_ceil(items_per_split);
    let Some(padded_rows) = tokens
        .checked_mul(heads)
        .and_then(|rows| rows.checked_add(HEAD_TILE))
    else {
        return Ok(None);
    };
    if items_per_split > MAX_EXACT_FLOAT_INTEGER
        || splits > MAX_GRID_SECONDARY_DIM
        || indexed_product(&[splits, items_per_split]).is_none()
        || indexed_product(&[padded_rows, splits, head_dim]).is_none()
        || indexed_product(&[padded_rows, splits, 2]).is_none()
    {
        return Ok(None);
    }
    Ok(Some(DecodeShape {
        tokens,
        heads,
        head_dim,
        items_per_split,
        splits,
    }))
}

pub fn try_cutile_qsa_decode(args: &QsaDecodeArgs<'_>, dev: &CudaDevice) -> Result<Option<Tensor>> {
    decode_launch(args, dev, false)
}

fn decode_launch(
    args: &QsaDecodeArgs<'_>,
    dev: &CudaDevice,
    compile_only: bool,
) -> Result<Option<Tensor>> {
    let Some(shape) = decode_shape(args, dev)? else {
        return Ok(None);
    };
    match args.q.dtype() {
        DType::BF16 => launch::<bf16>(args, dev, &shape, compile_only).map(Some),
        DType::F16 => launch::<f16>(args, dev, &shape, compile_only).map(Some),
        _ => unreachable!(),
    }
}

fn device_addr<'a>(
    storage: &'a CudaStorage,
    dtype: DType,
    offset: usize,
    stream: &'a candle_core::cuda::cudarc::driver::CudaStream,
) -> Result<(u64, SyncOnDrop<'a>)> {
    Ok(match dtype {
        DType::BF16 => slice_ptr_on_stream(storage.as_cuda_slice::<bf16>()?, offset, stream),
        DType::F16 => slice_ptr_on_stream(storage.as_cuda_slice::<f16>()?, offset, stream),
        DType::I32 => slice_ptr_on_stream(storage.as_cuda_slice::<i32>()?, offset, stream),
        DType::U32 => slice_ptr_on_stream(storage.as_cuda_slice::<u32>()?, offset, stream),
        _ => unreachable!(),
    })
}

macro_rules! cuda_addr {
    ($tensor:expr, $stream:expr, $held:ident, $addr:ident, $guard:ident) => {
        let $held = $tensor.storage_and_layout();
        let Storage::Cuda(storage) = &*$held.0 else {
            candle_core::bail!("QSA decode requires CUDA operands");
        };
        let ($addr, $guard) =
            device_addr(storage, $tensor.dtype(), $held.1.start_offset(), $stream)?;
    };
}

fn launch<T>(
    args: &QsaDecodeArgs<'_>,
    dev: &CudaDevice,
    shape: &DecodeShape,
    compile_only: bool,
) -> Result<Tensor>
where
    T: DeviceRepr + candle_core::cuda_backend::CudaDType + cutile::DType + Send,
{
    unsafe extern "C" {
        fn launch_sparse_attention_reduce(
            partial: *const f32,
            stats: *const f32,
            out: *mut std::ffi::c_void,
            rows: i32,
            splits: i32,
            head_dim: i32,
            dtype: i32,
            stream: *mut std::ffi::c_void,
        );
    }
    let stream = dev.cuda_stream();
    let mut out = unsafe { dev.alloc::<T>(shape.tokens * shape.heads * shape.head_dim)? };
    let mut partial =
        unsafe { dev.alloc::<f32>(shape.tokens * shape.heads * shape.splits * shape.head_dim)? };
    let mut stats = unsafe { dev.alloc::<f32>(shape.tokens * shape.heads * shape.splits * 2)? };
    let (selected, counts) = args.selection.unwrap_or((args.kv_lens, args.kv_lens));
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
    cuda_addr!(args.kv_lens, &stream, lens_held, lens_addr, _lens_guard);
    cuda_addr!(selected, &stream, sel_held, sel_addr, _sel_guard);
    cuda_addr!(counts, &stream, count_held, count_addr, _count_guard);
    let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
    let (partial_addr, partial_guard) = slice_ptr_mut_on_stream(&mut partial, 0, &stream);
    let (stats_addr, stats_guard) = slice_ptr_mut_on_stream(&mut stats, 0, &stream);
    let cutile_stream = context::stream(dev);
    let dtype = if args.q.dtype() == DType::F16 {
        "f16"
    } else {
        "bf16"
    };
    let kernel = unsafe {
        kernels::qsa_attention_split(
            DevicePointer::<f32>::from_cu_deviceptr(partial_addr),
            DevicePointer::<f32>::from_cu_deviceptr(stats_addr),
            DevicePointer::<T>::from_cu_deviceptr(q_addr),
            DevicePointer::<T>::from_cu_deviceptr(k_addr),
            DevicePointer::<T>::from_cu_deviceptr(v_addr),
            DevicePointer::<i32>::from_cu_deviceptr(tb_addr),
            DevicePointer::<i32>::from_cu_deviceptr(sel_addr),
            DevicePointer::<i32>::from_cu_deviceptr(count_addr),
            DevicePointer::<i32>::from_cu_deviceptr(lens_addr),
            args.paged.max_blocks_per_seq as f32,
            args.topk as i32,
            args.ratio as i32,
            args.paged.block_size as i32,
            (shape.heads / args.n_kv_heads) as i32,
            shape.heads as i32,
            args.n_kv_heads as i32,
            args.q_len as f32,
            shape.splits as f32,
            shape.items_per_split as f32,
            if args.selection.is_some() { 1.0 } else { 0.0 },
            args.scale,
        )
    }
    .generics(vec![
        dtype.to_owned(),
        HEAD_TILE.to_string(),
        shape.head_dim.to_string(),
        ITEM_TILE.to_string(),
        PROBABILITY_COMPONENTS.to_string(),
    ])
    .grid((
        shape.tokens as u32,
        args.n_kv_heads as u32,
        shape.splits as u32,
    ));
    catch_cutile_panic("QSA decode", || unsafe {
        if compile_only {
            kernel.compile_on(&cutile_stream).map_err(|error| {
                candle_core::Error::msg(format!("cuTile QSA decode compile: {error:?}"))
            })?;
        } else {
            kernel.async_on(&cutile_stream).map_err(|error| {
                candle_core::Error::msg(format!("cuTile QSA decode launch: {error:?}"))
            })?;
            launch_sparse_attention_reduce(
                partial_addr as *const f32,
                stats_addr as *const f32,
                out_addr as *mut _,
                (shape.tokens * shape.heads) as i32,
                shape.splits as i32,
                shape.head_dim as i32,
                i32::from(args.q.dtype() != DType::F16),
                stream.cu_stream() as *mut _,
            );
        }
        Ok(())
    })?;
    drop((out_guard, partial_guard, stats_guard));
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        args.q.shape().clone(),
    )))
}

pub(super) fn decode_warmup_steps() -> usize {
    SUPPORTED_HEAD_DIMS.len() * WARMUP_DTYPES.len()
}

pub(super) fn warm_decode(
    shape: QsaWarmShape,
    dev: &CudaDevice,
    progress: &WarmupProgress,
) -> Result<()> {
    let device = Device::Cuda(dev.clone());
    let tables = Tensor::zeros((1, 1), DType::I32, &device)?;
    let kv_lens = Tensor::ones(1, DType::U32, &device)?;
    let selected = Tensor::zeros((1, shape.topk), DType::I32, &device)?;
    let counts = Tensor::zeros(1, DType::I32, &device)?;
    let paged = QsaPaged {
        block_tables: &tables,
        max_blocks_per_seq: 1,
        block_size: shape.block_size,
    };
    for (head_dim, dtype) in SUPPORTED_HEAD_DIMS
        .iter()
        .copied()
        .flat_map(|dim| WARMUP_DTYPES.iter().copied().map(move |dtype| (dim, dtype)))
    {
        let cache = Tensor::zeros(
            (1, shape.n_kv_heads, shape.block_size, head_dim),
            dtype,
            &device,
        )?;
        progress.run(format!("decode {dtype:?} head_dim={head_dim}"), || {
            decode_launch(
                &QsaDecodeArgs {
                    q: &Tensor::zeros((1, shape.n_q_heads, head_dim), dtype, &device)?,
                    key_cache: &cache,
                    value_cache: &cache,
                    paged: &paged,
                    kv_lens: &kv_lens,
                    selection: Some((&selected, &counts)),
                    q_len: 1,
                    n_kv_heads: shape.n_kv_heads,
                    ratio: shape.ratio,
                    topk: shape.topk,
                    scale: 1.0,
                },
                dev,
                true,
            )
        })?;
    }
    Ok(())
}
