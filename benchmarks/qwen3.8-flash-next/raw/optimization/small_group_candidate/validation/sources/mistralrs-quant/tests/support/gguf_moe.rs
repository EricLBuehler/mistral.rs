#![allow(
    clippy::too_many_arguments,
    reason = "cuTile kernel arguments follow the device ABI"
)]

use candle_core::cuda::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{CudaStorage, DType, Result, Storage, Tensor};
use cutile::cuda_async::device_buffer::DevicePointer;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tile_kernel::TileKernel;
use half::{bf16, f16};

use mistralrs_quant::cutile::context;

const DEFAULT_BM: i32 = 16;
const DEFAULT_BN: i32 = 64;
const DEFAULT_BK: i32 = 128;

fn catch_cutile_panic<T>(
    operation: &str,
    f: impl FnOnce() -> candle_core::Result<T>,
) -> candle_core::Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let message = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic");
            candle_core::bail!("cuTile {operation} panicked: {message}")
        }
    }
}

#[cutile::module]
mod kernels {
    use cutile::core::*;

    const Q4K_VALUES: i32 = 256;
    const Q4K_BYTES: i32 = 144;
    const Q4_1_VALUES: i32 = 32;
    const Q4_1_BYTES: i32 = 20;
    const Q4K_SCALE_GROUPS: i32 = 8;
    const Q4K_LOW_SCALE_GROUPS: i32 = 4;
    const Q4K_GROUP_PAIR_VALUES: i32 = 64;
    const Q4K_QUANTS_OFFSET: i32 = 16;
    const QUANT_HEADERS_BYTES: i32 = 4;
    const Q4_1_HALF_VALUES: i32 = 16;
    const NIBBLE_MASK: i32 = 15;
    const NIBBLE_BITS: i32 = 4;
    const SCALE_MASK: i32 = 63;
    const SCALE_BITS: i32 = 6;

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn projection<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const GB: i32,
        const FORMAT: i32,
        const TOP_K: i32,
    >(
        output: *mut f32,
        input: *mut bf16,
        weight_bytes: *mut u8,
        weight_halves: *mut f16,
        sorted_ids: *mut i32,
        expert_ids: *mut i32,
        padded_count: *mut i32,
        n_size: i32,
        k_size: i32,
        assignments: i32,
        padded_capacity: i32,
    ) {
        let pid = get_tile_block_id().0;
        let n_tiles = ceil_div(n_size, BN);
        let m_tile = pid / n_tiles;
        let n_tile = pid % n_tiles;
        let padded_ptr: PointerTile<*mut i32, { [] }> = pointer_to_tile(padded_count);
        let padded_ptr: PointerTile<*mut i32, { [1] }> = padded_ptr.reshape(const_shape![1]);
        let padded_ptr: PointerTile<*mut i32, { [1] }> = padded_ptr.broadcast(const_shape![1]);
        let padded_ptr = padded_ptr.offset_tile(broadcast_scalar(0i32, const_shape![1]));
        let (padded, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
            padded_ptr,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        let padded: i32 = tile_to_scalar(padded.reshape(const_shape![]));
        if m_tile * BM < padded {
            let expert_ptr: PointerTile<*mut i32, { [] }> = pointer_to_tile(expert_ids);
            let expert_ptr: PointerTile<*mut i32, { [1] }> = expert_ptr.reshape(const_shape![1]);
            let expert_ptr: PointerTile<*mut i32, { [1] }> = expert_ptr.broadcast(const_shape![1]);
            let expert_ptr = expert_ptr.offset_tile(broadcast_scalar(m_tile, const_shape![1]));
            let (expert, _): (Tile<i32, { [1] }>, Token) = load_ptr_tko(
                expert_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            let expert: i32 = tile_to_scalar(expert.reshape(const_shape![]));
            let positions: Tile<i32, { [BM] }> =
                iota(const_shape![BM]) + broadcast_scalar(m_tile * BM, const_shape![BM]);
            let ids_ptr: PointerTile<*mut i32, { [] }> = pointer_to_tile(sorted_ids);
            let ids_ptr: PointerTile<*mut i32, { [1] }> = ids_ptr.reshape(const_shape![1]);
            let ids_ptr: PointerTile<*mut i32, { [BM] }> = ids_ptr.broadcast(const_shape![BM]);
            let ids_ptr = ids_ptr.offset_tile(positions);
            let capacity: Tile<i32, { [BM] }> = broadcast_scalar(padded_capacity, const_shape![BM]);
            let ids_in_bounds: Tile<bool, { [BM] }> = lt_tile(positions, capacity);
            let (ids, _): (Tile<i32, { [BM] }>, Token) = load_ptr_tko(
                ids_ptr,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(ids_in_bounds),
                Some(assignments),
                None,
                Latency::<0>,
            );
            let valid_rows: Tile<bool, { [BM] }> =
                lt_tile(ids, broadcast_scalar(assignments, const_shape![BM]));
            let rows: Tile<i32, { [BM] }> = select(
                valid_rows,
                ids / broadcast_scalar(TOP_K, const_shape![BM]),
                broadcast_scalar(0i32, const_shape![BM]),
            );
            let columns: Tile<i32, { [BN] }> =
                iota(const_shape![BN]) + broadcast_scalar(n_tile * BN, const_shape![BN]);
            let valid_columns: Tile<bool, { [BN] }> =
                lt_tile(columns, broadcast_scalar(n_size, const_shape![BN]));
            let safe_columns: Tile<i32, { [BN] }> = select(
                valid_columns,
                columns,
                broadcast_scalar(0i32, const_shape![BN]),
            );
            let k_iota: Tile<i32, { [BK] }> = iota(const_shape![BK]);
            let input_base: Tile<i64, { [BM] }> = exti(rows);
            let input_stride: Tile<i64, { [BM] }> =
                exti(broadcast_scalar(k_size, const_shape![BM]));
            let input_base: Tile<i64, { [BM] }> = input_base * input_stride;
            let weight_row: Tile<i64, { [BN] }> = exti(safe_columns);
            let expert64: Tile<i64, { [BN] }> = exti(broadcast_scalar(expert, const_shape![BN]));
            let n_size64: Tile<i64, { [BN] }> = exti(broadcast_scalar(n_size, const_shape![BN]));
            let weight_row: Tile<i64, { [BN] }> = weight_row + expert64 * n_size64;
            let block_values = if FORMAT == 0 { Q4K_VALUES } else { Q4_1_VALUES };
            let weight_stride: Tile<i64, { [BN] }> =
                exti(broadcast_scalar(k_size / block_values, const_shape![BN]));
            let weight_base: Tile<i64, { [BN] }> = weight_row * weight_stride;
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            if expert >= 0 {
                for kt in 0..ceil_div(k_size, BK) {
                    let k: Tile<i32, { [BK] }> =
                        k_iota + broadcast_scalar(kt * BK, const_shape![BK]);
                    let valid_k: Tile<bool, { [BK] }> =
                        lt_tile(k, broadcast_scalar(k_size, const_shape![BK]));
                    let safe_k: Tile<i32, { [BK] }> =
                        select(valid_k, k, broadcast_scalar(0i32, const_shape![BK]));
                    let k64: Tile<i64, { [BK] }> = exti(safe_k);
                    let a_offsets: Tile<i64, { [BM, BK] }> = input_base
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, BK])
                        + k64
                            .reshape(const_shape![1, BK])
                            .broadcast(const_shape![BM, BK]);
                    let a_mask: Tile<bool, { [BM, BK] }> = valid_rows
                        .reshape(const_shape![BM, 1])
                        .broadcast(const_shape![BM, BK])
                        & valid_k
                            .reshape(const_shape![1, BK])
                            .broadcast(const_shape![BM, BK]);
                    let a_ptr: PointerTile<*mut bf16, { [] }> = pointer_to_tile(input);
                    let a_ptr: PointerTile<*mut bf16, { [1, 1] }> =
                        a_ptr.reshape(const_shape![1, 1]);
                    let a_ptr: PointerTile<*mut bf16, { [BM, BK] }> =
                        a_ptr.broadcast(const_shape![BM, BK]);
                    let a_ptr = a_ptr.offset_tile(a_offsets);
                    let (a, _): (Tile<bf16, { [BM, BK] }>, Token) = load_ptr_tko(
                        a_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        Some(a_mask),
                        None,
                        None,
                        Latency::<0>,
                    );
                    let a_zero: Tile<bf16, { [BM, BK] }> =
                        constant(bf16::ZERO, const_shape![BM, BK]);
                    let a: Tile<bf16, { [BM, BK] }> = select(a_mask, a, a_zero);
                    let groups: Tile<i32, { [GB] }> =
                        iota(const_shape![GB]) + broadcast_scalar(kt * GB, const_shape![GB]);
                    let group_count: Tile<i32, { [GB] }> =
                        broadcast_scalar(k_size / Q4_1_VALUES, const_shape![GB]);
                    let valid_groups: Tile<bool, { [GB] }> = lt_tile(groups, group_count);
                    let groups: Tile<i32, { [GB] }> = select(
                        valid_groups,
                        groups,
                        broadcast_scalar(0i32, const_shape![GB]),
                    );
                    let blocks: Tile<i32, { [GB] }> = if FORMAT == 0 {
                        groups / broadcast_scalar(Q4K_SCALE_GROUPS, const_shape![GB])
                    } else {
                        groups
                    };
                    let blocks: Tile<i64, { [GB] }> = exti(blocks);
                    let bytes_per_block = if FORMAT == 0 { Q4K_BYTES } else { Q4_1_BYTES };
                    let block_bytes: Tile<i64, { [BN, GB] }> =
                        exti(broadcast_scalar(bytes_per_block, const_shape![BN, GB]));
                    let byte_base: Tile<i64, { [BN, GB] }> = (weight_base
                        .reshape(const_shape![BN, 1])
                        .broadcast(const_shape![BN, GB])
                        + blocks
                            .reshape(const_shape![1, GB])
                            .broadcast(const_shape![BN, GB]))
                        * block_bytes;
                    let half_base: Tile<i64, { [BN, GB] }> =
                        byte_base / broadcast_scalar(2i64, const_shape![BN, GB]);
                    let d_ptr: PointerTile<*mut f16, { [] }> = pointer_to_tile(weight_halves);
                    let d_ptr: PointerTile<*mut f16, { [1, 1] }> =
                        d_ptr.reshape(const_shape![1, 1]);
                    let d_ptr: PointerTile<*mut f16, { [BN, GB] }> =
                        d_ptr.broadcast(const_shape![BN, GB]);
                    let d_ptr = d_ptr.offset_tile(half_base);
                    let (d, _): (Tile<f16, { [BN, GB] }>, Token) = load_ptr_tko(
                        d_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let m_ptr: PointerTile<*mut f16, { [] }> = pointer_to_tile(weight_halves);
                    let m_ptr: PointerTile<*mut f16, { [1, 1] }> =
                        m_ptr.reshape(const_shape![1, 1]);
                    let m_ptr: PointerTile<*mut f16, { [BN, GB] }> =
                        m_ptr.broadcast(const_shape![BN, GB]);
                    let m_ptr =
                        m_ptr.offset_tile(half_base + broadcast_scalar(1i64, const_shape![BN, GB]));
                    let (m, _): (Tile<f16, { [BN, GB] }>, Token) = load_ptr_tko(
                        m_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let d: Tile<f32, { [BN, GB] }> = convert_tile(d);
                    let m: Tile<f32, { [BN, GB] }> = convert_tile(m);
                    let local_group: Tile<i32, { [GB] }> =
                        groups & broadcast_scalar(Q4K_SCALE_GROUPS - 1, const_shape![GB]);
                    let scale: Tile<f32, { [BN, GB] }> = if FORMAT == 0 {
                        let lower: Tile<bool, { [GB] }> = lt_tile(
                            local_group,
                            broadcast_scalar(Q4K_LOW_SCALE_GROUPS, const_shape![GB]),
                        );
                        let index: Tile<i32, { [GB] }> = local_group
                            & broadcast_scalar(Q4K_LOW_SCALE_GROUPS - 1, const_shape![GB]);
                        let index: Tile<i64, { [GB] }> = exti(index);
                        let offsets: Tile<i64, { [BN, GB] }> = byte_base
                            + broadcast_scalar(4i64, const_shape![BN, GB])
                            + index
                                .reshape(const_shape![1, GB])
                                .broadcast(const_shape![BN, GB]);
                        let scale_low_ptr: PointerTile<*mut u8, { [] }> =
                            pointer_to_tile(weight_bytes);
                        let scale_low_ptr: PointerTile<*mut u8, { [1, 1] }> =
                            scale_low_ptr.reshape(const_shape![1, 1]);
                        let scale_low_ptr: PointerTile<*mut u8, { [BN, GB] }> =
                            scale_low_ptr.broadcast(const_shape![BN, GB]);
                        let scale_low_ptr = scale_low_ptr.offset_tile(offsets);
                        let (scale_low, _): (Tile<u8, { [BN, GB] }>, Token) = load_ptr_tko(
                            scale_low_ptr,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        let scale_high_ptr: PointerTile<*mut u8, { [] }> =
                            pointer_to_tile(weight_bytes);
                        let scale_high_ptr: PointerTile<*mut u8, { [1, 1] }> =
                            scale_high_ptr.reshape(const_shape![1, 1]);
                        let scale_high_ptr: PointerTile<*mut u8, { [BN, GB] }> =
                            scale_high_ptr.broadcast(const_shape![BN, GB]);
                        let scale_high_ptr = scale_high_ptr
                            .offset_tile(offsets + broadcast_scalar(8i64, const_shape![BN, GB]));
                        let (scale_high, _): (Tile<u8, { [BN, GB] }>, Token) = load_ptr_tko(
                            scale_high_ptr,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        let scale_low: Tile<i32, { [BN, GB] }> = exti(scale_low);
                        let scale_high: Tile<i32, { [BN, GB] }> = exti(scale_high);
                        let lower: Tile<bool, { [BN, GB] }> = lower
                            .reshape(const_shape![1, GB])
                            .broadcast(const_shape![BN, GB]);
                        let scale: Tile<i32, { [BN, GB] }> = select(
                            lower,
                            scale_low & broadcast_scalar(SCALE_MASK, const_shape![BN, GB]),
                            (scale_high & broadcast_scalar(NIBBLE_MASK, const_shape![BN, GB]))
                                | ((scale_low
                                    >> broadcast_scalar(SCALE_BITS, const_shape![BN, GB]))
                                    << broadcast_scalar(NIBBLE_BITS, const_shape![BN, GB])),
                        );
                        let scale: Tile<f32, { [BN, GB] }> = convert_tile(scale);
                        d * scale
                    } else {
                        d
                    };
                    let bias: Tile<f32, { [BN, GB] }> = if FORMAT == 0 {
                        let lower: Tile<bool, { [GB] }> = lt_tile(
                            local_group,
                            broadcast_scalar(Q4K_LOW_SCALE_GROUPS, const_shape![GB]),
                        );
                        let index: Tile<i32, { [GB] }> = local_group
                            & broadcast_scalar(Q4K_LOW_SCALE_GROUPS - 1, const_shape![GB]);
                        let index: Tile<i64, { [GB] }> = exti(index);
                        let offsets: Tile<i64, { [BN, GB] }> = byte_base
                            + broadcast_scalar(8i64, const_shape![BN, GB])
                            + index
                                .reshape(const_shape![1, GB])
                                .broadcast(const_shape![BN, GB]);
                        let min_low_ptr: PointerTile<*mut u8, { [] }> =
                            pointer_to_tile(weight_bytes);
                        let min_low_ptr: PointerTile<*mut u8, { [1, 1] }> =
                            min_low_ptr.reshape(const_shape![1, 1]);
                        let min_low_ptr: PointerTile<*mut u8, { [BN, GB] }> =
                            min_low_ptr.broadcast(const_shape![BN, GB]);
                        let min_low_ptr = min_low_ptr.offset_tile(offsets);
                        let (min_low, _): (Tile<u8, { [BN, GB] }>, Token) = load_ptr_tko(
                            min_low_ptr,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        let min_high_ptr: PointerTile<*mut u8, { [] }> =
                            pointer_to_tile(weight_bytes);
                        let min_high_ptr: PointerTile<*mut u8, { [1, 1] }> =
                            min_high_ptr.reshape(const_shape![1, 1]);
                        let min_high_ptr: PointerTile<*mut u8, { [BN, GB] }> =
                            min_high_ptr.broadcast(const_shape![BN, GB]);
                        let min_high_ptr = min_high_ptr
                            .offset_tile(offsets + broadcast_scalar(4i64, const_shape![BN, GB]));
                        let (min_high, _): (Tile<u8, { [BN, GB] }>, Token) = load_ptr_tko(
                            min_high_ptr,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        let min_low: Tile<i32, { [BN, GB] }> = exti(min_low);
                        let min_high: Tile<i32, { [BN, GB] }> = exti(min_high);
                        let lower: Tile<bool, { [BN, GB] }> = lower
                            .reshape(const_shape![1, GB])
                            .broadcast(const_shape![BN, GB]);
                        let minimum: Tile<i32, { [BN, GB] }> = select(
                            lower,
                            min_low & broadcast_scalar(SCALE_MASK, const_shape![BN, GB]),
                            (min_high >> broadcast_scalar(NIBBLE_BITS, const_shape![BN, GB]))
                                | ((min_low >> broadcast_scalar(SCALE_BITS, const_shape![BN, GB]))
                                    << broadcast_scalar(NIBBLE_BITS, const_shape![BN, GB])),
                        );
                        let minimum: Tile<f32, { [BN, GB] }> = convert_tile(minimum);
                        m * minimum
                    } else {
                        m
                    };
                    let local: Tile<i32, { [BK] }> =
                        safe_k & broadcast_scalar(block_values - 1, const_shape![BK]);
                    let quant_offset: Tile<i32, { [BK] }> = if FORMAT == 0 {
                        broadcast_scalar(Q4K_QUANTS_OFFSET, const_shape![BK])
                            + (local / broadcast_scalar(Q4K_GROUP_PAIR_VALUES, const_shape![BK]))
                                * broadcast_scalar(Q4_1_VALUES, const_shape![BK])
                            + (local & broadcast_scalar(Q4_1_VALUES - 1, const_shape![BK]))
                    } else {
                        broadcast_scalar(QUANT_HEADERS_BYTES, const_shape![BK])
                            + (local & broadcast_scalar(NIBBLE_MASK, const_shape![BK]))
                    };
                    let quant_offset: Tile<i64, { [BK] }> = exti(quant_offset);
                    let expanded_base: Tile<i64, { [BN, BK] }> = byte_base
                        .reshape(const_shape![BN, GB, 1])
                        .broadcast(const_shape![BN, GB, 32])
                        .reshape(const_shape![BN, BK]);
                    let quant_offset: Tile<i64, { [BN, BK] }> = expanded_base
                        + quant_offset
                            .reshape(const_shape![1, BK])
                            .broadcast(const_shape![BN, BK]);
                    let packed_ptr: PointerTile<*mut u8, { [] }> = pointer_to_tile(weight_bytes);
                    let packed_ptr: PointerTile<*mut u8, { [1, 1] }> =
                        packed_ptr.reshape(const_shape![1, 1]);
                    let packed_ptr: PointerTile<*mut u8, { [BN, BK] }> =
                        packed_ptr.broadcast(const_shape![BN, BK]);
                    let packed_ptr = packed_ptr.offset_tile(quant_offset);
                    let (packed, _): (Tile<u8, { [BN, BK] }>, Token) = load_ptr_tko(
                        packed_ptr,
                        ordering::Weak,
                        None::<scope::TileBlock>,
                        None,
                        None,
                        None,
                        Latency::<0>,
                    );
                    let packed: Tile<i32, { [BN, BK] }> = exti(packed);
                    let shift: Tile<i32, { [BK] }> = if FORMAT == 0 {
                        ((local / broadcast_scalar(Q4_1_VALUES, const_shape![BK]))
                            & broadcast_scalar(1i32, const_shape![BK]))
                            * broadcast_scalar(NIBBLE_BITS, const_shape![BK])
                    } else {
                        (local / broadcast_scalar(Q4_1_HALF_VALUES, const_shape![BK]))
                            * broadcast_scalar(NIBBLE_BITS, const_shape![BK])
                    };
                    let shift: Tile<i32, { [BN, BK] }> = shift
                        .reshape(const_shape![1, BK])
                        .broadcast(const_shape![BN, BK]);
                    let q: Tile<i32, { [BN, BK] }> =
                        (packed >> shift) & broadcast_scalar(NIBBLE_MASK, const_shape![BN, BK]);
                    let q: Tile<f32, { [BN, BK] }> = convert_tile(q);
                    let scale: Tile<f32, { [BN, BK] }> = scale
                        .reshape(const_shape![BN, GB, 1])
                        .broadcast(const_shape![BN, GB, 32])
                        .reshape(const_shape![BN, BK]);
                    let bias: Tile<f32, { [BN, BK] }> = bias
                        .reshape(const_shape![BN, GB, 1])
                        .broadcast(const_shape![BN, GB, 32])
                        .reshape(const_shape![BN, BK]);
                    let decoded: Tile<f32, { [BN, BK] }> = if FORMAT == 0 {
                        scale * q - bias
                    } else {
                        scale * q + bias
                    };
                    let b: Tile<bf16, { [BN, BK] }> = convert_tile(decoded);
                    let b_mask: Tile<bool, { [BN, BK] }> = valid_columns
                        .reshape(const_shape![BN, 1])
                        .broadcast(const_shape![BN, BK])
                        & valid_k
                            .reshape(const_shape![1, BK])
                            .broadcast(const_shape![BN, BK]);
                    let b_zero: Tile<bf16, { [BN, BK] }> =
                        constant(bf16::ZERO, const_shape![BN, BK]);
                    let b: Tile<bf16, { [BN, BK] }> = select(b_mask, b, b_zero);
                    let b: Tile<bf16, { [BK, BN] }> = permute(b, const_array![1, 0]);
                    acc = mmaf(a, b, acc);
                }
            }
            let ids64: Tile<i64, { [BM] }> = exti(ids);
            let columns64: Tile<i64, { [BN] }> = exti(columns);
            let output_stride: Tile<i64, { [BM] }> =
                exti(broadcast_scalar(n_size, const_shape![BM]));
            let output_offsets: Tile<i64, { [BM, BN] }> = (ids64 * output_stride)
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BN])
                + columns64
                    .reshape(const_shape![1, BN])
                    .broadcast(const_shape![BM, BN]);
            let output_mask: Tile<bool, { [BM, BN] }> = valid_rows
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BN])
                & valid_columns
                    .reshape(const_shape![1, BN])
                    .broadcast(const_shape![BM, BN]);
            let out: PointerTile<*mut f32, { [] }> = pointer_to_tile(output);
            let out: PointerTile<*mut f32, { [1, 1] }> = out.reshape(const_shape![1, 1]);
            let out: PointerTile<*mut f32, { [BM, BN] }> = out.broadcast(const_shape![BM, BN]);
            let out = out.offset_tile(output_offsets);
            store_ptr_tko(
                out,
                acc,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                None,
                Latency::<0>,
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GgufMoeConfig {
    pub bm: i32,
    pub bn: i32,
    pub bk: i32,
}

impl Default for GgufMoeConfig {
    fn default() -> Self {
        Self {
            bm: DEFAULT_BM,
            bn: DEFAULT_BN,
            bk: DEFAULT_BK,
        }
    }
}

/// Routing must come from moe_align with the same BM, assignment count, and CUDA device.
pub struct GgufMoeProjection<'a> {
    pub input: &'a Tensor,
    pub weight: &'a QTensor,
    pub sorted_token_ids: &'a CudaSlice<i32>,
    pub expert_ids: &'a CudaSlice<i32>,
    pub num_tokens_post_pad: &'a CudaSlice<i32>,
    pub padded_capacity: usize,
    pub assignments: usize,
    pub top_k: usize,
    pub config: GgufMoeConfig,
}

/// Experimental native GGUF projection with BF16-rounded weights and FP32 accumulation.
pub fn gguf_moe_projection(args: GgufMoeProjection<'_>) -> Result<Tensor> {
    let format = match args.weight.dtype() {
        GgmlDType::Q4K => 0,
        GgmlDType::Q4_1 => 1,
        dtype => candle_core::bail!("cuTile GGUF MoE does not support {dtype:?}"),
    };
    let (_, n_size, k_size) = args.weight.shape().dims3()?;
    let (input_rows, input_k) = args.input.dims2()?;
    if args.top_k == 0
        || args.input.dtype() != DType::BF16
        || !args.input.is_contiguous()
        || input_k != k_size
        || input_rows * args.top_k != args.assignments
    {
        candle_core::bail!(
            "cuTile GGUF MoE requires contiguous BF16 inputs matching the routed weight shape"
        );
    }
    let cfg = args.config;
    if ![8, 16, 32].contains(&cfg.bm)
        || ![32, 64, 128].contains(&cfg.bn)
        || ![32, 64, 128, 256].contains(&cfg.bk)
    {
        candle_core::bail!("unsupported cuTile GGUF MoE tile {cfg:?}");
    }
    if args.sorted_token_ids.len() < args.padded_capacity
        || args.expert_ids.len() < args.padded_capacity.div_ceil(cfg.bm as usize)
        || args.num_tokens_post_pad.is_empty()
    {
        candle_core::bail!("cuTile GGUF MoE routing buffers do not cover padded capacity");
    }
    let dev = args.input.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let (storage, layout) = args.input.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        unreachable!()
    };
    let input = storage.as_cuda_slice::<bf16>()?;
    let mut output = unsafe { dev.alloc::<f32>(args.assignments * n_size)? };
    let (input_ptr, _input_guard) = input.device_ptr(&stream);
    let input_ptr = input_ptr + (layout.start_offset() * std::mem::size_of::<bf16>()) as u64;
    let (weight_ptr, _weight_guard) = args.weight.device_ptr_with_guard(&stream)?;
    let (output_ptr, output_guard) = output.device_ptr_mut(&stream);
    let (ids_ptr, _ids_guard) = args.sorted_token_ids.device_ptr(&stream);
    let (experts_ptr, _experts_guard) = args.expert_ids.device_ptr(&stream);
    let (padded_ptr, _padded_guard) = args.num_tokens_post_pad.device_ptr(&stream);
    let launcher = unsafe {
        kernels::projection(
            DevicePointer::<f32>::from_cu_deviceptr(output_ptr as CUdeviceptr),
            DevicePointer::<bf16>::from_cu_deviceptr(input_ptr as CUdeviceptr),
            DevicePointer::<u8>::from_cu_deviceptr(weight_ptr as CUdeviceptr),
            DevicePointer::<f16>::from_cu_deviceptr(weight_ptr as CUdeviceptr),
            DevicePointer::<i32>::from_cu_deviceptr(ids_ptr as CUdeviceptr),
            DevicePointer::<i32>::from_cu_deviceptr(experts_ptr as CUdeviceptr),
            DevicePointer::<i32>::from_cu_deviceptr(padded_ptr as CUdeviceptr),
            n_size as i32,
            k_size as i32,
            args.assignments as i32,
            args.padded_capacity as i32,
        )
    }
    .generics(vec![
        cfg.bm.to_string(),
        cfg.bn.to_string(),
        cfg.bk.to_string(),
        (cfg.bk / 32).to_string(),
        format.to_string(),
        args.top_k.to_string(),
    ])
    .grid((
        (args.padded_capacity.div_ceil(cfg.bm as usize) * n_size.div_ceil(cfg.bn as usize)) as u32,
        1,
        1,
    ));
    catch_cutile_panic("GGUF MoE projection", || unsafe {
        launcher
            .async_on(&context::stream(dev))
            .map_err(|error| candle_core::Error::Msg(format!("cuTile GGUF MoE: {error:?}")))
    })?;
    drop(output_guard);
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
        (args.assignments, n_size),
    )))
}
