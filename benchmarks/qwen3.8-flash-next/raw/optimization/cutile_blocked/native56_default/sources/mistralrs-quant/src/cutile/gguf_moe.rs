#![allow(
    clippy::too_many_arguments,
    reason = "cuTile kernel arguments follow the device ABI"
)]

use candle_core::cuda::cudarc::driver::CudaSlice;
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{CudaStorage, DType, Result, Storage, Tensor};
use cutile::cuda_async::device_buffer::DevicePointer;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tile_kernel::TileKernel;
use half::{bf16, f16};

use super::{catch_cutile_panic, context};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

const DEFAULT_BM: i32 = 16;
const DEFAULT_BN: i32 = 64;
const DEFAULT_BK: i32 = 128;

#[cutile::module]
mod kernels {
    use cutile::core::*;

    const Q4K_VALUES: i32 = 256;
    const Q4K_BYTES: i32 = 144;
    const Q4_1_VALUES: i32 = 32;
    const Q4_1_BYTES: i32 = 20;
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
                    let zero: Tile<i32, { [] }> = scalar_to_tile(0i32);
                    let one: Tile<i32, { [] }> = scalar_to_tile(1i32);
                    let b: Tile<bf16, { [BN, BK] }> = if FORMAT == 0 {
                        let block: Tile<i64, { [BN] }> =
                            exti(broadcast_scalar((kt * BK) / Q4K_VALUES, const_shape![BN]));
                        let block_bytes: Tile<i64, { [BN] }> =
                            exti(broadcast_scalar(Q4K_BYTES, const_shape![BN]));
                        let byte_base: Tile<i64, { [BN] }> = (weight_base + block) * block_bytes;
                        let half_base: Tile<i64, { [BN] }> =
                            byte_base / broadcast_scalar(2i64, const_shape![BN]);
                        let d_ptr: PointerTile<*mut f16, { [] }> = pointer_to_tile(weight_halves);
                        let d_ptr: PointerTile<*mut f16, { [1] }> = d_ptr.reshape(const_shape![1]);
                        let d_ptr: PointerTile<*mut f16, { [BN] }> =
                            d_ptr.broadcast(const_shape![BN]);
                        let d_ptr = d_ptr.offset_tile(half_base);
                        let (d, _): (Tile<f16, { [BN] }>, Token) = load_ptr_tko(
                            d_ptr,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        let m_ptr: PointerTile<*mut f16, { [] }> = pointer_to_tile(weight_halves);
                        let m_ptr: PointerTile<*mut f16, { [1] }> = m_ptr.reshape(const_shape![1]);
                        let m_ptr: PointerTile<*mut f16, { [BN] }> =
                            m_ptr.broadcast(const_shape![BN]);
                        let m_ptr =
                            m_ptr.offset_tile(half_base + broadcast_scalar(1i64, const_shape![BN]));
                        let (m, _): (Tile<f16, { [BN] }>, Token) = load_ptr_tko(
                            m_ptr,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        let d: Tile<f32, { [BN] }> = convert_tile(d);
                        let m: Tile<f32, { [BN] }> = convert_tile(m);
                        let group: Tile<i32, { [4] }> = iota(const_shape![4]);
                        let group: Tile<i64, { [4] }> = exti(group);
                        let group: Tile<i64, { [BN, 4] }> = group
                            .reshape(const_shape![1, 4])
                            .broadcast(const_shape![BN, 4]);
                        let scale_base: Tile<i64, { [BN, 4] }> = byte_base
                            .reshape(const_shape![BN, 1])
                            .broadcast(const_shape![BN, 4])
                            + group;
                        let s0_ptr: PointerTile<*mut u8, { [] }> = pointer_to_tile(weight_bytes);
                        let s0_ptr: PointerTile<*mut u8, { [1, 1] }> =
                            s0_ptr.reshape(const_shape![1, 1]);
                        let s0_ptr: PointerTile<*mut u8, { [BN, 4] }> =
                            s0_ptr.broadcast(const_shape![BN, 4]);
                        let s0_ptr = s0_ptr
                            .offset_tile(scale_base + broadcast_scalar(4i64, const_shape![BN, 4]));
                        let (s0, _): (Tile<u8, { [BN, 4] }>, Token) = load_ptr_tko(
                            s0_ptr,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        let s0: Tile<i32, { [BN, 4] }> = exti(s0);
                        let s1_ptr: PointerTile<*mut u8, { [] }> = pointer_to_tile(weight_bytes);
                        let s1_ptr: PointerTile<*mut u8, { [1, 1] }> =
                            s1_ptr.reshape(const_shape![1, 1]);
                        let s1_ptr: PointerTile<*mut u8, { [BN, 4] }> =
                            s1_ptr.broadcast(const_shape![BN, 4]);
                        let s1_ptr = s1_ptr
                            .offset_tile(scale_base + broadcast_scalar(8i64, const_shape![BN, 4]));
                        let (s1, _): (Tile<u8, { [BN, 4] }>, Token) = load_ptr_tko(
                            s1_ptr,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        let s1: Tile<i32, { [BN, 4] }> = exti(s1);
                        let s2_ptr: PointerTile<*mut u8, { [] }> = pointer_to_tile(weight_bytes);
                        let s2_ptr: PointerTile<*mut u8, { [1, 1] }> =
                            s2_ptr.reshape(const_shape![1, 1]);
                        let s2_ptr: PointerTile<*mut u8, { [BN, 4] }> =
                            s2_ptr.broadcast(const_shape![BN, 4]);
                        let s2_ptr = s2_ptr
                            .offset_tile(scale_base + broadcast_scalar(12i64, const_shape![BN, 4]));
                        let (s2, _): (Tile<u8, { [BN, 4] }>, Token) = load_ptr_tko(
                            s2_ptr,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        let s2: Tile<i32, { [BN, 4] }> = exti(s2);
                        let scale_lo: Tile<i32, { [BN, 4] }> =
                            s0 & broadcast_scalar(SCALE_MASK, const_shape![BN, 4]);
                        let scale_hi: Tile<i32, { [BN, 4] }> = (s2
                            & broadcast_scalar(NIBBLE_MASK, const_shape![BN, 4]))
                            | ((s0 >> broadcast_scalar(SCALE_BITS, const_shape![BN, 4]))
                                << broadcast_scalar(NIBBLE_BITS, const_shape![BN, 4]));
                        let min_lo: Tile<i32, { [BN, 4] }> =
                            s1 & broadcast_scalar(SCALE_MASK, const_shape![BN, 4]);
                        let min_hi: Tile<i32, { [BN, 4] }> = (s2
                            >> broadcast_scalar(NIBBLE_BITS, const_shape![BN, 4]))
                            | ((s1 >> broadcast_scalar(SCALE_BITS, const_shape![BN, 4]))
                                << broadcast_scalar(NIBBLE_BITS, const_shape![BN, 4]));
                        let scales: Tile<i32, { [BN, 8] }> = cat(scale_lo, scale_hi, 1i32);
                        let minima: Tile<i32, { [BN, 8] }> = cat(min_lo, min_hi, 1i32);
                        let scales: Tile<f32, { [BN, 8] }> = convert_tile(scales);
                        let minima: Tile<f32, { [BN, 8] }> = convert_tile(minima);
                        let scales: Tile<f32, { [BN, 8] }> = scales
                            * d.reshape(const_shape![BN, 1])
                                .broadcast(const_shape![BN, 8]);
                        let minima: Tile<f32, { [BN, 8] }> = minima
                            * m.reshape(const_shape![BN, 1])
                                .broadcast(const_shape![BN, 8]);
                        let scales: Tile<f32, { [BN, 4, 2] }> =
                            scales.reshape(const_shape![BN, 4, 2]);
                        let minima: Tile<f32, { [BN, 4, 2] }> =
                            minima.reshape(const_shape![BN, 4, 2]);
                        let scale_even: Tile<f32, { [BN, 4, 1] }> =
                            extract(scales, [zero, zero, zero]);
                        let scale_odd: Tile<f32, { [BN, 4, 1] }> =
                            extract(scales, [zero, zero, one]);
                        let min_even: Tile<f32, { [BN, 4, 1] }> =
                            extract(minima, [zero, zero, zero]);
                        let min_odd: Tile<f32, { [BN, 4, 1] }> = extract(minima, [zero, zero, one]);
                        let quant_index: Tile<i32, { [128] }> = iota(const_shape![128]);
                        let quant_index: Tile<i64, { [128] }> = exti(quant_index);
                        let quant_offsets: Tile<i64, { [BN, 128] }> = byte_base
                            .reshape(const_shape![BN, 1])
                            .broadcast(const_shape![BN, 128])
                            + quant_index
                                .reshape(const_shape![1, 128])
                                .broadcast(const_shape![BN, 128])
                            + broadcast_scalar(16i64, const_shape![BN, 128]);
                        let packed_ptr: PointerTile<*mut u8, { [] }> =
                            pointer_to_tile(weight_bytes);
                        let packed_ptr: PointerTile<*mut u8, { [1, 1] }> =
                            packed_ptr.reshape(const_shape![1, 1]);
                        let packed_ptr: PointerTile<*mut u8, { [BN, 128] }> =
                            packed_ptr.broadcast(const_shape![BN, 128]);
                        let packed_ptr = packed_ptr.offset_tile(quant_offsets);
                        let (packed, _): (Tile<u8, { [BN, 128] }>, Token) = load_ptr_tko(
                            packed_ptr,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        let packed: Tile<u8, { [BN, 4, 32] }> =
                            packed.reshape(const_shape![BN, 4, 32]);
                        let packed: Tile<i32, { [BN, 4, 32] }> = exti(packed);
                        let low: Tile<i32, { [BN, 4, 32] }> =
                            packed & broadcast_scalar(NIBBLE_MASK, const_shape![BN, 4, 32]);
                        let low: Tile<f32, { [BN, 4, 32] }> = convert_tile(low);
                        let low: Tile<f32, { [BN, 4, 32] }> = low
                            * scale_even.broadcast(const_shape![BN, 4, 32])
                            - min_even.broadcast(const_shape![BN, 4, 32]);
                        let low: Tile<bf16, { [BN, 4, 32] }> = convert_tile(low);
                        let high: Tile<i32, { [BN, 4, 32] }> =
                            packed >> broadcast_scalar(NIBBLE_BITS, const_shape![BN, 4, 32]);
                        let high: Tile<f32, { [BN, 4, 32] }> = convert_tile(high);
                        let high: Tile<f32, { [BN, 4, 32] }> = high
                            * scale_odd.broadcast(const_shape![BN, 4, 32])
                            - min_odd.broadcast(const_shape![BN, 4, 32]);
                        let high: Tile<bf16, { [BN, 4, 32] }> = convert_tile(high);
                        let low: Tile<bf16, { [BN, 4, 1, 32] }> =
                            low.reshape(const_shape![BN, 4, 1, 32]);
                        let high: Tile<bf16, { [BN, 4, 1, 32] }> =
                            high.reshape(const_shape![BN, 4, 1, 32]);
                        let decoded: Tile<bf16, { [BN, 4, 2, 32] }> = cat(low, high, 2i32);
                        let decoded: Tile<bf16, { [BN, 256] }> =
                            decoded.reshape(const_shape![BN, 256]);
                        let section: Tile<i32, { [] }> = scalar_to_tile(kt % (Q4K_VALUES / BK));
                        let decoded: Tile<bf16, { [BN, BK] }> = extract(decoded, [zero, section]);
                        decoded
                    } else {
                        let group: Tile<i32, { [GB] }> =
                            iota(const_shape![GB]) + broadcast_scalar(kt * GB, const_shape![GB]);
                        let group_count: Tile<i32, { [GB] }> =
                            broadcast_scalar(k_size / Q4_1_VALUES, const_shape![GB]);
                        let valid: Tile<bool, { [GB] }> = lt_tile(group, group_count);
                        let zero_group: Tile<i32, { [GB] }> =
                            broadcast_scalar(0i32, const_shape![GB]);
                        let group: Tile<i32, { [GB] }> = select(valid, group, zero_group);
                        let group: Tile<i64, { [GB] }> = exti(group);
                        let blocks: Tile<i64, { [BN, GB] }> = weight_base
                            .reshape(const_shape![BN, 1])
                            .broadcast(const_shape![BN, GB])
                            + group
                                .reshape(const_shape![1, GB])
                                .broadcast(const_shape![BN, GB]);
                        let bytes_per_block: Tile<i64, { [BN, GB] }> =
                            exti(broadcast_scalar(Q4_1_BYTES, const_shape![BN, GB]));
                        let byte_base: Tile<i64, { [BN, GB] }> = blocks * bytes_per_block;
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
                        let m_ptr = m_ptr
                            .offset_tile(half_base + broadcast_scalar(1i64, const_shape![BN, GB]));
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
                        let d: Tile<f32, { [BN, GB, 1] }> = d.reshape(const_shape![BN, GB, 1]);
                        let m: Tile<f32, { [BN, GB, 1] }> = m.reshape(const_shape![BN, GB, 1]);
                        let quant_index: Tile<i32, { [16] }> = iota(const_shape![16]);
                        let quant_index: Tile<i64, { [16] }> = exti(quant_index);
                        let quant_offsets: Tile<i64, { [BN, GB, 16] }> = byte_base
                            .reshape(const_shape![BN, GB, 1])
                            .broadcast(const_shape![BN, GB, 16])
                            + quant_index
                                .reshape(const_shape![1, 1, 16])
                                .broadcast(const_shape![BN, GB, 16])
                            + broadcast_scalar(4i64, const_shape![BN, GB, 16]);
                        let packed_ptr: PointerTile<*mut u8, { [] }> =
                            pointer_to_tile(weight_bytes);
                        let packed_ptr: PointerTile<*mut u8, { [1, 1, 1] }> =
                            packed_ptr.reshape(const_shape![1, 1, 1]);
                        let packed_ptr: PointerTile<*mut u8, { [BN, GB, 16] }> =
                            packed_ptr.broadcast(const_shape![BN, GB, 16]);
                        let packed_ptr = packed_ptr.offset_tile(quant_offsets);
                        let (packed, _): (Tile<u8, { [BN, GB, 16] }>, Token) = load_ptr_tko(
                            packed_ptr,
                            ordering::Weak,
                            None::<scope::TileBlock>,
                            None,
                            None,
                            None,
                            Latency::<0>,
                        );
                        let packed: Tile<i32, { [BN, GB, 16] }> = exti(packed);
                        let low: Tile<i32, { [BN, GB, 16] }> =
                            packed & broadcast_scalar(NIBBLE_MASK, const_shape![BN, GB, 16]);
                        let low: Tile<f32, { [BN, GB, 16] }> = convert_tile(low);
                        let low: Tile<f32, { [BN, GB, 16] }> =
                            d.broadcast(const_shape![BN, GB, 16]) * low
                                + m.broadcast(const_shape![BN, GB, 16]);
                        let low: Tile<bf16, { [BN, GB, 16] }> = convert_tile(low);
                        let high: Tile<i32, { [BN, GB, 16] }> =
                            packed >> broadcast_scalar(NIBBLE_BITS, const_shape![BN, GB, 16]);
                        let high: Tile<f32, { [BN, GB, 16] }> = convert_tile(high);
                        let high: Tile<f32, { [BN, GB, 16] }> =
                            d.broadcast(const_shape![BN, GB, 16]) * high
                                + m.broadcast(const_shape![BN, GB, 16]);
                        let high: Tile<bf16, { [BN, GB, 16] }> = convert_tile(high);
                        let low: Tile<bf16, { [BN, GB, 1, 16] }> =
                            low.reshape(const_shape![BN, GB, 1, 16]);
                        let high: Tile<bf16, { [BN, GB, 1, 16] }> =
                            high.reshape(const_shape![BN, GB, 1, 16]);
                        let decoded: Tile<bf16, { [BN, GB, 2, 16] }> = cat(low, high, 2i32);
                        let decoded: Tile<bf16, { [BN, BK] }> =
                            decoded.reshape(const_shape![BN, BK]);
                        decoded
                    };
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
    let (input_ptr, _input_guard) = slice_ptr_on_stream(input, layout.start_offset(), &stream);
    let (weight_ptr, _weight_guard) = args.weight.device_ptr_with_guard(&stream)?;
    let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
    let (ids_ptr, _ids_guard) = slice_ptr_on_stream(args.sorted_token_ids, 0, &stream);
    let (experts_ptr, _experts_guard) = slice_ptr_on_stream(args.expert_ids, 0, &stream);
    let (padded_ptr, _padded_guard) = slice_ptr_on_stream(args.num_tokens_post_pad, 0, &stream);
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
