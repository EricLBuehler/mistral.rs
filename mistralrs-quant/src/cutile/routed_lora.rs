//! One-shot BF16 routed MoE LoRA for cuTile.
#![allow(clippy::missing_safety_doc, clippy::too_many_arguments)]

use std::cell::RefCell;
use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};

use candle_core::{cuda::cudarc::driver::CudaSlice, CudaDevice, DType, Result};
use cuda_async::device_buffer::DevicePointer;
use cuda_async::device_operation::DeviceOp;
use cuda_core::sys::CUdeviceptr;
use cutile::tile_kernel::TileKernel;
use cutile_compiler::compiler::utils::CompileOptions;
use cutile_compiler::specialization::DivHint;
use half::bf16;

#[cfg(test)]
use crate::lora::RoutedLoraInputMode;
use crate::lora::{
    RoutedLoraCudaMetadata, RoutedLoraCudaWeightTable, RoutedLoraMetadataLayout,
    RoutedLoraProjectionLayout, ROUTED_LORA_BLOCK_SIZE,
};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

use super::{
    catch_cutile_panic, context, device_compute_capability, device_supported, jit_available,
};

pub const CUTILE_ROUTED_LORA_MAX_RANK: usize = 128;

const MIN_BLOCK_R: usize = 16;
const TARGET_CTA_PER_SM: usize = 2;
const MAX_N_AXIS_GROUPS: usize = 8;
const AUTOTUNE_TIMED_RUNS: usize = 3;
const TUNING_CACHE_CAPACITY: usize = 256;
const TUNING_LOCK_SHARDS: usize = 64;
const I32_INDEXABLE_ELEMENTS: usize = i32::MAX as usize + 1;
const POINTER_HINT_MAX_BYTES: usize = 16;

#[cutile::module]
mod routed_lora_kernel {
    use cutile::core::*;

    const DESCRIPTOR_BYTES: i32 = 40;

    unsafe fn load_scalar_tile<T: ElementType>(ptr: *mut T, offset: i32) -> Tile<T, { [] }> {
        let base_scalar: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        let base: PointerTile<*mut T, { [1] }> = base_scalar.reshape(const_shape![1]);
        let offset: Tile<i32, { [1] }> = broadcast_scalar(offset, const_shape![1]);
        let pointers: PointerTile<*mut T, { [1] }> = base.offset_tile(offset);
        let (value, _): (Tile<T, { [1] }>, Token) = load_ptr_tko(
            pointers,
            ordering::Weak,
            None::<scope::TileBlock>,
            None,
            None,
            None,
            Latency::<0>,
        );
        value.reshape(const_shape![])
    }

    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn routed_lora_one_shot<
        const BR: i32,
        const BK: i32,
        const BN: i32,
        const N_GROUPS: i32,
        const HAS_INPUT_ROWS: i32,
        const HAS_OUTPUT_ROWS: i32,
        const HAS_ROUTE_SCALES: i32,
        const INPUT_MODE: i32,
        const NAIVE_ASSIGNMENT: i32,
        const HAS_TOKEN_ADAPTER_SLOTS: i32,
    >(
        input_ptr: *mut bf16,
        output_ptr: *mut bf16,
        descriptors_ptr: *mut u8,
        sorted_route_ids_ptr: *mut i32,
        block_pair_ids_ptr: *mut i32,
        token_adapter_slots_ptr: *mut i32,
        topk_expert_ids_ptr: *mut i32,
        route_input_rows_ptr: *mut i32,
        route_output_rows_ptr: *mut i32,
        route_output_scales_ptr: *mut f32,
        num_routes: i32,
        top_k: i32,
        num_experts: i32,
        num_adapter_slots: i32,
        input_features: i32,
        output_features: i32,
        output_row_stride: i32,
        output_slice_stride: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block = pid.0 / N_GROUPS;
        let n_group = pid.0 - block * N_GROUPS;
        let slice = pid.1;
        let output_tiles = 1 + (output_features - 1) / BN;
        if n_group >= output_tiles {
            return;
        }

        if NAIVE_ASSIGNMENT != 0 && block >= num_routes {
            return;
        }
        let pair: i32 = if NAIVE_ASSIGNMENT != 0 {
            let token = block / top_k;
            let adapter_slot: i32 = if HAS_TOKEN_ADAPTER_SLOTS != 0 {
                tile_to_scalar(load_scalar_tile(token_adapter_slots_ptr, token))
            } else {
                0
            };
            let expert: i32 = tile_to_scalar(load_scalar_tile(topk_expert_ids_ptr, block));
            let assignment_valid = adapter_slot >= 0
                && adapter_slot < num_adapter_slots
                && expert >= 0
                && expert < num_experts;
            if assignment_valid {
                adapter_slot * num_experts + expert
            } else {
                -1
            }
        } else {
            tile_to_scalar(load_scalar_tile(block_pair_ids_ptr, block))
        };
        let pair_valid = pair >= 0 && pair < num_adapter_slots * num_experts;
        let safe_pair = if pair_valid { pair } else { 0 };
        let adapter_slot: i32 = safe_pair / num_experts;
        let expert: i32 = safe_pair - adapter_slot * num_experts;

        let descriptor_index: i32 = slice * num_adapter_slots + adapter_slot;
        let descriptor_byte_offset: Tile<i32, { [] }> =
            scalar_to_tile(descriptor_index * DESCRIPTOR_BYTES);
        let descriptors_base: PointerTile<*mut u8, { [] }> = pointer_to_tile(descriptors_ptr);
        let descriptor_base: PointerTile<*mut u8, { [] }> =
            descriptors_base.offset_tile(descriptor_byte_offset);
        let descriptor_u64: *mut u64 = tile_to_pointer(ptr_to_ptr(descriptor_base));
        let descriptor_i32: *mut i32 = tile_to_pointer(ptr_to_ptr(descriptor_base));
        let descriptor_f32: *mut f32 = tile_to_pointer(ptr_to_ptr(descriptor_base));
        let a_address: u64 = tile_to_scalar(load_scalar_tile(descriptor_u64, 0));
        let b_address: u64 = tile_to_scalar(load_scalar_tile(descriptor_u64, 1));
        let scales_address: u64 = tile_to_scalar(load_scalar_tile(descriptor_u64, 2));
        let rank: i32 = tile_to_scalar(load_scalar_tile(descriptor_i32, 6));
        let rank_stride: i32 = tile_to_scalar(load_scalar_tile(descriptor_i32, 7));
        let mut adapter_scale: f32 = tile_to_scalar(load_scalar_tile(descriptor_f32, 8));
        if rank <= 0 || rank > BR || rank_stride < rank || a_address == 0 || b_address == 0 {
            return;
        }
        if scales_address != 0 {
            let scales_tile: Tile<u64, { [] }> = scalar_to_tile(scales_address);
            let scales_pointer_tile: PointerTile<*mut f32, { [] }> = int_to_ptr(scales_tile);
            let scales_ptr: *mut f32 = tile_to_pointer(scales_pointer_tile);
            adapter_scale = tile_to_scalar(load_scalar_tile(scales_ptr, expert));
        }
        if adapter_scale == 0.0 {
            return;
        }

        let route_lane: Tile<i32, { [16] }> = iota(const_shape![16]);
        let route_index_base: Tile<i32, { [16] }> = broadcast_scalar(block * 16, const_shape![16]);
        let routes: Tile<i32, { [16] }> = if NAIVE_ASSIGNMENT != 0 {
            let route = broadcast_scalar(block, const_shape![16]);
            let lane_zero = eq_tile(route_lane, broadcast_scalar(0i32, const_shape![16]));
            select(lane_zero, route, broadcast_scalar(-1i32, const_shape![16]))
        } else {
            let route_indices = route_index_base + route_lane;
            let sorted_base_0: PointerTile<*mut i32, { [] }> =
                pointer_to_tile(sorted_route_ids_ptr);
            let sorted_base_1: PointerTile<*mut i32, { [1] }> =
                sorted_base_0.reshape(const_shape![1]);
            let sorted_base: PointerTile<*mut i32, { [16] }> =
                sorted_base_1.broadcast(const_shape![16]);
            let sorted_ptrs = sorted_base.offset_tile(route_indices);
            let (routes, _): (Tile<i32, { [16] }>, Token) = load_ptr_tko(
                sorted_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                None,
                None,
                None,
                Latency::<0>,
            );
            routes
        };
        let zero_routes: Tile<i32, { [16] }> = broadcast_scalar(0i32, const_shape![16]);
        let num_routes_tile: Tile<i32, { [16] }> = broadcast_scalar(num_routes, const_shape![16]);
        let pair_mask: Tile<bool, { [16] }> = broadcast_scalar(pair_valid, const_shape![16]);
        let route_mask =
            pair_mask & ge_tile(routes, zero_routes) & lt_tile(routes, num_routes_tile);
        let safe_routes = select(route_mask, routes, zero_routes);

        let input_rows: Tile<i32, { [16] }> = if HAS_INPUT_ROWS != 0 {
            let base_0: PointerTile<*mut i32, { [] }> = pointer_to_tile(route_input_rows_ptr);
            let base_1: PointerTile<*mut i32, { [1] }> = base_0.reshape(const_shape![1]);
            let base: PointerTile<*mut i32, { [16] }> = base_1.broadcast(const_shape![16]);
            let pointers = base.offset_tile(safe_routes);
            let (rows, _): (Tile<i32, { [16] }>, Token) = load_ptr_tko(
                pointers,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(route_mask),
                Some(0i32),
                None,
                Latency::<0>,
            );
            rows
        } else if INPUT_MODE == 1 {
            safe_routes / broadcast_scalar(top_k, const_shape![16])
        } else {
            safe_routes
        };
        let output_rows: Tile<i32, { [16] }> = if HAS_OUTPUT_ROWS != 0 {
            let base_0: PointerTile<*mut i32, { [] }> = pointer_to_tile(route_output_rows_ptr);
            let base_1: PointerTile<*mut i32, { [1] }> = base_0.reshape(const_shape![1]);
            let base: PointerTile<*mut i32, { [16] }> = base_1.broadcast(const_shape![16]);
            let pointers = base.offset_tile(safe_routes);
            let (rows, _): (Tile<i32, { [16] }>, Token) = load_ptr_tko(
                pointers,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(route_mask),
                Some(0i32),
                None,
                Latency::<0>,
            );
            rows
        } else {
            safe_routes
        };

        let a_address_tile: Tile<u64, { [] }> = scalar_to_tile(a_address);
        let a_pointer_tile: PointerTile<*mut bf16, { [] }> = int_to_ptr(a_address_tile);
        let a_ptr: *mut bf16 = tile_to_pointer(a_pointer_tile);
        let b_address_tile: Tile<u64, { [] }> = scalar_to_tile(b_address);
        let b_pointer_tile: PointerTile<*mut bf16, { [] }> = int_to_ptr(b_address_tile);
        let b_ptr: *mut bf16 = tile_to_pointer(b_pointer_tile);
        let rank_offsets: Tile<i32, { [BR] }> = iota(const_shape![BR]);
        let rank_limit: Tile<i32, { [BR] }> = broadcast_scalar(rank, const_shape![BR]);
        let rank_mask = lt_tile(rank_offsets, rank_limit);
        let safe_rank = select(
            rank_mask,
            rank_offsets,
            broadcast_scalar(0i32, const_shape![BR]),
        );
        let route_mask_2d: Tile<bool, { [16, BK] }> = route_mask
            .reshape(const_shape![16, 1])
            .broadcast(const_shape![16, BK]);
        let input_row_offsets = muli(
            input_rows,
            broadcast_scalar(input_features, const_shape![16]),
            overflow::NoSignedWrap,
        );
        let input_row_offsets: Tile<i32, { [16, BK] }> = input_row_offsets
            .reshape(const_shape![16, 1])
            .broadcast(const_shape![16, BK]);
        let input_base_0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(input_ptr);
        let input_base_1: PointerTile<*mut bf16, { [1, 1] }> =
            input_base_0.reshape(const_shape![1, 1]);
        let input_base: PointerTile<*mut bf16, { [16, BK] }> =
            input_base_1.broadcast(const_shape![16, BK]);
        let a_base_0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(a_ptr);
        let a_base_1: PointerTile<*mut bf16, { [1, 1] }> = a_base_0.reshape(const_shape![1, 1]);
        let a_base: PointerTile<*mut bf16, { [BK, BR] }> = a_base_1.broadcast(const_shape![BK, BR]);
        let expert_a_offset = expert * rank_stride * input_features;
        let safe_rank_a: Tile<i32, { [1, BR] }> = muli(
            safe_rank,
            broadcast_scalar(input_features, const_shape![BR]),
            overflow::NoSignedWrap,
        )
        .reshape(const_shape![1, BR]);
        let safe_rank_a: Tile<i32, { [BK, BR] }> = safe_rank_a.broadcast(const_shape![BK, BR]);
        let rank_mask_a: Tile<bool, { [BK, BR] }> = rank_mask
            .reshape(const_shape![1, BR])
            .broadcast(const_shape![BK, BR]);
        let input_k_lane: Tile<i32, { [BK] }> = iota(const_shape![BK]);
        let mut hidden: Tile<f32, { [16, BR] }> = constant(0.0f32, const_shape![16, BR]);

        let input_tiles = 1 + (input_features - 1) / BK;
        for input_tile in 0i32..input_tiles {
            let input_columns = input_k_lane + broadcast_scalar(input_tile * BK, const_shape![BK]);
            let input_column_mask = lt_tile(
                input_columns,
                broadcast_scalar(input_features, const_shape![BK]),
            );
            let safe_input_columns = select(
                input_column_mask,
                input_columns,
                broadcast_scalar(0i32, const_shape![BK]),
            );
            let input_columns_x: Tile<i32, { [16, BK] }> = safe_input_columns
                .reshape(const_shape![1, BK])
                .broadcast(const_shape![16, BK]);
            let x_offsets = input_row_offsets + input_columns_x;
            let x_ptrs = input_base.offset_tile(x_offsets);
            let input_column_mask_x: Tile<bool, { [16, BK] }> = input_column_mask
                .reshape(const_shape![1, BK])
                .broadcast(const_shape![16, BK]);
            let x_mask = route_mask_2d & input_column_mask_x;
            let (x_loaded, _): (Tile<bf16, { [16, BK] }>, Token) = load_ptr_tko(
                x_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(x_mask),
                None,
                None,
                Latency::<0>,
            );
            let x_zero: Tile<bf16, { [16, BK] }> = constant(bf16::ZERO, const_shape![16, BK]);
            let x_loaded = select(x_mask, x_loaded, x_zero);

            let input_columns_a: Tile<i32, { [BK, 1] }> =
                safe_input_columns.reshape(const_shape![BK, 1]);
            let input_columns_a: Tile<i32, { [BK, BR] }> =
                input_columns_a.broadcast(const_shape![BK, BR]);
            let a_offsets = input_columns_a
                + safe_rank_a
                + broadcast_scalar(expert_a_offset, const_shape![BK, BR]);
            let a_ptrs = a_base.offset_tile(a_offsets);
            let input_column_mask_a: Tile<bool, { [BK, BR] }> = input_column_mask
                .reshape(const_shape![BK, 1])
                .broadcast(const_shape![BK, BR]);
            let a_mask = input_column_mask_a & rank_mask_a;
            let (a_loaded, _): (Tile<bf16, { [BK, BR] }>, Token) = load_ptr_tko(
                a_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(a_mask),
                None,
                None,
                Latency::<0>,
            );
            let a_zero: Tile<bf16, { [BK, BR] }> = constant(bf16::ZERO, const_shape![BK, BR]);
            let a_loaded = select(a_mask, a_loaded, a_zero);
            hidden = mmaf(x_loaded, a_loaded, hidden);
        }
        let hidden: Tile<bf16, { [16, BR] }> = convert_tile(hidden);

        let tile_begin = output_tiles * n_group / N_GROUPS;
        let tile_end = output_tiles * (n_group + 1) / N_GROUPS;

        let mut route_scale: Tile<f32, { [16] }> =
            broadcast_scalar(adapter_scale, const_shape![16]);
        if HAS_ROUTE_SCALES != 0 {
            let base_0: PointerTile<*mut f32, { [] }> = pointer_to_tile(route_output_scales_ptr);
            let base_1: PointerTile<*mut f32, { [1] }> = base_0.reshape(const_shape![1]);
            let base: PointerTile<*mut f32, { [16] }> = base_1.broadcast(const_shape![16]);
            let pointers = base.offset_tile(safe_routes);
            let (scales, _): (Tile<f32, { [16] }>, Token) = load_ptr_tko(
                pointers,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(route_mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            route_scale = route_scale * scales;
        }
        let route_scale: Tile<f32, { [16, BN] }> = route_scale
            .reshape(const_shape![16, 1])
            .broadcast(const_shape![16, BN]);
        let output_row_offsets = muli(
            output_rows,
            broadcast_scalar(output_row_stride, const_shape![16]),
            overflow::NoSignedWrap,
        );
        let output_row_offsets: Tile<i32, { [16, BN] }> = output_row_offsets
            .reshape(const_shape![16, 1])
            .broadcast(const_shape![16, BN]);
        let output_base_0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(output_ptr);
        let output_base_1: PointerTile<*mut bf16, { [1, 1] }> =
            output_base_0.reshape(const_shape![1, 1]);
        let output_base: PointerTile<*mut bf16, { [16, BN] }> =
            output_base_1.broadcast(const_shape![16, BN]);
        let b_base_0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(b_ptr);
        let b_base_1: PointerTile<*mut bf16, { [1, 1] }> = b_base_0.reshape(const_shape![1, 1]);
        let b_base: PointerTile<*mut bf16, { [BR, BN] }> = b_base_1.broadcast(const_shape![BR, BN]);
        let expert_b_offset = expert * output_features * rank_stride;
        let rank_mask_b: Tile<bool, { [BR, BN] }> = rank_mask
            .reshape(const_shape![BR, 1])
            .broadcast(const_shape![BR, BN]);
        let safe_rank_b: Tile<i32, { [BR, BN] }> = safe_rank
            .reshape(const_shape![BR, 1])
            .broadcast(const_shape![BR, BN]);
        let output_lane: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let route_mask_output: Tile<bool, { [16, BN] }> = route_mask
            .reshape(const_shape![16, 1])
            .broadcast(const_shape![16, BN]);

        for output_tile in tile_begin..tile_end {
            let output_columns = output_lane + broadcast_scalar(output_tile * BN, const_shape![BN]);
            let output_column_mask = lt_tile(
                output_columns,
                broadcast_scalar(output_features, const_shape![BN]),
            );
            let safe_output_columns = select(
                output_column_mask,
                output_columns,
                broadcast_scalar(0i32, const_shape![BN]),
            );
            let output_columns_b: Tile<i32, { [BR, BN] }> = muli(
                safe_output_columns,
                broadcast_scalar(rank_stride, const_shape![BN]),
                overflow::NoSignedWrap,
            )
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BR, BN]);
            let b_offsets = output_columns_b
                + safe_rank_b
                + broadcast_scalar(expert_b_offset, const_shape![BR, BN]);
            let b_ptrs = b_base.offset_tile(b_offsets);
            let output_column_mask_b: Tile<bool, { [BR, BN] }> = output_column_mask
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BR, BN]);
            let b_mask = rank_mask_b & output_column_mask_b;
            let (b_loaded, _): (Tile<bf16, { [BR, BN] }>, Token) = load_ptr_tko(
                b_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(b_mask),
                None,
                None,
                Latency::<0>,
            );
            let b_zero: Tile<bf16, { [BR, BN] }> = constant(bf16::ZERO, const_shape![BR, BN]);
            let b_loaded = select(b_mask, b_loaded, b_zero);
            let mut delta: Tile<f32, { [16, BN] }> = constant(0.0f32, const_shape![16, BN]);
            delta = mmaf(hidden, b_loaded, delta) * route_scale;

            let output_columns_2d: Tile<i32, { [16, BN] }> = safe_output_columns
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![16, BN]);
            let slice_offset: Tile<i32, { [16, BN] }> =
                broadcast_scalar(slice * output_slice_stride, const_shape![16, BN]);
            let output_offsets = output_row_offsets + slice_offset + output_columns_2d;
            let output_ptrs = output_base.offset_tile(output_offsets);
            let output_column_mask_2d: Tile<bool, { [16, BN] }> = output_column_mask
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![16, BN]);
            let output_mask = route_mask_output & output_column_mask_2d;
            let (previous, _): (Tile<bf16, { [16, BN] }>, Token) = load_ptr_tko(
                output_ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                None,
                None,
                Latency::<0>,
            );
            let previous_zero: Tile<bf16, { [16, BN] }> =
                constant(bf16::ZERO, const_shape![16, BN]);
            let previous = select(output_mask, previous, previous_zero);
            let previous: Tile<f32, { [16, BN] }> = convert_tile(previous);
            let result: Tile<bf16, { [16, BN] }> = convert_tile(previous + delta);
            store_ptr_tko(
                output_ptrs,
                result,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(output_mask),
                None,
                Latency::<0>,
            );
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum CutileRoutedLoraOptimizationHint {
    Balanced,
    HighOccupancy,
    Cluster2,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct CutileRoutedLoraConfig {
    pub block_k: i32,
    pub block_n: i32,
    pub n_axis_groups: i32,
    pub optimization_hint: CutileRoutedLoraOptimizationHint,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct CutileRoutedLoraDeviceKey {
    pub ordinal: usize,
    pub compute_major: i32,
    pub compute_minor: i32,
    pub multiprocessors: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct CutileRoutedLoraShapeKey {
    pub top_k: usize,
    pub num_routes: usize,
    pub max_blocks: usize,
    pub route_block_size: usize,
    pub num_experts: usize,
    pub num_adapter_slots: usize,
    pub input_features: usize,
    pub output_features: usize,
    pub output_row_stride: usize,
    pub output_slice_stride: usize,
    pub num_slices: usize,
    pub weight_slice_offset: usize,
    pub max_rank: usize,
    pub max_rank_stride: usize,
    pub input_mode: i32,
    pub has_input_rows: bool,
    pub has_output_rows: bool,
    pub has_route_scales: bool,
    pub naive_assignment: bool,
    pub has_token_adapter_slots: bool,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct CutileRoutedLoraTuningKey {
    pub device: CutileRoutedLoraDeviceKey,
    pub shape: CutileRoutedLoraShapeKey,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct CutilePreparedKey {
    tuning: CutileRoutedLoraTuningKey,
    pointer_divisors: [i32; 10],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutileRoutedLoraUnsupported {
    DType(DType),
    Device,
    JitUnavailable,
    RouteBlockSize(usize),
    SliceCount(usize),
    Rank(usize),
    IndexRange,
    AutotuneFailed,
    LaunchFailed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutileRoutedLoraStatus {
    Launched,
    Unsupported(CutileRoutedLoraUnsupported),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CutileFailureBoundary {
    FallbackSafe,
    ScratchLaunchAttempted,
    OutputLaunchAttempted,
}

impl CutileFailureBoundary {
    fn severity(self) -> u8 {
        match self {
            Self::FallbackSafe => 0,
            Self::ScratchLaunchAttempted => 1,
            Self::OutputLaunchAttempted => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CutileAutotuneState {
    FallbackSafe,
    ScratchWorkPending,
}

impl CutileAutotuneState {
    fn failure_boundary(self) -> CutileFailureBoundary {
        match self {
            Self::FallbackSafe => CutileFailureBoundary::FallbackSafe,
            Self::ScratchWorkPending => CutileFailureBoundary::ScratchLaunchAttempted,
        }
    }

    fn mark_scratch_launch(&mut self) {
        *self = Self::ScratchWorkPending;
    }

    fn mark_synchronized(&mut self) {
        *self = Self::FallbackSafe;
    }

    fn failure(self, error: candle_core::Error) -> CutileAutotuneFailure {
        CutileAutotuneFailure {
            boundary: self.failure_boundary(),
            error,
        }
    }
}

#[derive(Debug)]
struct CutileAutotuneFailure {
    boundary: CutileFailureBoundary,
    error: candle_core::Error,
}

type CutileAutotuneResult<T> = std::result::Result<T, CutileAutotuneFailure>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CutileCachedFailure {
    reason: CutileRoutedLoraUnsupported,
    boundary: CutileFailureBoundary,
}

#[derive(Clone, Copy, Debug)]
pub struct CutileRoutedLoraLaunch {
    pub input: u64,
    pub output: u64,
    pub route_input_rows: u64,
    pub route_output_rows: u64,
    pub route_output_scales: u64,
    pub dtype: DType,
    pub projection: RoutedLoraProjectionLayout,
    pub weight_slice_offset: usize,
}

struct BoundedCache<K, V> {
    values: HashMap<K, V>,
    insertion_order: VecDeque<K>,
}

impl<K: Copy + Eq + Hash, V> BoundedCache<K, V> {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
            insertion_order: VecDeque::new(),
        }
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.values.get(key)
    }

    fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(current) = self.values.get_mut(&key) {
            return Some(std::mem::replace(current, value));
        }
        if self.values.len() == TUNING_CACHE_CAPACITY {
            let evicted = self.insertion_order.pop_front().unwrap();
            self.values.remove(&evicted);
        }
        self.insertion_order.push_back(key);
        self.values.insert(key, value)
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        let removed = self.values.remove(key)?;
        self.insertion_order.retain(|candidate| candidate != key);
        Some(removed)
    }
}

type TuningCache<V> = BoundedCache<CutileRoutedLoraTuningKey, V>;

static CONFIG_CACHE: OnceLock<Mutex<TuningCache<CutileRoutedLoraConfig>>> = OnceLock::new();
static BUCKET_CONFIG_CACHE: OnceLock<Mutex<TuningCache<CutileRoutedLoraConfig>>> = OnceLock::new();
static FAILED_KEYS: OnceLock<Mutex<TuningCache<CutileCachedFailure>>> = OnceLock::new();
static TUNING_LOCKS: OnceLock<[Mutex<()>; TUNING_LOCK_SHARDS]> = OnceLock::new();

thread_local! {
    static PREPARED_CONFIGS: RefCell<BoundedCache<CutilePreparedKey, CutileRoutedLoraConfig>> =
        RefCell::new(BoundedCache::new());
}

fn config_cache() -> &'static Mutex<TuningCache<CutileRoutedLoraConfig>> {
    CONFIG_CACHE.get_or_init(|| Mutex::new(BoundedCache::new()))
}

fn bucket_config_cache() -> &'static Mutex<TuningCache<CutileRoutedLoraConfig>> {
    BUCKET_CONFIG_CACHE.get_or_init(|| Mutex::new(BoundedCache::new()))
}

fn failed_keys() -> &'static Mutex<TuningCache<CutileCachedFailure>> {
    FAILED_KEYS.get_or_init(|| Mutex::new(BoundedCache::new()))
}

fn tuning_locks() -> &'static [Mutex<()>; TUNING_LOCK_SHARDS] {
    TUNING_LOCKS.get_or_init(|| std::array::from_fn(|_| Mutex::new(())))
}

fn prepared_config(key: CutilePreparedKey, config: CutileRoutedLoraConfig) -> bool {
    PREPARED_CONFIGS.with(|prepared| {
        prepared
            .borrow()
            .get(&key)
            .is_some_and(|current| *current == config)
    })
}

fn mark_config_prepared(key: CutilePreparedKey, config: CutileRoutedLoraConfig) {
    PREPARED_CONFIGS.with(|prepared| {
        prepared.borrow_mut().insert(key, config);
    });
}

fn tuning_lock_index(key: CutileRoutedLoraTuningKey) -> usize {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish() as usize % TUNING_LOCK_SHARDS
}

fn power_of_two_bucket(value: usize) -> usize {
    value
        .max(1)
        .checked_next_power_of_two()
        .unwrap_or(usize::MAX)
}

fn bucketed_tuning_key(mut key: CutileRoutedLoraTuningKey) -> CutileRoutedLoraTuningKey {
    key.shape.num_routes = power_of_two_bucket(key.shape.num_routes);
    key.shape.max_blocks = power_of_two_bucket(key.shape.max_blocks);
    key
}

fn cached_bucket_config(key: CutileRoutedLoraTuningKey) -> Option<CutileRoutedLoraConfig> {
    bucket_config_cache()
        .lock()
        .unwrap()
        .get(&bucketed_tuning_key(key))
        .copied()
        .filter(|config| valid_config(key, *config))
}

fn multiprocessor_count(dev: &CudaDevice) -> usize {
    use candle_core::cuda::cudarc::driver::{result, sys};

    let cu_device = dev.cuda_stream().context().cu_device();
    unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }
    .unwrap_or(1)
    .max(1) as usize
}

fn tuning_key(
    dev: &CudaDevice,
    metadata: RoutedLoraMetadataLayout,
    launch: CutileRoutedLoraLaunch,
    max_rank_stride: usize,
    addresses: KernelAddresses,
) -> CutileRoutedLoraTuningKey {
    let (compute_major, compute_minor) = device_compute_capability(dev);
    CutileRoutedLoraTuningKey {
        device: CutileRoutedLoraDeviceKey {
            ordinal: dev.cuda_stream().context().ordinal(),
            compute_major,
            compute_minor,
            multiprocessors: multiprocessor_count(dev),
        },
        shape: CutileRoutedLoraShapeKey {
            top_k: metadata.top_k(),
            num_routes: metadata.num_routes(),
            max_blocks: addresses.program_blocks(metadata),
            route_block_size: metadata.block_size(),
            num_experts: metadata.num_experts(),
            num_adapter_slots: metadata.num_adapter_slots(),
            input_features: launch.projection.input_features(),
            output_features: launch.projection.output_features(),
            output_row_stride: launch.projection.output_row_stride(),
            output_slice_stride: launch.projection.output_slice_stride(),
            num_slices: launch.projection.num_slices(),
            weight_slice_offset: launch.weight_slice_offset,
            max_rank: launch.projection.max_rank(),
            max_rank_stride,
            input_mode: launch.projection.input_mode() as i32,
            has_input_rows: launch.route_input_rows != 0,
            has_output_rows: launch.route_output_rows != 0,
            has_route_scales: launch.route_output_scales != 0,
            naive_assignment: addresses.naive_assignment,
            has_token_adapter_slots: addresses.token_adapter_slots != 0,
        },
    }
}

fn max_groups(shape: CutileRoutedLoraShapeKey, block_n: usize) -> usize {
    shape
        .output_features
        .div_ceil(block_n)
        .clamp(1, MAX_N_AXIS_GROUPS)
}

fn preferred_groups(key: CutileRoutedLoraTuningKey) -> usize {
    let shape = key.shape;
    let base_programs = (shape.max_blocks * shape.num_slices).max(1);
    let multiprocessors = key.device.multiprocessors.max(1);
    if base_programs * 2 >= multiprocessors * 3 {
        return 1;
    }
    let occupancy_groups = (TARGET_CTA_PER_SM * multiprocessors).div_ceil(base_programs);
    let shrink_ratio = shape.input_features as f64
        / shape
            .input_features
            .saturating_add(shape.output_features)
            .max(1) as f64;
    let duplication_budget = ((1.5 / shrink_ratio.max(1e-3)) as usize + 1).max(1);
    let target = occupancy_groups
        .min(duplication_budget)
        .min(max_groups(shape, 128));
    let maximum = max_groups(shape, 128);
    [1usize, 2, 4, 8]
        .into_iter()
        .filter(|candidate| *candidate <= maximum)
        .min_by_key(|candidate| candidate.abs_diff(target))
        .unwrap_or(1)
}

fn valid_config(key: CutileRoutedLoraTuningKey, config: CutileRoutedLoraConfig) -> bool {
    if key.shape.max_rank == 0
        || key.shape.max_rank > CUTILE_ROUTED_LORA_MAX_RANK
        || !matches!(
            key.shape.max_rank.max(MIN_BLOCK_R).next_power_of_two(),
            16 | 32 | 64 | 128
        )
        || !matches!(config.block_k, 64 | 128)
        || !matches!(config.block_n, 64 | 128)
        || !matches!(config.n_axis_groups, 1 | 2 | 4 | 8)
        || config.n_axis_groups as usize > max_groups(key.shape, config.block_n as usize)
    {
        return false;
    }
    let Some(grid_x) = key
        .shape
        .max_blocks
        .checked_mul(config.n_axis_groups as usize)
    else {
        return false;
    };
    if grid_x > i32::MAX as usize {
        return false;
    }
    if config.optimization_hint == CutileRoutedLoraOptimizationHint::Cluster2 {
        key.device.compute_major >= 12 && grid_x.is_multiple_of(2)
    } else {
        true
    }
}

/// Candidate order is the default heuristic followed by deterministic tuning alternatives.
pub fn cutile_routed_lora_candidate_configs(
    key: CutileRoutedLoraTuningKey,
) -> Vec<CutileRoutedLoraConfig> {
    let preferred_k = if key.shape.input_features >= 256
        || key.shape.num_routes / key.shape.num_experts.max(1) >= 16
    {
        128
    } else {
        64
    };
    let preferred_group = preferred_groups(key) as i32;
    let alternate_k = 192 - preferred_k;
    let mut candidates = Vec::new();
    let mut push = |block_k, block_n, n_axis_groups, optimization_hint| {
        let candidate = CutileRoutedLoraConfig {
            block_k,
            block_n,
            n_axis_groups,
            optimization_hint,
        };
        if valid_config(key, candidate) && !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    };
    push(
        preferred_k,
        128,
        preferred_group,
        CutileRoutedLoraOptimizationHint::Balanced,
    );
    push(
        alternate_k,
        128,
        preferred_group,
        CutileRoutedLoraOptimizationHint::Balanced,
    );
    push(
        preferred_k,
        64,
        preferred_group,
        CutileRoutedLoraOptimizationHint::Balanced,
    );
    push(
        alternate_k,
        64,
        preferred_group,
        CutileRoutedLoraOptimizationHint::Balanced,
    );
    for group in [1, 2, 4, 8] {
        push(
            preferred_k,
            128,
            group,
            CutileRoutedLoraOptimizationHint::Balanced,
        );
    }
    push(
        preferred_k,
        128,
        preferred_group,
        CutileRoutedLoraOptimizationHint::HighOccupancy,
    );
    push(
        preferred_k,
        128,
        preferred_group,
        CutileRoutedLoraOptimizationHint::Cluster2,
    );
    candidates
}

pub fn cached_cutile_routed_lora_config(
    key: CutileRoutedLoraTuningKey,
) -> Option<CutileRoutedLoraConfig> {
    config_cache().lock().unwrap().get(&key).copied()
}

pub fn set_cutile_routed_lora_tuned_config(
    key: CutileRoutedLoraTuningKey,
    config: CutileRoutedLoraConfig,
) -> Result<()> {
    if !valid_config(key, config) {
        candle_core::bail!("invalid cuTile routed LoRA config for tuning key");
    }
    config_cache().lock().unwrap().insert(key, config);
    clear_fallback_safe_failure(&mut failed_keys().lock().unwrap(), key);
    Ok(())
}

pub fn selected_cutile_routed_lora_config(
    key: CutileRoutedLoraTuningKey,
) -> Option<CutileRoutedLoraConfig> {
    if let Some(config) = cached_cutile_routed_lora_config(key) {
        return Some(config);
    }
    if let Some(config) = cached_bucket_config(key) {
        return Some(config);
    }
    cutile_routed_lora_candidate_configs(key).into_iter().next()
}

fn i32_indexable_product(factors: &[usize]) -> bool {
    factors
        .iter()
        .try_fold(1usize, |product, factor| product.checked_mul(*factor))
        .is_some_and(|elements| elements <= I32_INDEXABLE_ELEMENTS)
}

fn addressing_supported(
    metadata: RoutedLoraMetadataLayout,
    projection: RoutedLoraProjectionLayout,
    max_rank_stride: usize,
) -> bool {
    max_rank_stride <= i32::MAX as usize
        && i32_indexable_product(&[
            metadata.num_experts(),
            max_rank_stride,
            projection.input_features(),
        ])
        && i32_indexable_product(&[
            metadata.num_experts(),
            projection.output_features(),
            max_rank_stride,
        ])
        && i32_indexable_product(&[metadata.num_routes(), projection.input_features()])
        && i32_indexable_product(&[metadata.num_routes(), projection.output_row_stride()])
        && i32_indexable_product(&[
            projection.num_slices(),
            metadata.num_adapter_slots(),
            std::mem::size_of::<crate::lora::RoutedLoraAdapterWeight>(),
        ])
}

fn unsupported(
    dev: &CudaDevice,
    metadata: RoutedLoraMetadataLayout,
    launch: CutileRoutedLoraLaunch,
    max_rank_stride: usize,
) -> Option<CutileRoutedLoraUnsupported> {
    if launch.dtype != DType::BF16 {
        return Some(CutileRoutedLoraUnsupported::DType(launch.dtype));
    }
    if metadata.block_size() != ROUTED_LORA_BLOCK_SIZE {
        return Some(CutileRoutedLoraUnsupported::RouteBlockSize(
            metadata.block_size(),
        ));
    }
    if !matches!(launch.projection.num_slices(), 1 | 2) {
        return Some(CutileRoutedLoraUnsupported::SliceCount(
            launch.projection.num_slices(),
        ));
    }
    if launch.projection.max_rank() > CUTILE_ROUTED_LORA_MAX_RANK {
        return Some(CutileRoutedLoraUnsupported::Rank(
            launch.projection.max_rank(),
        ));
    }
    if !addressing_supported(metadata, launch.projection, max_rank_stride) {
        return Some(CutileRoutedLoraUnsupported::IndexRange);
    }
    if !device_supported(dev) {
        return Some(CutileRoutedLoraUnsupported::Device);
    }
    if !jit_available(dev) {
        return Some(CutileRoutedLoraUnsupported::JitUnavailable);
    }
    None
}

fn rank_block(rank: usize) -> i32 {
    rank.max(MIN_BLOCK_R).next_power_of_two() as i32
}

fn compile_options(hint: CutileRoutedLoraOptimizationHint) -> CompileOptions {
    match hint {
        CutileRoutedLoraOptimizationHint::Balanced => CompileOptions::default(),
        CutileRoutedLoraOptimizationHint::HighOccupancy => CompileOptions::default().occupancy(4),
        CutileRoutedLoraOptimizationHint::Cluster2 => CompileOptions::default().num_cta_in_cga(2),
    }
}

fn driver_error(operation: &str, error: impl std::fmt::Debug) -> candle_core::Error {
    candle_core::Error::Msg(format!("cuTile routed LoRA {operation}: {error:?}"))
}

#[derive(Clone, Copy)]
struct KernelAddresses {
    descriptors: u64,
    sorted_routes: u64,
    block_pairs: u64,
    token_adapter_slots: u64,
    topk_expert_ids: u64,
    naive_assignment: bool,
}

impl KernelAddresses {
    fn program_blocks(self, layout: RoutedLoraMetadataLayout) -> usize {
        if self.naive_assignment {
            layout.num_routes()
        } else {
            layout.max_blocks()
        }
    }
}

fn pointer_divisors(addresses: KernelAddresses, launch: CutileRoutedLoraLaunch) -> [i32; 10] {
    [
        launch.input,
        launch.output,
        addresses.descriptors,
        addresses.sorted_routes,
        addresses.block_pairs,
        addresses.token_adapter_slots,
        addresses.topk_expert_ids,
        launch.route_input_rows,
        launch.route_output_rows,
        launch.route_output_scales,
    ]
    .map(|address| DivHint::from_ptr(address).divisor)
}

fn prepared_key(
    tuning: CutileRoutedLoraTuningKey,
    addresses: KernelAddresses,
    launch: CutileRoutedLoraLaunch,
) -> CutilePreparedKey {
    CutilePreparedKey {
        tuning,
        pointer_divisors: pointer_divisors(addresses, launch),
    }
}

fn scratch_output_offset(base: u64, output: u64) -> Option<usize> {
    let target = DivHint::from_ptr(output).divisor;
    (0..POINTER_HINT_MAX_BYTES / std::mem::size_of::<bf16>()).find(|offset| {
        DivHint::from_ptr(base + (*offset * std::mem::size_of::<bf16>()) as u64).divisor == target
    })
}

unsafe fn launch_config(
    dev: &CudaDevice,
    layout: RoutedLoraMetadataLayout,
    addresses: KernelAddresses,
    launch: CutileRoutedLoraLaunch,
    config: CutileRoutedLoraConfig,
) -> Result<()> {
    let grid_x = addresses
        .program_blocks(layout)
        .checked_mul(config.n_axis_groups as usize)
        .filter(|value| *value <= i32::MAX as usize)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| candle_core::Error::msg("cuTile routed LoRA launch grid overflow"))?;
    let grid_y = u32::try_from(launch.projection.num_slices())
        .map_err(|_| candle_core::Error::msg("cuTile routed LoRA slice grid overflow"))?;
    let generics = vec![
        rank_block(launch.projection.max_rank()).to_string(),
        config.block_k.to_string(),
        config.block_n.to_string(),
        config.n_axis_groups.to_string(),
        i32::from(launch.route_input_rows != 0).to_string(),
        i32::from(launch.route_output_rows != 0).to_string(),
        i32::from(launch.route_output_scales != 0).to_string(),
        (launch.projection.input_mode() as i32).to_string(),
        i32::from(addresses.naive_assignment).to_string(),
        i32::from(addresses.token_adapter_slots != 0).to_string(),
    ];
    let ctx = context::execution_context(dev);
    let launcher = routed_lora_kernel::routed_lora_one_shot(
        DevicePointer::<bf16>::from_cu_deviceptr(launch.input as CUdeviceptr),
        DevicePointer::<bf16>::from_cu_deviceptr(launch.output as CUdeviceptr),
        DevicePointer::<u8>::from_cu_deviceptr(addresses.descriptors as CUdeviceptr),
        DevicePointer::<i32>::from_cu_deviceptr(addresses.sorted_routes as CUdeviceptr),
        DevicePointer::<i32>::from_cu_deviceptr(addresses.block_pairs as CUdeviceptr),
        DevicePointer::<i32>::from_cu_deviceptr(addresses.token_adapter_slots as CUdeviceptr),
        DevicePointer::<i32>::from_cu_deviceptr(addresses.topk_expert_ids as CUdeviceptr),
        DevicePointer::<i32>::from_cu_deviceptr(launch.route_input_rows as CUdeviceptr),
        DevicePointer::<i32>::from_cu_deviceptr(launch.route_output_rows as CUdeviceptr),
        DevicePointer::<f32>::from_cu_deviceptr(launch.route_output_scales as CUdeviceptr),
        layout.num_routes() as i32,
        layout.top_k() as i32,
        layout.num_experts() as i32,
        layout.num_adapter_slots() as i32,
        launch.projection.input_features() as i32,
        launch.projection.output_features() as i32,
        launch.projection.output_row_stride() as i32,
        launch.projection.output_slice_stride() as i32,
    )
    .generics(generics)
    .grid((grid_x, grid_y, 1))
    .compile_options(compile_options(config.optimization_hint));
    catch_cutile_panic("routed LoRA kernel execute", || unsafe {
        launcher
            .execute(&ctx)
            .map_err(|error| driver_error("launch", error))
    })?;
    Ok(())
}

fn tuning_output_elements(
    layout: RoutedLoraMetadataLayout,
    launch: CutileRoutedLoraLaunch,
) -> Result<usize> {
    layout
        .num_routes()
        .checked_mul(launch.projection.output_row_stride())
        .ok_or_else(|| candle_core::Error::msg("cuTile routed LoRA tuning output overflow"))
}

unsafe fn tuning_scratch(
    dev: &CudaDevice,
    layout: RoutedLoraMetadataLayout,
    launch: CutileRoutedLoraLaunch,
) -> Result<(CudaSlice<bf16>, usize)> {
    let alignment_elements = POINTER_HINT_MAX_BYTES / std::mem::size_of::<bf16>();
    let scratch_elements = tuning_output_elements(layout, launch)?
        .checked_add(alignment_elements)
        .ok_or_else(|| candle_core::Error::msg("cuTile routed LoRA scratch size overflow"))?;
    let mut scratch = unsafe { dev.alloc::<bf16>(scratch_elements)? };
    let stream = dev.cuda_stream();
    let (scratch_base, scratch_base_guard) = slice_ptr_mut_on_stream(&mut scratch, 0, &stream);
    drop(scratch_base_guard);
    let scratch_offset = scratch_output_offset(scratch_base, launch.output).ok_or_else(|| {
        candle_core::Error::msg("cuTile routed LoRA could not match output pointer alignment")
    })?;
    Ok((scratch, scratch_offset))
}

unsafe fn prepare_config_on_thread(
    dev: &CudaDevice,
    layout: RoutedLoraMetadataLayout,
    key: CutileRoutedLoraTuningKey,
    addresses: KernelAddresses,
    launch: CutileRoutedLoraLaunch,
    config: CutileRoutedLoraConfig,
) -> CutileAutotuneResult<()> {
    let prepared_key = prepared_key(key, addresses, launch);
    if prepared_config(prepared_key, config) {
        return Ok(());
    }
    let mut state = CutileAutotuneState::FallbackSafe;
    let stream = dev.cuda_stream();
    let (mut scratch, scratch_offset) =
        unsafe { tuning_scratch(dev, layout, launch) }.map_err(|error| state.failure(error))?;
    stream
        .memset_zeros(&mut scratch)
        .map_err(|error| state.failure(driver_error("scratch reset", error)))?;
    let (scratch_address, scratch_guard) =
        slice_ptr_mut_on_stream(&mut scratch, scratch_offset, &stream);
    let scratch_launch = CutileRoutedLoraLaunch {
        output: scratch_address,
        ..launch
    };
    launch_config(dev, layout, addresses, scratch_launch, config)
        .map_err(|error| state.failure(error))?;
    state.mark_scratch_launch();
    drop(scratch_guard);
    let completion = stream
        .record_event(None)
        .map_err(|error| state.failure(driver_error("JIT preparation event", error)))?;
    completion
        .synchronize()
        .map_err(|error| state.failure(driver_error("JIT preparation synchronization", error)))?;
    state.mark_synchronized();
    mark_config_prepared(prepared_key, config);
    Ok(())
}

unsafe fn autotune_config(
    dev: &CudaDevice,
    layout: RoutedLoraMetadataLayout,
    key: CutileRoutedLoraTuningKey,
    addresses: KernelAddresses,
    launch: CutileRoutedLoraLaunch,
) -> CutileAutotuneResult<CutileRoutedLoraConfig> {
    let cached = config_cache().lock().unwrap().get(&key).copied();
    if let Some(config) = cached {
        prepare_config_on_thread(dev, layout, key, addresses, launch, config)?;
        return Ok(config);
    }
    let bucket_key = bucketed_tuning_key(key);
    let _tuning_guard = tuning_locks()[tuning_lock_index(bucket_key)]
        .lock()
        .unwrap();
    let cached = config_cache().lock().unwrap().get(&key).copied();
    if let Some(config) = cached {
        prepare_config_on_thread(dev, layout, key, addresses, launch, config)?;
        return Ok(config);
    }
    if let Some(config) = cached_bucket_config(key) {
        prepare_config_on_thread(dev, layout, key, addresses, launch, config)?;
        config_cache().lock().unwrap().insert(key, config);
        return Ok(config);
    }

    let stream = dev.cuda_stream();
    let mut state = CutileAutotuneState::FallbackSafe;
    let (mut scratch, scratch_offset) =
        unsafe { tuning_scratch(dev, layout, launch) }.map_err(|error| state.failure(error))?;
    let mut best = None;
    let mut last_error = None;
    for config in cutile_routed_lora_candidate_configs(key) {
        stream
            .memset_zeros(&mut scratch)
            .map_err(|error| state.failure(driver_error("scratch reset", error)))?;
        let (scratch_address, scratch_guard) =
            slice_ptr_mut_on_stream(&mut scratch, scratch_offset, &stream);
        let scratch_launch = CutileRoutedLoraLaunch {
            output: scratch_address,
            ..launch
        };
        match launch_config(dev, layout, addresses, scratch_launch, config) {
            Ok(()) => state.mark_scratch_launch(),
            Err(error) => {
                drop(scratch_guard);
                last_error = Some(error.to_string());
                continue;
            }
        }
        drop(scratch_guard);
        let completion = stream
            .record_event(None)
            .map_err(|error| state.failure(driver_error("warmup event", error)))?;
        completion
            .synchronize()
            .map_err(|error| state.failure(driver_error("warmup synchronization", error)))?;
        state.mark_synchronized();

        let mut timings = Vec::with_capacity(AUTOTUNE_TIMED_RUNS);
        let mut failed = false;
        for _ in 0..AUTOTUNE_TIMED_RUNS {
            stream
                .memset_zeros(&mut scratch)
                .map_err(|error| state.failure(driver_error("scratch reset", error)))?;
            let (scratch_address, scratch_guard) =
                slice_ptr_mut_on_stream(&mut scratch, scratch_offset, &stream);
            let scratch_launch = CutileRoutedLoraLaunch {
                output: scratch_address,
                ..launch
            };
            let start = stream
                .record_event(Some(
                    candle_core::cuda::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                ))
                .map_err(|error| state.failure(driver_error("start event", error)))?;
            if let Err(error) = launch_config(dev, layout, addresses, scratch_launch, config) {
                last_error = Some(error.to_string());
                failed = true;
                drop(scratch_guard);
                break;
            }
            state.mark_scratch_launch();
            let end = stream
                .record_event(Some(
                    candle_core::cuda::cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                ))
                .map_err(|error| state.failure(driver_error("end event", error)))?;
            drop(scratch_guard);
            let elapsed_ms = start
                .elapsed_ms(&end)
                .map_err(|error| state.failure(driver_error("event timing", error)))?;
            state.mark_synchronized();
            timings.push(elapsed_ms);
        }
        if failed {
            continue;
        }
        timings.sort_by(f32::total_cmp);
        let elapsed_ms = timings[timings.len() / 2];
        if best
            .as_ref()
            .is_none_or(|(_, best_ms)| elapsed_ms < *best_ms)
        {
            best = Some((config, elapsed_ms));
        }
    }
    let Some((config, elapsed_ms)) = best else {
        return Err(state.failure(candle_core::Error::msg(format!(
            "all cuTile routed LoRA autotune candidates failed: {}",
            last_error.unwrap_or_else(|| "no candidates".to_string())
        ))));
    };
    mark_config_prepared(prepared_key(key, addresses, launch), config);
    let mut cache = config_cache().lock().unwrap();
    if let Some(existing) = cache.get(&key).copied() {
        drop(cache);
        prepare_config_on_thread(dev, layout, key, addresses, launch, existing)?;
        return Ok(existing);
    }
    tracing::debug!(?config, elapsed_ms, "autotuned cuTile routed LoRA kernel");
    cache.insert(key, config);
    drop(cache);
    bucket_config_cache()
        .lock()
        .unwrap()
        .insert(bucket_key, config);
    Ok(config)
}

fn mark_failed(
    key: CutileRoutedLoraTuningKey,
    reason: CutileRoutedLoraUnsupported,
    operation: &str,
    error: &candle_core::Error,
    boundary: CutileFailureBoundary,
) -> CutileCachedFailure {
    let failure = CutileCachedFailure { reason, boundary };
    let (effective, cache_changed) = {
        let mut cache = failed_keys().lock().unwrap();
        let changed = cache_failure(&mut cache, key, failure);
        (*cache.get(&key).unwrap(), changed)
    };
    if cache_changed {
        match boundary {
            CutileFailureBoundary::FallbackSafe => {
                tracing::warn!(
                    "cuTile routed LoRA {operation} failed, using CUDA fallback: {error}"
                );
            }
            CutileFailureBoundary::ScratchLaunchAttempted => {
                tracing::warn!(
                    "cuTile routed LoRA {operation} failed after attempting a scratch launch: {error}"
                );
            }
            CutileFailureBoundary::OutputLaunchAttempted => {
                tracing::warn!(
                    "cuTile routed LoRA {operation} failed after attempting the output launch: {error}"
                );
            }
        }
    }
    effective
}

fn cache_failure(
    cache: &mut TuningCache<CutileCachedFailure>,
    key: CutileRoutedLoraTuningKey,
    failure: CutileCachedFailure,
) -> bool {
    if cache
        .get(&key)
        .is_some_and(|current| current.boundary.severity() >= failure.boundary.severity())
    {
        return false;
    }
    cache.insert(key, failure);
    true
}

fn clear_fallback_safe_failure(
    cache: &mut TuningCache<CutileCachedFailure>,
    key: CutileRoutedLoraTuningKey,
) {
    if cache
        .get(&key)
        .is_some_and(|failure| failure.boundary == CutileFailureBoundary::FallbackSafe)
    {
        cache.remove(&key);
    }
}

fn failure_result(
    boundary: CutileFailureBoundary,
    reason: CutileRoutedLoraUnsupported,
    error: candle_core::Error,
) -> Result<CutileRoutedLoraStatus> {
    match boundary {
        CutileFailureBoundary::FallbackSafe => Ok(CutileRoutedLoraStatus::Unsupported(reason)),
        CutileFailureBoundary::ScratchLaunchAttempted
        | CutileFailureBoundary::OutputLaunchAttempted => Err(error),
    }
}

fn cached_failure_result(failure: CutileCachedFailure) -> Result<CutileRoutedLoraStatus> {
    failure_result(
        failure.boundary,
        failure.reason,
        candle_core::Error::msg(format!(
            "cuTile routed LoRA is disabled for this shape after a prior {:?} failure",
            failure.reason
        )),
    )
}

fn pointer_aligned_or_null(address: u64, alignment: usize) -> bool {
    address == 0 || address.is_multiple_of(alignment as u64)
}

fn validate_launch_shape(
    layout: RoutedLoraMetadataLayout,
    weights: &RoutedLoraCudaWeightTable,
    launch: CutileRoutedLoraLaunch,
) -> Result<usize> {
    if weights.num_adapter_slots() != layout.num_adapter_slots()
        || launch
            .weight_slice_offset
            .checked_add(launch.projection.num_slices())
            .is_none_or(|end| end > weights.num_slices())
    {
        candle_core::bail!("cuTile routed LoRA launch and weight table shape mismatch");
    }
    if launch.input == 0 || launch.output == 0 {
        candle_core::bail!("cuTile routed LoRA input and output pointers must be non-null");
    }
    if !pointer_aligned_or_null(launch.input, std::mem::align_of::<bf16>())
        || !pointer_aligned_or_null(launch.output, std::mem::align_of::<bf16>())
        || !pointer_aligned_or_null(launch.route_input_rows, std::mem::align_of::<u32>())
        || !pointer_aligned_or_null(launch.route_output_rows, std::mem::align_of::<u32>())
        || !pointer_aligned_or_null(launch.route_output_scales, std::mem::align_of::<f32>())
    {
        candle_core::bail!("cuTile routed LoRA pointer alignment is invalid");
    }
    weights.validate_projection_rank(
        launch.weight_slice_offset,
        launch.projection.num_slices(),
        launch.projection.max_rank(),
    )?;
    Ok(weights
        .max_rank_stride_in_range(launch.weight_slice_offset, launch.projection.num_slices())
        .expect("validated cuTile routed LoRA descriptor slice range"))
}

unsafe fn try_cutile_routed_lora_with_addresses(
    dev: &CudaDevice,
    layout: RoutedLoraMetadataLayout,
    launch: CutileRoutedLoraLaunch,
    max_rank_stride: usize,
    addresses: KernelAddresses,
) -> Result<CutileRoutedLoraStatus> {
    let key = tuning_key(dev, layout, launch, max_rank_stride, addresses);
    if let Some(failure) = failed_keys().lock().unwrap().get(&key).copied() {
        return cached_failure_result(failure);
    }
    let config = match autotune_config(dev, layout, key, addresses, launch) {
        Ok(config) => config,
        Err(failure) => {
            let cached = mark_failed(
                key,
                CutileRoutedLoraUnsupported::AutotuneFailed,
                "autotune",
                &failure.error,
                failure.boundary,
            );
            return failure_result(cached.boundary, cached.reason, failure.error);
        }
    };
    if let Some(failure) = failed_keys().lock().unwrap().get(&key).copied() {
        return cached_failure_result(failure);
    }
    if let Err(error) = launch_config(dev, layout, addresses, launch, config) {
        let cached = mark_failed(
            key,
            CutileRoutedLoraUnsupported::LaunchFailed,
            "launch",
            &error,
            CutileFailureBoundary::OutputLaunchAttempted,
        );
        return failure_result(cached.boundary, cached.reason, error);
    }
    Ok(CutileRoutedLoraStatus::Launched)
}

/// Tries the one-shot cuTile path. Unsupported shapes leave CUDA dispatch to the caller.
/// The first shape in each power-of-two route/block bucket synchronously times candidates;
/// compatible exact shapes reuse that measured config, while exact manual overrides stay exact.
/// A new bucket probes at most nine configs with one warmup and three event-timed launches each.
///
/// # Safety
/// Raw pointers must reference live allocations on `dev` with the shapes described by the layouts.
/// Input and output regions must not overlap. Explicit output rows and `RoutedRows` input rows must
/// be less than `metadata.layout().num_routes()`. `TokenRows` input rows must be less than
/// `metadata.layout().num_tokens()`. Output regions for every output row and slice must be disjoint
/// so the in-place add and tuning scratch are safe.
pub unsafe fn try_cutile_routed_lora(
    dev: &CudaDevice,
    metadata: &RoutedLoraCudaMetadata,
    weights: &RoutedLoraCudaWeightTable,
    launch: CutileRoutedLoraLaunch,
) -> Result<CutileRoutedLoraStatus> {
    let layout = metadata.layout();
    let max_rank_stride = validate_launch_shape(layout, weights, launch)?;
    if let Some(reason) = unsupported(dev, layout, launch, max_rank_stride) {
        return Ok(CutileRoutedLoraStatus::Unsupported(reason));
    }

    let ordinal = dev.cuda_stream().context().ordinal();
    if weights.descriptors().ordinal() != ordinal
        || metadata.sorted_route_ids().ordinal() != ordinal
        || metadata.block_pair_ids().ordinal() != ordinal
    {
        candle_core::bail!("cuTile routed LoRA metadata is on a different CUDA device");
    }
    let stream = dev.cuda_stream();
    let descriptor_offset = launch
        .weight_slice_offset
        .checked_mul(layout.num_adapter_slots())
        .ok_or_else(|| candle_core::Error::msg("cuTile routed LoRA descriptor offset overflow"))?;
    let (descriptor_address, descriptor_guard) =
        slice_ptr_on_stream(weights.descriptors(), descriptor_offset, &stream);
    let (sorted_address, sorted_guard) =
        slice_ptr_on_stream(metadata.sorted_route_ids(), 0, &stream);
    let (pair_address, pair_guard) = slice_ptr_on_stream(metadata.block_pair_ids(), 0, &stream);
    let addresses = KernelAddresses {
        descriptors: descriptor_address,
        sorted_routes: sorted_address,
        block_pairs: pair_address,
        token_adapter_slots: 0,
        topk_expert_ids: 0,
        naive_assignment: false,
    };
    let status =
        try_cutile_routed_lora_with_addresses(dev, layout, launch, max_rank_stride, addresses)?;
    drop((descriptor_guard, sorted_guard, pair_guard));
    Ok(status)
}

/// Tries the no-sort one-shot cuTile path using route-major token and expert mappings.
/// Unsupported shapes leave CUDA dispatch to the caller without modifying `launch.output`.
///
/// # Safety
/// `launch.input` and `launch.output` must reference live allocations on `dev` with the shapes
/// described by `layout` and `launch.projection` and must not overlap. Explicit output rows and
/// `RoutedRows` input rows must be less than `layout.num_routes()`. `TokenRows` input rows must be
/// less than `layout.num_tokens()`. Every output row and slice region must be disjoint.
pub unsafe fn try_cutile_routed_lora_no_sort(
    dev: &CudaDevice,
    layout: RoutedLoraMetadataLayout,
    token_adapter_slots: Option<&CudaSlice<u32>>,
    topk_expert_ids: &CudaSlice<u32>,
    topk_expert_ids_offset: usize,
    weights: &RoutedLoraCudaWeightTable,
    launch: CutileRoutedLoraLaunch,
) -> Result<CutileRoutedLoraStatus> {
    let max_rank_stride = validate_launch_shape(layout, weights, launch)?;
    if let Some(reason) = unsupported(dev, layout, launch, max_rank_stride) {
        return Ok(CutileRoutedLoraStatus::Unsupported(reason));
    }
    if token_adapter_slots.is_none() && layout.num_adapter_slots() != 1 {
        candle_core::bail!(
            "null cuTile routed LoRA token slots require one adapter descriptor slot"
        );
    }
    if token_adapter_slots.is_some_and(|slots| slots.len() < layout.num_tokens()) {
        candle_core::bail!("cuTile routed LoRA token slot buffer is too small");
    }
    if topk_expert_ids_offset
        .checked_add(layout.num_routes())
        .is_none_or(|end| end > topk_expert_ids.len())
    {
        candle_core::bail!("cuTile routed LoRA expert ID buffer is too small");
    }

    let ordinal = dev.cuda_stream().context().ordinal();
    if weights.descriptors().ordinal() != ordinal
        || token_adapter_slots.is_some_and(|slots| slots.ordinal() != ordinal)
        || topk_expert_ids.ordinal() != ordinal
    {
        candle_core::bail!("cuTile routed LoRA no-sort inputs are on a different CUDA device");
    }
    let stream = dev.cuda_stream();
    let descriptor_offset = launch
        .weight_slice_offset
        .checked_mul(layout.num_adapter_slots())
        .ok_or_else(|| candle_core::Error::msg("cuTile routed LoRA descriptor offset overflow"))?;
    let (descriptor_address, descriptor_guard) =
        slice_ptr_on_stream(weights.descriptors(), descriptor_offset, &stream);
    let (topk_address, topk_guard) =
        slice_ptr_on_stream(topk_expert_ids, topk_expert_ids_offset, &stream);
    let (token_slots_address, token_slots_guard) = match token_adapter_slots {
        Some(slots) => {
            let (address, guard) = slice_ptr_on_stream(slots, 0, &stream);
            (address, Some(guard))
        }
        None => (0, None),
    };
    let addresses = KernelAddresses {
        descriptors: descriptor_address,
        sorted_routes: 0,
        block_pairs: 0,
        token_adapter_slots: token_slots_address,
        topk_expert_ids: topk_address,
        naive_assignment: true,
    };
    let status =
        try_cutile_routed_lora_with_addresses(dev, layout, launch, max_rank_stride, addresses)?;
    drop((descriptor_guard, topk_guard, token_slots_guard));
    Ok(status)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> CutileRoutedLoraTuningKey {
        CutileRoutedLoraTuningKey {
            device: CutileRoutedLoraDeviceKey {
                ordinal: 0,
                compute_major: 12,
                compute_minor: 0,
                multiprocessors: 148,
            },
            shape: CutileRoutedLoraShapeKey {
                top_k: 8,
                num_routes: 64,
                max_blocks: 64,
                route_block_size: 16,
                num_experts: 128,
                num_adapter_slots: 4,
                input_features: 4096,
                output_features: 14336,
                output_row_stride: 28672,
                output_slice_stride: 14336,
                num_slices: 2,
                weight_slice_offset: 0,
                max_rank: 96,
                max_rank_stride: 128,
                input_mode: RoutedLoraInputMode::TokenRows as i32,
                has_input_rows: false,
                has_output_rows: false,
                has_route_scales: false,
                naive_assignment: false,
                has_token_adapter_slots: false,
            },
        }
    }

    #[test]
    fn candidates_cover_kernel_axes_and_hints() {
        let candidates = cutile_routed_lora_candidate_configs(key());
        assert_eq!(candidates.len(), 9);
        assert_eq!(candidates[0].block_k, 128);
        assert!(candidates.iter().any(|config| config.block_n == 64));
        assert!(candidates.iter().any(|config| config.n_axis_groups > 1));
        assert!(candidates.iter().any(|config| {
            config.optimization_hint == CutileRoutedLoraOptimizationHint::Cluster2
        }));
        assert!(candidates.iter().all(|config| valid_config(key(), *config)));
    }

    #[test]
    fn fallback_is_allowed_only_at_a_safe_boundary() {
        let safe = failure_result(
            CutileFailureBoundary::FallbackSafe,
            CutileRoutedLoraUnsupported::AutotuneFailed,
            candle_core::Error::msg("autotune failed"),
        );
        assert_eq!(
            safe.unwrap(),
            CutileRoutedLoraStatus::Unsupported(CutileRoutedLoraUnsupported::AutotuneFailed)
        );

        for boundary in [
            CutileFailureBoundary::ScratchLaunchAttempted,
            CutileFailureBoundary::OutputLaunchAttempted,
        ] {
            let attempted = failure_result(
                boundary,
                CutileRoutedLoraUnsupported::LaunchFailed,
                candle_core::Error::msg("launch failed"),
            );
            assert!(attempted.unwrap_err().to_string().contains("launch failed"));
        }
    }

    #[test]
    fn autotune_state_is_safe_again_only_after_synchronization() {
        let mut state = CutileAutotuneState::FallbackSafe;
        assert_eq!(
            state.failure_boundary(),
            CutileFailureBoundary::FallbackSafe
        );
        state.mark_scratch_launch();
        assert_eq!(
            state.failure_boundary(),
            CutileFailureBoundary::ScratchLaunchAttempted
        );
        state.mark_synchronized();
        assert_eq!(
            state.failure_boundary(),
            CutileFailureBoundary::FallbackSafe
        );
    }

    #[test]
    fn cached_fatal_failures_do_not_enable_fallback() {
        for boundary in [
            CutileFailureBoundary::ScratchLaunchAttempted,
            CutileFailureBoundary::OutputLaunchAttempted,
        ] {
            let failure = CutileCachedFailure {
                reason: CutileRoutedLoraUnsupported::LaunchFailed,
                boundary,
            };
            let mut cache = BoundedCache::new();
            cache.insert(key(), failure);
            assert!(cached_failure_result(*cache.get(&key()).unwrap()).is_err());
        }

        let safe = CutileCachedFailure {
            reason: CutileRoutedLoraUnsupported::AutotuneFailed,
            boundary: CutileFailureBoundary::FallbackSafe,
        };
        assert_eq!(
            cached_failure_result(safe).unwrap(),
            CutileRoutedLoraStatus::Unsupported(CutileRoutedLoraUnsupported::AutotuneFailed)
        );
    }

    #[test]
    fn cached_failure_boundaries_never_downgrade() {
        let tuning_key = key();
        let safe = CutileCachedFailure {
            reason: CutileRoutedLoraUnsupported::AutotuneFailed,
            boundary: CutileFailureBoundary::FallbackSafe,
        };
        let scratch = CutileCachedFailure {
            reason: CutileRoutedLoraUnsupported::AutotuneFailed,
            boundary: CutileFailureBoundary::ScratchLaunchAttempted,
        };
        let output = CutileCachedFailure {
            reason: CutileRoutedLoraUnsupported::LaunchFailed,
            boundary: CutileFailureBoundary::OutputLaunchAttempted,
        };
        let mut cache = BoundedCache::new();

        assert!(cache_failure(&mut cache, tuning_key, scratch));
        assert!(!cache_failure(&mut cache, tuning_key, safe));
        assert_eq!(cache.get(&tuning_key), Some(&scratch));
        assert!(cache_failure(&mut cache, tuning_key, output));
        assert!(!cache_failure(&mut cache, tuning_key, scratch));
        assert_eq!(cache.get(&tuning_key), Some(&output));
        assert!(cached_failure_result(*cache.get(&tuning_key).unwrap()).is_err());
        clear_fallback_safe_failure(&mut cache, tuning_key);
        assert_eq!(cache.get(&tuning_key), Some(&output));

        let mut safe_key = tuning_key;
        safe_key.shape.num_routes += 1;
        assert!(cache_failure(&mut cache, safe_key, safe));
        clear_fallback_safe_failure(&mut cache, safe_key);
        assert!(cache.get(&safe_key).is_none());
    }

    #[test]
    fn prepared_configs_are_thread_local_and_config_specific() {
        let mut tuning_key = key();
        tuning_key.device.ordinal = usize::MAX - 1;
        let prepared_key = CutilePreparedKey {
            tuning: tuning_key,
            pointer_divisors: [16; 10],
        };
        let mut other_alignment = prepared_key;
        other_alignment.pointer_divisors[1] = 8;
        let config = cutile_routed_lora_candidate_configs(tuning_key)[0];
        let alternate = CutileRoutedLoraConfig {
            block_k: 192 - config.block_k,
            ..config
        };
        assert!(!prepared_config(prepared_key, config));
        mark_config_prepared(prepared_key, config);
        assert!(prepared_config(prepared_key, config));
        assert!(!prepared_config(prepared_key, alternate));
        assert!(!prepared_config(other_alignment, config));
        assert!(
            !std::thread::spawn(move || prepared_config(prepared_key, config))
                .join()
                .unwrap()
        );
    }

    #[test]
    fn scratch_offsets_match_output_pointer_specialization() {
        let base = 0x1000u64;
        for output in [0x2000u64, 0x2008, 0x2004, 0x2002] {
            let offset = scratch_output_offset(base, output).unwrap();
            let scratch = base + (offset * std::mem::size_of::<bf16>()) as u64;
            assert_eq!(
                DivHint::from_ptr(scratch).divisor,
                DivHint::from_ptr(output).divisor
            );
        }
        assert!(scratch_output_offset(base, 0x2001).is_none());
    }

    #[test]
    fn preferred_groups_stay_on_supported_axes_for_odd_tile_limits() {
        let mut tuning_key = key();
        tuning_key.shape.input_features = 64;
        tuning_key.shape.output_features = 384;
        tuning_key.shape.output_row_stride = 768;
        tuning_key.shape.output_slice_stride = 384;
        assert_eq!(preferred_groups(tuning_key), 2);
        let candidates = cutile_routed_lora_candidate_configs(tuning_key);
        assert!(candidates.iter().any(|config| {
            config.optimization_hint == CutileRoutedLoraOptimizationHint::HighOccupancy
        }));
        assert!(candidates
            .iter()
            .all(|config| valid_config(tuning_key, *config)));
    }

    #[test]
    fn route_block_buckets_follow_power_of_two_boundaries() {
        for (value, expected) in [(0, 1), (1, 1), (2, 2), (3, 4), (4, 4), (5, 8)] {
            assert_eq!(power_of_two_bucket(value), expected);
        }
        let mut lower = key();
        lower.shape.num_routes = 65;
        lower.shape.max_blocks = 17;
        let mut upper = lower;
        upper.shape.num_routes = 128;
        upper.shape.max_blocks = 32;
        assert_eq!(bucketed_tuning_key(lower), bucketed_tuning_key(upper));
        upper.shape.num_routes += 1;
        assert_ne!(bucketed_tuning_key(lower), bucketed_tuning_key(upper));
    }

    #[test]
    fn route_buckets_keep_device_and_static_shape_separate() {
        let baseline = bucketed_tuning_key(key());
        let mut other_device = key();
        other_device.device.ordinal += 1;
        assert_ne!(baseline, bucketed_tuning_key(other_device));
        let mut other_shape = key();
        other_shape.shape.output_features += 1;
        assert_ne!(baseline, bucketed_tuning_key(other_shape));
        let mut other_slice = key();
        other_slice.shape.weight_slice_offset += 1;
        assert_ne!(baseline, bucketed_tuning_key(other_slice));
    }

    #[test]
    fn no_sort_and_grouped_tuning_keys_are_separate() {
        let grouped = key();
        let mut no_sort = grouped;
        no_sort.shape.max_blocks = no_sort.shape.num_routes;
        no_sort.shape.naive_assignment = true;
        no_sort.shape.has_token_adapter_slots = true;
        assert_ne!(bucketed_tuning_key(grouped), bucketed_tuning_key(no_sort));

        let mut implicit_single_adapter = no_sort;
        implicit_single_adapter.shape.has_token_adapter_slots = false;
        assert_ne!(
            bucketed_tuning_key(no_sort),
            bucketed_tuning_key(implicit_single_adapter)
        );
    }

    #[test]
    fn bucket_reuse_rejects_invalid_cluster_config() {
        let mut representative = key();
        representative.device.ordinal = usize::MAX;
        representative.shape.max_blocks = 4;
        let config = CutileRoutedLoraConfig {
            block_k: 128,
            block_n: 128,
            n_axis_groups: 1,
            optimization_hint: CutileRoutedLoraOptimizationHint::Cluster2,
        };
        assert!(valid_config(representative, config));
        bucket_config_cache()
            .lock()
            .unwrap()
            .insert(bucketed_tuning_key(representative), config);
        let mut odd_grid = representative;
        odd_grid.shape.max_blocks = 3;
        assert_eq!(
            bucketed_tuning_key(representative),
            bucketed_tuning_key(odd_grid)
        );
        assert_eq!(cached_bucket_config(representative), Some(config));
        assert_eq!(cached_bucket_config(odd_grid), None);
        bucket_config_cache()
            .lock()
            .unwrap()
            .remove(&bucketed_tuning_key(representative));
    }

    #[test]
    fn tuned_config_round_trips_through_shape_device_cache() -> Result<()> {
        let mut tuning_key = key();
        tuning_key.shape.output_features += 1;
        let bucket_config = CutileRoutedLoraConfig {
            block_k: 128,
            block_n: 128,
            n_axis_groups: 1,
            optimization_hint: CutileRoutedLoraOptimizationHint::Balanced,
        };
        bucket_config_cache()
            .lock()
            .unwrap()
            .insert(bucketed_tuning_key(tuning_key), bucket_config);
        assert_eq!(
            selected_cutile_routed_lora_config(tuning_key),
            Some(bucket_config)
        );
        let config = CutileRoutedLoraConfig {
            block_k: 64,
            block_n: 64,
            n_axis_groups: 2,
            optimization_hint: CutileRoutedLoraOptimizationHint::HighOccupancy,
        };
        set_cutile_routed_lora_tuned_config(tuning_key, config)?;
        assert_eq!(cached_cutile_routed_lora_config(tuning_key), Some(config));
        assert_eq!(selected_cutile_routed_lora_config(tuning_key), Some(config));
        bucket_config_cache()
            .lock()
            .unwrap()
            .remove(&bucketed_tuning_key(tuning_key));
        Ok(())
    }

    #[test]
    fn tuning_caches_evict_the_oldest_shape_at_capacity() {
        let first = key();
        let mut cache = BoundedCache::new();
        for ordinal in 0..=TUNING_CACHE_CAPACITY {
            let mut tuning_key = first;
            tuning_key.device.ordinal = ordinal;
            cache.insert(tuning_key, ordinal);
        }
        let mut latest = first;
        latest.device.ordinal = TUNING_CACHE_CAPACITY;
        assert_eq!(cache.values.len(), TUNING_CACHE_CAPACITY);
        assert!(cache.get(&first).is_none());
        assert_eq!(cache.get(&latest), Some(&TUNING_CACHE_CAPACITY));
        cache.insert(latest, TUNING_CACHE_CAPACITY + 1);
        assert_eq!(cache.values.len(), TUNING_CACHE_CAPACITY);
    }

    #[test]
    fn invalid_cluster_config_is_rejected() {
        let mut tuning_key = key();
        tuning_key.shape.max_blocks = 3;
        let config = CutileRoutedLoraConfig {
            block_k: 128,
            block_n: 128,
            n_axis_groups: 1,
            optimization_hint: CutileRoutedLoraOptimizationHint::Cluster2,
        };
        assert!(set_cutile_routed_lora_tuned_config(tuning_key, config).is_err());
    }

    #[test]
    fn configs_reject_grids_beyond_i32_block_ids() {
        let mut tuning_key = key();
        tuning_key.shape.max_blocks = i32::MAX as usize;
        let single_group = CutileRoutedLoraConfig {
            block_k: 128,
            block_n: 128,
            n_axis_groups: 1,
            optimization_hint: CutileRoutedLoraOptimizationHint::Balanced,
        };
        let two_groups = CutileRoutedLoraConfig {
            n_axis_groups: 2,
            ..single_group
        };
        assert!(valid_config(tuning_key, single_group));

        tuning_key.shape.max_blocks = i32::MAX as usize + 1;
        assert!(!valid_config(tuning_key, single_group));

        tuning_key.shape.max_blocks = i32::MAX as usize / 2 + 1;
        assert!(!valid_config(tuning_key, two_groups));
    }

    #[test]
    fn rank_blocks_match_one_shot_register_tiles() {
        assert_eq!(rank_block(4), 16);
        assert_eq!(rank_block(16), 16);
        assert_eq!(rank_block(17), 32);
        assert_eq!(rank_block(96), 128);
    }

    #[test]
    fn addressing_gate_accepts_i32_boundary_and_rejects_overflow() -> Result<()> {
        let expert_layout = RoutedLoraMetadataLayout::new(1, 1, 128, 1)?;
        let expert_boundary = RoutedLoraProjectionLayout::new(
            131_072,
            64,
            64,
            0,
            1,
            128,
            RoutedLoraInputMode::RoutedRows,
        )?;
        let expert_overflow = RoutedLoraProjectionLayout::new(
            131_073,
            64,
            64,
            0,
            1,
            128,
            RoutedLoraInputMode::RoutedRows,
        )?;
        assert!(addressing_supported(expert_layout, expert_boundary, 128));
        assert!(!addressing_supported(expert_layout, expert_overflow, 128));

        let row_layout = RoutedLoraMetadataLayout::new(65_536, 1, 1, 1)?;
        let row_boundary = RoutedLoraProjectionLayout::new(
            1,
            1,
            32_768,
            0,
            1,
            1,
            RoutedLoraInputMode::RoutedRows,
        )?;
        let row_overflow = RoutedLoraProjectionLayout::new(
            1,
            1,
            32_769,
            0,
            1,
            1,
            RoutedLoraInputMode::RoutedRows,
        )?;
        assert!(addressing_supported(row_layout, row_boundary, 1));
        assert!(!addressing_supported(row_layout, row_overflow, 1));

        let descriptor_bytes = std::mem::size_of::<crate::lora::RoutedLoraAdapterWeight>();
        let descriptor_boundary_slots = I32_INDEXABLE_ELEMENTS / descriptor_bytes;
        let descriptor_boundary =
            RoutedLoraMetadataLayout::new(1, 1, 1, descriptor_boundary_slots)?;
        let descriptor_overflow =
            RoutedLoraMetadataLayout::new(1, 1, 1, descriptor_boundary_slots + 1)?;
        let descriptor_projection =
            RoutedLoraProjectionLayout::new(1, 1, 1, 0, 1, 1, RoutedLoraInputMode::RoutedRows)?;
        assert!(addressing_supported(
            descriptor_boundary,
            descriptor_projection,
            1
        ));
        assert!(!addressing_supported(
            descriptor_overflow,
            descriptor_projection,
            1
        ));
        Ok(())
    }

    #[test]
    fn feature_ceil_div_handles_i32_boundary() -> Result<()> {
        let max_feature = i32::MAX;
        for block in [64, 128] {
            let tiles = 1 + (max_feature - 1) / block;
            assert_eq!(
                tiles as usize,
                (max_feature as usize).div_ceil(block as usize)
            );
        }

        let layout = RoutedLoraMetadataLayout::new(1, 1, 1, 1)?;
        let boundary = RoutedLoraProjectionLayout::new(
            i32::MAX as usize,
            1,
            1,
            0,
            1,
            1,
            RoutedLoraInputMode::RoutedRows,
        )?;
        assert!(addressing_supported(layout, boundary, 1));
        assert!(RoutedLoraProjectionLayout::new(
            i32::MAX as usize + 1,
            1,
            1,
            0,
            1,
            1,
            RoutedLoraInputMode::RoutedRows,
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn compact_descriptor_offsets_match_kernel_loads() {
        use crate::lora::RoutedLoraAdapterWeight;

        assert_eq!(std::mem::size_of::<RoutedLoraAdapterWeight>(), 40);
        assert_eq!(std::mem::offset_of!(RoutedLoraAdapterWeight, a), 0);
        assert_eq!(std::mem::offset_of!(RoutedLoraAdapterWeight, b), 8);
        assert_eq!(std::mem::offset_of!(RoutedLoraAdapterWeight, scales), 16);
        assert_eq!(std::mem::offset_of!(RoutedLoraAdapterWeight, rank), 24);
        assert_eq!(
            std::mem::offset_of!(RoutedLoraAdapterWeight, rank_stride),
            28
        );
        assert_eq!(std::mem::offset_of!(RoutedLoraAdapterWeight, scale), 32);
    }

    #[test]
    fn one_shot_kernel_compiles_to_tile_ir() {
        use cutile::compile_api::KernelCompiler;

        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                for generics in [
                    ["16", "64", "64", "1", "0", "0", "0", "1", "0", "0"],
                    ["128", "128", "128", "8", "1", "1", "1", "0", "0", "0"],
                    ["128", "128", "128", "4", "0", "0", "1", "0", "1", "1"],
                ] {
                    let artifacts = KernelCompiler::new(
                        routed_lora_kernel::__module_ast_self,
                        "routed_lora_kernel",
                        "routed_lora_one_shot",
                    )
                    .generics(generics.into_iter().map(str::to_string).collect())
                    .target("sm_80")
                    .compile()
                    .expect("compile routed LoRA Tile IR");
                    assert!(artifacts.ir_text().contains("entry"));
                    assert!(!artifacts.bytecode().expect("serialize Tile IR").is_empty());
                }
            })
            .expect("spawn Tile IR compiler")
            .join()
            .expect("join Tile IR compiler");
    }
}
