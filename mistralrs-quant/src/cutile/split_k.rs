//! Second pass of a split-K grouped GEMM: sums the f32 partial slices into the bf16 output.

use candle_core::cuda::cudarc::driver::CudaSlice;
use candle_core::{CudaDevice, Result};
use cutile::cuda_async::device_buffer::DevicePointer;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tile_kernel::TileKernel;
use half::bf16;

use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

use super::{catch_cutile_panic, context};

const REDUCE_BLOCK: i32 = 1024;

#[cutile::module]
pub mod split_k {
    use cutile::core::*;
    use cutile::cutile_compiler;

    #[cutile::entry(unchecked_accesses = true)]
    pub unsafe fn split_k_reduce_kernel<const BLOCK: i32, const SPLITS: i32>(
        out_ptr: *mut bf16,    // [numel]
        partial_ptr: *mut f32, // [SPLITS, numel]
        numel: i32,
    ) {
        let pid: i32 = get_tile_block_id().0;
        let iota_b: Tile<i32, { [BLOCK] }> = iota(const_shape![BLOCK]);
        let base: Tile<i32, { [BLOCK] }> = broadcast_scalar(pid * BLOCK, const_shape![BLOCK]);
        let offs: Tile<i32, { [BLOCK] }> = iota_b + base;
        let numel_t: Tile<i32, { [BLOCK] }> = broadcast_scalar(numel, const_shape![BLOCK]);
        let mask: Tile<bool, { [BLOCK] }> = lt_tile(offs, numel_t);
        let p0: PointerTile<*mut f32, { [] }> = pointer_to_tile(partial_ptr);
        let p1: PointerTile<*mut f32, { [1] }> = p0.reshape(const_shape![1]);
        let p2: PointerTile<*mut f32, { [BLOCK] }> = p1.broadcast(const_shape![BLOCK]);
        let mut acc: Tile<f32, { [BLOCK] }> = constant(0.0f32, const_shape![BLOCK]);
        let mut off: Tile<i32, { [BLOCK] }> = offs;
        for _s in 0i32..SPLITS {
            let ptrs: PointerTile<*mut f32, { [BLOCK] }> = p2.offset_tile(off);
            let (v, _): (Tile<f32, { [BLOCK] }>, Token) = load_ptr_tko(
                ptrs,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                Some(0.0f32),
                None,
                Latency::<0>,
            );
            acc = acc + v;
            off = off + numel_t;
        }
        let o0: PointerTile<*mut bf16, { [] }> = pointer_to_tile(out_ptr);
        let o1: PointerTile<*mut bf16, { [1] }> = o0.reshape(const_shape![1]);
        let o2: PointerTile<*mut bf16, { [BLOCK] }> = o1.broadcast(const_shape![BLOCK]);
        let o_ptrs: PointerTile<*mut bf16, { [BLOCK] }> = o2.offset_tile(offs);
        let acc_bf: Tile<bf16, { [BLOCK] }> = convert_tile(acc);
        store_ptr_tko(
            o_ptrs,
            acc_bf,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }
}

/// Sums `splits` slices of `numel` f32 partials into `out`.
pub(super) fn reduce_split_k(
    partial: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
    splits: i32,
    numel: usize,
    dev: &CudaDevice,
    compile_only: bool,
) -> Result<()> {
    let stream = dev.cuda_stream();
    let (partial_addr, _partial_guard) = slice_ptr_on_stream(partial, 0, &stream);
    let (out_addr, out_guard) = slice_ptr_mut_on_stream(out, 0, &stream);
    let grid_x = numel.div_ceil(REDUCE_BLOCK as usize) as u32;
    let cutile_stream = context::stream(dev);
    let launcher = unsafe {
        split_k::split_k_reduce_kernel(
            DevicePointer::<bf16>::from_cu_deviceptr(out_addr as CUdeviceptr),
            DevicePointer::<f32>::from_cu_deviceptr(partial_addr as CUdeviceptr),
            numel as i32,
        )
    }
    .generics(vec![REDUCE_BLOCK.to_string(), splits.to_string()])
    .grid((grid_x, 1, 1));
    if compile_only {
        catch_cutile_panic("split-K reduce compile", || {
            launcher
                .compile_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile split_k compile: {e:?}")))
        })?;
    } else {
        catch_cutile_panic("split-K reduce execute", || unsafe {
            launcher
                .async_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile split_k launch: {e:?}")))
        })?;
    }
    drop(out_guard);
    Ok(())
}
