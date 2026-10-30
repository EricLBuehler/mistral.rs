#![allow(
    clippy::too_many_arguments,
    reason = "cuTile kernel arguments follow the device ABI"
)]

#[cutile::module]
mod kernels {
    use cutile::core::*;

    const BLOCK: i32 = 16;
    const FP4_MAX: f32 = 6.0;
    const FP4_MIN: f32 = -6.0;
    const FP8_MAX: f32 = 448.0;

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn quantize<
        T: ElementType,
        const BM: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
    >(
        mut q: MappedPartitionMut<f4e2m1fnx2, { [BM, PK] }, { [1, 1] }>,
        s: *mut f8e4m3fn,
        rows: i32,
        scale_stride: i32,
        x: &Tensor<T, { [-1, -1] }>,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [0]);
        for idx in q.iter_indices() {
            let (m, kg) = idx.components();
            let xt: Tile<T, { [BM, BK] }> = px.load([m, kg]);
            let xf: Tile<f32, { [BM, BK] }> = convert_tile(xt);
            let xf = xf
                / global
                    .reshape(const_shape![1, 1])
                    .broadcast(const_shape![BM, BK]);
            let blocks: Tile<f32, { [BM, SK, BLOCK] }> = xf.reshape(const_shape![BM, SK, BLOCK]);
            let maxima: Tile<f32, { [BM, SK] }> = reduce_max(absf(blocks), 2);
            let maxima = maxima.reshape(const_shape![BM, SK]);
            let fp4_max: Tile<f32, { [BM, SK] }> = constant(FP4_MAX, const_shape![BM, SK]);
            let fp8_max: Tile<f32, { [BM, SK] }> = constant(FP8_MAX, const_shape![BM, SK]);
            let zeros: Tile<f32, { [BM, SK] }> = constant(0.0f32, const_shape![BM, SK]);
            let ones: Tile<f32, { [BM, SK] }> = constant(1.0f32, const_shape![BM, SK]);
            let upper: Tile<f32, { [BM, BK] }> = constant(FP4_MAX, const_shape![BM, BK]);
            let lower: Tile<f32, { [BM, BK] }> = constant(FP4_MIN, const_shape![BM, BK]);
            let raw_scale = min_tile(maxima / fp4_max, fp8_max);
            let sx: Tile<f8e4m3fn, { [BM, SK] }> = convert_tile(raw_scale);
            let rounded: Tile<f32, { [BM, SK] }> = convert_tile(sx);
            let denominator = select(eq_tile(rounded, zeros), ones, rounded);
            let denominator = denominator
                .reshape(const_shape![BM, SK, 1])
                .broadcast(const_shape![BM, SK, BLOCK]);
            let normalized = (blocks / denominator).reshape(const_shape![BM, BK]);
            let normalized = max_tile(min_tile(normalized, upper), lower);
            let xq: Tile<f4e2m1fn, { [BM, BK] }> = convert_tile(normalized);
            let packed: Tile<f4e2m1fnx2, { [BM, PK] }> = xq.pack(const_shape![BM, PK]);
            q.store(packed, idx);
            let row: Tile<i32, { [BM] }> =
                iota(const_shape![BM]) + broadcast_scalar(m * BM, const_shape![BM]);
            let col: Tile<i32, { [SK] }> =
                iota(const_shape![SK]) + broadcast_scalar(kg * SK, const_shape![SK]);
            let row_offset: Tile<i64, { [BM] }> = exti(row);
            let stride: Tile<i64, { [BM] }> =
                exti(broadcast_scalar(scale_stride, const_shape![BM]));
            let col_offset: Tile<i64, { [SK] }> = exti(col);
            let offset = (row_offset * stride)
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, SK])
                + col_offset
                    .reshape(const_shape![1, SK])
                    .broadcast(const_shape![BM, SK]);
            let mask = lt_tile(row, broadcast_scalar(rows, const_shape![BM]))
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, SK])
                & lt_tile(col, broadcast_scalar(scale_stride, const_shape![SK]))
                    .reshape(const_shape![1, SK])
                    .broadcast(const_shape![BM, SK]);
            let base: PointerTile<*mut f8e4m3fn, { [] }> = pointer_to_tile(s);
            let base: PointerTile<*mut f8e4m3fn, { [1, 1] }> = base.reshape(const_shape![1, 1]);
            let base: PointerTile<*mut f8e4m3fn, { [BM, SK] }> =
                base.broadcast(const_shape![BM, SK]);
            let ptrs: PointerTile<*mut f8e4m3fn, { [BM, SK] }> = base.offset_tile(offset);
            store_ptr_tko(
                ptrs,
                sx,
                ordering::Weak,
                None::<scope::TileBlock>,
                Some(mask),
                None,
                Latency::<0>,
            );
        }
    }

    #[cutile::entry(unchecked_accesses = false)]
    fn matmul<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const A4: bool,
        const LATENCY: i32,
        const GROUP_M: i32,
    >(
        mut y: MappedPartitionMut<T, { [BM, BN] }, { [GROUP_M, 1] }>,
        x: &Tensor<T, { [-1, -1] }>,
        q: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        qs: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        ws: &Tensor<f8e4m3fn, { [-1, -1] }>,
        wg: &Tensor<f32, { [-1] }>,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pq = q.partition(const_shape![BM, PK]);
        let pqs = qs.partition(const_shape![BM, SK]);
        let pw = w.partition(const_shape![BN, PK]);
        let pws = ws.partition(const_shape![BN, SK]);
        let pwg = wg.partition(const_shape![BN]);
        let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [0]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let k = num_tiles(&pw, 1);
        for idx in y.iter_indices() {
            let (m, n) = idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let packed: Tile<f4e2m1fnx2, { [BN, PK] }> = pw.load_pipelined::<LATENCY>([n, kg]);
                let wt: Tile<f4e2m1fn, { [BN, BK] }> = packed.unpack(const_shape![BN, BK]);
                let sw: Tile<f8e4m3fn, { [BN, SK] }> = pws.load_pipelined::<LATENCY>([n, kg]);
                if A4 {
                    let xp: Tile<f4e2m1fnx2, { [BM, PK] }> = pq.load_pipelined::<LATENCY>([m, kg]);
                    let xq: Tile<f4e2m1fn, { [BM, BK] }> = xp.unpack(const_shape![BM, BK]);
                    let sx: Tile<f8e4m3fn, { [BM, SK] }> = pqs.load_pipelined::<LATENCY>([m, kg]);
                    let wt: Tile<f4e2m1fn, { [BK, BN] }> = permute(wt, transpose);
                    let sw: Tile<f8e4m3fn, { [SK, BN] }> = permute(sw, transpose);
                    acc = mmaf_scaled(xq, wt, acc, sx, sw);
                } else {
                    let xt: Tile<T, { [BM, BK] }> = px.load_pipelined::<LATENCY>([m, kg]);
                    let wf: Tile<f32, { [BN, BK] }> = convert_tile(wt);
                    let sw: Tile<f32, { [BN, SK] }> = convert_tile(sw);
                    let sw = sw
                        .reshape(const_shape![BN, SK, 1])
                        .broadcast(const_shape![BN, SK, BLOCK])
                        .reshape(const_shape![BN, BK]);
                    let wd: Tile<T, { [BN, BK] }> = convert_tile(wf * sw);
                    let wd: Tile<T, { [BK, BN] }> = permute(wd, transpose);
                    acc = mmaf(xt, wd, acc);
                }
            }
            let weight_global: Tile<f32, { [BN] }> = pwg.load([n]);
            let weight_global = weight_global
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BM, BN]);
            let result = acc * weight_global;
            let result = if A4 {
                result
                    * global
                        .reshape(const_shape![1, 1])
                        .broadcast(const_shape![BM, BN])
            } else {
                result
            };
            let out: Tile<T, { [BM, BN] }> = convert_tile(result);
            y.store(out, idx);
        }
    }

    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn routed_matmul<
        T: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const PK: i32,
        const SK: i32,
        const A4: bool,
        const LATENCY: i32,
        const PERSISTENT: bool,
    >(
        y: *mut T,
        sids: &Tensor<i32, { [-1] }>,
        eids: &Tensor<i32, { [-1] }>,
        ntpp: &Tensor<i32, { [-1] }>,
        routes: i32,
        n_size: i32,
        x: &Tensor<T, { [-1, -1] }>,
        q: &Tensor<f4e2m1fnx2, { [-1, -1] }>,
        qs: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w: &Tensor<f4e2m1fnx2, { [-1, -1, -1] }>,
        ws: &Tensor<f8e4m3fn, { [-1, -1, -1] }>,
        wg: &Tensor<f32, { [-1, -1] }>,
        ag: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pq = q.partition(const_shape![BM, PK]);
        let pqs = qs.partition(const_shape![BM, SK]);
        let pw = w.partition(const_shape![1, BN, PK]);
        let pws = ws.partition(const_shape![1, BN, SK]);
        let pwg = wg.partition(const_shape![1, BN]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let k = num_tiles(&pw, 2);
        let block_id = get_tile_block_id().0;
        let workers = get_num_tile_blocks().0;
        let ntiles = ceil_div(n_size, BN);
        let aligned: Tile<i32, { [1] }> = load_tile(ntpp, const_shape![1], [0]);
        let aligned: i32 = tile_to_scalar(aligned.reshape(const_shape![]));
        let rounds = if PERSISTENT {
            ceil_div(ceil_div(aligned, BM) * ntiles, workers)
        } else {
            1i32
        };
        for round in 0..rounds {
            let pid = block_id + round * workers;
            let m = pid / ntiles;
            let n = pid % ntiles;
            if m * BM < aligned {
                let expert: Tile<i32, { [1] }> = load_tile(eids, const_shape![1], [m]);
                let expert: i32 = tile_to_scalar(expert.reshape(const_shape![]));
                let global: Tile<f32, { [1] }> = load_tile(ag, const_shape![1], [expert]);
                let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                for kg in 0..k {
                    let packed: Tile<f4e2m1fnx2, { [BN, PK] }> = pw
                        .load_pipelined::<LATENCY>([expert, n, kg])
                        .reshape(const_shape![BN, PK]);
                    let wt: Tile<f4e2m1fn, { [BN, BK] }> = packed.unpack(const_shape![BN, BK]);
                    let sw: Tile<f8e4m3fn, { [BN, SK] }> = pws
                        .load_pipelined::<LATENCY>([expert, n, kg])
                        .reshape(const_shape![BN, SK]);
                    if A4 {
                        let xp: Tile<f4e2m1fnx2, { [BM, PK] }> =
                            pq.load_pipelined::<LATENCY>([m, kg]);
                        let xq: Tile<f4e2m1fn, { [BM, BK] }> = xp.unpack(const_shape![BM, BK]);
                        let sx: Tile<f8e4m3fn, { [BM, SK] }> =
                            pqs.load_pipelined::<LATENCY>([m, kg]);
                        let wt: Tile<f4e2m1fn, { [BK, BN] }> = permute(wt, transpose);
                        let sw: Tile<f8e4m3fn, { [SK, BN] }> = permute(sw, transpose);
                        acc = mmaf_scaled(xq, wt, acc, sx, sw);
                    } else {
                        let xt: Tile<T, { [BM, BK] }> = px.load_pipelined::<LATENCY>([m, kg]);
                        let wf: Tile<f32, { [BN, BK] }> = convert_tile(wt);
                        let sw: Tile<f32, { [BN, SK] }> = convert_tile(sw);
                        let sw = sw
                            .reshape(const_shape![BN, SK, 1])
                            .broadcast(const_shape![BN, SK, BLOCK])
                            .reshape(const_shape![BN, BK]);
                        let wd: Tile<T, { [BN, BK] }> = convert_tile(wf * sw);
                        let wd: Tile<T, { [BK, BN] }> = permute(wd, transpose);
                        acc = mmaf(xt, wd, acc);
                    }
                }
                let weight_global: Tile<f32, { [BN] }> =
                    pwg.load([expert, n]).reshape(const_shape![BN]);
                let weight_global = weight_global
                    .reshape(const_shape![1, BN])
                    .broadcast(const_shape![BM, BN]);
                let result = acc * weight_global;
                let result = if A4 {
                    result
                        * global
                            .reshape(const_shape![1, 1])
                            .broadcast(const_shape![BM, BN])
                } else {
                    result
                };
                let out: Tile<T, { [BM, BN] }> = convert_tile(result);
                let ids: Tile<i32, { [BM] }> = load_tile(sids, const_shape![BM], [m]);
                let cols: Tile<i32, { [BN] }> =
                    iota(const_shape![BN]) + broadcast_scalar(n * BN, const_shape![BN]);
                let row_offset: Tile<i64, { [BM] }> = exti(ids);
                let stride: Tile<i64, { [BM] }> = exti(broadcast_scalar(n_size, const_shape![BM]));
                let col_offset: Tile<i64, { [BN] }> = exti(cols);
                let offsets = (row_offset * stride)
                    .reshape(const_shape![BM, 1])
                    .broadcast(const_shape![BM, BN])
                    + col_offset
                        .reshape(const_shape![1, BN])
                        .broadcast(const_shape![BM, BN]);
                let mask = lt_tile(ids, broadcast_scalar(routes, const_shape![BM]))
                    .reshape(const_shape![BM, 1])
                    .broadcast(const_shape![BM, BN])
                    & lt_tile(cols, broadcast_scalar(n_size, const_shape![BN]))
                        .reshape(const_shape![1, BN])
                        .broadcast(const_shape![BM, BN]);
                let ptr: PointerTile<*mut T, { [] }> = pointer_to_tile(y);
                let ptr: PointerTile<*mut T, { [1, 1] }> = ptr.reshape(const_shape![1, 1]);
                let ptr: PointerTile<*mut T, { [BM, BN] }> = ptr.broadcast(const_shape![BM, BN]);
                let ptr: PointerTile<*mut T, { [BM, BN] }> = ptr.offset_tile(offsets);
                store_ptr_tko(
                    ptr,
                    out,
                    ordering::Weak,
                    None::<scope::TileBlock>,
                    Some(mask),
                    None,
                    Latency::<0>,
                );
            }
        }
    }
}

use std::sync::Arc;

use candle_core::{CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor};
use cutile::core::{f4e2m1fnx2, f8e4m3fn};
use cutile::cuda_async::device_buffer::DevicePointer;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::TileKernel;
use float8::F8E4M3;
use half::{bf16, f16};

use super::nvfp4::Nvfp4GemmArgs;
use super::{catch_cutile_panic, context, device_multiprocessor_count};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

pub(super) const KERNEL_SOURCE_HASH: &str = kernels::_SOURCE_HASH;

pub(super) fn with_element<T: cutile::DType>(generics: &[String]) -> Vec<String> {
    std::iter::once(T::DTYPE.as_str().to_owned())
        .chain(generics.iter().cloned())
        .collect()
}

const BLOCK_SIZE: usize = 16;
const QUANT_ROWS: usize = 4;
const QUANT_K: usize = 256;
const SMALL_MATMUL_ROWS: usize = 16;
const SMALL_MATMUL_COLUMNS: usize = 64;
const MEDIUM_MATMUL_ROWS: usize = 32;
const MEDIUM_MATMUL_COLUMNS: usize = 64;
const MATMUL_ROWS: usize = 64;
const MATMUL_COLUMNS: usize = 128;
const MATMUL_K: usize = 256;
const WIDE_MATMUL_ROWS: usize = 128;
const WIDE_MATMUL_COLUMNS: usize = 64;
const WIDE_MATMUL_MIN_K: usize = 8192;
const WIDE_MATMUL_MIN_N: usize = 4096;
const GROUPED_MATMUL_MIN_K: usize = 4096;
const GROUPED_MATMUL_ROWS: usize = 8;
const LOAD_LATENCY: usize = 3;
const BLOCKS_PER_SM: usize = 2;
const WORKER_HINT_MIN_N: usize = 1024;
const WORKER_HINT_WARPS: i32 = 16;

#[derive(Clone, Copy)]
pub(super) struct MatmulDevice {
    pub(super) l2_bytes: usize,
}

pub(super) fn matmul_device(dev: &CudaDevice) -> MatmulDevice {
    MatmulDevice {
        l2_bytes: crate::cuda_device::cuda_device_info(dev).l2_bytes,
    }
}

pub(super) fn dense_geometry(
    m: usize,
    n: usize,
    k: usize,
    a4: bool,
    device: MatmulDevice,
) -> (usize, usize, usize) {
    if a4
        && m >= WIDE_MATMUL_ROWS
        && n >= WIDE_MATMUL_MIN_N
        && k >= GROUPED_MATMUL_MIN_K
        && device.l2_bytes > 0
        && n * k / 2 + n * k / BLOCK_SIZE > device.l2_bytes
    {
        // Grouping M tiles reuses weights that would otherwise be evicted between rows.
        (WIDE_MATMUL_ROWS, WIDE_MATMUL_COLUMNS, GROUPED_MATMUL_ROWS)
    } else if m <= SMALL_MATMUL_ROWS {
        (SMALL_MATMUL_ROWS, SMALL_MATMUL_COLUMNS, 1)
    } else if m <= MEDIUM_MATMUL_ROWS && dense_worker_warps(m, n, k, a4).is_some() {
        (MEDIUM_MATMUL_ROWS, MEDIUM_MATMUL_COLUMNS, 1)
    } else if m >= WIDE_MATMUL_ROWS && n >= WIDE_MATMUL_MIN_N && k >= WIDE_MATMUL_MIN_K {
        (WIDE_MATMUL_ROWS, WIDE_MATMUL_COLUMNS, 1)
    } else {
        (MATMUL_ROWS, MATMUL_COLUMNS, 1)
    }
}

pub(super) fn dense_worker_warps(m: usize, n: usize, k: usize, a4: bool) -> Option<i32> {
    (a4 && (m >= WIDE_MATMUL_ROWS || (SMALL_MATMUL_ROWS < m && m <= MEDIUM_MATMUL_ROWS))
        && n >= WORKER_HINT_MIN_N
        && k >= GROUPED_MATMUL_MIN_K)
        .then_some(WORKER_HINT_WARPS)
}

pub(super) fn quantize(
    x: &Tensor,
    activation_global: &Tensor,
    compile_only: bool,
) -> Result<(Tensor, Tensor)> {
    let (m, k) = x.dims2()?;
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let cutile_stream = context::stream(dev);
    let (x_storage, x_layout) = x.storage_and_layout();
    let (ag_storage, ag_layout) = activation_global.storage_and_layout();
    let (Storage::Cuda(x_cuda), Storage::Cuda(ag_cuda)) = (&*x_storage, &*ag_storage) else {
        candle_core::bail!("cuTile NVFP4 quantization requires CUDA tensors");
    };
    let (ag_addr, _ag_guard) = slice_ptr_on_stream(
        ag_cuda.as_cuda_slice::<f32>()?,
        ag_layout.start_offset(),
        &stream,
    );
    let ag = Arc::new(unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            ag_addr as CUdeviceptr,
            ordinal,
            vec![1],
            vec![1],
        )
    });
    let mut packed = unsafe { dev.alloc::<u8>(m * k / 2)? };
    let mut scales = unsafe { dev.alloc::<F8E4M3>(m * k / BLOCK_SIZE)? };
    let (q_addr, q_guard) = slice_ptr_mut_on_stream(&mut packed, 0, &stream);
    let (s_addr, s_guard) = slice_ptr_mut_on_stream(&mut scales, 0, &stream);
    let q = unsafe {
        cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
            q_addr as CUdeviceptr,
            ordinal,
            vec![m as i32, (k / 2) as i32],
            vec![(k / 2) as i32, 1],
        )
    };
    let blocks = (BLOCKS_PER_SM * device_multiprocessor_count(dev)) as u32;
    let generic = vec![
        QUANT_ROWS.to_string(),
        QUANT_K.to_string(),
        (QUANT_K / 2).to_string(),
        (QUANT_K / BLOCK_SIZE).to_string(),
    ];
    macro_rules! run {
        ($dtype:ty, $kernel:path) => {{
            let (x_addr, _x_guard) = slice_ptr_on_stream(
                x_cuda.as_cuda_slice::<$dtype>()?,
                x_layout.start_offset(),
                &stream,
            );
            let input = Arc::new(unsafe {
                cutile::tensor::Tensor::<$dtype>::borrow_raw_parts(
                    x_addr as CUdeviceptr,
                    ordinal,
                    vec![m as i32, k as i32],
                    vec![k as i32, 1],
                )
            });
            let mapped = q.partition([QUANT_ROWS, QUANT_K / 2]).map(
                [1, 1],
                blocks.min((m.div_ceil(QUANT_ROWS) * k.div_ceil(QUANT_K)) as u32),
            );
            let launcher = unsafe {
                $kernel(
                    mapped,
                    DevicePointer::<f8e4m3fn>::from_cu_deviceptr(s_addr as CUdeviceptr),
                    m as i32,
                    (k / BLOCK_SIZE) as i32,
                    input,
                    ag,
                )
            }
            .generics(with_element::<$dtype>(&generic));
            if compile_only {
                catch_cutile_panic("NVFP4 quantization compile", || {
                    launcher.compile_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::msg(format!(
                            "cuTile NVFP4 quantization compile failed: {error:?}"
                        ))
                    })
                })?;
            } else {
                catch_cutile_panic("NVFP4 quantization launch", || unsafe {
                    launcher.async_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::msg(format!(
                            "cuTile NVFP4 quantization launch failed: {error:?}"
                        ))
                    })
                })?;
            }
        }};
    }
    match x.dtype() {
        DType::BF16 => run!(bf16, kernels::quantize),
        DType::F16 => run!(f16, kernels::quantize),
        dtype => candle_core::bail!("cuTile NVFP4 quantization does not support {dtype:?}"),
    }
    drop(q_guard);
    drop(s_guard);
    Ok((
        Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(packed, dev.clone())),
            Shape::from_dims(&[m, k / 2]),
        )),
        Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(scales, dev.clone())),
            Shape::from_dims(&[m, k / BLOCK_SIZE]),
        )),
    ))
}

/// Launch parameters of the dense NVFP4 matmul; the policy comes from [`dense_policy`] and the
/// autotuner may replace it per shape and row bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct DenseConfig {
    pub(super) bm: usize,
    pub(super) bn: usize,
    pub(super) bk: usize,
    pub(super) group_m: usize,
    pub(super) latency: usize,
    /// Worker warps per CTA; 0 leaves it to the compiler.
    pub(super) warps: i32,
    pub(super) blocks_per_sm: usize,
}

pub(super) fn dense_policy(
    m: usize,
    n: usize,
    k: usize,
    a4: bool,
    device: MatmulDevice,
) -> DenseConfig {
    let (bm, bn, group_m) = dense_geometry(m, n, k, a4, device);
    DenseConfig {
        bm,
        bn,
        bk: MATMUL_K,
        group_m,
        latency: LOAD_LATENCY,
        warps: dense_worker_warps(m, n, k, a4).unwrap_or(0),
        blocks_per_sm: BLOCKS_PER_SM,
    }
}

/// One dense NVFP4 matmul: an A16 activation, or an already quantized A4 one.
#[derive(Clone, Copy)]
pub(super) struct DenseOperands<'a> {
    pub(super) x: Option<&'a Tensor>,
    pub(super) prequantized: Option<(&'a Tensor, &'a Tensor)>,
    pub(super) dtype: DType,
    pub(super) args: Nvfp4GemmArgs<'a>,
}

impl DenseOperands<'_> {
    fn rows_and_k(&self) -> Result<(usize, usize)> {
        match self.prequantized {
            Some((packed, _)) => {
                let (m, packed_k) = packed.dims2()?;
                Ok((m, packed_k * 2))
            }
            None => self.x.unwrap().dims2(),
        }
    }
}

pub(super) fn launch(x: &Tensor, args: Nvfp4GemmArgs<'_>, compile_only: bool) -> Result<Tensor> {
    if let Some(global) = args.activation_global_scale {
        let (packed, scales) = quantize(x, global, compile_only)?;
        launch_prequantized(&packed, &scales, x.dtype(), args, compile_only)
    } else {
        let operands = DenseOperands {
            x: Some(x),
            prequantized: None,
            dtype: x.dtype(),
            args,
        };
        launch_dense(operands, None, compile_only)
    }
}

pub(super) fn launch_prequantized(
    packed: &Tensor,
    scales: &Tensor,
    dtype: DType,
    args: Nvfp4GemmArgs<'_>,
    compile_only: bool,
) -> Result<Tensor> {
    let operands = DenseOperands {
        x: None,
        prequantized: Some((packed, scales)),
        dtype,
        args,
    };
    launch_dense(operands, None, compile_only)
}

/// `config` overrides the tuned-or-policy choice, for the autotuner's candidates.
pub(super) fn launch_dense(
    operands: DenseOperands<'_>,
    config: Option<DenseConfig>,
    compile_only: bool,
) -> Result<Tensor> {
    let DenseOperands {
        x,
        prequantized,
        dtype,
        args,
    } = operands;
    let (m, k) = operands.rows_and_k()?;
    let (n, packed_k) = args.weights.dims2()?;
    let dev = args.weights.device().as_cuda_device()?;
    let a4 = prequantized.is_some();
    let cfg = config.unwrap_or_else(|| {
        super::nvfp4_tune::dense_config(
            super::nvfp4_tune::DenseKey::new(dev, n, k, a4, dtype),
            m,
            matmul_device(dev),
        )
    });
    let DenseConfig {
        bm,
        bn,
        bk,
        group_m,
        ..
    } = cfg;
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let cutile_stream = context::stream(dev);
    let (w_storage, w_layout) = args.weights.storage_and_layout();
    let (s_storage, s_layout) = args.weight_scales.storage_and_layout();
    let (wg_storage, wg_layout) = args.weight_global_scale.storage_and_layout();
    let activation_global = args
        .activation_global_scale
        .unwrap_or(args.weight_global_scale);
    let (ag_storage, ag_layout) = activation_global.storage_and_layout();
    let (
        Storage::Cuda(w_cuda),
        Storage::Cuda(s_cuda),
        Storage::Cuda(wg_cuda),
        Storage::Cuda(ag_cuda),
    ) = (&*w_storage, &*s_storage, &*wg_storage, &*ag_storage)
    else {
        candle_core::bail!("cuTile NVFP4 matmul operands must be CUDA tensors");
    };
    let (w_addr, _w_guard) = slice_ptr_on_stream(
        w_cuda.as_cuda_slice::<u8>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (s_addr, _s_guard) = slice_ptr_on_stream(
        s_cuda.as_cuda_slice::<F8E4M3>()?,
        s_layout.start_offset(),
        &stream,
    );
    let (wg_addr, _wg_guard) = slice_ptr_on_stream(
        wg_cuda.as_cuda_slice::<f32>()?,
        wg_layout.start_offset(),
        &stream,
    );
    let (ag_addr, _ag_guard) = slice_ptr_on_stream(
        ag_cuda.as_cuda_slice::<f32>()?,
        ag_layout.start_offset(),
        &stream,
    );
    let (packed, scales) = match prequantized {
        Some((packed, scales)) => (packed.clone(), scales.clone()),
        None => (
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(
                    unsafe { dev.alloc::<u8>(1)? },
                    dev.clone(),
                )),
                Shape::from_dims(&[1, 1]),
            )),
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(
                    unsafe { dev.alloc::<F8E4M3>(1)? },
                    dev.clone(),
                )),
                Shape::from_dims(&[1, 1]),
            )),
        ),
    };
    let (q_storage, q_layout) = packed.storage_and_layout();
    let (qs_storage, qs_layout) = scales.storage_and_layout();
    let (Storage::Cuda(q_cuda), Storage::Cuda(qs_cuda)) = (&*q_storage, &*qs_storage) else {
        candle_core::bail!("cuTile NVFP4 activations must be CUDA tensors");
    };
    let (q_addr, _q_guard) = slice_ptr_on_stream(
        q_cuda.as_cuda_slice::<u8>()?,
        q_layout.start_offset(),
        &stream,
    );
    let (qs_addr, _qs_guard) = slice_ptr_on_stream(
        qs_cuda.as_cuda_slice::<F8E4M3>()?,
        qs_layout.start_offset(),
        &stream,
    );
    let w = Arc::new(unsafe {
        cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
            w_addr as CUdeviceptr,
            ordinal,
            vec![n as i32, packed_k as i32],
            vec![packed_k as i32, 1],
        )
    });
    let ws = Arc::new(unsafe {
        cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            s_addr as CUdeviceptr,
            ordinal,
            vec![n as i32, (k / BLOCK_SIZE) as i32],
            vec![(k / BLOCK_SIZE) as i32, 1],
        )
    });
    let wg = Arc::new(unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            wg_addr as CUdeviceptr,
            ordinal,
            vec![n as i32],
            vec![1],
        )
    });
    let ag = Arc::new(unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            ag_addr as CUdeviceptr,
            ordinal,
            vec![1],
            vec![1],
        )
    });
    let q = Arc::new(unsafe {
        cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
            q_addr as CUdeviceptr,
            ordinal,
            if a4 {
                vec![m as i32, packed_k as i32]
            } else {
                vec![1, 1]
            },
            if a4 {
                vec![packed_k as i32, 1]
            } else {
                vec![1, 1]
            },
        )
    });
    let qs = Arc::new(unsafe {
        cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            qs_addr as CUdeviceptr,
            ordinal,
            if a4 {
                vec![m as i32, (k / BLOCK_SIZE) as i32]
            } else {
                vec![1, 1]
            },
            if a4 {
                vec![(k / BLOCK_SIZE) as i32, 1]
            } else {
                vec![1, 1]
            },
        )
    });
    let blocks = (cfg.blocks_per_sm.max(1) * device_multiprocessor_count(dev)) as u32;
    let generic = vec![
        bm.to_string(),
        bn.to_string(),
        bk.to_string(),
        (bk / 2).to_string(),
        (bk / BLOCK_SIZE).to_string(),
        a4.to_string(),
        cfg.latency.to_string(),
        group_m.to_string(),
    ];
    let options = if cfg.warps > 0 {
        cutile::cutile_compiler::compiler::utils::CompileOptions::default()
            .num_worker_warps_per_cta(cfg.warps)
    } else {
        cutile::cutile_compiler::compiler::utils::CompileOptions::default()
    };
    macro_rules! run {
        ($dtype:ty, $kernel:path) => {{
            let input_storage = x.map(Tensor::storage_and_layout);
            let (x_addr, _input_guard) = if a4 {
                // A4 never reads its A16 operand; the F32 calibration scalar owns this two-byte view.
                (ag_addr, None)
            } else {
                let (storage, layout) = input_storage.as_ref().unwrap();
                let Storage::Cuda(storage) = &**storage else {
                    candle_core::bail!("cuTile NVFP4 matmul input must be CUDA");
                };
                let (address, guard) = slice_ptr_on_stream(storage.as_cuda_slice::<$dtype>()?, layout.start_offset(), &stream);
                (address, Some(guard))
            };
            let input = Arc::new(unsafe {
                cutile::tensor::Tensor::<$dtype>::borrow_raw_parts(
                    x_addr as CUdeviceptr, ordinal,
                    if a4 { vec![1, 1] } else { vec![m as i32, k as i32] },
                    if a4 { vec![1, 1] } else { vec![k as i32, 1] },
                )
            });
            let mut output = unsafe { dev.alloc::<$dtype>(m * n)? };
            let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let y = unsafe {
                cutile::tensor::Tensor::<$dtype>::borrow_raw_parts(out_addr as CUdeviceptr, ordinal, vec![m as i32, n as i32], vec![n as i32, 1])
            };
            let mapped = y.partition([bm, bn]).map([group_m, 1], blocks.min((m.div_ceil(bm) * n.div_ceil(bn)) as u32));
            let launcher = $kernel(mapped, input, q, qs, w, ws, wg, ag).generics(with_element::<$dtype>(&generic)).compile_options(options);
            if compile_only {
                catch_cutile_panic("NVFP4 matmul compile", || {
                    launcher.compile_on(&cutile_stream).map_err(|error| candle_core::Error::msg(format!("cuTile NVFP4 matmul compile failed: {error:?}")))
                })?;
            } else {
                catch_cutile_panic("NVFP4 matmul launch", || unsafe {
                    launcher.async_on(&cutile_stream).map_err(|error| candle_core::Error::msg(format!("cuTile NVFP4 matmul launch failed: {error:?}")))
                })?;
            }
            drop(out_guard);
            Ok(Tensor::from((Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())), Shape::from_dims(&[m, n]))))
        }};
    }
    match dtype {
        DType::BF16 => run!(bf16, kernels::matmul),
        DType::F16 => run!(f16, kernels::matmul),
        dtype => candle_core::bail!("cuTile NVFP4 matmul does not support {dtype:?}"),
    }
}

const ROUTED_ROWS: usize = 16;
const ROUTED_COLUMNS: usize = 64;
pub(super) const ROUTED_PREFILL_MIN_TOKENS: usize = 512;
pub(super) const ROUTED_LARGE_PREFILL_MIN_TOKENS: usize = 4096;
pub(super) const MAX_ROUTED_ROWS: usize = 64;
const ROUTED_LARGE_PREFILL_ROWS: usize = 32;
const ROUTED_PREFILL_COLUMNS: usize = 128;
const ROUTED_PREFILL_MIN_WIDTH: usize = 512;
const ROUTED_PREFILL_SHORT_K: usize = 128;
const ROUTED_WORKERS_PER_SM: usize = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct RoutedGeometry {
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) k: usize,
    pub(super) persistent: bool,
    pub(super) latency: usize,
    /// Persistent tile blocks per SM; unused by the one-block-per-tile grid.
    pub(super) workers_per_sm: usize,
}

pub(super) fn routed_geometry(tokens: usize, n: usize, k: usize, a4: bool) -> RoutedGeometry {
    if a4
        && tokens >= ROUTED_PREFILL_MIN_TOKENS
        && n >= ROUTED_PREFILL_MIN_WIDTH
        && k >= ROUTED_PREFILL_MIN_WIDTH
    {
        RoutedGeometry {
            rows: if n >= k {
                MAX_ROUTED_ROWS
            } else if tokens >= ROUTED_LARGE_PREFILL_MIN_TOKENS {
                ROUTED_LARGE_PREFILL_ROWS
            } else {
                ROUTED_ROWS
            },
            columns: ROUTED_PREFILL_COLUMNS,
            k: if n >= k {
                ROUTED_PREFILL_SHORT_K
            } else {
                MATMUL_K
            },
            persistent: true,
            latency: LOAD_LATENCY,
            workers_per_sm: ROUTED_WORKERS_PER_SM,
        }
    } else {
        RoutedGeometry {
            rows: ROUTED_ROWS,
            columns: ROUTED_COLUMNS,
            k: MATMUL_K,
            persistent: false,
            latency: LOAD_LATENCY,
            workers_per_sm: ROUTED_WORKERS_PER_SM,
        }
    }
}

pub(super) struct PreparedRoutedInput {
    alignment: super::fused_moe::MoeAlign,
    packed: Tensor,
    scales: Tensor,
    aligned: Tensor,
    tokens: usize,
    topk: usize,
    geometry: RoutedGeometry,
}

/// How a routed activation is prepared; `geometry` overrides the tuned-or-policy choice.
#[derive(Clone, Copy)]
pub(super) struct GatherPlan {
    pub(super) geometry: Option<RoutedGeometry>,
    pub(super) uniform_input_scale: bool,
    pub(super) compile_only: bool,
}

pub(super) fn prepare_gather(
    x: &Tensor,
    indices: &Tensor,
    args: Nvfp4GemmArgs<'_>,
    uniform_input_scale: bool,
    compile_only: bool,
) -> Result<PreparedRoutedInput> {
    let plan = GatherPlan {
        geometry: None,
        uniform_input_scale,
        compile_only,
    };
    prepare_gather_with(x, indices, args, plan)
}

pub(super) fn prepare_gather_with(
    x: &Tensor,
    indices: &Tensor,
    args: Nvfp4GemmArgs<'_>,
    plan: GatherPlan,
) -> Result<PreparedRoutedInput> {
    let GatherPlan {
        uniform_input_scale,
        compile_only,
        ..
    } = plan;
    let (tokens, topk) = indices.dims2()?;
    let (experts, n, packed_k) = args.weights.dims3()?;
    let routes = tokens * topk;
    let k = packed_k * 2;
    let input_rows = x.elem_count() / k;
    let x_stride = routes / input_rows;
    let x = x.contiguous()?.reshape((input_rows, k))?;
    let Device::Cuda(dev) = x.device() else {
        candle_core::bail!("cuTile NVFP4 grouped matmul requires CUDA tensors")
    };
    let a4 = args.activation_global_scale.is_some();
    let geometry = plan.geometry.unwrap_or_else(|| {
        let key = super::nvfp4_tune::RoutedKey::new(dev, experts, n, k, topk, a4, x.dtype());
        super::nvfp4_tune::routed_config(key, tokens)
    });
    let indices = indices.contiguous()?;
    let (id_storage, id_layout) = indices.storage_and_layout();
    let Storage::Cuda(id_cuda) = &*id_storage else {
        candle_core::bail!("cuTile NVFP4 grouped indices must be CUDA tensors")
    };
    let alignment = if compile_only {
        let em = crate::moe::cuda::moe_align_em(tokens, topk, experts, geometry.rows);
        super::fused_moe::MoeAlign {
            sids: unsafe { dev.alloc::<i32>(em)? },
            eids: unsafe { dev.alloc::<i32>(em.div_ceil(geometry.rows))? },
            ntpp: unsafe { dev.alloc::<i32>(1)? },
            em,
        }
    } else {
        super::fused_moe::MoeAlign::build(
            dev,
            &id_cuda
                .as_cuda_slice::<u32>()?
                .slice(id_layout.start_offset()..id_layout.start_offset() + routes),
            tokens,
            experts,
            topk,
            geometry.rows as i32,
        )?
    };
    let em = alignment.em;
    let stream = dev.cuda_stream();
    let (x_storage, x_layout) = x.storage_and_layout();
    let activation_global = args
        .activation_global_scale
        .unwrap_or(args.weight_global_scale);
    let (ag_storage, ag_layout) = activation_global.storage_and_layout();
    let (Storage::Cuda(x_cuda), Storage::Cuda(ag_cuda)) = (&*x_storage, &*ag_storage) else {
        candle_core::bail!("cuTile NVFP4 grouped preparation requires CUDA tensors")
    };
    let (ag_addr, _ag_guard) = slice_ptr_on_stream(
        ag_cuda.as_cuda_slice::<f32>()?,
        ag_layout.start_offset(),
        &stream,
    );
    let (sid_addr, _sid_guard) = slice_ptr_on_stream(&alignment.sids, 0, &stream);
    let (eid_addr, _eid_guard) = slice_ptr_on_stream(&alignment.eids, 0, &stream);
    let (ntpp_addr, _ntpp_guard) = slice_ptr_on_stream(&alignment.ntpp, 0, &stream);
    let quantize_source = a4
        && uniform_input_scale
        && crate::nvfp4::prepare::supports_source_quantization(input_rows, k, x_stride);
    macro_rules! run {
        ($dtype:ty) => {{
            let (x_addr, _x_guard) = slice_ptr_on_stream(
                x_cuda.as_cuda_slice::<$dtype>()?,
                x_layout.start_offset(),
                &stream,
            );
            let mut packed = unsafe { dev.alloc::<u8>(if a4 { em * packed_k } else { 1 })? };
            let mut scales =
                unsafe { dev.alloc::<F8E4M3>(if a4 { em * k / BLOCK_SIZE } else { 1 })? };
            let mut aligned = unsafe { dev.alloc::<$dtype>(if a4 { 1 } else { em * k })? };
            let (q_addr, q_guard) = slice_ptr_mut_on_stream(&mut packed, 0, &stream);
            let (qs_addr, qs_guard) = slice_ptr_mut_on_stream(&mut scales, 0, &stream);
            let (xa_addr, xa_guard) = slice_ptr_mut_on_stream(&mut aligned, 0, &stream);
            let mut source_packed = if quantize_source {
                Some(unsafe { dev.alloc::<u8>(input_rows * packed_k)? })
            } else {
                None
            };
            let mut source_scales = if quantize_source {
                Some(unsafe { dev.alloc::<u8>(input_rows * k / BLOCK_SIZE)? })
            } else {
                None
            };
            if !compile_only {
                unsafe {
                    let preparation = crate::nvfp4::prepare::RoutedActivation {
                        input: x_addr as *const _,
                        global: ag_addr as *const _,
                        sorted_ids: sid_addr as *const _,
                        expert_ids: eid_addr as *const _,
                        num_padded: ntpp_addr as *const _,
                        packed: q_addr as *mut _,
                        scales: qs_addr as *mut _,
                        aligned: xa_addr as *mut _,
                        rows: em as i32,
                        routes: routes as i32,
                        k: k as i32,
                        input_stride: x_stride as i32,
                        alignment: geometry.rows as i32,
                        a4: i32::from(a4),
                    };
                    if let (Some(packed), Some(scales)) = (&mut source_packed, &mut source_scales) {
                        let (source_q, _source_q_guard) =
                            slice_ptr_mut_on_stream(packed, 0, &stream);
                        let (source_s, _source_s_guard) =
                            slice_ptr_mut_on_stream(scales, 0, &stream);
                        crate::nvfp4::prepare::launch_source(
                            &preparation,
                            &crate::nvfp4::prepare::SourceActivation {
                                packed: source_q as *mut _,
                                scales: source_s as *mut _,
                            },
                            x.dtype(),
                            stream.cu_stream(),
                        )?;
                    } else {
                        crate::nvfp4::prepare::launch(&preparation, x.dtype(), stream.cu_stream())?;
                    }
                }
            }
            drop(q_guard);
            drop(qs_guard);
            drop(xa_guard);
            let tensor = |storage, shape: [usize; 2]| {
                Tensor::from((Storage::Cuda(storage), Shape::from_dims(&shape)))
            };
            let packed = tensor(
                CudaStorage::wrap_cuda_slice(packed, dev.clone()),
                if a4 { [em, packed_k] } else { [1, 1] },
            );
            let scales = tensor(
                CudaStorage::wrap_cuda_slice(scales, dev.clone()),
                if a4 { [em, k / BLOCK_SIZE] } else { [1, 1] },
            );
            let aligned = tensor(
                CudaStorage::wrap_cuda_slice(aligned, dev.clone()),
                if a4 { [1, 1] } else { [em, k] },
            );
            (packed, scales, aligned)
        }};
    }
    let (packed, scales, aligned) = match x.dtype() {
        DType::BF16 => run!(bf16),
        DType::F16 => run!(f16),
        dtype => {
            candle_core::bail!("cuTile NVFP4 grouped matmul does not support {dtype:?} activations")
        }
    };
    drop(_sid_guard);
    drop(_eid_guard);
    drop(_ntpp_guard);
    Ok(PreparedRoutedInput {
        alignment,
        packed,
        scales,
        aligned,
        tokens,
        topk,
        geometry,
    })
}

pub(super) fn launch_gather(
    x: &Tensor,
    indices: &Tensor,
    args: Nvfp4GemmArgs<'_>,
    uniform_input_scale: bool,
    compile_only: bool,
) -> Result<Tensor> {
    let prepared = prepare_gather(x, indices, args, uniform_input_scale, compile_only)?;
    launch_gather_prepared(&prepared, args, compile_only)
}

pub(super) fn launch_gather_prepared(
    prepared: &PreparedRoutedInput,
    args: Nvfp4GemmArgs<'_>,
    compile_only: bool,
) -> Result<Tensor> {
    let (experts, n, packed_k) = args.weights.dims3()?;
    let k = packed_k * 2;
    let tokens = prepared.tokens;
    let topk = prepared.topk;
    let routes = tokens * topk;
    let geometry = prepared.geometry;
    let alignment = &prepared.alignment;
    let em = alignment.em;
    let x = &prepared.aligned;
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let cutile_stream = context::stream(dev);
    let (x_storage, x_layout) = x.storage_and_layout();
    let (w_storage, w_layout) = args.weights.storage_and_layout();
    let (s_storage, s_layout) = args.weight_scales.storage_and_layout();
    let (wg_storage, wg_layout) = args.weight_global_scale.storage_and_layout();
    let activation_global = args
        .activation_global_scale
        .unwrap_or(args.weight_global_scale);
    let (ag_storage, ag_layout) = activation_global.storage_and_layout();
    let (
        Storage::Cuda(x_cuda),
        Storage::Cuda(w_cuda),
        Storage::Cuda(s_cuda),
        Storage::Cuda(wg_cuda),
        Storage::Cuda(ag_cuda),
    ) = (
        &*x_storage,
        &*w_storage,
        &*s_storage,
        &*wg_storage,
        &*ag_storage,
    )
    else {
        candle_core::bail!("cuTile NVFP4 grouped matmul operands must be CUDA tensors")
    };
    let (w_addr, _w_guard) = slice_ptr_on_stream(
        w_cuda.as_cuda_slice::<u8>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (s_addr, _s_guard) = slice_ptr_on_stream(
        s_cuda.as_cuda_slice::<F8E4M3>()?,
        s_layout.start_offset(),
        &stream,
    );
    let (wg_addr, _wg_guard) = slice_ptr_on_stream(
        wg_cuda.as_cuda_slice::<f32>()?,
        wg_layout.start_offset(),
        &stream,
    );
    let (ag_addr, _ag_guard) = slice_ptr_on_stream(
        ag_cuda.as_cuda_slice::<f32>()?,
        ag_layout.start_offset(),
        &stream,
    );
    let (sid_addr, _sid_guard) = slice_ptr_on_stream(&alignment.sids, 0, &stream);
    let (eid_addr, _eid_guard) = slice_ptr_on_stream(&alignment.eids, 0, &stream);
    let (ntpp_addr, _ntpp_guard) = slice_ptr_on_stream(&alignment.ntpp, 0, &stream);
    let sids = Arc::new(unsafe {
        cutile::tensor::Tensor::<i32>::borrow_raw_parts(
            sid_addr as CUdeviceptr,
            ordinal,
            vec![em as i32],
            vec![1],
        )
    });
    let eids = Arc::new(unsafe {
        cutile::tensor::Tensor::<i32>::borrow_raw_parts(
            eid_addr as CUdeviceptr,
            ordinal,
            vec![em.div_ceil(geometry.rows) as i32],
            vec![1],
        )
    });
    let ntpp = Arc::new(unsafe {
        cutile::tensor::Tensor::<i32>::borrow_raw_parts(
            ntpp_addr as CUdeviceptr,
            ordinal,
            vec![1],
            vec![1],
        )
    });
    let w = unsafe {
        cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
            w_addr as CUdeviceptr,
            ordinal,
            vec![experts as i32, n as i32, packed_k as i32],
            vec![(n * packed_k) as i32, packed_k as i32, 1],
        )
    };
    let ws = unsafe {
        cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            s_addr as CUdeviceptr,
            ordinal,
            vec![experts as i32, n as i32, (k / BLOCK_SIZE) as i32],
            vec![(n * k / BLOCK_SIZE) as i32, (k / BLOCK_SIZE) as i32, 1],
        )
    };
    let wg = unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            wg_addr as CUdeviceptr,
            ordinal,
            vec![experts as i32, n as i32],
            vec![n as i32, 1],
        )
    };
    let ag = Arc::new(unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            ag_addr as CUdeviceptr,
            ordinal,
            vec![experts as i32],
            vec![1],
        )
    });
    let blocks = if geometry.persistent {
        (device_multiprocessor_count(dev) * geometry.workers_per_sm.max(1)) as u32
    } else {
        (em.div_ceil(geometry.rows) * n.div_ceil(geometry.columns)) as u32
    };
    let a4 = args.activation_global_scale.is_some();
    let generic = vec![
        geometry.rows.to_string(),
        geometry.columns.to_string(),
        geometry.k.to_string(),
        (geometry.k / 2).to_string(),
        (geometry.k / BLOCK_SIZE).to_string(),
        a4.to_string(),
        geometry.latency.to_string(),
        geometry.persistent.to_string(),
    ];
    macro_rules! dispatch {
        ($launcher:expr) => {{
            let launcher = $launcher;
            if compile_only {
                catch_cutile_panic("NVFP4 grouped compile", || {
                    launcher.compile_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "cuTile NVFP4 grouped compile failed: {error:?}"
                        ))
                    })
                })?;
            } else {
                catch_cutile_panic("NVFP4 grouped launch", || unsafe {
                    launcher.async_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "cuTile NVFP4 grouped launch failed: {error:?}"
                        ))
                    })
                })?;
            }
        }};
    }
    macro_rules! run {
        ($dtype:ty, $matmul:path) => {{
            let (xa_addr, _xa_guard) = slice_ptr_on_stream(
                x_cuda.as_cuda_slice::<$dtype>()?,
                x_layout.start_offset(),
                &stream,
            );
            let mut output = unsafe { dev.alloc::<$dtype>(routes * n)? };
            let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let (q_storage, q_layout) = prepared.packed.storage_and_layout();
            let (qs_storage, qs_layout) = prepared.scales.storage_and_layout();
            let (Storage::Cuda(q_cuda), Storage::Cuda(qs_cuda)) = (&*q_storage, &*qs_storage)
            else {
                unreachable!()
            };
            let (q_addr, _q_guard) = slice_ptr_on_stream(
                q_cuda.as_cuda_slice::<u8>()?,
                q_layout.start_offset(),
                &stream,
            );
            let (qs_addr, _qs_guard) = slice_ptr_on_stream(
                qs_cuda.as_cuda_slice::<F8E4M3>()?,
                qs_layout.start_offset(),
                &stream,
            );
            let qs = unsafe {
                cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
                    qs_addr as CUdeviceptr,
                    ordinal,
                    if a4 {
                        vec![em as i32, (k / BLOCK_SIZE) as i32]
                    } else {
                        vec![1, 1]
                    },
                    if a4 {
                        vec![(k / BLOCK_SIZE) as i32, 1]
                    } else {
                        vec![1, 1]
                    },
                )
            };
            let xa = unsafe {
                cutile::tensor::Tensor::<$dtype>::borrow_raw_parts(
                    xa_addr as CUdeviceptr,
                    ordinal,
                    if a4 {
                        vec![1, 1]
                    } else {
                        vec![em as i32, k as i32]
                    },
                    if a4 { vec![1, 1] } else { vec![k as i32, 1] },
                )
            };
            let q_read = Arc::new(unsafe {
                cutile::tensor::Tensor::<f4e2m1fnx2>::borrow_raw_parts(
                    q_addr as CUdeviceptr,
                    ordinal,
                    if a4 {
                        vec![em as i32, packed_k as i32]
                    } else {
                        vec![1, 1]
                    },
                    if a4 {
                        vec![packed_k as i32, 1]
                    } else {
                        vec![1, 1]
                    },
                )
            });
            let matmul = unsafe {
                $matmul(
                    DevicePointer::<$dtype>::from_cu_deviceptr(out_addr as CUdeviceptr),
                    sids,
                    eids,
                    ntpp,
                    routes as i32,
                    n as i32,
                    Arc::new(xa),
                    q_read,
                    Arc::new(qs),
                    Arc::new(w),
                    Arc::new(ws),
                    Arc::new(wg),
                    ag,
                )
            }
            .generics(with_element::<$dtype>(&generic))
            .grid((blocks, 1, 1));
            dispatch!(matmul);
            drop(out_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from_dims(&[tokens, topk, n]),
            ))
        }};
    }
    match x.dtype() {
        DType::BF16 => Ok(run!(bf16, kernels::routed_matmul)),
        DType::F16 => Ok(run!(f16, kernels::routed_matmul)),
        dtype => {
            candle_core::bail!("cuTile NVFP4 grouped matmul does not support {dtype:?} activations")
        }
    }
}
