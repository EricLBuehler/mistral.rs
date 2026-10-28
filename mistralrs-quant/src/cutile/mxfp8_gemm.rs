use std::collections::HashSet;
use std::sync::{Arc, LazyLock, Mutex};

use candle_core::{
    backend::BackendStorage, CudaDevice, CudaStorage, DType, Result, Shape, Storage, Tensor,
};
use cutile::core::{bf16, f16, f8e4m3fn, f8e8m0fnu};
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::TileKernel;
use float8::F8E4M3;

use super::warmup::{CutileKernel, WarmupProgress};
use super::{catch_cutile_panic, context, device_multiprocessor_count};
use crate::mxfp8::{MXFP8_ACTIVATION_SCHEME, MXFP8_BLOCK_SIZE};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};
use crate::{ActivationScaleLayout, QuantizedActivation};

const TILE_N: usize = 128;
const TILE_K: usize = 128;
const LOAD_LATENCY: i32 = 3;
const LARGE_ROW_TILES: usize = 32;
const WIDE_COLUMN_TILES: usize = 32;
const LARGE_COLUMN_TILES: usize = 64;
const NARROW_PREFILL_MAX_ROWS: usize = 1024;
const MIN_TILE_M: usize = 16;
const MAX_TILE_M: usize = 64;
const STATIC_K_MIN_ROWS: usize = 2048;
const STATIC_K_MIN_COLUMNS: usize = 8192;
const STATIC_K_MIN_INNER: usize = 2048;
const STATIC_K_TILE_M: usize = 128;
const STATIC_K_TILE_K: usize = 64;
const MAX_POINTER_ALIGNMENT: usize = 16;
const WARM_ROWS: [usize; 28] = [
    2, 3, 4, 5, 8, 9, 16, 17, 18, 20, 24, 32, 33, 34, 36, 40, 48, 64, 1985, 1986, 1988, 1992, 2000,
    2049, 2050, 2052, 2056, 2064,
];

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry(unchecked_accesses = false)]
    fn mxfp8_gemm<
        O: ElementType,
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const GS: i32,
        const GROUP_M: i32,
        const LATENCY: i32,
        const K_TILES: i32,
    >(
        mut y: MappedPartitionMut<O, { [BM, BN] }, { [GROUP_M, 1] }>,
        x: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        xs: &Tensor<f8e8m0fnu, { [-1, -1] }>,
        ws: &Tensor<f8e8m0fnu, { [-1, -1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pw = w.partition(const_shape![BN, BK]);
        let psx = xs.partition(const_shape![BM, GS]);
        let psw = ws.partition(const_shape![BN, GS]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let groups = if K_TILES == 0 {
            num_tiles(&px, 1)
        } else {
            K_TILES
        };
        for index in y.iter_indices() {
            let (m, n) = index.components();
            let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for group in 0..groups {
                let a: Tile<f8e4m3fn, { [BM, BK] }> = px.load_pipelined::<LATENCY>([m, group]);
                let b: Tile<f8e4m3fn, { [BN, BK] }> = pw.load_pipelined::<LATENCY>([n, group]);
                let bt: Tile<f8e4m3fn, { [BK, BN] }> = permute(b, transpose);
                let sa: Tile<f8e8m0fnu, { [BM, GS] }> = psx.load_pipelined::<LATENCY>([m, group]);
                let sb: Tile<f8e8m0fnu, { [BN, GS] }> = psw.load_pipelined::<LATENCY>([n, group]);
                let sbt: Tile<f8e8m0fnu, { [GS, BN] }> = permute(sb, transpose);
                accumulator = mmaf_scaled(a, bt, accumulator, sa, sbt);
            }
            let output: Tile<O, { [BM, BN] }> = convert_tile(accumulator);
            y.store(output, index);
        }
    }
}

#[derive(Clone, Copy)]
struct Config {
    bm: usize,
    bk: usize,
    static_k: bool,
    group_m: usize,
    blocks_per_sm: usize,
}

fn config(rows: usize, columns: usize, inner: usize) -> Config {
    let static_k =
        rows >= STATIC_K_MIN_ROWS && columns >= STATIC_K_MIN_COLUMNS && inner >= STATIC_K_MIN_INNER;
    let bm = if static_k {
        STATIC_K_TILE_M
    } else if columns <= TILE_N && rows <= NARROW_PREFILL_MAX_ROWS {
        MIN_TILE_M
    } else {
        rows.next_power_of_two().clamp(MIN_TILE_M, MAX_TILE_M)
    };
    let row_tiles = rows.div_ceil(bm);
    let column_tiles = columns.div_ceil(TILE_N);
    let large_rows = static_k || row_tiles >= LARGE_ROW_TILES;
    Config {
        bm,
        bk: if static_k { STATIC_K_TILE_K } else { TILE_K },
        static_k,
        group_m: if large_rows && column_tiles >= LARGE_COLUMN_TILES {
            16
        } else {
            8
        },
        blocks_per_sm: if large_rows && column_tiles >= WIDE_COLUMN_TILES {
            1
        } else {
            2
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct WarmShape {
    ordinal: usize,
    n: usize,
    k: usize,
    dtype: DType,
    scale_offset: usize,
}

static SHAPES: LazyLock<Mutex<HashSet<WarmShape>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

pub(crate) fn register(weight: &Tensor, scales: &Tensor, dtype: DType) -> bool {
    let candle_core::Device::Cuda(dev) = weight.device() else {
        return false;
    };
    let Ok((n, k)) = weight.dims2() else {
        return false;
    };
    if n == 0
        || !super::block_scaled_mma_supported(dev)
        || !matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
    {
        return false;
    }
    if SHAPES.lock().unwrap().insert(WarmShape {
        ordinal: dev.cuda_stream().context().ordinal(),
        n,
        k,
        dtype,
        scale_offset: scales.storage_and_layout().1.start_offset() % MAX_POINTER_ALIGNMENT,
    }) {
        super::warmup::mark_dirty();
    }
    true
}

struct Tiles<T: cutile::DType> {
    output: cutile::tensor::Tensor<T>,
    activation: cutile::tensor::Tensor<f8e4m3fn>,
    weight: cutile::tensor::Tensor<f8e4m3fn>,
    activation_scales: cutile::tensor::Tensor<f8e8m0fnu>,
    weight_scales: cutile::tensor::Tensor<f8e8m0fnu>,
    scale_offset: usize,
}

fn dispatch<T: cutile::DType>(
    tiles: Tiles<T>,
    rows: usize,
    n: usize,
    dev: &CudaDevice,
    compile_only: bool,
) -> Result<()> {
    let k = tiles.weight.size() / n;
    let cfg = config(rows, n, k);
    let count = rows.div_ceil(cfg.bm) * n.div_ceil(TILE_N);
    let blocks = (device_multiprocessor_count(dev) * cfg.blocks_per_sm).clamp(1, count);
    let mapped = tiles
        .output
        .partition([cfg.bm, TILE_N])
        .map([cfg.group_m, 1], blocks as u32);
    let scale_elements = tiles.weight_scales.size();
    let flat_scales = tiles
        .weight_scales
        .view(&[scale_elements])
        .map_err(candle_core::Error::wrap)?;
    let scale_view = flat_scales
        .slice(std::slice::from_ref(&(tiles.scale_offset..scale_elements)))
        .map_err(candle_core::Error::wrap)?;
    let scale_view = scale_view
        .view(&[n, (scale_elements - tiles.scale_offset) / n])
        .map_err(candle_core::Error::wrap)?;
    // cuTile 0.3 reshape loses view-offset alignment; slicing refreshes it.
    let scale_view = scale_view
        .slice(std::slice::from_ref(&(0..n)))
        .map_err(candle_core::Error::wrap)?;
    let launcher = kernels::mxfp8_gemm(
        mapped,
        Arc::new(tiles.activation),
        Arc::new(tiles.weight),
        Arc::new(tiles.activation_scales),
        &scale_view,
    )
    .generics(vec![
        T::DTYPE.as_str().to_owned(),
        cfg.bm.to_string(),
        TILE_N.to_string(),
        cfg.bk.to_string(),
        (cfg.bk / MXFP8_BLOCK_SIZE).to_string(),
        cfg.group_m.to_string(),
        LOAD_LATENCY.to_string(),
        if cfg.static_k { k.div_ceil(cfg.bk) } else { 0 }.to_string(),
    ]);
    let stream = context::stream(dev);
    catch_cutile_panic("MXFP8 GEMM", || {
        if compile_only {
            launcher.compile_on(&stream).map_err(|error| {
                candle_core::Error::msg(format!("cuTile MXFP8 compile: {error:?}"))
            })?;
        } else {
            unsafe { launcher.async_on(&stream) }.map_err(|error| {
                candle_core::Error::msg(format!("cuTile MXFP8 launch: {error:?}"))
            })?;
        }
        Ok(())
    })
}

unsafe fn borrowed<T: cutile::DType>(
    pointer: u64,
    ordinal: usize,
    rows: usize,
    columns: usize,
) -> cutile::tensor::Tensor<T> {
    cutile::tensor::Tensor::borrow_raw_parts(
        pointer,
        ordinal,
        vec![rows as i32, columns as i32],
        vec![columns as i32, 1],
    )
}

pub(crate) fn forward(
    activation: &QuantizedActivation,
    weight: &Tensor,
    scales: &Tensor,
) -> Result<Tensor> {
    let (rows, k) = activation.quantized().dims2()?;
    let (n, weight_k) = weight.dims2()?;
    if activation.scheme() != MXFP8_ACTIVATION_SCHEME
        || activation.scale_layout() != ActivationScaleLayout::RowMajor
        || k != weight_k
        || rows == 0
        || [rows, n, k].into_iter().any(|dim| dim > i32::MAX as usize)
    {
        candle_core::bail!("cuTile MXFP8 activation format or dimensions do not match the weight");
    }
    if !weight.device().same_device(activation.quantized().device())
        || !weight.device().same_device(activation.scales().device())
        || !weight.device().same_device(scales.device())
        || [activation.quantized(), activation.scales(), weight, scales]
            .into_iter()
            .any(|tensor| !tensor.is_contiguous())
    {
        candle_core::bail!("cuTile MXFP8 requires contiguous operands on the same device");
    }
    let (input, il) = activation.quantized().storage_and_layout();
    let (input_scales, isl) = activation.scales().storage_and_layout();
    let (weight_storage, wl) = weight.storage_and_layout();
    let (weight_scales, wsl) = scales.storage_and_layout();
    let (
        Storage::Cuda(input),
        Storage::Cuda(input_scales),
        Storage::Cuda(weight_storage),
        Storage::Cuda(weight_scales),
    ) = (&*input, &*input_scales, &*weight_storage, &*weight_scales)
    else {
        candle_core::bail!("cuTile MXFP8 requires CUDA operands");
    };
    let dev = input.device();
    let stream = dev.cuda_stream();
    let (xp, xg) =
        slice_ptr_on_stream(input.as_cuda_slice::<F8E4M3>()?, il.start_offset(), &stream);
    let (xsp, xsg) = slice_ptr_on_stream(
        input_scales.as_cuda_slice::<u8>()?,
        isl.start_offset(),
        &stream,
    );
    let (wp, wg) = slice_ptr_on_stream(
        weight_storage.as_cuda_slice::<F8E4M3>()?,
        wl.start_offset(),
        &stream,
    );
    let (wsp, wsg) = slice_ptr_on_stream(
        weight_scales.as_cuda_slice::<u8>()?,
        wsl.start_offset(),
        &stream,
    );
    macro_rules! run {
        ($ty:ty) => {{
            let mut output = unsafe { dev.alloc::<$ty>(rows * n)? };
            let (yp, yg) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let tiles = unsafe {
                Tiles::<$ty> {
                    output: borrowed(yp, dev.cuda_stream().context().ordinal(), rows, n),
                    activation: borrowed(xp, dev.cuda_stream().context().ordinal(), rows, k),
                    weight: borrowed(wp, dev.cuda_stream().context().ordinal(), n, k),
                    activation_scales: borrowed(
                        xsp,
                        dev.cuda_stream().context().ordinal(),
                        rows,
                        k / MXFP8_BLOCK_SIZE,
                    ),
                    scale_offset: 0,
                    weight_scales: borrowed(
                        wsp,
                        dev.cuda_stream().context().ordinal(),
                        n,
                        k / MXFP8_BLOCK_SIZE,
                    ),
                }
            };
            dispatch(tiles, rows, n, dev, false)?;
            drop(yg);
            CudaStorage::wrap_cuda_slice(output, dev.clone())
        }};
    }
    let output = match activation.source_dtype() {
        DType::F32 => run!(f32),
        DType::F16 => run!(f16),
        DType::BF16 => run!(bf16),
        _ => candle_core::bail!("cuTile MXFP8 output must be F32, F16 or BF16"),
    };
    drop((xg, xsg, wg, wsg));
    Ok(Tensor::from((
        Storage::Cuda(output),
        Shape::from((rows, n)),
    )))
}

pub(super) struct Mxfp8Gemm;
pub(super) static MXFP8_GEMM: Mxfp8Gemm = Mxfp8Gemm;

impl CutileKernel for Mxfp8Gemm {
    fn warm(&self, dev: &CudaDevice, silent: bool) -> Result<()> {
        if !super::block_scaled_mma_supported(dev) {
            return Ok(());
        }
        let shapes = SHAPES
            .lock()
            .unwrap()
            .iter()
            .copied()
            .filter(|shape| shape.ordinal == dev.cuda_stream().context().ordinal())
            .collect::<Vec<_>>();
        if shapes.is_empty() {
            return Ok(());
        }
        let progress = WarmupProgress::new("MXFP8 GEMM", shapes.len() * WARM_ROWS.len(), silent);
        for shape in shapes {
            warm_shape(shape, dev, &progress)?;
        }
        Ok(())
    }
}

fn warm_shape(shape: WarmShape, dev: &CudaDevice, progress: &WarmupProgress) -> Result<()> {
    let stream = context::stream(dev);
    for rows in WARM_ROWS {
        progress.run(format!("n={} k={} rows={rows}", shape.n, shape.k), || {
            macro_rules! compile {
                ($ty:ty) => {{
                    let meta = |dimensions: &[usize]| {
                        unsafe { cutile::api::meta::<f8e4m3fn>(dimensions).async_on(&stream) }
                            .map_err(|error| {
                                candle_core::Error::msg(format!("cuTile MXFP8 metadata: {error:?}"))
                            })
                    };
                    let scale = |dimensions: &[usize]| {
                        unsafe { cutile::api::meta::<f8e8m0fnu>(dimensions).async_on(&stream) }
                            .map_err(|error| {
                                candle_core::Error::msg(format!("cuTile MXFP8 metadata: {error:?}"))
                            })
                    };
                    let output =
                        unsafe { cutile::api::meta::<$ty>(&[rows, shape.n]).async_on(&stream) }
                            .map_err(|error| {
                                candle_core::Error::msg(format!("cuTile MXFP8 metadata: {error:?}"))
                            })?;
                    dispatch(
                        Tiles {
                            output,
                            activation: meta(&[rows, shape.k])?,
                            weight: meta(&[shape.n, shape.k])?,
                            activation_scales: scale(&[rows, shape.k / MXFP8_BLOCK_SIZE])?,
                            weight_scales: scale(&[
                                shape.n * (shape.k / MXFP8_BLOCK_SIZE) + shape.scale_offset
                            ])?,
                            scale_offset: shape.scale_offset,
                        },
                        rows,
                        shape.n,
                        dev,
                        true,
                    )
                }};
            }
            match shape.dtype {
                DType::F32 => compile!(f32),
                DType::F16 => compile!(f16),
                DType::BF16 => compile!(bf16),
                _ => unreachable!(),
            }
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
