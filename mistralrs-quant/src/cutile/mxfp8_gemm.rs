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

use super::tune::{
    bucket_warm_rows, buckets_from_breakpoints, policy_breakpoints, probe_input, random_activation,
    tunable, Bucket, Prepared, Space, Tunable, TuneMode, TuneRequest, Tuned, TunedTable,
};
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
const SMALL_BATCH_MAX_ROWS: usize = 8;
const SMALL_BATCH_TILE_N: usize = 64;
const SMALL_BATCH_NARROW_SM_DIVISOR: usize = 8;
const SMALL_BATCH_SPLITS: usize = 4;
const SMALL_BATCH_MIN_K_TILES_PER_SPLIT: usize = 4;
const SMALL_BATCH_SPLIT_SM_DIVISOR: usize = 3;
const TUNE_KERNEL: &str = "mxfp8_gemm";
// Single-token layers use GEMV rather than this GEMM.
const TUNE_MIN_ROWS: usize = 1;
/// Rows scanned for policy changes; past this the policy depends on rows only through the tail.
const TUNE_SCAN_ROWS: usize = 8192;
const TUNE_PREFILL_PROBE: usize = 2048;
const TUNE_TAIL_PROBE: usize = 4096;
const WARM_ROWS: [usize; 30] = [
    2, 3, 4, 5, 8, 9, 10, 12, 16, 17, 18, 20, 24, 32, 33, 34, 36, 40, 48, 64, 1985, 1986, 1988,
    1992, 2000, 2049, 2050, 2052, 2056, 2064,
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
        const SPLITS: i32,
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
            let (output_m, n) = index.components();
            let m = if SPLITS == 1 {
                output_m
            } else {
                output_m / SPLITS
            };
            let split = if SPLITS == 1 { 0 } else { output_m % SPLITS };
            let start = if SPLITS == 1 {
                0
            } else {
                split * groups / SPLITS
            };
            let steps = if SPLITS == 1 {
                groups
            } else {
                (split + 1) * groups / SPLITS - start
            };
            let mut accumulator: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for step in 0..steps {
                let group = start + step;
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

    #[cutile::entry(unchecked_accesses = false)]
    fn reduce_partials<O: ElementType, const BM: i32, const BN: i32, const SPLITS: i32>(
        mut y: MappedPartitionMut<O, { [BM, BN] }, { [1, 1] }>,
        partial: &Tensor<f32, { [-1, -1] }>,
    ) {
        let partial = partial.partition(const_shape![BM, BN]);
        for index in y.iter_indices() {
            let (m, n) = index.components();
            let mut sum: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for split in 0..SPLITS {
                sum = sum + partial.load([m * SPLITS + split, n]);
            }
            let output: Tile<O, { [BM, BN] }> = convert_tile(sum);
            y.store(output, index);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Config {
    bm: usize,
    bn: usize,
    bk: usize,
    splits: usize,
    static_k: bool,
    group_m: usize,
    blocks_per_sm: usize,
    latency: usize,
}

tunable!(Config {
    bm,
    bn,
    bk,
    splits,
    static_k,
    group_m,
    blocks_per_sm,
    latency
});

fn config(rows: usize, columns: usize, inner: usize, multiprocessors: usize) -> Config {
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
    let narrow_batch = (2..=SMALL_BATCH_MAX_ROWS).contains(&rows)
        && column_tiles <= multiprocessors / SMALL_BATCH_NARROW_SM_DIVISOR;
    let splits = if narrow_batch
        && columns.div_ceil(SMALL_BATCH_TILE_N) * SMALL_BATCH_SPLITS
            <= multiprocessors / SMALL_BATCH_SPLIT_SM_DIVISOR
        && inner >= TILE_K * SMALL_BATCH_SPLITS * SMALL_BATCH_MIN_K_TILES_PER_SPLIT
    {
        SMALL_BATCH_SPLITS
    } else {
        1
    };
    Config {
        bm,
        bn: if narrow_batch {
            SMALL_BATCH_TILE_N
        } else {
            TILE_N
        },
        bk: if static_k { STATIC_K_TILE_K } else { TILE_K },
        splits,
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
        latency: LOAD_LATENCY as usize,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TuneKey {
    ordinal: usize,
    n: usize,
    k: usize,
    dtype: DType,
}

static TUNED: TunedTable<TuneKey, Config> = TunedTable::new();

fn launch_config(key: TuneKey, rows: usize, multiprocessors: usize) -> Config {
    (rows > TUNE_MIN_ROWS)
        .then(|| TUNED.get(key, rows))
        .flatten()
        .unwrap_or_else(|| config(rows, key.n, key.k, multiprocessors))
}

fn tune_buckets(key: TuneKey, multiprocessors: usize) -> Vec<Bucket> {
    let breakpoints = policy_breakpoints(TUNE_MIN_ROWS + 1, TUNE_SCAN_ROWS, |rows| {
        config(rows, key.n, key.k, multiprocessors)
    });
    buckets_from_breakpoints(&breakpoints, TUNE_PREFILL_PROBE, TUNE_TAIL_PROBE)
}

/// Pipelined operand stages must fit in opt-in shared memory; 0 means unknown and skips the check.
fn fits_smem(cfg: Config, smem: usize) -> bool {
    let stage = (cfg.bm + cfg.bn) * cfg.bk * (1 + MXFP8_BLOCK_SIZE) / MXFP8_BLOCK_SIZE;
    smem == 0 || stage * cfg.latency.max(1) <= smem
}

fn tune_space(bucket: Bucket, key: TuneKey, multiprocessors: usize, smem: usize) -> Space {
    let policy = config(bucket.probe, key.n, key.k, multiprocessors);
    Space::new()
        .joint(
            ["bm", "bn"],
            [
                [16, 64],
                [16, 128],
                [32, 64],
                [32, 128],
                [64, 64],
                [64, 128],
                [128, 64],
                [128, 128],
            ],
        )
        .axis("bk", [128, 64, 256])
        .axis("static_k", [0, 1])
        .axis("group_m", [8, 16, 4])
        .axis("blocks_per_sm", [2, 1, 3])
        .axis("latency", [3, 2, 4])
        .axis("splits", [1, 2, 4, 8])
        .constrain(move |c| {
            Config::from_config(c).is_some_and(|cfg| {
                fits_smem(cfg, smem)
                    && (cfg.splits == 1
                        || (bucket.probe.div_ceil(cfg.bm) * key.n.div_ceil(cfg.bn)
                            < multiprocessors
                            && key.k.div_ceil(cfg.bk)
                                >= cfg.splits * SMALL_BATCH_MIN_K_TILES_PER_SPLIT))
            })
        })
        .policy(policy.to_config())
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

#[derive(Clone, Copy)]
struct DispatchCtx<'a> {
    rows: usize,
    n: usize,
    dev: &'a CudaDevice,
    compile_only: bool,
    cfg: Config,
}

fn dispatch_config<T: cutile::DType>(tiles: Tiles<T>, ctx: DispatchCtx<'_>) -> Result<()> {
    let DispatchCtx {
        rows,
        n,
        dev,
        compile_only,
        cfg,
    } = ctx;
    if cfg.splits == 1 {
        return dispatch_gemm(tiles, ctx);
    }
    let partial_rows = rows.div_ceil(cfg.bm) * cfg.bm * cfg.splits;
    let stream = dev.cuda_stream();
    let cutile_stream = context::stream(dev);
    let mut partial_storage = if compile_only {
        None
    } else {
        Some(unsafe { dev.alloc::<f32>(partial_rows * n)? })
    };
    let partial_guard;
    let partial_pointer = if let Some(storage) = &mut partial_storage {
        let (pointer, guard) = slice_ptr_mut_on_stream(storage, 0, &stream);
        partial_guard = Some(guard);
        Some(pointer)
    } else {
        partial_guard = None;
        None
    };
    let partial_tensor = || {
        if let Some(pointer) = partial_pointer {
            Ok(unsafe { borrowed(pointer, stream.context().ordinal(), partial_rows, n) })
        } else {
            unsafe { cutile::api::meta::<f32>(&[partial_rows, n]).async_on(&cutile_stream) }
                .map_err(|error| {
                    candle_core::Error::msg(format!("cuTile MXFP8 metadata: {error:?}"))
                })
        }
    };
    let output = tiles.output;
    dispatch_gemm(
        Tiles {
            output: partial_tensor()?,
            activation: tiles.activation,
            weight: tiles.weight,
            activation_scales: tiles.activation_scales,
            weight_scales: tiles.weight_scales,
            scale_offset: tiles.scale_offset,
        },
        DispatchCtx {
            rows: partial_rows,
            ..ctx
        },
    )?;
    let blocks = rows.div_ceil(cfg.bm) * n.div_ceil(cfg.bn);
    let mapped = output
        .partition([cfg.bm, cfg.bn])
        .map([1, 1], blocks as u32);
    let partial = partial_tensor()?;
    let launcher = kernels::reduce_partials(mapped, &partial).generics(vec![
        T::DTYPE.as_str().to_owned(),
        cfg.bm.to_string(),
        cfg.bn.to_string(),
        cfg.splits.to_string(),
    ]);
    catch_cutile_panic("MXFP8 split-K reduction", || {
        if compile_only {
            launcher.compile_on(&cutile_stream).map_err(|error| {
                candle_core::Error::msg(format!("cuTile MXFP8 reduction compile: {error:?}"))
            })?;
        } else {
            unsafe { launcher.async_on(&cutile_stream) }.map_err(|error| {
                candle_core::Error::msg(format!("cuTile MXFP8 reduction launch: {error:?}"))
            })?;
        }
        Ok(())
    })?;
    drop(partial_guard);
    Ok(())
}

fn dispatch_gemm<T: cutile::DType>(tiles: Tiles<T>, ctx: DispatchCtx<'_>) -> Result<()> {
    let DispatchCtx {
        rows,
        n,
        dev,
        compile_only,
        cfg,
    } = ctx;
    let k = tiles.weight.size() / n;
    let count = rows.div_ceil(cfg.bm) * n.div_ceil(cfg.bn);
    let blocks = (device_multiprocessor_count(dev) * cfg.blocks_per_sm).clamp(1, count);
    let mapped = tiles
        .output
        .partition([cfg.bm, cfg.bn])
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
        cfg.bn.to_string(),
        cfg.bk.to_string(),
        (cfg.bk / MXFP8_BLOCK_SIZE).to_string(),
        cfg.group_m.to_string(),
        cfg.latency.to_string(),
        if cfg.static_k { k.div_ceil(cfg.bk) } else { 0 }.to_string(),
        cfg.splits.to_string(),
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
    forward_config(activation, weight, scales, None)
}

fn forward_config(
    activation: &QuantizedActivation,
    weight: &Tensor,
    scales: &Tensor,
    config_override: Option<Config>,
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
            let cfg = config_override.unwrap_or_else(|| {
                let key = TuneKey {
                    ordinal: dev.cuda_stream().context().ordinal(),
                    n,
                    k,
                    dtype: activation.source_dtype(),
                };
                launch_config(key, rows, device_multiprocessor_count(dev))
            });
            dispatch_config(
                tiles,
                DispatchCtx {
                    rows,
                    n,
                    dev,
                    compile_only: false,
                    cfg,
                },
            )?;
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
        for shape in &shapes {
            let key = shape.key();
            TUNED.tune_once(key, |mode| tune_shape(key, dev, mode))?;
        }
        let multiprocessors = device_multiprocessor_count(dev);
        let plans: Vec<(WarmShape, Vec<usize>)> = shapes
            .into_iter()
            .map(|shape| {
                let rows = warm_rows(shape.key(), multiprocessors);
                (shape, rows)
            })
            .collect();
        let total = plans.iter().map(|(_, rows)| rows.len()).sum();
        let progress = WarmupProgress::new("MXFP8 GEMM", total, silent);
        for (shape, rows) in plans {
            warm_shape(shape, &rows, dev, &progress)?;
        }
        Ok(())
    }
}

impl WarmShape {
    fn key(&self) -> TuneKey {
        TuneKey {
            ordinal: self.ordinal,
            n: self.n,
            k: self.k,
            dtype: self.dtype,
        }
    }
}

fn random_quantized(
    rows: usize,
    k: usize,
    dtype: DType,
    dev: &CudaDevice,
) -> Result<QuantizedActivation> {
    let device = candle_core::Device::Cuda(dev.clone());
    crate::mxfp8::quantize_packed(&random_activation(rows, k, dtype, &device)?)
}

/// Tunes one weight shape on synthetic weights.
fn tune_shape(key: TuneKey, dev: &CudaDevice, mode: TuneMode) -> Result<Vec<Tuned>> {
    let multiprocessors = device_multiprocessor_count(dev);
    let smem = crate::cuda_device::cuda_device_info(dev).smem_optin;
    let buckets = tune_buckets(key, multiprocessors);
    let space = move |bucket: Bucket| tune_space(bucket, key, multiprocessors, smem);
    let request = TuneRequest {
        kernel: TUNE_KERNEL,
        source_hash: kernels::_SOURCE_HASH,
        shape: format!("n{}_k{}_{:?}", key.n, key.k, key.dtype).to_ascii_lowercase(),
        buckets: &buckets,
        space: &space,
    };
    let mut weight = None;
    let mut inputs = std::collections::HashMap::new();
    Ok(TUNED.tune(key, dev, mode, &request, |rows, cfg| {
        if weight.is_none() {
            weight = Some(random_quantized(key.n, key.k, key.dtype, dev)?);
        }
        let weight = weight.as_ref().unwrap();
        let activation = probe_input(&mut inputs, rows, || {
            random_quantized(rows, key.k, key.dtype, dev)
        })?
        .clone();
        let (weight, scales) = (weight.quantized().clone(), weight.scales().clone());
        Prepared::from_launch(move || forward_config(&activation, &weight, &scales, Some(cfg)))
    }))
}

/// The static row set plus every divisor class of each tuned bucket.
fn warm_rows(key: TuneKey, multiprocessors: usize) -> Vec<usize> {
    let mut rows: Vec<usize> = WARM_ROWS
        .into_iter()
        .chain(bucket_warm_rows(
            &tune_buckets(key, multiprocessors),
            TUNE_MIN_ROWS + 1,
        ))
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

fn warm_shape(
    shape: WarmShape,
    rows: &[usize],
    dev: &CudaDevice,
    progress: &WarmupProgress,
) -> Result<()> {
    let stream = context::stream(dev);
    for &rows in rows {
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
                    let cfg = launch_config(shape.key(), rows, device_multiprocessor_count(dev));
                    dispatch_config(
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
                        DispatchCtx {
                            rows,
                            n: shape.n,
                            dev,
                            compile_only: true,
                            cfg,
                        },
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
