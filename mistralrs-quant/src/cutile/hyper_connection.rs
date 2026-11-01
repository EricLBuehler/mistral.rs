use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::cuda::cudarc::driver::DeviceRepr;
use candle_core::cuda_backend::DeviceId;
use candle_core::{CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor};
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::{CompileOptions, TileKernel};
use half::{bf16, f16};

use super::warmup::{CutileKernel, WarmupProgress};
use super::{blackwell_tile_features_supported, catch_cutile_panic, context};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

const TILE_ROWS: usize = 64;
const TILE_COLUMNS: usize = 128;
const TILE_RANK: usize = 64;
const WORKER_WARPS: i32 = 4;
const MIN_ROWS: usize = 512;
const ALIGNMENT_ELEMENTS: usize = 8;
const MIN_BLOCKS_PER_SM: usize = 2;
const CACHE_BUDGET_PARTS: usize = 3;
const WARMUP_ROWS: [usize; 5] = [
    MIN_ROWS,
    MIN_ROWS + 1,
    MIN_ROWS + 2,
    MIN_ROWS + 4,
    MIN_ROWS + 8,
];

#[cutile::module]
mod kernels {
    use cutile::core::*;

    #[cutile::entry(unchecked_accesses = false)]
    fn up_mix<T: ElementType, const BM: i32, const BN: i32, const BK: i32, const STREAMS: i32>(
        mut y: MappedPartitionMut<T, { [BM, BN] }, { [1, 1] }>,
        lo: &Tensor<T, { [-1, -1] }>,
        up: &Tensor<T, { [-1, -1, -1] }>,
        xn: &Tensor<T, { [-1, -1, -1] }>,
        streams_f: f32,
    ) {
        let pl = lo.partition(const_shape![BM, BK]);
        let pw = up.partition(const_shape![1, BN, BK]);
        let px = xn.partition(const_shape![BM, 1, BN]);
        let steps = num_tiles(&pl, 1);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        let one: Tile<f32, { [BM, BN] }> = constant(1.0f32, const_shape![BM, BN]);
        let neg_log2_e: Tile<f32, { [BM, BN] }> =
            constant(-1.4426950216293334961f32, const_shape![BM, BN]);
        let divisor: Tile<f32, { [BM, BN] }> = broadcast_scalar(streams_f, const_shape![BM, BN]);
        let inv_streams = divf(one, divisor, rounding::Approx, ftz::Enabled);
        for out_idx in y.iter_indices() {
            let (row, col) = out_idx.components();
            let mut mixed: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for stream in 0..STREAMS {
                let mut logits: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                for step in 0..steps {
                    let a: Tile<T, { [BM, BK] }> = pl.load([row, step]);
                    let b: Tile<T, { [1, BN, BK] }> = pw.load([stream, col, step]);
                    let b = b.reshape(const_shape![BN, BK]);
                    let b: Tile<T, { [BK, BN] }> = permute(b, transpose);
                    logits = mmaf(a, b, logits);
                }
                let rounded: Tile<T, { [BM, BN] }> = ftof(logits, rounding::NearestEven);
                let logits: Tile<f32, { [BM, BN] }> = ftof(rounded, rounding::NearestEven);
                let scaled = mulf(logits, neg_log2_e, rounding::NearestEven, ftz::Enabled);
                let exponential = exp2(scaled, ftz::Enabled);
                let denominator = addf(one, exponential, rounding::NearestEven, ftz::Enabled);
                let gate = divf(one, denominator, rounding::Approx, ftz::Enabled);
                let x: Tile<T, { [BM, 1, BN] }> = px.load([row, stream, col]);
                let x: Tile<f32, { [BM, BN] }> = convert_tile(x.reshape(const_shape![BM, BN]));
                mixed = fma(x, gate, mixed, rounding::NearestEven, ftz::Enabled);
            }
            let mean = mulf(mixed, inv_streams, rounding::NearestEven, ftz::Enabled);
            let out: Tile<T, { [BM, BN] }> = ftof(mean, rounding::NearestEven);
            y.store(out, out_idx);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ShapeKey {
    device: DeviceId,
    dtype: DType,
    width: usize,
    rank: usize,
    streams: usize,
}

#[derive(Clone)]
struct RegisteredShape {
    key: ShapeKey,
    device: CudaDevice,
    hardware: DeviceGeometry,
}

#[derive(Clone, Copy)]
struct DeviceGeometry {
    multiprocessors: usize,
    cache_budget: usize,
}

impl DeviceGeometry {
    fn read(dev: &CudaDevice) -> Option<Self> {
        let info = crate::cuda_device::cuda_device_info(dev);
        (info.l2_bytes >= CACHE_BUDGET_PARTS).then_some(Self {
            multiprocessors: info.sms,
            cache_budget: info.l2_bytes / CACHE_BUDGET_PARTS,
        })
    }

    fn permits(self, shape: ShapeKey, rows: usize) -> bool {
        let element_bytes = shape.dtype.size_in_bytes();
        let blocks =
            rows.div_ceil(TILE_ROWS) * (shape.width / shape.streams).div_ceil(TILE_COLUMNS);
        self.permits_workload(
            rows,
            shape.width * shape.rank * element_bytes,
            rows * shape.width * element_bytes,
            blocks,
        )
    }

    fn permits_workload(
        self,
        rows: usize,
        weight_bytes: usize,
        input_bytes: usize,
        blocks: usize,
    ) -> bool {
        rows >= MIN_ROWS
            && blocks >= self.multiprocessors * MIN_BLOCKS_PER_SM
            && weight_bytes <= self.cache_budget
            && input_bytes >= self.cache_budget
    }
}

static SHAPES: OnceLock<Mutex<Vec<RegisteredShape>>> = OnceLock::new();
static READY: OnceLock<Mutex<HashMap<ShapeKey, DeviceGeometry>>> = OnceLock::new();

fn weight_shape(weight: &Tensor, streams: usize) -> Option<ShapeKey> {
    if streams == 0
        || weight.rank() != 2
        || !matches!(weight.dtype(), DType::BF16 | DType::F16)
        || !weight.is_contiguous()
        || !weight
            .layout()
            .start_offset()
            .is_multiple_of(ALIGNMENT_ELEMENTS)
    {
        return None;
    }
    let width = weight.dims()[0];
    let rank = weight.dims()[1];
    if !width.is_multiple_of(streams)
        || !(width / streams).is_multiple_of(TILE_COLUMNS)
        || !rank.is_multiple_of(TILE_RANK)
        || width == 0
        || rank == 0
    {
        return None;
    }
    let Device::Cuda(dev) = weight.device() else {
        return None;
    };
    Some(ShapeKey {
        device: dev.id(),
        dtype: weight.dtype(),
        width,
        rank,
        streams,
    })
}

pub fn hc_prefill_supported(dev: &CudaDevice) -> bool {
    blackwell_tile_features_supported(dev)
}

pub fn register_hc_up_mix(weight: &Tensor, streams: usize) {
    let Some(key) = weight_shape(weight, streams) else {
        return;
    };
    let Device::Cuda(dev) = weight.device() else {
        return;
    };
    let Some(hardware) = hc_prefill_supported(dev)
        .then(|| DeviceGeometry::read(dev))
        .flatten()
    else {
        return;
    };
    if weight.elem_count() * weight.dtype().size_in_bytes() > hardware.cache_budget {
        return;
    }
    let mut shapes = SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if shapes.iter().any(|existing| existing.key == key) {
        return;
    }
    shapes.push(RegisteredShape {
        key,
        device: dev.clone(),
        hardware,
    });
    super::warmup::mark_dirty();
}

struct UpMix<'a> {
    xn: &'a Tensor,
    lo: &'a Tensor,
    weight: &'a Tensor,
    shape: ShapeKey,
}

pub fn try_hc_prefill_up_mix(
    xn: &Tensor,
    lo: &Tensor,
    weight: &Tensor,
    streams: usize,
) -> Result<Option<Tensor>> {
    let Some(shape) = weight_shape(weight, streams) else {
        return Ok(None);
    };
    if !xn.is_contiguous()
        || !lo.is_contiguous()
        || xn.dtype() != shape.dtype
        || lo.dtype() != shape.dtype
        || !xn.device().same_device(weight.device())
        || !lo.device().same_device(weight.device())
        || !xn
            .layout()
            .start_offset()
            .is_multiple_of(ALIGNMENT_ELEMENTS)
        || !lo
            .layout()
            .start_offset()
            .is_multiple_of(ALIGNMENT_ELEMENTS)
    {
        return Ok(None);
    }
    let Some(hardware) = READY
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap()
        .get(&shape)
        .copied()
    else {
        return Ok(None);
    };
    let width = xn.dim(candle_core::D::Minus1)?;
    let rank = lo.dim(candle_core::D::Minus1)?;
    if shape.width != width || shape.rank != rank {
        return Ok(None);
    }
    let rows = xn.elem_count() / width;
    if !hardware.permits(shape, rows) || lo.elem_count() / rank != rows {
        return Ok(None);
    }
    let args = UpMix {
        xn,
        lo,
        weight,
        shape,
    };
    match shape.dtype {
        DType::BF16 => launch::<bf16>(&args).map(Some),
        DType::F16 => launch::<f16>(&args).map(Some),
        _ => unreachable!(),
    }
}

struct Tiles<T: cutile::DType> {
    output: cutile::tensor::Tensor<T>,
    lo: cutile::tensor::Tensor<T>,
    up: cutile::tensor::Tensor<T>,
    xn: cutile::tensor::Tensor<T>,
}

fn dispatch<T: cutile::DType>(
    tiles: Tiles<T>,
    shape: ShapeKey,
    rows: usize,
    dev: &CudaDevice,
    compile_only: bool,
) -> Result<()> {
    let hidden = shape.width / shape.streams;
    let blocks = rows.div_ceil(TILE_ROWS) * hidden.div_ceil(TILE_COLUMNS);
    let mapped = tiles
        .output
        .partition([TILE_ROWS, TILE_COLUMNS])
        .map([1, 1], blocks as u32);
    let options = CompileOptions::new()
        .num_worker_warps_per_cta(WORKER_WARPS)
        .num_cta_in_cga(1);
    let dtype = match shape.dtype {
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        _ => unreachable!(),
    };
    let generics = vec![
        dtype.to_owned(),
        TILE_ROWS.to_string(),
        TILE_COLUMNS.to_string(),
        TILE_RANK.to_string(),
        shape.streams.to_string(),
    ];
    let kernel = kernels::up_mix(
        mapped,
        Arc::new(tiles.lo),
        Arc::new(tiles.up),
        Arc::new(tiles.xn),
        shape.streams as f32,
    )
    .generics(generics)
    .compile_options(options);
    let stream = context::stream(dev);
    catch_cutile_panic("hyper-connection up-mix", || {
        if compile_only {
            kernel.compile_on(&stream).map_err(|error| {
                candle_core::Error::msg(format!("cuTile hyper-connection compile: {error:?}"))
            })?;
        } else {
            unsafe { kernel.async_on(&stream) }.map_err(|error| {
                candle_core::Error::msg(format!("cuTile hyper-connection launch: {error:?}"))
            })?;
        }
        Ok(())
    })
}

fn launch<T>(args: &UpMix<'_>) -> Result<Tensor>
where
    T: DeviceRepr + candle_core::cuda_backend::CudaDType + cutile::DType + Send,
{
    let ShapeKey {
        width,
        rank,
        streams,
        ..
    } = args.shape;
    let rows = args.xn.elem_count() / width;
    let hidden = width / streams;
    let dev = args.xn.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let (ls, ll) = args.lo.storage_and_layout();
    let (ws, wl) = args.weight.storage_and_layout();
    let (xs, xl) = args.xn.storage_and_layout();
    let (Storage::Cuda(ls), Storage::Cuda(ws), Storage::Cuda(xs)) = (&*ls, &*ws, &*xs) else {
        unreachable!()
    };
    let (lp, _lg) = slice_ptr_on_stream(ls.as_cuda_slice::<T>()?, ll.start_offset(), &stream);
    let (wp, _wg) = slice_ptr_on_stream(ws.as_cuda_slice::<T>()?, wl.start_offset(), &stream);
    let (xp, _xg) = slice_ptr_on_stream(xs.as_cuda_slice::<T>()?, xl.start_offset(), &stream);
    let mut output = unsafe { dev.alloc::<T>(rows * hidden)? };
    let (op, og) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
    let borrow = |ptr, dims: &[usize]| unsafe {
        let mut step = 1;
        let mut strides = Vec::new();
        for &dim in dims.iter().rev() {
            strides.push(step as i32);
            step *= dim;
        }
        strides.reverse();
        cutile::tensor::Tensor::<T>::borrow_raw_parts(
            ptr,
            ordinal,
            dims.iter().map(|&d| d as i32).collect(),
            strides,
        )
    };
    dispatch(
        Tiles {
            lo: borrow(lp, &[rows, rank]),
            up: borrow(wp, &[streams, hidden, rank]),
            xn: borrow(xp, &[rows, streams, hidden]),
            output: borrow(op, &[rows, hidden]),
        },
        args.shape,
        rows,
        dev,
        false,
    )?;
    drop(og);
    let mut shape = args.xn.dims().to_vec();
    *shape.last_mut().unwrap() = hidden;
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
        Shape::from(shape),
    )))
}

pub(super) struct HyperConnectionKernel;
pub(super) static HYPER_CONNECTION: HyperConnectionKernel = HyperConnectionKernel;

fn warm_shape<T: cutile::DType>(entry: &RegisteredShape, rows: usize) -> Result<()> {
    let shape = entry.key;
    let stream = context::stream(&entry.device);
    let meta = |dimensions: &[usize]| {
        unsafe { cutile::api::meta::<T>(dimensions).async_on(&stream) }.map_err(|error| {
            candle_core::Error::msg(format!("cuTile hyper-connection metadata: {error:?}"))
        })
    };
    dispatch(
        Tiles {
            lo: meta(&[rows, shape.rank])?,
            up: meta(&[shape.streams, shape.width / shape.streams, shape.rank])?,
            xn: meta(&[rows, shape.streams, shape.width / shape.streams])?,
            output: meta(&[rows, shape.width / shape.streams])?,
        },
        shape,
        rows,
        &entry.device,
        true,
    )
}

impl CutileKernel for HyperConnectionKernel {
    fn warm(&self, _dev: &CudaDevice, silent: bool) -> Result<()> {
        let shapes = SHAPES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap()
            .clone();
        let pending = {
            let ready = READY
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .unwrap();
            shapes
                .into_iter()
                .filter(|entry| !ready.contains_key(&entry.key))
                .collect::<Vec<_>>()
        };
        let progress = WarmupProgress::new(
            "hyper-connection up-mix",
            pending.len() * WARMUP_ROWS.len(),
            silent,
        );
        for entry in &pending {
            for rows in WARMUP_ROWS {
                progress.run(
                    format!(
                        "width={} rank={} rows={rows}",
                        entry.key.width, entry.key.rank
                    ),
                    || match entry.key.dtype {
                        DType::BF16 => warm_shape::<bf16>(entry, rows),
                        DType::F16 => warm_shape::<f16>(entry, rows),
                        _ => unreachable!(),
                    },
                )?;
            }
            READY
                .get()
                .unwrap()
                .lock()
                .unwrap()
                .insert(entry.key, entry.hardware);
        }
        Ok(())
    }
}
