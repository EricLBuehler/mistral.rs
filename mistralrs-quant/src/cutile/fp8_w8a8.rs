//! Retained-weight E4M3 W8A8 GEMM for tensor and channel weight scales.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::{
    CudaDevice, CudaStorage, DType, Device, DeviceLocation, Result, Shape, Storage, Tensor,
};
use cutile::core::f8e4m3fn;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::{CompileOptions, TileKernel};
use float8::F8E4M3;
use half::{bf16, f16};

use super::tune::{
    buckets_from_breakpoints, config, cutile_error, tune, Bucket, Prepared, Space, TuneMode,
    TuneRequest, TunedTable, TUNE_WEIGHT_SETS,
};
use super::warmup::CutileKernel;
use super::{
    catch_cutile_panic, context, device_compute_capability, device_multiprocessor_count,
    jit_available,
};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};
use crate::{Fp8ActivationMode, Fp8WeightScaleLayout};

const TILE_SIZE: usize = 128;
const TUNE_KERNEL: &str = "fp8_w8a8";
const ROW_BREAKPOINTS: [usize; 3] = [16, 64, 256];
const PREFILL_PROBE_ROWS: usize = 1024;

#[cutile::module]
mod kernels {
    #![allow(deprecated)]
    use cutile::core::*;
    use cutile::cutile_compiler;

    #[cutile::entry(
        unchecked_accesses = false,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2, occupancy = 2,),
            sm_121 = (num_cta_in_cga = 2, occupancy = 2,),
        )
    )]
    fn fp8_w8a8_bf16<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
        const LATENCY: i32,
    >(
        mut y: MappedPartitionMut<bf16, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        xs: &Tensor<f32, { [-1] }>,
        ws: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pw = w.partition(const_shape![BN, BK]);
        let pxs = xs.partition(const_shape![BM]);
        let pws = ws.partition(const_shape![BN]);
        let k = num_tiles(&px, 1);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        for out_idx in y.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let xt: Tile<f8e4m3fn, { [BM, BK] }> = if LATENCY > 0 {
                    px.load_pipelined::<LATENCY>([bid_m, kg])
                } else {
                    px.load([bid_m, kg])
                };
                let wt: Tile<f8e4m3fn, { [BN, BK] }> = if LATENCY > 0 {
                    pw.load_pipelined::<LATENCY>([bid_n, kg])
                } else {
                    pw.load([bid_n, kg])
                };
                let wt: Tile<f8e4m3fn, { [BK, BN] }> = permute(wt, transpose);
                acc = mmaf(xt, wt, acc);
            }
            let sx: Tile<f32, { [BM] }> = pxs.load([bid_m]);
            let sx: Tile<f32, { [BM, BN] }> = sx
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BN]);
            let sw: Tile<f32, { [BN] }> = pws.load([bid_n]);
            let sw: Tile<f32, { [BM, BN] }> = sw
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BM, BN]);
            let out: Tile<bf16, { [BM, BN] }> = convert_tile(acc * sx * sw);
            y.store(out, out_idx);
        }
    }

    #[cutile::entry(
        unchecked_accesses = false,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2, occupancy = 2,),
            sm_121 = (num_cta_in_cga = 2, occupancy = 2,),
        )
    )]
    fn fp8_w8a8_f16<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
        const LATENCY: i32,
    >(
        mut y: MappedPartitionMut<f16, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        xs: &Tensor<f32, { [-1] }>,
        ws: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pw = w.partition(const_shape![BN, BK]);
        let pxs = xs.partition(const_shape![BM]);
        let pws = ws.partition(const_shape![BN]);
        let k = num_tiles(&px, 1);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        for out_idx in y.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let xt: Tile<f8e4m3fn, { [BM, BK] }> = if LATENCY > 0 {
                    px.load_pipelined::<LATENCY>([bid_m, kg])
                } else {
                    px.load([bid_m, kg])
                };
                let wt: Tile<f8e4m3fn, { [BN, BK] }> = if LATENCY > 0 {
                    pw.load_pipelined::<LATENCY>([bid_n, kg])
                } else {
                    pw.load([bid_n, kg])
                };
                let wt: Tile<f8e4m3fn, { [BK, BN] }> = permute(wt, transpose);
                acc = mmaf(xt, wt, acc);
            }
            let sx: Tile<f32, { [BM] }> = pxs.load([bid_m]);
            let sx: Tile<f32, { [BM, BN] }> = sx
                .reshape(const_shape![BM, 1])
                .broadcast(const_shape![BM, BN]);
            let sw: Tile<f32, { [BN] }> = pws.load([bid_n]);
            let sw: Tile<f32, { [BM, BN] }> = sw
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BM, BN]);
            let out: Tile<f16, { [BM, BN] }> = convert_tile(acc * sx * sw);
            y.store(out, out_idx);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum OutputDType {
    Bf16,
    F16,
}

impl TryFrom<DType> for OutputDType {
    type Error = candle_core::Error;

    fn try_from(dtype: DType) -> Result<Self> {
        match dtype {
            DType::BF16 => Ok(Self::Bf16),
            DType::F16 => Ok(Self::F16),
            dtype => candle_core::bail!("cuTile W8A8 does not support {dtype:?} output"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fp8W8A8Scheme {
    pub weight_scale: Fp8WeightScaleLayout,
    pub activation: Fp8ActivationMode,
    pub output_dtype: DType,
}

impl Fp8W8A8Scheme {
    fn validate(self) -> Result<()> {
        if !matches!(
            self.weight_scale,
            Fp8WeightScaleLayout::Tensor | Fp8WeightScaleLayout::Channel
        ) {
            candle_core::bail!(
                "cuTile W8A8 supports tensor or channel weight scales, got {:?}",
                self.weight_scale
            )
        }
        if !matches!(
            self.activation,
            Fp8ActivationMode::StaticTensor | Fp8ActivationMode::DynamicToken
        ) {
            candle_core::bail!(
                "cuTile W8A8 supports static tensor or dynamic token activations, got {:?}",
                self.activation
            )
        }
        let _ = OutputDType::try_from(self.output_dtype)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GemmShape {
    device: DeviceLocation,
    n: usize,
    k: usize,
    weight_scale: Fp8WeightScaleLayout,
    activation: Fp8ActivationMode,
    output_dtype: OutputDType,
}

impl GemmShape {
    fn new(n: usize, k: usize, scheme: Fp8W8A8Scheme, device: DeviceLocation) -> Result<Self> {
        scheme.validate()?;
        Ok(Self {
            device,
            n,
            k,
            weight_scale: scheme.weight_scale,
            activation: scheme.activation,
            output_dtype: scheme.output_dtype.try_into()?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CutileFp8W8A8Config {
    pub bm: i32,
    pub map_m: i32,
    pub map_n: i32,
    pub blocks_per_sm: i32,
    pub latency: i32,
    pub warps: i32,
    pub occupancy: i32,
    pub cluster: i32,
}

const POLICY_SMALL: CutileFp8W8A8Config = CutileFp8W8A8Config {
    bm: 16,
    map_m: 1,
    map_n: 8,
    blocks_per_sm: 2,
    latency: 0,
    warps: 0,
    occupancy: 0,
    cluster: 0,
};

const POLICY_LARGE: CutileFp8W8A8Config = CutileFp8W8A8Config {
    bm: 128,
    map_m: 8,
    map_n: 1,
    blocks_per_sm: 2,
    latency: 0,
    warps: 0,
    occupancy: 0,
    cluster: 0,
};

impl CutileFp8W8A8Config {
    fn to_config(self) -> cutile::tune::Config {
        config([
            ("bm", i64::from(self.bm)),
            ("map_m", i64::from(self.map_m)),
            ("map_n", i64::from(self.map_n)),
            ("blocks_per_sm", i64::from(self.blocks_per_sm)),
            ("latency", i64::from(self.latency)),
            ("warps", i64::from(self.warps)),
            ("occupancy", i64::from(self.occupancy)),
            ("cluster", i64::from(self.cluster)),
        ])
    }

    fn from_config(config: &cutile::tune::Config) -> Option<Self> {
        let int = |key: &str| config.int(key).and_then(|value| i32::try_from(value).ok());
        Some(Self {
            bm: int("bm")?,
            map_m: int("map_m")?,
            map_n: int("map_n")?,
            blocks_per_sm: int("blocks_per_sm")?,
            latency: int("latency")?,
            warps: int("warps")?,
            occupancy: int("occupancy")?,
            cluster: int("cluster")?,
        })
    }

    fn compile_options(self) -> CompileOptions {
        let mut options = CompileOptions::new();
        if self.warps > 0 {
            options = options.num_worker_warps_per_cta(self.warps);
        }
        if self.occupancy > 0 {
            options = options.occupancy(self.occupancy);
        }
        if self.cluster > 0 {
            options = options.num_cta_in_cga(self.cluster);
        }
        options
    }
}

static TUNED: TunedTable<GemmShape, CutileFp8W8A8Config> = TunedTable::new();

fn policy(rows: usize) -> CutileFp8W8A8Config {
    if rows <= ROW_BREAKPOINTS[0] {
        POLICY_SMALL
    } else {
        POLICY_LARGE
    }
}

fn gemm_config(shape: GemmShape, rows: usize) -> CutileFp8W8A8Config {
    TUNED.get(shape, rows).unwrap_or_else(|| policy(rows))
}

fn gemm_space(bucket: Bucket) -> Space {
    Space::new()
        .joint(["bm"], [[16], [32], [64], [128]])
        .joint(["map_m", "map_n"], [[1, 8], [2, 4], [4, 1], [8, 1], [1, 1]])
        .axis("blocks_per_sm", [2, 1, 4])
        .axis("latency", [0, 2, 4])
        .axis("warps", [0, 4, 8])
        .axis("occupancy", [0, 4])
        .axis("cluster", [0, 2])
        .policy(policy(bucket.probe).to_config())
}

fn gemm_buckets() -> Vec<Bucket> {
    buckets_from_breakpoints(&ROW_BREAKPOINTS, PREFILL_PROBE_ROWS, PREFILL_PROBE_ROWS)
}

pub struct Fp8W8A8Kernel;

pub(super) static FP8_W8A8: Fp8W8A8Kernel = Fp8W8A8Kernel;

#[derive(Clone)]
struct GemmWeights {
    weight: Tensor,
    scales: Tensor,
    key: GemmShape,
}

static SHAPES: OnceLock<Mutex<Vec<Vec<GemmWeights>>>> = OnceLock::new();

pub fn register_fp8_w8a8_shape(weight: &Tensor, weight_scales: &Tensor, scheme: Fp8W8A8Scheme) {
    let Ok((n, k)) = weight.dims2() else {
        return;
    };
    let Ok(key) = GemmShape::new(n, k, scheme, weight.device().location()) else {
        return;
    };
    let entry = GemmWeights {
        weight: weight.clone(),
        scales: weight_scales.clone(),
        key,
    };
    let mut shapes = SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if let Some(sets) = shapes.iter_mut().find(|sets| {
        sets[0].key == key && sets[0].weight.device().same_device(entry.weight.device())
    }) {
        if sets.len() < TUNE_WEIGHT_SETS {
            sets.push(entry);
            super::warmup::mark_dirty();
        }
        return;
    }
    shapes.push(vec![entry]);
    super::warmup::mark_dirty();
}

pub fn fp8_w8a8_supported(
    dev: &CudaDevice,
    output_features: usize,
    input_features: usize,
    activation_dtype: DType,
    scheme: Fp8W8A8Scheme,
) -> bool {
    let (major, minor) = device_compute_capability(dev);
    let has_native_fp8_mma = major > 8 || (major == 8 && minor >= 9);
    cfg!(has_blockwise_fp8_kernels)
        && jit_available(dev)
        && has_native_fp8_mma
        && matches!(activation_dtype, DType::BF16 | DType::F16)
        && output_features > 0
        && input_features > 0
        && output_features.is_multiple_of(TILE_SIZE)
        && input_features.is_multiple_of(TILE_SIZE)
        && scheme.validate().is_ok()
}

pub struct CutileFp8W8A8Args<'a> {
    pub weight: &'a Tensor,
    pub weight_scales: &'a Tensor,
    pub scheme: Fp8W8A8Scheme,
    pub activation_scale: Option<&'a Tensor>,
}

pub fn cutile_fp8_w8a8(activation: &Tensor, args: CutileFp8W8A8Args<'_>) -> Result<Tensor> {
    args.scheme.validate()?;
    if activation.rank() == 0 {
        candle_core::bail!("cuTile W8A8 activation cannot be scalar")
    }
    if !matches!(activation.dtype(), DType::BF16 | DType::F16) {
        candle_core::bail!("cuTile W8A8 activation must be BF16 or F16")
    }
    let source_shape = activation.dims().to_vec();
    let k = *source_shape.last().unwrap();
    let rows = source_shape[..source_shape.len() - 1]
        .iter()
        .try_fold(1usize, |rows, dim| rows.checked_mul(*dim))
        .ok_or_else(|| candle_core::Error::msg("cuTile W8A8 activation shape overflows usize"))?;
    let (n, weight_k) = args.weight.dims2()?;
    if weight_k != k {
        candle_core::bail!("cuTile W8A8 activation K={k} does not match weight K={weight_k}")
    }
    let Device::Cuda(dev) = activation.device() else {
        candle_core::bail!("cuTile W8A8 activation must be a CUDA tensor")
    };
    if !fp8_w8a8_supported(dev, n, k, activation.dtype(), args.scheme) {
        candle_core::bail!("cuTile W8A8 does not support rows={rows} n={n} k={k}")
    }
    let activation = activation.reshape((rows, k))?.contiguous()?;
    let (quantized, activation_scales) =
        quantize_activation(&activation, args.scheme.activation, args.activation_scale)?;
    let operands = GemmOperands {
        activation: &quantized,
        activation_scales: &activation_scales,
        weight: args.weight,
        weight_scales: args.weight_scales,
        scheme: args.scheme,
    };
    let key = GemmShape::new(n, k, args.scheme, activation.device().location())?;
    let output = launch(&operands, gemm_config(key, rows), false)?;
    let mut output_shape = source_shape[..source_shape.len() - 1].to_vec();
    output_shape.push(n);
    output.reshape(output_shape)
}

fn quantize_activation(
    activation: &Tensor,
    mode: Fp8ActivationMode,
    static_scale: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    match mode {
        Fp8ActivationMode::StaticTensor => {
            let scale = static_scale.ok_or_else(|| {
                candle_core::Error::msg("cuTile static W8A8 requires an activation scale")
            })?;
            if scale.elem_count() != 1 {
                candle_core::bail!(
                    "cuTile static W8A8 activation scale has shape {:?}, expected a scalar",
                    scale.dims()
                )
            }
            if !activation.device().same_device(scale.device()) {
                candle_core::bail!("cuTile W8A8 activation and scale must be on the same device")
            }
            let scale = scale.reshape(())?.to_dtype(DType::F32)?;
            let quantized =
                crate::blockwise_fp8::ops::fp8_quantize_activation_static(activation, &scale)?;
            Ok((quantized, scale))
        }
        Fp8ActivationMode::DynamicToken => {
            if static_scale.is_some() {
                candle_core::bail!("cuTile dynamic W8A8 does not use a stored activation scale")
            }
            crate::blockwise_fp8::ops::fp8_quantize_activation_rowwise(activation)
        }
        mode => candle_core::bail!("cuTile W8A8 does not support {mode:?} activations"),
    }
}

struct GemmOperands<'a> {
    activation: &'a Tensor,
    activation_scales: &'a Tensor,
    weight: &'a Tensor,
    weight_scales: &'a Tensor,
    scheme: Fp8W8A8Scheme,
}

fn validate_scale_shapes(operands: &GemmOperands<'_>, rows: usize, n: usize) -> Result<()> {
    let activation_valid = match operands.scheme.activation {
        Fp8ActivationMode::StaticTensor => operands.activation_scales.elem_count() == 1,
        Fp8ActivationMode::DynamicToken => operands.activation_scales.elem_count() == rows,
        _ => false,
    };
    let weight_valid = match operands.scheme.weight_scale {
        Fp8WeightScaleLayout::Tensor => operands.weight_scales.elem_count() == 1,
        Fp8WeightScaleLayout::Channel => operands.weight_scales.elem_count() == n,
        Fp8WeightScaleLayout::Block(_) => false,
    };
    if !activation_valid || !weight_valid {
        candle_core::bail!(
            "cuTile W8A8 scale shapes activation={:?} weight={:?} do not match rows={rows} n={n}",
            operands.activation_scales.dims(),
            operands.weight_scales.dims()
        )
    }
    Ok(())
}

fn launch(
    operands: &GemmOperands<'_>,
    cfg: CutileFp8W8A8Config,
    compile_only: bool,
) -> Result<Tensor> {
    operands.scheme.validate()?;
    let activation = operands.activation.contiguous()?;
    let activation_scales = operands.activation_scales.contiguous()?;
    let weight = operands.weight.contiguous()?;
    let weight_scales = operands.weight_scales.contiguous()?;
    let (rows, k) = activation.dims2()?;
    let (n, weight_k) = weight.dims2()?;
    if rows == 0
        || n == 0
        || k == 0
        || weight_k != k
        || !n.is_multiple_of(TILE_SIZE)
        || !k.is_multiple_of(TILE_SIZE)
    {
        candle_core::bail!("cuTile W8A8 got unsupported shape rows={rows} n={n} k={k}")
    }
    if activation.dtype() != DType::F8E4M3
        || weight.dtype() != DType::F8E4M3
        || activation_scales.dtype() != DType::F32
        || weight_scales.dtype() != DType::F32
    {
        candle_core::bail!("cuTile W8A8 needs E4M3 operands with F32 scales")
    }
    if !activation.device().same_device(weight.device())
        || !activation.device().same_device(activation_scales.device())
        || !activation.device().same_device(weight_scales.device())
    {
        candle_core::bail!("cuTile W8A8 operands must be on the same device")
    }
    let Device::Cuda(dev) = activation.device() else {
        candle_core::bail!("cuTile W8A8 operands must be CUDA tensors")
    };
    validate_scale_shapes(operands, rows, n)?;
    let bm = usize::try_from(cfg.bm).unwrap_or(0);
    if bm == 0 || !TILE_SIZE.is_multiple_of(bm) {
        candle_core::bail!("cuTile W8A8 row tile {} must divide {TILE_SIZE}", cfg.bm)
    }
    if cfg.map_m <= 0 || cfg.map_n <= 0 {
        candle_core::bail!("cuTile W8A8 map dimensions must be positive")
    }
    let padded_rows = rows.div_ceil(bm) * bm;
    let padded_activation;
    let activation = if padded_rows == rows {
        &activation
    } else {
        padded_activation = Tensor::zeros((padded_rows, k), DType::F8E4M3, activation.device())?;
        padded_activation.slice_set(&activation, 0, 0)?;
        &padded_activation
    };
    let padded_activation_scales;
    let activation_scales = match operands.scheme.activation {
        Fp8ActivationMode::StaticTensor => &activation_scales,
        Fp8ActivationMode::DynamicToken if padded_rows == rows => &activation_scales,
        Fp8ActivationMode::DynamicToken => {
            padded_activation_scales =
                Tensor::zeros(padded_rows, DType::F32, activation_scales.device())?;
            padded_activation_scales.slice_set(&activation_scales.flatten_all()?, 0, 0)?;
            &padded_activation_scales
        }
        _ => unreachable!(),
    };

    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let (a_storage, a_layout) = activation.storage_and_layout();
    let (as_storage, as_layout) = activation_scales.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let (ws_storage, ws_layout) = weight_scales.storage_and_layout();
    let (
        Storage::Cuda(a_cuda),
        Storage::Cuda(as_cuda),
        Storage::Cuda(w_cuda),
        Storage::Cuda(ws_cuda),
    ) = (&*a_storage, &*as_storage, &*w_storage, &*ws_storage)
    else {
        candle_core::bail!("cuTile W8A8 operands must be CUDA tensors")
    };
    let (a_addr, _a_guard) = slice_ptr_on_stream(
        a_cuda.as_cuda_slice::<F8E4M3>()?,
        a_layout.start_offset(),
        &stream,
    );
    let (as_addr, _as_guard) = slice_ptr_on_stream(
        as_cuda.as_cuda_slice::<f32>()?,
        as_layout.start_offset(),
        &stream,
    );
    let (w_addr, _w_guard) = slice_ptr_on_stream(
        w_cuda.as_cuda_slice::<F8E4M3>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (ws_addr, _ws_guard) = slice_ptr_on_stream(
        ws_cuda.as_cuda_slice::<f32>()?,
        ws_layout.start_offset(),
        &stream,
    );
    let dims = |r: usize, c: usize| (vec![r as i32, c as i32], vec![c as i32, 1]);
    let (shape, strides) = dims(padded_rows, k);
    let x = unsafe {
        cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            a_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        )
    };
    let (shape, strides) = dims(n, k);
    let w = unsafe {
        cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            w_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        )
    };
    let activation_scale_stride =
        usize::from(operands.scheme.activation == Fp8ActivationMode::DynamicToken);
    let xs = unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            as_addr as CUdeviceptr,
            ordinal,
            vec![padded_rows as i32],
            vec![activation_scale_stride as i32],
        )
    };
    let weight_scale_stride =
        usize::from(operands.scheme.weight_scale == Fp8WeightScaleLayout::Channel);
    let ws = unsafe {
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            ws_addr as CUdeviceptr,
            ordinal,
            vec![n as i32],
            vec![weight_scale_stride as i32],
        )
    };
    let tiles = (padded_rows / bm) * (n / TILE_SIZE);
    let blocks_per_sm = usize::try_from(cfg.blocks_per_sm).unwrap_or(1).max(1);
    let tile_blocks = (blocks_per_sm * device_multiprocessor_count(dev)).clamp(1, tiles) as u32;
    let generics = vec![
        cfg.bm.to_string(),
        TILE_SIZE.to_string(),
        TILE_SIZE.to_string(),
        cfg.map_m.to_string(),
        cfg.map_n.to_string(),
        cfg.latency.to_string(),
    ];
    let cutile_stream = context::stream(dev);

    macro_rules! run {
        ($launcher:expr, $label:literal) => {{
            let launcher = $launcher
                .generics(generics.clone())
                .compile_options(cfg.compile_options());
            if compile_only {
                catch_cutile_panic("W8A8 compile", || {
                    launcher.compile_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "cuTile {} compile failed: {error:?}",
                            $label
                        ))
                    })
                })?;
            } else {
                catch_cutile_panic("W8A8 launch", || unsafe {
                    launcher.async_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "cuTile {} launch failed: {error:?}",
                            $label
                        ))
                    })
                })?;
            }
        }};
    }

    let output = match operands.scheme.output_dtype {
        DType::BF16 => {
            let mut output = unsafe { dev.alloc::<bf16>(padded_rows * n)? };
            let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let (shape, strides) = dims(padded_rows, n);
            let y = unsafe {
                cutile::tensor::Tensor::<bf16>::borrow_raw_parts(
                    out_addr as CUdeviceptr,
                    ordinal,
                    shape,
                    strides,
                )
            };
            let mapped = y
                .partition([bm, TILE_SIZE])
                .map([cfg.map_m as usize, cfg.map_n as usize], tile_blocks);
            run!(
                kernels::fp8_w8a8_bf16(
                    mapped,
                    Arc::new(x),
                    Arc::new(w),
                    Arc::new(xs),
                    Arc::new(ws)
                ),
                "W8A8 BF16 GEMM"
            );
            drop(out_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from_dims(&[padded_rows, n]),
            ))
        }
        DType::F16 => {
            let mut output = unsafe { dev.alloc::<f16>(padded_rows * n)? };
            let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let (shape, strides) = dims(padded_rows, n);
            let y = unsafe {
                cutile::tensor::Tensor::<f16>::borrow_raw_parts(
                    out_addr as CUdeviceptr,
                    ordinal,
                    shape,
                    strides,
                )
            };
            let mapped = y
                .partition([bm, TILE_SIZE])
                .map([cfg.map_m as usize, cfg.map_n as usize], tile_blocks);
            run!(
                kernels::fp8_w8a8_f16(mapped, Arc::new(x), Arc::new(w), Arc::new(xs), Arc::new(ws)),
                "W8A8 F16 GEMM"
            );
            drop(out_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from_dims(&[padded_rows, n]),
            ))
        }
        _ => unreachable!(),
    };
    output.narrow(0, 0, rows)
}

struct GemmTuner {
    dev: CudaDevice,
    sets: Vec<GemmWeights>,
    operands: HashMap<usize, (Tensor, Tensor)>,
}

impl GemmTuner {
    fn new(dev: &CudaDevice, sets: &[GemmWeights]) -> Self {
        Self {
            dev: dev.clone(),
            sets: sets.to_vec(),
            operands: HashMap::new(),
        }
    }

    fn prepare(&mut self, rows: usize, cfg: CutileFp8W8A8Config) -> Result<Prepared> {
        let dev = self.dev.clone();
        let sets = self.sets.clone();
        let key = sets[0].key;
        if let std::collections::hash_map::Entry::Vacant(entry) = self.operands.entry(rows) {
            let device = Device::Cuda(dev.clone());
            let elements = rows.checked_mul(key.k).ok_or_else(|| {
                candle_core::Error::msg("cuTile W8A8 tuning activation shape overflows usize")
            })?;
            let values = (0..elements)
                .map(|index| F8E4M3::from_f32((index % 15) as f32 - 7.0))
                .collect::<Vec<_>>();
            let activation =
                Tensor::from_vec(values, (rows, key.k), &Device::Cpu)?.to_device(&device)?;
            let scales = match key.activation {
                Fp8ActivationMode::StaticTensor => Tensor::new(0.03125f32, &device)?,
                Fp8ActivationMode::DynamicToken => Tensor::from_vec(
                    (0..rows)
                        .map(|row| 0.015625f32 * (row % 7 + 1) as f32)
                        .collect::<Vec<_>>(),
                    rows,
                    &device,
                )?,
                _ => unreachable!(),
            };
            entry.insert((activation, scales));
        }
        let (activation, activation_scales) = self.operands[&rows].clone();
        let launch = move |weights: &GemmWeights, compile_only: bool| -> Result<Tensor> {
            let scheme = Fp8W8A8Scheme {
                weight_scale: key.weight_scale,
                activation: key.activation,
                output_dtype: match key.output_dtype {
                    OutputDType::Bf16 => DType::BF16,
                    OutputDType::F16 => DType::F16,
                },
            };
            let operands = GemmOperands {
                activation: &activation,
                activation_scales: &activation_scales,
                weight: &weights.weight,
                weight_scales: &weights.scales,
                scheme,
            };
            launch(&operands, cfg, compile_only)
        };
        launch(&sets[0], true)?;
        let sample = launch(&sets[0], false)?;
        let mut next = 0usize;
        let run = Box::new(move |_: &Arc<cutile::cuda_core::Stream>| {
            let weights = &sets[next % sets.len()];
            next += 1;
            launch(weights, false).map(|_| ()).map_err(cutile_error)
        });
        Ok(Prepared { run, sample })
    }
}

impl CutileKernel for Fp8W8A8Kernel {
    fn warm(&self, _dev: &CudaDevice) -> Result<()> {
        let shapes = std::mem::take(
            &mut *SHAPES
                .get_or_init(|| Mutex::new(Vec::new()))
                .lock()
                .unwrap(),
        );
        if shapes.is_empty() {
            return Ok(());
        }
        let mode = TuneMode::from_env();
        let buckets = gemm_buckets();
        for sets in &shapes {
            let key = sets[0].key;
            let Device::Cuda(dev) = sets[0].weight.device() else {
                continue;
            };
            let request = TuneRequest {
                kernel: TUNE_KERNEL,
                source_hash: kernels::_SOURCE_HASH,
                shape: format!(
                    "n{}_k{}_w{:?}_a{:?}_o{:?}",
                    key.n, key.k, key.weight_scale, key.activation, key.output_dtype
                ),
                buckets: &buckets,
                space: &gemm_space,
            };
            let mut tuner = GemmTuner::new(dev, sets);
            let tuned = tune(dev, mode, &request, |rows, candidate| {
                let cfg = CutileFp8W8A8Config::from_config(candidate)
                    .ok_or_else(|| candle_core::Error::msg("config outside the space"))?;
                tuner.prepare(rows, cfg)
            });
            TUNED.set(key, &tuned, CutileFp8W8A8Config::from_config);
        }
        tracing::info!("Warming {} cuTile W8A8 GEMM kernels.", shapes.len());
        for sets in &shapes {
            let key = sets[0].key;
            let Device::Cuda(dev) = sets[0].weight.device() else {
                continue;
            };
            let device = Device::Cuda(dev.clone());
            let scheme = Fp8W8A8Scheme {
                weight_scale: key.weight_scale,
                activation: key.activation,
                output_dtype: match key.output_dtype {
                    OutputDType::Bf16 => DType::BF16,
                    OutputDType::F16 => DType::F16,
                },
            };
            for bucket in &buckets {
                let result = (|| -> Result<()> {
                    let activation = Tensor::zeros((bucket.probe, key.k), DType::F8E4M3, &device)?;
                    let activation_scales = match key.activation {
                        Fp8ActivationMode::StaticTensor => Tensor::ones((), DType::F32, &device)?,
                        Fp8ActivationMode::DynamicToken => {
                            Tensor::ones(bucket.probe, DType::F32, &device)?
                        }
                        _ => unreachable!(),
                    };
                    let operands = GemmOperands {
                        activation: &activation,
                        activation_scales: &activation_scales,
                        weight: &sets[0].weight,
                        weight_scales: &sets[0].scales,
                        scheme,
                    };
                    launch(&operands, gemm_config(key, bucket.probe), true)?;
                    Ok(())
                })();
                if let Err(error) = result {
                    tracing::warn!(
                        "cuTile W8A8 warmup failed (n={} k={} rows={}): {error}",
                        key.n,
                        key.k,
                        bucket.probe
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::{
        cutile_fp8_w8a8, quantize_activation, validate_scale_shapes, CutileFp8W8A8Args,
        Fp8W8A8Scheme, GemmOperands,
    };
    use crate::{Fp8ActivationMode, Fp8WeightScaleLayout};

    #[test]
    fn scale_shapes_are_validated_by_scheme() -> Result<()> {
        let device = Device::Cpu;
        let activation = Tensor::zeros((7, 128), DType::F8E4M3, &device)?;
        let weight = Tensor::zeros((256, 128), DType::F8E4M3, &device)?;
        let scalar = Tensor::zeros((), DType::F32, &device)?;
        let token = Tensor::zeros(7, DType::F32, &device)?;
        let channel = Tensor::zeros(256, DType::F32, &device)?;
        let static_tensor = Fp8W8A8Scheme {
            weight_scale: Fp8WeightScaleLayout::Tensor,
            activation: Fp8ActivationMode::StaticTensor,
            output_dtype: DType::BF16,
        };
        validate_scale_shapes(
            &GemmOperands {
                activation: &activation,
                activation_scales: &scalar,
                weight: &weight,
                weight_scales: &scalar,
                scheme: static_tensor,
            },
            7,
            256,
        )?;
        validate_scale_shapes(
            &GemmOperands {
                activation: &activation,
                activation_scales: &token,
                weight: &weight,
                weight_scales: &channel,
                scheme: Fp8W8A8Scheme {
                    weight_scale: Fp8WeightScaleLayout::Channel,
                    activation: Fp8ActivationMode::DynamicToken,
                    output_dtype: DType::F16,
                },
            },
            7,
            256,
        )?;
        assert!(validate_scale_shapes(
            &GemmOperands {
                activation: &activation,
                activation_scales: &token,
                weight: &weight,
                weight_scales: &scalar,
                scheme: static_tensor,
            },
            7,
            256,
        )
        .is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_fp8_w8a8_matches_dequantized_reference() -> Result<()> {
        use float8::F8E4M3;

        const ROWS: usize = 17;
        const N: usize = 256;
        const K: usize = 256;

        let device = Device::new_cuda(0)?;
        let Device::Cuda(_cuda) = &device else {
            unreachable!()
        };
        let weight_values = (0..N * K)
            .map(|index| F8E4M3::from_f32((index % 17) as f32 - 8.0))
            .collect::<Vec<_>>();
        let weight = Tensor::from_vec(weight_values, (N, K), &Device::Cpu)?.to_device(&device)?;
        let input_values = (0..ROWS * K)
            .map(|index| match index {
                index if index < K => 0.0,
                index if index == K => 1000.0,
                index if index == K + 1 => -1000.0,
                _ => ((index * 7919 + 104729) % 2001) as f32 / 160.0 - 6.0,
            })
            .collect::<Vec<_>>();

        for (dtype, weight_layout, activation_mode) in [
            (
                DType::BF16,
                Fp8WeightScaleLayout::Tensor,
                Fp8ActivationMode::StaticTensor,
            ),
            (
                DType::BF16,
                Fp8WeightScaleLayout::Channel,
                Fp8ActivationMode::DynamicToken,
            ),
            (
                DType::F16,
                Fp8WeightScaleLayout::Tensor,
                Fp8ActivationMode::DynamicToken,
            ),
            (
                DType::F16,
                Fp8WeightScaleLayout::Channel,
                Fp8ActivationMode::StaticTensor,
            ),
        ] {
            let input =
                Tensor::from_vec(input_values.clone(), (1, ROWS, K), &device)?.to_dtype(dtype)?;
            let weight_scales = match weight_layout {
                Fp8WeightScaleLayout::Tensor => Tensor::new(0.03125f32, &device)?,
                Fp8WeightScaleLayout::Channel => Tensor::from_vec(
                    (0..N)
                        .map(|index| 0.015625f32 * (1.0 + (index % 5) as f32))
                        .collect::<Vec<_>>(),
                    N,
                    &device,
                )?,
                Fp8WeightScaleLayout::Block(_) => unreachable!(),
            };
            let static_scale = (activation_mode == Fp8ActivationMode::StaticTensor)
                .then(|| Tensor::new(0.031337f32, &device))
                .transpose()?;
            let scheme = Fp8W8A8Scheme {
                weight_scale: weight_layout,
                activation: activation_mode,
                output_dtype: dtype,
            };
            let direct = cutile_fp8_w8a8(
                &input,
                CutileFp8W8A8Args {
                    weight: &weight,
                    weight_scales: &weight_scales,
                    scheme,
                    activation_scale: static_scale.as_ref(),
                },
            )?;
            let runtime = crate::fp8_w8a8_linear(crate::Fp8W8A8LinearArgs {
                weight: weight.clone(),
                weight_scale: weight_scales.clone(),
                weight_scale_layout: weight_layout,
                activation_mode,
                activation_scale: static_scale.clone(),
                bias: None,
                dequant_dtype: dtype,
            })?
            .forward(&input)?;
            let runtime_error = direct
                .to_dtype(DType::F32)?
                .sub(&runtime.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            assert_eq!(runtime_error, 0.0);
            let output = direct
                .reshape((ROWS, N))?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?;

            let flat = input.reshape((ROWS, K))?;
            let (activation_q, activation_scales) =
                quantize_activation(&flat, activation_mode, static_scale.as_ref())?;
            if activation_mode == Fp8ActivationMode::DynamicToken {
                let source = flat
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .to_vec2::<f32>()?;
                let scales = activation_scales
                    .to_device(&Device::Cpu)?
                    .to_vec1::<f32>()?;
                for (row, scale) in source.iter().zip(scales) {
                    let expected = (row.iter().copied().map(f32::abs).fold(0.0, f32::max) / 448.0)
                        .max(1.0e-12);
                    assert!(
                        (scale - expected).abs() <= expected.max(1.0e-6) * 1.0e-5,
                        "dynamic scale {scale} does not match {expected}"
                    );
                }
            } else {
                let source = flat
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .to_vec2::<f32>()?;
                let quantized = activation_q.to_device(&Device::Cpu)?.to_vec2::<F8E4M3>()?;
                let scale = static_scale.as_ref().unwrap().to_scalar::<f32>()?;
                for (source, quantized) in source.iter().zip(quantized) {
                    for (source, quantized) in source.iter().zip(quantized) {
                        let expected = F8E4M3::from_f32((source / scale).clamp(-448.0, 448.0));
                        assert_eq!(quantized.to_f32(), expected.to_f32());
                    }
                }
            }
            let activation_q = crate::scalar_fp8::ops::fp8_to_dtype(
                &activation_q.to_device(&Device::Cpu)?,
                DType::F32,
            )?;
            let activation_scales = activation_scales.to_device(&Device::Cpu)?;
            let activation_ref = match activation_mode {
                Fp8ActivationMode::StaticTensor => {
                    activation_q.broadcast_mul(&activation_scales)?
                }
                Fp8ActivationMode::DynamicToken => activation_q
                    .broadcast_mul(&activation_scales.flatten_all()?.reshape((ROWS, 1))?)?,
                _ => unreachable!(),
            };
            let weight_q =
                crate::scalar_fp8::ops::fp8_to_dtype(&weight.to_device(&Device::Cpu)?, DType::F32)?;
            let weight_scales = weight_scales.to_device(&Device::Cpu)?;
            let weight_ref = match weight_layout {
                Fp8WeightScaleLayout::Tensor => weight_q.broadcast_mul(&weight_scales)?,
                Fp8WeightScaleLayout::Channel => {
                    weight_q.broadcast_mul(&weight_scales.reshape((N, 1))?)?
                }
                Fp8WeightScaleLayout::Block(_) => unreachable!(),
            };
            let reference = activation_ref.matmul(&weight_ref.t()?)?;
            let max_error = output
                .sub(&reference)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let max_reference = reference.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                max_error <= 0.03 * max_reference.max(1.0),
                "dtype={dtype:?} weight={weight_layout:?} activation={activation_mode:?}: max error {max_error} vs {max_reference}"
            );
        }

        let input =
            Tensor::from_vec(input_values[..K].to_vec(), K, &device)?.to_dtype(DType::BF16)?;
        let weight_scales = Tensor::new(0.03125f32, &device)?;
        let scheme = Fp8W8A8Scheme {
            weight_scale: Fp8WeightScaleLayout::Tensor,
            activation: Fp8ActivationMode::DynamicToken,
            output_dtype: DType::BF16,
        };
        let direct = cutile_fp8_w8a8(
            &input,
            CutileFp8W8A8Args {
                weight: &weight,
                weight_scales: &weight_scales,
                scheme,
                activation_scale: None,
            },
        )?;
        let runtime = crate::fp8_w8a8_linear(crate::Fp8W8A8LinearArgs {
            weight,
            weight_scale: weight_scales,
            weight_scale_layout: Fp8WeightScaleLayout::Tensor,
            activation_mode: Fp8ActivationMode::DynamicToken,
            activation_scale: None,
            bias: None,
            dequant_dtype: DType::BF16,
        })?
        .forward(&input)?;
        assert_eq!(direct.dims(), [N]);
        assert_eq!(runtime.dims(), [N]);
        assert_eq!(
            direct
                .to_dtype(DType::F32)?
                .sub(&runtime.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?,
            0.0
        );
        Ok(())
    }
}
