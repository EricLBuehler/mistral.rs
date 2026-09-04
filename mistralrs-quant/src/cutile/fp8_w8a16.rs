//! Retained-weight E4M3 W8A16 GEMM for tensor, channel, and 128x128 block scales.

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
use super::{catch_cutile_panic, context, device_multiprocessor_count, jit_available};
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};
use crate::Fp8WeightScaleLayout;

const BLOCK_SIZE: usize = 128;
const TUNE_KERNEL: &str = "fp8_w8a16";
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
    fn fp8_w8a16_post_bf16<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
        const LATENCY: i32,
    >(
        mut y: MappedPartitionMut<bf16, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<bf16, { [-1, -1] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        ws: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pw = w.partition(const_shape![BN, BK]);
        let k = num_tiles(&px, 1);
        let pws = ws.partition(const_shape![BN]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        for out_idx in y.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let xt: Tile<bf16, { [BM, BK] }> = if LATENCY > 0 {
                    px.load_pipelined::<LATENCY>([bid_m, kg])
                } else {
                    px.load([bid_m, kg])
                };
                let wt: Tile<f8e4m3fn, { [BN, BK] }> = if LATENCY > 0 {
                    pw.load_pipelined::<LATENCY>([bid_n, kg])
                } else {
                    pw.load([bid_n, kg])
                };
                let wt: Tile<bf16, { [BN, BK] }> = convert_tile(wt);
                let wt: Tile<bf16, { [BK, BN] }> = permute(wt, transpose);
                acc = mmaf(xt, wt, acc);
            }
            let scale: Tile<f32, { [BN] }> = pws.load([bid_n]);
            let scale: Tile<f32, { [BM, BN] }> = scale
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BM, BN]);
            let out: Tile<bf16, { [BM, BN] }> = convert_tile(acc * scale);
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
    fn fp8_w8a16_block_bf16<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
        const LATENCY: i32,
    >(
        mut y: MappedPartitionMut<bf16, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<bf16, { [-1, -1] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        ws: &Tensor<f32, { [-1, -1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pw = w.partition(const_shape![BN, BK]);
        let pws = ws.partition(const_shape![1, 1]);
        let k = num_tiles(&px, 1);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        for out_idx in y.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let xt: Tile<bf16, { [BM, BK] }> = if LATENCY > 0 {
                    px.load_pipelined::<LATENCY>([bid_m, kg])
                } else {
                    px.load([bid_m, kg])
                };
                let wt: Tile<f8e4m3fn, { [BN, BK] }> = if LATENCY > 0 {
                    pw.load_pipelined::<LATENCY>([bid_n, kg])
                } else {
                    pw.load([bid_n, kg])
                };
                let wt: Tile<bf16, { [BN, BK] }> = convert_tile(wt);
                let wt: Tile<bf16, { [BK, BN] }> = permute(wt, transpose);
                let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                let part: Tile<f32, { [BM, BN] }> = mmaf(xt, wt, zero);
                let scale: Tile<f32, { [1, 1] }> = pws.load([bid_n, kg]);
                acc = acc + part * scale.broadcast(const_shape![BM, BN]);
            }
            let out: Tile<bf16, { [BM, BN] }> = convert_tile(acc);
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
    fn fp8_w8a16_post_f16<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
        const LATENCY: i32,
    >(
        mut y: MappedPartitionMut<f16, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f16, { [-1, -1] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        ws: &Tensor<f32, { [-1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pw = w.partition(const_shape![BN, BK]);
        let k = num_tiles(&px, 1);
        let pws = ws.partition(const_shape![BN]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        for out_idx in y.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let xt: Tile<f16, { [BM, BK] }> = if LATENCY > 0 {
                    px.load_pipelined::<LATENCY>([bid_m, kg])
                } else {
                    px.load([bid_m, kg])
                };
                let wt: Tile<f8e4m3fn, { [BN, BK] }> = if LATENCY > 0 {
                    pw.load_pipelined::<LATENCY>([bid_n, kg])
                } else {
                    pw.load([bid_n, kg])
                };
                let wt: Tile<f16, { [BN, BK] }> = convert_tile(wt);
                let wt: Tile<f16, { [BK, BN] }> = permute(wt, transpose);
                acc = mmaf(xt, wt, acc);
            }
            let scale: Tile<f32, { [BN] }> = pws.load([bid_n]);
            let scale: Tile<f32, { [BM, BN] }> = scale
                .reshape(const_shape![1, BN])
                .broadcast(const_shape![BM, BN]);
            let out: Tile<f16, { [BM, BN] }> = convert_tile(acc * scale);
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
    fn fp8_w8a16_block_f16<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
        const LATENCY: i32,
    >(
        mut y: MappedPartitionMut<f16, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f16, { [-1, -1] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        ws: &Tensor<f32, { [-1, -1] }>,
    ) {
        let px = x.partition(const_shape![BM, BK]);
        let pw = w.partition(const_shape![BN, BK]);
        let pws = ws.partition(const_shape![1, 1]);
        let k = num_tiles(&px, 1);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        for out_idx in y.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in 0..k {
                let xt: Tile<f16, { [BM, BK] }> = if LATENCY > 0 {
                    px.load_pipelined::<LATENCY>([bid_m, kg])
                } else {
                    px.load([bid_m, kg])
                };
                let wt: Tile<f8e4m3fn, { [BN, BK] }> = if LATENCY > 0 {
                    pw.load_pipelined::<LATENCY>([bid_n, kg])
                } else {
                    pw.load([bid_n, kg])
                };
                let wt: Tile<f16, { [BN, BK] }> = convert_tile(wt);
                let wt: Tile<f16, { [BK, BN] }> = permute(wt, transpose);
                let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                let part: Tile<f32, { [BM, BN] }> = mmaf(xt, wt, zero);
                let scale: Tile<f32, { [1, 1] }> = pws.load([bid_n, kg]);
                acc = acc + part * scale.broadcast(const_shape![BM, BN]);
            }
            let out: Tile<f16, { [BM, BN] }> = convert_tile(acc);
            y.store(out, out_idx);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GemmShape {
    device: DeviceLocation,
    n: usize,
    k: usize,
    dtype: ActivationDType,
    scales: Fp8WeightScaleLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ActivationDType {
    Bf16,
    F16,
}

impl TryFrom<DType> for ActivationDType {
    type Error = candle_core::Error;

    fn try_from(dtype: DType) -> Result<Self> {
        match dtype {
            DType::BF16 => Ok(Self::Bf16),
            DType::F16 => Ok(Self::F16),
            dtype => candle_core::bail!("cuTile W8A16 does not support {dtype:?} activations"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fp8W8A16Config {
    pub bm: i32,
    pub map_m: i32,
    pub map_n: i32,
    pub blocks_per_sm: i32,
    pub latency: i32,
    pub warps: i32,
    pub occupancy: i32,
    pub cluster: i32,
}

const POLICY_SMALL: Fp8W8A16Config = Fp8W8A16Config {
    bm: 16,
    map_m: 1,
    map_n: 8,
    blocks_per_sm: 2,
    latency: 0,
    warps: 0,
    occupancy: 0,
    cluster: 0,
};

const POLICY_LARGE: Fp8W8A16Config = Fp8W8A16Config {
    bm: 64,
    map_m: 4,
    map_n: 1,
    blocks_per_sm: 2,
    latency: 0,
    warps: 0,
    occupancy: 0,
    cluster: 0,
};

impl Fp8W8A16Config {
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

static TUNED: TunedTable<GemmShape, Fp8W8A16Config> = TunedTable::new();

fn policy(rows: usize) -> Fp8W8A16Config {
    if rows <= ROW_BREAKPOINTS[0] {
        POLICY_SMALL
    } else {
        POLICY_LARGE
    }
}

fn gemm_config(shape: GemmShape, rows: usize) -> Fp8W8A16Config {
    TUNED.get(shape, rows).unwrap_or_else(|| policy(rows))
}

fn gemm_space(bucket: Bucket) -> Space {
    let policy = policy(bucket.probe);
    Space::new()
        .joint(["bm"], [[16], [32], [64], [128]])
        .joint(["map_m", "map_n"], [[1, 8], [2, 4], [4, 1], [8, 1], [1, 1]])
        .axis("blocks_per_sm", [2, 1, 4])
        .axis("latency", [0, 2, 4])
        .axis("warps", [0, 4, 8])
        .axis("occupancy", [0, 4])
        .axis("cluster", [0, 2])
        .policy(policy.to_config())
}

fn gemm_buckets() -> Vec<Bucket> {
    buckets_from_breakpoints(&ROW_BREAKPOINTS, PREFILL_PROBE_ROWS, PREFILL_PROBE_ROWS)
}

pub struct Fp8W8A16Kernel;

pub(super) static FP8_W8A16: Fp8W8A16Kernel = Fp8W8A16Kernel;

#[derive(Clone)]
struct GemmWeights {
    weight: Tensor,
    scales: Tensor,
    key: GemmShape,
}

static SHAPES: OnceLock<Mutex<Vec<Vec<GemmWeights>>>> = OnceLock::new();

pub fn register_fp8_w8a16_shape(
    weight: &Tensor,
    weight_scales: &Tensor,
    scale_layout: Fp8WeightScaleLayout,
    activation_dtype: DType,
) {
    let Ok((n, k)) = weight.dims2() else {
        return;
    };
    let Ok(dtype) = ActivationDType::try_from(activation_dtype) else {
        return;
    };
    let entry = GemmWeights {
        weight: weight.clone(),
        scales: weight_scales.clone(),
        key: GemmShape {
            device: weight.device().location(),
            n,
            k,
            dtype,
            scales: scale_layout,
        },
    };
    let mut shapes = SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if let Some(sets) = shapes.iter_mut().find(|sets| {
        sets[0].key == entry.key && sets[0].weight.device().same_device(entry.weight.device())
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

pub fn fp8_w8a16_supported(
    dev: &CudaDevice,
    output_features: usize,
    input_features: usize,
    activation_dtype: DType,
) -> bool {
    jit_available(dev)
        && matches!(activation_dtype, DType::BF16 | DType::F16)
        && output_features > 0
        && input_features > 0
        && output_features.is_multiple_of(BLOCK_SIZE)
        && input_features.is_multiple_of(BLOCK_SIZE)
}

struct GemmOperands<'a> {
    activation: &'a Tensor,
    weight: &'a Tensor,
    weight_scales: &'a Tensor,
    scale_layout: Fp8WeightScaleLayout,
}

pub fn cutile_fp8_w8a16(
    activation: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    scale_layout: Fp8WeightScaleLayout,
) -> Result<Tensor> {
    let operands = GemmOperands {
        activation,
        weight,
        weight_scales,
        scale_layout,
    };
    let (n, k) = weight.dims2()?;
    let key = GemmShape {
        device: activation.device().location(),
        n,
        k,
        dtype: activation.dtype().try_into()?,
        scales: scale_layout,
    };
    let cfg = gemm_config(key, activation.dim(0)?);
    launch(&operands, cfg, false)
}

fn validate_scale_shape(
    scales: &Tensor,
    layout: Fp8WeightScaleLayout,
    n: usize,
    k: usize,
) -> Result<()> {
    let valid = match layout {
        Fp8WeightScaleLayout::Tensor => scales.elem_count() == 1,
        Fp8WeightScaleLayout::Channel => scales.elem_count() == n,
        Fp8WeightScaleLayout::Block([BLOCK_SIZE, BLOCK_SIZE]) => {
            scales.dims() == [n / BLOCK_SIZE, k / BLOCK_SIZE]
        }
        Fp8WeightScaleLayout::Block(_) => false,
    };
    if !valid {
        candle_core::bail!(
            "cuTile W8A16 scale shape {:?} does not match {layout:?} for weight [{n}, {k}]",
            scales.dims()
        )
    }
    Ok(())
}

fn launch(operands: &GemmOperands<'_>, cfg: Fp8W8A16Config, compile_only: bool) -> Result<Tensor> {
    let activation = operands.activation.contiguous()?;
    let weight = operands.weight.contiguous()?;
    let scales = operands.weight_scales.contiguous()?;
    let (rows, k) = activation.dims2()?;
    let (n, weight_k) = weight.dims2()?;
    if rows == 0
        || n == 0
        || k == 0
        || weight_k != k
        || !n.is_multiple_of(BLOCK_SIZE)
        || !k.is_multiple_of(BLOCK_SIZE)
    {
        candle_core::bail!("cuTile W8A16 got unsupported shape rows={rows} n={n} k={k}")
    }
    if !matches!(activation.dtype(), DType::BF16 | DType::F16)
        || weight.dtype() != DType::F8E4M3
        || scales.dtype() != DType::F32
    {
        candle_core::bail!("cuTile W8A16 needs A16 activations, E4M3 weights, and F32 scales")
    }
    if !activation.device().same_device(weight.device())
        || !activation.device().same_device(scales.device())
    {
        candle_core::bail!("cuTile W8A16 operands must be on the same device")
    }
    let Device::Cuda(dev) = activation.device() else {
        candle_core::bail!("cuTile W8A16 operands must be CUDA tensors")
    };
    validate_scale_shape(&scales, operands.scale_layout, n, k)?;
    let bm = usize::try_from(cfg.bm).unwrap_or(0);
    if bm == 0 || !BLOCK_SIZE.is_multiple_of(bm) {
        candle_core::bail!("cuTile W8A16 row tile {} must divide {BLOCK_SIZE}", cfg.bm)
    }
    if cfg.map_m <= 0 || cfg.map_n <= 0 {
        candle_core::bail!("cuTile W8A16 map dimensions must be positive")
    }
    let padded_rows = rows.div_ceil(bm) * bm;
    let padded_activation;
    let activation = if padded_rows == rows {
        &activation
    } else {
        padded_activation =
            Tensor::zeros((padded_rows, k), activation.dtype(), activation.device())?;
        padded_activation.slice_set(&activation, 0, 0)?;
        &padded_activation
    };

    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let (a_storage, a_layout) = activation.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let (s_storage, s_layout) = scales.storage_and_layout();
    let (Storage::Cuda(a_cuda), Storage::Cuda(w_cuda), Storage::Cuda(s_cuda)) =
        (&*a_storage, &*w_storage, &*s_storage)
    else {
        candle_core::bail!("cuTile W8A16 operands must be CUDA tensors")
    };
    let (w_addr, _w_guard) = slice_ptr_on_stream(
        w_cuda.as_cuda_slice::<F8E4M3>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (s_addr, _s_guard) = slice_ptr_on_stream(
        s_cuda.as_cuda_slice::<f32>()?,
        s_layout.start_offset(),
        &stream,
    );
    let dims = |r: usize, c: usize| (vec![r as i32, c as i32], vec![c as i32, 1]);
    let (shape, strides) = dims(n, k);
    let w = unsafe {
        cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            w_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        )
    };
    let post_scales = || unsafe {
        let stride = usize::from(operands.scale_layout == Fp8WeightScaleLayout::Channel);
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            s_addr as CUdeviceptr,
            ordinal,
            vec![n as i32],
            vec![stride as i32],
        )
    };
    let block_scales = || unsafe {
        let (shape, strides) = dims(n / BLOCK_SIZE, k / BLOCK_SIZE);
        cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            s_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        )
    };
    let tiles = (padded_rows / bm) * (n / BLOCK_SIZE);
    let blocks_per_sm = usize::try_from(cfg.blocks_per_sm).unwrap_or(1).max(1);
    let tile_blocks = (blocks_per_sm * device_multiprocessor_count(dev)).clamp(1, tiles) as u32;
    let generics = vec![
        cfg.bm.to_string(),
        BLOCK_SIZE.to_string(),
        BLOCK_SIZE.to_string(),
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
                catch_cutile_panic("W8A16 compile", || {
                    launcher.compile_on(&cutile_stream).map_err(|error| {
                        candle_core::Error::Msg(format!(
                            "cuTile {} compile failed: {error:?}",
                            $label
                        ))
                    })
                })?;
            } else {
                catch_cutile_panic("W8A16 launch", || unsafe {
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

    let output = match activation.dtype() {
        DType::BF16 => {
            let (a_addr, _a_guard) = slice_ptr_on_stream(
                a_cuda.as_cuda_slice::<bf16>()?,
                a_layout.start_offset(),
                &stream,
            );
            let mut output = unsafe { dev.alloc::<bf16>(padded_rows * n)? };
            let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let (shape, strides) = dims(padded_rows, k);
            let x = unsafe {
                cutile::tensor::Tensor::<bf16>::borrow_raw_parts(
                    a_addr as CUdeviceptr,
                    ordinal,
                    shape,
                    strides,
                )
            };
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
                .partition([bm, BLOCK_SIZE])
                .map([cfg.map_m as usize, cfg.map_n as usize], tile_blocks);
            match operands.scale_layout {
                Fp8WeightScaleLayout::Tensor | Fp8WeightScaleLayout::Channel => run!(
                    kernels::fp8_w8a16_post_bf16(
                        mapped,
                        Arc::new(x),
                        Arc::new(w),
                        Arc::new(post_scales())
                    ),
                    "W8A16 BF16 post-scale GEMM"
                ),
                Fp8WeightScaleLayout::Block([BLOCK_SIZE, BLOCK_SIZE]) => run!(
                    kernels::fp8_w8a16_block_bf16(
                        mapped,
                        Arc::new(x),
                        Arc::new(w),
                        Arc::new(block_scales())
                    ),
                    "W8A16 BF16 block-scale GEMM"
                ),
                Fp8WeightScaleLayout::Block(_) => unreachable!(),
            }
            drop(out_guard);
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from_dims(&[padded_rows, n]),
            ))
        }
        DType::F16 => {
            let (a_addr, _a_guard) = slice_ptr_on_stream(
                a_cuda.as_cuda_slice::<f16>()?,
                a_layout.start_offset(),
                &stream,
            );
            let mut output = unsafe { dev.alloc::<f16>(padded_rows * n)? };
            let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let (shape, strides) = dims(padded_rows, k);
            let x = unsafe {
                cutile::tensor::Tensor::<f16>::borrow_raw_parts(
                    a_addr as CUdeviceptr,
                    ordinal,
                    shape,
                    strides,
                )
            };
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
                .partition([bm, BLOCK_SIZE])
                .map([cfg.map_m as usize, cfg.map_n as usize], tile_blocks);
            match operands.scale_layout {
                Fp8WeightScaleLayout::Tensor | Fp8WeightScaleLayout::Channel => run!(
                    kernels::fp8_w8a16_post_f16(
                        mapped,
                        Arc::new(x),
                        Arc::new(w),
                        Arc::new(post_scales())
                    ),
                    "W8A16 F16 post-scale GEMM"
                ),
                Fp8WeightScaleLayout::Block([BLOCK_SIZE, BLOCK_SIZE]) => run!(
                    kernels::fp8_w8a16_block_f16(
                        mapped,
                        Arc::new(x),
                        Arc::new(w),
                        Arc::new(block_scales())
                    ),
                    "W8A16 F16 block-scale GEMM"
                ),
                Fp8WeightScaleLayout::Block(_) => unreachable!(),
            }
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
    operands: HashMap<usize, Tensor>,
}

impl GemmTuner {
    fn new(dev: &CudaDevice, sets: &[GemmWeights]) -> Self {
        Self {
            dev: dev.clone(),
            sets: sets.to_vec(),
            operands: HashMap::new(),
        }
    }

    fn prepare(&mut self, rows: usize, cfg: Fp8W8A16Config) -> Result<Prepared> {
        let dev = self.dev.clone();
        let sets = self.sets.clone();
        let key = sets[0].key;
        if let std::collections::hash_map::Entry::Vacant(slot) = self.operands.entry(rows) {
            let device = candle_core::Device::Cuda(dev.clone());
            let activation =
                Tensor::rand(-1f32, 1f32, (rows, key.k), &device)?.to_dtype(match key.dtype {
                    ActivationDType::Bf16 => DType::BF16,
                    ActivationDType::F16 => DType::F16,
                })?;
            slot.insert(activation);
        }
        let activation = self.operands[&rows].clone();
        let launch = move |weights: &GemmWeights, compile_only: bool| -> Result<Tensor> {
            let operands = GemmOperands {
                activation: &activation,
                weight: &weights.weight,
                weight_scales: &weights.scales,
                scale_layout: key.scales,
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

impl CutileKernel for Fp8W8A16Kernel {
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
                shape: format!("n{}_k{}_a{:?}_s{:?}", key.n, key.k, key.dtype, key.scales),
                buckets: &buckets,
                space: &gemm_space,
            };
            let mut tuner = GemmTuner::new(dev, sets);
            let tuned = tune(dev, mode, &request, |rows, candidate| {
                let cfg = Fp8W8A16Config::from_config(candidate)
                    .ok_or_else(|| candle_core::Error::msg("config outside the space"))?;
                tuner.prepare(rows, cfg)
            });
            TUNED.set(key, &tuned, Fp8W8A16Config::from_config);
        }
        tracing::info!("Warming {} cuTile W8A16 GEMM kernels.", shapes.len());
        for sets in &shapes {
            let key = sets[0].key;
            let Device::Cuda(dev) = sets[0].weight.device() else {
                continue;
            };
            let device = Device::Cuda(dev.clone());
            for bucket in &buckets {
                let result = (|| -> Result<()> {
                    let dtype = match key.dtype {
                        ActivationDType::Bf16 => DType::BF16,
                        ActivationDType::F16 => DType::F16,
                    };
                    let activation = Tensor::zeros((bucket.probe, key.k), dtype, &device)?;
                    let operands = GemmOperands {
                        activation: &activation,
                        weight: &sets[0].weight,
                        weight_scales: &sets[0].scales,
                        scale_layout: key.scales,
                    };
                    launch(&operands, gemm_config(key, bucket.probe), true)?;
                    Ok(())
                })();
                if let Err(error) = result {
                    tracing::warn!(
                        "cuTile W8A16 warmup failed (n={} k={} rows={}): {error}",
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
    use super::{cutile_fp8_w8a16, validate_scale_shape};
    use crate::Fp8WeightScaleLayout;
    use candle_core::{DType, Device, Result, Tensor};
    use float8::F8E4M3;

    type ScaleAt = Box<dyn Fn(usize, usize) -> f32>;

    #[test]
    fn scale_shapes_are_validated_by_layout() -> Result<()> {
        let device = Device::Cpu;
        let scalar = Tensor::zeros((), DType::F32, &device)?;
        let channel = Tensor::zeros(256, DType::F32, &device)?;
        let block = Tensor::zeros((2, 4), DType::F32, &device)?;
        validate_scale_shape(&scalar, Fp8WeightScaleLayout::Tensor, 256, 512)?;
        validate_scale_shape(&channel, Fp8WeightScaleLayout::Channel, 256, 512)?;
        validate_scale_shape(&block, Fp8WeightScaleLayout::Block([128, 128]), 256, 512)?;
        assert!(
            validate_scale_shape(&channel, Fp8WeightScaleLayout::Block([128, 128]), 256, 512)
                .is_err()
        );
        Ok(())
    }

    fn value(index: usize, modulus: usize, offset: i32) -> f32 {
        ((index * 17 + 11) % modulus) as f32 + offset as f32
    }

    fn run_correctness(dtype: DType, scale_layout: Fp8WeightScaleLayout) -> Result<()> {
        const ROWS: usize = 7;
        const N: usize = 256;
        const K: usize = 256;

        let device = Device::new_cuda(0)?;
        let Device::Cuda(_cuda) = &device else {
            unreachable!()
        };
        let x_values = (0..ROWS * K)
            .map(|index| value(index, 9, -4) * 0.25)
            .collect::<Vec<_>>();
        let q_values = (0..N * K)
            .map(|index| F8E4M3::from_f32(value(index, 7, -3)))
            .collect::<Vec<_>>();
        let (scales, scale_at): (Vec<f32>, ScaleAt) = match scale_layout {
            Fp8WeightScaleLayout::Tensor => (vec![0.125], Box::new(|_, _| 0.125)),
            Fp8WeightScaleLayout::Channel => {
                let scales = (0..N)
                    .map(|row| 0.0625 * (row % 4 + 1) as f32)
                    .collect::<Vec<_>>();
                let copy = scales.clone();
                (scales, Box::new(move |row, _| copy[row]))
            }
            Fp8WeightScaleLayout::Block([128, 128]) => (
                vec![0.125, 0.25, 0.375, 0.5],
                Box::new(|row, col| [0.125, 0.25, 0.375, 0.5][(row / 128) * 2 + col / 128]),
            ),
            Fp8WeightScaleLayout::Block(_) => unreachable!(),
        };
        let scale_shape = match scale_layout {
            Fp8WeightScaleLayout::Tensor => candle_core::Shape::from_dims(&[]),
            Fp8WeightScaleLayout::Channel => candle_core::Shape::from_dims(&[N]),
            Fp8WeightScaleLayout::Block(_) => candle_core::Shape::from_dims(&[2, 2]),
        };
        let x = Tensor::from_vec(x_values.clone(), (ROWS, K), &device)?.to_dtype(dtype)?;
        let weight = Tensor::from_vec(q_values.clone(), (N, K), &device)?;
        let scales = Tensor::from_vec(scales, scale_shape, &device)?;
        let output = cutile_fp8_w8a16(&x, &weight, &scales, scale_layout)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        let mut max_error = 0f32;
        let mut max_reference = 0f32;
        for row in 0..ROWS {
            for out in 0..N {
                let reference = (0..K)
                    .map(|col| {
                        x_values[row * K + col]
                            * q_values[out * K + col].to_f32()
                            * scale_at(out, col)
                    })
                    .sum::<f32>();
                max_error = max_error.max((output[row][out] - reference).abs());
                max_reference = max_reference.max(reference.abs());
            }
        }
        assert!(
            max_error <= max_reference * 0.02 + 0.25,
            "dtype={dtype:?} scales={scale_layout:?}: error {max_error}, reference {max_reference}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn retained_fp8_w8a16_matches_reference() -> Result<()> {
        for dtype in [DType::BF16, DType::F16] {
            for layout in [
                Fp8WeightScaleLayout::Tensor,
                Fp8WeightScaleLayout::Channel,
                Fp8WeightScaleLayout::Block([128, 128]),
            ] {
                run_correctness(dtype, layout)?;
            }
        }
        Ok(())
    }
}
