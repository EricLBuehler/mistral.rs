//! Persistent blockwise FP8 W8A8 GEMM: 128x128 weight scales, 1x128 activation scales, BF16 output.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use candle_core::{CudaDevice, CudaStorage, DType, Result, Shape, Storage, Tensor};
use cutile::core::f8e4m3fn;
use cutile::cuda_async::device_operation::DeviceOp;
use cutile::cuda_core::sys::CUdeviceptr;
use cutile::tensor::IntoPartition;
use cutile::tile_kernel::{CompileOptions, TileKernel};
use float8::F8E4M3;
use half::bf16;

use super::tune::{
    buckets_from_breakpoints, config, cutile_error, tune, Bucket, Prepared, Space, TuneMode,
    TuneRequest, TunedTable, TUNE_WEIGHT_SETS,
};
use super::warmup::CutileKernel;
use super::{catch_cutile_panic, context, device_multiprocessor_count, jit_available};
use crate::blockwise_fp8::mma::quantize_activation_padded;
use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

pub const FP8_GEMM_BLOCK_ROWS: usize = 128;
const BLOCK_COLS: usize = 128;
const GROUP_SIZE: usize = 128;
const TUNE_KERNEL: &str = "fp8_gemm";
/// Row counts where the launch policy may change; the GEMV handles anything below the first.
const ROW_BREAKPOINTS: [usize; 2] = [128, 512];
const PREFILL_PROBE_ROWS: usize = 4096;

// Bounded partitions are deprecated in cutile 0.3 but plain indexed loads compile to checked loads
// that run 3x slower on sm_121.
#[cutile::module]
mod kernels {
    #![allow(deprecated)]
    use cutile::core::*;
    use cutile::cutile_compiler;

    // BN stays at one weight-scale column and BK at one scale group, so every k step is scaled by a
    // single (row, group) product before it joins the running accumulator.
    #[allow(deprecated)]
    #[cutile::entry(
        unchecked_accesses = false,
        optimization_hints = (
            sm_120 = (num_cta_in_cga = 2, occupancy = 2,),
            sm_121 = (num_cta_in_cga = 2, occupancy = 2,),
        )
    )]
    fn fp8_blockwise_gemm<
        const BM: i32,
        const BN: i32,
        const BK: i32,
        const MAP_SHAPE: [i32; 2],
        const LATENCY: i32, // operand-load pipelining hint, 0 = compiler default
    >(
        mut y: MappedPartitionMut<bf16, { [BM, BN] }, MAP_SHAPE>,
        x: &Tensor<f8e4m3fn, { [-1, -1] }>,
        w: &Tensor<f8e4m3fn, { [-1, -1] }>,
        xs: &Tensor<f32, { [-1, -1] }>,
        ws: &Tensor<f32, { [-1, -1] }>,
    ) {
        let m = num_tiles(&y, 0);
        let n = num_tiles(&y, 1);
        let k = Dim::new(x.shape()[1] / BK);
        let px = x.partition(const_shape![BM, BK]).with_bounds((m, k));
        let pw = w.partition(const_shape![BN, BK]).with_bounds((n, k));
        let pxs = xs.partition(const_shape![1, BM]).with_bounds((k, m));
        let pws = ws.partition(const_shape![1, 1]).with_bounds((n, k));
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };
        for out_idx in y.iter_indices() {
            let (bid_m, bid_n) = out_idx.components();
            let mut acc: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            for kg in k {
                // a zero latency hint is rejected by tileiras, so 0 means the plain load
                let xt: Tile<f8e4m3fn, { [BM, BK] }> = if LATENCY > 0 {
                    px.load_pipelined::<LATENCY>(coord((bid_m, kg)))
                } else {
                    px.load(coord((bid_m, kg)))
                };
                let wt: Tile<f8e4m3fn, { [BN, BK] }> = if LATENCY > 0 {
                    pw.load_pipelined::<LATENCY>(coord((bid_n, kg)))
                } else {
                    pw.load(coord((bid_n, kg)))
                };
                let wtt: Tile<f8e4m3fn, { [BK, BN] }> = permute(wt, transpose);
                let zero: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
                let part: Tile<f32, { [BM, BN] }> = mmaf(xt, wtt, zero);
                let sx: Tile<f32, { [1, BM] }> = pxs.load(coord((kg, bid_m)));
                let sxc: Tile<f32, { [BM, 1] }> = sx.reshape(const_shape![BM, 1]);
                let sw: Tile<f32, { [1, 1] }> = pws.load(coord((bid_n, kg)));
                let scale: Tile<f32, { [BM, BN] }> =
                    sxc.broadcast(const_shape![BM, BN]) * sw.broadcast(const_shape![BM, BN]);
                acc = acc + part * scale;
            }
            let out: Tile<bf16, { [BM, BN] }> = convert_tile(acc);
            y.store(out, out_idx);
        }
    }
}

/// Launch config: the row tile, the swizzle map over output tiles, persistent tile blocks per SM,
/// and the knobs the autotuner sweeps. The column tile is pinned to one weight-scale column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fp8GemmConfig {
    pub bm: i32,
    pub map_m: i32,
    pub map_n: i32,
    pub blocks_per_sm: i32,
    pub latency: i32,
    pub warps: i32,
    pub occupancy: i32,
    pub cluster: i32,
}

/// Measured on GB10: 128-row tiles with an 8x1 swizzle and two persistent blocks per SM.
const POLICY: Fp8GemmConfig = Fp8GemmConfig {
    bm: 128,
    map_m: 8,
    map_n: 1,
    blocks_per_sm: 2,
    latency: 0,
    warps: 0,
    occupancy: 0,
    cluster: 0,
};

impl Fp8GemmConfig {
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
        let int = |key: &str| config.int(key).and_then(|v| i32::try_from(v).ok());
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

/// Output features and input features of a registered weight.
type GemmShape = (usize, usize);

static TUNED: TunedTable<GemmShape, Fp8GemmConfig> = TunedTable::new();

fn gemm_config(shape: GemmShape, rows: usize) -> Fp8GemmConfig {
    TUNED.get(shape, rows).unwrap_or(POLICY)
}

fn gemm_space(bucket: Bucket) -> Space {
    let _ = bucket;
    Space::new()
        .joint(["bm"], [[128], [64]])
        .joint(["map_m", "map_n"], [[8, 1], [4, 1], [1, 1]])
        .axis("blocks_per_sm", [2, 1, 4])
        .axis("latency", [0, 2, 4])
        .axis("warps", [0, 4, 8])
        .axis("occupancy", [0, 4])
        .axis("cluster", [0, 2])
        .policy(POLICY.to_config())
}

fn gemm_buckets() -> Vec<Bucket> {
    buckets_from_breakpoints(&ROW_BREAKPOINTS, PREFILL_PROBE_ROWS, PREFILL_PROBE_ROWS)
}

pub struct Fp8GemmKernel;

pub(super) static FP8_GEMM: Fp8GemmKernel = Fp8GemmKernel;

#[derive(Clone)]
struct GemmWeights {
    weight: Tensor,
    scales: Tensor,
}

impl GemmWeights {
    fn shape(&self) -> Result<GemmShape> {
        self.weight.dims2()
    }
}

static SHAPES: OnceLock<Mutex<Vec<Vec<GemmWeights>>>> = OnceLock::new();

/// Register a linear layer's FP8 weight so warmup tunes and compiles the kernel keys the forward
/// will launch. Up to `TUNE_WEIGHT_SETS` layers of the same shape are kept as weight handles.
pub fn register_fp8_gemm_shape(weight: &Tensor, weight_scales: &Tensor) {
    let entry = GemmWeights {
        weight: weight.clone(),
        scales: weight_scales.clone(),
    };
    let Ok(shape) = entry.shape() else {
        return;
    };
    let mut shapes = SHAPES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap();
    if let Some(sets) = shapes
        .iter_mut()
        .find(|sets| sets[0].shape().ok() == Some(shape))
    {
        if sets.len() < TUNE_WEIGHT_SETS {
            sets.push(entry);
        }
        return;
    }
    shapes.push(vec![entry]);
}

pub fn fp8_gemm_supported(dev: &CudaDevice, output_features: usize, input_features: usize) -> bool {
    jit_available(dev)
        && output_features.is_multiple_of(BLOCK_COLS)
        && input_features.is_multiple_of(GROUP_SIZE)
}

struct GemmOperands<'a> {
    activation: &'a Tensor,
    activation_scales: &'a Tensor,
    weight: &'a Tensor,
    weight_scales: &'a Tensor,
}

/// `activation` [rows, K] E4M3 whose storage holds `padded` rows, `activation_scales` [K/128, padded]
/// F32 with `padded` a multiple of [`FP8_GEMM_BLOCK_ROWS`] and rows <= padded, `weight` [N, K] E4M3,
/// `weight_scales` [N/128, K/128] F32 -> [padded, N] BF16; rows past `rows` are unspecified.
pub fn cutile_fp8_gemm(
    activation: &Tensor,
    activation_scales: &Tensor,
    weight: &Tensor,
    weight_scales: &Tensor,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let operands = GemmOperands {
        activation,
        activation_scales,
        weight,
        weight_scales,
    };
    let cfg = gemm_config(weight.dims2()?, activation.dim(0)?);
    launch(&operands, cfg, dev, false)
}

/// Rows an activation view's storage can supply past its start offset.
fn activation_storage_rows(activation: &Tensor) -> Result<usize> {
    let (_, k) = activation.dims2()?;
    let (storage, layout) = activation.storage_and_layout();
    let Storage::Cuda(cuda) = &*storage else {
        candle_core::bail!("cuTile FP8 GEMM operands must be CUDA tensors")
    };
    Ok((cuda.as_cuda_slice::<F8E4M3>()?.len() - layout.start_offset()) / k)
}

fn launch(
    operands: &GemmOperands<'_>,
    cfg: Fp8GemmConfig,
    dev: &CudaDevice,
    compile_only: bool,
) -> Result<Tensor> {
    let (logical_rows, k) = operands.activation.dims2()?;
    let (groups_dim, rows) = operands.activation_scales.dims2()?;
    let (n, weight_k) = operands.weight.dims2()?;
    let groups = k / GROUP_SIZE;
    if weight_k != k
        || !rows.is_multiple_of(FP8_GEMM_BLOCK_ROWS)
        || !n.is_multiple_of(BLOCK_COLS)
        || !k.is_multiple_of(GROUP_SIZE)
        || logical_rows == 0
        || logical_rows > rows
        || !operands.activation.is_contiguous()
    {
        candle_core::bail!(
            "cuTile FP8 GEMM got unsupported shape rows={logical_rows} padded={rows} n={n} k={k}"
        )
    }
    if operands.activation.dtype() != DType::F8E4M3
        || operands.weight.dtype() != DType::F8E4M3
        || operands.activation_scales.dtype() != DType::F32
        || operands.weight_scales.dtype() != DType::F32
    {
        candle_core::bail!("cuTile FP8 GEMM needs E4M3 operands with F32 scales")
    }
    if groups_dim != groups || operands.weight_scales.dims2()? != (n / BLOCK_COLS, groups) {
        candle_core::bail!("cuTile FP8 GEMM scale shapes do not match the operands")
    }
    let bm = usize::try_from(cfg.bm).unwrap_or(0);
    if bm == 0 || !FP8_GEMM_BLOCK_ROWS.is_multiple_of(bm) {
        candle_core::bail!(
            "cuTile FP8 GEMM row tile {} must divide {FP8_GEMM_BLOCK_ROWS}",
            cfg.bm
        )
    }
    let stream = dev.cuda_stream();
    let ordinal = stream.context().ordinal();
    let mut out = unsafe { dev.alloc::<bf16>(rows * n)? };
    // producers pad their FP8 rows to the scale stride; anything else gets copied into a padded buffer
    let padded_copy;
    let activation = if activation_storage_rows(operands.activation)? >= rows {
        operands.activation
    } else {
        padded_copy = Tensor::zeros((rows, k), DType::F8E4M3, operands.activation.device())?;
        padded_copy.slice_set(operands.activation, 0, 0)?;
        &padded_copy
    };
    let (a_storage, a_layout) = activation.storage_and_layout();
    let (as_storage, as_layout) = operands.activation_scales.storage_and_layout();
    let (w_storage, w_layout) = operands.weight.storage_and_layout();
    let (ws_storage, ws_layout) = operands.weight_scales.storage_and_layout();
    let (
        Storage::Cuda(a_cuda),
        Storage::Cuda(as_cuda),
        Storage::Cuda(w_cuda),
        Storage::Cuda(ws_cuda),
    ) = (&*a_storage, &*as_storage, &*w_storage, &*ws_storage)
    else {
        candle_core::bail!("cuTile FP8 GEMM operands must be CUDA tensors")
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
    let (out_addr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
    let dims = |r: usize, c: usize| (vec![r as i32, c as i32], vec![c as i32, 1]);
    // SAFETY: every borrowed buffer is a live candle allocation that outlives the launch on the
    // same stream, and the output is written only by this kernel.
    let (y, x, w, xs, ws) = unsafe {
        let (shape, strides) = dims(rows, n);
        let y = cutile::tensor::Tensor::<bf16>::borrow_raw_parts(
            out_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        );
        let (shape, strides) = dims(rows, k);
        let x = cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            a_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        );
        let (shape, strides) = dims(n, k);
        let w = cutile::tensor::Tensor::<f8e4m3fn>::borrow_raw_parts(
            w_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        );
        let (shape, strides) = dims(groups, rows);
        let xs = cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            as_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        );
        let (shape, strides) = dims(n / BLOCK_COLS, groups);
        let ws = cutile::tensor::Tensor::<f32>::borrow_raw_parts(
            ws_addr as CUdeviceptr,
            ordinal,
            shape,
            strides,
        );
        (y, x, w, xs, ws)
    };
    let tiles = (rows / bm) * (n / BLOCK_COLS);
    let blocks_per_sm = usize::try_from(cfg.blocks_per_sm).unwrap_or(1).max(1);
    let tile_blocks = (blocks_per_sm * device_multiprocessor_count(dev)).clamp(1, tiles) as u32;
    let mapped = y
        .partition([bm, BLOCK_COLS])
        .map([cfg.map_m as usize, cfg.map_n as usize], tile_blocks);
    let generics = vec![
        cfg.bm.to_string(),
        BLOCK_COLS.to_string(),
        GROUP_SIZE.to_string(),
        cfg.map_m.to_string(),
        cfg.map_n.to_string(),
        cfg.latency.to_string(),
    ];
    let cutile_stream = context::stream(dev);
    let launcher =
        kernels::fp8_blockwise_gemm(mapped, Arc::new(x), Arc::new(w), Arc::new(xs), Arc::new(ws))
            .generics(generics)
            .compile_options(cfg.compile_options());
    if compile_only {
        catch_cutile_panic("FP8 GEMM compile", || {
            launcher
                .compile_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile fp8 gemm compile: {e:?}")))
        })?;
    } else {
        catch_cutile_panic("FP8 GEMM launch", || unsafe {
            launcher
                .async_on(&cutile_stream)
                .map_err(|e| candle_core::Error::Msg(format!("cutile fp8 gemm launch: {e:?}")))
        })?;
    }
    drop(out_guard);
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from_dims(&[rows, n]),
    )))
}

/// Prepares timed launch sets for the tuner, rotating through the registered weights of a shape.
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

    fn prepare(&mut self, rows: usize, cfg: Fp8GemmConfig) -> Result<Prepared> {
        let dev = self.dev.clone();
        let sets = self.sets.clone();
        let (_, k) = sets[0].shape()?;
        if let std::collections::hash_map::Entry::Vacant(slot) = self.operands.entry(rows) {
            let device = candle_core::Device::Cuda(dev.clone());
            let padded = rows.div_ceil(FP8_GEMM_BLOCK_ROWS) * FP8_GEMM_BLOCK_ROWS;
            let x = Tensor::rand(-1f32, 1f32, (rows, k), &device)?.to_dtype(DType::BF16)?;
            slot.insert(quantize_activation_padded(&x, padded)?);
        }
        let (activation, activation_scales) = self.operands[&rows].clone();
        let launch = move |w: &GemmWeights, compile_only: bool| -> Result<Tensor> {
            let operands = GemmOperands {
                activation: &activation,
                activation_scales: &activation_scales,
                weight: &w.weight,
                weight_scales: &w.scales,
            };
            launch(&operands, cfg, &dev, compile_only)
        };
        launch(&sets[0], true)?;
        let sample = launch(&sets[0], false)?;
        let mut next = 0usize;
        let run = Box::new(move |_: &Arc<cutile::cuda_core::Stream>| {
            let w = &sets[next % sets.len()];
            next += 1;
            launch(w, false).map(|_| ()).map_err(cutile_error)
        });
        Ok(Prepared { run, sample })
    }
}

impl CutileKernel for Fp8GemmKernel {
    fn warm(&self, dev: &CudaDevice) -> Result<()> {
        let shapes: Vec<Vec<GemmWeights>> = SHAPES
            .get_or_init(|| Mutex::new(Vec::new()))
            .lock()
            .unwrap()
            .clone();
        if shapes.is_empty() {
            return Ok(());
        }
        let mode = TuneMode::from_env();
        let buckets = gemm_buckets();
        for sets in &shapes {
            let shape = sets[0].shape()?;
            let request = TuneRequest {
                kernel: TUNE_KERNEL,
                source_hash: kernels::_SOURCE_HASH,
                shape: format!("n{}_k{}", shape.0, shape.1),
                buckets: &buckets,
                space: &gemm_space,
            };
            let mut tuner = GemmTuner::new(dev, sets);
            let tuned = tune(dev, mode, &request, |rows, candidate| {
                let cfg = Fp8GemmConfig::from_config(candidate)
                    .ok_or_else(|| candle_core::Error::Msg("config outside the space".into()))?;
                tuner.prepare(rows, cfg)
            });
            TUNED.set(shape, &tuned, Fp8GemmConfig::from_config);
        }
        tracing::info!("Warming {} cuTile FP8 GEMM kernels.", shapes.len());
        let device = candle_core::Device::Cuda(dev.clone());
        for sets in &shapes {
            let (n, k) = sets[0].shape()?;
            for bucket in &buckets {
                let rows = bucket.probe.div_ceil(FP8_GEMM_BLOCK_ROWS) * FP8_GEMM_BLOCK_ROWS;
                let activation = Tensor::zeros((rows, k), DType::F8E4M3, &device)?;
                let activation_scales = Tensor::zeros((k / GROUP_SIZE, rows), DType::F32, &device)?;
                let operands = GemmOperands {
                    activation: &activation,
                    activation_scales: &activation_scales,
                    weight: &sets[0].weight,
                    weight_scales: &sets[0].scales,
                };
                let cfg = gemm_config((n, k), bucket.probe);
                if let Err(err) = launch(&operands, cfg, dev, true) {
                    tracing::warn!(
                        "cuTile FP8 GEMM warmup failed (n={n} k={k} rows={rows}): {err}"
                    );
                }
            }
        }
        Ok(())
    }
}

#[cfg(all(test, has_blockwise_fp8_kernels))]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::{cutile_fp8_gemm, Fp8GemmConfig, FP8_GEMM_BLOCK_ROWS, POLICY, TUNED};
    use crate::blockwise_fp8::{mma, ops};
    use crate::cutile::tune::{Bucket, Source, Tuned};

    const GROUP_SIZE: usize = 128;

    fn patterned(len: usize, seed: usize, amplitude: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|index| ((index * 7919 + seed * 104729) % 2001) as f32 / 1000.0 - 1.0)
            .map(|value| value * amplitude + offset)
            .collect()
    }

    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_fp8_gemm_reads_row_views_over_padded_storage() -> Result<()> {
        const N: usize = 256;
        const K: usize = 512;
        const ROWS: usize = 200;
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let weight =
            Tensor::from_vec(patterned(N * K, 5, 2.0, 0.1), (N, K), &dev)?.to_dtype(DType::BF16)?;
        let (weight, weight_scales) =
            ops::fp8_blockwise_quantize(&weight, vec![GROUP_SIZE, GROUP_SIZE])?;
        let x = Tensor::from_vec(patterned(ROWS * K, 9, 3.0, -0.2), (ROWS, K), &dev)?
            .to_dtype(DType::BF16)?;
        let padded_rows = ROWS.div_ceil(FP8_GEMM_BLOCK_ROWS) * FP8_GEMM_BLOCK_ROWS;
        let (quantized, scales) = mma::quantize_activation_padded(&x, padded_rows)?;
        let full = cutile_fp8_gemm(&quantized, &scales, &weight, &weight_scales, cuda)?;
        let view = quantized.narrow(0, 0, ROWS)?;
        let viewed = cutile_fp8_gemm(&view, &scales, &weight, &weight_scales, cuda)?;
        assert_eq!(full.dims(), &[padded_rows, N]);
        assert_eq!(viewed.dims(), &[padded_rows, N]);
        let difference = full
            .narrow(0, 0, ROWS)?
            .to_dtype(DType::F32)?
            .sub(&viewed.narrow(0, 0, ROWS)?.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(difference, 0.0);
        // unpadded storage takes the copy fallback and must agree too
        let (unpadded, _) = mma::quantize_activation_padded(&x, ROWS)?;
        let copied = cutile_fp8_gemm(&unpadded, &scales, &weight, &weight_scales, cuda)?;
        let difference = full
            .narrow(0, 0, ROWS)?
            .to_dtype(DType::F32)?
            .sub(&copied.narrow(0, 0, ROWS)?.to_dtype(DType::F32)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(difference, 0.0);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device with cuTile support"]
    fn cutile_fp8_gemm_matches_dequantized_reference() -> Result<()> {
        const N: usize = 256;
        const K: usize = 1024;
        let dev = Device::new_cuda(0)?;
        let Device::Cuda(cuda) = &dev else {
            unreachable!()
        };
        let weight =
            Tensor::from_vec(patterned(N * K, 3, 2.0, 0.1), (N, K), &dev)?.to_dtype(DType::BF16)?;
        let (weight, weight_scales) =
            ops::fp8_blockwise_quantize(&weight, vec![GROUP_SIZE, GROUP_SIZE])?;
        let weight_ref = ops::fp8_blockwise_dequantize(
            &weight,
            &weight_scales,
            vec![GROUP_SIZE, GROUP_SIZE],
            DType::F32,
        )?
        .to_device(&Device::Cpu)?;
        // the second pass forces a tuned config through the table, as the tuner would
        let forced = Fp8GemmConfig {
            bm: 64,
            map_m: 4,
            blocks_per_sm: 1,
            latency: 2,
            ..POLICY
        };
        for (pass, rows) in [(0, 33usize), (0, 128), (0, 300), (1, 300)] {
            TUNED.set(
                (N, K),
                &[Tuned {
                    bucket: Bucket {
                        upper: usize::MAX,
                        probe: rows,
                    },
                    config: if pass == 1 { forced } else { POLICY }.to_config(),
                    source: Source::Measured,
                    ms: 0.0,
                    policy_ms: 0.0,
                }],
                Fp8GemmConfig::from_config,
            );
            let padded_rows = rows.div_ceil(FP8_GEMM_BLOCK_ROWS) * FP8_GEMM_BLOCK_ROWS;
            let x = Tensor::from_vec(patterned(rows * K, rows, 3.0, -0.2), (rows, K), &dev)?
                .to_dtype(DType::BF16)?;
            let (quantized, scales) = mma::quantize_activation_padded(&x, padded_rows)?;
            let output = cutile_fp8_gemm(&quantized, &scales, &weight, &weight_scales, cuda)?
                .narrow(0, 0, rows)?
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?;
            let cpu = Device::Cpu;
            let values = quantized
                .narrow(0, 0, rows)?
                .to_device(&cpu)?
                .to_dtype(DType::F32)?;
            let row_scales = scales
                .to_device(&cpu)?
                .t()?
                .narrow(0, 0, rows)?
                .contiguous()?;
            let dequantized = values
                .reshape((rows, K / GROUP_SIZE, GROUP_SIZE))?
                .broadcast_mul(&row_scales.reshape((rows, K / GROUP_SIZE, 1))?)?
                .reshape((rows, K))?;
            let reference = dequantized.matmul(&weight_ref.t()?)?;
            let max_error = output
                .sub(&reference)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let max_reference = reference.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(
                max_error <= 1.0e-2 * max_reference,
                "rows={rows} pass={pass}: max error {max_error} vs reference {max_reference}"
            );
        }
        TUNED.set((N, K), &[], Fp8GemmConfig::from_config);
        Ok(())
    }
}
