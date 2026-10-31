//! Autotuning for the dense and routed NVFP4 cuTile matmuls on top of [`super::tune`].

use std::collections::HashMap;

use candle_core::{CudaDevice, DType, Result, Tensor};

use super::nvfp4::{Nvfp4GemmArgs, Nvfp4Weights};
use super::nvfp4_matmul::{
    dense_policy, launch_dense, launch_gather_prepared, matmul_device, prepare_gather_with,
    quantize, routed_geometry, DenseConfig, DenseOperands, GatherPlan, MatmulDevice,
    RoutedGeometry, KERNEL_SOURCE_HASH, MAX_ROUTED_ROWS, ROUTED_LARGE_PREFILL_MIN_TOKENS,
    ROUTED_PREFILL_MIN_TOKENS,
};
use super::tune::{
    bucket_warm_rows, buckets_from_breakpoints, probe_input, random_activation, tunable, Bucket,
    Prepared, Space, Tunable, TuneMode, TuneRequest, TuneRouting, Tuned, TunedTable,
};
use crate::cuda_device::cuda_device_info;

const DENSE_KERNEL: &str = "nvfp4_dense";
const ROUTED_KERNEL: &str = "nvfp4_routed";
const BLOCK_SIZE: usize = 16;
/// Row counts where `dense_geometry` changes tiles, plus a split between short and long prefill.
const DENSE_BREAKPOINTS: [usize; 4] = [16, 32, 127, 511];
const DENSE_PREFILL_PROBE: usize = 2048;
const DENSE_DECODE_MAX_ROWS: usize = 32;
const MIN_TILE_ROWS: usize = 16;
const ROUTED_SHORT_PROBE: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct DenseKey {
    ordinal: usize,
    n: usize,
    k: usize,
    a4: bool,
    dtype: DType,
}

impl DenseKey {
    pub(super) fn new(dev: &CudaDevice, n: usize, k: usize, a4: bool, dtype: DType) -> Self {
        Self {
            ordinal: dev.cuda_stream().context().ordinal(),
            n,
            k,
            a4,
            dtype,
        }
    }

    fn shape(&self) -> String {
        format!(
            "n{}_k{}_{}_{:?}",
            self.n,
            self.k,
            if self.a4 { "a4" } else { "a16" },
            self.dtype
        )
        .to_ascii_lowercase()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct RoutedKey {
    ordinal: usize,
    experts: usize,
    n: usize,
    k: usize,
    topk: usize,
    a4: bool,
    dtype: DType,
}

impl RoutedKey {
    pub(super) fn new(
        dev: &CudaDevice,
        experts: usize,
        n: usize,
        k: usize,
        topk: usize,
        a4: bool,
        dtype: DType,
    ) -> Self {
        Self {
            ordinal: dev.cuda_stream().context().ordinal(),
            experts,
            n,
            k,
            topk,
            a4,
            dtype,
        }
    }

    fn shape(&self) -> String {
        format!(
            "e{}_n{}_k{}_t{}_{}_{:?}",
            self.experts,
            self.n,
            self.k,
            self.topk,
            if self.a4 { "a4" } else { "a16" },
            self.dtype
        )
        .to_ascii_lowercase()
    }
}

static DENSE_TUNED: TunedTable<DenseKey, DenseConfig> = TunedTable::new();
static ROUTED_TUNED: TunedTable<RoutedKey, RoutedGeometry> = TunedTable::new();

pub(super) fn dense_config(key: DenseKey, m: usize, device: MatmulDevice) -> DenseConfig {
    DENSE_TUNED
        .get(key, m)
        .unwrap_or_else(|| dense_policy(m, key.n, key.k, key.a4, device))
}

/// Decode-sized batches keep the routing-cost policy; only prefill buckets are tuned.
pub(super) fn routed_config(key: RoutedKey, tokens: usize) -> RoutedGeometry {
    (tokens >= ROUTED_PREFILL_MIN_TOKENS)
        .then(|| ROUTED_TUNED.get(key, tokens))
        .flatten()
        .unwrap_or_else(|| routed_geometry(tokens, key.n, key.k, key.a4))
}

tunable!(DenseConfig {
    bm,
    bn,
    bk,
    group_m,
    latency,
    warps,
    blocks_per_sm
});
tunable!(RoutedGeometry {
    rows,
    columns,
    k,
    persistent,
    latency,
    workers_per_sm
});

/// Shared-memory bytes one pipeline stage of a (rows x columns x k) tile needs; A16 stages the
/// activation at two bytes per value next to the packed weight.
fn stage_bytes(rows: usize, columns: usize, k: usize, a4: bool) -> usize {
    let weight = columns * (k / 2 + k / BLOCK_SIZE);
    let activation = if a4 {
        rows * (k / 2 + k / BLOCK_SIZE)
    } else {
        rows * k * 2
    };
    weight + activation
}

/// Drops tiles whose pipelined stages cannot fit in the device's opt-in shared memory.
fn fits_smem(rows: usize, columns: usize, k: usize, latency: usize, a4: bool, smem: usize) -> bool {
    smem == 0 || stage_bytes(rows, columns, k, a4) * latency.max(1) <= smem
}

fn dense_space(bucket: Bucket, key: DenseKey, device: MatmulDevice, smem: usize) -> Space {
    let policy = dense_policy(bucket.probe, key.n, key.k, key.a4, device);
    let tiles: &[[i64; 2]] = if bucket.upper <= DENSE_DECODE_MAX_ROWS {
        &[[16, 64], [16, 128], [32, 64], [32, 128]]
    } else {
        &[[64, 64], [64, 128], [128, 64], [128, 128], [32, 128]]
    };
    let row_cap = bucket.probe.next_power_of_two().max(MIN_TILE_ROWS) as i64;
    let a4 = key.a4;
    Space::new()
        .joint(["bm", "bn"], tiles.iter().copied())
        .axis("group_m", [1, 8, 16])
        .axis("bk", [256, 128, 512])
        .axis("warps", [0, 8, 16])
        .axis("latency", [3, 2, 4])
        .axis("blocks_per_sm", [2, 1, 3])
        .constrain(move |c| {
            DenseConfig::from_config(c).is_some_and(|cfg| {
                cfg.bm as i64 <= row_cap && fits_smem(cfg.bm, cfg.bn, cfg.bk, cfg.latency, a4, smem)
            })
        })
        .policy(policy.to_config())
}

fn routed_space(bucket: Bucket, key: RoutedKey, smem: usize) -> Space {
    let policy = routed_geometry(bucket.probe, key.n, key.k, key.a4);
    let a4 = key.a4;
    Space::new()
        .joint(
            ["rows", "columns"],
            [
                [16, 64],
                [16, 128],
                [32, 64],
                [32, 128],
                [64, 64],
                [64, 128],
            ],
        )
        .axis("k", [256, 128])
        .axis("persistent", [1, 0])
        .axis("latency", [3, 2, 4])
        .axis("workers_per_sm", [1, 2])
        .constrain(move |c| {
            RoutedGeometry::from_config(c).is_some_and(|cfg| {
                cfg.rows <= MAX_ROUTED_ROWS
                    && fits_smem(cfg.rows, cfg.columns, cfg.k, cfg.latency, a4, smem)
            })
        })
        .policy(policy.to_config())
}

/// Rows that compile every JIT variant a tuned dense bucket can launch.
pub(super) fn dense_warm_rows() -> Vec<usize> {
    bucket_warm_rows(&dense_buckets(), 1)
}

fn dense_buckets() -> Vec<Bucket> {
    buckets_from_breakpoints(&DENSE_BREAKPOINTS, DENSE_PREFILL_PROBE, DENSE_PREFILL_PROBE)
}

/// Prefill buckets on either side of the policy's large-prefill breakpoint.
fn routed_buckets() -> [Bucket; 2] {
    [
        Bucket {
            upper: ROUTED_LARGE_PREFILL_MIN_TOKENS - 1,
            probe: ROUTED_SHORT_PROBE,
        },
        Bucket {
            upper: usize::MAX,
            probe: ROUTED_LARGE_PREFILL_MIN_TOKENS,
        },
    ]
}

/// Tunes one dense NVFP4 weight shape once per process; later layers of the shape reuse it.
pub(super) fn tune_dense(args: Nvfp4GemmArgs<'_>, dtype: DType) -> Result<()> {
    let dev = args.weights.device().as_cuda_device()?;
    let (n, packed_k) = args.weights.dims2()?;
    let a4 = args.activation_global_scale.is_some();
    let key = DenseKey::new(dev, n, packed_k * 2, a4, dtype);
    DENSE_TUNED.tune_once(key, |mode| tune_dense_with(args, key, mode))
}

fn tune_dense_with(args: Nvfp4GemmArgs<'_>, key: DenseKey, mode: TuneMode) -> Result<Vec<Tuned>> {
    let device = args.weights.device().clone();
    let dev = device.as_cuda_device()?;
    let (k, dtype) = (key.k, key.dtype);
    let buckets = dense_buckets();
    let matmul = matmul_device(dev);
    let smem = cuda_device_info(dev).smem_optin;
    let space = move |bucket: Bucket| dense_space(bucket, key, matmul, smem);
    let request = TuneRequest {
        kernel: DENSE_KERNEL,
        source_hash: KERNEL_SOURCE_HASH,
        shape: key.shape(),
        buckets: &buckets,
        space: &space,
    };
    let layer = Nvfp4Weights::new(args);
    let mut inputs = HashMap::new();
    Ok(DENSE_TUNED.tune(key, dev, mode, &request, |rows, cfg| {
        let input = probe_input(&mut inputs, rows, || {
            let x = random_activation(rows, k, dtype, &device)?;
            Ok(match &layer.activation_global_scale {
                Some(global) => (None, Some(quantize(&x, global, false)?)),
                None => (Some(x), None),
            })
        })?;
        let (x, prequantized) = input.clone();
        let layer = layer.clone();
        let launch = move |compile_only| {
            let operands = DenseOperands {
                x: x.as_ref(),
                prequantized: prequantized.as_ref().map(|(p, s)| (p, s)),
                dtype,
                args: layer.args(),
            };
            launch_dense(operands, Some(cfg), compile_only)
        };
        launch(true)?;
        Prepared::from_launch(move || launch(false))
    }))
}

/// Tunes one routed NVFP4 expert shape at prefill token counts with spread routing.
pub(super) fn tune_routed(args: Nvfp4GemmArgs<'_>, key: RoutedKey) -> Result<()> {
    ROUTED_TUNED.tune_once(key, |mode| tune_routed_with(args, key, mode))
}

fn tune_routed_with(args: Nvfp4GemmArgs<'_>, key: RoutedKey, mode: TuneMode) -> Result<Vec<Tuned>> {
    let device = args.weights.device().clone();
    let dev = device.as_cuda_device()?;
    let buckets = routed_buckets();
    let smem = cuda_device_info(dev).smem_optin;
    let space = move |bucket: Bucket| routed_space(bucket, key, smem);
    let request = TuneRequest {
        kernel: ROUTED_KERNEL,
        source_hash: KERNEL_SOURCE_HASH,
        shape: key.shape(),
        buckets: &buckets,
        space: &space,
    };
    let layer = Nvfp4Weights::new(args);
    let mut inputs = HashMap::new();
    Ok(
        ROUTED_TUNED.tune(key, dev, mode, &request, |tokens, geometry| {
            let (x, indices) = probe_input(&mut inputs, tokens, || {
                let ids = TuneRouting::for_tokens(tokens).expert_ids(tokens, key.topk, key.experts);
                let indices = Tensor::from_vec(ids, (tokens, key.topk), &device)?;
                Ok((
                    random_activation(tokens, key.k, key.dtype, &device)?,
                    indices,
                ))
            })?;
            let plan = |compile_only| GatherPlan {
                geometry: Some(geometry),
                uniform_input_scale: false,
                compile_only,
            };
            let compiled = prepare_gather_with(x, indices, layer.args(), plan(true))?;
            launch_gather_prepared(&compiled, layer.args(), true)?;
            let prepared = prepare_gather_with(x, indices, layer.args(), plan(false))?;
            let layer = layer.clone();
            Prepared::from_launch(move || launch_gather_prepared(&prepared, layer.args(), false))
        }),
    )
}
