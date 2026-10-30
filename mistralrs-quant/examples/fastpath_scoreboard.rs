use std::{
    cell::OnceCell,
    collections::{HashMap, HashSet},
    env,
    hint::black_box,
    io::{BufRead, BufReader, Write},
    panic::{self, AssertUnwindSafe},
    process::{self, Command, Stdio},
    sync::{atomic::AtomicUsize, mpsc, Arc},
    thread,
    time::{Duration, Instant},
};

use candle_core::{cuda::cudarc::driver::sys, DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Linear;
use float8::F8E4M3;
use mistralrs_quant::{
    blockwise_fp8_moe, linear_no_bias, quantize_expert_stack, try_fused_gate_up, try_fused_qkv,
    try_routed_ffn, EmbeddingOutput, Fp8ActivationScheme, GluActivationType, IsqType, MXFP4Layer,
    PreQuantizedExperts, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig,
    ShardedSafeTensors, UnquantLinear,
};
use serde_json::{json, Value};

const DENSE_K: usize = 4096;
const DENSE_N: usize = 12288;
const QKV_Q_OUT: usize = 4096;
const QKV_KV_OUT: usize = 1024;
const MOE_HIDDEN: usize = 2048;
const MOE_INTER: usize = 768;
const MOE_EXPERTS: usize = 64;
const MOE_TOPK: usize = 8;
const FFN_LAYERS: f64 = 3.0;
const DECODE_MAX_ROWS: usize = 8;
const EMBED_VOCAB: usize = 151936;
const EMBED_HIDDEN: usize = 2048;
const BYTES_PER_MB: f64 = 1e6;

const WEIGHT_STD: f32 = 0.02;
// static activation scales are calibrated for N(0, 1) inputs
const ACT_ABS_MAX: f32 = 8.0;
const FP8_MAX: f32 = 448.0;
const FP4_MAX: f32 = 6.0;
const FP4_MIDPOINTS: [f64; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
const FP4_SIGN: f64 = 8.0;
const NIBBLE_SHIFT: f64 = 16.0;
const NVFP4_GROUP: usize = 16;
const MX_GROUP: usize = 32;
const E8M0_BIAS: f64 = 127.0;
const FP8_BLOCK: usize = 128;

const FAST_BW: f64 = 0.6;
const PARTIAL_BW: f64 = 0.25;
const FAST_RATIO: f64 = 0.7;
const PARTIAL_RATIO: f64 = 0.3;

const SAMPLES: usize = 7;
const SAMPLE_BUDGET_MS: f64 = 25.0;
const MAX_ITERS_PER_SAMPLE: usize = 200;
const SLOW_CELL_MS: f64 = 1500.0;
const PEAK_COPY_ELEMS: usize = 256 << 20;
const PEAK_GEMV_ROWS: usize = 32768;
const PEAK_PROBE_REPEATS: usize = 3;
const L2_FLUSH_ELEMS: usize = 32 << 20;
const COLD_ITERS: usize = 41;
const ROW_TIMEOUT: Duration = Duration::from_secs(1200);
const UNQUANT_GATHER_LIMIT_BYTES: usize = 4 << 30;
const ROUTING_SEED: u64 = 0x5eed_c0de;
const LCG_MUL: u64 = 6364136223846793005;
const LCG_INC: u64 = 1442695040888963407;
const DENSE_PREFIX: &str = "model.layers.0.proj";
const MOE_PREFIX: &str = "model.layers.0.mlp";
const CELL_PREFIX: &str = "SCOREBOARD_CELL";
// set to run cuTile warmup (and its autotuner, see MISTRALRS_CUTILE_TUNE) after each layer build
const WARMUP_ENV: &str = "SCOREBOARD_WARMUP";
const NO_DEQUANT_NOTE: &str = " ref=orig weight";
const STICKY_EXIT_CODE: i32 = 3;
const NOTE_MAX_CHARS: usize = 120;
const LABEL_WIDTH: usize = 20;
const CELL_WIDTH: usize = 9;
// rows that force the pre-Blackwell NVFP4 kernels
const PORTABLE_NVFP4_ROWS: [&str; 2] = ["nvfp4_fallback", "nvfp4_w4a16_fallback"];

#[derive(Clone, Copy, PartialEq)]
enum OpKind {
    Dense(usize),
    Moe(usize),
    MoeFfn(usize),
    GateUp(usize),
    Qkv(usize),
    Embed(usize),
}

struct Op {
    id: &'static str,
    label: &'static str,
    kind: OpKind,
}

const OPS: [Op; 16] = [
    Op {
        id: "m1",
        label: "decode M=1",
        kind: OpKind::Dense(1),
    },
    Op {
        id: "m4",
        label: "batch M=4",
        kind: OpKind::Dense(4),
    },
    Op {
        id: "m8",
        label: "batch M=8",
        kind: OpKind::Dense(8),
    },
    Op {
        id: "m512",
        label: "prefill M=512",
        kind: OpKind::Dense(512),
    },
    Op {
        id: "m2048",
        label: "prefill M=2048",
        kind: OpKind::Dense(2048),
    },
    Op {
        id: "moe_t1",
        label: "MoE decode T=1",
        kind: OpKind::Moe(1),
    },
    Op {
        id: "moe_t8",
        label: "MoE decode T=8",
        kind: OpKind::Moe(8),
    },
    Op {
        id: "ffn_t1",
        label: "routed FFN T=1",
        kind: OpKind::MoeFfn(1),
    },
    Op {
        id: "ffn_t8",
        label: "routed FFN T=8",
        kind: OpKind::MoeFfn(8),
    },
    Op {
        id: "moe_t512",
        label: "MoE prefill T=512",
        kind: OpKind::Moe(512),
    },
    Op {
        id: "gateup_m1",
        label: "fused gate/up M=1",
        kind: OpKind::GateUp(1),
    },
    Op {
        id: "gateup_m512",
        label: "fused gate/up M=512",
        kind: OpKind::GateUp(512),
    },
    Op {
        id: "qkv_m1",
        label: "fused QKV M=1",
        kind: OpKind::Qkv(1),
    },
    Op {
        id: "qkv_m512",
        label: "fused QKV M=512",
        kind: OpKind::Qkv(512),
    },
    Op {
        id: "embed_t1",
        label: "embedding T=1",
        kind: OpKind::Embed(1),
    },
    Op {
        id: "embed_t512",
        label: "embedding T=512",
        kind: OpKind::Embed(512),
    },
];

impl OpKind {
    fn rows(self) -> usize {
        match self {
            Self::Dense(m)
            | Self::Moe(m)
            | Self::MoeFfn(m)
            | Self::GateUp(m)
            | Self::Qkv(m)
            | Self::Embed(m) => m,
        }
    }

    fn is_decode(self) -> bool {
        self.rows() <= DECODE_MAX_ROWS
    }

    // ideal single bf16 GEMM with the same FLOPs
    fn baseline_shape(self) -> (usize, usize, usize) {
        match self {
            Self::Dense(m) => (m, DENSE_N, DENSE_K),
            Self::Moe(t) => (t * MOE_TOPK, MOE_INTER, MOE_HIDDEN),
            Self::MoeFfn(t) => (t * MOE_TOPK, 3 * MOE_INTER, MOE_HIDDEN),
            Self::GateUp(m) => (m, 2 * DENSE_N, DENSE_K),
            Self::Qkv(m) => (m, QKV_Q_OUT + 2 * QKV_KV_OUT, DENSE_K),
            // embeddings are ranked against a bf16 index_select instead
            Self::Embed(t) => (t, EMBED_HIDDEN, 1),
        }
    }
}

#[derive(Clone, Copy)]
enum Ckpt {
    Fp8Block,
    Fp8TensorW8A16,
    Fp8ChannelW8A16,
    Fp8TensorW8A8,
    Fp8ChannelW8A8,
    Mxfp8,
    Nvfp4,
    Nvfp4W4A16,
}

#[derive(Clone, Copy)]
enum Build {
    Bf16,
    Isq(IsqType),
    Ckpt(Ckpt),
    Mxfp4,
}

struct Format {
    id: &'static str,
    label: &'static str,
    group: &'static str,
    build: Build,
    bits_per_weight: f64,
    // vs the layer's own dequantized weight, covers activation quantization
    kernel_tol: f64,
    // vs the original bf16 weight, covers weight quantization error
    quant_tol: f64,
    // activation and output dtype of checkpoint layers
    dtype: DType,
}

macro_rules! fmt_row {
    ($id:literal, $label:literal, $group:literal, $build:expr, $bpw:literal, $ktol:literal, $qtol:literal) => {
        Format {
            id: $id,
            label: $label,
            group: $group,
            build: $build,
            bits_per_weight: $bpw,
            kernel_tol: $ktol,
            quant_tol: $qtol,
            dtype: DType::BF16,
        }
    };
    ($id:literal, $label:literal, $group:literal, $build:expr, $bpw:literal, $ktol:literal, $qtol:literal, $dtype:expr) => {
        Format {
            dtype: $dtype,
            ..fmt_row!($id, $label, $group, $build, $bpw, $ktol, $qtol)
        }
    };
}

const FORMATS: [Format; 29] = [
    fmt_row!("bf16", "bf16", "unquantized", Build::Bf16, 16.0, 0.01, 0.01),
    fmt_row!(
        "q4_0",
        "Q4_0 (ISQ)",
        "gguf",
        Build::Isq(IsqType::Q4_0),
        4.5,
        0.03,
        0.15
    ),
    fmt_row!(
        "q8_0",
        "Q8_0 (ISQ)",
        "gguf",
        Build::Isq(IsqType::Q8_0),
        8.5,
        0.03,
        0.03
    ),
    fmt_row!(
        "q2k",
        "Q2K (ISQ)",
        "gguf",
        Build::Isq(IsqType::Q2K),
        2.625,
        0.03,
        0.5
    ),
    fmt_row!(
        "q3k",
        "Q3K (ISQ)",
        "gguf",
        Build::Isq(IsqType::Q3K),
        3.4375,
        0.03,
        0.25
    ),
    fmt_row!(
        "q4k",
        "Q4K (ISQ)",
        "gguf",
        Build::Isq(IsqType::Q4K),
        4.5,
        0.03,
        0.12
    ),
    fmt_row!(
        "q5k",
        "Q5K (ISQ)",
        "gguf",
        Build::Isq(IsqType::Q5K),
        5.5,
        0.03,
        0.07
    ),
    fmt_row!(
        "q6k",
        "Q6K (ISQ)",
        "gguf",
        Build::Isq(IsqType::Q6K),
        6.5625,
        0.03,
        0.05
    ),
    fmt_row!(
        "afq4",
        "AFQ4 (ISQ)",
        "affine",
        Build::Isq(IsqType::AFQ4),
        4.5,
        0.02,
        0.15
    ),
    fmt_row!(
        "afq8",
        "AFQ8 (ISQ)",
        "affine",
        Build::Isq(IsqType::AFQ8),
        8.5,
        0.02,
        0.03
    ),
    fmt_row!(
        "hqq4",
        "HQQ4 (ISQ)",
        "affine",
        Build::Isq(IsqType::HQQ4),
        4.5,
        0.02,
        0.15
    ),
    fmt_row!(
        "hqq8",
        "HQQ8 (ISQ)",
        "affine",
        Build::Isq(IsqType::HQQ8),
        8.5,
        0.02,
        0.03
    ),
    fmt_row!(
        "f8q8",
        "F8Q8 (ISQ)",
        "fp8",
        Build::Isq(IsqType::F8Q8),
        8.5,
        0.03,
        0.08
    ),
    fmt_row!(
        "isq_fp8",
        "FP8 E4M3 (ISQ)",
        "fp8",
        Build::Isq(IsqType::F8E4M3),
        8.0,
        0.08,
        0.12
    ),
    fmt_row!(
        "fp8_block",
        "FP8 block128 W8A8",
        "fp8",
        Build::Ckpt(Ckpt::Fp8Block),
        8.0,
        0.08,
        0.12
    ),
    fmt_row!(
        "fp8_t_w8a16",
        "FP8 tensor W8A16",
        "fp8",
        Build::Ckpt(Ckpt::Fp8TensorW8A16),
        8.0,
        0.02,
        0.08
    ),
    fmt_row!(
        "fp8_c_w8a16",
        "FP8 channel W8A16",
        "fp8",
        Build::Ckpt(Ckpt::Fp8ChannelW8A16),
        8.0,
        0.02,
        0.08
    ),
    fmt_row!(
        "fp8_t_w8a8",
        "FP8 tensor W8A8",
        "fp8",
        Build::Ckpt(Ckpt::Fp8TensorW8A8),
        8.0,
        0.08,
        0.12
    ),
    fmt_row!(
        "fp8_c_w8a8",
        "FP8 channel W8A8",
        "fp8",
        Build::Ckpt(Ckpt::Fp8ChannelW8A8),
        8.0,
        0.08,
        0.12
    ),
    fmt_row!(
        "mxfp8",
        "MXFP8",
        "fp8",
        Build::Ckpt(Ckpt::Mxfp8),
        8.25,
        0.08,
        0.12
    ),
    fmt_row!(
        "nvfp4",
        "NVFP4 W4A4",
        "fp4",
        Build::Ckpt(Ckpt::Nvfp4),
        4.5,
        0.25,
        0.35
    ),
    fmt_row!(
        "nvfp4_w4a16",
        "NVFP4 W4A16",
        "fp4",
        Build::Ckpt(Ckpt::Nvfp4W4A16),
        4.5,
        0.02,
        0.25
    ),
    fmt_row!(
        "nvfp4_fallback",
        "NVFP4 W4A4 pre-BW",
        "fp4",
        Build::Ckpt(Ckpt::Nvfp4),
        4.5,
        0.25,
        0.35
    ),
    fmt_row!(
        "nvfp4_w4a16_fallback",
        "NVFP4 W4A16 pre-BW",
        "fp4",
        Build::Ckpt(Ckpt::Nvfp4W4A16),
        4.5,
        0.02,
        0.25
    ),
    fmt_row!(
        "mxfp8_f16",
        "MXFP8 f16",
        "fp8",
        Build::Ckpt(Ckpt::Mxfp8),
        8.25,
        0.08,
        0.12,
        DType::F16
    ),
    fmt_row!(
        "nvfp4_f16",
        "NVFP4 W4A4 f16",
        "fp4",
        Build::Ckpt(Ckpt::Nvfp4),
        4.5,
        0.25,
        0.35,
        DType::F16
    ),
    fmt_row!(
        "nvfp4_w4a16_f16",
        "NVFP4 W4A16 f16",
        "fp4",
        Build::Ckpt(Ckpt::Nvfp4W4A16),
        4.5,
        0.02,
        0.25,
        DType::F16
    ),
    fmt_row!("mxfp4", "MXFP4", "fp4", Build::Mxfp4, 4.25, 0.02, 0.3),
    fmt_row!(
        "mxfp4_isq",
        "MXFP4 (ISQ)",
        "fp4",
        Build::Isq(IsqType::MXFP4),
        4.25,
        0.02,
        0.3
    ),
];

#[derive(Clone, Copy, PartialEq)]
enum Status {
    Fast,
    Partial,
    Fallback,
    None,
    Wrong,
}

impl Status {
    fn name(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Partial => "partial",
            Self::Fallback => "fallback",
            Self::None => "none",
            Self::Wrong => "wrong",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "fast" => Self::Fast,
            "partial" => Self::Partial,
            "fallback" => Self::Fallback,
            "wrong" => Self::Wrong,
            _ => Self::None,
        }
    }

    fn letter(self) -> &'static str {
        match self {
            Self::Fast => "F",
            Self::Partial => "P",
            Self::Fallback => "B",
            Self::None => "-",
            Self::Wrong => "W!",
        }
    }
}

struct Cell {
    status: Status,
    short: String,
    note: String,
}

impl Cell {
    fn none(note: impl Into<String>) -> Self {
        Self {
            status: Status::None,
            short: String::new(),
            note: note.into(),
        }
    }
}

fn sanitize(text: &str) -> String {
    let flat: String = text
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    flat.chars().take(NOTE_MAX_CHARS).collect()
}

fn find_format(id: &str) -> Option<&'static Format> {
    FORMATS.iter().find(|f| f.id == id)
}

fn find_op(id: &str) -> Option<&'static Op> {
    OPS.iter().find(|o| o.id == id)
}

fn measure(dev: &Device, mut launch: impl FnMut() -> Result<Tensor>) -> Result<f64> {
    black_box(launch()?);
    dev.synchronize()?;
    let start = Instant::now();
    black_box(launch()?);
    dev.synchronize()?;
    let first = start.elapsed().as_secs_f64() * 1e3;
    if first > SLOW_CELL_MS {
        return Ok(first);
    }
    let iters = ((SAMPLE_BUDGET_MS / first.max(1e-3)) as usize).clamp(1, MAX_ITERS_PER_SAMPLE);
    let stream = dev.as_cuda_device()?.cuda_stream();
    let event = || {
        stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::msg)
    };
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let begin = event()?;
        for _ in 0..iters {
            black_box(launch()?);
        }
        let end = event()?;
        end.synchronize().map_err(candle_core::Error::msg)?;
        let ms = begin.elapsed_ms(&end).map_err(candle_core::Error::msg)? as f64;
        samples.push(ms / iters as f64);
    }
    samples.sort_by(f64::total_cmp);
    Ok(samples[SAMPLES / 2])
}

// per-launch event timing with L2 evicted first, since a real decode step never finds the weights cached
fn measure_cold(
    dev: &Device,
    flush: &Tensor,
    mut launch: impl FnMut() -> Result<Tensor>,
) -> Result<f64> {
    black_box(launch()?);
    dev.synchronize()?;
    let stream = dev.as_cuda_device()?.cuda_stream();
    let event = || {
        stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(candle_core::Error::msg)
    };
    let mut pairs = Vec::with_capacity(COLD_ITERS);
    for _ in 0..COLD_ITERS {
        black_box(flush.sum_all()?);
        let begin = event()?;
        black_box(launch()?);
        pairs.push((begin, event()?));
    }
    dev.synchronize()?;
    let mut samples = pairs
        .iter()
        .map(|(begin, end)| begin.elapsed_ms(end).map(|ms| ms as f64))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(candle_core::Error::msg)?;
    samples.sort_by(f64::total_cmp);
    Ok(samples[COLD_ITERS / 2])
}

fn rel_err(y: &Tensor, reference: &Tensor) -> Result<f64> {
    let y = y.to_dtype(DType::F32)?.flatten_all()?;
    let r = reference.to_dtype(DType::F32)?.flatten_all()?;
    let num = (y - &r)?.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    let den = r.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
    Ok((num / den.max(f64::MIN_POSITIVE)).sqrt())
}

fn to_f8(t: &Tensor) -> Result<Tensor> {
    let values = t
        .to_dtype(DType::F32)?
        .clamp(-FP8_MAX, FP8_MAX)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let codes: Vec<F8E4M3> = values.iter().map(|&v| F8E4M3::from_f32(v)).collect();
    Tensor::from_vec(codes, t.shape(), t.device())
}

fn f8_to_f32(t: &Tensor) -> Result<Tensor> {
    let codes = t.flatten_all()?.to_vec1::<F8E4M3>()?;
    let values: Vec<f32> = codes.iter().map(|c| c.to_f32()).collect();
    Tensor::from_vec(values, t.shape(), t.device())
}

// rows x k f32 already divided by its block scale -> packed e2m1, first value in the low nibble
fn fp4_pack(scaled: &Tensor) -> Result<Tensor> {
    let (rows, k) = scaled.dims2()?;
    let mag = scaled.abs()?;
    let mut code = mag.zeros_like()?;
    for mid in FP4_MIDPOINTS {
        code = (code + mag.gt(mid)?.to_dtype(DType::F32)?)?;
    }
    let sign = (scaled.lt(0f64)?.to_dtype(DType::F32)? * FP4_SIGN)?;
    let nibbles = (code + sign)?.reshape((rows, k / 2, 2))?;
    let lo = nibbles.i((.., .., 0))?;
    let hi = (nibbles.i((.., .., 1))? * NIBBLE_SHIFT)?;
    (lo + hi)?.to_dtype(DType::U8)
}

fn block_amax(w: &Tensor, group: usize) -> Result<Tensor> {
    let (rows, k) = w.dims2()?;
    w.reshape((rows, k / group, group))?.abs()?.max(D::Minus1)
}

fn expand_groups(scale: &Tensor, group: usize) -> Result<Tensor> {
    let (rows, groups) = scale.dims2()?;
    scale
        .unsqueeze(2)?
        .broadcast_as((rows, groups, group))?
        .reshape((rows, groups * group))
}

// e8m0 exponent large enough that amax / 2^e fits under max_code
fn e8m0_exponent(amax: &Tensor, max_code: f32) -> Result<Tensor> {
    let ln2 = std::f64::consts::LN_2;
    ((amax / max_code as f64)?.log()? / ln2)?
        .ceil()?
        .clamp(-E8M0_BIAS as f32, E8M0_BIAS as f32)
}

fn exp2(e: &Tensor) -> Result<Tensor> {
    (e * std::f64::consts::LN_2)?.exp()
}

struct MxQuant {
    codes: Tensor,
    scales: Tensor,
}

fn mxfp8_quant(w: &Tensor) -> Result<MxQuant> {
    let e = e8m0_exponent(&block_amax(w, MX_GROUP)?, FP8_MAX)?;
    let scaled = (w / expand_groups(&exp2(&e)?, MX_GROUP)?)?;
    Ok(MxQuant {
        codes: to_f8(&scaled)?,
        scales: (e + E8M0_BIAS)?.to_dtype(DType::U8)?,
    })
}

fn mxfp4_quant(w: &Tensor) -> Result<MxQuant> {
    let e = e8m0_exponent(&block_amax(w, MX_GROUP)?, FP4_MAX)?;
    let scaled = (w / expand_groups(&exp2(&e)?, MX_GROUP)?)?;
    Ok(MxQuant {
        codes: fp4_pack(&scaled)?,
        scales: (e + E8M0_BIAS)?.to_dtype(DType::U8)?,
    })
}

struct Nvfp4Quant {
    packed: Tensor,
    block_scales: Tensor,
    global_scale: f32,
}

fn nvfp4_quant(w: &Tensor) -> Result<Nvfp4Quant> {
    let amax = w.abs()?.max_all()?.to_scalar::<f32>()?;
    let global_scale = amax / (FP4_MAX * FP8_MAX);
    let block = ((block_amax(w, NVFP4_GROUP)? / FP4_MAX as f64)? / global_scale as f64)?;
    let block_scales = to_f8(&block)?;
    let rounded = f8_to_f32(&block_scales)?;
    let safe = (&rounded + rounded.eq(0f64)?.to_dtype(DType::F32)?)?;
    let denom = (expand_groups(&safe, NVFP4_GROUP)? * global_scale as f64)?;
    Ok(Nvfp4Quant {
        packed: fp4_pack(&(w / denom)?)?,
        block_scales,
        global_scale,
    })
}

fn fp8_block_quant(w: &Tensor) -> Result<(Tensor, Tensor)> {
    let (n, k) = w.dims2()?;
    let (bn, bk) = (n / FP8_BLOCK, k / FP8_BLOCK);
    let scale = (w
        .reshape((bn, FP8_BLOCK, bk, FP8_BLOCK))?
        .abs()?
        .max(3)?
        .max(1)?
        / FP8_MAX as f64)?;
    let expanded = scale
        .reshape((bn, 1, bk, 1))?
        .broadcast_as((bn, FP8_BLOCK, bk, FP8_BLOCK))?
        .reshape((n, k))?;
    Ok((to_f8(&(w / expanded)?)?, scale))
}

fn act_scale(max_code: f32) -> f32 {
    ACT_ABS_MAX / max_code
}

fn ckpt_config(kind: Ckpt) -> Result<QuantizedConfig> {
    let ct = |strategy: &str| {
        json!({
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {"group_0": {
                "targets": ["Linear"],
                "weights": {"type": "float", "num_bits": 8, "dynamic": false, "symmetric": true, "strategy": strategy},
                "input_activations": null
            }}
        })
    };
    let modelopt = |algo: &str| json!({"quant_method": "modelopt", "quant_algo": algo});
    let value = match kind {
        Ckpt::Fp8Block => json!({
            "quant_method": "fp8",
            "weight_block_size": [FP8_BLOCK, FP8_BLOCK],
            "activation_scheme": "dynamic"
        }),
        Ckpt::Fp8TensorW8A16 => ct("tensor"),
        Ckpt::Fp8ChannelW8A16 => ct("channel"),
        Ckpt::Fp8TensorW8A8 => modelopt("FP8"),
        Ckpt::Fp8ChannelW8A8 => modelopt("FP8_PER_CHANNEL_PER_TOKEN"),
        Ckpt::Mxfp8 => modelopt("MXFP8"),
        Ckpt::Nvfp4 => modelopt("NVFP4"),
        Ckpt::Nvfp4W4A16 => modelopt("W4A16_NVFP4"),
    };
    serde_json::from_value(value).map_err(candle_core::Error::msg)
}

// checkpoint tensors for one [N, K] f32 weight, named the way each dialect stores them
fn ckpt_entries(
    kind: Ckpt,
    w: &Tensor,
    prefix: &str,
    map: &mut HashMap<String, Tensor>,
) -> Result<()> {
    let dev = w.device();
    let mut put = |name: &str, t: Tensor| {
        map.insert(format!("{prefix}.{name}"), t);
    };
    match kind {
        Ckpt::Fp8Block => {
            let (q, s) = fp8_block_quant(w)?;
            put("weight", q);
            put("weight_scale_inv", s);
        }
        Ckpt::Fp8TensorW8A16 | Ckpt::Fp8TensorW8A8 => {
            let scale = w.abs()?.max_all()?.to_scalar::<f32>()? / FP8_MAX;
            put("weight", to_f8(&(w / scale as f64)?)?);
            put("weight_scale", Tensor::new(scale, dev)?);
            if matches!(kind, Ckpt::Fp8TensorW8A8) {
                put("input_scale", Tensor::new(act_scale(FP8_MAX), dev)?);
            }
        }
        Ckpt::Fp8ChannelW8A16 | Ckpt::Fp8ChannelW8A8 => {
            let scale = (w.abs()?.max_keepdim(1)? / FP8_MAX as f64)?;
            put("weight", to_f8(&w.broadcast_div(&scale)?)?);
            let scale = if matches!(kind, Ckpt::Fp8ChannelW8A8) {
                scale.squeeze(1)?
            } else {
                scale
            };
            put("weight_scale", scale);
        }
        Ckpt::Mxfp8 => {
            let q = mxfp8_quant(w)?;
            put("weight", q.codes);
            put("weight_scale", q.scales);
        }
        Ckpt::Nvfp4 | Ckpt::Nvfp4W4A16 => {
            let q = nvfp4_quant(w)?;
            put("weight", q.packed);
            put("weight_scale", q.block_scales);
            put("weight_scale_2", Tensor::new(q.global_scale, dev)?);
            if matches!(kind, Ckpt::Nvfp4) {
                put(
                    "input_scale",
                    Tensor::new(act_scale(FP4_MAX * FP8_MAX), dev)?,
                );
            }
        }
    }
    Ok(())
}

fn unquant(w: Tensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(<UnquantLinear as QuantMethod>::new(
        QuantMethodConfig::Unquantized(Linear::new(w, None)),
    )?))
}

fn build_dense(fmt: &Format, w: &Tensor) -> Result<Arc<dyn QuantMethod>> {
    let dev = w.device().clone();
    let (n, k) = w.dims2()?;
    match fmt.build {
        Build::Bf16 => unquant(w.to_dtype(DType::BF16)?),
        Build::Isq(ty) => unquant(w.to_dtype(DType::BF16)?)?.apply_isq(
            Some(ty),
            dev,
            &AtomicUsize::new(0),
            None,
            QuantizeOntoGuard::new(),
        ),
        Build::Mxfp4 => {
            let q = mxfp4_quant(w)?;
            Ok(Arc::new(<MXFP4Layer as QuantMethod>::new(
                QuantMethodConfig::MXFP4 {
                    blocks: q.codes,
                    scales: q.scales,
                    bias: None,
                },
            )?))
        }
        Build::Ckpt(kind) => {
            let mut map = HashMap::new();
            ckpt_entries(kind, w, DENSE_PREFIX, &mut map)?;
            let vb = ShardedSafeTensors::wrap(map, fmt.dtype, dev);
            linear_no_bias(k, n, &Some(ckpt_config(kind)?), vb.pp(DENSE_PREFIX))
        }
    }
}

fn build_moe(fmt: &Format, w: &Tensor) -> Result<Arc<dyn QuantMethod>> {
    let dev = w.device().clone();
    let (experts, n, k) = w.dims3()?;
    match fmt.build {
        Build::Bf16 => unquant(w.to_dtype(DType::BF16)?),
        Build::Isq(ty) if ty.supports_stacked_gather() => quantize_expert_stack(
            w.to_dtype(DType::BF16)?,
            ty,
            None,
            &dev,
            QuantizeOntoGuard::new(),
        ),
        Build::Isq(ty) => unquant(w.to_dtype(DType::BF16)?)?.apply_isq(
            Some(ty),
            dev,
            &AtomicUsize::new(0),
            None,
            QuantizeOntoGuard::new(),
        ),
        Build::Mxfp4 => {
            let q = mxfp4_quant(&w.reshape((experts * n, k))?)?;
            Ok(Arc::new(<MXFP4Layer as QuantMethod>::new(
                QuantMethodConfig::MXFP4 {
                    blocks: q.codes.reshape((experts, n, k / 2))?,
                    scales: q.scales.reshape((experts, n, k / MX_GROUP))?,
                    bias: None,
                },
            )?))
        }
        Build::Ckpt(Ckpt::Fp8Block) => {
            let mut codes = Vec::with_capacity(experts);
            let mut scales = Vec::with_capacity(experts);
            for e in 0..experts {
                let (q, s) = fp8_block_quant(&w.i(e)?)?;
                codes.push(q);
                scales.push(s);
            }
            blockwise_fp8_moe(
                Tensor::stack(&codes, 0)?,
                Tensor::stack(&scales, 0)?,
                vec![FP8_BLOCK, FP8_BLOCK],
                Some(Fp8ActivationScheme::Dynamic),
                DType::BF16,
            )
        }
        Build::Ckpt(kind) => {
            let mut map = HashMap::new();
            for e in 0..experts {
                let w_e = w.i(e)?.contiguous()?;
                let base = format!("{MOE_PREFIX}.experts.{e}");
                ckpt_entries(kind, &w_e, &format!("{base}.gate_proj"), &mut map)?;
                ckpt_entries(kind, &w_e, &format!("{base}.up_proj"), &mut map)?;
                let down = w_e.t()?.contiguous()?;
                ckpt_entries(kind, &down, &format!("{base}.down_proj"), &mut map)?;
            }
            let vb = ShardedSafeTensors::wrap(map, fmt.dtype, dev);
            let experts = PreQuantizedExperts::new(
                k,
                n,
                experts,
                &Some(ckpt_config(kind)?),
                vb.pp(MOE_PREFIX),
            )?;
            Ok(experts.fused_gate_proj)
        }
    }
}

struct LayerCase {
    layer: Arc<dyn QuantMethod>,
    orig: Tensor,
    deq: Tensor,
    deq_note: String,
}

fn make_case(
    fmt: &Format,
    shape: &[usize],
    dev: &Device,
    build: fn(&Format, &Tensor) -> Result<Arc<dyn QuantMethod>>,
) -> Result<LayerCase> {
    let orig = Tensor::randn(0f32, WEIGHT_STD, shape, dev)?
        .to_dtype(DType::BF16)?
        .to_dtype(DType::F32)?;
    let layer = build(fmt, &orig)?;
    if env::var_os(WARMUP_ENV).is_some() {
        // inference only warns on a failed warmup too; the cell still runs and JITs lazily
        if let Err(e) = mistralrs_quant::cutile::warmup_moe_kernels(dev, true) {
            eprintln!("warmup failed: {}", sanitize(&e.to_string()));
        }
    }
    let (deq, deq_note) = match layer.dequantize_w() {
        Ok(d) if d.elem_count() == orig.elem_count() => (
            d.to_device(dev)?.to_dtype(DType::F32)?.reshape(shape)?,
            String::new(),
        ),
        Ok(d) => (
            orig.clone(),
            format!("{NO_DEQUANT_NOTE} (dequantize_w shape {:?})", d.dims()),
        ),
        Err(e) => (orig.clone(), format!("{NO_DEQUANT_NOTE} ({e})")),
    };
    Ok(LayerCase {
        layer,
        orig,
        deq,
        deq_note,
    })
}

type Lazy = OnceCell<std::result::Result<LayerCase, String>>;

struct Worker {
    fmt: &'static Format,
    dev: Device,
    peak_gbs: f64,
    flush: Tensor,
    up: Lazy,
    gate: Lazy,
    q: Lazy,
    k: Lazy,
    v: Lazy,
    moe: Lazy,
    moe_up: Lazy,
    moe_down: Lazy,
    embed: Lazy,
}

impl Worker {
    fn get<'a>(
        &self,
        cell: &'a Lazy,
        shape: &[usize],
        build: fn(&Format, &Tensor) -> Result<Arc<dyn QuantMethod>>,
    ) -> Result<&'a LayerCase> {
        cell.get_or_init(|| make_case(self.fmt, shape, &self.dev, build).map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| candle_core::Error::msg(format!("build: {e}")))
    }

    fn input(&self, shape: &[usize]) -> Result<(Tensor, Tensor)> {
        let x = Tensor::randn(0f32, 1f32, shape, &self.dev)?.to_dtype(self.fmt.dtype)?;
        let xf = x.to_dtype(DType::F32)?;
        Ok((x, xf))
    }

    fn check(&self, kernel_rel: f64, quant_rel: f64, note: &str) -> Option<Cell> {
        let kernel_ok = note.starts_with(NO_DEQUANT_NOTE) || kernel_rel <= self.fmt.kernel_tol;
        let ok = kernel_ok && quant_rel <= self.fmt.quant_tol;
        (!ok).then(|| Cell {
            status: Status::Wrong,
            short: String::new(),
            note: format!(
                "rel vs dequant {kernel_rel:.4} (tol {}), vs orig {quant_rel:.4} (tol {}){note}",
                self.fmt.kernel_tol, self.fmt.quant_tol
            ),
        })
    }

    fn classify(&self, op: &Op, bytes: f64, ms: f64, note: &str) -> Result<Cell> {
        let (frac, fast, partial, short, text) = if op.kind.is_decode() {
            let eff = bytes / (ms * 1e-3) / (self.peak_gbs * 1e9);
            (
                eff,
                FAST_BW,
                PARTIAL_BW,
                format!("{:.0}%", eff * 100.0),
                format!("{:.0}% of peak bw, {ms:.3} ms", eff * 100.0),
            )
        } else {
            // measured right after the cell so shared-GPU contention hits both alike
            let base = bf16_gemm_ms(&self.dev, op.kind.baseline_shape())?;
            let ratio = base / ms;
            (
                ratio,
                FAST_RATIO,
                PARTIAL_RATIO,
                format!("{ratio:.2}x"),
                format!("{ratio:.2}x bf16 GEMM, {ms:.3} ms"),
            )
        };
        let status = if frac >= fast {
            Status::Fast
        } else if frac >= partial {
            Status::Partial
        } else {
            Status::Fallback
        };
        Ok(Cell {
            status,
            short,
            note: format!("{text}{note}"),
        })
    }

    fn time(&self, op: &Op, launch: impl FnMut() -> Result<Tensor>) -> Result<f64> {
        if op.kind.is_decode() {
            measure_cold(&self.dev, &self.flush, launch)
        } else {
            measure(&self.dev, launch)
        }
    }

    fn weight_bytes(&self, elems: usize) -> f64 {
        elems as f64 * self.fmt.bits_per_weight / 8.0
    }

    fn dense(&self, op: &Op, m: usize) -> Result<Cell> {
        let case = self.get(&self.up, &[DENSE_N, DENSE_K], build_dense)?;
        let (x, xf) = self.input(&[1, m, DENSE_K])?;
        let xf = xf.squeeze(0)?;
        let y = case.layer.forward(&x)?;
        let kernel_rel = rel_err(&y, &xf.matmul(&case.deq.t()?)?)?;
        let quant_rel = rel_err(&y, &xf.matmul(&case.orig.t()?)?)?;
        if let Some(cell) = self.check(kernel_rel, quant_rel, &case.deq_note) {
            return Ok(cell);
        }
        let ms = self.time(op, || case.layer.forward(&x))?;
        self.classify(op, self.weight_bytes(DENSE_N * DENSE_K), ms, &case.deq_note)
    }

    fn moe(&self, op: &Op, tokens: usize) -> Result<Cell> {
        if matches!(self.fmt.build, Build::Bf16) {
            let gathered = tokens * MOE_TOPK * MOE_INTER * MOE_HIDDEN * 2;
            if gathered > UNQUANT_GATHER_LIMIT_BYTES {
                return Ok(Cell {
                    status: Status::Fallback,
                    short: "skip".into(),
                    note: format!(
                        "not run: unquantized gather materializes {:.1} GB of expert weights",
                        gathered as f64 / 1e9
                    ),
                });
            }
        }
        let case = self.get(&self.moe, &[MOE_EXPERTS, MOE_INTER, MOE_HIDDEN], build_moe)?;
        let ids = routing(tokens);
        let unique = ids.iter().collect::<HashSet<_>>().len();
        let idx = Tensor::from_vec(ids.clone(), (tokens, MOE_TOPK), &self.dev)?;
        let (x, xf) = self.input(&[tokens, 1, MOE_HIDDEN])?;
        let xt = xf.squeeze(1)?.t()?.contiguous()?;
        let y = case.layer.gather_forward(&x, &idx)?;
        if y.elem_count() != tokens * MOE_TOPK * MOE_INTER {
            return Ok(Cell {
                status: Status::Wrong,
                short: String::new(),
                note: format!(
                    "gather_forward on ({tokens}, 1, {MOE_HIDDEN}) routes with ({tokens}, {MOE_TOPK}) ids returned {:?}",
                    y.dims()
                ),
            });
        }
        let kernel_rel = rel_err(&y, &moe_reference(&case.deq, &xt, &ids)?)?;
        let quant_rel = rel_err(&y, &moe_reference(&case.orig, &xt, &ids)?)?;
        if let Some(cell) = self.check(kernel_rel, quant_rel, &case.deq_note) {
            return Ok(cell);
        }
        let ms = self.time(op, || case.layer.gather_forward(&x, &idx))?;
        let bytes = self.weight_bytes(unique * MOE_INTER * MOE_HIDDEN);
        self.classify(op, bytes, ms, &case.deq_note)
    }

    // gate/up/down + weighted reduce through the routed decode path the MoE block takes first
    fn moe_ffn(&self, op: &Op, tokens: usize) -> Result<Cell> {
        let gate_shape = [MOE_EXPERTS, MOE_INTER, MOE_HIDDEN];
        let gate = self.get(&self.moe, &gate_shape, build_moe)?;
        let up = self.get(&self.moe_up, &gate_shape, build_moe)?;
        let down = self.get(
            &self.moe_down,
            &[MOE_EXPERTS, MOE_HIDDEN, MOE_INTER],
            build_moe,
        )?;
        let ids = routing(tokens);
        let unique = ids.iter().collect::<HashSet<_>>().len();
        let idx = Tensor::from_vec(ids.clone(), (tokens, MOE_TOPK), &self.dev)?;
        let weights = (0..ids.len())
            .map(|i| (i % MOE_TOPK + 1) as f32 / MOE_TOPK as f32)
            .collect::<Vec<_>>();
        let route_w = Tensor::from_slice(&weights, (tokens, MOE_TOPK), &self.dev)?;
        let (x, xf) = self.input(&[tokens, MOE_HIDDEN])?;
        let xt = xf.t()?.contiguous()?;
        let members = [&*gate.layer, &*up.layer, &*down.layer];
        let run = || try_routed_ffn(&x, &idx, &route_w, members, GluActivationType::Silu);
        let Some(y) = run()? else {
            return Ok(Cell::none("no routed FFN path"));
        };
        let reference = |w: [&Tensor; 3]| ffn_reference(w, &xt, &ids, &weights);
        let kernel_rel = rel_err(&y, &reference([&gate.deq, &up.deq, &down.deq])?)?;
        let quant_rel = rel_err(&y, &reference([&gate.orig, &up.orig, &down.orig])?)?;
        // quant error from the three stacked layers adds roughly in quadrature vs the one-layer tolerance
        let quant_rel = quant_rel / FFN_LAYERS.sqrt();
        if let Some(cell) = self.check(kernel_rel, quant_rel, &gate.deq_note) {
            return Ok(cell);
        }
        let ms = self.time(op, || {
            run()?.ok_or_else(|| candle_core::Error::msg("routed path disappeared"))
        })?;
        let bytes = self.weight_bytes(3 * unique * MOE_INTER * MOE_HIDDEN);
        self.classify(op, bytes, ms, &gate.deq_note)
    }

    fn gate_up(&self, op: &Op, m: usize) -> Result<Cell> {
        let up = self.get(&self.up, &[DENSE_N, DENSE_K], build_dense)?;
        let gate = self.get(&self.gate, &[DENSE_N, DENSE_K], build_dense)?;
        let (x, xf) = self.input(&[1, m, DENSE_K])?;
        let xf = xf.squeeze(0)?;
        let run = || try_fused_gate_up(&x, &*gate.layer, &*up.layer, GluActivationType::Silu);
        let Some(y) = run()? else {
            return Ok(Cell::none("no fused gate/up path"));
        };
        let glu = |g: &Tensor, u: &Tensor| -> Result<Tensor> {
            xf.matmul(&g.t()?)?.silu()? * xf.matmul(&u.t()?)?
        };
        let kernel_rel = rel_err(&y, &glu(&gate.deq, &up.deq)?)?;
        let quant_rel = rel_err(&y, &glu(&gate.orig, &up.orig)?)?;
        if let Some(cell) = self.check(kernel_rel, quant_rel, &up.deq_note) {
            return Ok(cell);
        }
        let ms = self.time(op, || {
            run()?.ok_or_else(|| candle_core::Error::msg("fused path disappeared"))
        })?;
        self.classify(
            op,
            self.weight_bytes(2 * DENSE_N * DENSE_K),
            ms,
            &up.deq_note,
        )
    }

    fn qkv(&self, op: &Op, m: usize) -> Result<Cell> {
        let q = self.get(&self.q, &[QKV_Q_OUT, DENSE_K], build_dense)?;
        let k = self.get(&self.k, &[QKV_KV_OUT, DENSE_K], build_dense)?;
        let v = self.get(&self.v, &[QKV_KV_OUT, DENSE_K], build_dense)?;
        let (x, xf) = self.input(&[1, m, DENSE_K])?;
        let xf = xf.squeeze(0)?;
        let run = || try_fused_qkv(&x, &*q.layer, &*k.layer, &*v.layer);
        let Some((yq, yk, yv)) = run()? else {
            return Ok(Cell::none("no fused QKV path"));
        };
        let flat = |ts: [&Tensor; 3]| -> Result<Tensor> {
            let parts = ts
                .iter()
                .map(|t| t.to_dtype(DType::F32)?.flatten_all())
                .collect::<Result<Vec<_>>>()?;
            Tensor::cat(&parts, 0)
        };
        let reference = |wq: &Tensor, wk: &Tensor, wv: &Tensor| -> Result<Tensor> {
            flat([
                &xf.matmul(&wq.t()?)?,
                &xf.matmul(&wk.t()?)?,
                &xf.matmul(&wv.t()?)?,
            ])
        };
        let y = flat([&yq, &yk, &yv])?;
        let kernel_rel = rel_err(&y, &reference(&q.deq, &k.deq, &v.deq)?)?;
        let quant_rel = rel_err(&y, &reference(&q.orig, &k.orig, &v.orig)?)?;
        if let Some(cell) = self.check(kernel_rel, quant_rel, &q.deq_note) {
            return Ok(cell);
        }
        let ms = self.time(op, || {
            run()?
                .map(|(q, _, _)| q)
                .ok_or_else(|| candle_core::Error::msg("fused path disappeared"))
        })?;
        let elems = (QKV_Q_OUT + 2 * QKV_KV_OUT) * DENSE_K;
        self.classify(op, self.weight_bytes(elems), ms, &q.deq_note)
    }

    fn embed(&self, op: &Op, tokens: usize) -> Result<Cell> {
        let case = self.get(&self.embed, &[EMBED_VOCAB, EMBED_HIDDEN], build_dense)?;
        let ids = Tensor::from_vec(embed_ids(tokens), (1, tokens), &self.dev)?;
        let flat = ids.flatten_all()?;
        let out = EmbeddingOutput::from(DType::BF16);
        let y = case.layer.embedding_forward(&ids, out)?;
        let kernel_rel = rel_err(&y, &case.deq.index_select(&flat, 0)?)?;
        let quant_rel = rel_err(&y, &case.orig.index_select(&flat, 0)?)?;
        if let Some(cell) = self.check(kernel_rel, quant_rel, &case.deq_note) {
            return Ok(cell);
        }
        let ms = self.time(op, || case.layer.embedding_forward(&ids, out))?;
        let table = case.orig.to_dtype(DType::BF16)?;
        let base = self.time(op, || table.index_select(&flat, 0))?;
        let ratio = base / ms;
        let status = if ratio >= FAST_RATIO {
            Status::Fast
        } else if ratio >= PARTIAL_RATIO {
            Status::Partial
        } else {
            Status::Fallback
        };
        let elems = EMBED_VOCAB * EMBED_HIDDEN;
        Ok(Cell {
            status,
            short: format!("{ratio:.2}x"),
            note: format!(
                "{ratio:.2}x bf16 index_select ({base:.4} ms), {ms:.4} ms, table {:.0} MB vs bf16 {:.0} MB{}",
                self.weight_bytes(elems) / BYTES_PER_MB,
                (elems * DType::BF16.size_in_bytes()) as f64 / BYTES_PER_MB,
                case.deq_note
            ),
        })
    }

    fn run(&self, op: &Op) -> Result<Cell> {
        match op.kind {
            OpKind::Dense(m) => self.dense(op, m),
            OpKind::Moe(t) => self.moe(op, t),
            OpKind::MoeFfn(t) => self.moe_ffn(op, t),
            OpKind::GateUp(m) => self.gate_up(op, m),
            OpKind::Qkv(m) => self.qkv(op, m),
            OpKind::Embed(t) => self.embed(op, t),
        }
    }
}

// distinct top-k experts per token from a fixed LCG so every row sees the same routing
fn routing(tokens: usize) -> Vec<u32> {
    let mut state = ROUTING_SEED;
    let mut next = || {
        state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
        (state >> 33) as usize
    };
    let mut ids = Vec::with_capacity(tokens * MOE_TOPK);
    for _ in 0..tokens {
        let mut picked = Vec::with_capacity(MOE_TOPK);
        while picked.len() < MOE_TOPK {
            let e = (next() % MOE_EXPERTS) as u32;
            if !picked.contains(&e) {
                picked.push(e);
            }
        }
        ids.extend(picked);
    }
    ids
}

// first and last rows plus LCG rows with a repeat
fn embed_ids(tokens: usize) -> Vec<u32> {
    let mut state = ROUTING_SEED;
    let mut ids = (0..tokens)
        .map(|_| {
            state = state.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
            ((state >> 33) as usize % EMBED_VOCAB) as u32
        })
        .collect::<Vec<_>>();
    if tokens > 2 {
        ids[0] = 0;
        ids[1] = ids[tokens / 2];
        ids[tokens - 1] = (EMBED_VOCAB - 1) as u32;
    }
    ids
}

// w [E, N, K] f32, xt [K, T] -> [T, topk, N]
fn moe_reference(w: &Tensor, xt: &Tensor, ids: &[u32]) -> Result<Tensor> {
    let all = w.broadcast_matmul(xt)?;
    let (_, n, tokens) = all.dims3()?;
    let host = all.flatten_all()?.to_vec1::<f32>()?;
    let mut out = Vec::with_capacity(ids.len() * n);
    for (route, &e) in ids.iter().enumerate() {
        let t = route / MOE_TOPK;
        let base = e as usize * n * tokens;
        out.extend((0..n).map(|row| host[base + row * tokens + t]));
    }
    Tensor::from_vec(out, (tokens, MOE_TOPK, n), w.device())
}

// [gate, up, down] f32 weights, xt [K, T] -> [T, hidden] silu-GLU FFN summed over routes
fn ffn_reference(w: [&Tensor; 3], xt: &Tensor, ids: &[u32], weights: &[f32]) -> Result<Tensor> {
    let mid =
        (moe_reference(w[0], xt, ids)?.silu()? * moe_reference(w[1], xt, ids)?)?.flatten_to(1)?;
    let mut rows = Vec::with_capacity(ids.len() / MOE_TOPK);
    for (t, routes) in ids.as_chunks::<MOE_TOPK>().0.iter().enumerate() {
        let mut acc = Tensor::zeros((MOE_HIDDEN, 1), DType::F32, xt.device())?;
        for (k, &e) in routes.iter().enumerate() {
            let r = t * MOE_TOPK + k;
            let y = w[2].i(e as usize)?.matmul(&mid.i(r)?.unsqueeze(1)?)?;
            acc = (acc + (y * f64::from(weights[r]))?)?;
        }
        rows.push(acc.squeeze(1)?);
    }
    Tensor::stack(&rows, 0)
}

fn is_sticky(message: &str) -> bool {
    [
        "ILLEGAL",
        "LAUNCH_FAILED",
        "MISALIGNED",
        "CUDA_ERROR_ASSERT",
        "UNKNOWN",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

fn emit(op: &Op, cell: &Cell) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(
        out,
        "{CELL_PREFIX}\t{}\t{}\t{}\t{}",
        op.id,
        cell.status.name(),
        sanitize(&cell.short),
        sanitize(&cell.note)
    );
    let _ = out.flush();
}

fn worker_main(args: &Args) -> Result<()> {
    let fmt = find_format(&args.worker)
        .ok_or_else(|| candle_core::Error::msg(format!("unknown format {}", args.worker)))?;
    let dev = Device::new_cuda(0)?;
    mistralrs_quant::nvfp4::force_portable_kernels(PORTABLE_NVFP4_ROWS.contains(&fmt.id));
    if env::var_os(WARMUP_ENV).is_some() {
        mistralrs_quant::cutile::register_moe_routing(&dev, MOE_EXPERTS, MOE_TOPK)?;
    }
    let worker = Worker {
        fmt,
        flush: Tensor::zeros(L2_FLUSH_ELEMS, DType::F32, &dev)?,
        dev,
        peak_gbs: args.peak_gbs,
        up: OnceCell::new(),
        gate: OnceCell::new(),
        q: OnceCell::new(),
        k: OnceCell::new(),
        v: OnceCell::new(),
        moe: OnceCell::new(),
        moe_up: OnceCell::new(),
        moe_down: OnceCell::new(),
        embed: OnceCell::new(),
    };
    for id in &args.cols {
        let Some(op) = find_op(id) else { continue };
        eprintln!("  [{}] {} ...", fmt.id, op.id);
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| worker.run(op)));
        let (cell, sticky) = match outcome {
            Ok(Ok(cell)) => (cell, false),
            Ok(Err(e)) => {
                let message = e.to_string();
                let sticky = is_sticky(&message);
                (Cell::none(format!("error: {message}")), sticky)
            }
            Err(payload) => {
                let message = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown panic".into());
                (Cell::none(format!("panic: {message}")), true)
            }
        };
        emit(op, &cell);
        if sticky {
            process::exit(STICKY_EXIT_CODE);
        }
    }
    Ok(())
}

#[derive(Default)]
struct Args {
    json: Option<String>,
    filter: Option<String>,
    worker: String,
    cols: Vec<String>,
    peak_gbs: f64,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self::default();
        let mut it = env::args().skip(1);
        let value = |it: &mut std::iter::Skip<env::Args>, flag: &str| {
            it.next()
                .ok_or_else(|| candle_core::Error::msg(format!("missing value for {flag}")))
        };
        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--json" => args.json = Some(value(&mut it, &flag)?),
                "--filter" => args.filter = Some(value(&mut it, &flag)?),
                "--worker" => args.worker = value(&mut it, &flag)?,
                "--cols" => {
                    args.cols = value(&mut it, &flag)?.split(',').map(String::from).collect()
                }
                "--peak-gbs" => {
                    args.peak_gbs = value(&mut it, &flag)?
                        .parse()
                        .map_err(candle_core::Error::msg)?
                }
                _ => candle_core::bail!(
                    "unknown option {flag}; usage: fastpath_scoreboard [--json out.json] [--filter substr] [--peak-gbs N]"
                ),
            }
        }
        Ok(args)
    }
}

// best of a few streaming kernels into preallocated buffers
fn peak_bandwidth(dev: &Device) -> Result<f64> {
    let bytes = (PEAK_COPY_ELEMS * DType::F32.size_in_bytes()) as f64;
    let gbs = |ms: f64, traffic: f64| traffic / (ms * 1e-3) / 1e9;
    let src = Tensor::randn(0f32, 1f32, PEAK_COPY_ELEMS, dev)?;
    let rows = src.reshape((PEAK_GEMV_ROWS, ()))?;
    let w = rows.to_dtype(DType::BF16)?;
    let x = Tensor::randn(0f32, 1f32, (1, w.dim(1)?), dev)?.to_dtype(DType::BF16)?;
    let dst = src.zeros_like()?;
    let mut best = 0f64;
    for _ in 0..PEAK_PROBE_REPEATS {
        let copy = gbs(
            measure(dev, || dst.slice_set(&src, 0, 0).map(|_| dst.clone()))?,
            2.0 * bytes,
        );
        let reduce = gbs(measure(dev, || rows.sum_keepdim(1))?, bytes);
        let gemv = gbs(measure(dev, || x.matmul(&w.t()?))?, bytes / 2.0);
        eprintln!(
            "bandwidth probes GB/s: copy {copy:.1}, row-sum {reduce:.1}, bf16 gemv {gemv:.1}"
        );
        best = best.max(copy).max(reduce).max(gemv);
    }
    Ok(best)
}

fn bf16_gemm_ms(dev: &Device, (m, n, k): (usize, usize, usize)) -> Result<f64> {
    let w = Tensor::randn(0f32, WEIGHT_STD, (n, k), dev)?.to_dtype(DType::BF16)?;
    let x = Tensor::randn(0f32, 1f32, (1, m, k), dev)?.to_dtype(DType::BF16)?;
    let layer = unquant(w)?;
    measure(dev, || layer.forward(&x))
}

struct RowRun<'a> {
    fmt: &'a Format,
    cols: Vec<&'static Op>,
    peak_gbs: f64,
}

fn run_row(run: RowRun<'_>) -> HashMap<&'static str, Cell> {
    let mut cells = HashMap::new();
    let mut remaining = run.cols.clone();
    let exe = env::current_exe().expect("current exe");
    while !remaining.is_empty() {
        let ids = remaining.iter().map(|o| o.id).collect::<Vec<_>>().join(",");
        let spawned = Command::new(&exe)
            .args(["--worker", run.fmt.id, "--cols", &ids])
            .args(["--peak-gbs", &run.peak_gbs.to_string()])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn();
        let mut child = match spawned {
            Ok(child) => child,
            Err(e) => {
                for op in remaining.drain(..) {
                    cells.insert(op.id, Cell::none(format!("spawn failed: {e}")));
                }
                break;
            }
        };
        let stdout = child.stdout.take().expect("piped stdout");
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout)
                .lines()
                .map_while(std::result::Result::ok)
            {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let deadline = Instant::now() + ROW_TIMEOUT;
        let mut timed_out = false;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(left) {
                Ok(line) => {
                    let parts: Vec<&str> = line.splitn(5, '\t').collect();
                    if let [CELL_PREFIX, id, s, short, note] = parts.as_slice() {
                        if let Some(op) = find_op(id) {
                            cells.insert(
                                op.id,
                                Cell {
                                    status: Status::parse(s),
                                    short: short.to_string(),
                                    note: note.to_string(),
                                },
                            );
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    timed_out = true;
                    let _ = child.kill();
                    break;
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        let status = child.wait().map(|s| s.to_string()).unwrap_or_default();
        let before = remaining.len();
        remaining.retain(|op| !cells.contains_key(op.id));
        if remaining.len() == before && !remaining.is_empty() {
            let op = remaining.remove(0);
            let why = if timed_out {
                "timeout"
            } else {
                "worker crashed"
            };
            cells.insert(op.id, Cell::none(format!("{why} ({status})")));
        }
        if timed_out {
            for op in remaining.drain(..) {
                cells.insert(op.id, Cell::none("not run: row timeout"));
            }
        }
    }
    cells
}

fn print_table(rows: &[(&Format, HashMap<&'static str, Cell>)], cols: &[&Op]) {
    print!("{:<LABEL_WIDTH$}", "format");
    for op in cols {
        print!("{:>CELL_WIDTH$}", op.id);
    }
    println!();
    for (fmt, cells) in rows {
        print!("{:<LABEL_WIDTH$}", fmt.label);
        for op in cols {
            let text = cells
                .get(op.id)
                .map(|c| format!("{} {}", c.status.letter(), c.short))
                .unwrap_or_default();
            print!("{:>CELL_WIDTH$}", text.trim_end());
        }
        println!();
    }
    println!(
        "F fast (>={:.0}% bw | >={FAST_RATIO}x bf16)  P partial  B fallback  - none/error  W! wrong",
        FAST_BW * 100.0
    );
    for (fmt, cells) in rows {
        for op in cols {
            if let Some(cell) = cells.get(op.id) {
                if matches!(cell.status, Status::Wrong | Status::None) {
                    println!(
                        "  {}:{} [{}] {}",
                        fmt.id,
                        op.id,
                        cell.status.name(),
                        cell.note
                    );
                }
            }
        }
    }
}

fn main_inner() -> Result<()> {
    let args = Args::parse()?;
    if !args.worker.is_empty() {
        return worker_main(&args);
    }
    let selected = |fmt: &Format, op: &Op| {
        args.filter
            .as_deref()
            .is_none_or(|f| format!("{}:{}", fmt.id, op.id).contains(f))
    };
    let peak_gbs = if args.peak_gbs > 0.0 {
        args.peak_gbs
    } else {
        peak_bandwidth(&Device::new_cuda(0)?)?
    };
    eprintln!("peak bandwidth: {peak_gbs:.1} GB/s");
    let used_cols: Vec<&'static Op> = OPS
        .iter()
        .filter(|op| FORMATS.iter().any(|f| selected(f, op)))
        .collect();
    let mut rows = Vec::new();
    for fmt in FORMATS.iter() {
        let cols: Vec<&'static Op> = OPS.iter().filter(|op| selected(fmt, op)).collect();
        if cols.is_empty() {
            continue;
        }
        eprintln!("== {} ==", fmt.label);
        let cells = run_row(RowRun {
            fmt,
            cols,
            peak_gbs,
        });
        rows.push((fmt, cells));
    }
    print_table(&rows, &used_cols);
    if let Some(path) = &args.json {
        let mut cells_json = serde_json::Map::new();
        for (fmt, cells) in &rows {
            let mut row = serde_json::Map::new();
            for (id, cell) in cells {
                row.insert(
                    id.to_string(),
                    json!({"s": cell.status.name(), "n": cell.note}),
                );
            }
            cells_json.insert(fmt.id.to_string(), Value::Object(row));
        }
        let doc = json!({
            "title": format!("mistral.rs CUDA fast-path scoreboard (peak copy bw {peak_gbs:.0} GB/s)"),
            "rows": rows.iter().map(|(f, _)| json!({"id": f.id, "label": f.label, "group": f.group})).collect::<Vec<_>>(),
            "cols": used_cols.iter().map(|o| json!({"id": o.id, "label": o.label})).collect::<Vec<_>>(),
            "cells": cells_json,
        });
        std::fs::write(
            path,
            serde_json::to_string_pretty(&doc).map_err(candle_core::Error::msg)?,
        )?;
        eprintln!("wrote {path}");
    }
    Ok(())
}

fn main() {
    if let Err(e) = main_inner() {
        eprintln!("fastpath_scoreboard: {e}");
        process::exit(1);
    }
}
