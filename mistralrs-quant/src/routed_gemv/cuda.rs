use std::{ffi::c_void, sync::OnceLock};

use candle_core::{
    cuda::cudarc::driver::{sys::CUstream, CudaStream, SyncOnDrop},
    CudaDevice, CudaStorage, DType, Result, Storage, Tensor,
};
use half::{bf16, f16};

use super::{RoutedCodec, RoutedWeights};
use crate::{
    cuda_device::cuda_device_info,
    utils::{slice_ptr_mut_on_stream, CudaReadView},
    GluActivationType,
};

// Escape hatch: `0` sends every format back to its own gather kernels.
const ENABLE_ENV: &str = "MISTRALRS_ROUTED_GEMV";
// Past this many tokens per call, routes re-stream experts that a grouped GEMM would read once.
pub(crate) const MAX_TOKENS: usize = 16;
const GRID_Y_LIMIT: usize = 65_535;
const WIDE_ROW_UNITS: usize = 64;
const MEDIUM_ROW_UNITS: usize = 32;
const LANES_WIDE: i32 = 32;
const LANES_MEDIUM: i32 = 16;
const LANES_NARROW: i32 = 8;
const VECTOR_BYTES: usize = 16;
const NVFP4_WIDE_UNIT: usize = 32;
const NVFP4_NARROW_UNIT: usize = 16;
const MX_UNIT: usize = 32;
const FP8_UNIT: usize = 16;
const AFQ_UNIT: usize = 32;
const MX_GROUP: usize = 32;
const NVFP4_GROUP: usize = 16;

pub(super) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(ENABLE_ENV).map_or(true, |value| value != "0"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Plain = 0,
    Glu = 1,
    Reduce = 2,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Operand {
    data: *const u8,
    scales: *const c_void,
    extra: *const c_void,
    bias: *const c_void,
    data_expert: i64,
    data_row: i64,
    scale_expert: i64,
    scale_row: i64,
    extra_expert: i64,
    extra_row: i64,
    bias_expert: i64,
    plane: i64,
    group: i32,
    block_rows: i32,
}

impl Operand {
    fn empty() -> Self {
        Self {
            data: std::ptr::null(),
            scales: std::ptr::null(),
            extra: std::ptr::null(),
            bias: std::ptr::null(),
            data_expert: 0,
            data_row: 0,
            scale_expert: 0,
            scale_row: 0,
            extra_expert: 0,
            extra_row: 0,
            bias_expert: 0,
            plane: 0,
            group: 0,
            block_rows: 1,
        }
    }
}

#[repr(C)]
struct Launch {
    a: Operand,
    b: Operand,
    input: *const c_void,
    ids: *const u32,
    routing: *const f32,
    output: *mut c_void,
    input_stride: i64,
    tokens: i32,
    topk: i32,
    n: i32,
    k: i32,
    input_routes: i32,
    activation: i32,
    prefetch: i32,
}

unsafe extern "C" {
    fn routed_gemv_nvfp4(
        a: *const Launch,
        unit: i32,
        dtype: i32,
        mode: i32,
        lanes: i32,
        stream: CUstream,
    ) -> i32;
    fn routed_gemv_mxfp4(
        a: *const Launch,
        dtype: i32,
        mode: i32,
        lanes: i32,
        stream: CUstream,
    ) -> i32;
    fn routed_gemv_mxfp8(
        a: *const Launch,
        dtype: i32,
        mode: i32,
        lanes: i32,
        stream: CUstream,
    ) -> i32;
    fn routed_gemv_fp8(
        a: *const Launch,
        dtype: i32,
        mode: i32,
        lanes: i32,
        stream: CUstream,
    ) -> i32;
    fn routed_gemv_afq(
        a: *const Launch,
        bits: i32,
        dtype: i32,
        mode: i32,
        lanes: i32,
        stream: CUstream,
    ) -> i32;
}

fn prefetch(dev: &CudaDevice) -> bool {
    crate::cuda_policy::prefers_weight_prefetch(&cuda_device_info(dev))
}

fn lanes(units: usize) -> i32 {
    if units >= WIDE_ROW_UNITS {
        LANES_WIDE
    } else if units >= MEDIUM_ROW_UNITS {
        LANES_MEDIUM
    } else {
        LANES_NARROW
    }
}

fn dtype_code(dtype: DType) -> Option<i32> {
    match dtype {
        DType::F32 => Some(0),
        DType::F16 => Some(1),
        DType::BF16 => Some(2),
        _ => None,
    }
}

struct WeightViews<'a> {
    data: CudaReadView<'a>,
    scales: CudaReadView<'a>,
    extra: Option<CudaReadView<'a>>,
    bias: Option<CudaReadView<'a>>,
}

impl<'a> WeightViews<'a> {
    fn new(weights: &'a RoutedWeights) -> Result<Option<Self>> {
        if [
            Some(&weights.data),
            Some(&weights.scales),
            weights.extra.as_ref(),
            weights.bias.as_ref(),
        ]
        .into_iter()
        .flatten()
        .any(|tensor| {
            !tensor.is_contiguous()
                || !tensor.device().is_cuda()
                || !matches!(
                    tensor.dtype(),
                    DType::U8 | DType::U32 | DType::F8E4M3 | DType::F32 | DType::F16 | DType::BF16
                )
        }) {
            return Ok(None);
        }
        Ok(Some(Self {
            data: CudaReadView::new(&weights.data)?,
            scales: CudaReadView::new(&weights.scales)?,
            extra: CudaReadView::optional(weights.extra.as_ref())?,
            bias: CudaReadView::optional(weights.bias.as_ref())?,
        }))
    }
}

fn aligned(address: u64, bytes: usize) -> bool {
    address.is_multiple_of(bytes as u64)
}

// Widest vector a codec unit loads, which both the row stride and base address must honor.
fn load_alignment(unit_bytes: usize) -> usize {
    [VECTOR_BYTES, 8, 4]
        .into_iter()
        .find(|&align| unit_bytes.is_multiple_of(align))
        .unwrap_or(1)
}

struct Codec {
    unit: usize,
    unit_bytes: usize,
    row_bytes: usize,
}

fn codec(w: &RoutedWeights) -> Option<Codec> {
    let k = w.k;
    let (unit, unit_bytes, row_bytes) = match w.codec {
        RoutedCodec::Nvfp4 => {
            let unit = if k.is_multiple_of(NVFP4_WIDE_UNIT) {
                NVFP4_WIDE_UNIT
            } else {
                NVFP4_NARROW_UNIT
            };
            (unit, unit / 2, k / 2)
        }
        RoutedCodec::Mxfp4 => (MX_UNIT, MX_UNIT / 2, k / 2),
        RoutedCodec::Mxfp8 => (MX_UNIT, MX_UNIT, k),
        RoutedCodec::Fp8 { block_cols, .. } => {
            if block_cols < k && !block_cols.is_multiple_of(FP8_UNIT) {
                return None;
            }
            (FP8_UNIT, FP8_UNIT, k)
        }
        RoutedCodec::Afq { bits, group } => {
            if !group.is_multiple_of(AFQ_UNIT) {
                return None;
            }
            (AFQ_UNIT, AFQ_UNIT * bits / 8, k * bits / 8)
        }
    };
    (k > 0 && k.is_multiple_of(unit) && row_bytes.is_multiple_of(load_alignment(unit_bytes)))
        .then_some(Codec {
            unit,
            unit_bytes,
            row_bytes,
        })
}

fn operand<'a>(
    w: &RoutedWeights,
    views: &'a WeightViews<'_>,
    dtype: DType,
    stream: &'a CudaStream,
) -> Result<Option<(Operand, [SyncOnDrop<'a>; 4])>> {
    let Some(codec) = codec(w) else {
        return Ok(None);
    };
    let (n, k) = (w.n, w.k);
    let (data, data_guard) = views.data.read(stream)?;
    let (scales, scale_guard) = views.scales.read(stream)?;
    if !aligned(data, load_alignment(codec.unit_bytes)) {
        return Ok(None);
    }
    let (extra, extra_guard) = CudaReadView::read_optional(views.extra.as_ref(), stream)?;
    let (bias, bias_guard) = CudaReadView::read_optional(views.bias.as_ref(), stream)?;
    let bias_expert = match &w.bias {
        Some(bias) if bias.dtype() == dtype && bias.dims().last() == Some(&n) => {
            if bias.rank() > 1 {
                n as i64
            } else {
                0
            }
        }
        Some(_) => return Ok(None),
        None => 0,
    };
    let mut op = Operand {
        data: data as *const u8,
        scales: scales as *const c_void,
        extra: extra as *const c_void,
        bias: bias as *const c_void,
        data_expert: (n * codec.row_bytes) as i64,
        data_row: codec.row_bytes as i64,
        bias_expert,
        ..Operand::empty()
    };
    match w.codec {
        RoutedCodec::Nvfp4 => {
            if w.extra.is_none() {
                return Ok(None);
            }
            op.scale_row = (k / NVFP4_GROUP) as i64;
            op.scale_expert = op.scale_row * n as i64;
            op.extra_row = 1;
            op.extra_expert = n as i64;
        }
        RoutedCodec::Mxfp4 | RoutedCodec::Mxfp8 => {
            op.scale_row = (k / MX_GROUP) as i64;
            op.scale_expert = op.scale_row * n as i64;
        }
        RoutedCodec::Fp8 {
            block_rows,
            block_cols,
        } => {
            if w.scales.dtype() != DType::F32 {
                return Ok(None);
            }
            let cols = k.div_ceil(block_cols);
            op.scale_row = cols as i64;
            op.scale_expert = if w.scales.elem_count() == 1 {
                0
            } else {
                (n.div_ceil(block_rows) * cols) as i64
            };
            op.block_rows = i32::try_from(block_rows.min(n)).map_err(candle_core::Error::wrap)?;
            op.group = if cols == 1 {
                0
            } else {
                i32::try_from(block_cols).map_err(candle_core::Error::wrap)?
            };
        }
        RoutedCodec::Afq { group, .. } => {
            if w.scales.dtype() != dtype || w.extra.as_ref().is_none_or(|b| b.dtype() != dtype) {
                return Ok(None);
            }
            op.scale_row = (k / group) as i64;
            op.scale_expert = op.scale_row * n as i64;
            op.extra_row = op.scale_row;
            op.extra_expert = op.scale_expert;
            op.group = i32::try_from(group).map_err(candle_core::Error::wrap)?;
        }
    }
    Ok(Some((
        op,
        [data_guard, scale_guard, extra_guard, bias_guard],
    )))
}

struct RouteShape {
    tokens: usize,
    topk: usize,
    input_routes: usize,
}

fn route_shape(x: &Tensor, indices: &Tensor, k: usize) -> Result<Option<RouteShape>> {
    if indices.dtype() != DType::U32 || indices.rank() != 2 {
        return Ok(None);
    }
    let (tokens, topk) = indices.dims2()?;
    let input_routes = match *x.dims() {
        [t, xk] if t == tokens && xk == k => 1,
        [t, routes, xk] if t == tokens && xk == k && (routes == 1 || routes == topk) => routes,
        _ => return Ok(None),
    };
    Ok(
        (tokens > 0 && topk > 0 && tokens * topk <= GRID_Y_LIMIT).then_some(RouteShape {
            tokens,
            topk,
            input_routes,
        }),
    )
}

struct Run<'a> {
    mode: Mode,
    x: &'a Tensor,
    shape: RouteShape,
    indices: &'a Tensor,
    routing: Option<&'a Tensor>,
    a: &'a RoutedWeights,
    b: Option<&'a RoutedWeights>,
    activation: i32,
}

fn launch(run: Run<'_>) -> Result<Option<Tensor>> {
    let x = run.x;
    let Some(dtype) = dtype_code(x.dtype()) else {
        return Ok(None);
    };
    let Some(codec) = codec(run.a) else {
        return Ok(None);
    };
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let w = run.a;
    if !x.device().same_device(w.data.device()) {
        return Ok(None);
    }
    if !(w.k * x.dtype().size_in_bytes()).is_multiple_of(VECTOR_BYTES) {
        return Ok(None);
    }
    let mut x = x.contiguous()?;
    let input_aligned = {
        let view = CudaReadView::new(&x)?;
        let (pointer, _guard) = view.read(&stream)?;
        aligned(pointer, VECTOR_BYTES)
    };
    if !input_aligned {
        x = x.force_contiguous()?;
    }
    let indices = run.indices.contiguous()?;
    let routing = run
        .routing
        .map(|r| r.to_dtype(DType::F32)?.contiguous())
        .transpose()?;
    let Some(a_views) = WeightViews::new(w)? else {
        return Ok(None);
    };
    let b_views = match run.b {
        Some(b) => match WeightViews::new(b)? {
            Some(views) => Some(views),
            None => return Ok(None),
        },
        None => None,
    };
    let Some((a, _a_guards)) = operand(w, &a_views, x.dtype(), &stream)? else {
        return Ok(None);
    };
    let (b, _b_guards) = match run.b.zip(b_views.as_ref()) {
        Some((b, views)) => match operand(b, views, x.dtype(), &stream)? {
            Some((operand, guards)) => (operand, Some(guards)),
            None => return Ok(None),
        },
        None => (Operand::empty(), None),
    };
    let input_view = CudaReadView::new(&x)?;
    let ids_view = CudaReadView::new(&indices)?;
    let routing_view = CudaReadView::optional(routing.as_ref())?;
    let (input, _input_guard) = input_view.read(&stream)?;
    let (ids, _ids_guard) = ids_view.read(&stream)?;
    let (routing_ptr, _routing_guard) =
        CudaReadView::read_optional(routing_view.as_ref(), &stream)?;
    let RouteShape {
        tokens,
        topk,
        input_routes,
    } = run.shape;
    let out_rows = if run.mode == Mode::Reduce {
        tokens
    } else {
        tokens * topk
    };
    let lanes = lanes(w.k / codec.unit);
    let mut launch = Launch {
        a,
        b,
        input: input as *const c_void,
        ids: ids as *const u32,
        routing: routing_ptr as *const f32,
        output: std::ptr::null_mut(),
        input_stride: w.k as i64,
        tokens: tokens as i32,
        topk: topk as i32,
        n: w.n as i32,
        k: w.k as i32,
        input_routes: input_routes as i32,
        activation: run.activation,
        prefetch: i32::from(prefetch(dev)),
    };
    macro_rules! dispatch {
        ($ty:ty) => {{
            let mut out = unsafe { dev.alloc::<$ty>(out_rows * w.n)? };
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
            launch.output = out_ptr as *mut c_void;
            let mode = run.mode as i32;
            let cu = stream.cu_stream();
            let status = unsafe {
                match w.codec {
                    RoutedCodec::Nvfp4 => {
                        routed_gemv_nvfp4(&launch, codec.unit as i32, dtype, mode, lanes, cu)
                    }
                    RoutedCodec::Mxfp4 => routed_gemv_mxfp4(&launch, dtype, mode, lanes, cu),
                    RoutedCodec::Mxfp8 => routed_gemv_mxfp8(&launch, dtype, mode, lanes, cu),
                    RoutedCodec::Fp8 { .. } => routed_gemv_fp8(&launch, dtype, mode, lanes, cu),
                    RoutedCodec::Afq { bits, .. } => {
                        routed_gemv_afq(&launch, bits as i32, dtype, mode, lanes, cu)
                    }
                }
            };
            drop(out_guard);
            if status != 0 {
                candle_core::bail!("routed GEMV launch failed with CUDA error {status}");
            }
            CudaStorage::wrap_cuda_slice(out, dev.clone())
        }};
    }
    let storage = match x.dtype() {
        DType::F32 => dispatch!(f32),
        DType::F16 => dispatch!(f16),
        DType::BF16 => dispatch!(bf16),
        _ => unreachable!(),
    };
    Ok(Some(Tensor::from((
        Storage::Cuda(storage),
        (out_rows, w.n),
    ))))
}

pub(super) fn gather(x: &Tensor, indices: &Tensor, w: &RoutedWeights) -> Result<Option<Tensor>> {
    let Some(shape) = route_shape(x, indices, w.k)? else {
        return Ok(None);
    };
    if shape.tokens > MAX_TOKENS {
        return Ok(None);
    }
    let (tokens, topk) = (shape.tokens, shape.topk);
    let out = launch(Run {
        mode: Mode::Plain,
        x,
        shape,
        indices,
        routing: None,
        a: w,
        b: None,
        activation: 0,
    })?;
    out.map(|out| out.reshape((tokens, topk, w.n))).transpose()
}

pub(super) fn glu(
    x: &Tensor,
    indices: &Tensor,
    gate: &RoutedWeights,
    up: &RoutedWeights,
    activation: GluActivationType,
) -> Result<Option<Tensor>> {
    let Some(shape) = route_shape(x, indices, gate.k)? else {
        return Ok(None);
    };
    if shape.tokens > MAX_TOKENS || shape.input_routes != 1 {
        return Ok(None);
    }
    let (tokens, topk) = (shape.tokens, shape.topk);
    let out = launch(Run {
        mode: Mode::Glu,
        x,
        shape,
        indices,
        routing: None,
        a: gate,
        b: Some(up),
        activation: activation as i32,
    })?;
    out.map(|out| out.reshape((tokens, topk, gate.n)))
        .transpose()
}

pub(super) fn reduce(
    x: &Tensor,
    indices: &Tensor,
    routing: &Tensor,
    w: &RoutedWeights,
) -> Result<Option<Tensor>> {
    let Some(shape) = route_shape(x, indices, w.k)? else {
        return Ok(None);
    };
    if shape.tokens > MAX_TOKENS || routing.dims() != indices.dims() {
        return Ok(None);
    }
    launch(Run {
        mode: Mode::Reduce,
        x,
        shape,
        indices,
        routing: Some(routing),
        a: w,
        b: None,
        activation: 0,
    })
}
