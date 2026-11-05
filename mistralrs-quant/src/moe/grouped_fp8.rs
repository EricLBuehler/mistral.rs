//! E4M3-weight W8A16 GEMM (`kernels/moe/grouped_fp8.cu`) shared by every FP8-family format: routed
//! over `moe_align` blocks for MoE prefill, or over plain rows for dense layers.

use candle_core::cuda::cudarc::driver::sys::CUstream;
use candle_core::{CudaDevice, DType, Result, Shape, Storage, Tensor};
use half::{bf16, f16};

use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

/// Routes per expert below which the per-route GEMV re-reads less weight than the grouped tiles.
pub(crate) const GROUPED_MIN_ROUTES_PER_EXPERT: usize = 2;
const BLOCK_M_OPTIONS: [usize; 3] = [32, 64, 128];
const BLOCK_K_OPTIONS: [usize; 2] = [128, 64];
// mirrors kernels/moe/grouped_fp8.cu: pipeline depth, N tile, A row padding, and the route table
const STAGES: usize = 3;
const BLOCK_N: usize = 128;
const A_PAD: usize = 8;
const ROUTE_BYTES: usize = 4;
const CHUNK_BYTES: usize = 16;
const MMA_K: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fp8ScaleEncoding {
    F32 = 0,
    E8m0 = 1,
}

/// E4M3 weights `[E, N, K]` (or `[N, K]`) with scales laid out as `[E, ceil(N / bn), ceil(K / bk)]`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Fp8WeightView<'a> {
    pub weight: &'a Tensor,
    pub scales: &'a Tensor,
    pub encoding: Fp8ScaleEncoding,
    pub block: [usize; 2],
}

mod ffi {
    use super::CUstream;

    macro_rules! decl {
        ($name:ident) => {
            extern "C" {
                pub fn $name(
                    a: *const core::ffi::c_void,
                    w: *const u8,
                    scales: *const core::ffi::c_void,
                    sorted_ids: *const i32,
                    expert_ids: *const i32,
                    num_post_pad: *const i32,
                    out: *mut core::ffi::c_void,
                    n: i32,
                    k: i32,
                    em: i32,
                    num_valid: i32,
                    top_k: i32,
                    scale_block_n: i32,
                    scale_block_k: i32,
                    scale_kind: i32,
                    block_m: i32,
                    block_k: i32,
                    stream: CUstream,
                ) -> i32;
            }
        };
    }
    decl!(mistralrs_w8a16_grouped_bf16);
    decl!(mistralrs_w8a16_grouped_f16);
}

struct Geometry {
    experts: usize,
    n: usize,
    k: usize,
}

fn geometry(view: &Fp8WeightView<'_>) -> Option<Geometry> {
    let (experts, n, k) = match *view.weight.dims() {
        [n, k] => (1, n, k),
        [experts, n, k] => (experts, n, k),
        _ => return None,
    };
    let [bn, bk] = view.block;
    let scale_dtype = match view.encoding {
        Fp8ScaleEncoding::F32 => DType::F32,
        Fp8ScaleEncoding::E8m0 => DType::U8,
    };
    let epilogue_scale = bk >= k;
    let fits = [experts * n * k, n, k]
        .iter()
        .all(|&dim| dim <= i32::MAX as usize);
    (view.weight.dtype() == DType::F8E4M3
        && view.weight.is_contiguous()
        && view.scales.dtype() == scale_dtype
        && view.scales.is_contiguous()
        && view.scales.elem_count() == experts * n.div_ceil(bn.max(1)) * k.div_ceil(bk.max(1))
        && bn > 0
        && bk > 0
        && n > 0
        && k.is_multiple_of(CHUNK_BYTES)
        && (epilogue_scale || bk.is_multiple_of(MMA_K))
        && fits)
        .then_some(Geometry { experts, n, k })
}

fn device_supported(x: &Tensor, view: &Fp8WeightView<'_>) -> Option<CudaDevice> {
    let candle_core::Device::Cuda(dev) = x.device() else {
        return None;
    };
    (matches!(x.dtype(), DType::BF16 | DType::F16)
        && x.device().same_device(view.weight.device())
        && x.device().same_device(view.scales.device())
        && crate::cuda_device::cuda_device_info(dev).has_bf16_mma())
    .then(|| dev.clone())
}

#[derive(Clone, Copy, Debug)]
struct Tiles {
    m: usize,
    k: usize,
}

fn smem_bytes(tiles: Tiles, elem: usize) -> usize {
    STAGES * (tiles.m * (tiles.k + A_PAD) * elem + BLOCK_N * tiles.k) + tiles.m * ROUTE_BYTES
}

// Smallest M tile covering a group's rows, then the widest K tile whose pipeline fits the opt-in shared memory.
fn tiles_for(rows_per_group: usize, elem: usize, dev: &CudaDevice) -> Option<Tiles> {
    let smem = crate::cuda_device::cuda_device_info(dev).smem_optin;
    let m = BLOCK_M_OPTIONS
        .into_iter()
        .find(|&bm| rows_per_group <= bm)
        .unwrap_or(BLOCK_M_OPTIONS[BLOCK_M_OPTIONS.len() - 1]);
    BLOCK_M_OPTIONS
        .into_iter()
        .rev()
        .filter(|&bm| bm <= m)
        .flat_map(|m| BLOCK_K_OPTIONS.into_iter().map(move |k| Tiles { m, k }))
        .find(|&tiles| smem_bytes(tiles, elem) <= smem)
}

struct Routing {
    sorted_ids: u64,
    expert_ids: u64,
    num_post_pad: u64,
    em: usize,
    num_valid: usize,
    top_k: usize,
    tiles: Tiles,
}

struct Launch<'a> {
    x: &'a Tensor,
    view: Fp8WeightView<'a>,
    geometry: Geometry,
    routing: Routing,
}

fn launch(dev: &CudaDevice, args: Launch<'_>) -> Result<Option<Tensor>> {
    let Launch {
        x,
        view,
        geometry,
        routing,
    } = args;
    let stream = dev.cuda_stream();
    let (x_storage, x_layout) = x.storage_and_layout();
    let (w_storage, w_layout) = view.weight.storage_and_layout();
    let (s_storage, s_layout) = view.scales.storage_and_layout();
    let (Storage::Cuda(xs), Storage::Cuda(ws), Storage::Cuda(ss)) =
        (&*x_storage, &*w_storage, &*s_storage)
    else {
        candle_core::bail!("FP8 W8A16 operands must be CUDA tensors")
    };
    let (w_ptr, _w_guard) = slice_ptr_on_stream(
        ws.as_cuda_slice::<float8::F8E4M3>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (s_ptr, _s_guard) = match view.encoding {
        Fp8ScaleEncoding::F32 => {
            slice_ptr_on_stream(ss.as_cuda_slice::<f32>()?, s_layout.start_offset(), &stream)
        }
        Fp8ScaleEncoding::E8m0 => {
            slice_ptr_on_stream(ss.as_cuda_slice::<u8>()?, s_layout.start_offset(), &stream)
        }
    };
    let elements = routing.num_valid * geometry.n;
    let shape = Shape::from((routing.num_valid, geometry.n));
    macro_rules! run {
        ($ty:ty, $kernel:ident) => {{
            let (x_ptr, _x_guard) =
                slice_ptr_on_stream(xs.as_cuda_slice::<$ty>()?, x_layout.start_offset(), &stream);
            if x_ptr % CHUNK_BYTES as u64 != 0 || w_ptr % CHUNK_BYTES as u64 != 0 {
                return Ok(None);
            }
            let mut out = unsafe { dev.alloc::<$ty>(elements)? };
            let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
            let status = unsafe {
                ffi::$kernel(
                    x_ptr as *const core::ffi::c_void,
                    w_ptr as *const u8,
                    s_ptr as *const core::ffi::c_void,
                    routing.sorted_ids as *const i32,
                    routing.expert_ids as *const i32,
                    routing.num_post_pad as *const i32,
                    out_ptr as *mut core::ffi::c_void,
                    geometry.n as i32,
                    geometry.k as i32,
                    routing.em as i32,
                    routing.num_valid as i32,
                    routing.top_k as i32,
                    view.block[0].min(geometry.n) as i32,
                    view.block[1].min(geometry.k) as i32,
                    view.encoding as i32,
                    routing.tiles.m as i32,
                    routing.tiles.k as i32,
                    stream.cu_stream(),
                )
            };
            drop(out_guard);
            if status != 0 {
                candle_core::bail!("FP8 W8A16 grouped kernel failed with CUDA error {status}");
            }
            candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone())
        }};
    }
    let storage = match x.dtype() {
        DType::BF16 => run!(bf16, mistralrs_w8a16_grouped_bf16),
        DType::F16 => run!(f16, mistralrs_w8a16_grouped_f16),
        dtype => candle_core::bail!("FP8 W8A16 does not support {dtype:?} activations"),
    };
    Ok(Some(Tensor::from((Storage::Cuda(storage), shape))))
}

/// Routed `[T, topk, N]` output, or `None` when the routing is too sparse or the operands are unsupported.
pub(crate) fn gather(
    x: &Tensor,
    indices: &Tensor,
    view: Fp8WeightView<'_>,
) -> Result<Option<Tensor>> {
    let experts = match *view.weight.dims() {
        [experts, _, _] => experts,
        _ => return Ok(None),
    };
    if indices.elem_count() < GROUPED_MIN_ROUTES_PER_EXPERT * experts {
        return Ok(None);
    }
    gather_grouped(x, indices, view)
}

fn gather_grouped(x: &Tensor, indices: &Tensor, view: Fp8WeightView<'_>) -> Result<Option<Tensor>> {
    let Some(geometry) = geometry(&view) else {
        return Ok(None);
    };
    let Some(dev) = device_supported(x, &view) else {
        return Ok(None);
    };
    let Ok((tokens, topk)) = indices.dims2() else {
        return Ok(None);
    };
    let input_routes = match *x.dims() {
        [t, routes, k] if t == tokens && k == geometry.k && (routes == 1 || routes == topk) => {
            routes
        }
        _ => return Ok(None),
    };
    let routes = tokens * topk;
    if indices.dtype() != DType::U32 || routes == 0 || routes > i32::MAX as usize {
        return Ok(None);
    }
    let Some(tiles) = tiles_for(routes / geometry.experts, x.dtype().size_in_bytes(), &dev) else {
        return Ok(None);
    };
    let x = x.contiguous()?;
    let indices = indices.contiguous()?;
    let (i_storage, i_layout) = indices.storage_and_layout();
    let Storage::Cuda(is) = &*i_storage else {
        return Ok(None);
    };
    let ids = is
        .as_cuda_slice::<u32>()?
        .slice(i_layout.start_offset()..i_layout.start_offset() + routes);
    let (sorted, experts, post_pad, em) =
        super::cuda::moe_align(&ids, tokens, geometry.experts, topk, tiles.m as i32, &dev)?;
    let stream = dev.cuda_stream();
    let (sorted_ptr, _sorted_guard) = slice_ptr_on_stream(&sorted, 0, &stream);
    let (experts_ptr, _experts_guard) = slice_ptr_on_stream(&experts, 0, &stream);
    let (post_pad_ptr, _post_pad_guard) = slice_ptr_on_stream(&post_pad, 0, &stream);
    let n = geometry.n;
    let output = launch(
        &dev,
        Launch {
            x: &x,
            view,
            geometry,
            routing: Routing {
                sorted_ids: sorted_ptr,
                expert_ids: experts_ptr,
                num_post_pad: post_pad_ptr,
                em,
                num_valid: routes,
                top_k: if input_routes == 1 { topk } else { 1 },
                tiles,
            },
        },
    )?;
    output.map(|y| y.reshape((tokens, topk, n))).transpose()
}

/// Whether `dense` serves `[N, K]` weights for `dtype` activations on their device.
pub(crate) fn dense_supported(view: Fp8WeightView<'_>, dtype: DType) -> bool {
    let candle_core::Device::Cuda(dev) = view.weight.device() else {
        return false;
    };
    view.weight.rank() == 2
        && matches!(dtype, DType::BF16 | DType::F16)
        && view
            .weight
            .layout()
            .start_offset()
            .is_multiple_of(CHUNK_BYTES)
        && geometry(&view).is_some()
        && view.scales.device().same_device(view.weight.device())
        && crate::cuda_device::cuda_device_info(dev).has_bf16_mma()
}

/// Dense `[.., N]` output for `[N, K]` weights, or `None` when the operands are unsupported.
pub(crate) fn dense(x: &Tensor, view: Fp8WeightView<'_>) -> Result<Option<Tensor>> {
    let Some(geometry) = geometry(&view) else {
        return Ok(None);
    };
    let Some(dev) = device_supported(x, &view) else {
        return Ok(None);
    };
    if geometry.experts != 1 || view.weight.rank() != 2 || x.dims().last() != Some(&geometry.k) {
        return Ok(None);
    }
    let rows = x.elem_count() / geometry.k;
    if rows == 0 || rows > i32::MAX as usize {
        return Ok(None);
    }
    let Some(tiles) = tiles_for(rows, x.dtype().size_in_bytes(), &dev) else {
        return Ok(None);
    };
    let mut shape = x.dims().to_vec();
    *shape.last_mut().unwrap() = geometry.n;
    let x = x.contiguous()?;
    let output = launch(
        &dev,
        Launch {
            x: &x,
            view,
            geometry,
            routing: Routing {
                sorted_ids: 0,
                expert_ids: 0,
                num_post_pad: 0,
                em: rows,
                num_valid: rows,
                top_k: 1,
                tiles,
            },
        },
    )?;
    output.map(|y| y.reshape(shape)).transpose()
}
