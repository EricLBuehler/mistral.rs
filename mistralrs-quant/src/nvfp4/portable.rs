use std::ffi::c_void;

use candle_core::{
    cuda::cudarc::driver::{sys::CUstream, CudaSlice},
    CudaDevice, CudaStorage, DType, Result, Storage, Tensor,
};
use float8::F8E4M3;
use half::{bf16, f16};

use crate::{
    cuda_device::cuda_device_info,
    utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream},
    NVFP4_BLOCK_SIZE,
};

const TILE_N: usize = 128;
const TILE_K: usize = 64;
const MIN_SPLIT_K_TILES: usize = 4;
const MAX_SPLITS: usize = 16;
const ACTIVATION_ALIGNMENT: usize = 16;
pub(crate) const WEIGHT_ALIGNMENT: usize = 8;
pub(crate) const SCALE_ALIGNMENT: usize = 4;
const MAX_DIMENSION: usize = i32::MAX as usize;

#[repr(C)]
struct Args {
    input: *const c_void,
    weight: *const u8,
    scales: *const u8,
    global: *const f32,
    sorted_ids: *const i32,
    expert_ids: *const i32,
    num_padded: *const i32,
    output: *mut c_void,
    partial: *mut f32,
    rows: i32,
    n: i32,
    k: i32,
    input_stride: i32,
    splits: i32,
    k_tiles_per_split: i32,
}

unsafe extern "C" {
    fn mistralrs_nvfp4_w4a16_f16(
        args: *const Args,
        config: i32,
        row_tiles: i32,
        stream: CUstream,
    ) -> i32;
    fn mistralrs_nvfp4_w4a16_bf16(
        args: *const Args,
        config: i32,
        row_tiles: i32,
        stream: CUstream,
    ) -> i32;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tile {
    Rows16,
    Rows64,
    Rows128,
}

impl Tile {
    const ALL: [Self; 3] = [Self::Rows16, Self::Rows64, Self::Rows128];

    fn rows(self) -> usize {
        match self {
            Self::Rows16 => 16,
            Self::Rows64 => 64,
            Self::Rows128 => 128,
        }
    }

    // Small-row tiles stream weights, so they need several resident blocks per SM to hide latency.
    fn target_blocks_per_sm(self) -> usize {
        match self {
            Self::Rows16 => 8,
            Self::Rows64 | Self::Rows128 => 2,
        }
    }

    fn for_rows(rows: usize) -> Self {
        Self::ALL
            .into_iter()
            .find(|tile| rows <= tile.rows())
            .unwrap_or(Self::Rows128)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Weights<'a> {
    /// Packed E2M1 [N, K / 2] or [E, N, K / 2].
    pub weight: &'a Tensor,
    /// E4M3 block scales [N, K / 16] or [E, N, K / 16].
    pub scales: &'a Tensor,
    /// F32 per-row multipliers [N] or [E, N].
    pub global: &'a Tensor,
}

impl Weights<'_> {
    fn dims(&self) -> (usize, usize, usize) {
        let dims = self.weight.dims();
        let experts = if dims.len() == 3 { dims[0] } else { 1 };
        (experts, dims[dims.len() - 2], dims[dims.len() - 1] * 2)
    }
}

/// bf16/f16 mma.sync plus cp.async is all the W4A16 kernel needs.
pub(crate) fn supported(dev: &CudaDevice) -> bool {
    cuda_device_info(dev).has_bf16_mma()
}

struct Routing {
    sorted_ids: CudaSlice<i32>,
    expert_ids: CudaSlice<i32>,
    num_padded: CudaSlice<i32>,
}

struct Plan<'a> {
    x: &'a Tensor,
    weights: Weights<'a>,
    rows: usize,
    input_stride: usize,
    tile: Tile,
    row_tiles: usize,
    active_row_tiles: usize,
    routing: Option<Routing>,
}

/// x [rows, K] -> [rows, N].
pub(crate) fn matmul(x: &Tensor, weights: Weights<'_>) -> Result<Tensor> {
    let (rows, k) = x.dims2()?;
    let (_, n, weight_k) = weights.dims();
    if k != weight_k || weights.weight.rank() != 2 {
        candle_core::bail!("NVFP4 W4A16 matmul expects [rows, {weight_k}] against rank-2 weights");
    }
    let x = realigned(x, ACTIVATION_ALIGNMENT)?;
    let tile = Tile::for_rows(rows);
    let row_tiles = rows.div_ceil(tile.rows());
    let output = launch(Plan {
        x: &x,
        weights,
        rows,
        input_stride: 1,
        tile,
        row_tiles,
        active_row_tiles: row_tiles,
        routing: None,
    })?;
    output.reshape((rows, n))
}

/// x [tokens, 1 | topk, K] with expert ids [tokens, topk] -> [tokens, topk, N].
pub(crate) fn gather(x: &Tensor, indices: &Tensor, weights: Weights<'_>) -> Result<Tensor> {
    let (tokens, topk) = indices.dims2()?;
    let (experts, n, k) = weights.dims();
    let routes = tokens * topk;
    let input_rows = x.elem_count() / k;
    if weights.weight.rank() != 3
        || x.dim(candle_core::D::Minus1)? != k
        || input_rows == 0
        || routes % input_rows != 0
        || !(input_rows == tokens || input_rows == routes)
        || indices.dtype() != DType::U32
    {
        candle_core::bail!(
            "NVFP4 W4A16 gather got activations {:?} and U32 ids {:?} for weights {:?}",
            x.dims(),
            indices.dims(),
            weights.weight.dims()
        );
    }
    let x = realigned(&x.reshape((input_rows, k))?, ACTIVATION_ALIGNMENT)?;
    let dev = x.device().as_cuda_device()?;
    let active = routes.min(experts);
    let tile = Tile::for_rows(routes.div_ceil(active));
    let indices = indices.contiguous()?;
    let (sorted_ids, expert_ids, num_padded, em) = {
        let (storage, layout) = indices.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            candle_core::bail!("NVFP4 W4A16 gather ids must be CUDA tensors");
        };
        let start = layout.start_offset();
        crate::moe::cuda::moe_align(
            &storage.as_cuda_slice::<u32>()?.slice(start..start + routes),
            tokens,
            experts,
            topk,
            i32::try_from(tile.rows())?,
            dev,
        )?
    };
    let row_tiles = em.div_ceil(tile.rows());
    let output = launch(Plan {
        x: &x,
        weights,
        rows: routes,
        input_stride: routes / input_rows,
        tile,
        row_tiles,
        active_row_tiles: active,
        routing: Some(Routing {
            sorted_ids,
            expert_ids,
            num_padded,
        }),
    })?;
    output.reshape((tokens, topk, n))
}

/// Copies `tensor` into a fresh allocation when its view does not start on an `alignment`-byte boundary.
pub(crate) fn realigned(tensor: &Tensor, alignment: usize) -> Result<Tensor> {
    let tensor = crate::utils::contiguous_fp8(tensor)?;
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle_core::bail!("NVFP4 W4A16 operands must be CUDA tensors");
    };
    let start = layout.start_offset();
    if (start * tensor.dtype().size_in_bytes()).is_multiple_of(alignment) {
        return Ok(tensor.clone());
    }
    let range = start..start + tensor.elem_count();
    let dev = storage.device.clone();
    let stream = dev.cuda_stream();
    macro_rules! copy {
        ($ty:ty) => {
            CudaStorage::wrap_cuda_slice(
                stream
                    .clone_dtod(&storage.as_cuda_slice::<$ty>()?.slice(range.clone()))
                    .map_err(candle_core::Error::wrap)?,
                dev.clone(),
            )
        };
    }
    let copied = match tensor.dtype() {
        DType::U8 => copy!(u8),
        DType::F8E4M3 => copy!(F8E4M3),
        DType::BF16 => copy!(bf16),
        DType::F16 => copy!(f16),
        DType::F32 => copy!(f32),
        dtype => candle_core::bail!("NVFP4 W4A16 cannot realign {dtype:?} operands"),
    };
    Ok(Tensor::from((
        Storage::Cuda(copied),
        tensor.shape().clone(),
    )))
}

// Split K only when the output grid cannot fill the device; partial sums stay deterministic.
fn split_k(blocks: usize, k_tiles: usize, target: usize) -> (usize, usize) {
    let wanted = if blocks >= target {
        1
    } else {
        target
            .div_ceil(blocks)
            .min(MAX_SPLITS)
            .min((k_tiles / MIN_SPLIT_K_TILES).max(1))
    };
    let per_split = k_tiles.div_ceil(wanted);
    (k_tiles.div_ceil(per_split), per_split)
}

fn launch(plan: Plan<'_>) -> Result<Tensor> {
    let Plan {
        x,
        weights,
        rows,
        input_stride,
        tile,
        row_tiles,
        active_row_tiles,
        routing,
    } = plan;
    let (_, n, k) = weights.dims();
    if !matches!(x.dtype(), DType::BF16 | DType::F16)
        || weights.weight.dtype() != DType::U8
        || weights.scales.dtype() != DType::F8E4M3
        || weights.global.dtype() != DType::F32
        || !k.is_multiple_of(NVFP4_BLOCK_SIZE)
        || [rows, n, k, row_tiles].iter().any(|&d| d > MAX_DIMENSION)
    {
        candle_core::bail!(
            "NVFP4 W4A16 requires BF16/F16 activations, U8 codes, E4M3 scales, F32 globals and 32-bit dimensions"
        );
    }
    let dev = x.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let k_tiles = k.div_ceil(TILE_K);
    let (splits, k_tiles_per_split) = split_k(
        n.div_ceil(TILE_N) * active_row_tiles,
        k_tiles,
        cuda_device_info(dev).sms * tile.target_blocks_per_sm(),
    );
    let weight = weights.weight.contiguous()?;
    let scales = crate::utils::contiguous_fp8(weights.scales)?;
    let global = weights.global.contiguous()?;
    let (x_storage, x_layout) = x.storage_and_layout();
    let (w_storage, w_layout) = weight.storage_and_layout();
    let (s_storage, s_layout) = scales.storage_and_layout();
    let (g_storage, g_layout) = global.storage_and_layout();
    let (
        Storage::Cuda(x_storage),
        Storage::Cuda(w_storage),
        Storage::Cuda(s_storage),
        Storage::Cuda(g_storage),
    ) = (&*x_storage, &*w_storage, &*s_storage, &*g_storage)
    else {
        candle_core::bail!("NVFP4 W4A16 operands must be CUDA tensors");
    };
    let (weight_ptr, _weight_guard) = slice_ptr_on_stream(
        w_storage.as_cuda_slice::<u8>()?,
        w_layout.start_offset(),
        &stream,
    );
    let (scales_ptr, _scales_guard) = slice_ptr_on_stream(
        s_storage.as_cuda_slice::<F8E4M3>()?,
        s_layout.start_offset(),
        &stream,
    );
    let (global_ptr, _global_guard) = slice_ptr_on_stream(
        g_storage.as_cuda_slice::<f32>()?,
        g_layout.start_offset(),
        &stream,
    );
    let routing_ptrs = routing.as_ref().map(|routing| {
        [
            slice_ptr_on_stream(&routing.sorted_ids, 0, &stream),
            slice_ptr_on_stream(&routing.expert_ids, 0, &stream),
            slice_ptr_on_stream(&routing.num_padded, 0, &stream),
        ]
    });
    let routing_ptr = |slot: usize| {
        routing_ptrs
            .as_ref()
            .map_or(std::ptr::null(), |ptrs| ptrs[slot].0 as *const i32)
    };
    let mut partial = if splits > 1 {
        Some(unsafe { dev.alloc::<f32>(splits * rows * n)? })
    } else {
        None
    };
    let partial_ptr = partial
        .as_mut()
        .map(|partial| slice_ptr_mut_on_stream(partial, 0, &stream));
    macro_rules! run {
        ($ty:ty, $kernel:ident) => {{
            let (input_ptr, _input_guard) = slice_ptr_on_stream(
                x_storage.as_cuda_slice::<$ty>()?,
                x_layout.start_offset(),
                &stream,
            );
            let mut output = unsafe { dev.alloc::<$ty>(rows * n)? };
            let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let args = Args {
                input: input_ptr as *const c_void,
                weight: weight_ptr as *const u8,
                scales: scales_ptr as *const u8,
                global: global_ptr as *const f32,
                sorted_ids: routing_ptr(0),
                expert_ids: routing_ptr(1),
                num_padded: routing_ptr(2),
                output: output_ptr as *mut c_void,
                partial: partial_ptr
                    .as_ref()
                    .map_or(std::ptr::null_mut(), |(ptr, _)| *ptr as *mut f32),
                rows: rows as i32,
                n: n as i32,
                k: k as i32,
                input_stride: input_stride as i32,
                splits: splits as i32,
                k_tiles_per_split: k_tiles_per_split as i32,
            };
            let status =
                unsafe { $kernel(&args, tile as i32, row_tiles as i32, stream.cu_stream()) };
            if status != 0 {
                candle_core::bail!("NVFP4 W4A16 CUDA launch failed with error {status}");
            }
            drop(output_guard);
            CudaStorage::wrap_cuda_slice(output, dev.clone())
        }};
    }
    let output = match x.dtype() {
        DType::F16 => run!(f16, mistralrs_nvfp4_w4a16_f16),
        DType::BF16 => run!(bf16, mistralrs_nvfp4_w4a16_bf16),
        _ => unreachable!(),
    };
    drop(partial_ptr);
    drop(routing_ptrs);
    Ok(Tensor::from((Storage::Cuda(output), (rows, n))))
}
