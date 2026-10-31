//! CUDA fast path for GGUF matmul with BF16/F16/F32 activations.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

use candle_core::cuda::cudarc::driver::{CudaSlice, CudaStream, DevicePtrMut, SyncOnDrop};
use candle_core::{
    quantized::{GgmlDType, QTensor},
    CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

use super::{ffi, q8_1};
use crate::cuda_device::cuda_device_info;
use crate::{
    utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream},
    GluActivationType,
};

const PREFETCH_MIN_COLS: usize = 2048;
const DENSE_PAIR_MIN_COLUMNS: usize = 512;
const DENSE_PAIR_THREADS: usize = 256;
const DENSE_PAIR_PREFETCH_BYTES_PER_THREAD: usize = 32;
const PREFETCH_MIN_WEIGHT_BYTES: usize = 8 * 1024 * 1024;
const PREFETCH_LOW_BIT_MIN_COLS: usize = 4096;
const ROWS_PER_BLOCK: usize = 2;
const SINGLE_TOKEN_WARPS: usize = 8;
const FOUR_TOKEN_WARPS: usize = 4;
const EIGHT_TOKEN_WARPS: usize = 2;

fn use_single_row_prefetch(dtype: GgmlDType, columns: usize, weight_bytes: usize) -> bool {
    let block_bytes = dtype.type_size();
    let block_elements = dtype.block_size();
    // Low-density blocks need more rows in flight to hide unpacking latency.
    let enough_work = block_bytes * 2 >= block_elements
        && (block_bytes * 4 > block_elements * 3 || columns >= PREFETCH_LOW_BIT_MIN_COLS);
    columns >= PREFETCH_MIN_COLS && weight_bytes >= PREFETCH_MIN_WEIGHT_BYTES && enough_work
}

#[inline]
fn pad(p: usize, q: usize) -> usize {
    p.div_ceil(q) * q
}

fn output_shape(xs: &Tensor, nrows: usize) -> Shape {
    let mut out_dims = xs.dims().to_vec();
    let last = out_dims.len() - 1;
    out_dims[last] = nrows;
    Shape::from(out_dims)
}

/// Quant types supported by `mmvq_gguf.cu`.
pub fn supports(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

/// Maximum flattened batch handled by the CUDA launcher table.
pub const MMVQ_MAX_BATCH: usize = 8;

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
}

struct WorkspaceGuard<'a> {
    slot: MutexGuard<'static, WorkspaceSlot>,
    stream: &'a CudaStream,
}

impl WorkspaceGuard<'_> {
    fn ptr_mut(&mut self) -> (u64, SyncOnDrop<'_>) {
        self.slot.slice.device_ptr_mut(self.stream)
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct WorkspaceKey {
    device: candle_core::cuda::DeviceId,
    stream: usize,
    thread: ThreadId,
    capacity: usize,
}

type WsMap = Mutex<HashMap<WorkspaceKey, &'static Mutex<WorkspaceSlot>>>;

static WORKSPACE: OnceLock<WsMap> = OnceLock::new();

pub(super) fn prefers_weight_prefetch(dev: &CudaDevice) -> bool {
    crate::cuda_policy::prefers_weight_prefetch(&cuda_device_info(dev))
}

pub(crate) fn prefers_fused_pair(dev: &CudaDevice, tokens: usize, rows: [usize; 2]) -> bool {
    let info = cuda_device_info(dev);
    let warps = match tokens {
        1 => SINGLE_TOKEN_WARPS,
        2..=4 => FOUR_TOKEN_WARPS,
        _ => EIGHT_TOKEN_WARPS,
    };
    let blocks = rows.map(|rows| rows.div_ceil(ROWS_PER_BLOCK));
    crate::cuda_policy::fuse_underfilled_grids(&info, &blocks, info.warp_size * warps)
}

fn workspace_ensure<'a>(
    dev: &CudaDevice,
    bytes: usize,
    stream: &'a CudaStream,
) -> Result<WorkspaceGuard<'a>> {
    let map = WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let capacity = bytes.max(1).next_power_of_two();
    let key = WorkspaceKey {
        device: dev.id(),
        stream: stream.cu_stream() as usize,
        thread: std::thread::current().id(),
        capacity,
    };
    let workspace_mtx: &'static Mutex<WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(capacity)? };
                // CUDA graph replay requires process-stable workspace addresses.
                let leaked = Box::leak(Box::new(Mutex::new(WorkspaceSlot { slice })));
                guard.insert(key, leaked);
                leaked
            }
        }
    };
    Ok(WorkspaceGuard {
        slot: workspace_mtx.lock().unwrap(),
        stream,
    })
}

// Launcher dispatch by weight and output dtype.

type PlainLauncher = unsafe extern "C" fn(
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

type FusedGluLauncher = unsafe extern "C" fn(
    vx_gate: *const std::ffi::c_void,
    vx_up: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

type FusedQkvLauncher = unsafe extern "C" fn(
    vx_q: *const std::ffi::c_void,
    vx_k: *const std::ffi::c_void,
    vx_v: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    q_dst: *mut std::ffi::c_void,
    k_dst: *mut std::ffi::c_void,
    v_dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_q: i32,
    nrows_k: i32,
    nrows_v: i32,
    stride_col_y: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

fn plain_launcher_bf16(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_bf16_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_bf16_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_bf16_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_bf16_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_bf16_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_bf16_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_bf16_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_bf16_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_bf16_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_bf16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f16(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f16_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f16_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f16_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f16_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f16_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_f16_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_f16_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f16_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f16_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f32(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f32_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f32_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f32_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f32_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f32_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_f32_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_f32_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f32_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f32_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f32_plain,
        _ => return None,
    };
    Some(f)
}

fn fused_glu_launcher(input_ty: DType, dtype: GgmlDType) -> Option<FusedGluLauncher> {
    match (input_ty, dtype) {
        (DType::BF16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_bf16_fused_glu),
        (DType::BF16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_bf16_fused_glu),

        (DType::F16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f16_fused_glu),
        (DType::F16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f16_fused_glu),
        (DType::F16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f16_fused_glu),
        (DType::F16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f16_fused_glu),
        (DType::F16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f16_fused_glu),

        (DType::F32, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f32_fused_glu),
        (DType::F32, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f32_fused_glu),
        (DType::F32, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f32_fused_glu),
        (DType::F32, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f32_fused_glu),
        (DType::F32, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f32_fused_glu),
        _ => None,
    }
}

pub fn supports_fused_glu(input_ty: DType, dtype: GgmlDType) -> bool {
    fused_glu_launcher(input_ty, dtype).is_some()
}

fn fused_qkv_launcher(input_ty: DType, dtype: GgmlDType) -> Option<FusedQkvLauncher> {
    match (input_ty, dtype) {
        (DType::BF16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_bf16_fused_qkv),
        (DType::BF16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_bf16_fused_qkv),

        (DType::F16, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f16_fused_qkv),
        (DType::F16, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f16_fused_qkv),
        (DType::F16, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f16_fused_qkv),
        (DType::F16, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f16_fused_qkv),

        (DType::F32, GgmlDType::Q4_0) => Some(ffi::launch_mmvq_gguf_q4_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q4_1) => Some(ffi::launch_mmvq_gguf_q4_1_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5_0) => Some(ffi::launch_mmvq_gguf_q5_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5_1) => Some(ffi::launch_mmvq_gguf_q5_1_f32_fused_qkv),
        (DType::F32, GgmlDType::Q8_0) => Some(ffi::launch_mmvq_gguf_q8_0_f32_fused_qkv),
        (DType::F32, GgmlDType::Q2K) => Some(ffi::launch_mmvq_gguf_q2_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q3K) => Some(ffi::launch_mmvq_gguf_q3_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q4K) => Some(ffi::launch_mmvq_gguf_q4_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q5K) => Some(ffi::launch_mmvq_gguf_q5_k_f32_fused_qkv),
        (DType::F32, GgmlDType::Q6K) => Some(ffi::launch_mmvq_gguf_q6_k_f32_fused_qkv),
        _ => None,
    }
}

fn qkv_prefetch_launcher(input_ty: DType, dtype: GgmlDType) -> Option<FusedQkvLauncher> {
    Some(match (input_ty, dtype) {
        (DType::BF16, GgmlDType::Q4_0) => ffi::launch_mmvq_gguf_q4_0_bf16_qkv_prefetch,
        (DType::F16, GgmlDType::Q4_0) => ffi::launch_mmvq_gguf_q4_0_f16_qkv_prefetch,
        (DType::F32, GgmlDType::Q4_0) => ffi::launch_mmvq_gguf_q4_0_f32_qkv_prefetch,
        (DType::BF16, GgmlDType::Q4_1) => ffi::launch_mmvq_gguf_q4_1_bf16_qkv_prefetch,
        (DType::F16, GgmlDType::Q4_1) => ffi::launch_mmvq_gguf_q4_1_f16_qkv_prefetch,
        (DType::F32, GgmlDType::Q4_1) => ffi::launch_mmvq_gguf_q4_1_f32_qkv_prefetch,
        (DType::BF16, GgmlDType::Q5_0) => ffi::launch_mmvq_gguf_q5_0_bf16_qkv_prefetch,
        (DType::F16, GgmlDType::Q5_0) => ffi::launch_mmvq_gguf_q5_0_f16_qkv_prefetch,
        (DType::F32, GgmlDType::Q5_0) => ffi::launch_mmvq_gguf_q5_0_f32_qkv_prefetch,
        (DType::BF16, GgmlDType::Q5_1) => ffi::launch_mmvq_gguf_q5_1_bf16_qkv_prefetch,
        (DType::F16, GgmlDType::Q5_1) => ffi::launch_mmvq_gguf_q5_1_f16_qkv_prefetch,
        (DType::F32, GgmlDType::Q5_1) => ffi::launch_mmvq_gguf_q5_1_f32_qkv_prefetch,
        (DType::BF16, GgmlDType::Q8_0) => ffi::launch_mmvq_gguf_q8_0_bf16_qkv_prefetch,
        (DType::F16, GgmlDType::Q8_0) => ffi::launch_mmvq_gguf_q8_0_f16_qkv_prefetch,
        (DType::F32, GgmlDType::Q8_0) => ffi::launch_mmvq_gguf_q8_0_f32_qkv_prefetch,
        (DType::BF16, GgmlDType::Q4K) => ffi::launch_mmvq_gguf_q4k_bf16_qkv_prefetch,
        (DType::F16, GgmlDType::Q4K) => ffi::launch_mmvq_gguf_q4k_f16_qkv_prefetch,
        (DType::F32, GgmlDType::Q4K) => ffi::launch_mmvq_gguf_q4k_f32_qkv_prefetch,
        (DType::BF16, GgmlDType::Q5K) => ffi::launch_mmvq_gguf_q5k_bf16_qkv_prefetch,
        (DType::F16, GgmlDType::Q5K) => ffi::launch_mmvq_gguf_q5k_f16_qkv_prefetch,
        (DType::F32, GgmlDType::Q5K) => ffi::launch_mmvq_gguf_q5k_f32_qkv_prefetch,
        (DType::BF16, GgmlDType::Q6K) => ffi::launch_mmvq_gguf_q6k_bf16_qkv_prefetch,
        (DType::F16, GgmlDType::Q6K) => ffi::launch_mmvq_gguf_q6k_f16_qkv_prefetch,
        (DType::F32, GgmlDType::Q6K) => ffi::launch_mmvq_gguf_q6k_f32_qkv_prefetch,
        _ => return None,
    })
}

fn prefetch_launcher(dtype: GgmlDType, output: DType) -> Option<PlainLauncher> {
    Some(match (dtype, output) {
        (GgmlDType::Q8_0, DType::BF16) => ffi::launch_mmvq_gguf_q8_0_bf16_prefetch,
        (GgmlDType::Q8_0, DType::F16) => ffi::launch_mmvq_gguf_q8_0_f16_prefetch,
        (GgmlDType::Q8_0, DType::F32) => ffi::launch_mmvq_gguf_q8_0_f32_prefetch,
        (GgmlDType::Q4_0, DType::BF16) => ffi::launch_mmvq_gguf_q4_0_bf16_prefetch,
        (GgmlDType::Q4_0, DType::F16) => ffi::launch_mmvq_gguf_q4_0_f16_prefetch,
        (GgmlDType::Q4_0, DType::F32) => ffi::launch_mmvq_gguf_q4_0_f32_prefetch,
        (GgmlDType::Q4_1, DType::BF16) => ffi::launch_mmvq_gguf_q4_1_bf16_prefetch,
        (GgmlDType::Q4_1, DType::F16) => ffi::launch_mmvq_gguf_q4_1_f16_prefetch,
        (GgmlDType::Q4_1, DType::F32) => ffi::launch_mmvq_gguf_q4_1_f32_prefetch,
        (GgmlDType::Q5_0, DType::BF16) => ffi::launch_mmvq_gguf_q5_0_bf16_prefetch,
        (GgmlDType::Q5_0, DType::F16) => ffi::launch_mmvq_gguf_q5_0_f16_prefetch,
        (GgmlDType::Q5_0, DType::F32) => ffi::launch_mmvq_gguf_q5_0_f32_prefetch,
        (GgmlDType::Q5_1, DType::BF16) => ffi::launch_mmvq_gguf_q5_1_bf16_prefetch,
        (GgmlDType::Q5_1, DType::F16) => ffi::launch_mmvq_gguf_q5_1_f16_prefetch,
        (GgmlDType::Q5_1, DType::F32) => ffi::launch_mmvq_gguf_q5_1_f32_prefetch,
        (GgmlDType::Q4K, DType::BF16) => ffi::launch_mmvq_gguf_q4k_bf16_prefetch,
        (GgmlDType::Q4K, DType::F16) => ffi::launch_mmvq_gguf_q4k_f16_prefetch,
        (GgmlDType::Q4K, DType::F32) => ffi::launch_mmvq_gguf_q4k_f32_prefetch,
        (GgmlDType::Q5K, DType::BF16) => ffi::launch_mmvq_gguf_q5k_bf16_prefetch,
        (GgmlDType::Q5K, DType::F16) => ffi::launch_mmvq_gguf_q5k_f16_prefetch,
        (GgmlDType::Q5K, DType::F32) => ffi::launch_mmvq_gguf_q5k_f32_prefetch,
        (GgmlDType::Q6K, DType::BF16) => ffi::launch_mmvq_gguf_q6k_bf16_prefetch,
        (GgmlDType::Q6K, DType::F16) => ffi::launch_mmvq_gguf_q6k_f16_prefetch,
        (GgmlDType::Q6K, DType::F32) => ffi::launch_mmvq_gguf_q6k_f32_prefetch,
        _ => return None,
    })
}

/// Compute `w @ xs^T` where `w` is a Q8_1-quantizable GGUF weight tensor and
/// `xs` is a contiguous BF16 / F16 / F32 activation on the same CUDA device.
///
/// The product of the leading input dimensions must be in `1..=8`.
///
/// Output has the same leading dimensions as `xs` with the last axis replaced
/// by `w.shape().dims2()?.0` (nrows of the weight).
///
/// The output dtype matches the input dtype (BF16 -> BF16, F16 -> F16, F32 -> F32).
pub fn plain(w: &QTensor, xs: &Tensor) -> Result<Tensor> {
    let dtype = w.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmvq: unsupported quant dtype {dtype:?}");
    }
    let Device::Cuda(dev) = w.device() else {
        candle_core::bail!("fast_mmvq: weight must live on CUDA");
    };
    if !xs.device().same_device(&w.device()) {
        candle_core::bail!("fast_mmvq: input and weight are on different devices");
    }
    let (nrows, ncols) = w.shape().dims2()?;

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq: shape mismatch: weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("fast_mmvq: input dtype must be BF16, F16, or F32, got {input_ty:?}");
    }
    let prefetch = b_size == 1
        && use_single_row_prefetch(
            dtype,
            ncols,
            nrows * ncols / dtype.block_size() * dtype.type_size(),
        )
        && prefers_weight_prefetch(&dev);

    let stream = dev.cuda_stream();
    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmvq: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, q8_1::ROW_PADDING);
    let num_blocks_per_row = k_padded / q8_1::BLOCK_ELEMENTS;
    let dst_row_bytes = num_blocks_per_row * q8_1::BLOCK_BYTES;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / q8_1::BLOCK_ELEMENTS) as i32;
    let stride_col_dst = nrows as i32;
    let (weight_ptr, _weight_guard) = w.device_ptr_with_guard(&stream)?;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;

    match input_ty {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    let launcher = if prefetch {
                        prefetch_launcher(dtype, DType::BF16).expect("supported dtype")
                    } else {
                        plain_launcher_bf16(dtype).expect("supports() checked")
                    };
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    let launcher = if prefetch {
                        prefetch_launcher(dtype, DType::F16).expect("supported dtype")
                    } else {
                        plain_launcher_f16(dtype).expect("supports() checked")
                    };
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    let launcher = if prefetch {
                        prefetch_launcher(dtype, DType::F32).expect("supported dtype")
                    } else {
                        plain_launcher_f32(dtype).expect("supports() checked")
                    };
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        _ => unreachable!(),
    }
}

pub(crate) fn shared_lhs(weights: &[&QTensor], xs: &Tensor) -> Result<Vec<Tensor>> {
    let Device::Cuda(dev) = xs.device() else {
        candle_core::bail!("shared MMVQ input must live on CUDA");
    };
    let Some((&columns, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("shared MMVQ input must have at least one dimension");
    };
    let batch = batch_dims.iter().product::<usize>();
    if !(1..=MMVQ_MAX_BATCH).contains(&batch) {
        candle_core::bail!("shared MMVQ batch {batch} is unsupported");
    }
    let input_dtype = xs.dtype();
    if !matches!(input_dtype, DType::F32 | DType::F16 | DType::BF16) {
        candle_core::bail!("shared MMVQ input dtype {input_dtype:?} is unsupported");
    }
    for weight in weights {
        if !supports(weight.dtype()) {
            candle_core::bail!(
                "shared MMVQ weight dtype {:?} is unsupported",
                weight.dtype()
            );
        }
        if !xs.device().same_device(&weight.device()) || weight.shape().dims2()?.1 != columns {
            candle_core::bail!("shared MMVQ weights must match the input device and columns");
        }
    }
    let prefetch = batch == 1 && prefers_weight_prefetch(dev);
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let xs = xs.contiguous()?;
    let (storage, layout) = xs.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        unreachable!()
    };
    let padded_columns = pad(columns, q8_1::ROW_PADDING);
    let stride = padded_columns / q8_1::BLOCK_ELEMENTS;
    let mut workspace = workspace_ensure(dev, batch * stride * q8_1::BLOCK_BYTES, &stream)?;
    let (scratch, _scratch_guard) = workspace.ptr_mut();
    let scratch = scratch as *mut std::ffi::c_void;
    macro_rules! launch {
        ($type:ty, $quantize:ident, $plain:ident) => {{
            let input = storage.as_cuda_slice::<$type>()?;
            let (input, _input_guard) = slice_ptr_on_stream(input, layout.start_offset(), &stream);
            unsafe {
                ffi::$quantize(
                    input as *const _,
                    scratch,
                    columns as i32,
                    padded_columns as i32,
                    batch as i32,
                    stream_ptr,
                );
            }
            weights
                .iter()
                .map(|weight| {
                    let (rows, _) = weight.shape().dims2()?;
                    let format = weight.dtype();
                    let bytes = rows * columns / format.block_size() * format.type_size();
                    let launcher = if prefetch && use_single_row_prefetch(format, columns, bytes) {
                        prefetch_launcher(format, input_dtype).expect("supported dtype")
                    } else {
                        $plain(format).expect("supported dtype")
                    };
                    let mut output = unsafe { dev.alloc::<$type>(rows * batch)? };
                    {
                        let (weight, _weight_guard) = weight.device_ptr_with_guard(&stream)?;
                        let (output, _output_guard) =
                            slice_ptr_mut_on_stream(&mut output, 0, &stream);
                        unsafe {
                            launcher(
                                weight as *const _,
                                scratch,
                                output as *mut _,
                                columns as i32,
                                rows as i32,
                                stride as i32,
                                rows as i32,
                                batch as i32,
                                stream_ptr,
                            );
                        }
                    }
                    Ok(Tensor::from((
                        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                        output_shape(&xs, rows),
                    )))
                })
                .collect()
        }};
    }
    match input_dtype {
        DType::BF16 => launch!(
            half::bf16,
            launch_mmvq_gguf_quantize_q8_1_bf16,
            plain_launcher_bf16
        ),
        DType::F16 => launch!(
            half::f16,
            launch_mmvq_gguf_quantize_q8_1_f16,
            plain_launcher_f16
        ),
        DType::F32 => launch!(f32, launch_mmvq_gguf_quantize_q8_1_f32, plain_launcher_f32),
        _ => unreachable!(),
    }
}

fn dense_pair_prefetch_rows(dev: &CudaDevice) -> usize {
    let info = cuda_device_info(dev);
    info.sms * (info.threads_per_sm / DENSE_PAIR_THREADS)
}

pub(crate) fn try_dense_pair(
    quant_weight: &QTensor,
    dense_weight: &Tensor,
    xs: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    let Ok((quant_rows, columns)) = quant_weight.shape().dims2() else {
        return Ok(None);
    };
    let Ok((dense_rows, dense_columns)) = dense_weight.dims2() else {
        return Ok(None);
    };
    let format = quant_weight.dtype();
    let Some(&input_columns) = xs.dims().last() else {
        return Ok(None);
    };
    if !supports(format)
        || columns < DENSE_PAIR_MIN_COLUMNS
        || !columns.is_multiple_of(format.block_size())
        || columns != dense_columns
        || columns != input_columns
        || xs.elem_count() != columns
        || quant_rows == 0
        || dense_rows == 0
        // The established two-row MMVQ core reads both rows before masking output.
        || !quant_rows.is_multiple_of(2)
        || quant_rows.checked_mul(columns).is_none_or(|n| n > i32::MAX as usize)
        || dense_rows.checked_mul(columns).is_none_or(|n| n > i32::MAX as usize)
        || !dense_weight.is_contiguous()
        || !dense_weight.layout().start_offset().is_multiple_of(2)
        || (xs.is_contiguous() && !xs.layout().start_offset().is_multiple_of(2))
        || !xs.device().same_device(&quant_weight.device())
        || !xs.device().same_device(dense_weight.device())
        || xs.dtype() != dense_weight.dtype()
        || !crate::gemv::should_use_gemv(xs, dense_weight)
    {
        return Ok(None);
    }
    let dev = xs.device().as_cuda_device()?;
    let weight_bytes = quant_rows * columns / format.block_size() * format.type_size();
    let prefetch_device = prefers_weight_prefetch(dev);
    if prefetch_device && use_single_row_prefetch(format, columns, weight_bytes) {
        return Ok(None);
    }
    let row_bytes = weight_bytes / quant_rows;
    let prefetch = prefetch_device
        && row_bytes >= DENSE_PAIR_PREFETCH_BYTES_PER_THREAD * DENSE_PAIR_THREADS
        && quant_rows >= dense_pair_prefetch_rows(dev);
    let xs = xs.contiguous()?;
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let (input_storage, input_layout) = xs.storage_and_layout();
    let (dense_storage, dense_layout) = dense_weight.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(dense_storage)) =
        (&*input_storage, &*dense_storage)
    else {
        unreachable!()
    };
    let padded_columns = pad(columns, q8_1::ROW_PADDING);
    let mut workspace = workspace_ensure(
        dev,
        padded_columns / q8_1::BLOCK_ELEMENTS * q8_1::BLOCK_BYTES,
        &stream,
    )?;
    let (scratch, _scratch_guard) = workspace.ptr_mut();
    let (quant, _quant_guard) = quant_weight.device_ptr_with_guard(&stream)?;
    macro_rules! launch {
        ($type:ty, $dtype:expr, $quantize:ident) => {{
            let (input, _input_guard) = slice_ptr_on_stream(
                input_storage.as_cuda_slice::<$type>()?,
                input_layout.start_offset(),
                &stream,
            );
            let (dense, _dense_guard) = slice_ptr_on_stream(
                dense_storage.as_cuda_slice::<$type>()?,
                dense_layout.start_offset(),
                &stream,
            );
            let mut quant_output = unsafe { dev.alloc::<$type>(quant_rows)? };
            let mut dense_output = unsafe { dev.alloc::<$type>(dense_rows)? };
            {
                let (quant_out, _quant_out_guard) =
                    slice_ptr_mut_on_stream(&mut quant_output, 0, &stream);
                let (dense_out, _dense_out_guard) =
                    slice_ptr_mut_on_stream(&mut dense_output, 0, &stream);
                unsafe {
                    ffi::$quantize(
                        input as *const _,
                        scratch as *mut _,
                        columns as i32,
                        padded_columns as i32,
                        1,
                        stream_ptr,
                    );
                    ffi::launch_mmvq_gguf_dense_pair(
                        ffi::MixedMmvqArgs {
                            quant_weight: quant as *const _,
                            quant_input: scratch as *const _,
                            dense_weight: dense as *const _,
                            input: input as *const _,
                            quant_output: quant_out as *mut _,
                            dense_output: dense_out as *mut _,
                            columns: columns as i32,
                            quant_rows: quant_rows as i32,
                            dense_rows: dense_rows as i32,
                            prefetch: i32::from(prefetch),
                        },
                        super::ggml_dtype_to_uqff_code(format) as i32,
                        $dtype,
                        stream_ptr,
                    );
                }
            }
            Ok(Some((
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(quant_output, dev.clone())),
                    output_shape(&xs, quant_rows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(dense_output, dev.clone())),
                    output_shape(&xs, dense_rows),
                )),
            )))
        }};
    }
    match xs.dtype() {
        DType::F16 => launch!(half::f16, 0, launch_mmvq_gguf_quantize_q8_1_f16),
        DType::BF16 => launch!(half::bf16, 1, launch_mmvq_gguf_quantize_q8_1_bf16),
        DType::F32 => launch!(f32, 2, launch_mmvq_gguf_quantize_q8_1_f32),
        _ => Ok(None),
    }
}

pub fn fused_glu(
    gate_w: &QTensor,
    up_w: &QTensor,
    xs: &Tensor,
    activation: GluActivationType,
) -> Result<Tensor> {
    let dtype = gate_w.dtype();
    if dtype != up_w.dtype() {
        candle_core::bail!(
            "fast_mmvq fused_glu: gate/up dtype mismatch {:?} vs {:?}",
            dtype,
            up_w.dtype()
        );
    }
    let Some(launcher) = fused_glu_launcher(xs.dtype(), dtype) else {
        candle_core::bail!("fast_mmvq fused_glu: unsupported dtype combination");
    };

    let Device::Cuda(dev) = gate_w.device() else {
        candle_core::bail!("fast_mmvq fused_glu: gate weight must live on CUDA");
    };
    let Device::Cuda(up_dev) = up_w.device() else {
        candle_core::bail!("fast_mmvq fused_glu: up weight must live on CUDA");
    };
    if dev.id() != up_dev.id() {
        candle_core::bail!("fast_mmvq fused_glu: gate/up weights are on different CUDA devices");
    }
    if !xs.device().same_device(&gate_w.device()) {
        candle_core::bail!("fast_mmvq fused_glu: input and weights are on different devices");
    }

    let (nrows, ncols) = gate_w.shape().dims2()?;
    let (up_nrows, up_ncols) = up_w.shape().dims2()?;
    if (nrows, ncols) != (up_nrows, up_ncols) {
        candle_core::bail!(
            "fast_mmvq fused_glu: gate/up shape mismatch [{nrows}, {ncols}] vs [{up_nrows}, {up_ncols}]"
        );
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq fused_glu: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq fused_glu: shape mismatch: weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq fused_glu: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmvq fused_glu: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmvq fused_glu: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, q8_1::ROW_PADDING);
    let num_blocks_per_row = k_padded / q8_1::BLOCK_ELEMENTS;
    let dst_row_bytes = num_blocks_per_row * q8_1::BLOCK_BYTES;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / q8_1::BLOCK_ELEMENTS) as i32;
    let stride_col_dst = nrows as i32;
    let (gate_ptr, _gate_guard) = gate_w.device_ptr_with_guard(&stream)?;
    let (up_ptr, _up_guard) = up_w.device_ptr_with_guard(&stream)?;
    let gate_ptr = gate_ptr as *const std::ffi::c_void;
    let up_ptr = up_ptr as *const std::ffi::c_void;
    let activation = activation as i32;

    match input_ty {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        gate_ptr,
                        up_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        activation,
                        stream_ptr,
                    );
                }
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(out_storage),
                output_shape(&xs, nrows),
            )))
        }
        _ => unreachable!(),
    }
}

/// Compute Q, K, and V matvecs with one input quantization pass and one MMVQ
/// kernel. The result tensors match the unfused `plain` outputs for each
/// projection and preserve the input dtype.
pub fn fused_qkv(
    q_w: &QTensor,
    k_w: &QTensor,
    v_w: &QTensor,
    xs: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    fused_qkv_optional(q_w, k_w, Some(v_w), xs)
}

pub fn fused_pair(left: &QTensor, right: &QTensor, xs: &Tensor) -> Result<(Tensor, Tensor)> {
    let (left, right, _) = fused_qkv_optional(left, right, None, xs)?;
    Ok((left, right))
}

fn fused_qkv_optional(
    q_w: &QTensor,
    k_w: &QTensor,
    v_w: Option<&QTensor>,
    xs: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let has_v = v_w.is_some();
    let v_w = v_w.unwrap_or(q_w);
    let dtype = q_w.dtype();
    if dtype != k_w.dtype() || dtype != v_w.dtype() {
        candle_core::bail!(
            "fast_mmvq fused_qkv: q/k/v dtype mismatch {:?}, {:?}, {:?}",
            dtype,
            k_w.dtype(),
            v_w.dtype()
        );
    }
    let Some(launcher) = fused_qkv_launcher(xs.dtype(), dtype) else {
        candle_core::bail!("fast_mmvq fused_qkv: unsupported dtype combination");
    };

    let Device::Cuda(dev) = q_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: q weight must live on CUDA");
    };
    let Device::Cuda(k_dev) = k_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: k weight must live on CUDA");
    };
    let Device::Cuda(v_dev) = v_w.device() else {
        candle_core::bail!("fast_mmvq fused_qkv: v weight must live on CUDA");
    };
    if dev.id() != k_dev.id() || dev.id() != v_dev.id() {
        candle_core::bail!("fast_mmvq fused_qkv: q/k/v weights are on different CUDA devices");
    }
    if !xs.device().same_device(&q_w.device()) {
        candle_core::bail!("fast_mmvq fused_qkv: input and weights are on different devices");
    }

    let (q_nrows, ncols) = q_w.shape().dims2()?;
    let (k_nrows, k_ncols) = k_w.shape().dims2()?;
    let (v_nrows, v_ncols) = v_w.shape().dims2()?;
    let v_nrows = if has_v { v_nrows } else { 0 };
    if ncols != k_ncols || ncols != v_ncols {
        candle_core::bail!(
            "fast_mmvq fused_qkv: q/k/v ncols mismatch {ncols}, {k_ncols}, {v_ncols}"
        );
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        candle_core::bail!("fast_mmvq fused_qkv: input must have at least one dimension");
    };
    let b_size = batch_dims.iter().product::<usize>();
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq fused_qkv: shape mismatch: weight ncols {ncols} vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq fused_qkv: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmvq fused_qkv: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let weight_bytes =
        (q_nrows + k_nrows + v_nrows) * ncols / dtype.block_size() * dtype.type_size();
    let launcher = if b_size == 1
        && use_single_row_prefetch(dtype, ncols, weight_bytes)
        && prefers_weight_prefetch(&dev)
    {
        qkv_prefetch_launcher(input_ty, dtype).expect("supported dtype")
    } else {
        launcher
    };

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmvq fused_qkv: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, q8_1::ROW_PADDING);
    let num_blocks_per_row = k_padded / q8_1::BLOCK_ELEMENTS;
    let dst_row_bytes = num_blocks_per_row * q8_1::BLOCK_BYTES;
    let scratch_bytes = b_size * dst_row_bytes;

    let mut workspace = workspace_ensure(&dev, scratch_bytes, &stream)?;
    let (scratch_ptr, _scratch_guard) = workspace.ptr_mut();
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;
    let stride_col_y = (k_padded / q8_1::BLOCK_ELEMENTS) as i32;
    let (q_ptr, _q_guard) = q_w.device_ptr_with_guard(&stream)?;
    let (k_ptr, _k_guard) = k_w.device_ptr_with_guard(&stream)?;
    let (v_ptr, _v_guard) = v_w.device_ptr_with_guard(&stream)?;
    let q_ptr = q_ptr as *const std::ffi::c_void;
    let k_ptr = k_ptr as *const std::ffi::c_void;
    let v_ptr = v_ptr as *const std::ffi::c_void;

    match input_ty {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut q_out = unsafe { dev.alloc::<half::bf16>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<half::bf16>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<half::bf16>((v_nrows * b_size).max(1))? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut q_out = unsafe { dev.alloc::<half::f16>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<half::f16>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<half::f16>((v_nrows * b_size).max(1))? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut q_out = unsafe { dev.alloc::<f32>(q_nrows * b_size)? };
            let mut k_out = unsafe { dev.alloc::<f32>(k_nrows * b_size)? };
            let mut v_out = unsafe { dev.alloc::<f32>((v_nrows * b_size).max(1))? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr_on_stream(slice, xs_offset, &stream);
                let (q_out_ptr, _q_out_guard) = slice_ptr_mut_on_stream(&mut q_out, 0, &stream);
                let (k_out_ptr, _k_out_guard) = slice_ptr_mut_on_stream(&mut k_out, 0, &stream);
                let (v_out_ptr, _v_out_guard) = slice_ptr_mut_on_stream(&mut v_out, 0, &stream);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    launcher(
                        q_ptr,
                        k_ptr,
                        v_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        q_out_ptr as *mut std::ffi::c_void,
                        k_out_ptr as *mut std::ffi::c_void,
                        v_out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        q_nrows as i32,
                        k_nrows as i32,
                        v_nrows as i32,
                        stride_col_y,
                        b_size as i32,
                        stream_ptr,
                    );
                }
            }

            Ok((
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(q_out, dev.clone())),
                    output_shape(&xs, q_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(k_out, dev.clone())),
                    output_shape(&xs, k_nrows),
                )),
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(v_out, dev.clone())),
                    output_shape(&xs, v_nrows),
                )),
            ))
        }
        _ => unreachable!(),
    }
}
