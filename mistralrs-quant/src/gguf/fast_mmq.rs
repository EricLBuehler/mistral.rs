//! CUDA fast path for GGUF tiled matmul (prompt/prefill phase).
//! Handles batch > 8 (complement to fast_mmvq which handles batch 1-8).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::cuda::cudarc::driver::{CudaSlice, DevicePtr};
use candle_core::{
    quantized::{GgmlDType, QTensor},
    CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

use super::ffi;
use crate::utils::slice_ptr;

const QK8_1: usize = 32;
const BLOCK_Q8_1_MMQ_SIZE: usize = 4 * QK8_1 + 4 * 4; // 128 qs + 16 scale bytes = 144
const MATRIX_ROW_PADDING: usize = 512;

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

/// Quant types supported by MMQ kernels (same as MMVQ).
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

/// qk (block quantization size) per dtype.
fn qk_for(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => 32,
        GgmlDType::Q2K | GgmlDType::Q3K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K => 256,
        _ => unreachable!(),
    }
}

// ds_layout mapping: which Q8_1_mmq scale layout to use per weight type.
// D4 = scale only, DS4 = scale+partial_sum, D2S6 = 2 scales + 6 partial_sums
enum DsLayout {
    D4,
    DS4,
    D2S6,
}

fn ds_layout_for(dtype: GgmlDType) -> DsLayout {
    match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 => DsLayout::DS4,
        GgmlDType::Q5_0 => DsLayout::D4,
        GgmlDType::Q5_1 => DsLayout::DS4,
        GgmlDType::Q8_0 => DsLayout::D4,
        GgmlDType::Q2K => DsLayout::D2S6,
        GgmlDType::Q3K => DsLayout::D4,
        GgmlDType::Q4K | GgmlDType::Q5K => DsLayout::DS4,
        GgmlDType::Q6K => DsLayout::D4,
        _ => unreachable!(),
    }
}

type QuantizeLauncher = unsafe extern "C" fn(
    x: *const std::ffi::c_void,
    ids: *const i32,
    vy: *mut std::ffi::c_void,
    type_x: i32,
    ne00: i64,
    s01: i64,
    s02: i64,
    s03: i64,
    ne0: i64,
    ne1: i64,
    ne2: i64,
    ne3: i64,
    stream: *mut std::ffi::c_void,
);

fn quantize_launcher(layout: DsLayout) -> QuantizeLauncher {
    match layout {
        DsLayout::D4 => ffi::launch_mmq_quantize_q8_1_D4,
        DsLayout::DS4 => ffi::launch_mmq_quantize_q8_1_DS4,
        DsLayout::D2S6 => ffi::launch_mmq_quantize_q8_1_D2S6,
    }
}

type MmqLauncher = unsafe extern "C" fn(
    tmp_fixup: *mut std::ffi::c_void,
    x: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i64,
    nrows_x: i64,
    ncols_y: i64,
    stride_row_x: i64,
    stride_col_dst: i64,
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
    stream: *mut std::ffi::c_void,
);

fn mmq_launcher(dtype: GgmlDType) -> Option<MmqLauncher> {
    let f: MmqLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmq_gguf_q4_0,
        GgmlDType::Q4_1 => ffi::launch_mmq_gguf_q4_1,
        GgmlDType::Q5_0 => ffi::launch_mmq_gguf_q5_0,
        GgmlDType::Q5_1 => ffi::launch_mmq_gguf_q5_1,
        GgmlDType::Q8_0 => ffi::launch_mmq_gguf_q8_0,
        GgmlDType::Q2K => ffi::launch_mmq_gguf_q2_k,
        GgmlDType::Q3K => ffi::launch_mmq_gguf_q3_k,
        GgmlDType::Q4K => ffi::launch_mmq_gguf_q4_k,
        GgmlDType::Q5K => ffi::launch_mmq_gguf_q5_k,
        GgmlDType::Q6K => ffi::launch_mmq_gguf_q6_k,
        _ => return None,
    };
    Some(f)
}

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

type WsMap = Mutex<HashMap<candle_core::cuda::DeviceId, &'static Mutex<WorkspaceSlot>>>;

static MMQ_WORKSPACE: OnceLock<WsMap> = OnceLock::new();
static FIXUP_WORKSPACE: OnceLock<WsMap> = OnceLock::new();

#[derive(Clone, Copy)]
struct DeviceInfo {
    cc: i32,
    nsm: i32,
    smpbo: i64,
    warp_size: i32,
}

static DEVICE_INFO: OnceLock<Mutex<HashMap<candle_core::cuda::DeviceId, DeviceInfo>>> =
    OnceLock::new();

fn get_device_info(dev: &CudaDevice) -> DeviceInfo {
    use candle_core::cuda::cudarc::driver::{result, sys};
    let map = DEVICE_INFO.get_or_init(|| Mutex::new(HashMap::new()));
    let key = dev.id();
    let mut guard = map.lock().unwrap();
    if let Some(info) = guard.get(&key) {
        return *info;
    }
    let cu_device = dev.cuda_stream().context().cu_device();
    let major = unsafe {
        result::device::get_attribute(cu_device, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
    }.unwrap_or(8);
    let minor = unsafe {
        result::device::get_attribute(cu_device, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
    }.unwrap_or(0);
    let nsm = unsafe {
        result::device::get_attribute(cu_device, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    }.unwrap_or(1);
    let smpbo = unsafe {
        result::device::get_attribute(cu_device, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    }.unwrap_or(49152);
    let warp_size = unsafe {
        result::device::get_attribute(cu_device, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    }.unwrap_or(32);
    let info = DeviceInfo {
        cc: major * 100 + minor * 10,
        nsm,
        smpbo: smpbo as i64,
        warp_size,
    };
    guard.insert(key, info);
    info
}

fn workspace_ensure(
    ws: &'static OnceLock<WsMap>,
    dev: &CudaDevice,
    bytes: usize,
) -> Result<(u64, std::sync::MutexGuard<'static, WorkspaceSlot>)> {
    let map = ws.get_or_init(|| Mutex::new(HashMap::new()));
    let device_key = dev.id();
    let device_mtx: &'static Mutex<WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&device_key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(bytes.max(1))? };
                let leaked = Box::leak(Box::new(Mutex::new(WorkspaceSlot {
                    slice,
                    cap: bytes.max(1),
                })));
                guard.insert(device_key, leaked);
                leaked
            }
        }
    };
    let mut slot = device_mtx.lock().unwrap();
    if slot.cap < bytes {
        slot.slice = unsafe { dev.alloc::<u8>(bytes)? };
        slot.cap = bytes;
    }
    let ptr = slot.slice.device_ptr(slot.slice.stream()).0;
    Ok((ptr, slot))
}

/// Compute `w @ xs^T` where `w` is a GGUF-quantized weight tensor and
/// `xs` is a contiguous BF16 / F16 / F32 activation on the same CUDA device.
///
/// This is the prompt/prefill kernel path for batch > 8.
/// Output is always computed in f32 internally, then cast to the input dtype.
pub fn plain(w: &QTensor, xs: &Tensor) -> Result<Tensor> {
    let dtype = w.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmq: unsupported quant dtype {dtype:?}");
    }
    let Device::Cuda(dev) = w.device() else {
        candle_core::bail!("fast_mmq: weight must live on CUDA");
    };
    let (nrows, ncols) = w.shape().dims2()?;

    let (b_size, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        other => candle_core::bail!("fast_mmq: unexpected input rank {other:?}"),
    };
    if k != ncols {
        candle_core::bail!(
            "fast_mmq: shape mismatch — weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 {
        candle_core::bail!("fast_mmq: batch size must be > 0");
    }

    let qk = qk_for(dtype);
    if k % qk != 0 {
        candle_core::bail!("fast_mmq: k={k} not divisible by qk={qk}");
    }

    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!("fast_mmq: input dtype must be BF16, F16, or F32, got {input_ty:?}");
    }

    // Convert to f32 if needed (MMQ quantize expects f32 input)
    let xs_f32 = if input_ty == DType::F32 {
        xs.contiguous()?
    } else {
        xs.to_dtype(DType::F32)?.contiguous()?
    };

    let (xs_storage, xs_layout) = xs_f32.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmq: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();

    let stream_ptr = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;

    // Compute padded dimensions
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    // Must also be multiple of 4*QK8_1 = 128 for block_q8_1_mmq
    let k_padded = pad(k_padded, 4 * QK8_1);

    // Workspace for block_q8_1_mmq quantized activations
    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = b_size * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    // Extra padding for mmq_x_max (128 for MMA path)
    let workspace_extra = 128 * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;

    let (scratch_ptr, _workspace_guard) = workspace_ensure(&MMQ_WORKSPACE, &dev, workspace_bytes)?;
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    // Stream-k fixup workspace: nsm * mmq_x_max * mmq_y_max * sizeof(float)
    // mmq_x_max=128 (MMA path), mmq_y_max=128 (Volta+)
    // We use cudaDeviceGetAttribute to get SM count, but since cudarc doesn't
    // expose it directly, we allocate a generous upper bound (256 SMs * 128 * 128 * 4 = 16 MB).
    // The actual usage is much smaller and the workspace is reused across launches.
    const MMQ_X_MAX: usize = 128;
    const MMQ_Y_MAX: usize = 128;
    const MAX_SMS: usize = 256; // covers all current GPUs
    let fixup_bytes = MAX_SMS * MMQ_X_MAX * MMQ_Y_MAX * std::mem::size_of::<f32>();
    let (fixup_ptr, _fixup_guard) = workspace_ensure(&FIXUP_WORKSPACE, &dev, fixup_bytes)?;
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let weight_ptr = w.device_ptr()? as *const std::ffi::c_void;
    let stride_row_x = (k / qk) as i64;
    let di = get_device_info(&dev);

    let out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
    let stride_col_dst = nrows as i64;

    {
        let slice = xs_cuda.as_cuda_slice::<f32>()?;
        let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
        let (out_ptr, _out_guard) = slice_ptr(&out, 0);

        unsafe {
            let quantize = quantize_launcher(ds_layout_for(dtype));
            quantize(
                xs_ptr as *const std::ffi::c_void,
                std::ptr::null(),
                scratch_ptr,
                0,
                k as i64,
                k as i64,
                0,
                0,
                k_padded as i64,
                b_size as i64,
                1,
                1,
                stream_ptr,
            );

            let launcher = mmq_launcher(dtype).expect("supports() checked");
            launcher(
                fixup_ptr,
                weight_ptr,
                scratch_ptr as *const std::ffi::c_void,
                out_ptr as *mut std::ffi::c_void,
                k as i64,
                nrows as i64,
                b_size as i64,
                stride_row_x,
                stride_col_dst,
                di.cc,
                di.nsm,
                di.smpbo,
                di.warp_size,
                stream_ptr,
            );
        }
    }

    let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
    let out_tensor = Tensor::from((Storage::Cuda(out_storage), output_shape(&xs_f32, nrows)));

    if input_ty == DType::F32 {
        Ok(out_tensor)
    } else {
        out_tensor.to_dtype(input_ty)
    }
}
