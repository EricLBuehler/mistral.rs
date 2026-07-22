//! CUDA fast path for GGUF tiled matmul (prompt/prefill phase).
//! Handles batch > 8 (complement to fast_mmvq which handles batch 1-8).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::cuda::cudarc::driver::{CudaSlice, DevicePtr, DeviceRepr};
use candle_core::cuda_backend::CudaDType;
use candle_core::{
    quantized::{GgmlDType, QTensor},
    CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

use super::ffi;
use crate::utils::{slice_ptr, slice_ptr_mut_on_stream};

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

fn wrap_cuda_output<T: CudaDType + DeviceRepr>(
    out: CudaSlice<T>,
    dev: &CudaDevice,
    shape: Shape,
) -> Tensor {
    Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        shape,
    ))
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
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 => {
            32
        }
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

type QuantizeGluF32Launcher = unsafe extern "C" fn(
    gate: *const f32,
    up: *const f32,
    ids: *const i32,
    vy: *mut std::ffi::c_void,
    ne00: i64,
    s01: i64,
    ne0: i64,
    ne1: i64,
    activation: i32,
    stream: *mut std::ffi::c_void,
);

fn quantize_launcher(layout: DsLayout) -> QuantizeLauncher {
    match layout {
        DsLayout::D4 => ffi::launch_mmq_quantize_q8_1_D4,
        DsLayout::DS4 => ffi::launch_mmq_quantize_q8_1_DS4,
        DsLayout::D2S6 => ffi::launch_mmq_quantize_q8_1_D2S6,
    }
}

fn quantize_glu_f32_launcher(layout: DsLayout) -> QuantizeGluF32Launcher {
    match layout {
        DsLayout::D4 => ffi::launch_mmq_quantize_glu_q8_1_D4_f32,
        DsLayout::DS4 => ffi::launch_mmq_quantize_glu_q8_1_DS4_f32,
        DsLayout::D2S6 => ffi::launch_mmq_quantize_glu_q8_1_D2S6_f32,
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
    type_dst: i32,
    stream: *mut std::ffi::c_void,
);

type MmqMoeLauncher = unsafe extern "C" fn(
    tmp_fixup: *mut std::ffi::c_void,
    x: *const std::ffi::c_void,
    y: *const std::ffi::c_void,
    ids_dst: *const i32,
    expert_bounds: *const i32,
    dst: *mut std::ffi::c_void,
    ncols_x: i64,
    nrows_x: i64,
    ncols_dst: i64,
    stride_row_x: i64,
    stride_col_dst: i64,
    num_experts: i64,
    ncols_max: i64,
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

fn mmq_moe_launcher(dtype: GgmlDType) -> Option<MmqMoeLauncher> {
    let f: MmqMoeLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmq_gguf_q4_0_moe,
        GgmlDType::Q4_1 => ffi::launch_mmq_gguf_q4_1_moe,
        GgmlDType::Q5_0 => ffi::launch_mmq_gguf_q5_0_moe,
        GgmlDType::Q5_1 => ffi::launch_mmq_gguf_q5_1_moe,
        GgmlDType::Q8_0 => ffi::launch_mmq_gguf_q8_0_moe,
        GgmlDType::Q2K => ffi::launch_mmq_gguf_q2_k_moe,
        GgmlDType::Q3K => ffi::launch_mmq_gguf_q3_k_moe,
        GgmlDType::Q4K => ffi::launch_mmq_gguf_q4_k_moe,
        GgmlDType::Q5K => ffi::launch_mmq_gguf_q5_k_moe,
        GgmlDType::Q6K => ffi::launch_mmq_gguf_q6_k_moe,
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
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
    }
    .unwrap_or(8);
    let minor = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
    }
    .unwrap_or(0);
    let nsm = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
        )
    }
    .unwrap_or(1);
    let smpbo = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        )
    }
    .unwrap_or(49152);
    let warp_size = unsafe {
        result::device::get_attribute(
            cu_device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE,
        )
    }
    .unwrap_or(32);
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
/// Output is stored in the input dtype.
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

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmq: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let type_x = match input_ty {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 30,
        _ => unreachable!(),
    };

    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut std::ffi::c_void;

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

    let (weight_ptr, _weight_guard) = w.device_ptr_with_guard(&stream)?;
    let weight_ptr = weight_ptr as *const std::ffi::c_void;
    let stride_row_x = (k / qk) as i64;
    let di = get_device_info(&dev);

    let stride_col_dst = nrows as i64;

    let quantize = quantize_launcher(ds_layout_for(dtype));
    let launcher = mmq_launcher(dtype).expect("supports() checked");

    match input_ty {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let mut out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };
            let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
            let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

            unsafe {
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    std::ptr::null(),
                    scratch_ptr,
                    type_x,
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
                    type_x,
                    stream_ptr,
                );
            }
            drop(_out_guard);
            Ok(wrap_cuda_output(out, &dev, output_shape(&xs, nrows)))
        }
        DType::F16 => {
            let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            let mut out = unsafe { dev.alloc::<half::f16>(nrows * b_size)? };
            let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
            let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

            unsafe {
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    std::ptr::null(),
                    scratch_ptr,
                    type_x,
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
                    type_x,
                    stream_ptr,
                );
            }
            drop(_out_guard);
            Ok(wrap_cuda_output(out, &dev, output_shape(&xs, nrows)))
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let mut out = unsafe { dev.alloc::<f32>(nrows * b_size)? };
            let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
            let (out_ptr, _out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

            unsafe {
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    std::ptr::null(),
                    scratch_ptr,
                    type_x,
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
                    type_x,
                    stream_ptr,
                );
            }
            drop(_out_guard);
            Ok(wrap_cuda_output(out, &dev, output_shape(&xs, nrows)))
        }
        _ => unreachable!(),
    }
}

/// Run one GGUF-quantized MoE projection with llama.cpp-style grouped MMQ.
///
/// `ids_src` maps compact expert-sorted rows to input token rows. `ids_dst`
/// maps those same compact rows to output assignment rows. For Gemma4 MoE this
/// lets callers produce rows in flat assignment order for downstream grouped
/// MoE stages.
#[allow(clippy::too_many_arguments)]
pub fn grouped(
    weight: &QTensor,
    xs: &Tensor,
    ids_src: &CudaSlice<u32>,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    ncols_max: usize,
    num_experts: usize,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let dtype = weight.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmq grouped: unsupported quant dtype {dtype:?}");
    }

    let (_, k) = xs.dims2()?;

    let (weight_experts, nrows, ncols) = weight.shape().dims3()?;
    if weight_experts != num_experts {
        candle_core::bail!(
            "fast_mmq grouped: expected {num_experts} experts, got {weight_experts}"
        );
    }
    if k != ncols {
        candle_core::bail!(
            "fast_mmq grouped: shape mismatch — weight cols {ncols} vs input tail {k}"
        );
    }
    let qk = qk_for(dtype);
    if k % qk != 0 {
        candle_core::bail!("fast_mmq grouped: k={k} not divisible by qk={qk}");
    }

    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmq grouped: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmq grouped: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let type_x = match input_ty {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 30,
        _ => unreachable!(),
    };

    let stream_ptr = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);

    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = total_assignments * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_extra = 128 * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;
    let (scratch_ptr, _workspace_guard) = workspace_ensure(&MMQ_WORKSPACE, dev, workspace_bytes)?;
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    const MMQ_X_MAX: usize = 128;
    const MMQ_Y_MAX: usize = 128;
    const MAX_SMS: usize = 256;
    let fixup_bytes = MAX_SMS * MMQ_X_MAX * MMQ_Y_MAX * std::mem::size_of::<f32>();
    let (fixup_ptr, _fixup_guard) = workspace_ensure(&FIXUP_WORKSPACE, dev, fixup_bytes)?;
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let out = unsafe { dev.alloc::<f32>(total_assignments * nrows)? };

    let weight_ptr = weight.device_ptr()? as *const std::ffi::c_void;
    let stride_row_x = (k / qk) as i64;
    let stride_col_dst = nrows as i64;
    let di = get_device_info(dev);

    let quantize = quantize_launcher(ds_layout_for(dtype));
    let launcher = mmq_moe_launcher(dtype).expect("supports() checked");

    let (ids_src_ptr, _ids_src_guard) = slice_ptr(ids_src, 0);
    let (ids_dst_ptr, _ids_dst_guard) = slice_ptr(ids_dst, 0);
    let (bounds_ptr, _bounds_guard) = slice_ptr(expert_bounds, 0);
    let (out_ptr, _out_guard) = slice_ptr(&out, 0);

    unsafe {
        match input_ty {
            DType::BF16 => {
                let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            DType::F16 => {
                let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            DType::F32 => {
                let slice = xs_cuda.as_cuda_slice::<f32>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            _ => unreachable!(),
        }

        launcher(
            fixup_ptr,
            weight_ptr,
            scratch_ptr as *const std::ffi::c_void,
            ids_dst_ptr as *const i32,
            bounds_ptr as *const i32,
            out_ptr as *mut std::ffi::c_void,
            k as i64,
            nrows as i64,
            total_assignments as i64,
            stride_row_x,
            stride_col_dst,
            num_experts as i64,
            ncols_max as i64,
            di.cc,
            di.nsm,
            di.smpbo,
            di.warp_size,
            stream_ptr,
        );
    }

    drop(_out_guard);
    drop(_bounds_guard);
    drop(_ids_dst_guard);
    drop(_ids_src_guard);

    let out_shape: Shape = vec![total_assignments, nrows].into();
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        out_shape,
    )))
}

struct GroupedGluRun<'a> {
    weight: &'a QTensor,
    gate: &'a Tensor,
    up: &'a Tensor,
    row_stride: usize,
    ids_src: Option<&'a CudaSlice<u32>>,
    ids_dst: &'a CudaSlice<u32>,
    expert_bounds: &'a CudaSlice<u32>,
    total_assignments: usize,
    ncols_max: usize,
    num_experts: usize,
    activation: i32,
    dev: &'a CudaDevice,
}

fn grouped_from_glu(run: GroupedGluRun<'_>) -> Result<Tensor> {
    let GroupedGluRun {
        weight,
        gate,
        up,
        row_stride,
        ids_src,
        ids_dst,
        expert_bounds,
        total_assignments,
        ncols_max,
        num_experts,
        activation,
        dev,
    } = run;
    let dtype = weight.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmq grouped_from_glu_pair: unsupported quant dtype {dtype:?}");
    }

    let (gate_rows, k) = gate.dims2()?;
    let (up_rows, up_k) = up.dims2()?;
    if gate_rows != total_assignments || up_rows != total_assignments || up_k != k {
        candle_core::bail!(
            "fast_mmq grouped_from_glu_pair: gate/up shape mismatch {:?} vs {:?}, total_assignments={total_assignments}",
            gate.shape(),
            up.shape()
        );
    }
    if gate.dtype() != DType::F32 || up.dtype() != DType::F32 {
        candle_core::bail!(
            "fast_mmq grouped_from_glu_pair: gate/up must be F32, got {:?} and {:?}",
            gate.dtype(),
            up.dtype()
        );
    }

    let (weight_experts, nrows, ncols) = weight.shape().dims3()?;
    if weight_experts != num_experts {
        candle_core::bail!(
            "fast_mmq grouped_from_glu_pair: expected {num_experts} experts, got {weight_experts}"
        );
    }
    if k != ncols {
        candle_core::bail!(
            "fast_mmq grouped_from_glu_pair: shape mismatch — weight cols {ncols} vs input tail {k}"
        );
    }
    let qk = qk_for(dtype);
    if k % qk != 0 {
        candle_core::bail!("fast_mmq grouped_from_glu_pair: k={k} not divisible by qk={qk}");
    }

    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let Storage::Cuda(gate_cuda) = &*gate_storage else {
        candle_core::bail!("fast_mmq grouped_from_glu_pair: gate must live on CUDA");
    };
    let (up_storage, up_layout) = up.storage_and_layout();
    let Storage::Cuda(up_cuda) = &*up_storage else {
        candle_core::bail!("fast_mmq grouped_from_glu_pair: up must live on CUDA");
    };
    if gate_layout.stride() != [row_stride, 1] || up_layout.stride() != [row_stride, 1] {
        candle_core::bail!("fast_mmq grouped_from_glu_pair: invalid gate/up row stride");
    }

    let stream_ptr = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);

    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = total_assignments * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_extra = 128 * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;
    let (scratch_ptr, _workspace_guard) = workspace_ensure(&MMQ_WORKSPACE, dev, workspace_bytes)?;
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    const MMQ_X_MAX: usize = 128;
    const MMQ_Y_MAX: usize = 128;
    const MAX_SMS: usize = 256;
    let fixup_bytes = MAX_SMS * MMQ_X_MAX * MMQ_Y_MAX * std::mem::size_of::<f32>();
    let (fixup_ptr, _fixup_guard) = workspace_ensure(&FIXUP_WORKSPACE, dev, fixup_bytes)?;
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let out = unsafe { dev.alloc::<f32>(total_assignments * nrows)? };

    let weight_ptr = weight.device_ptr()? as *const std::ffi::c_void;
    let stride_row_x = (k / qk) as i64;
    let stride_col_dst = nrows as i64;
    let di = get_device_info(dev);

    let quantize = quantize_glu_f32_launcher(ds_layout_for(dtype));
    let launcher = mmq_moe_launcher(dtype).expect("supports() checked");

    let gate_slice = gate_cuda.as_cuda_slice::<f32>()?;
    let up_slice = up_cuda.as_cuda_slice::<f32>()?;
    let (gate_ptr, _gate_guard) = slice_ptr(gate_slice, gate_layout.start_offset());
    let (up_ptr, _up_guard) = slice_ptr(up_slice, up_layout.start_offset());
    let (ids_src_ptr, _ids_src_guard) = match ids_src {
        Some(ids_src) => {
            let (ptr, guard) = slice_ptr(ids_src, 0);
            (ptr, Some(guard))
        }
        None => (0, None),
    };
    let (ids_dst_ptr, _ids_dst_guard) = slice_ptr(ids_dst, 0);
    let (bounds_ptr, _bounds_guard) = slice_ptr(expert_bounds, 0);
    let (out_ptr, _out_guard) = slice_ptr(&out, 0);

    unsafe {
        quantize(
            gate_ptr as *const f32,
            up_ptr as *const f32,
            ids_src_ptr as *const i32,
            scratch_ptr,
            k as i64,
            row_stride as i64,
            k_padded as i64,
            total_assignments as i64,
            activation,
            stream_ptr,
        );

        launcher(
            fixup_ptr,
            weight_ptr,
            scratch_ptr as *const std::ffi::c_void,
            ids_dst_ptr as *const i32,
            bounds_ptr as *const i32,
            out_ptr as *mut std::ffi::c_void,
            k as i64,
            nrows as i64,
            total_assignments as i64,
            stride_row_x,
            stride_col_dst,
            num_experts as i64,
            ncols_max as i64,
            di.cc,
            di.nsm,
            di.smpbo,
            di.warp_size,
            stream_ptr,
        );
    }

    drop(_out_guard);
    drop(_bounds_guard);
    drop(_ids_dst_guard);
    drop(_ids_src_guard);
    drop(_up_guard);
    drop(_gate_guard);

    let out_shape: Shape = vec![total_assignments, nrows].into();
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        out_shape,
    )))
}

/// Run one grouped MoE projection after fusing `activation(gate) * up` directly
/// into the MMQ activation quantization layout.
#[allow(clippy::too_many_arguments)]
pub fn grouped_from_glu_pair(
    weight: &QTensor,
    gate: &Tensor,
    up: &Tensor,
    ids_src: &CudaSlice<u32>,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    ncols_max: usize,
    num_experts: usize,
    activation: i32,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let row_stride = gate.dim(1)?;
    grouped_from_glu(GroupedGluRun {
        weight,
        gate: &gate,
        up: &up,
        row_stride,
        ids_src: Some(ids_src),
        ids_dst,
        expert_bounds,
        total_assignments,
        ncols_max,
        num_experts,
        activation,
        dev,
    })
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn grouped_from_glu_sorted_pair(
    weight: &QTensor,
    gate: &Tensor,
    up: &Tensor,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    ncols_max: usize,
    num_experts: usize,
    activation: i32,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let gate = gate.contiguous()?;
    let up = up.contiguous()?;
    let row_stride = gate.dim(1)?;
    grouped_from_glu(GroupedGluRun {
        weight,
        gate: &gate,
        up: &up,
        row_stride,
        ids_src: None,
        ids_dst,
        expert_bounds,
        total_assignments,
        ncols_max,
        num_experts,
        activation,
        dev,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn grouped_from_glu_packed(
    weight: &QTensor,
    gate_up: &Tensor,
    ids_src: &CudaSlice<u32>,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    ncols_max: usize,
    num_experts: usize,
    activation: i32,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let gate_up = gate_up.contiguous()?;
    let (_, _, k) = weight.shape().dims3()?;
    if gate_up.dims2()? != (total_assignments, 2 * k) {
        candle_core::bail!("fast_mmq grouped_from_glu_packed: gate/up shape mismatch");
    }
    let gate = gate_up.narrow(1, 0, k)?;
    let up = gate_up.narrow(1, k, k)?;
    grouped_from_glu(GroupedGluRun {
        weight,
        gate: &gate,
        up: &up,
        row_stride: 2 * k,
        ids_src: Some(ids_src),
        ids_dst,
        expert_bounds,
        total_assignments,
        ncols_max,
        num_experts,
        activation,
        dev,
    })
}

/// Run two GGUF-quantized MoE projections with llama.cpp-style grouped MMQ.
///
/// Gate/up share one MMQ activation quantization pass and one packed output.
#[allow(clippy::too_many_arguments)]
pub fn grouped_pair_packed(
    gate: &QTensor,
    up: &QTensor,
    xs: &Tensor,
    ids_src: &CudaSlice<u32>,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    topk: usize,
    num_experts: usize,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let dtype = gate.dtype();
    if dtype != up.dtype() {
        candle_core::bail!(
            "fast_mmq grouped_pair requires matching gate/up dtypes, got {:?} and {:?}",
            dtype,
            up.dtype()
        );
    }
    if !supports(dtype) {
        candle_core::bail!("fast_mmq grouped_pair: unsupported quant dtype {dtype:?}");
    }

    let (num_tokens, k) = xs.dims2()?;
    if total_assignments != num_tokens * topk {
        candle_core::bail!(
            "fast_mmq grouped_pair: total_assignments={total_assignments} does not match num_tokens={num_tokens} * topk={topk}"
        );
    }

    let (gate_experts, nrows, ncols) = gate.shape().dims3()?;
    let (up_experts, up_nrows, up_ncols) = up.shape().dims3()?;
    if gate_experts != num_experts || up_experts != num_experts {
        candle_core::bail!(
            "fast_mmq grouped_pair: expected {num_experts} experts, got gate={gate_experts} up={up_experts}"
        );
    }
    if nrows != up_nrows || ncols != up_ncols {
        candle_core::bail!(
            "fast_mmq grouped_pair: gate/up shape mismatch {:?} vs {:?}",
            gate.shape(),
            up.shape()
        );
    }
    if k != ncols {
        candle_core::bail!(
            "fast_mmq grouped_pair: shape mismatch — weight cols {ncols} vs input tail {k}"
        );
    }
    let qk = qk_for(dtype);
    if k % qk != 0 {
        candle_core::bail!("fast_mmq grouped_pair: k={k} not divisible by qk={qk}");
    }

    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "fast_mmq grouped_pair: input dtype must be BF16, F16, or F32, got {input_ty:?}"
        );
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmq grouped_pair: input must live on CUDA");
    };
    let xs_offset = xs_layout.start_offset();
    let type_x = match input_ty {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 30,
        _ => unreachable!(),
    };

    let stream_ptr = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(pad(k, MATRIX_ROW_PADDING), 4 * QK8_1);

    let blocks_per_row = k_padded / (4 * QK8_1);
    let workspace_main = total_assignments * blocks_per_row * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_extra = 128 * BLOCK_Q8_1_MMQ_SIZE;
    let workspace_bytes = workspace_main + workspace_extra;
    let (scratch_ptr, _workspace_guard) = workspace_ensure(&MMQ_WORKSPACE, dev, workspace_bytes)?;
    let scratch_ptr = scratch_ptr as *mut std::ffi::c_void;

    const MMQ_X_MAX: usize = 128;
    const MMQ_Y_MAX: usize = 128;
    const MAX_SMS: usize = 256;
    let fixup_bytes = MAX_SMS * MMQ_X_MAX * MMQ_Y_MAX * std::mem::size_of::<f32>();
    let (fixup_ptr, _fixup_guard) = workspace_ensure(&FIXUP_WORKSPACE, dev, fixup_bytes)?;
    let fixup_ptr = fixup_ptr as *mut std::ffi::c_void;

    let output = unsafe { dev.alloc::<f32>(total_assignments * nrows * 2)? };

    let gate_ptr = gate.device_ptr()? as *const std::ffi::c_void;
    let up_ptr = up.device_ptr()? as *const std::ffi::c_void;
    let stride_row_x = (k / qk) as i64;
    let stride_col_dst = (2 * nrows) as i64;
    let di = get_device_info(dev);

    let quantize = quantize_launcher(ds_layout_for(dtype));
    let launcher = mmq_moe_launcher(dtype).expect("supports() checked");

    let (ids_src_ptr, _ids_src_guard) = slice_ptr(ids_src, 0);
    let (ids_dst_ptr, _ids_dst_guard) = slice_ptr(ids_dst, 0);
    let (bounds_ptr, _bounds_guard) = slice_ptr(expert_bounds, 0);
    let (gate_out_ptr, _gate_out_guard) = slice_ptr(&output, 0);
    let (up_out_ptr, _up_out_guard) = slice_ptr(&output, nrows);

    unsafe {
        match input_ty {
            DType::BF16 => {
                let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            DType::F16 => {
                let slice = xs_cuda.as_cuda_slice::<half::f16>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            DType::F32 => {
                let slice = xs_cuda.as_cuda_slice::<f32>()?;
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                quantize(
                    xs_ptr as *const std::ffi::c_void,
                    ids_src_ptr as *const i32,
                    scratch_ptr,
                    type_x,
                    k as i64,
                    k as i64,
                    0,
                    0,
                    k_padded as i64,
                    total_assignments as i64,
                    1,
                    1,
                    stream_ptr,
                );
            }
            _ => unreachable!(),
        }

        for (weight_ptr, out_ptr) in [
            (gate_ptr, gate_out_ptr as *mut std::ffi::c_void),
            (up_ptr, up_out_ptr as *mut std::ffi::c_void),
        ] {
            launcher(
                fixup_ptr,
                weight_ptr,
                scratch_ptr as *const std::ffi::c_void,
                ids_dst_ptr as *const i32,
                bounds_ptr as *const i32,
                out_ptr,
                k as i64,
                nrows as i64,
                total_assignments as i64,
                stride_row_x,
                stride_col_dst,
                num_experts as i64,
                num_tokens as i64,
                di.cc,
                di.nsm,
                di.smpbo,
                di.warp_size,
                stream_ptr,
            );
        }
    }

    drop(_gate_out_guard);
    drop(_up_out_guard);
    drop(_bounds_guard);
    drop(_ids_dst_guard);
    drop(_ids_src_guard);

    let out_shape: Shape = vec![total_assignments, 2 * nrows].into();
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
        out_shape,
    )))
}

/// Run two GGUF-quantized MoE projections with llama.cpp-style grouped MMQ.
#[allow(clippy::too_many_arguments)]
pub fn grouped_pair(
    gate: &QTensor,
    up: &QTensor,
    xs: &Tensor,
    ids_src: &CudaSlice<u32>,
    ids_dst: &CudaSlice<u32>,
    expert_bounds: &CudaSlice<u32>,
    total_assignments: usize,
    topk: usize,
    num_experts: usize,
    dev: &CudaDevice,
) -> Result<(Tensor, Tensor)> {
    let output = grouped_pair_packed(
        gate,
        up,
        xs,
        ids_src,
        ids_dst,
        expert_bounds,
        total_assignments,
        topk,
        num_experts,
        dev,
    )?;
    let (_, nrows, _) = gate.shape().dims3()?;
    let gate = output.narrow(1, 0, nrows)?.contiguous()?;
    let up = output.narrow(1, nrows, nrows)?.contiguous()?;
    Ok((gate, up))
}
