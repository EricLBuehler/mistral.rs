use std::{
    collections::HashMap,
    ffi::{c_char, c_void, CStr},
    sync::{Arc, Mutex, OnceLock},
    thread::ThreadId,
};

use candle_core::{
    cuda::{
        cudarc::driver::{
            result,
            sys::{CUdevice_attribute, CUstreamCaptureStatus},
            CudaSlice,
        },
        DeviceId,
    },
    CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};
use float8::F8E4M3;
use half::{bf16, f16};

use super::Nvfp4LayerParts;
use crate::{
    cutile::Nvfp4GemmArgs,
    utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream},
    Nvfp4ActivationMode, NVFP4_BLOCK_SIZE,
};

const MIN_ROWS: usize = 1024;
const DECODE_MIN_ROWS: usize = 2;
const DECODE_MAX_ROWS: usize = 64;
const MIN_COLUMNS: usize = 4096;
const MIN_REDUCTION: usize = 4096;
const COLUMN_ALIGNMENT: usize = 32;
const REDUCTION_ALIGNMENT: usize = 64;
const POINTER_ALIGNMENT: u64 = 16;
const CUDA_STREAM_PER_THREAD_HANDLE: usize = 2;
const BF16_CONTEXT: usize = 0;
const F16_CONTEXT: usize = 1;
const DTYPE_COUNT: usize = 2;
const PREFILL_KERNEL: i32 = 0;
const DECODE_KERNEL: i32 = 1;
const CONTEXT_COUNT: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Context {
    device: i32,
    sm_count: i32,
    dtype: i32,
    kernel: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Problem {
    m: i32,
    n: i32,
    k: i32,
}

#[repr(C)]
struct Launch {
    shape: Problem,
    context: Context,
    a_packed: *const c_void,
    w_packed: *const c_void,
    a_scale_swizzled: *const c_void,
    w_scale_swizzled: *const c_void,
    weight_global: *const f32,
    activation_global: *const f32,
    output: *mut c_void,
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
}

#[repr(C)]
#[derive(Default)]
struct Resources {
    context: Context,
    major: i32,
    minor: i32,
    threads: i32,
    registers_per_thread: i32,
    shared_bytes: usize,
    local_bytes: usize,
}

unsafe extern "C" {
    fn mistralrs_nvfp4_error_string(status: i32) -> *const c_char;
    fn mistralrs_nvfp4_prepare(
        device: i32,
        dtype: i32,
        kernel: i32,
        resources: *mut Resources,
    ) -> i32;
    fn mistralrs_nvfp4_workspace_size(
        context: *const Context,
        shape: *const Problem,
        bytes: *mut usize,
    ) -> i32;
    fn mistralrs_nvfp4_gemm(launch: *const Launch) -> i32;
    fn mistralrs_nvfp4_scale_bytes(rows: i32, k: i32, bytes: *mut usize) -> i32;
    fn mistralrs_nvfp4_swizzle_cuda(
        source: *const c_void,
        destination: *mut c_void,
        rows: i32,
        k: i32,
        destination_bytes: usize,
        stream: *mut c_void,
    ) -> i32;
}

fn check_status(operation: &str, status: i32) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let message = unsafe { mistralrs_nvfp4_error_string(status) };
    let message = if message.is_null() {
        std::borrow::Cow::Borrowed("unknown error")
    } else {
        unsafe { CStr::from_ptr(message) }.to_string_lossy()
    };
    candle_core::bail!("CUTLASS NVFP4 {operation} failed: {message} (status {status})")
}

fn outside_capture(device: &CudaDevice, operation: &str) -> Result<()> {
    let status = device
        .cuda_stream()
        .capture_status()
        .map_err(candle_core::Error::msg)?;
    if status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
        candle_core::bail!("CUTLASS NVFP4 {operation} must be warmed before CUDA graph capture");
    }
    Ok(())
}

type PreparedDevices = Mutex<HashMap<DeviceId, [Context; CONTEXT_COUNT]>>;
static PREPARED_DEVICES: OnceLock<PreparedDevices> = OnceLock::new();

fn prepare(device: &CudaDevice) -> Result<[Context; CONTEXT_COUNT]> {
    let stream = device.cuda_stream();
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::msg)?;
    let prepared = PREPARED_DEVICES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut prepared = prepared.lock().unwrap();
    if let Some(contexts) = prepared.get(&device.id()) {
        return Ok(*contexts);
    }
    outside_capture(device, "kernel preparation")?;
    let ordinal = i32::try_from(stream.context().ordinal())?;
    let mut contexts = [Context::default(); CONTEXT_COUNT];
    for kernel in [PREFILL_KERNEL, DECODE_KERNEL] {
        for dtype in 0..DTYPE_COUNT {
            let mut resources = Resources::default();
            check_status("kernel preparation", unsafe {
                mistralrs_nvfp4_prepare(ordinal, dtype as i32, kernel, &mut resources)
            })?;
            contexts[kernel as usize * DTYPE_COUNT + dtype] = resources.context;
        }
    }
    prepared.insert(device.id(), contexts);
    Ok(contexts)
}

#[derive(Eq, Hash, PartialEq)]
struct WorkspaceKey {
    device: DeviceId,
    stream: usize,
    thread: Option<ThreadId>,
    capacity: usize,
}

type Workspace = Arc<Mutex<CudaSlice<u8>>>;
type Workspaces = Mutex<HashMap<WorkspaceKey, Workspace>>;
static WORKSPACES: OnceLock<Workspaces> = OnceLock::new();

fn workspace(device: &CudaDevice, bytes: usize) -> Result<Option<Workspace>> {
    if bytes == 0 {
        return Ok(None);
    }
    let capacity = bytes
        .checked_next_power_of_two()
        .ok_or_else(|| candle_core::Error::msg("CUTLASS NVFP4 workspace capacity overflow"))?;
    let stream = device.cuda_stream().cu_stream() as usize;
    let key = WorkspaceKey {
        device: device.id(),
        stream,
        thread: (stream == CUDA_STREAM_PER_THREAD_HANDLE).then(|| std::thread::current().id()),
        capacity,
    };
    let workspaces = WORKSPACES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut workspaces = workspaces.lock().unwrap();
    if let Some(workspace) = workspaces.get(&key) {
        return Ok(Some(Arc::clone(workspace)));
    }
    outside_capture(device, "workspace allocation")?;
    // Each capacity stays alive so previously captured graphs retain their workspace addresses.
    let workspace = Arc::new(Mutex::new(unsafe { device.alloc::<u8>(capacity)? }));
    workspaces.insert(key, Arc::clone(&workspace));
    Ok(Some(workspace))
}

fn aligned_bytes(tensor: &Tensor) -> Result<Tensor> {
    let tensor = tensor.contiguous()?;
    let aligned = {
        let device = tensor.device().as_cuda_device()?;
        let stream = device.cuda_stream();
        let (storage, layout) = tensor.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            unreachable!()
        };
        let (address, _guard) = slice_ptr_on_stream(
            storage.as_cuda_slice::<u8>()?,
            layout.start_offset(),
            &stream,
        );
        address.is_multiple_of(POINTER_ALIGNMENT)
    };
    if aligned {
        Ok(tensor)
    } else {
        tensor.force_contiguous()
    }
}

fn scale_bytes(rows: usize, k: usize) -> Result<usize> {
    let mut bytes = 0;
    check_status("scale size query", unsafe {
        mistralrs_nvfp4_scale_bytes(i32::try_from(rows)?, i32::try_from(k)?, &mut bytes)
    })?;
    Ok(bytes)
}

pub(crate) fn swizzle_scales(canonical: &Tensor) -> Result<Tensor> {
    let (rows, columns) = canonical.dims2()?;
    if canonical.dtype() != DType::F8E4M3 {
        candle_core::bail!("CUTLASS NVFP4 canonical scales must be F8E4M3");
    }
    let k = columns
        .checked_mul(NVFP4_BLOCK_SIZE)
        .ok_or_else(|| candle_core::Error::msg("CUTLASS NVFP4 scale dimension overflow"))?;
    let bytes = scale_bytes(rows, k)?;
    let canonical = crate::utils::contiguous_fp8(canonical)?;
    let device = canonical.device().as_cuda_device()?;
    let stream = device.cuda_stream();
    stream
        .context()
        .bind_to_thread()
        .map_err(candle_core::Error::msg)?;
    let (storage, layout) = canonical.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        unreachable!()
    };
    let (source, _source_guard) = slice_ptr_on_stream(
        storage.as_cuda_slice::<F8E4M3>()?,
        layout.start_offset(),
        &stream,
    );
    let mut output = unsafe { device.alloc::<u8>(bytes)? };
    let (destination, destination_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
    check_status("scale swizzle", unsafe {
        mistralrs_nvfp4_swizzle_cuda(
            source as *const c_void,
            destination as *mut c_void,
            i32::try_from(rows)?,
            i32::try_from(k)?,
            bytes,
            stream.cu_stream() as *mut c_void,
        )
    })?;
    drop(destination_guard);
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
        Shape::from_dims(&[bytes]),
    )))
}

#[derive(Debug)]
pub(crate) struct State {
    weights: Tensor,
    weight_scales: Tensor,
    contexts: [Context; CONTEXT_COUNT],
    large_decode_weights: bool,
    n: usize,
    k: usize,
}

impl State {
    pub(crate) fn new(parts: &Nvfp4LayerParts) -> Result<Option<Self>> {
        let Device::Cuda(device) = parts.weight.device() else {
            return Ok(None);
        };
        if parts.activation != Nvfp4ActivationMode::DynamicBlock
            || parts.weight.rank() != 2
            || !matches!(parts.dtype, DType::BF16 | DType::F16)
            || crate::cutile::device_compute_capability(device) != (12, 1)
        {
            return Ok(None);
        }
        let (n, packed_k) = parts.weight.dims2()?;
        let Some(k) = packed_k.checked_mul(2) else {
            return Ok(None);
        };
        if n < MIN_COLUMNS
            || k < MIN_REDUCTION
            || !n.is_multiple_of(COLUMN_ALIGNMENT)
            || !k.is_multiple_of(REDUCTION_ALIGNMENT)
            || n > i32::MAX as usize
            || k > i32::MAX as usize
        {
            return Ok(None);
        }
        outside_capture(device, "layer construction")?;
        let contexts = prepare(device)?;
        let l2_bytes = unsafe {
            result::device::get_attribute(
                device.cuda_stream().context().cu_device(),
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
            )
        }
        .map_err(candle_core::Error::msg)? as usize;
        let weight_bytes = parts.weight.elem_count() + parts.scales.elem_count();
        Ok(Some(Self {
            weights: aligned_bytes(&parts.weight)?,
            weight_scales: swizzle_scales(&parts.scales)?,
            contexts,
            large_decode_weights: weight_bytes >= l2_bytes,
            n,
            k,
        }))
    }

    pub(crate) fn supports(&self, rows: usize, dtype: DType) -> bool {
        rows <= i32::MAX as usize
            && (rows >= MIN_ROWS
                || (self.large_decode_weights
                    && (DECODE_MIN_ROWS..=DECODE_MAX_ROWS).contains(&rows)))
            && matches!(dtype, DType::BF16 | DType::F16)
    }

    pub(crate) fn forward(
        &self,
        packed: &Tensor,
        native_scales: &Tensor,
        dtype: DType,
        args: Nvfp4GemmArgs<'_>,
    ) -> Result<Tensor> {
        let (rows, packed_k) = packed.dims2()?;
        let activation_global = args.activation_global_scale.ok_or_else(|| {
            candle_core::Error::msg("CUTLASS NVFP4 requires an activation global scale")
        })?;
        if !self.supports(rows, dtype)
            || packed_k != self.k / 2
            || packed.dtype() != DType::U8
            || native_scales.dtype() != DType::U8
            || native_scales.dims() != [scale_bytes(rows, self.k)?]
            || args.weight_global_scale.dtype() != DType::F32
            || args.weight_global_scale.dims() != [self.n]
            || activation_global.dtype() != DType::F32
            || activation_global.elem_count() != 1
        {
            candle_core::bail!("invalid CUTLASS NVFP4 activation shape, dtype, or global scale");
        }
        for tensor in [
            packed,
            native_scales,
            args.weight_global_scale,
            activation_global,
        ] {
            if !tensor.device().same_device(self.weights.device()) {
                candle_core::bail!("CUTLASS NVFP4 operands must be on the layer device");
            }
        }
        let packed = aligned_bytes(packed)?;
        let native_scales = aligned_bytes(native_scales)?;
        let weight_global = args.weight_global_scale.contiguous()?;
        let activation_global = activation_global.contiguous()?;
        let device = self.weights.device().as_cuda_device()?;
        let stream = device.cuda_stream();
        stream
            .context()
            .bind_to_thread()
            .map_err(candle_core::Error::msg)?;
        let dtype_index = if dtype == DType::BF16 {
            BF16_CONTEXT
        } else {
            F16_CONTEXT
        };
        let kernel = if rows <= DECODE_MAX_ROWS {
            DECODE_KERNEL
        } else {
            PREFILL_KERNEL
        };
        let context = self.contexts[kernel as usize * DTYPE_COUNT + dtype_index];
        let shape = Problem {
            m: i32::try_from(rows)?,
            n: self.n as i32,
            k: self.k as i32,
        };
        let mut workspace_bytes = 0;
        check_status("workspace query", unsafe {
            mistralrs_nvfp4_workspace_size(&context, &shape, &mut workspace_bytes)
        })?;
        let workspace = workspace(device, workspace_bytes)?;
        let mut workspace_lock = workspace
            .as_ref()
            .map(|workspace| workspace.lock().unwrap());
        let (workspace_pointer, workspace_guard) = match workspace_lock.as_mut() {
            Some(workspace) => {
                let (pointer, guard) = slice_ptr_mut_on_stream(workspace, 0, &stream);
                (pointer, Some(guard))
            }
            None => (0, None),
        };
        macro_rules! pointer {
            ($tensor:expr, $ty:ty, $storage:ident, $layout:ident, $address:ident, $guard:ident) => {
                let ($storage, $layout) = $tensor.storage_and_layout();
                let Storage::Cuda($storage) = &*$storage else {
                    unreachable!()
                };
                let ($address, $guard) = slice_ptr_on_stream(
                    $storage.as_cuda_slice::<$ty>()?,
                    $layout.start_offset(),
                    &stream,
                );
            };
        }
        pointer!(packed, u8, a_storage, a_layout, a_pointer, a_guard);
        pointer!(
            native_scales,
            u8,
            as_storage,
            as_layout,
            as_pointer,
            as_guard
        );
        pointer!(self.weights, u8, w_storage, w_layout, w_pointer, w_guard);
        pointer!(
            self.weight_scales,
            u8,
            ws_storage,
            ws_layout,
            ws_pointer,
            ws_guard
        );
        pointer!(
            weight_global,
            f32,
            wg_storage,
            wg_layout,
            wg_pointer,
            wg_guard
        );
        pointer!(
            activation_global,
            f32,
            ag_storage,
            ag_layout,
            ag_pointer,
            ag_guard
        );
        let mut launch = Launch {
            shape,
            context,
            a_packed: a_pointer as *const c_void,
            w_packed: w_pointer as *const c_void,
            a_scale_swizzled: as_pointer as *const c_void,
            w_scale_swizzled: ws_pointer as *const c_void,
            weight_global: wg_pointer as *const f32,
            activation_global: ag_pointer as *const f32,
            output: std::ptr::null_mut(),
            workspace: workspace_pointer as *mut c_void,
            workspace_bytes,
            stream: stream.cu_stream() as *mut c_void,
        };
        let elements = rows
            .checked_mul(self.n)
            .ok_or_else(|| candle_core::Error::msg("CUTLASS NVFP4 output dimension overflow"))?;
        macro_rules! run {
            ($dtype:ty) => {{
                let mut output = unsafe { device.alloc::<$dtype>(elements)? };
                let (pointer, guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
                launch.output = pointer as *mut c_void;
                check_status("matmul", unsafe { mistralrs_nvfp4_gemm(&launch) })?;
                drop(guard);
                Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
                    Shape::from_dims(&[rows, self.n]),
                ))
            }};
        }
        let output = match dtype {
            DType::BF16 => run!(bf16),
            DType::F16 => run!(f16),
            _ => unreachable!(),
        };
        drop((
            a_guard,
            as_guard,
            w_guard,
            ws_guard,
            wg_guard,
            ag_guard,
            workspace_guard,
        ));
        drop(workspace_lock);
        Ok(output)
    }
}
