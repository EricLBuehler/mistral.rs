use std::{
    collections::HashMap,
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, OnceLock,
    },
    thread::ThreadId,
};

use candle_core::cuda::cudarc::driver::{sys, CudaEvent, CudaSlice, CudaStream, DeviceRepr};
use candle_core::cuda_backend::{CudaDType, DeviceId, WrapErr};
use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

const MARLIN_N_TILE: usize = 64;
const MARLIN_WIDE_TILE: usize = 128;
const MARLIN_K_TILE: usize = 64;
const MARLIN_MAX_PARALLEL: usize = 16;
const MARLIN_INPUT_ALIGNMENT: usize = 16;
const MIN_COMPUTE_CAPABILITY_MAJOR: i32 = 8;
const MIN_MEMORY_HEADROOM: usize = 1024 * 1024 * 1024;
const MEMORY_HEADROOM_DIVISOR: usize = 20;
const AFFINE_ONLY_MIN_BATCH: usize = 1;
const Q5_0_MIN_BATCH: usize = 16;
const Q5_1_MIN_BATCH: usize = 128;
const Q6K_MIN_BATCH: usize = 128;
const BACKEND_ENV: &str = "MISTRALRS_GGUF_AFFINE_BACKEND";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AffineFormatSpec {
    source_dtype: GgmlDType,
    format_code: i32,
    source_block_size: usize,
    payload_bits: usize,
    group_size: usize,
    min_batch: usize,
}

impl AffineFormatSpec {
    fn for_dtype(source_dtype: GgmlDType) -> Option<Self> {
        let (payload_bits, group_size) = match source_dtype {
            GgmlDType::Q4_0 | GgmlDType::Q4_1 => (4, 32),
            GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 | GgmlDType::Q8_1 => (8, 32),
            GgmlDType::Q2K | GgmlDType::Q3K => (4, 16),
            GgmlDType::Q4K => (4, 32),
            GgmlDType::Q5K | GgmlDType::Q8K => (8, 32),
            GgmlDType::Q6K => (8, 16),
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => return None,
        };
        Some(Self {
            source_dtype,
            format_code: super::ggml_dtype_to_uqff_code(source_dtype) as i32,
            source_block_size: source_dtype.block_size(),
            payload_bits,
            group_size,
            min_batch: match source_dtype {
                GgmlDType::Q8_1 | GgmlDType::Q8K => AFFINE_ONLY_MIN_BATCH,
                GgmlDType::Q5_0 => Q5_0_MIN_BATCH,
                GgmlDType::Q5_1 => Q5_1_MIN_BATCH,
                GgmlDType::Q6K => Q6K_MIN_BATCH,
                _ => super::GGUF_AFFINE_MIN_BATCH,
            },
        })
    }

    fn supports_f32_input(self) -> bool {
        matches!(self.source_dtype, GgmlDType::Q8_1 | GgmlDType::Q8K)
    }
}

pub(crate) fn minimum_batch(dtype: GgmlDType) -> Option<usize> {
    Some(AffineFormatSpec::for_dtype(dtype)?.min_batch)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PackedAffinePlan {
    format: AffineFormatSpec,
    n: usize,
    padded_n: usize,
    k: usize,
    payload_bytes: usize,
    metadata_values: usize,
    metadata_bytes: usize,
    workspace_len: usize,
    total_bytes: usize,
}

impl PackedAffinePlan {
    fn new(weight: &QTensor) -> Option<Self> {
        if !enabled() || !supports_device(&weight.device()) || weight.shape().rank() != 2 {
            return None;
        }
        let format = AffineFormatSpec::for_dtype(weight.dtype())?;
        let (n, k) = weight.shape().dims2().ok()?;
        let padded_n = padded_n_for_shape(n, k)?;
        if n == 0
            || k == 0
            || !k.is_multiple_of(format.source_block_size)
            || i32::try_from(n).is_err()
            || i32::try_from(padded_n).is_err()
            || i32::try_from(k).is_err()
        {
            return None;
        }
        let payload_bytes = padded_n
            .checked_mul(k)?
            .checked_mul(format.payload_bits)?
            .checked_div(8)?;
        let metadata_values = k.checked_div(format.group_size)?.checked_mul(padded_n)?;
        let metadata_bytes = metadata_values.checked_mul(std::mem::size_of::<half::f16>())?;
        let workspace_len = padded_n
            .checked_div(MARLIN_N_TILE)?
            .checked_mul(MARLIN_MAX_PARALLEL)?;
        let workspace_bytes = workspace_len.checked_mul(std::mem::size_of::<u32>())?;
        let total_bytes = payload_bytes
            .checked_add(metadata_bytes)?
            .checked_add(metadata_bytes)?
            .checked_add(workspace_bytes)?;
        Some(Self {
            format,
            n,
            padded_n,
            k,
            payload_bytes,
            metadata_values,
            metadata_bytes,
            workspace_len,
            total_bytes,
        })
    }
}

fn supports_marlin_shape(n: usize, k: usize) -> bool {
    (k.is_multiple_of(MARLIN_WIDE_TILE) && n.is_multiple_of(MARLIN_N_TILE))
        || (k.is_multiple_of(MARLIN_K_TILE) && n.is_multiple_of(MARLIN_WIDE_TILE))
}

fn padded_n_for_shape(n: usize, k: usize) -> Option<usize> {
    let tile = if k.is_multiple_of(MARLIN_WIDE_TILE) {
        MARLIN_N_TILE
    } else if k.is_multiple_of(MARLIN_K_TILE) {
        MARLIN_WIDE_TILE
    } else {
        return None;
    };
    let padded_n = n
        .checked_add(tile - 1)?
        .checked_div(tile)?
        .checked_mul(tile)?;
    supports_marlin_shape(padded_n, k).then_some(padded_n)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Backend {
    Auto,
    Off,
}

#[derive(Default)]
struct DeviceReservations {
    remaining_bytes: usize,
    remaining_f32_bytes: usize,
    live: usize,
    memory_budget: Option<usize>,
}

impl DeviceReservations {
    fn reserve(&mut self, format: AffineFormatSpec, bytes: usize) {
        self.remaining_bytes = self.remaining_bytes.saturating_add(bytes);
        if format.supports_f32_input() {
            self.remaining_f32_bytes = self.remaining_f32_bytes.saturating_add(bytes);
        }
    }

    fn release(&mut self, format: AffineFormatSpec, bytes: usize) {
        self.remaining_bytes = self.remaining_bytes.saturating_sub(bytes);
        if format.supports_f32_input() {
            self.remaining_f32_bytes = self.remaining_f32_bytes.saturating_sub(bytes);
        }
    }

    fn bytes_for_dtype(&self, dtype: DType) -> usize {
        match dtype {
            DType::F16 | DType::BF16 => self.remaining_bytes,
            DType::F32 => self.remaining_f32_bytes,
            _ => 0,
        }
    }
}

fn reservations() -> &'static Mutex<HashMap<DeviceId, DeviceReservations>> {
    static RESERVATIONS: OnceLock<Mutex<HashMap<DeviceId, DeviceReservations>>> = OnceLock::new();
    RESERVATIONS.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Debug)]
pub(crate) struct Reservation {
    device_id: DeviceId,
    plan: PackedAffinePlan,
    bytes: usize,
    pending: AtomicBool,
    source_ready: CudaEvent,
}

impl Reservation {
    pub(crate) fn new(weight: &QMatMul) -> Option<Self> {
        let QMatMul::QTensor(weight) = weight else {
            return None;
        };
        let plan = PackedAffinePlan::new(weight)?;
        let bytes = plan.total_bytes;
        let Device::Cuda(dev) = weight.device() else {
            return None;
        };
        let stream = dev.cuda_stream();
        let source_ready = stream.context().new_event(None).ok()?;
        source_ready.record(&stream).ok()?;
        let device_id = dev.id();
        let mut reservations = reservations().lock().unwrap();
        let state = reservations.entry(device_id).or_default();
        state.reserve(plan.format, bytes);
        state.live += 1;
        state.memory_budget = None;
        Some(Self {
            device_id,
            plan,
            bytes,
            pending: AtomicBool::new(true),
            source_ready,
        })
    }

    pub(crate) fn consume(&self) {
        if self.pending.swap(false, Ordering::AcqRel) {
            let mut reservations = reservations().lock().unwrap();
            if let Some(state) = reservations.get_mut(&self.device_id) {
                state.release(self.plan.format, self.bytes);
            }
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        let mut reservations = reservations().lock().unwrap();
        let remove = if let Some(state) = reservations.get_mut(&self.device_id) {
            if self.pending.swap(false, Ordering::AcqRel) {
                state.release(self.plan.format, self.bytes);
            }
            state.live = state.live.saturating_sub(1);
            state.live == 0
        } else {
            false
        };
        if remove {
            reservations.remove(&self.device_id);
        }
    }
}

pub(crate) fn reserved_bytes(device: &Device, dtype: DType) -> usize {
    let Device::Cuda(dev) = device else {
        return 0;
    };
    reservations()
        .lock()
        .unwrap()
        .get(&dev.id())
        .filter(|state| state.memory_budget != Some(0))
        .map(|state| state.bytes_for_dtype(dtype))
        .unwrap_or(0)
}

pub(crate) fn adjust_cache_bytes(
    device: &Device,
    dtype: DType,
    available_bytes: usize,
    requested_cache_bytes: usize,
    minimum_cache_bytes: usize,
    may_reduce_cache: bool,
) -> Result<usize> {
    let Device::Cuda(dev) = device else {
        return Ok(requested_cache_bytes);
    };
    let reserved = reserved_bytes(device, dtype);
    if reserved == 0 {
        return Ok(requested_cache_bytes);
    }
    let stream = dev.cuda_stream();
    let context = stream.context();
    let total = context.mem_get_info().w()?.1;
    let integrated = context
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
        .w()?
        != 0;
    let planned = cache_bytes_with_sidecars(
        reserved,
        memory_headroom(total, integrated),
        available_bytes,
        requested_cache_bytes,
        minimum_cache_bytes,
        may_reduce_cache,
    );
    set_memory_budget(dev.id(), planned.map(|_| reserved).unwrap_or(0));
    Ok(planned.unwrap_or(requested_cache_bytes))
}

fn cache_bytes_with_sidecars(
    reserved: usize,
    headroom: usize,
    available_bytes: usize,
    requested_cache_bytes: usize,
    minimum_cache_bytes: usize,
    may_reduce_cache: bool,
) -> Option<usize> {
    let required_free = reserved.saturating_add(headroom);
    let max_cache_bytes = available_bytes.saturating_sub(required_free);
    let adjusted_cache_bytes = if may_reduce_cache {
        max_cache_bytes.min(requested_cache_bytes.saturating_sub(reserved))
    } else {
        requested_cache_bytes
    };
    let enabled = adjusted_cache_bytes <= max_cache_bytes
        && (!may_reduce_cache || adjusted_cache_bytes >= minimum_cache_bytes);
    enabled.then_some(adjusted_cache_bytes)
}

fn backend() -> Backend {
    static BACKEND: OnceLock<Backend> = OnceLock::new();
    *BACKEND.get_or_init(|| match std::env::var(BACKEND_ENV).as_deref() {
        Ok("off") => Backend::Off,
        _ => Backend::Auto,
    })
}

pub(crate) fn enabled() -> bool {
    cfg!(has_marlin_kernels) && backend() != Backend::Off
}

fn memory_headroom(total: usize, integrated: bool) -> usize {
    if integrated {
        MIN_MEMORY_HEADROOM
    } else {
        (total / MEMORY_HEADROOM_DIVISOR).max(MIN_MEMORY_HEADROOM)
    }
}

fn set_memory_budget(device_id: DeviceId, bytes: usize) {
    if let Some(state) = reservations().lock().unwrap().get_mut(&device_id) {
        state.memory_budget = Some(bytes);
    }
}

fn claim_memory_budget(
    device_id: DeviceId,
    bytes: usize,
    free: usize,
    headroom: usize,
) -> Option<bool> {
    let mut reservations = reservations().lock().unwrap();
    let state = reservations.get_mut(&device_id)?;
    if state.memory_budget.is_none() {
        let required = state.remaining_bytes.max(bytes);
        state.memory_budget = Some(if required <= free.saturating_sub(headroom) {
            required
        } else {
            0
        });
    }
    let budget = state
        .memory_budget
        .as_mut()
        .expect("memory budget initialized");
    if bytes > *budget {
        return Some(false);
    }
    *budget -= bytes;
    Some(true)
}

enum AffineParams {
    F16 {
        _scales: CudaSlice<half::f16>,
        scales_ptr: u64,
        _offsets: CudaSlice<half::f16>,
        offsets_ptr: u64,
    },
    Bf16 {
        _scales: CudaSlice<half::bf16>,
        scales_ptr: u64,
        _offsets: CudaSlice<half::bf16>,
        offsets_ptr: u64,
    },
}

impl AffineParams {
    fn dtype(&self) -> DType {
        match self {
            Self::F16 { .. } => DType::F16,
            Self::Bf16 { .. } => DType::BF16,
        }
    }
}

struct Workspace {
    _data: CudaSlice<u32>,
    ptr: u64,
    completion: CudaEvent,
}

pub(crate) struct PackedAffine {
    _payload: CudaSlice<u8>,
    payload_ptr: u64,
    params: AffineParams,
    // Graph replay reuses the capture thread's workspace; the engine serializes model forwards.
    workspaces: Mutex<HashMap<ThreadId, Workspace>>,
    ready: CudaEvent,
    workspace_len: usize,
    dev: CudaDevice,
    format: AffineFormatSpec,
    n: usize,
    padded_n: usize,
    k: usize,
}

impl fmt::Debug for PackedAffine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PackedAffine")
            .field("source_dtype", &self.format.source_dtype)
            .field("dtype", &self.params.dtype())
            .field("n", &self.n)
            .field("padded_n", &self.padded_n)
            .field("k", &self.k)
            .finish()
    }
}

impl Drop for PackedAffine {
    fn drop(&mut self) {
        let workspaces = self
            .workspaces
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for workspace in workspaces.values() {
            let _ = workspace.completion.synchronize();
        }
        let _ = self.ready.synchronize();
    }
}

impl PackedAffine {
    pub(crate) fn dtype(&self) -> DType {
        self.params.dtype()
    }

    pub(crate) fn supports(weight: &QTensor, dtype: DType) -> bool {
        matches!(dtype, DType::F16 | DType::BF16) && PackedAffinePlan::new(weight).is_some()
    }

    pub(crate) fn dtype_for_input(weight: &QTensor, input: &Tensor) -> Option<DType> {
        let dtype = match input.dtype() {
            DType::F16 | DType::BF16 => input.dtype(),
            DType::F32
                if AffineFormatSpec::for_dtype(weight.dtype())
                    .is_some_and(AffineFormatSpec::supports_f32_input) =>
            {
                DType::BF16
            }
            _ => return None,
        };
        if !Self::supports(weight, dtype) || !weight.device().same_device(input.device()) {
            return None;
        }
        let Ok((_, weight_k)) = weight.shape().dims2() else {
            return None;
        };
        input
            .dims()
            .split_last()
            .is_some_and(|(&input_k, _)| input_k == weight_k)
            .then_some(dtype)
    }

    pub(crate) fn new(
        weight: &QTensor,
        dtype: DType,
        reservation: Option<&Reservation>,
    ) -> Result<Self> {
        if !Self::supports(weight, dtype) {
            candle_core::bail!(
                "packed GGUF affine does not support {:?} {:?} {:?} on {:?}",
                weight.dtype(),
                weight.shape(),
                dtype,
                weight.device()
            );
        }
        let Device::Cuda(dev) = weight.device() else {
            unreachable!("supports() checked CUDA")
        };
        let plan = PackedAffinePlan::new(weight).expect("supports() checked packed plan");
        let PackedAffinePlan {
            format,
            n,
            padded_n,
            k,
            payload_bytes,
            metadata_values,
            workspace_len,
            total_bytes: required_bytes,
            ..
        } = plan;
        let stream = dev.cuda_stream();
        let context = stream.context();
        if let Some(reservation) = reservation {
            if reservation.plan != plan || reservation.device_id != dev.id() {
                candle_core::bail!("packed GGUF affine reservation does not match the weight");
            }
            stream.wait(&reservation.source_ready).w()?;
        }
        let (free, total) = context.mem_get_info().w()?;
        let integrated = context
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_INTEGRATED)
            .w()?
            != 0;
        let headroom = memory_headroom(total, integrated);
        let device_id = dev.id();
        let budget_claim = reservation
            .is_some()
            .then(|| claim_memory_budget(device_id, required_bytes, free, headroom))
            .flatten();
        if budget_claim == Some(false) {
            candle_core::bail!("packed GGUF affine memory plan is disabled for this device");
        }
        let available = if budget_claim == Some(true) {
            free
        } else {
            free.saturating_sub(headroom)
        };
        if required_bytes > available {
            set_memory_budget(device_id, 0);
            candle_core::bail!(
                "packed GGUF affine needs {required_bytes} bytes with {free} free and {headroom} reserved"
            );
        }
        let (weight_ptr, weight_guard) = weight.device_ptr_with_guard(&stream)?;

        let mut payload = unsafe { dev.alloc::<u8>(payload_bytes)? };
        let (payload_ptr, payload_guard) = slice_ptr_mut_on_stream(&mut payload, 0, &stream);
        let params = match dtype {
            DType::F16 => {
                let mut scales = unsafe { dev.alloc::<half::f16>(metadata_values)? };
                let mut offsets = unsafe { dev.alloc::<half::f16>(metadata_values)? };
                let (scales_ptr, scales_guard) = slice_ptr_mut_on_stream(&mut scales, 0, &stream);
                let (offsets_ptr, offsets_guard) =
                    slice_ptr_mut_on_stream(&mut offsets, 0, &stream);
                let status = unsafe {
                    ffi::mrs_gguf_affine_repack_f16(
                        format.format_code,
                        weight_ptr.cast(),
                        payload_ptr as *mut _,
                        scales_ptr as *mut _,
                        offsets_ptr as *mut _,
                        k as i32,
                        n as i32,
                        padded_n as i32,
                        stream.cu_stream() as usize,
                    )
                };
                check_status("repack F16", status)?;
                drop(scales_guard);
                drop(offsets_guard);
                AffineParams::F16 {
                    _scales: scales,
                    scales_ptr,
                    _offsets: offsets,
                    offsets_ptr,
                }
            }
            DType::BF16 => {
                let mut scales = unsafe { dev.alloc::<half::bf16>(metadata_values)? };
                let mut offsets = unsafe { dev.alloc::<half::bf16>(metadata_values)? };
                let (scales_ptr, scales_guard) = slice_ptr_mut_on_stream(&mut scales, 0, &stream);
                let (offsets_ptr, offsets_guard) =
                    slice_ptr_mut_on_stream(&mut offsets, 0, &stream);
                let status = unsafe {
                    ffi::mrs_gguf_affine_repack_bf16(
                        format.format_code,
                        weight_ptr.cast(),
                        payload_ptr as *mut _,
                        scales_ptr as *mut _,
                        offsets_ptr as *mut _,
                        k as i32,
                        n as i32,
                        padded_n as i32,
                        stream.cu_stream() as usize,
                    )
                };
                check_status("repack BF16", status)?;
                drop(scales_guard);
                drop(offsets_guard);
                AffineParams::Bf16 {
                    _scales: scales,
                    scales_ptr,
                    _offsets: offsets,
                    offsets_ptr,
                }
            }
            _ => unreachable!("supports() checked activation dtype"),
        };
        drop(payload_guard);
        drop(weight_guard);
        let ready = context.new_event(None).w()?;
        ready.record(&stream).w()?;
        let workspace = Self::new_workspace(&dev, workspace_len, &stream, &ready)?;
        let workspaces = HashMap::from([(std::thread::current().id(), workspace)]);
        Ok(Self {
            _payload: payload,
            payload_ptr,
            params,
            workspaces: Mutex::new(workspaces),
            ready,
            workspace_len,
            dev,
            format,
            n,
            padded_n,
            k,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        if !xs.device().same_device(&Device::Cuda(self.dev.clone())) {
            candle_core::bail!("packed GGUF affine input and weight are on different devices");
        }
        if xs.dtype() != self.params.dtype() {
            candle_core::bail!(
                "packed GGUF affine parameter dtype {:?} does not match input {:?}",
                self.params.dtype(),
                xs.dtype()
            );
        }
        let Some((&k, batch_dims)) = xs.dims().split_last() else {
            candle_core::bail!("packed GGUF affine input must have at least one dimension");
        };
        let m = batch_dims.iter().product::<usize>();
        if m == 0 || k != self.k || i32::try_from(m).is_err() {
            candle_core::bail!(
                "packed GGUF affine shape mismatch: input {:?}, weight [{}, {}]",
                xs.shape(),
                self.n,
                self.k
            );
        }

        let xs = xs.contiguous()?;
        let offset_bytes = {
            let (_storage, layout) = xs.storage_and_layout();
            layout
                .start_offset()
                .checked_mul(xs.dtype().size_in_bytes())
                .ok_or_else(|| {
                    candle_core::Error::Msg("packed GGUF affine input offset overflow".into())
                })?
        };
        let xs = if offset_bytes.is_multiple_of(MARLIN_INPUT_ALIGNMENT) {
            xs
        } else {
            xs.force_contiguous()?
        };
        let (storage, layout) = xs.storage_and_layout();
        let Storage::Cuda(storage) = &*storage else {
            candle_core::bail!("packed GGUF affine input must live on CUDA");
        };
        match &self.params {
            AffineParams::F16 {
                scales_ptr,
                offsets_ptr,
                ..
            } => self.forward_t(
                storage.as_cuda_slice::<half::f16>()?,
                layout.start_offset(),
                *scales_ptr,
                *offsets_ptr,
                m,
                &xs,
            ),
            AffineParams::Bf16 {
                scales_ptr,
                offsets_ptr,
                ..
            } => self.forward_t(
                storage.as_cuda_slice::<half::bf16>()?,
                layout.start_offset(),
                *scales_ptr,
                *offsets_ptr,
                m,
                &xs,
            ),
        }
    }

    fn forward_t<T: CudaDType + DeviceRepr>(
        &self,
        input: &CudaSlice<T>,
        input_offset: usize,
        scales_ptr: u64,
        offsets_ptr: u64,
        m: usize,
        xs: &Tensor,
    ) -> Result<Tensor> {
        let stream = self.dev.cuda_stream();
        let (input_ptr, _input_guard) = slice_ptr_on_stream(input, input_offset, &stream);
        let mut workspaces = self.workspaces.lock().unwrap();
        let thread_id = std::thread::current().id();
        let workspace =
            match workspaces.entry(thread_id) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => entry.insert(
                    Self::new_workspace(&self.dev, self.workspace_len, &stream, &self.ready)?,
                ),
            };
        let mut output = unsafe { self.dev.alloc::<T>(m * self.padded_n)? };
        let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);

        let status = unsafe {
            match (self.params.dtype(), self.format.payload_bits) {
                (DType::F16, 4) => ffi::marlin_affine_u4_f16(
                    input_ptr as *const _,
                    self.payload_ptr as *const _,
                    scales_ptr as *mut _,
                    offsets_ptr as *mut _,
                    output_ptr as *mut _,
                    m as i32,
                    self.k as i32,
                    self.padded_n as i32,
                    self.format.group_size as i32,
                    workspace.ptr as *mut _,
                    stream.cu_stream() as i64,
                ),
                (DType::BF16, 4) => ffi::marlin_affine_u4_bf16(
                    input_ptr as *const _,
                    self.payload_ptr as *const _,
                    scales_ptr as *mut _,
                    offsets_ptr as *mut _,
                    output_ptr as *mut _,
                    m as i32,
                    self.k as i32,
                    self.padded_n as i32,
                    self.format.group_size as i32,
                    workspace.ptr as *mut _,
                    stream.cu_stream() as i64,
                ),
                (DType::F16, 8) => ffi::marlin_affine_u8_f16(
                    input_ptr as *const _,
                    self.payload_ptr as *const _,
                    scales_ptr as *mut _,
                    offsets_ptr as *mut _,
                    output_ptr as *mut _,
                    m as i32,
                    self.k as i32,
                    self.padded_n as i32,
                    self.format.group_size as i32,
                    workspace.ptr as *mut _,
                    stream.cu_stream() as i64,
                ),
                (DType::BF16, 8) => ffi::marlin_affine_u8_bf16(
                    input_ptr as *const _,
                    self.payload_ptr as *const _,
                    scales_ptr as *mut _,
                    offsets_ptr as *mut _,
                    output_ptr as *mut _,
                    m as i32,
                    self.k as i32,
                    self.padded_n as i32,
                    self.format.group_size as i32,
                    workspace.ptr as *mut _,
                    stream.cu_stream() as i64,
                ),
                _ => unreachable!(),
            }
        };
        workspace.completion.record(&stream).w()?;
        check_status("Marlin matmul", status)?;

        drop(output_guard);
        let mut padded_dims = xs.dims().to_vec();
        *padded_dims.last_mut().expect("input has a final dimension") = self.padded_n;
        let output = Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(output, self.dev.clone())),
            Shape::from(padded_dims),
        ));
        if self.padded_n == self.n {
            Ok(output)
        } else {
            output.narrow(output.rank() - 1, 0, self.n)?.contiguous()
        }
    }

    fn new_workspace(
        dev: &CudaDevice,
        len: usize,
        stream: &CudaStream,
        ready: &CudaEvent,
    ) -> Result<Workspace> {
        stream.wait(ready).w()?;
        let mut data = dev.alloc_zeros::<u32>(len)?;
        let (ptr, guard) = slice_ptr_mut_on_stream(&mut data, 0, stream);
        drop(guard);
        Ok(Workspace {
            _data: data,
            ptr,
            completion: stream.context().new_event(None).w()?,
        })
    }
}

fn supports_device(device: &Device) -> bool {
    let Device::Cuda(dev) = device else {
        return false;
    };
    static SUPPORTED: OnceLock<Mutex<HashMap<usize, bool>>> = OnceLock::new();
    let ordinal = dev.cuda_stream().context().ordinal();
    let mut supported = SUPPORTED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    *supported.entry(ordinal).or_insert_with(|| {
        dev.cuda_stream()
            .context()
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .is_ok_and(|major| major >= MIN_COMPUTE_CAPABILITY_MAJOR)
    })
}

fn check_status(operation: &str, status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        candle_core::bail!("packed GGUF affine {operation} failed with CUDA status {status}")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use candle_core::{quantized::QTensor, DType, Device};

    use super::super::GGUF_AFFINE_MIN_BATCH;
    use super::*;
    use crate::{
        try_fused_quantized_ffn, try_fused_quantized_gate_up, try_fused_quantized_qkv, GgufMatMul,
        GluActivationType, QuantMethod,
    };

    const AFFINE_DTYPES: &[GgmlDType] = &[
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q8_1,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8K,
    ];

    fn patterned(rows: usize, cols: usize, seed: usize, scale: f32) -> Result<Tensor> {
        let values = (0..rows * cols)
            .map(|index| {
                let phase = (index.wrapping_mul(37).wrapping_add(seed * 17) % 251) as f32;
                (phase * 0.071).sin() * scale
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows, cols), &Device::Cpu)
    }

    fn assert_close(
        actual: &Tensor,
        expected: &Tensor,
        max_limit: f32,
        mean_limit: f32,
    ) -> Result<()> {
        let actual = actual
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let expected = expected
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        assert_values_close(&actual, &expected, max_limit, mean_limit)
    }

    fn assert_values_close(
        actual: &[f32],
        expected: &[f32],
        max_limit: f32,
        mean_limit: f32,
    ) -> Result<()> {
        assert_eq!(actual.len(), expected.len());
        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        for (actual, expected) in actual.iter().zip(expected) {
            let diff = (actual - expected).abs();
            max_diff = max_diff.max(diff);
            sum_diff += diff;
        }
        let mean_diff = sum_diff / actual.len() as f32;
        assert!(
            max_diff <= max_limit && mean_diff <= mean_limit,
            "max diff {max_diff}, mean diff {mean_diff}"
        );
        Ok(())
    }

    #[test]
    fn cache_plan_counts_sidecars_inside_utilization_budget() {
        assert_eq!(cache_bytes_with_sidecars(4, 1, 19, 16, 2, true), Some(12));
        assert_eq!(cache_bytes_with_sidecars(4, 1, 19, 13, 2, false), Some(13));
        assert_eq!(cache_bytes_with_sidecars(4, 1, 19, 15, 2, false), None);
        assert_eq!(cache_bytes_with_sidecars(4, 2, 6, 5, 1, true), None);
    }

    #[test]
    fn reservation_dtype_accounting_tracks_f32_capable_formats() {
        const Q4_BYTES: usize = 11;
        const Q8_BYTES: usize = 17;

        let q4 = AffineFormatSpec::for_dtype(GgmlDType::Q4K).expect("Q4_K format");
        let q8 = AffineFormatSpec::for_dtype(GgmlDType::Q8K).expect("Q8_K format");
        let mut state = DeviceReservations::default();
        state.reserve(q4, Q4_BYTES);
        state.reserve(q8, Q8_BYTES);
        assert_eq!(state.bytes_for_dtype(DType::F16), Q4_BYTES + Q8_BYTES);
        assert_eq!(state.bytes_for_dtype(DType::BF16), Q4_BYTES + Q8_BYTES);
        assert_eq!(state.bytes_for_dtype(DType::F32), Q8_BYTES);
        assert_eq!(state.bytes_for_dtype(DType::U8), 0);
        state.release(q8, Q8_BYTES);
        assert_eq!(state.bytes_for_dtype(DType::F32), 0);
        assert_eq!(state.bytes_for_dtype(DType::BF16), Q4_BYTES);
    }

    #[test]
    fn format_specs_cover_all_quantized_gguf_types() {
        let actual = AFFINE_DTYPES
            .iter()
            .map(|dtype| {
                let spec = AffineFormatSpec::for_dtype(*dtype).expect("affine format");
                (
                    spec.source_dtype,
                    spec.format_code,
                    spec.source_block_size,
                    spec.payload_bits,
                    spec.group_size,
                    spec.min_batch,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            vec![
                (GgmlDType::Q4_0, 2, 32, 4, 32, GGUF_AFFINE_MIN_BATCH),
                (GgmlDType::Q4_1, 3, 32, 4, 32, GGUF_AFFINE_MIN_BATCH),
                (GgmlDType::Q5_0, 6, 32, 8, 32, Q5_0_MIN_BATCH),
                (GgmlDType::Q5_1, 7, 32, 8, 32, Q5_1_MIN_BATCH),
                (GgmlDType::Q8_0, 8, 32, 8, 32, GGUF_AFFINE_MIN_BATCH),
                (GgmlDType::Q8_1, 9, 32, 8, 32, AFFINE_ONLY_MIN_BATCH),
                (GgmlDType::Q2K, 10, 256, 4, 16, GGUF_AFFINE_MIN_BATCH),
                (GgmlDType::Q3K, 11, 256, 4, 16, GGUF_AFFINE_MIN_BATCH),
                (GgmlDType::Q4K, 12, 256, 4, 32, GGUF_AFFINE_MIN_BATCH),
                (GgmlDType::Q5K, 13, 256, 8, 32, GGUF_AFFINE_MIN_BATCH),
                (GgmlDType::Q6K, 14, 256, 8, 16, Q6K_MIN_BATCH),
                (GgmlDType::Q8K, 15, 256, 8, 32, AFFINE_ONLY_MIN_BATCH),
            ]
        );
        assert!(AffineFormatSpec::for_dtype(GgmlDType::F32).is_none());
        assert!(AffineFormatSpec::for_dtype(GgmlDType::F16).is_none());
        assert!(AffineFormatSpec::for_dtype(GgmlDType::BF16).is_none());
    }

    #[test]
    fn marlin_shape_filter_matches_available_tiles() {
        assert!(supports_marlin_shape(64, 128));
        assert!(supports_marlin_shape(128, 64));
        assert!(supports_marlin_shape(256, 64));
        assert!(!supports_marlin_shape(64, 64));
        assert!(!supports_marlin_shape(64, 96));
        assert!(!supports_marlin_shape(96, 128));
        assert!(!supports_marlin_shape(192, 64));
        assert_eq!(padded_n_for_shape(96, 256), Some(128));
        assert_eq!(padded_n_for_shape(64, 64), Some(128));
        assert_eq!(padded_n_for_shape(129, 128), Some(192));
        assert_eq!(padded_n_for_shape(64, 32), None);
    }

    fn run_case(source_dtype: GgmlDType, dtype: DType, m: usize, n: usize, k: usize) -> Result<()> {
        let cuda = Device::new_cuda(0)?;
        let weight = QTensor::quantize_onto(&patterned(n, k, 11, 0.04)?, source_dtype, &cuda)?;
        let xs = patterned(m, k, 29, 0.1)?
            .reshape((1, m, k))?
            .to_dtype(dtype)?
            .to_device(&cuda)?;
        let packed = PackedAffine::new(&weight, dtype, None)?;
        let actual = packed.forward(&xs)?;
        let dequantized = weight.dequantize(&cuda)?.to_dtype(dtype)?;
        let expected = xs
            .reshape((m, k))?
            .matmul(&dequantized.t()?)?
            .reshape((1, m, n))?;
        assert_close(&actual, &expected, 0.08, 0.01)
    }

    #[test]
    fn marlin_matches_dequantized_q4k() -> Result<()> {
        for &(m, n, k) in &[
            (1, 64, 256),
            (8, 128, 256),
            (16, 128, 256),
            (17, 192, 256),
            (33, 256, 256),
            (49, 320, 256),
            (65, 64, 512),
            (127, 192, 512),
        ] {
            run_case(GgmlDType::Q4K, DType::F16, m, n, k)?;
            run_case(GgmlDType::Q4K, DType::BF16, m, n, k)?;
        }
        Ok(())
    }

    #[test]
    fn marlin_matches_dequantized_all_affine_formats() -> Result<()> {
        for &source_dtype in AFFINE_DTYPES {
            run_case(source_dtype, DType::F16, 17, 128, 256)?;
            run_case(source_dtype, DType::BF16, 17, 128, 256)?;
            run_case(source_dtype, DType::F16, 17, 96, 256)?;
            run_case(source_dtype, DType::BF16, 17, 96, 256)?;
        }
        Ok(())
    }

    #[test]
    fn marlin_lock_workspace_matches_dequantized_q4k() -> Result<()> {
        run_case(GgmlDType::Q4K, DType::F16, 512, 192, 512)?;
        run_case(GgmlDType::Q4K, DType::BF16, 512, 192, 512)
    }

    #[test]
    fn gguf_dispatch_preserves_bias_and_reuses_pack() -> Result<()> {
        const M: usize = 17;
        const N: usize = 128;
        const K: usize = 256;

        let cuda = Device::new_cuda(0)?;
        let weight = QTensor::quantize_onto(&patterned(N, K, 41, 0.04)?, GgmlDType::Q4K, &cuda)?;
        let dequantized = weight.dequantize(&cuda)?.to_dtype(DType::BF16)?;
        let bias = patterned(1, N, 53, 0.01)?
            .reshape(N)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;
        let layer = GgufMatMul::from_qtensor(weight, Some(bias.clone()));
        let xs = patterned(K, M, 67, 0.1)?
            .reshape((1, K, M))?
            .transpose(1, 2)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;
        let expected = xs
            .contiguous()?
            .reshape((M, K))?
            .matmul(&dequantized.t()?)?
            .reshape((1, M, N))?
            .broadcast_add(&bias)?;

        let actual = layer.forward_raw(&xs)?;
        assert_close(&actual, &expected, 0.08, 0.01)?;
        let first = Arc::as_ptr(
            layer
                .packed_affine
                .get()
                .and_then(Option::as_ref)
                .expect("packed Q4_K cache"),
        );
        let repeated = layer.forward_raw(&xs)?;
        assert_close(&repeated, &expected, 0.08, 0.01)?;
        let second = Arc::as_ptr(
            layer
                .packed_affine
                .get()
                .and_then(Option::as_ref)
                .expect("packed Q4_K cache"),
        );
        assert_eq!(first, second);
        Ok(())
    }

    #[test]
    fn direct_quant_method_dispatch_reaches_affine_only_formats_at_batch_one() -> Result<()> {
        const M: usize = AFFINE_ONLY_MIN_BATCH;
        const N: usize = 128;
        const K: usize = 256;

        let cuda = Device::new_cuda(0)?;
        for source_dtype in [GgmlDType::Q8_1, GgmlDType::Q8K] {
            let weight = QTensor::quantize_onto(&patterned(N, K, 59, 0.04)?, source_dtype, &cuda)?;
            let dequantized_f32 = weight.dequantize(&cuda)?;
            let dequantized = dequantized_f32.to_dtype(DType::BF16)?;
            let layer = GgufMatMul::from_qtensor(weight, None);
            if source_dtype == GgmlDType::Q8_1 {
                assert_close(
                    &QuantMethod::dequantize_w(&layer)?,
                    &dequantized_f32,
                    0.0,
                    0.0,
                )?;
            }

            let xs = patterned(M, K, 61, 0.1)?
                .to_dtype(DType::BF16)?
                .to_device(&cuda)?;
            let actual = QuantMethod::forward(&layer, &xs)?;
            let expected = xs.matmul(&dequantized.t()?)?;
            assert_close(&actual, &expected, 0.08, 0.01)?;
            assert!(layer.packed_affine.get().is_some_and(Option::is_some));

            let xs_f16 = patterned(M, K, 63, 0.1)?
                .to_dtype(DType::F16)?
                .to_device(&cuda)?;
            let actual_f16 = QuantMethod::forward(&layer, &xs_f16)?;
            let expected_f16 = xs_f16.matmul(&dequantized_f32.to_dtype(DType::F16)?.t()?)?;
            assert_eq!(actual_f16.dtype(), DType::F16);
            assert_close(&actual_f16, &expected_f16, 0.08, 0.01)?;

            let xs_f32 = patterned(M, K, 65, 0.1)?.to_device(&cuda)?;
            let actual_f32 = QuantMethod::forward(&layer, &xs_f32)?;
            let expected_f32 = xs_f32.matmul(&dequantized_f32.t()?)?;
            assert_eq!(actual_f32.dtype(), DType::F32);
            assert_close(&actual_f32, &expected_f32, 0.08, 0.01)?;
        }
        Ok(())
    }

    #[test]
    fn unaligned_width_uses_padded_packed_dispatch() -> Result<()> {
        const M: usize = 17;
        const N: usize = 96;
        const K: usize = 256;

        let cuda = Device::new_cuda(0)?;
        let xs = patterned(M, K, 83, 0.1)?
            .reshape((1, M, K))?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;
        for source_dtype in [GgmlDType::Q8_1, GgmlDType::Q8K] {
            let weight = QTensor::quantize_onto(&patterned(N, K, 71, 0.04)?, source_dtype, &cuda)?;
            let dequantized = weight.dequantize(&cuda)?.to_dtype(DType::BF16)?;
            let layer = GgufMatMul::from_qtensor(weight, None);
            let actual = QuantMethod::forward(&layer, &xs)?;
            let expected = xs
                .reshape((M, K))?
                .matmul(&dequantized.t()?)?
                .reshape((1, M, N))?;
            assert_close(&actual, &expected, 0.08, 0.01)?;
            assert!(layer.packed_affine.get().is_some_and(Option::is_some));
        }
        Ok(())
    }

    #[test]
    fn unsupported_k_tile_uses_canonical_dispatch() -> Result<()> {
        const M: usize = 17;
        const N: usize = 64;
        const K: usize = 32;

        let cuda = Device::new_cuda(0)?;
        let weight = QTensor::quantize_onto(&patterned(N, K, 73, 0.04)?, GgmlDType::Q4_0, &cuda)?;
        let dequantized = weight.dequantize(&cuda)?.to_dtype(DType::BF16)?;
        let layer = GgufMatMul::from_qtensor(weight, None);
        let xs = patterned(M, K, 77, 0.1)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;
        let actual = layer.forward_raw(&xs)?;
        let expected = xs.matmul(&dequantized.t()?)?;
        assert_close(&actual, &expected, 0.08, 0.01)?;
        assert!(layer.packed_affine.get().is_none());
        Ok(())
    }

    #[test]
    fn affine_only_unsupported_k_reports_required_backend() -> Result<()> {
        const M: usize = AFFINE_ONLY_MIN_BATCH;
        const N: usize = 64;
        const K: usize = 32;

        let cuda = Device::new_cuda(0)?;
        let weight = QTensor::quantize_onto(&patterned(N, K, 75, 0.04)?, GgmlDType::Q8_1, &cuda)?;
        let layer = GgufMatMul::from_qtensor(weight, None);
        let xs = patterned(M, K, 77, 0.1)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;
        let error =
            QuantMethod::forward(&layer, &xs).expect_err("Q8_1 requires packed dispatch on CUDA");
        assert!(
            error
                .to_string()
                .contains("require the packed GGUF affine backend"),
            "unexpected error: {error}"
        );
        Ok(())
    }

    #[test]
    fn dispatch_switches_to_packed_at_minimum_batch() -> Result<()> {
        const N: usize = 128;
        const K: usize = 256;

        let cuda = Device::new_cuda(0)?;
        let weight = QTensor::quantize_onto(&patterned(N, K, 79, 0.04)?, GgmlDType::Q4K, &cuda)?;
        let layer = GgufMatMul::from_qtensor(weight, None);

        let small = patterned(GGUF_AFFINE_MIN_BATCH - 1, K, 81, 0.1)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;
        layer.forward_raw(&small)?;
        assert!(layer.packed_affine.get().is_none());

        let packed = patterned(GGUF_AFFINE_MIN_BATCH, K, 83, 0.1)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;
        layer.forward_raw(&packed)?;
        assert!(layer.packed_affine.get().is_some_and(Option::is_some));
        Ok(())
    }

    #[test]
    fn format_specific_minimum_batches_are_enforced() -> Result<()> {
        const N: usize = 128;
        const K: usize = 256;

        let cuda = Device::new_cuda(0)?;
        for (source_dtype, min_batch) in [
            (GgmlDType::Q5_0, Q5_0_MIN_BATCH),
            (GgmlDType::Q5_1, Q5_1_MIN_BATCH),
            (GgmlDType::Q6K, Q6K_MIN_BATCH),
        ] {
            let weight = QTensor::quantize_onto(&patterned(N, K, 85, 0.04)?, source_dtype, &cuda)?;
            let layer = GgufMatMul::from_qtensor(weight, None);

            let below = patterned(min_batch - 1, K, 87, 0.1)?
                .to_dtype(DType::BF16)?
                .to_device(&cuda)?;
            layer.forward_raw(&below)?;
            assert!(layer.packed_affine.get().is_none());

            let eligible = patterned(min_batch, K, 89, 0.1)?
                .to_dtype(DType::BF16)?
                .to_device(&cuda)?;
            layer.forward_raw(&eligible)?;
            assert!(layer.packed_affine.get().is_some_and(Option::is_some));
        }
        Ok(())
    }

    #[test]
    fn aligned_copy_handles_contiguous_offset_view() -> Result<()> {
        const M: usize = 17;
        const N: usize = 128;
        const K: usize = 256;

        let cuda = Device::new_cuda(0)?;
        let weight = QTensor::quantize_onto(&patterned(N, K, 89, 0.04)?, GgmlDType::Q4K, &cuda)?;
        let dequantized = weight.dequantize(&cuda)?.to_dtype(DType::BF16)?;
        let xs = patterned(1, M * K + 1, 91, 0.1)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?
            .narrow(1, 1, M * K)?
            .reshape((M, K))?;
        assert!(xs.is_contiguous());
        assert_eq!(xs.layout().start_offset(), 1);

        let actual = PackedAffine::new(&weight, DType::BF16, None)?.forward(&xs)?;
        let expected = xs.matmul(&dequantized.t()?)?;
        assert_close(&actual, &expected, 0.08, 0.01)
    }

    #[test]
    fn concurrent_per_thread_streams_use_distinct_workspaces() -> Result<()> {
        const M: usize = 17;
        const N: usize = 128;
        const K: usize = 256;

        let cuda = Device::new_cuda(0)?;
        let source_weight = patterned(N, K, 109, 0.04)?;
        let reference_weight = QTensor::quantize(&source_weight, GgmlDType::Q4K)?;
        let weight = QTensor::quantize_onto(&source_weight, GgmlDType::Q4K, &cuda)?;
        let layer = Arc::new(GgufMatMul::from_qtensor(weight, None));
        let xs = Arc::new(patterned(M, K, 113, 0.1)?.to_dtype(DType::BF16)?);
        let expected = Arc::new(
            xs.to_dtype(DType::F32)?
                .matmul(&reference_weight.dequantize(&Device::Cpu)?.t()?)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        );
        let barrier = Arc::new(Barrier::new(2));

        std::thread::scope(|scope| -> Result<()> {
            let mut handles = Vec::new();
            for _ in 0..2 {
                let layer = layer.clone();
                let xs = xs.clone();
                let expected = expected.clone();
                let barrier = barrier.clone();
                let device = cuda.clone();
                handles.push(scope.spawn(move || -> Result<()> {
                    let xs = xs.to_device(&device)?;
                    let Device::Cuda(cuda) = xs.device() else {
                        unreachable!()
                    };
                    let stream = cuda.cuda_stream();
                    barrier.wait();
                    for _ in 0..4 {
                        let actual = layer.forward_raw(&xs)?;
                        let actual_f32 = actual.flatten_all()?.to_dtype(DType::F32)?;
                        let actual_values = actual_f32.to_vec1::<f32>()?;
                        stream.synchronize().w()?;
                        assert_values_close(&actual_values, &expected, 0.08, 0.01)?;
                    }
                    stream.synchronize().w()?;
                    Ok(())
                }));
            }
            for handle in handles {
                handle.join().expect("packed Q4_K worker")?;
            }
            Ok(())
        })?;
        let Device::Cuda(cuda) = cuda else {
            unreachable!()
        };
        cuda.cuda_stream().context().synchronize().w()
    }

    #[test]
    fn packing_consumes_paged_attention_reservation() -> Result<()> {
        const M: usize = 17;
        const N: usize = 128;
        const K: usize = 256;

        let cuda = Device::new_cuda(0)?;
        let weight = QTensor::quantize_onto(&patterned(N, K, 127, 0.04)?, GgmlDType::Q8K, &cuda)?;
        let layer = GgufMatMul::from_qtensor(weight, None);
        let reservation = layer
            ._gguf_affine_reservation
            .as_ref()
            .expect("packed GGUF affine reservation");
        let payload_bytes = N * K;
        let metadata_bytes = (K / 32) * N * std::mem::size_of::<half::f16>();
        let workspace_bytes =
            (N / MARLIN_N_TILE) * MARLIN_MAX_PARALLEL * std::mem::size_of::<u32>();
        assert_eq!(
            reservation.bytes,
            payload_bytes + 2 * metadata_bytes + workspace_bytes
        );
        assert!(reservation.pending.load(Ordering::Acquire));
        assert!(reserved_bytes(&cuda, DType::F32) >= reservation.bytes);
        let xs = patterned(M, K, 131, 0.1)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;
        layer.forward_raw(&xs)?;
        assert!(!reservation.pending.load(Ordering::Acquire));
        Ok(())
    }

    #[test]
    fn fused_ffn_prepares_different_down_shape() -> Result<()> {
        const M: usize = Q6K_MIN_BATCH;
        const HIDDEN: usize = 256;
        const INTERMEDIATE: usize = 512;

        let cuda = Device::new_cuda(0)?;
        let gate_weight = QTensor::quantize_onto(
            &patterned(INTERMEDIATE, HIDDEN, 97, 0.04)?,
            GgmlDType::Q3K,
            &cuda,
        )?;
        let up_weight = QTensor::quantize_onto(
            &patterned(INTERMEDIATE, HIDDEN, 101, 0.04)?,
            GgmlDType::Q5K,
            &cuda,
        )?;
        let down_weight = QTensor::quantize_onto(
            &patterned(HIDDEN, INTERMEDIATE, 103, 0.04)?,
            GgmlDType::Q6K,
            &cuda,
        )?;
        let gate_dequantized = gate_weight.dequantize(&cuda)?.to_dtype(DType::BF16)?;
        let up_dequantized = up_weight.dequantize(&cuda)?.to_dtype(DType::BF16)?;
        let down_dequantized = down_weight.dequantize(&cuda)?.to_dtype(DType::BF16)?;
        let gate = GgufMatMul::from_qtensor(gate_weight, None);
        let up = GgufMatMul::from_qtensor(up_weight, None);
        let down = GgufMatMul::from_qtensor(down_weight, None);
        let xs = patterned(M, HIDDEN, 107, 0.1)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;

        let actual = try_fused_quantized_ffn(&xs, &gate, &up, &down, GluActivationType::Silu)?
            .expect("mixed packed GGUF affine fused FFN");
        let gate_expected = xs.matmul(&gate_dequantized.t()?)?;
        let up_expected = xs.matmul(&up_dequantized.t()?)?;
        let intermediate = crate::fused_glu(&gate_expected, &up_expected, GluActivationType::Silu)?;
        let expected = intermediate.matmul(&down_dequantized.t()?)?;

        assert_close(&actual, &expected, 0.2, 0.03)?;
        assert!(gate.packed_affine.get().is_some_and(Option::is_some));
        assert!(up.packed_affine.get().is_some_and(Option::is_some));
        assert!(down.packed_affine.get().is_some_and(Option::is_some));
        Ok(())
    }

    #[test]
    fn fused_qkv_and_gate_up_use_packed_layers() -> Result<()> {
        const M: usize = 17;
        const Q_OUT: usize = 256;
        const KV_OUT: usize = 64;
        const K: usize = 256;

        let cuda = Device::new_cuda(0)?;
        let q_weight =
            QTensor::quantize_onto(&patterned(Q_OUT, K, 137, 0.04)?, GgmlDType::Q2K, &cuda)?;
        let k_weight =
            QTensor::quantize_onto(&patterned(KV_OUT, K, 139, 0.04)?, GgmlDType::Q5K, &cuda)?;
        let v_weight =
            QTensor::quantize_onto(&patterned(KV_OUT, K, 149, 0.04)?, GgmlDType::Q8_0, &cuda)?;
        let q_dequantized = q_weight.dequantize(&cuda)?.to_dtype(DType::BF16)?;
        let k_dequantized = k_weight.dequantize(&cuda)?.to_dtype(DType::BF16)?;
        let v_dequantized = v_weight.dequantize(&cuda)?.to_dtype(DType::BF16)?;
        let q = GgufMatMul::from_qtensor(q_weight, None);
        let k = GgufMatMul::from_qtensor(k_weight, None);
        let v = GgufMatMul::from_qtensor(v_weight, None);
        let xs = patterned(M, K, 151, 0.1)?
            .to_dtype(DType::BF16)?
            .to_device(&cuda)?;

        let (q_actual, k_actual, v_actual) =
            try_fused_quantized_qkv(&xs, &q, &k, &v)?.expect("mixed packed GGUF affine fused QKV");
        assert_close(&q_actual, &xs.matmul(&q_dequantized.t()?)?, 0.08, 0.01)?;
        assert_close(&k_actual, &xs.matmul(&k_dequantized.t()?)?, 0.08, 0.01)?;
        assert_close(&v_actual, &xs.matmul(&v_dequantized.t()?)?, 0.08, 0.01)?;

        let gate_up = try_fused_quantized_gate_up(&xs, &k, &v, GluActivationType::Silu)?
            .expect("mixed packed GGUF affine fused gate-up");
        let gate_expected = xs.matmul(&k_dequantized.t()?)?;
        let up_expected = xs.matmul(&v_dequantized.t()?)?;
        let expected = crate::fused_glu(&gate_expected, &up_expected, GluActivationType::Silu)?;
        assert_close(&gate_up, &expected, 0.08, 0.01)?;
        assert!(q.packed_affine.get().is_some_and(Option::is_some));
        assert!(k.packed_affine.get().is_some_and(Option::is_some));
        assert!(v.packed_affine.get().is_some_and(Option::is_some));
        Ok(())
    }
}

#[cfg(has_marlin_kernels)]
mod ffi {
    use std::ffi::c_void;

    extern "C" {
        pub fn mrs_gguf_affine_repack_f16(
            format: i32,
            source: *const c_void,
            payload: *mut c_void,
            scales: *mut c_void,
            offsets: *mut c_void,
            k: i32,
            n: i32,
            padded_n: i32,
            stream: usize,
        ) -> i32;
        pub fn mrs_gguf_affine_repack_bf16(
            format: i32,
            source: *const c_void,
            payload: *mut c_void,
            scales: *mut c_void,
            offsets: *mut c_void,
            k: i32,
            n: i32,
            padded_n: i32,
            stream: usize,
        ) -> i32;
        pub fn marlin_affine_u4_f16(
            input: *const c_void,
            weight: *const c_void,
            scales: *mut c_void,
            offsets: *mut c_void,
            output: *mut c_void,
            m: i32,
            k: i32,
            n: i32,
            group_size: i32,
            workspace: *mut c_void,
            stream: i64,
        ) -> i32;
        pub fn marlin_affine_u4_bf16(
            input: *const c_void,
            weight: *const c_void,
            scales: *mut c_void,
            offsets: *mut c_void,
            output: *mut c_void,
            m: i32,
            k: i32,
            n: i32,
            group_size: i32,
            workspace: *mut c_void,
            stream: i64,
        ) -> i32;
        pub fn marlin_affine_u8_f16(
            input: *const c_void,
            weight: *const c_void,
            scales: *mut c_void,
            offsets: *mut c_void,
            output: *mut c_void,
            m: i32,
            k: i32,
            n: i32,
            group_size: i32,
            workspace: *mut c_void,
            stream: i64,
        ) -> i32;
        pub fn marlin_affine_u8_bf16(
            input: *const c_void,
            weight: *const c_void,
            scales: *mut c_void,
            offsets: *mut c_void,
            output: *mut c_void,
            m: i32,
            k: i32,
            n: i32,
            group_size: i32,
            workspace: *mut c_void,
            stream: i64,
        ) -> i32;
    }
}
