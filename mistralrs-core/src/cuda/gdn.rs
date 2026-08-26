#![allow(clippy::cast_possible_truncation)]

use candle_core::{DType, Device, Result, Tensor};

use crate::kv_cache::RecurrentStateLayout;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const GDN_PAD_SLOT: u32 = u32::MAX;

#[cfg(any(feature = "cuda", test))]
const GDN_DECODE_MIN_COMPUTE_MAJOR: i32 = 9;
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
const GDN_DECODE_TUNED_COMPUTE_MAJOR: i32 = 9;
pub(crate) const GDN_DECODE_K_DIM: usize = 128;
pub(crate) const GDN_DECODE_V_DIM: usize = 128;
#[cfg(feature = "cuda")]
const GDN_DECODE_FALLBACK_MAX_K: usize = 256;
#[cfg(any(feature = "cuda", test))]
const GDN_DECODE_V_MAJOR_LARGE_TILE: usize = 32;
#[cfg(any(feature = "cuda", test))]
const GDN_DECODE_V_MAJOR_LARGE_CTA_WAVES: usize = 8;
#[cfg(any(feature = "cuda", test))]
const GDN_DECODE_COOPERATIVE_V_TILE: usize = 16;
#[cfg(any(feature = "cuda", test))]
const GDN_DECODE_PIPELINED_V_TILE: usize = 32;
#[cfg(any(feature = "cuda", test))]
const GDN_DECODE_COOPERATIVE_STATE_WAVES: usize = 2;
#[cfg(any(feature = "cuda", test))]
const GDN_DECODE_PIPELINED_OCCUPANCY_WAVES: usize = 4;
#[cfg(any(feature = "cuda", test))]
const GDN_DECODE_PIPELINED_AMORTIZED_WAVES: usize = 8;
#[cfg(feature = "cuda")]
const GDN_DECODE_VECTOR_ALIGNMENT: usize = 16;
#[cfg(feature = "cuda")]
const GDN_DECODE_INPUT_ALIGNMENT: usize = 8;
pub(crate) const GDN_SPEC_COMMIT_MAX_K: usize = 256;
pub(crate) const GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH: usize = 16;
pub(crate) const GDN_SPEC_CHECKPOINT_MAX_K: usize = 256;
pub(crate) const GDN_SPEC_FUSED_MAX_TOKENS: usize = 8;
pub(crate) const GDN_CHANNEL_BLOCK_SIZE: usize = 256;
#[cfg(feature = "cuda")]
const GDN_TRANSITION_STAGE_POINTER_SEGMENTS: usize = 10;
#[cfg(feature = "cuda")]
const GDN_TRANSITION_APPLY_POINTER_SEGMENTS: usize = 10;
#[cfg(feature = "cuda")]
const GDN_DECODE_KERNEL_ENV: &str = "MISTRALRS_GDN_DECODE_KERNEL";
#[cfg(feature = "cuda")]
const GDN_PREFILL_KERNEL_ENV: &str = "MISTRALRS_GDN_PREFILL_KERNEL";
#[cfg(feature = "cuda")]
const GDN_STATE_DTYPE_F16: i32 = 0;
#[cfg(feature = "cuda")]
const GDN_STATE_DTYPE_BF16: i32 = 1;
#[cfg(feature = "cuda")]
const GDN_STATE_DTYPE_F32: i32 = 2;
#[cfg(feature = "cuda")]
const FLASHINFER_GDN_MIN_SEQ_LEN: usize = 32;

pub(crate) fn recurrent_state_dtype_supported(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16 | DType::F32)
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
enum GdnDecodeKernel {
    Baseline = 0,
    Cooperative = 1,
    Pipelined = 2,
    ValueMajor4 = 3,
    ValueMajor32 = 4,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum GdnPrefillKernel {
    FlashInferSm90,
    ValueMajor1,
    ValueMajor2,
    ValueMajor4,
    ValueMajor8,
    LegacyChunked,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy)]
struct GdnPrefillPolicy {
    compute_major: i32,
    multiprocessor_count: usize,
    state_blocks: usize,
    seq_len: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    bf16: bool,
    state_layout: RecurrentStateLayout,
}

#[cfg(any(feature = "cuda", test))]
fn prefill_kernel_supported(kernel: GdnPrefillKernel, policy: GdnPrefillPolicy) -> bool {
    kernel != GdnPrefillKernel::FlashInferSm90
        && policy.state_layout == RecurrentStateLayout::GdnValueMajor
        && policy.compute_major == GDN_DECODE_TUNED_COMPUTE_MAJOR
        && policy.multiprocessor_count > 0
        && policy.state_blocks > 0
        && policy.seq_len > 0
        && policy.bf16
        && policy.head_k_dim == GDN_DECODE_K_DIM
        && policy.head_v_dim == GDN_DECODE_V_DIM
}

#[cfg(any(feature = "cuda", test))]
fn automatic_prefill_kernel(policy: GdnPrefillPolicy) -> GdnPrefillKernel {
    if policy.state_blocks <= policy.multiprocessor_count {
        GdnPrefillKernel::ValueMajor2
    } else if policy.state_blocks <= policy.multiprocessor_count.saturating_mul(4) {
        GdnPrefillKernel::ValueMajor4
    } else {
        GdnPrefillKernel::ValueMajor8
    }
}

#[cfg(any(feature = "cuda", test))]
fn select_prefill_kernel(
    policy: GdnPrefillPolicy,
    requested: Option<GdnPrefillKernel>,
) -> std::result::Result<GdnPrefillKernel, GdnPrefillKernel> {
    let kernel = requested.unwrap_or_else(|| automatic_prefill_kernel(policy));
    prefill_kernel_supported(kernel, policy)
        .then_some(kernel)
        .ok_or(kernel)
}

#[cfg(any(feature = "cuda", test))]
fn parse_prefill_kernel(value: &str) -> std::result::Result<Option<GdnPrefillKernel>, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "flashinfer-sm90" => Ok(Some(GdnPrefillKernel::FlashInferSm90)),
        "vmajor1" => Ok(Some(GdnPrefillKernel::ValueMajor1)),
        "vmajor2" => Ok(Some(GdnPrefillKernel::ValueMajor2)),
        "vmajor4" => Ok(Some(GdnPrefillKernel::ValueMajor4)),
        "vmajor8" => Ok(Some(GdnPrefillKernel::ValueMajor8)),
        "legacy-chunked" => Ok(Some(GdnPrefillKernel::LegacyChunked)),
        other => Err(format!(
            "invalid GDN prefill kernel '{other}', expected auto, flashinfer-sm90, vmajor1, vmajor2, vmajor4, vmajor8, or legacy-chunked"
        )),
    }
}

#[cfg(feature = "cuda")]
fn prefill_kernel_override() -> Result<Option<GdnPrefillKernel>> {
    use std::sync::OnceLock;

    static OVERRIDE: OnceLock<std::result::Result<Option<GdnPrefillKernel>, String>> =
        OnceLock::new();
    match OVERRIDE.get_or_init(|| {
        std::env::var(GDN_PREFILL_KERNEL_ENV).map_or(Ok(None), |value| parse_prefill_kernel(&value))
    }) {
        Ok(kernel) => Ok(*kernel),
        Err(error) => Err(candle_core::Error::msg(error.clone())),
    }
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy)]
struct GdnDecodePolicy {
    compute_major: i32,
    multiprocessor_count: usize,
    state_blocks: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    vector_aligned: bool,
    bf16: bool,
    state_layout: RecurrentStateLayout,
}

#[cfg(any(feature = "cuda", test))]
fn decode_kernel_supported(kernel: GdnDecodeKernel, policy: GdnDecodePolicy) -> bool {
    match kernel {
        GdnDecodeKernel::Baseline => policy.state_layout == RecurrentStateLayout::GdnKeyMajor,
        GdnDecodeKernel::Cooperative => {
            policy.state_layout == RecurrentStateLayout::GdnKeyMajor
                && policy.compute_major >= GDN_DECODE_MIN_COMPUTE_MAJOR
                && policy.vector_aligned
                && policy.head_k_dim == GDN_DECODE_K_DIM
                && policy
                    .head_v_dim
                    .is_multiple_of(GDN_DECODE_COOPERATIVE_V_TILE)
        }
        GdnDecodeKernel::Pipelined => {
            policy.state_layout == RecurrentStateLayout::GdnKeyMajor
                && policy.compute_major >= GDN_DECODE_MIN_COMPUTE_MAJOR
                && policy.vector_aligned
                && policy.head_k_dim == GDN_DECODE_K_DIM
                && policy.head_v_dim >= GDN_DECODE_PIPELINED_V_TILE
                && policy
                    .head_v_dim
                    .is_multiple_of(GDN_DECODE_PIPELINED_V_TILE)
        }
        GdnDecodeKernel::ValueMajor4 | GdnDecodeKernel::ValueMajor32 => {
            policy.state_layout == RecurrentStateLayout::GdnValueMajor
                && policy.compute_major == GDN_DECODE_TUNED_COMPUTE_MAJOR
                && policy.bf16
                && policy.vector_aligned
                && policy.head_k_dim == GDN_DECODE_K_DIM
                && policy.head_v_dim == GDN_DECODE_V_DIM
        }
    }
}

#[cfg(any(feature = "cuda", test))]
fn automatic_decode_kernel(policy: GdnDecodePolicy) -> GdnDecodeKernel {
    if policy.state_layout == RecurrentStateLayout::GdnValueMajor {
        let large_tile_blocks = policy
            .state_blocks
            .saturating_mul(policy.head_v_dim / GDN_DECODE_V_MAJOR_LARGE_TILE);
        let large_tile_target = policy
            .multiprocessor_count
            .saturating_mul(GDN_DECODE_V_MAJOR_LARGE_CTA_WAVES);
        if large_tile_blocks >= large_tile_target
            && decode_kernel_supported(GdnDecodeKernel::ValueMajor32, policy)
        {
            return GdnDecodeKernel::ValueMajor32;
        }
        return GdnDecodeKernel::ValueMajor4;
    }
    if policy.compute_major != GDN_DECODE_TUNED_COMPUTE_MAJOR || policy.multiprocessor_count == 0 {
        return GdnDecodeKernel::Baseline;
    }

    let cooperative_limit = policy
        .multiprocessor_count
        .saturating_mul(GDN_DECODE_COOPERATIVE_STATE_WAVES);
    let pipelined_occupancy_limit = policy
        .multiprocessor_count
        .saturating_mul(GDN_DECODE_PIPELINED_OCCUPANCY_WAVES);
    let pipelined_amortized_start = policy
        .multiprocessor_count
        .saturating_mul(GDN_DECODE_PIPELINED_AMORTIZED_WAVES);
    let pipelined_has_overlap = policy.head_v_dim >= GDN_DECODE_PIPELINED_V_TILE.saturating_mul(2);

    if decode_kernel_supported(GdnDecodeKernel::Cooperative, policy)
        && policy.state_blocks < cooperative_limit
    {
        GdnDecodeKernel::Cooperative
    } else if decode_kernel_supported(GdnDecodeKernel::Pipelined, policy)
        && pipelined_has_overlap
        && policy.state_blocks >= cooperative_limit
        && (policy.state_blocks < pipelined_occupancy_limit
            || policy.state_blocks >= pipelined_amortized_start)
    {
        // The 256-thread, 39 KiB CTA has an SM90 4-8 state-wave scheduling valley.
        GdnDecodeKernel::Pipelined
    } else {
        GdnDecodeKernel::Baseline
    }
}

#[cfg(any(feature = "cuda", test))]
fn select_decode_kernel(
    policy: GdnDecodePolicy,
    requested: Option<GdnDecodeKernel>,
) -> std::result::Result<GdnDecodeKernel, GdnDecodeKernel> {
    match requested {
        Some(kernel) if decode_kernel_supported(kernel, policy) => Ok(kernel),
        Some(kernel) => Err(kernel),
        None => {
            let kernel = automatic_decode_kernel(policy);
            decode_kernel_supported(kernel, policy)
                .then_some(kernel)
                .ok_or(kernel)
        }
    }
}

#[cfg(any(feature = "cuda", test))]
fn parse_decode_kernel(value: &str) -> std::result::Result<Option<GdnDecodeKernel>, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "baseline" => Ok(Some(GdnDecodeKernel::Baseline)),
        "cooperative" => Ok(Some(GdnDecodeKernel::Cooperative)),
        "pipelined" => Ok(Some(GdnDecodeKernel::Pipelined)),
        "vmajor4" => Ok(Some(GdnDecodeKernel::ValueMajor4)),
        "vmajor32" => Ok(Some(GdnDecodeKernel::ValueMajor32)),
        other => Err(format!(
            "invalid GDN decode kernel '{other}', expected auto, baseline, cooperative, pipelined, vmajor4, or vmajor32"
        )),
    }
}

#[cfg(feature = "cuda")]
fn decode_kernel_override() -> Result<Option<GdnDecodeKernel>> {
    use std::sync::OnceLock;

    static OVERRIDE: OnceLock<std::result::Result<Option<GdnDecodeKernel>, String>> =
        OnceLock::new();
    match OVERRIDE.get_or_init(|| {
        std::env::var(GDN_DECODE_KERNEL_ENV).map_or(Ok(None), |value| parse_decode_kernel(&value))
    }) {
        Ok(kernel) => Ok(*kernel),
        Err(error) => Err(candle_core::Error::msg(error.clone())),
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
struct GdnCudaDeviceProperties {
    compute_major: i32,
    multiprocessor_count: usize,
}

#[cfg(feature = "cuda")]
fn gdn_cuda_device_properties(dev: &candle_core::CudaDevice) -> Result<GdnCudaDeviceProperties> {
    use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<HashMap<i32, GdnCudaDeviceProperties>>> = OnceLock::new();
    let stream = dev.cuda_stream();
    let context = stream.context();
    let device = context.cu_device();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(properties) = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&device)
        .copied()
    {
        return Ok(properties);
    }
    let compute_major = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(candle_core::Error::wrap)?;
    let multiprocessor_count = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(candle_core::Error::wrap)?;
    let multiprocessor_count = usize::try_from(multiprocessor_count)
        .ok()
        .filter(|count| *count > 0)
        .ok_or_else(|| candle_core::Error::msg("CUDA device reported no multiprocessors"))?;
    let properties = GdnCudaDeviceProperties {
        compute_major,
        multiprocessor_count,
    };
    cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(device, properties);
    Ok(properties)
}

pub(crate) fn v_major_state_supported(
    device: &Device,
    input_dtype: DType,
    key_dim: usize,
    value_dim: usize,
) -> Result<bool> {
    if input_dtype != DType::BF16 || key_dim != GDN_DECODE_K_DIM || value_dim != GDN_DECODE_V_DIM {
        return Ok(false);
    }
    #[cfg(feature = "cuda")]
    {
        if !device.is_cuda() {
            return Ok(false);
        }
        let properties = gdn_cuda_device_properties(device.as_cuda_device()?)?;
        Ok(properties.compute_major == GDN_DECODE_TUNED_COMPUTE_MAJOR)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = device;
        Ok(false)
    }
}

#[cfg(feature = "cuda")]
fn cuda_recurrent_state_ptr(tensor: &Tensor, name: &str) -> Result<(*mut core::ffi::c_void, i32)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;

    let (storage, layout) = tensor.storage_and_layout();
    let offset = layout.start_offset();
    let (pointer, dtype) = match tensor.dtype() {
        DType::F16 => {
            let storage = match &*storage {
                candle::Storage::Cuda(storage) => storage.as_cuda_slice::<half::f16>()?,
                _ => candle::bail!("{name} must be a CUDA tensor"),
            };
            (
                storage.slice(offset..).device_ptr(storage.stream()).0,
                GDN_STATE_DTYPE_F16,
            )
        }
        DType::BF16 => {
            let storage = match &*storage {
                candle::Storage::Cuda(storage) => storage.as_cuda_slice::<half::bf16>()?,
                _ => candle::bail!("{name} must be a CUDA tensor"),
            };
            (
                storage.slice(offset..).device_ptr(storage.stream()).0,
                GDN_STATE_DTYPE_BF16,
            )
        }
        DType::F32 => {
            let storage = match &*storage {
                candle::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
                _ => candle::bail!("{name} must be a CUDA tensor"),
            };
            (
                storage.slice(offset..).device_ptr(storage.stream()).0,
                GDN_STATE_DTYPE_F32,
            )
        }
        dtype => candle::bail!("{name} has unsupported recurrent state dtype {dtype:?}"),
    };
    Ok((pointer as *mut core::ffi::c_void, dtype))
}

#[cfg(feature = "cuda")]
fn cuda_read_ptr_with_guard<'a, T: candle_core::cuda_backend::CudaDType + 'a>(
    storage: &'a candle_core::Storage,
    layout: &candle_core::Layout,
    stream: &'a candle_core::cuda_backend::cudarc::driver::CudaStream,
    name: &str,
) -> Result<(
    candle_core::cuda_backend::cudarc::driver::sys::CUdeviceptr,
    candle_core::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
)> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;

    let slice = match storage {
        candle_core::Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?,
        _ => candle_core::bail!("{name} must be CUDA"),
    };
    let (base, guard) = slice.device_ptr(stream);
    let pointer = unsafe { (base as *const T).add(layout.start_offset()) as u64 };
    Ok((pointer, guard))
}

#[cfg(feature = "cuda")]
fn cuda_inplace_ptr_with_guard<'a, T: candle_core::cuda_backend::CudaDType + 'a>(
    storage: &'a candle_core::Storage,
    layout: &candle_core::Layout,
    stream: &'a candle_core::cuda_backend::cudarc::driver::CudaStream,
    name: &str,
) -> Result<(
    candle_core::cuda_backend::cudarc::driver::sys::CUdeviceptr,
    candle_core::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
)> {
    let slice = match storage {
        candle_core::Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?,
        _ => candle_core::bail!("{name} must be CUDA"),
    };
    if slice.stream().cu_stream() != stream.cu_stream() {
        candle_core::bail!("{name} must be owned by the GDN launch stream");
    }
    cuda_read_ptr_with_guard::<T>(storage, layout, stream, name)
}

#[cfg(feature = "cuda")]
fn cuda_recurrent_state_inplace_ptr_with_guard<'a>(
    dtype: DType,
    storage: &'a candle_core::Storage,
    layout: &candle_core::Layout,
    stream: &'a candle_core::cuda_backend::cudarc::driver::CudaStream,
    name: &str,
) -> Result<(
    candle_core::cuda_backend::cudarc::driver::sys::CUdeviceptr,
    i32,
    candle_core::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
)> {
    let (pointer, dtype_code, guard) = match dtype {
        DType::F16 => {
            let (pointer, guard) =
                cuda_inplace_ptr_with_guard::<half::f16>(storage, layout, stream, name)?;
            (pointer, GDN_STATE_DTYPE_F16, guard)
        }
        DType::BF16 => {
            let (pointer, guard) =
                cuda_inplace_ptr_with_guard::<half::bf16>(storage, layout, stream, name)?;
            (pointer, GDN_STATE_DTYPE_BF16, guard)
        }
        DType::F32 => {
            let (pointer, guard) =
                cuda_inplace_ptr_with_guard::<f32>(storage, layout, stream, name)?;
            (pointer, GDN_STATE_DTYPE_F32, guard)
        }
        other => candle_core::bail!("{name} has unsupported dtype {other:?}"),
    };
    Ok((pointer, dtype_code, guard))
}

/// Which rows of the recurrent state a GDN kernel reads and writes. `Gathered` means the state is a
/// `[B*H, ...]` copy addressed by batch row; `Pooled` addresses the whole pool `[cap, H, ...]`
/// through a `[B]` u32 slot table, so the kernels update it in place without gather/scatter copies.
/// A pooled slot of `u32::MAX` is padding: kernels emit zeros and leave the state pool untouched.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Copy)]
pub enum GdnStateSlots<'a> {
    Gathered,
    Pooled(&'a Tensor),
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl<'a> GdnStateSlots<'a> {
    pub fn from_option(slots: Option<&'a Tensor>) -> Self {
        match slots {
            Some(slots) => Self::Pooled(slots),
            None => Self::Gathered,
        }
    }
}

#[cfg(feature = "cuda")]
fn with_slot_indices<T>(
    slots: GdnStateSlots<'_>,
    f: impl FnOnce(*const i32, usize) -> Result<T>,
) -> Result<T> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    match slots {
        GdnStateSlots::Gathered => f(std::ptr::null(), 0),
        GdnStateSlots::Pooled(slots) => {
            let batch = slots.dim(0)?;
            let (s, l) = slots.storage_and_layout();
            let s = match &*s {
                candle_core::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle_core::bail!("slot indices must be a cuda tensor"),
            };
            let ptr = s.slice(l.start_offset()..).device_ptr(s.stream()).0 as *const i32;
            f(ptr, batch)
        }
    }
}

/// Contiguous f32 recurrence inputs: q, k `[BH, S, K]`, v `[BH, S, V]`, g, beta `[BH, S]`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Clone, Copy)]
pub struct RecurrenceInputs<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub g: &'a Tensor,
    pub beta: &'a Tensor,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
enum RecurrenceKernel {
    Scalar,
    Warp,
    ValueMajorWarp,
    ValueMajorWarp2,
    ValueMajorWarp4,
    ValueMajorWarp8,
    Chunked,
    #[allow(dead_code)]
    ValueMajorChunked,
}

/// `state` is `[BH, K, V]` or `[BH, V, K]` (gathered), or the matching pooled layout, mutated in place.
/// Returns output `[BH, S, V]`.
#[cfg(feature = "cuda")]
fn launch_recurrence(
    kernel: RecurrenceKernel,
    inputs: RecurrenceInputs<'_>,
    state: &mut Tensor,
    slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;

    let RecurrenceInputs { q, k, v, g, beta } = inputs;
    let (bh, seq_len, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;
    if matches!(
        kernel,
        RecurrenceKernel::ValueMajorWarp
            | RecurrenceKernel::ValueMajorWarp2
            | RecurrenceKernel::ValueMajorWarp4
            | RecurrenceKernel::ValueMajorWarp8
            | RecurrenceKernel::ValueMajorChunked
    ) && (k_dim != GDN_DECODE_K_DIM || v_dim != GDN_DECODE_V_DIM)
    {
        candle::bail!("value-major GDN prefill requires K=V=128, got K={k_dim}, V={v_dim}");
    }
    let dev = q.device().as_cuda_device()?;

    macro_rules! f32_ptr {
        ($t:expr, $name:literal) => {{
            let (s, l) = $t.storage_and_layout();
            let offset = l.start_offset();
            let s = match &*s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!(concat!($name, " must be a cuda tensor")),
            };
            let ptr = s.slice(offset..).device_ptr(s.stream()).0 as *mut f32;
            ptr
        }};
    }
    let q_ptr = f32_ptr!(q, "q");
    let k_ptr = f32_ptr!(k, "k");
    let v_ptr = f32_ptr!(v, "v");
    let g_ptr = f32_ptr!(g, "g");
    let beta_ptr = f32_ptr!(beta, "beta");
    let (state_ptr, state_dtype) = cuda_recurrent_state_ptr(state, "state")?;

    let output_buf = unsafe { dev.alloc::<f32>(bh * seq_len * v_dim) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    with_slot_indices(slots, |slot_ptr, batch| {
        let num_heads = bh.checked_div(batch).unwrap_or(1);
        let values_per_warp = match kernel {
            RecurrenceKernel::ValueMajorWarp2 => Some(2),
            RecurrenceKernel::ValueMajorWarp4 => Some(4),
            RecurrenceKernel::ValueMajorWarp8 => Some(8),
            _ => None,
        };
        if let Some(values_per_warp) = values_per_warp {
            let status = unsafe {
                crate::cuda::ffi::vmajor_grouped_warp_gated_delta_rule_recurrence(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    g_ptr,
                    beta_ptr,
                    state_ptr,
                    output_buf.device_ptr(output_buf.stream()).0 as *mut f32,
                    bh as i32,
                    seq_len as i32,
                    k_dim as i32,
                    v_dim as i32,
                    slot_ptr,
                    num_heads as i32,
                    values_per_warp,
                    state_dtype,
                    stream,
                )
            };
            if status != 0 {
                candle::bail!(
                    "vmajor_grouped_warp_gated_delta_rule_recurrence failed with status {status}"
                );
            }
            return Ok(());
        }
        if matches!(kernel, RecurrenceKernel::ValueMajorChunked) {
            let status = unsafe {
                crate::cuda::ffi::vmajor_chunked_gated_delta_rule_recurrence(
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    g_ptr,
                    beta_ptr,
                    state_ptr,
                    output_buf.device_ptr(output_buf.stream()).0 as *mut f32,
                    bh as i32,
                    seq_len as i32,
                    k_dim as i32,
                    v_dim as i32,
                    slot_ptr,
                    num_heads as i32,
                    state_dtype,
                    stream,
                )
            };
            if status != 0 {
                candle::bail!(
                    "vmajor_chunked_gated_delta_rule_recurrence failed with status {status}"
                );
            }
            return Ok(());
        }
        let launcher = match kernel {
            RecurrenceKernel::Scalar => crate::cuda::ffi::gated_delta_rule_recurrence,
            RecurrenceKernel::Warp => crate::cuda::ffi::warp_gated_delta_rule_recurrence,
            RecurrenceKernel::ValueMajorWarp => {
                crate::cuda::ffi::vmajor_warp_gated_delta_rule_recurrence
            }
            RecurrenceKernel::ValueMajorWarp2
            | RecurrenceKernel::ValueMajorWarp4
            | RecurrenceKernel::ValueMajorWarp8 => unreachable!(),
            RecurrenceKernel::Chunked => crate::cuda::ffi::chunked_gated_delta_rule_recurrence,
            RecurrenceKernel::ValueMajorChunked => unreachable!(),
        };
        unsafe {
            launcher(
                q_ptr,
                k_ptr,
                v_ptr,
                g_ptr,
                beta_ptr,
                state_ptr,
                output_buf.device_ptr(output_buf.stream()).0 as *mut f32,
                bh as i32,
                seq_len as i32,
                k_dim as i32,
                v_dim as i32,
                slot_ptr,
                num_heads as i32,
                state_dtype,
                stream,
            );
        }
        Ok(())
    })?;

    let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
    Ok(Tensor::from((
        candle::Storage::Cuda(output_storage),
        (bh, seq_len, v_dim),
    )))
}

/// Sequential (one token at a time) gated delta rule recurrence; see `launch_recurrence`.
#[cfg(feature = "cuda")]
pub fn gated_delta_rule_recurrence_cuda(
    inputs: RecurrenceInputs<'_>,
    state: &mut Tensor,
    slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    launch_recurrence(RecurrenceKernel::Scalar, inputs, state, slots)
}

/// Prefill recurrence in 64-token chunks; see `launch_recurrence`.
#[cfg(feature = "cuda")]
pub fn chunked_gated_delta_rule_recurrence_cuda(
    inputs: RecurrenceInputs<'_>,
    state: &mut Tensor,
    slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    launch_recurrence(RecurrenceKernel::Chunked, inputs, state, slots)
}

/// Warp-per-value-column prefill recurrence; see `launch_recurrence`.
#[cfg(feature = "cuda")]
pub fn warp_gated_delta_rule_recurrence_cuda(
    inputs: RecurrenceInputs<'_>,
    state: &mut Tensor,
    slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    launch_recurrence(RecurrenceKernel::Warp, inputs, state, slots)
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub fn vmajor_warp_gated_delta_rule_recurrence_cuda(
    inputs: RecurrenceInputs<'_>,
    state: &mut Tensor,
    slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    launch_recurrence(RecurrenceKernel::ValueMajorWarp, inputs, state, slots)
}

#[cfg(feature = "cuda")]
fn vmajor_prefill_gated_delta_rule_recurrence_cuda_impl(
    inputs: RecurrenceInputs<'_>,
    state: &mut Tensor,
    slots: GdnStateSlots<'_>,
    activation_dtype: DType,
    requested: Option<GdnPrefillKernel>,
) -> Result<Tensor> {
    let (state_blocks, seq_len, head_k_dim) = inputs.q.dims3()?;
    let head_v_dim = inputs.v.dim(2)?;
    let properties = gdn_cuda_device_properties(inputs.q.device().as_cuda_device()?)?;
    let policy = GdnPrefillPolicy {
        compute_major: properties.compute_major,
        multiprocessor_count: properties.multiprocessor_count,
        state_blocks,
        seq_len,
        head_k_dim,
        head_v_dim,
        bf16: activation_dtype == DType::BF16,
        state_layout: RecurrentStateLayout::GdnValueMajor,
    };
    let kernel = select_prefill_kernel(policy, requested).map_err(|kernel| {
        candle_core::Error::msg(format!(
            "GDN prefill kernel {kernel:?} does not support compute {}, BH={state_blocks}, S={seq_len}, K={head_k_dim}, V={head_v_dim}, dtype={activation_dtype:?}",
            properties.compute_major
        ))
    })?;
    let recurrence_kernel = match kernel {
        GdnPrefillKernel::ValueMajor1 => RecurrenceKernel::ValueMajorWarp,
        GdnPrefillKernel::ValueMajor2 => RecurrenceKernel::ValueMajorWarp2,
        GdnPrefillKernel::ValueMajor4 => RecurrenceKernel::ValueMajorWarp4,
        GdnPrefillKernel::ValueMajor8 => RecurrenceKernel::ValueMajorWarp8,
        GdnPrefillKernel::FlashInferSm90 => {
            candle_core::bail!("FlashInfer GDN prefill requires fused convolved inputs")
        }
        GdnPrefillKernel::LegacyChunked => RecurrenceKernel::ValueMajorChunked,
    };
    launch_recurrence(recurrence_kernel, inputs, state, slots)
}

#[cfg(feature = "cuda")]
pub fn vmajor_prefill_gated_delta_rule_recurrence_cuda(
    inputs: RecurrenceInputs<'_>,
    state: &mut Tensor,
    slots: GdnStateSlots<'_>,
    activation_dtype: DType,
) -> Result<Tensor> {
    let requested =
        prefill_kernel_override()?.filter(|kernel| *kernel != GdnPrefillKernel::FlashInferSm90);
    vmajor_prefill_gated_delta_rule_recurrence_cuda_impl(
        inputs,
        state,
        slots,
        activation_dtype,
        requested,
    )
}

/// Runs chunked prefill against value-major K=V=128 state.
#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub fn vmajor_chunked_gated_delta_rule_recurrence_cuda(
    inputs: RecurrenceInputs<'_>,
    state: &mut Tensor,
    slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    launch_recurrence(RecurrenceKernel::ValueMajorChunked, inputs, state, slots)
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn gated_delta_rule_recurrence_cuda(
    _inputs: RecurrenceInputs<'_>,
    _state: &mut Tensor,
    _slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    candle_core::bail!("gated_delta_rule_recurrence_cuda requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn chunked_gated_delta_rule_recurrence_cuda(
    _inputs: RecurrenceInputs<'_>,
    _state: &mut Tensor,
    _slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    candle_core::bail!("chunked_gated_delta_rule_recurrence_cuda requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn warp_gated_delta_rule_recurrence_cuda(
    _inputs: RecurrenceInputs<'_>,
    _state: &mut Tensor,
    _slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    candle_core::bail!("warp_gated_delta_rule_recurrence_cuda requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn vmajor_warp_gated_delta_rule_recurrence_cuda(
    _inputs: RecurrenceInputs<'_>,
    _state: &mut Tensor,
    _slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    candle_core::bail!("vmajor_warp_gated_delta_rule_recurrence_cuda requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn vmajor_prefill_gated_delta_rule_recurrence_cuda(
    _inputs: RecurrenceInputs<'_>,
    _state: &mut Tensor,
    _slots: GdnStateSlots<'_>,
    _activation_dtype: DType,
) -> Result<Tensor> {
    candle_core::bail!("vmajor_prefill_gated_delta_rule_recurrence_cuda requires the cuda feature")
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn vmajor_chunked_gated_delta_rule_recurrence_cuda(
    _inputs: RecurrenceInputs<'_>,
    _state: &mut Tensor,
    _slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    candle_core::bail!("vmajor_chunked_gated_delta_rule_recurrence_cuda requires the cuda feature")
}

/// CUDA-accelerated causal conv1d (both update and full paths).
///
/// x: [B, S, conv_dim] (S=1 for update)  weight: [conv_dim, kernel_size]
/// conv_state: [B, conv_dim, kernel_size], or the [cap, conv_dim, kernel_size] pool with `Pooled` slots.
/// Update mutates `conv_state` in place; full writes a fresh state (gathered) or the pool rows (pooled).
/// Returns (output [B, S, conv_dim], conv_state after the step).
#[cfg(feature = "cuda")]
pub fn causal_conv1d_cuda(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
    is_update: bool,
    slots: GdnStateSlots<'_>,
) -> Result<(Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        x: &Tensor,
        weight: &Tensor,
        conv_state: &Tensor,
        kernel_size: usize,
        is_update: bool,
        slots: GdnStateSlots<'_>,
        dtype_code: i32,
    ) -> Result<(Tensor, Tensor)> {
        let dev = x.device().as_cuda_device()?;
        let (batch_size, seq_len, conv_dim) = x.dims3()?;
        let pooled = matches!(slots, GdnStateSlots::Pooled(_));

        let (x_s, x_l) = x.storage_and_layout();
        let x_s = match &*x_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("x must be a cuda tensor"),
        };
        let x_offset = x_l.start_offset();
        let x_stride = x_l.stride();

        let (w_s, w_l) = weight.storage_and_layout();
        let w_s = match &*w_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };
        let w_offset = w_l.start_offset();

        let stream = dev.cuda_stream().cu_stream() as i64;

        if is_update {
            // Clone conv_state so the kernel can mutate it in place
            let conv_state_new = conv_state.clone();

            let output_buf = unsafe { dev.alloc::<T>(batch_size * conv_dim) }?;

            // Scope the borrow of conv_state_new so we can move it later
            {
                let (cs_s, cs_l) = conv_state_new.storage_and_layout();
                let cs_s = match &*cs_s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                    _ => candle::bail!("conv_state must be a cuda tensor"),
                };
                let cs_offset = cs_l.start_offset();

                with_slot_indices(slots, |slot_ptr, _| {
                    unsafe {
                        crate::cuda::ffi::causal_conv1d_update(
                            x_s.slice(x_offset..).device_ptr(x_s.stream()).0 as *const c_void,
                            w_s.slice(w_offset..).device_ptr(w_s.stream()).0 as *const c_void,
                            cs_s.slice(cs_offset..).device_ptr(cs_s.stream()).0 as *mut c_void,
                            output_buf.device_ptr(output_buf.stream()).0 as *mut c_void,
                            batch_size as i32,
                            conv_dim as i32,
                            kernel_size as i32,
                            x_stride[0] as i64,
                            x_stride[1] as i64,
                            x_stride[2] as i64,
                            slot_ptr,
                            dtype_code,
                            stream,
                        );
                    }
                    Ok(())
                })?;
            }

            let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
            let output = Tensor::from((
                candle::Storage::Cuda(output_storage),
                (batch_size, 1usize, conv_dim),
            ));

            Ok((output, conv_state_new))
        } else {
            let output_buf = unsafe { dev.alloc::<T>(batch_size * conv_dim * seq_len) }?;
            // Pooled: the save kernel rewrites the pool rows in place (it reads ahead of every write)
            let cs_buf = if pooled {
                None
            } else {
                Some(unsafe { dev.alloc::<T>(batch_size * conv_dim * kernel_size) }?)
            };
            let (cs_s, cs_l) = conv_state.storage_and_layout();
            let cs_s = match &*cs_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
                _ => candle::bail!("conv_state must be a cuda tensor"),
            };
            let cs_offset = cs_l.start_offset();
            let cs_in_ptr = cs_s.slice(cs_offset..).device_ptr(cs_s.stream()).0;
            let cs_out_ptr = match &cs_buf {
                Some(buf) => buf.device_ptr(buf.stream()).0,
                None => cs_in_ptr,
            };

            with_slot_indices(slots, |slot_ptr, _| {
                unsafe {
                    crate::cuda::ffi::causal_conv1d_full(
                        x_s.slice(x_offset..).device_ptr(x_s.stream()).0 as *const c_void,
                        w_s.slice(w_offset..).device_ptr(w_s.stream()).0 as *const c_void,
                        cs_in_ptr as *const c_void,
                        cs_out_ptr as *mut c_void,
                        output_buf.device_ptr(output_buf.stream()).0 as *mut c_void,
                        batch_size as i32,
                        conv_dim as i32,
                        seq_len as i32,
                        kernel_size as i32,
                        x_stride[0] as i64,
                        x_stride[1] as i64,
                        x_stride[2] as i64,
                        slot_ptr,
                        dtype_code,
                        stream,
                    );
                }
                Ok(())
            })?;

            let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
            let output = Tensor::from((
                candle::Storage::Cuda(output_storage),
                (batch_size, seq_len, conv_dim),
            ));

            let new_conv_state = match cs_buf {
                Some(cs_buf) => Tensor::from((
                    candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(
                        cs_buf,
                        dev.clone(),
                    )),
                    (batch_size, conv_dim, kernel_size),
                )),
                None => conv_state.clone(),
            };

            Ok((output, new_conv_state))
        }
    }

    let weight = weight.contiguous()?;
    if matches!(slots, GdnStateSlots::Pooled(_)) && !conv_state.is_contiguous() {
        candle_core::bail!("pooled conv state must be contiguous");
    }
    let conv_state = conv_state.contiguous()?;
    match x.dtype() {
        DType::F16 => {
            cuda_fwd::<half::f16>(x, &weight, &conv_state, kernel_size, is_update, slots, 0)
        }
        DType::BF16 => {
            cuda_fwd::<half::bf16>(x, &weight, &conv_state, kernel_size, is_update, slots, 1)
        }
        other => candle_core::bail!("causal_conv1d_cuda only supports f16/bf16, got {:?}", other),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn causal_conv1d_cuda(
    _x: &Tensor,
    _weight: &Tensor,
    _conv_state: &Tensor,
    _kernel_size: usize,
    _is_update: bool,
    _slots: GdnStateSlots<'_>,
) -> Result<(Tensor, Tensor)> {
    candle_core::bail!("causal_conv1d_cuda requires the cuda feature")
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn prepare_recurrence_inputs_cuda(
    mixed_qkv: &Tensor,
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    batch_size: usize,
    seq_len: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    tiled_v_heads: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        mixed_qkv: &Tensor,
        b: &Tensor,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        batch_size: usize,
        seq_len: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        tiled_v_heads: bool,
        dtype_code: i32,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let dev = mixed_qkv.device().as_cuda_device()?;

        let (mixed_s, mixed_l) = mixed_qkv.storage_and_layout();
        let mixed_s = match &*mixed_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("mixed_qkv must be a cuda tensor"),
        };
        let mixed_offset = mixed_l.start_offset();

        let (b_s, b_l) = b.storage_and_layout();
        let b_s = match &*b_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("b must be a cuda tensor"),
        };
        let b_offset = b_l.start_offset();

        let (a_s, a_l) = a.storage_and_layout();
        let a_s = match &*a_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("a must be a cuda tensor"),
        };
        let a_offset = a_l.start_offset();

        let (alog_s, alog_l) = a_log.storage_and_layout();
        let alog_s = match &*alog_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("a_log must be a cuda tensor"),
        };
        let alog_offset = alog_l.start_offset();

        let (dtb_s, dtb_l) = dt_bias.storage_and_layout();
        let dtb_s = match &*dtb_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("dt_bias must be a cuda tensor"),
        };
        let dtb_offset = dtb_l.start_offset();

        let bh = batch_size * num_v_heads;
        let q_buf = unsafe { dev.alloc::<f32>(bh * seq_len * head_k_dim) }?;
        let k_buf = unsafe { dev.alloc::<f32>(bh * seq_len * head_k_dim) }?;
        let v_buf = unsafe { dev.alloc::<f32>(bh * seq_len * head_v_dim) }?;
        let g_buf = unsafe { dev.alloc::<f32>(bh * seq_len) }?;
        let beta_buf = unsafe { dev.alloc::<f32>(bh * seq_len) }?;

        let stream = dev.cuda_stream().cu_stream() as i64;

        unsafe {
            crate::cuda::ffi::gdn_prepare_recurrence(
                mixed_s.slice(mixed_offset..).device_ptr(mixed_s.stream()).0 as *const c_void,
                b_s.slice(b_offset..).device_ptr(b_s.stream()).0 as *const c_void,
                a_s.slice(a_offset..).device_ptr(a_s.stream()).0 as *const c_void,
                alog_s.slice(alog_offset..).device_ptr(alog_s.stream()).0 as *const f32,
                dtb_s.slice(dtb_offset..).device_ptr(dtb_s.stream()).0 as *const f32,
                q_buf.device_ptr(q_buf.stream()).0 as *mut f32,
                k_buf.device_ptr(k_buf.stream()).0 as *mut f32,
                v_buf.device_ptr(v_buf.stream()).0 as *mut f32,
                g_buf.device_ptr(g_buf.stream()).0 as *mut f32,
                beta_buf.device_ptr(beta_buf.stream()).0 as *mut f32,
                batch_size as i32,
                seq_len as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                i32::from(tiled_v_heads),
                dtype_code,
                stream,
            );
        }

        let q = Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(q_buf, dev.clone())),
            (bh, seq_len, head_k_dim),
        ));
        let k = Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(k_buf, dev.clone())),
            (bh, seq_len, head_k_dim),
        ));
        let v = Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(v_buf, dev.clone())),
            (bh, seq_len, head_v_dim),
        ));
        let g = Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(g_buf, dev.clone())),
            (bh, seq_len),
        ));
        let beta = Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(beta_buf, dev.clone())),
            (bh, seq_len),
        ));

        Ok((q, k, v, g, beta))
    }

    match mixed_qkv.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(
            mixed_qkv,
            b,
            a,
            a_log,
            dt_bias,
            batch_size,
            seq_len,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads,
            0,
        ),
        DType::BF16 => cuda_fwd::<half::bf16>(
            mixed_qkv,
            b,
            a,
            a_log,
            dt_bias,
            batch_size,
            seq_len,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads,
            1,
        ),
        other => candle_core::bail!(
            "prepare_recurrence_inputs_cuda only supports f16/bf16, got {:?}",
            other
        ),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused, clippy::too_many_arguments)]
pub fn prepare_recurrence_inputs_cuda(
    _mixed_qkv: &Tensor,
    _b: &Tensor,
    _a: &Tensor,
    _a_log: &Tensor,
    _dt_bias: &Tensor,
    _batch_size: usize,
    _seq_len: usize,
    _num_k_heads: usize,
    _num_v_heads: usize,
    _head_k_dim: usize,
    _head_v_dim: usize,
    _tiled_v_heads: bool,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    candle_core::bail!("prepare_recurrence_inputs_cuda requires the cuda feature")
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct FusedPrefillRecurrence<'a> {
    pub mixed_qkv: &'a Tensor,
    pub b: &'a Tensor,
    pub a: &'a Tensor,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub state: &'a mut Tensor,
    pub batch_size: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub tiled_v_heads: bool,
    pub state_layout: RecurrentStateLayout,
    pub slots: GdnStateSlots<'a>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub enum FusedPrefillOutput {
    TokenMajor(Tensor),
}

#[cfg(all(feature = "cuda", has_flashinfer_gdn_sm90_kernel))]
#[repr(C)]
struct FlashInferGdnSm90Params {
    mixed_qkv: *const core::ffi::c_void,
    b: *const core::ffi::c_void,
    a: *const core::ffi::c_void,
    a_log: *const f32,
    dt_bias: *const f32,
    state: *mut f32,
    slots: *const u32,
    output: *mut core::ffi::c_void,
    workspace: *mut core::ffi::c_void,
    workspace_bytes: u64,
    batch_size: i32,
    seq_len: i32,
    num_k_heads: i32,
    num_v_heads: i32,
    sm_count: i32,
    stream: i64,
}

#[cfg(all(feature = "cuda", has_flashinfer_gdn_sm90_kernel))]
struct FlashInferGdnSm90Launch<'a> {
    mixed_qkv: &'a Tensor,
    b: &'a Tensor,
    a: &'a Tensor,
    a_log: &'a Tensor,
    dt_bias: &'a Tensor,
    slots: Option<&'a Tensor>,
    output: std::cell::RefCell<
        Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<half::bf16>>,
    >,
    workspace: std::cell::RefCell<Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<u8>>>,
    workspace_bytes: u64,
    batch_size: i32,
    seq_len: i32,
    num_k_heads: i32,
    num_v_heads: i32,
    sm_count: i32,
}

#[cfg(all(feature = "cuda", has_flashinfer_gdn_sm90_kernel))]
impl candle_core::InplaceOp1 for FlashInferGdnSm90Launch<'_> {
    fn name(&self) -> &'static str {
        "flashinfer-gdn-sm90-prefill"
    }

    fn cpu_fwd(
        &self,
        _storage: &mut candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> Result<()> {
        candle_core::bail!("FlashInfer GDN SM90 prefill requires CUDA storage")
    }

    fn cuda_fwd(
        &self,
        state_storage: &mut candle_core::CudaStorage,
        state_layout: &candle_core::Layout,
    ) -> Result<()> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};

        if !state_layout.is_contiguous() {
            candle_core::bail!("FlashInfer GDN SM90 prefill requires contiguous state")
        }
        let dev = state_storage.device();
        let stream = dev.cuda_stream();

        let (mixed_storage, mixed_layout) = self.mixed_qkv.storage_and_layout();
        let mixed_slice = match &*mixed_storage {
            candle_core::Storage::Cuda(storage) => storage.as_cuda_slice::<half::bf16>()?,
            _ => candle_core::bail!("mixed_qkv must be a CUDA tensor"),
        };
        let (mixed_base, mixed_guard) = mixed_slice.device_ptr(&stream);
        let mixed_ptr = unsafe {
            (mixed_base as *const half::bf16).add(mixed_layout.start_offset())
                as *const core::ffi::c_void
        };

        let (b_storage, b_layout) = self.b.storage_and_layout();
        let b_slice = match &*b_storage {
            candle_core::Storage::Cuda(storage) => storage.as_cuda_slice::<half::bf16>()?,
            _ => candle_core::bail!("b must be a CUDA tensor"),
        };
        let (b_base, b_guard) = b_slice.device_ptr(&stream);
        let b_ptr = unsafe {
            (b_base as *const half::bf16).add(b_layout.start_offset()) as *const core::ffi::c_void
        };

        let (a_storage, a_layout) = self.a.storage_and_layout();
        let a_slice = match &*a_storage {
            candle_core::Storage::Cuda(storage) => storage.as_cuda_slice::<half::bf16>()?,
            _ => candle_core::bail!("a must be a CUDA tensor"),
        };
        let (a_base, a_guard) = a_slice.device_ptr(&stream);
        let a_ptr = unsafe {
            (a_base as *const half::bf16).add(a_layout.start_offset()) as *const core::ffi::c_void
        };

        let (a_log_storage, a_log_layout) = self.a_log.storage_and_layout();
        let a_log_slice = match &*a_log_storage {
            candle_core::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
            _ => candle_core::bail!("a_log must be a CUDA tensor"),
        };
        let (a_log_base, a_log_guard) = a_log_slice.device_ptr(&stream);
        let a_log_ptr = unsafe { (a_log_base as *const f32).add(a_log_layout.start_offset()) };

        let (dt_bias_storage, dt_bias_layout) = self.dt_bias.storage_and_layout();
        let dt_bias_slice = match &*dt_bias_storage {
            candle_core::Storage::Cuda(storage) => storage.as_cuda_slice::<f32>()?,
            _ => candle_core::bail!("dt_bias must be a CUDA tensor"),
        };
        let (dt_bias_base, dt_bias_guard) = dt_bias_slice.device_ptr(&stream);
        let dt_bias_ptr =
            unsafe { (dt_bias_base as *const f32).add(dt_bias_layout.start_offset()) };

        let slots_storage_layout = self.slots.map(Tensor::storage_and_layout);
        let mut slots_guard = None;
        let slots_ptr = if let Some((storage, layout)) = &slots_storage_layout {
            let slice = match &**storage {
                candle_core::Storage::Cuda(storage) => storage.as_cuda_slice::<u32>()?,
                _ => candle_core::bail!("slot indices must be a CUDA tensor"),
            };
            let (base, guard) = slice.device_ptr(&stream);
            slots_guard = Some(guard);
            unsafe { (base as *const u32).add(layout.start_offset()) }
        } else {
            std::ptr::null()
        };

        let state_slice = state_storage.as_cuda_slice_mut::<f32>()?;
        let (state_base, state_guard) = state_slice.device_ptr_mut(&stream);
        let state_ptr = unsafe { (state_base as *mut f32).add(state_layout.start_offset()) };

        let mut output = self.output.borrow_mut();
        let output = output
            .as_mut()
            .ok_or_else(|| candle_core::Error::msg("FlashInfer GDN output was already consumed"))?;
        let (output_ptr, output_guard) = output.device_ptr_mut(&stream);

        let mut workspace = self.workspace.borrow_mut();
        let workspace = workspace.as_mut().ok_or_else(|| {
            candle_core::Error::msg("FlashInfer GDN workspace was already consumed")
        })?;
        let (workspace_ptr, workspace_guard) = workspace.device_ptr_mut(&stream);

        let params = FlashInferGdnSm90Params {
            mixed_qkv: mixed_ptr,
            b: b_ptr,
            a: a_ptr,
            a_log: a_log_ptr,
            dt_bias: dt_bias_ptr,
            state: state_ptr,
            slots: slots_ptr,
            output: output_ptr as *mut core::ffi::c_void,
            workspace: workspace_ptr as *mut core::ffi::c_void,
            workspace_bytes: self.workspace_bytes,
            batch_size: self.batch_size,
            seq_len: self.seq_len,
            num_k_heads: self.num_k_heads,
            num_v_heads: self.num_v_heads,
            sm_count: self.sm_count,
            stream: stream.cu_stream() as i64,
        };
        let status = unsafe {
            crate::cuda::ffi::mistralrs_flashinfer_gdn_sm90_launch(
                &params as *const FlashInferGdnSm90Params as *const core::ffi::c_void,
            )
        };

        drop(workspace_guard);
        drop(output_guard);
        drop(state_guard);
        drop(slots_guard);
        drop(dt_bias_guard);
        drop(a_log_guard);
        drop(a_guard);
        drop(b_guard);
        drop(mixed_guard);
        if status != 0 {
            candle_core::bail!("FlashInfer GDN SM90 prefill failed with status {status}")
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn flashinfer_sm90_prefill_supported(launch: &FusedPrefillRecurrence<'_>) -> Result<bool> {
    #[cfg(not(has_flashinfer_gdn_sm90_kernel))]
    {
        let _ = launch;
        Ok(false)
    }
    #[cfg(has_flashinfer_gdn_sm90_kernel)]
    {
        if launch.batch_size == 0
            || launch.num_k_heads == 0
            || launch.num_v_heads < launch.num_k_heads
            || !launch.num_v_heads.is_multiple_of(launch.num_k_heads)
            || launch.head_k_dim != GDN_DECODE_K_DIM
            || launch.head_v_dim != GDN_DECODE_V_DIM
            || launch.tiled_v_heads
            || launch.state_layout != RecurrentStateLayout::GdnValueMajor
            || launch.state.dtype() != DType::F32
            || !launch.state.is_contiguous()
            || [launch.mixed_qkv, launch.b, launch.a]
                .iter()
                .any(|tensor| tensor.dtype() != DType::BF16 || !tensor.is_contiguous())
            || [launch.a_log, launch.dt_bias]
                .iter()
                .any(|tensor| tensor.dtype() != DType::F32 || !tensor.is_contiguous())
        {
            return Ok(false);
        }
        let (_, seq_len, _) = launch.mixed_qkv.dims3()?;
        if seq_len < FLASHINFER_GDN_MIN_SEQ_LEN
            || i32::try_from(launch.batch_size).is_err()
            || i32::try_from(seq_len).is_err()
            || launch.batch_size.checked_mul(seq_len).is_none()
            || i32::try_from(launch.batch_size * seq_len).is_err()
        {
            return Ok(false);
        }
        let device = launch.mixed_qkv.device();
        if !device.is_cuda()
            || [
                launch.b,
                launch.a,
                launch.a_log,
                launch.dt_bias,
                launch.state,
            ]
            .iter()
            .any(|tensor| !device.same_device(tensor.device()))
        {
            return Ok(false);
        }
        if let GdnStateSlots::Pooled(slots) = launch.slots {
            if slots.dtype() != DType::U32
                || !slots.is_contiguous()
                || !device.same_device(slots.device())
            {
                return Ok(false);
            }
        }
        Ok(
            gdn_cuda_device_properties(device.as_cuda_device()?)?.compute_major
                == GDN_DECODE_TUNED_COMPUTE_MAJOR,
        )
    }
}

#[cfg(all(feature = "cuda", has_flashinfer_gdn_sm90_kernel))]
fn flashinfer_sm90_prefill(launch: FusedPrefillRecurrence<'_>) -> Result<FusedPrefillOutput> {
    use candle_core as candle;

    if !flashinfer_sm90_prefill_supported(&launch)? {
        candle::bail!(
            "FlashInfer GDN SM90 prefill does not support this device, layout, dtype, or shape"
        );
    }
    let FusedPrefillRecurrence {
        mixed_qkv,
        b,
        a,
        a_log,
        dt_bias,
        state,
        batch_size,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        tiled_v_heads: _,
        state_layout: _,
        slots,
    } = launch;
    let (input_batch, seq_len, conv_dim) = mixed_qkv.dims3()?;
    let expected_conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
    if input_batch != batch_size || conv_dim != expected_conv_dim {
        candle::bail!(
            "FlashInfer GDN input shape {:?} is incompatible with B={batch_size}, K heads={num_k_heads}, V heads={num_v_heads}, K={head_k_dim}, V={head_v_dim}",
            mixed_qkv.dims()
        );
    }
    if b.dims3()? != (batch_size, seq_len, num_v_heads)
        || a.dims3()? != (batch_size, seq_len, num_v_heads)
        || a_log.dims1()? != num_v_heads
        || dt_bias.dims1()? != num_v_heads
    {
        candle::bail!("FlashInfer GDN gate tensors have incompatible shapes")
    }
    match slots {
        GdnStateSlots::Gathered => {
            if state.dims3()? != (batch_size * num_v_heads, head_v_dim, head_k_dim) {
                candle::bail!("FlashInfer GDN gathered state has an incompatible shape")
            }
        }
        GdnStateSlots::Pooled(slot_indices) => {
            let state_dims = state.dims4()?;
            if state_dims.1 != num_v_heads
                || state_dims.2 != head_v_dim
                || state_dims.3 != head_k_dim
                || slot_indices.dims1()? != batch_size
            {
                candle::bail!("FlashInfer GDN pooled state or slot table has an incompatible shape")
            }
        }
    }

    let dev = mixed_qkv.device().as_cuda_device()?;
    let properties = gdn_cuda_device_properties(dev)?;
    let batch_size_i32 = i32::try_from(batch_size).map_err(candle::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle::Error::wrap)?;
    let num_k_heads_i32 = i32::try_from(num_k_heads).map_err(candle::Error::wrap)?;
    let num_v_heads_i32 = i32::try_from(num_v_heads).map_err(candle::Error::wrap)?;
    let sm_count_i32 =
        i32::try_from(properties.multiprocessor_count).map_err(candle::Error::wrap)?;
    let workspace_bytes = unsafe {
        crate::cuda::ffi::mistralrs_flashinfer_gdn_sm90_workspace_size(
            batch_size_i32,
            seq_len_i32,
            num_k_heads_i32,
            num_v_heads_i32,
            sm_count_i32,
        )
    };
    if workspace_bytes == 0 {
        candle::bail!("FlashInfer GDN SM90 returned an invalid workspace size")
    }
    let workspace_len = usize::try_from(workspace_bytes).map_err(candle::Error::wrap)?;
    let output_elements = batch_size
        .checked_mul(num_v_heads)
        .and_then(|elements| elements.checked_mul(seq_len))
        .and_then(|elements| elements.checked_mul(head_v_dim))
        .ok_or_else(|| candle::Error::msg("FlashInfer GDN output size overflow"))?;
    let output = unsafe { dev.alloc::<half::bf16>(output_elements) }?;
    let workspace = unsafe { dev.alloc::<u8>(workspace_len) }?;
    let op = FlashInferGdnSm90Launch {
        mixed_qkv,
        b,
        a,
        a_log,
        dt_bias,
        slots: match slots {
            GdnStateSlots::Gathered => None,
            GdnStateSlots::Pooled(slots) => Some(slots),
        },
        output: std::cell::RefCell::new(Some(output)),
        workspace: std::cell::RefCell::new(Some(workspace)),
        workspace_bytes,
        batch_size: batch_size_i32,
        seq_len: seq_len_i32,
        num_k_heads: num_k_heads_i32,
        num_v_heads: num_v_heads_i32,
        sm_count: sm_count_i32,
    };
    state.inplace_op1(&op)?;
    let output = op
        .output
        .into_inner()
        .ok_or_else(|| candle::Error::msg("FlashInfer GDN output is unavailable"))?;
    let output_storage = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(FusedPrefillOutput::TokenMajor(Tensor::from((
        candle::Storage::Cuda(output_storage),
        (batch_size, seq_len, num_v_heads, head_v_dim),
    ))))
}

#[cfg(feature = "cuda")]
pub fn try_fused_vmajor_prefill_recurrence_cuda(
    launch: FusedPrefillRecurrence<'_>,
) -> Result<Option<FusedPrefillOutput>> {
    let requested = prefill_kernel_override()?;
    let flashinfer_supported = flashinfer_sm90_prefill_supported(&launch)?;
    if requested == Some(GdnPrefillKernel::FlashInferSm90) {
        if !flashinfer_supported {
            candle_core::bail!(
                "FlashInfer GDN SM90 prefill does not support this device, layout, dtype, or shape"
            );
        }
        return flashinfer_sm90_prefill_dispatch(launch).map(Some);
    }
    if requested.is_none() && flashinfer_supported {
        return flashinfer_sm90_prefill_dispatch(launch).map(Some);
    }
    Ok(None)
}

#[cfg(feature = "cuda")]
fn flashinfer_sm90_prefill_dispatch(
    launch: FusedPrefillRecurrence<'_>,
) -> Result<FusedPrefillOutput> {
    #[cfg(has_flashinfer_gdn_sm90_kernel)]
    return flashinfer_sm90_prefill(launch);
    #[cfg(not(has_flashinfer_gdn_sm90_kernel))]
    {
        let _ = launch;
        candle_core::bail!("FlashInfer GDN SM90 prefill was not built for this target")
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn try_fused_vmajor_prefill_recurrence_cuda(
    _launch: FusedPrefillRecurrence<'_>,
) -> Result<Option<FusedPrefillOutput>> {
    candle_core::bail!("try_fused_vmajor_prefill_recurrence_cuda requires the cuda feature")
}

#[cfg(feature = "cuda")]
struct GdnDecodeLaunch<'a> {
    mixed_qkv: &'a Tensor,
    b: &'a Tensor,
    a: &'a Tensor,
    a_log: &'a Tensor,
    dt_bias: &'a Tensor,
    state: &'a mut Tensor,
    batch_size: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    tiled_v_heads: bool,
    state_layout: RecurrentStateLayout,
    slots: GdnStateSlots<'a>,
    requested_kernel: Option<GdnDecodeKernel>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct FusedDecodeRecurrence<'a> {
    pub mixed_qkv: &'a Tensor,
    pub b: &'a Tensor,
    pub a: &'a Tensor,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub state: &'a mut Tensor,
    pub batch_size: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub tiled_v_heads: bool,
    pub state_layout: RecurrentStateLayout,
    pub slots: GdnStateSlots<'a>,
}

#[cfg(feature = "cuda")]
pub fn fused_decode_recurrence_cuda(launch: FusedDecodeRecurrence<'_>) -> Result<Tensor> {
    fused_decode_recurrence_cuda_impl(GdnDecodeLaunch {
        mixed_qkv: launch.mixed_qkv,
        b: launch.b,
        a: launch.a,
        a_log: launch.a_log,
        dt_bias: launch.dt_bias,
        state: launch.state,
        batch_size: launch.batch_size,
        num_k_heads: launch.num_k_heads,
        num_v_heads: launch.num_v_heads,
        head_k_dim: launch.head_k_dim,
        head_v_dim: launch.head_v_dim,
        tiled_v_heads: launch.tiled_v_heads,
        state_layout: launch.state_layout,
        slots: launch.slots,
        requested_kernel: decode_kernel_override()?,
    })
}

#[cfg(feature = "cuda")]
fn fused_decode_recurrence_cuda_impl(launch: GdnDecodeLaunch<'_>) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        launch: GdnDecodeLaunch<'_>,
        dtype_code: i32,
    ) -> Result<Tensor> {
        let GdnDecodeLaunch {
            mixed_qkv,
            b,
            a,
            a_log,
            dt_bias,
            state,
            batch_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads,
            state_layout,
            slots,
            requested_kernel,
        } = launch;
        if head_k_dim > GDN_DECODE_FALLBACK_MAX_K {
            candle::bail!(
                "GDN decode key dimension {head_k_dim} exceeds the CUDA fallback limit {GDN_DECODE_FALLBACK_MAX_K}"
            );
        }
        let dev = mixed_qkv.device().as_cuda_device()?;
        let device_properties = gdn_cuda_device_properties(dev)?;

        let (mixed_s, mixed_l) = mixed_qkv.storage_and_layout();
        let mixed_s = match &*mixed_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("mixed_qkv must be a cuda tensor"),
        };
        let mixed_offset = mixed_l.start_offset();
        let mixed_ptr =
            mixed_s.slice(mixed_offset..).device_ptr(mixed_s.stream()).0 as *const c_void;

        let (b_s, b_l) = b.storage_and_layout();
        let b_s = match &*b_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("b must be a cuda tensor"),
        };
        let b_offset = b_l.start_offset();
        let b_stride = b_l.stride();
        let b_batch_stride = b_stride[0];
        let b_head_stride = b_stride[b_stride.len() - 1];

        let (a_s, a_l) = a.storage_and_layout();
        let a_s = match &*a_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("a must be a cuda tensor"),
        };
        let a_offset = a_l.start_offset();
        let a_stride = a_l.stride();
        let a_batch_stride = a_stride[0];
        let a_head_stride = a_stride[a_stride.len() - 1];

        let (alog_s, alog_l) = a_log.storage_and_layout();
        let alog_s = match &*alog_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("a_log must be a cuda tensor"),
        };
        let alog_offset = alog_l.start_offset();

        let (dtb_s, dtb_l) = dt_bias.storage_and_layout();
        let dtb_s = match &*dtb_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("dt_bias must be a cuda tensor"),
        };
        let dtb_offset = dtb_l.start_offset();

        let (state_ptr, state_dtype) = cuda_recurrent_state_ptr(state, "state")?;
        let policy = GdnDecodePolicy {
            compute_major: device_properties.compute_major,
            multiprocessor_count: device_properties.multiprocessor_count,
            state_blocks: batch_size.saturating_mul(num_v_heads),
            head_k_dim,
            head_v_dim,
            vector_aligned: (state_ptr as usize).is_multiple_of(GDN_DECODE_VECTOR_ALIGNMENT)
                && (mixed_ptr as usize).is_multiple_of(GDN_DECODE_INPUT_ALIGNMENT),
            bf16: dtype_code == 1,
            state_layout,
        };
        let decode_kernel = select_decode_kernel(policy, requested_kernel).map_err(|kernel| {
            candle_core::Error::msg(format!(
                "requested {kernel:?} GDN kernel is unsupported on compute {}, K={}, V={}, layout={:?}, bf16={}, aligned={}",
                policy.compute_major,
                policy.head_k_dim,
                policy.head_v_dim,
                policy.state_layout,
                policy.bf16,
                policy.vector_aligned
            ))
        })?;

        let bh = batch_size * num_v_heads;
        let output_buf = unsafe { dev.alloc::<T>(bh * head_v_dim) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        with_slot_indices(slots, |slot_ptr, _| {
            unsafe {
                crate::cuda::ffi::gdn_decode_recurrence(
                    mixed_ptr,
                    b_s.slice(b_offset..).device_ptr(b_s.stream()).0 as *const c_void,
                    a_s.slice(a_offset..).device_ptr(a_s.stream()).0 as *const c_void,
                    alog_s.slice(alog_offset..).device_ptr(alog_s.stream()).0 as *const f32,
                    dtb_s.slice(dtb_offset..).device_ptr(dtb_s.stream()).0 as *const f32,
                    state_ptr,
                    output_buf.device_ptr(output_buf.stream()).0 as *mut c_void,
                    batch_size as i32,
                    num_k_heads as i32,
                    num_v_heads as i32,
                    head_k_dim as i32,
                    head_v_dim as i32,
                    i32::from(tiled_v_heads),
                    b_batch_stride as i64,
                    b_head_stride as i64,
                    a_batch_stride as i64,
                    a_head_stride as i64,
                    slot_ptr,
                    decode_kernel as i32,
                    dtype_code,
                    state_dtype,
                    stream,
                );
            }
            Ok(())
        })?;

        Ok(Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(
                output_buf,
                dev.clone(),
            )),
            (bh, 1, head_v_dim),
        )))
    }

    match launch.mixed_qkv.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(launch, 0),
        DType::BF16 => cuda_fwd::<half::bf16>(launch, 1),
        other => candle_core::bail!(
            "fused_decode_recurrence_cuda only supports f16/bf16, got {:?}",
            other
        ),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn fused_decode_recurrence_cuda(_launch: FusedDecodeRecurrence<'_>) -> Result<Tensor> {
    candle_core::bail!("fused_decode_recurrence_cuda requires the cuda feature")
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct GdnSpeculativeStateCommit<'a> {
    pub mixed_qkv: &'a Tensor,
    pub convolved_qkv: &'a Tensor,
    pub b: &'a Tensor,
    pub a: &'a Tensor,
    pub initial_conv_state: &'a Tensor,
    pub initial_recurrent_state: &'a Tensor,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub conv_state_pool: &'a Tensor,
    pub recurrent_state_pool: &'a Tensor,
    pub keep_rows: &'a Tensor,
    pub slot_indices: &'a Tensor,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub tiled_v_heads: bool,
    pub state_layout: RecurrentStateLayout,
}

#[cfg(feature = "cuda")]
pub fn speculative_state_commit_cuda(commit: GdnSpeculativeStateCommit<'_>) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        commit: GdnSpeculativeStateCommit<'_>,
        dtype_code: i32,
    ) -> Result<()> {
        let GdnSpeculativeStateCommit {
            mixed_qkv,
            convolved_qkv,
            b,
            a,
            initial_conv_state,
            initial_recurrent_state,
            a_log,
            dt_bias,
            conv_state_pool,
            recurrent_state_pool,
            keep_rows,
            slot_indices,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads,
            state_layout,
        } = commit;
        let (batch_size, seq_len, conv_dim) = mixed_qkv.dims3()?;
        let expected_conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
        if conv_dim != expected_conv_dim {
            candle::bail!(
                "GDN speculative commit has conv dim {conv_dim}, expected {expected_conv_dim}"
            );
        }
        if head_k_dim == 0 || head_k_dim > GDN_SPEC_COMMIT_MAX_K || !head_k_dim.is_multiple_of(32) {
            candle::bail!(
                "GDN speculative commit requires a K dimension divisible by 32 and no larger than {GDN_SPEC_COMMIT_MAX_K}, got {head_k_dim}"
            );
        }
        if !head_v_dim.is_multiple_of(4) {
            candle::bail!(
                "GDN speculative commit requires a V dimension divisible by 4, got {head_v_dim}"
            );
        }
        let kernel_size = initial_conv_state.dim(candle_core::D::Minus1)?;
        if keep_rows.dims1()? != batch_size || slot_indices.dims1()? != batch_size {
            candle::bail!("GDN speculative commit index tensors must match batch size");
        }
        if b.dims3()? != (batch_size, seq_len, num_v_heads)
            || a.dims3()? != (batch_size, seq_len, num_v_heads)
        {
            candle::bail!("GDN speculative commit gate tensors have incompatible shapes");
        }
        if initial_conv_state.dims3()? != (batch_size, conv_dim, kernel_size) {
            candle::bail!("GDN speculative commit convolution state has an incompatible shape");
        }
        if convolved_qkv.dims3()? != (batch_size, seq_len, conv_dim) {
            candle::bail!("GDN speculative commit convolved input has an incompatible shape");
        }
        if [convolved_qkv, b, a, initial_conv_state, conv_state_pool]
            .iter()
            .any(|tensor| tensor.dtype() != mixed_qkv.dtype())
        {
            candle::bail!("GDN speculative commit activation tensors must share one dtype");
        }
        let expected_state_elements = batch_size * num_v_heads * head_k_dim * head_v_dim;
        if initial_recurrent_state.elem_count() != expected_state_elements {
            candle::bail!("GDN speculative commit recurrent state has an incompatible shape");
        }
        if !recurrent_state_dtype_supported(initial_recurrent_state.dtype())
            || initial_recurrent_state.dtype() != recurrent_state_pool.dtype()
        {
            candle::bail!("GDN speculative commit recurrent states must share a supported dtype");
        }
        let value_major = match state_layout {
            RecurrentStateLayout::GdnKeyMajor => false,
            RecurrentStateLayout::GdnValueMajor => true,
            RecurrentStateLayout::Opaque => {
                candle::bail!("GDN speculative commit does not support opaque state")
            }
        };

        let mixed_qkv = mixed_qkv.contiguous()?;
        let convolved_qkv = convolved_qkv.contiguous()?;
        let b = b.contiguous()?;
        let a = a.contiguous()?;
        let initial_conv_state = initial_conv_state.contiguous()?;
        let initial_recurrent_state = initial_recurrent_state.contiguous()?;
        let a_log = a_log.to_dtype(DType::F32)?.contiguous()?;
        let dt_bias = dt_bias.to_dtype(DType::F32)?.contiguous()?;
        if !conv_state_pool.is_contiguous() || !recurrent_state_pool.is_contiguous() {
            candle::bail!("GDN speculative commit state pools must be contiguous");
        }

        macro_rules! typed_ptr {
            ($tensor:expr, $ty:ty, $name:literal) => {{
                let (storage, layout) = $tensor.storage_and_layout();
                let storage = match &*storage {
                    candle::Storage::Cuda(storage) => storage.as_cuda_slice::<$ty>()?,
                    _ => candle::bail!(concat!($name, " must be a CUDA tensor")),
                };
                let pointer = storage
                    .slice(layout.start_offset()..)
                    .device_ptr(storage.stream())
                    .0;
                pointer
            }};
        }

        let mixed_ptr = typed_ptr!(mixed_qkv, T, "mixed_qkv") as *const c_void;
        let convolved_ptr = typed_ptr!(convolved_qkv, T, "convolved_qkv") as *const c_void;
        let b_ptr = typed_ptr!(b, T, "b") as *const c_void;
        let a_ptr = typed_ptr!(a, T, "a") as *const c_void;
        let initial_conv_ptr =
            typed_ptr!(initial_conv_state, T, "initial_conv_state") as *const c_void;
        let (initial_recurrent_ptr, state_dtype) =
            cuda_recurrent_state_ptr(&initial_recurrent_state, "initial_recurrent_state")?;
        let a_log_ptr = typed_ptr!(a_log, f32, "a_log") as *const f32;
        let dt_bias_ptr = typed_ptr!(dt_bias, f32, "dt_bias") as *const f32;
        let conv_pool_ptr = typed_ptr!(conv_state_pool, T, "conv_state_pool") as *mut c_void;
        let (recurrent_pool_ptr, pool_state_dtype) =
            cuda_recurrent_state_ptr(recurrent_state_pool, "recurrent_state_pool")?;
        if state_dtype != pool_state_dtype {
            candle::bail!("GDN speculative commit recurrent state dtype mismatch");
        }
        let keep_rows_ptr = typed_ptr!(keep_rows, u32, "keep_rows") as *const u32;
        let slot_indices_ptr = typed_ptr!(slot_indices, u32, "slot_indices") as *const u32;
        let dev = mixed_qkv.device().as_cuda_device()?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        unsafe {
            crate::cuda::ffi::gdn_speculative_state_commit(
                mixed_ptr,
                convolved_ptr,
                b_ptr,
                a_ptr,
                initial_conv_ptr,
                initial_recurrent_ptr as *const c_void,
                a_log_ptr,
                dt_bias_ptr,
                conv_pool_ptr,
                recurrent_pool_ptr,
                keep_rows_ptr,
                slot_indices_ptr,
                batch_size as i32,
                seq_len as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                kernel_size as i32,
                i32::from(tiled_v_heads),
                i32::from(value_major),
                dtype_code,
                state_dtype,
                stream,
            );
        }
        Ok(())
    }

    match commit.mixed_qkv.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(commit, 0),
        DType::BF16 => cuda_fwd::<half::bf16>(commit, 1),
        other => candle_core::bail!(
            "GDN speculative state commit only supports f16/bf16, got {:?}",
            other
        ),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn speculative_state_commit_cuda(_commit: GdnSpeculativeStateCommit<'_>) -> Result<()> {
    candle_core::bail!("speculative_state_commit_cuda requires the cuda feature")
}

#[allow(dead_code)]
pub struct GdnPendingSpeculativeConv<'a> {
    pub conv_input: &'a Tensor,
    pub keep_rows: &'a Tensor,
    pub epochs: &'a Tensor,
    pub applied_epochs: &'a Tensor,
}

#[allow(dead_code)]
pub struct GdnSpeculativeConvCheckpoints<'a> {
    pub x: &'a Tensor,
    pub weight: &'a Tensor,
    pub state_pool: &'a Tensor,
    pub active_slots: &'a Tensor,
    pub checkpoint_lanes: usize,
    pub write_checkpoints: bool,
    pub pending: Option<GdnPendingSpeculativeConv<'a>>,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub fn speculative_conv_checkpoints_cuda(
    context: GdnSpeculativeConvCheckpoints<'_>,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        context: GdnSpeculativeConvCheckpoints<'_>,
        dtype_code: i32,
    ) -> Result<Tensor> {
        let GdnSpeculativeConvCheckpoints {
            x,
            weight,
            state_pool,
            active_slots,
            checkpoint_lanes,
            write_checkpoints,
            pending,
        } = context;
        let (batch_size, seq_len, conv_dim) = x.dims3()?;
        let (weight_conv_dim, kernel_size) = weight.dims2()?;
        if batch_size == 0 || seq_len == 0 || conv_dim == 0 {
            candle::bail!("GDN speculative convolution requires non-empty dimensions");
        }
        if weight_conv_dim != conv_dim {
            candle::bail!(
                "GDN speculative convolution weight has {weight_conv_dim} channels, expected {conv_dim}"
            );
        }
        if kernel_size == 0 || kernel_size > GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH {
            candle::bail!(
                "GDN speculative convolution width must be in 1..={GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH}, got {kernel_size}"
            );
        }
        let (capacity, state_conv_dim, state_width) = state_pool.dims3()?;
        if state_conv_dim != conv_dim || state_width != kernel_size {
            candle::bail!(
                "GDN speculative convolution state shape {:?} is incompatible with [{capacity}, {conv_dim}, {kernel_size}]",
                state_pool.dims()
            );
        }
        if checkpoint_lanes == 0 || (write_checkpoints && seq_len > checkpoint_lanes) {
            candle::bail!(
                "GDN speculative convolution has query length {seq_len}, checkpoint lane count {checkpoint_lanes}"
            );
        }
        if !capacity.is_multiple_of(checkpoint_lanes) {
            candle::bail!(
                "GDN speculative convolution capacity {capacity} is not divisible by {checkpoint_lanes} checkpoint lanes"
            );
        }
        if active_slots.dims1()? != batch_size || active_slots.dtype() != DType::U32 {
            candle::bail!("GDN speculative convolution active slots must be u32 [batch]");
        }
        if weight.dtype() != x.dtype() || state_pool.dtype() != x.dtype() {
            candle::bail!("GDN speculative convolution tensors must share one dtype");
        }
        if !weight.is_contiguous() || !state_pool.is_contiguous() || !active_slots.is_contiguous() {
            candle::bail!(
                "GDN speculative convolution weight, state pool, and slots must be contiguous"
            );
        }
        for tensor in [weight, state_pool, active_slots] {
            if !tensor.device().same_device(x.device()) {
                candle::bail!("GDN speculative convolution tensors must share one device");
            }
        }
        let pending = pending
            .map(|pending| {
                let logical_capacity = capacity / checkpoint_lanes;
                let (pending_capacity, max_rows, pending_conv_dim) = pending.conv_input.dims3()?;
                let conv_blocks = conv_dim.div_ceil(GDN_CHANNEL_BLOCK_SIZE);
                if pending_capacity != logical_capacity
                    || max_rows == 0
                    || max_rows > GDN_SPEC_FUSED_MAX_TOKENS
                    || pending_conv_dim != conv_dim
                    || pending.keep_rows.dims() != [logical_capacity]
                    || pending.epochs.dims() != [logical_capacity]
                    || pending.applied_epochs.dims() != [logical_capacity, conv_blocks]
                    || pending.conv_input.dtype() != x.dtype()
                    || pending.keep_rows.dtype() != DType::U32
                    || pending.epochs.dtype() != DType::U32
                    || pending.applied_epochs.dtype() != DType::U32
                {
                    candle::bail!("GDN pending convolution transition shape is incompatible");
                }
                for tensor in [
                    pending.conv_input,
                    pending.keep_rows,
                    pending.epochs,
                    pending.applied_epochs,
                ] {
                    if !tensor.device().same_device(x.device()) || !tensor.is_contiguous() {
                        candle::bail!(
                            "GDN pending convolution transitions must be contiguous on one device"
                        );
                    }
                }
                Ok((pending, max_rows, logical_capacity, conv_blocks))
            })
            .transpose()?;
        if pending.is_some() && (write_checkpoints || checkpoint_lanes != 1) {
            candle::bail!(
                "GDN pending convolution transitions require single-lane transition logging"
            );
        }

        let dev = x.device().as_cuda_device()?;
        let (x_storage, x_layout) = x.storage_and_layout();
        let x_storage = match &*x_storage {
            candle::Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?,
            _ => candle::bail!("GDN speculative convolution input must be CUDA"),
        };
        let x_strides = x_layout.stride();
        let (weight_storage, weight_layout) = weight.storage_and_layout();
        let weight_storage = match &*weight_storage {
            candle::Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?,
            _ => candle::bail!("GDN speculative convolution weight must be CUDA"),
        };
        let (state_storage, state_layout) = state_pool.storage_and_layout();
        let state_storage = match &*state_storage {
            candle::Storage::Cuda(storage) => storage.as_cuda_slice::<T>()?,
            _ => candle::bail!("GDN speculative convolution state must be CUDA"),
        };
        let (slots_storage, slots_layout) = active_slots.storage_and_layout();
        let slots_storage = match &*slots_storage {
            candle::Storage::Cuda(storage) => storage.as_cuda_slice::<u32>()?,
            _ => candle::bail!("GDN speculative convolution slots must be CUDA"),
        };
        let output = unsafe { dev.alloc::<T>(batch_size * seq_len * conv_dim) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        macro_rules! cuda_ptr {
            ($tensor:expr, $ty:ty, $name:literal) => {{
                let (storage, layout) = $tensor.storage_and_layout();
                let storage = match &*storage {
                    candle::Storage::Cuda(storage) => storage.as_cuda_slice::<$ty>()?,
                    _ => candle::bail!(concat!($name, " must be CUDA")),
                };
                let pointer = storage
                    .slice(layout.start_offset()..)
                    .device_ptr(storage.stream())
                    .0;
                pointer
            }};
        }
        let (
            pending_conv,
            pending_keep,
            pending_epochs,
            applied_epochs,
            max_rows,
            pending_capacity,
            conv_blocks,
        ) = if let Some((pending, max_rows, pending_capacity, conv_blocks)) = pending {
            (
                cuda_ptr!(pending.conv_input, T, "pending_conv_input") as *const c_void,
                cuda_ptr!(pending.keep_rows, u32, "pending_keep_rows") as *const u32,
                cuda_ptr!(pending.epochs, u32, "pending_epochs") as *const u32,
                cuda_ptr!(pending.applied_epochs, u32, "conv_applied_epochs") as *mut u32,
                max_rows,
                pending_capacity,
                conv_blocks,
            )
        } else {
            (
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                0,
                0,
            )
        };

        unsafe {
            crate::cuda::ffi::gdn_speculative_conv_checkpoints(
                x_storage
                    .slice(x_layout.start_offset()..)
                    .device_ptr(x_storage.stream())
                    .0 as *const c_void,
                weight_storage
                    .slice(weight_layout.start_offset()..)
                    .device_ptr(weight_storage.stream())
                    .0 as *const c_void,
                state_storage
                    .slice(state_layout.start_offset()..)
                    .device_ptr(state_storage.stream())
                    .0 as *mut c_void,
                output.device_ptr(output.stream()).0 as *mut c_void,
                slots_storage
                    .slice(slots_layout.start_offset()..)
                    .device_ptr(slots_storage.stream())
                    .0 as *const u32,
                batch_size as i32,
                seq_len as i32,
                conv_dim as i32,
                kernel_size as i32,
                checkpoint_lanes as i32,
                i32::from(write_checkpoints),
                pending_conv,
                pending_keep,
                pending_epochs,
                applied_epochs,
                max_rows as i32,
                pending_capacity as i32,
                conv_blocks as i32,
                x_strides[0] as i64,
                x_strides[1] as i64,
                x_strides[2] as i64,
                dtype_code,
                stream,
            );
        }

        Ok(Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(output, dev.clone())),
            (batch_size, seq_len, conv_dim),
        )))
    }

    match context.x.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(context, 0),
        DType::BF16 => cuda_fwd::<half::bf16>(context, 1),
        other => {
            candle_core::bail!("GDN speculative convolution only supports f16/bf16, got {other:?}")
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn speculative_conv_checkpoints_cuda(
    _context: GdnSpeculativeConvCheckpoints<'_>,
) -> Result<Tensor> {
    candle_core::bail!("speculative_conv_checkpoints_cuda requires the cuda feature")
}

#[allow(dead_code)]
pub struct GdnPendingSpeculativeRecurrence<'a> {
    pub key: &'a Tensor,
    pub delta: &'a Tensor,
    pub decay: &'a Tensor,
    pub keep_rows: &'a Tensor,
    pub epochs: &'a Tensor,
    pub applied_epochs: &'a Tensor,
}

#[allow(dead_code)]
pub struct GdnSpeculativeRmsNormGate<'a> {
    pub gate: &'a Tensor,
    pub weight: &'a Tensor,
    pub eps: f64,
}

#[allow(dead_code)]
pub struct GdnSpeculativeRecurrenceCheckpoints<'a> {
    pub mixed_qkv: &'a Tensor,
    pub b: &'a Tensor,
    pub a: &'a Tensor,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub state_pool: &'a Tensor,
    pub active_slots: &'a Tensor,
    pub checkpoint_lanes: usize,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub tiled_v_heads: bool,
    pub state_layout: RecurrentStateLayout,
    pub post_op: Option<GdnSpeculativeRmsNormGate<'a>>,
    pub record_transitions: bool,
    pub pending: Option<GdnPendingSpeculativeRecurrence<'a>>,
}

#[derive(Clone)]
pub struct GdnSpeculativeTransitions {
    pub key: Tensor,
    pub delta: Tensor,
    pub decay: Tensor,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct GdnSpeculativeRecurrenceOutput {
    pub output: Tensor,
    pub transitions: Option<GdnSpeculativeTransitions>,
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub fn speculative_recurrence_checkpoints_cuda(
    context: GdnSpeculativeRecurrenceCheckpoints<'_>,
) -> Result<GdnSpeculativeRecurrenceOutput> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        context: GdnSpeculativeRecurrenceCheckpoints<'_>,
        dtype_code: i32,
    ) -> Result<GdnSpeculativeRecurrenceOutput> {
        let GdnSpeculativeRecurrenceCheckpoints {
            mixed_qkv,
            b,
            a,
            a_log,
            dt_bias,
            state_pool,
            active_slots,
            checkpoint_lanes,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads,
            state_layout,
            post_op,
            record_transitions,
            pending,
        } = context;
        let (batch_size, seq_len, conv_dim) = mixed_qkv.dims3()?;
        if batch_size == 0 || seq_len == 0 || num_k_heads == 0 || num_v_heads == 0 {
            candle::bail!("GDN speculative recurrence requires non-empty dimensions");
        }
        if head_k_dim == 0 || head_k_dim > GDN_SPEC_CHECKPOINT_MAX_K {
            candle::bail!(
                "GDN speculative recurrence key width must be in 1..={GDN_SPEC_CHECKPOINT_MAX_K}, got {head_k_dim}"
            );
        }
        if head_v_dim == 0 || !num_v_heads.is_multiple_of(num_k_heads) {
            candle::bail!("GDN speculative recurrence has incompatible head dimensions");
        }
        let expected_conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
        if conv_dim != expected_conv_dim {
            candle::bail!(
                "GDN speculative recurrence input width is {conv_dim}, expected {expected_conv_dim}"
            );
        }
        if b.dims3()? != (batch_size, seq_len, num_v_heads)
            || a.dims3()? != (batch_size, seq_len, num_v_heads)
        {
            candle::bail!("GDN speculative recurrence gate tensors have incompatible shapes");
        }
        let (physical_dim_2, physical_dim_3, value_major) = match state_layout {
            RecurrentStateLayout::GdnKeyMajor => (head_k_dim, head_v_dim, false),
            RecurrentStateLayout::GdnValueMajor => (head_v_dim, head_k_dim, true),
            RecurrentStateLayout::Opaque => {
                candle::bail!("GDN speculative recurrence does not support opaque state")
            }
        };
        let (capacity, state_heads, state_dim_2, state_dim_3) = state_pool.dims4()?;
        if state_heads != num_v_heads
            || state_dim_2 != physical_dim_2
            || state_dim_3 != physical_dim_3
        {
            candle::bail!(
                "GDN speculative recurrence state shape {:?} is incompatible with [{capacity}, {num_v_heads}, {physical_dim_2}, {physical_dim_3}]",
                state_pool.dims()
            );
        }
        if checkpoint_lanes == 0 || (!record_transitions && seq_len > checkpoint_lanes) {
            candle::bail!(
                "GDN speculative recurrence has query length {seq_len}, checkpoint lane count {checkpoint_lanes}"
            );
        }
        if !capacity.is_multiple_of(checkpoint_lanes) {
            candle::bail!(
                "GDN speculative recurrence capacity {capacity} is not divisible by {checkpoint_lanes} checkpoint lanes"
            );
        }
        if active_slots.dims1()? != batch_size || active_slots.dtype() != DType::U32 {
            candle::bail!("GDN speculative recurrence active slots must be u32 [batch]");
        }
        if a_log.dims1()? != num_v_heads || dt_bias.dims1()? != num_v_heads {
            candle::bail!("GDN speculative recurrence parameter tensors must match value heads");
        }
        if b.dtype() != mixed_qkv.dtype() || a.dtype() != mixed_qkv.dtype() {
            candle::bail!("GDN speculative recurrence activations must share one dtype");
        }
        if !recurrent_state_dtype_supported(state_pool.dtype()) {
            candle::bail!("GDN speculative recurrence state has unsupported dtype");
        }
        for tensor in [b, a, a_log, dt_bias, state_pool, active_slots] {
            if !tensor.device().same_device(mixed_qkv.device()) {
                candle::bail!("GDN speculative recurrence tensors must share one device");
            }
        }
        if !state_pool.is_contiguous() || !active_slots.is_contiguous() {
            candle::bail!("GDN speculative recurrence state and slots must be contiguous");
        }
        let pending = pending
            .map(|pending| {
                let logical_capacity = capacity / checkpoint_lanes;
                let (pending_capacity, max_rows, pending_k_heads, pending_k_dim) =
                    pending.key.dims4()?;
                if pending_capacity != logical_capacity
                    || max_rows == 0
                    || max_rows > GDN_SPEC_FUSED_MAX_TOKENS
                    || pending_k_heads != num_k_heads
                    || pending_k_dim != head_k_dim
                    || pending.delta.dims() != [logical_capacity, max_rows, num_v_heads, head_v_dim]
                    || pending.decay.dims() != [logical_capacity, max_rows, num_v_heads]
                    || pending.keep_rows.dims() != [logical_capacity]
                    || pending.epochs.dims() != [logical_capacity]
                    || pending.applied_epochs.dims() != [logical_capacity, num_v_heads]
                    || pending.key.dtype() != DType::F32
                    || pending.delta.dtype() != DType::F32
                    || pending.decay.dtype() != DType::F32
                    || pending.keep_rows.dtype() != DType::U32
                    || pending.epochs.dtype() != DType::U32
                    || pending.applied_epochs.dtype() != DType::U32
                {
                    candle::bail!("GDN pending recurrence transition shape is incompatible");
                }
                for tensor in [
                    pending.key,
                    pending.delta,
                    pending.decay,
                    pending.keep_rows,
                    pending.epochs,
                    pending.applied_epochs,
                ] {
                    if !tensor.device().same_device(mixed_qkv.device()) || !tensor.is_contiguous() {
                        candle::bail!(
                            "GDN pending recurrence transitions must be contiguous on one device"
                        );
                    }
                }
                Ok((pending, max_rows, logical_capacity))
            })
            .transpose()?;

        let fused_post_op = post_op.is_some();
        if pending.is_some() && (!record_transitions || !fused_post_op || checkpoint_lanes != 1) {
            candle::bail!(
                "GDN pending recurrence transitions require fused single-lane transition logging"
            );
        }
        if record_transitions
            && (state_layout != RecurrentStateLayout::GdnValueMajor
                || head_k_dim != GDN_DECODE_K_DIM
                || head_v_dim != GDN_DECODE_V_DIM
                || seq_len > GDN_SPEC_FUSED_MAX_TOKENS
                || !fused_post_op)
        {
            candle::bail!(
                "GDN transition logging requires fused value-major 128x128 speculative recurrence"
            );
        }
        let post_op = post_op
            .map(|post_op| {
                if state_layout != RecurrentStateLayout::GdnValueMajor
                    || head_k_dim != GDN_DECODE_K_DIM
                    || head_v_dim != GDN_DECODE_V_DIM
                    || seq_len > GDN_SPEC_FUSED_MAX_TOKENS
                {
                    candle::bail!(
                        "fused GDN speculative normalization requires value-major 128x128 state and at most 8 tokens"
                    );
                }
                if post_op.gate.dtype() != mixed_qkv.dtype()
                    || post_op.weight.dtype() != mixed_qkv.dtype()
                {
                    candle::bail!("fused GDN speculative normalization tensors must share the activation dtype");
                }
                if !post_op.gate.device().same_device(mixed_qkv.device())
                    || !post_op.weight.device().same_device(mixed_qkv.device())
                {
                    candle::bail!("fused GDN speculative normalization tensors must share one device");
                }
                if post_op.weight.dims() != [head_v_dim] {
                    candle::bail!("fused GDN speculative normalization weight must match the value head width");
                }
                if !post_op.eps.is_finite() || post_op.eps < 0.0 {
                    candle::bail!("fused GDN speculative normalization epsilon must be finite and non-negative");
                }
                let (_storage, layout) = post_op.gate.storage_and_layout();
                let strides = layout.stride();
                let gate_strides = match post_op.gate.dims() {
                    [gate_batch, gate_seq, gate_heads, gate_width]
                        if (*gate_batch, *gate_seq, *gate_heads, *gate_width)
                            == (batch_size, seq_len, num_v_heads, head_v_dim) =>
                    {
                        [strides[0], strides[1], strides[2], strides[3]]
                    }
                    [gate_batch, gate_seq, gate_width]
                        if (*gate_batch, *gate_seq, *gate_width)
                            == (batch_size, seq_len, num_v_heads * head_v_dim) =>
                    {
                        [strides[0], strides[1], head_v_dim * strides[2], strides[2]]
                    }
                    _ => candle::bail!(
                        "fused GDN speculative normalization gate has incompatible shape {:?}",
                        post_op.gate.dims()
                    ),
                };
                Ok((post_op, gate_strides))
            })
            .transpose()?;

        let mixed_qkv = mixed_qkv.contiguous()?;
        let b_strides = {
            let (_storage, layout) = b.storage_and_layout();
            let strides = layout.stride();
            [strides[0] as i64, strides[1] as i64, strides[2] as i64]
        };
        let a_strides = {
            let (_storage, layout) = a.storage_and_layout();
            let strides = layout.stride();
            [strides[0] as i64, strides[1] as i64, strides[2] as i64]
        };
        let a_log = a_log.to_dtype(DType::F32)?.contiguous()?;
        let dt_bias = dt_bias.to_dtype(DType::F32)?.contiguous()?;
        let dev = mixed_qkv.device().as_cuda_device()?;

        macro_rules! cuda_ptr {
            ($tensor:expr, $ty:ty, $name:literal) => {{
                let (storage, layout) = $tensor.storage_and_layout();
                let storage = match &*storage {
                    candle::Storage::Cuda(storage) => storage.as_cuda_slice::<$ty>()?,
                    _ => candle::bail!(concat!($name, " must be CUDA")),
                };
                let pointer = storage
                    .slice(layout.start_offset()..)
                    .device_ptr(storage.stream())
                    .0;
                pointer
            }};
        }

        let mixed_ptr = cuda_ptr!(mixed_qkv, T, "mixed_qkv") as *const c_void;
        let b_ptr = cuda_ptr!(b, T, "b") as *const c_void;
        let a_ptr = cuda_ptr!(a, T, "a") as *const c_void;
        let a_log_ptr = cuda_ptr!(a_log, f32, "a_log") as *const f32;
        let dt_bias_ptr = cuda_ptr!(dt_bias, f32, "dt_bias") as *const f32;
        let (state_ptr, state_dtype) = cuda_recurrent_state_ptr(state_pool, "state_pool")?;
        let slots_ptr = cuda_ptr!(active_slots, u32, "active_slots") as *const u32;
        let (gate_ptr, norm_weight, gate_strides, norm_eps) =
            if let Some((post_op, gate_strides)) = post_op {
                let gate_ptr = cuda_ptr!(post_op.gate, T, "gate") as *const c_void;
                let norm_weight = post_op.weight.contiguous()?;
                (
                    gate_ptr,
                    Some(norm_weight),
                    gate_strides,
                    post_op.eps as f32,
                )
            } else {
                (std::ptr::null(), None, [0; 4], 0.0)
            };
        let norm_weight_ptr = if let Some(norm_weight) = norm_weight.as_ref() {
            cuda_ptr!(norm_weight, T, "norm_weight") as *const c_void
        } else {
            std::ptr::null()
        };
        let output = unsafe { dev.alloc::<T>(batch_size * num_v_heads * seq_len * head_v_dim) }?;
        let mut transition_key = record_transitions
            .then(|| unsafe { dev.alloc::<f32>(batch_size * seq_len * num_k_heads * head_k_dim) })
            .transpose()?;
        let mut transition_delta = record_transitions
            .then(|| unsafe { dev.alloc::<f32>(batch_size * seq_len * num_v_heads * head_v_dim) })
            .transpose()?;
        let mut transition_decay = record_transitions
            .then(|| unsafe { dev.alloc::<f32>(batch_size * seq_len * num_v_heads) })
            .transpose()?;
        let transition_key_ptr = transition_key
            .as_mut()
            .map_or(std::ptr::null_mut(), |buffer| {
                buffer.device_ptr(buffer.stream()).0 as *mut f32
            });
        let transition_delta_ptr = transition_delta
            .as_mut()
            .map_or(std::ptr::null_mut(), |buffer| {
                buffer.device_ptr(buffer.stream()).0 as *mut f32
            });
        let transition_decay_ptr = transition_decay
            .as_mut()
            .map_or(std::ptr::null_mut(), |buffer| {
                buffer.device_ptr(buffer.stream()).0 as *mut f32
            });
        let (
            pending_key,
            pending_delta,
            pending_decay,
            pending_keep,
            pending_epochs,
            applied_epochs,
            max_pending_rows,
            pending_capacity,
        ) = if let Some((pending, max_rows, pending_capacity)) = pending {
            (
                cuda_ptr!(pending.key, f32, "pending_key") as *const f32,
                cuda_ptr!(pending.delta, f32, "pending_delta") as *const f32,
                cuda_ptr!(pending.decay, f32, "pending_decay") as *const f32,
                cuda_ptr!(pending.keep_rows, u32, "pending_keep_rows") as *const u32,
                cuda_ptr!(pending.epochs, u32, "pending_epochs") as *const u32,
                cuda_ptr!(pending.applied_epochs, u32, "recurrent_applied_epochs") as *mut u32,
                max_rows,
                pending_capacity,
            )
        } else {
            (
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                0,
            )
        };
        let stream = dev.cuda_stream().cu_stream() as i64;

        unsafe {
            crate::cuda::ffi::gdn_speculative_recurrence_checkpoints(
                mixed_ptr,
                b_ptr,
                a_ptr,
                a_log_ptr,
                dt_bias_ptr,
                state_ptr,
                output.device_ptr(output.stream()).0 as *mut c_void,
                slots_ptr,
                gate_ptr,
                norm_weight_ptr,
                transition_key_ptr,
                transition_delta_ptr,
                transition_decay_ptr,
                pending_key,
                pending_delta,
                pending_decay,
                pending_keep,
                pending_epochs,
                applied_epochs,
                max_pending_rows as i32,
                pending_capacity as i32,
                b_strides[0],
                b_strides[1],
                b_strides[2],
                a_strides[0],
                a_strides[1],
                a_strides[2],
                gate_strides[0] as i64,
                gate_strides[1] as i64,
                gate_strides[2] as i64,
                gate_strides[3] as i64,
                batch_size as i32,
                seq_len as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                checkpoint_lanes as i32,
                i32::from(tiled_v_heads),
                i32::from(value_major),
                norm_eps,
                dtype_code,
                state_dtype,
                stream,
            );
        }

        let output_storage =
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(output, dev.clone()));
        let output = if fused_post_op {
            Tensor::from((
                output_storage,
                (batch_size, seq_len, num_v_heads, head_v_dim),
            ))
        } else {
            Tensor::from((
                output_storage,
                (batch_size * num_v_heads, seq_len, head_v_dim),
            ))
        };
        let transitions = match (transition_key, transition_delta, transition_decay) {
            (Some(key), Some(delta), Some(decay)) => Some(GdnSpeculativeTransitions {
                key: Tensor::from((
                    candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(key, dev.clone())),
                    (batch_size, seq_len, num_k_heads, head_k_dim),
                )),
                delta: Tensor::from((
                    candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(delta, dev.clone())),
                    (batch_size, seq_len, num_v_heads, head_v_dim),
                )),
                decay: Tensor::from((
                    candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(decay, dev.clone())),
                    (batch_size, seq_len, num_v_heads),
                )),
            }),
            (None, None, None) => None,
            _ => unreachable!("transition buffers are allocated together"),
        };
        Ok(GdnSpeculativeRecurrenceOutput {
            output,
            transitions,
        })
    }

    match context.mixed_qkv.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(context, 0),
        DType::BF16 => cuda_fwd::<half::bf16>(context, 1),
        other => {
            candle_core::bail!("GDN speculative recurrence only supports f16/bf16, got {other:?}")
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn speculative_recurrence_checkpoints_cuda(
    _context: GdnSpeculativeRecurrenceCheckpoints<'_>,
) -> Result<GdnSpeculativeRecurrenceOutput> {
    candle_core::bail!("speculative_recurrence_checkpoints_cuda requires the cuda feature")
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct GdnSpeculativeTransitionLayer<'a> {
    pub conv_input: &'a Tensor,
    pub key: &'a Tensor,
    pub delta: &'a Tensor,
    pub decay: &'a Tensor,
    pub conv_state: &'a Tensor,
    pub recurrent_state: &'a Tensor,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct GdnSpeculativeTransitionCommit<'a> {
    pub layers: &'a [GdnSpeculativeTransitionLayer<'a>],
    pub keep_rows: &'a Tensor,
    pub active_slots: &'a Tensor,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_dim: usize,
    pub conv_width: usize,
    pub tiled_v_heads: bool,
    pub state_layout: RecurrentStateLayout,
}

#[cfg(feature = "cuda")]
pub fn speculative_transition_commit_batched_cuda(
    commit: GdnSpeculativeTransitionCommit<'_>,
) -> Result<()> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;

    fn cuda_commit<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        commit: GdnSpeculativeTransitionCommit<'_>,
        activation_dtype: i32,
    ) -> Result<()> {
        let GdnSpeculativeTransitionCommit {
            layers,
            keep_rows,
            active_slots,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            conv_width,
            tiled_v_heads,
            state_layout,
        } = commit;
        if layers.is_empty() {
            return Ok(());
        }
        let batch_size = keep_rows.dims1()?;
        if batch_size == 0
            || active_slots.dims1()? != batch_size
            || keep_rows.dtype() != DType::U32
            || active_slots.dtype() != DType::U32
        {
            candle::bail!("GDN transition commit indices must be non-empty u32 [batch]");
        }
        if conv_width == 0 || conv_width > GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH {
            candle::bail!("GDN transition commit has unsupported convolution width");
        }
        if num_k_heads == 0
            || num_v_heads == 0
            || !num_v_heads.is_multiple_of(num_k_heads)
            || head_k_dim != GDN_DECODE_K_DIM
            || head_v_dim != GDN_DECODE_V_DIM
        {
            candle::bail!("GDN transition commit has incompatible head dimensions");
        }
        let value_major = match state_layout {
            RecurrentStateLayout::GdnKeyMajor => false,
            RecurrentStateLayout::GdnValueMajor => true,
            RecurrentStateLayout::Opaque => {
                candle::bail!("GDN transition commit does not support opaque state")
            }
        };
        let (first_batch, seq_len, first_conv_dim) = layers[0].conv_input.dims3()?;
        if first_batch != batch_size
            || seq_len == 0
            || seq_len > GDN_SPEC_FUSED_MAX_TOKENS
            || first_conv_dim != conv_dim
        {
            candle::bail!("GDN transition commit input shape is incompatible with its indices");
        }
        let device = layers[0].conv_input.device();
        if !device.is_cuda()
            || !keep_rows.device().same_device(device)
            || !active_slots.device().same_device(device)
        {
            candle::bail!("GDN transition commit tensors must share one CUDA device");
        }
        let activation_tensor_dtype = layers[0].conv_input.dtype();
        let recurrent_dtype = layers[0].recurrent_state.dtype();
        if !recurrent_state_dtype_supported(recurrent_dtype) {
            candle::bail!("GDN transition commit state has unsupported dtype");
        }

        let cuda_stream = device.as_cuda_device()?.cuda_stream();
        let phase_timer = crate::pipeline::cuda_graph::CudaPhaseTimer::start(&cuda_stream)?;
        let mut pointer_segments = (0..6)
            .map(|_| Vec::with_capacity(layers.len()))
            .collect::<Vec<_>>();
        let mut recurrent_state_dtype = None;
        for layer in layers {
            let (layer_batch, layer_seq, layer_conv_dim) = layer.conv_input.dims3()?;
            let conv_capacity = layer.conv_state.dim(0)?;
            let recurrent_capacity = layer.recurrent_state.dim(0)?;
            if (layer_batch, layer_seq, layer_conv_dim) != (batch_size, seq_len, conv_dim)
                || layer.key.dims() != [batch_size, seq_len, num_k_heads, head_k_dim]
                || layer.delta.dims() != [batch_size, seq_len, num_v_heads, head_v_dim]
                || layer.decay.dims() != [batch_size, seq_len, num_v_heads]
                || layer.conv_state.dims() != [conv_capacity, conv_dim, conv_width]
                || recurrent_capacity != conv_capacity
            {
                candle::bail!("GDN transition commit layer shapes are incompatible");
            }
            let expected_state_shape = if value_major {
                [recurrent_capacity, num_v_heads, head_v_dim, head_k_dim]
            } else {
                [recurrent_capacity, num_v_heads, head_k_dim, head_v_dim]
            };
            if layer.recurrent_state.dims() != expected_state_shape {
                candle::bail!("GDN transition commit recurrent state shape is incompatible");
            }
            if layer.conv_input.dtype() != activation_tensor_dtype
                || layer.conv_state.dtype() != activation_tensor_dtype
                || layer.key.dtype() != DType::F32
                || layer.delta.dtype() != DType::F32
                || layer.decay.dtype() != DType::F32
                || layer.recurrent_state.dtype() != recurrent_dtype
            {
                candle::bail!("GDN transition commit layer dtypes are incompatible");
            }
            for tensor in [
                layer.conv_input,
                layer.key,
                layer.delta,
                layer.decay,
                layer.conv_state,
                layer.recurrent_state,
            ] {
                if !tensor.device().same_device(device) || !tensor.is_contiguous() {
                    candle::bail!(
                        "GDN transition commit layer tensors must be contiguous on one device"
                    );
                }
            }

            macro_rules! tensor_ptr {
                ($tensor:expr, $ty:ty, $name:literal) => {{
                    let (storage, layout) = $tensor.storage_and_layout();
                    let storage = match &*storage {
                        candle::Storage::Cuda(storage) => storage.as_cuda_slice::<$ty>()?,
                        _ => candle::bail!(concat!($name, " must be CUDA")),
                    };
                    let pointer = storage
                        .slice(layout.start_offset()..)
                        .device_ptr(storage.stream())
                        .0;
                    pointer
                }};
            }
            pointer_segments[0].push(tensor_ptr!(layer.conv_input, T, "conv_input"));
            pointer_segments[1].push(tensor_ptr!(layer.key, f32, "key"));
            pointer_segments[2].push(tensor_ptr!(layer.delta, f32, "delta"));
            pointer_segments[3].push(tensor_ptr!(layer.decay, f32, "decay"));
            pointer_segments[4].push(tensor_ptr!(layer.conv_state, T, "conv_state"));
            let (state_ptr, state_dtype) =
                cuda_recurrent_state_ptr(layer.recurrent_state, "recurrent_state")?;
            if recurrent_state_dtype
                .replace(state_dtype)
                .is_some_and(|dtype| dtype != state_dtype)
            {
                candle::bail!("GDN transition commit recurrent state dtypes diverged");
            }
            pointer_segments[5].push(state_ptr as u64);
        }
        let pointers = pointer_segments.into_iter().flatten().collect::<Vec<_>>();
        let pointers = pointers
            .into_iter()
            .map(|pointer| pointer as i64)
            .collect::<Vec<_>>();
        let pointer_table = Tensor::from_vec(pointers, (6 * layers.len(),), device)?;

        macro_rules! tensor_ptr {
            ($tensor:expr, $ty:ty, $name:literal) => {{
                let (storage, layout) = $tensor.storage_and_layout();
                let storage = match &*storage {
                    candle::Storage::Cuda(storage) => storage.as_cuda_slice::<$ty>()?,
                    _ => candle::bail!(concat!($name, " must be CUDA")),
                };
                let pointer = storage
                    .slice(layout.start_offset()..)
                    .device_ptr(storage.stream())
                    .0;
                pointer
            }};
        }
        let pointer_table_ptr = tensor_ptr!(pointer_table, i64, "pointer_table") as *const u64;
        let keep_rows_ptr = tensor_ptr!(keep_rows, u32, "keep_rows") as *const u32;
        let active_slots_ptr = tensor_ptr!(active_slots, u32, "active_slots") as *const u32;
        let stream = cuda_stream.cu_stream() as i64;
        unsafe {
            crate::cuda::ffi::gdn_speculative_transition_commit_batched(
                pointer_table_ptr,
                keep_rows_ptr,
                active_slots_ptr,
                layers.len() as i32,
                batch_size as i32,
                seq_len as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                conv_dim as i32,
                conv_width as i32,
                i32::from(tiled_v_heads),
                i32::from(value_major),
                activation_dtype,
                recurrent_state_dtype.expect("layers are non-empty"),
                stream,
            );
        }
        if let Some(timer) = phase_timer {
            timer.finish("gdn_transition_commit", batch_size, seq_len)?;
        }
        Ok(())
    }

    match commit.layers.first().map(|layer| layer.conv_input.dtype()) {
        Some(DType::F16) => cuda_commit::<half::f16>(commit, 0),
        Some(DType::BF16) => cuda_commit::<half::bf16>(commit, 1),
        Some(other) => candle_core::bail!(
            "GDN transition commit only supports f16/bf16 activations, got {other:?}"
        ),
        None => Ok(()),
    }
}

#[cfg(not(feature = "cuda"))]
pub fn speculative_transition_commit_batched_cuda(
    _commit: GdnSpeculativeTransitionCommit<'_>,
) -> Result<()> {
    candle_core::bail!("speculative_transition_commit_batched_cuda requires the cuda feature")
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct GdnSpeculativeTransitionStageLayer<'a> {
    pub conv_input: &'a Tensor,
    pub key: &'a Tensor,
    pub delta: &'a Tensor,
    pub decay: &'a Tensor,
    pub pending_conv_input: &'a Tensor,
    pub pending_key: &'a Tensor,
    pub pending_delta: &'a Tensor,
    pub pending_decay: &'a Tensor,
    pub pending_keep_rows: &'a Tensor,
    pub pending_epochs: &'a Tensor,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct GdnSpeculativeTransitionStage<'a> {
    pub layers: &'a [GdnSpeculativeTransitionStageLayer<'a>],
    pub keep_rows: &'a Tensor,
    pub destination_slots: &'a Tensor,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_dim: usize,
}

#[cfg(feature = "cuda")]
pub fn speculative_transition_stage_batched_cuda(
    stage: GdnSpeculativeTransitionStage<'_>,
) -> Result<()> {
    use candle_core as candle;

    fn cuda_stage<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        stage: GdnSpeculativeTransitionStage<'_>,
        activation_dtype: i32,
    ) -> Result<()> {
        let GdnSpeculativeTransitionStage {
            layers,
            keep_rows,
            destination_slots,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
        } = stage;
        if layers.is_empty() {
            return Ok(());
        }
        let batch_size = keep_rows.dims1()?;
        if batch_size == 0
            || destination_slots.dims1()? != batch_size
            || keep_rows.dtype() != DType::U32
            || destination_slots.dtype() != DType::U32
        {
            candle::bail!("GDN transition stage indices must be non-empty u32 [batch]");
        }
        if num_k_heads == 0
            || num_v_heads == 0
            || !num_v_heads.is_multiple_of(num_k_heads)
            || head_k_dim == 0
            || head_v_dim == 0
            || conv_dim == 0
        {
            candle::bail!("GDN transition stage has incompatible dimensions");
        }
        let (first_batch, seq_len, first_conv_dim) = layers[0].conv_input.dims3()?;
        if first_batch != batch_size
            || seq_len == 0
            || seq_len > GDN_SPEC_FUSED_MAX_TOKENS
            || first_conv_dim != conv_dim
        {
            candle::bail!("GDN transition stage input shape is incompatible with its indices");
        }
        let (destination_capacity, max_rows, destination_conv_dim) =
            layers[0].pending_conv_input.dims3()?;
        if destination_capacity == 0
            || max_rows < seq_len
            || max_rows > GDN_SPEC_FUSED_MAX_TOKENS
            || destination_conv_dim != conv_dim
        {
            candle::bail!("GDN transition stage destination shape is incompatible");
        }
        let device = layers[0].conv_input.device();
        if !device.is_cuda()
            || !keep_rows.device().same_device(device)
            || !destination_slots.device().same_device(device)
            || !keep_rows.is_contiguous()
            || !destination_slots.is_contiguous()
        {
            candle::bail!("GDN transition stage tensors must be contiguous on one CUDA device");
        }
        let activation_tensor_dtype = layers[0].conv_input.dtype();
        for layer in layers {
            if layer.conv_input.dims() != [batch_size, seq_len, conv_dim]
                || layer.key.dims() != [batch_size, seq_len, num_k_heads, head_k_dim]
                || layer.delta.dims() != [batch_size, seq_len, num_v_heads, head_v_dim]
                || layer.decay.dims() != [batch_size, seq_len, num_v_heads]
                || layer.pending_conv_input.dims() != [destination_capacity, max_rows, conv_dim]
                || layer.pending_key.dims()
                    != [destination_capacity, max_rows, num_k_heads, head_k_dim]
                || layer.pending_delta.dims()
                    != [destination_capacity, max_rows, num_v_heads, head_v_dim]
                || layer.pending_decay.dims() != [destination_capacity, max_rows, num_v_heads]
                || layer.pending_keep_rows.dims() != [destination_capacity]
                || layer.pending_epochs.dims() != [destination_capacity]
            {
                candle::bail!("GDN transition stage layer shapes are incompatible");
            }
            if layer.conv_input.dtype() != activation_tensor_dtype
                || layer.pending_conv_input.dtype() != activation_tensor_dtype
                || layer.key.dtype() != DType::F32
                || layer.delta.dtype() != DType::F32
                || layer.decay.dtype() != DType::F32
                || layer.pending_key.dtype() != DType::F32
                || layer.pending_delta.dtype() != DType::F32
                || layer.pending_decay.dtype() != DType::F32
                || layer.pending_keep_rows.dtype() != DType::U32
                || layer.pending_epochs.dtype() != DType::U32
            {
                candle::bail!("GDN transition stage layer dtypes are incompatible");
            }
            for tensor in [
                layer.conv_input,
                layer.key,
                layer.delta,
                layer.decay,
                layer.pending_conv_input,
                layer.pending_key,
                layer.pending_delta,
                layer.pending_decay,
                layer.pending_keep_rows,
                layer.pending_epochs,
            ] {
                if !tensor.device().same_device(device) || !tensor.is_contiguous() {
                    candle::bail!(
                        "GDN transition stage layer tensors must be contiguous on one device"
                    );
                }
            }
        }
        let cuda_stream = device.as_cuda_device()?.cuda_stream();
        let layer_storage_layouts = layers
            .iter()
            .map(|layer| {
                [
                    layer.conv_input.storage_and_layout(),
                    layer.key.storage_and_layout(),
                    layer.delta.storage_and_layout(),
                    layer.decay.storage_and_layout(),
                    layer.pending_conv_input.storage_and_layout(),
                    layer.pending_key.storage_and_layout(),
                    layer.pending_delta.storage_and_layout(),
                    layer.pending_decay.storage_and_layout(),
                    layer.pending_keep_rows.storage_and_layout(),
                    layer.pending_epochs.storage_and_layout(),
                ]
            })
            .collect::<Vec<_>>();
        let mut pointer_segments = (0..GDN_TRANSITION_STAGE_POINTER_SEGMENTS)
            .map(|_| Vec::with_capacity(layers.len()))
            .collect::<Vec<_>>();
        let mut pointer_guards =
            Vec::with_capacity(GDN_TRANSITION_STAGE_POINTER_SEGMENTS.saturating_mul(layers.len()));
        for storage_layouts in &layer_storage_layouts {
            macro_rules! read_ptr {
                ($segment:expr, $ty:ty, $name:literal) => {{
                    let (storage, layout) = &storage_layouts[$segment];
                    let (pointer, guard) =
                        cuda_read_ptr_with_guard::<$ty>(storage, layout, &cuda_stream, $name)?;
                    pointer_segments[$segment].push(pointer);
                    pointer_guards.push(guard);
                }};
            }
            macro_rules! inplace_ptr {
                ($segment:expr, $ty:ty, $name:literal) => {{
                    let (storage, layout) = &storage_layouts[$segment];
                    let (pointer, guard) =
                        cuda_inplace_ptr_with_guard::<$ty>(storage, layout, &cuda_stream, $name)?;
                    pointer_segments[$segment].push(pointer);
                    pointer_guards.push(guard);
                }};
            }
            read_ptr!(0, T, "conv_input");
            read_ptr!(1, f32, "key");
            read_ptr!(2, f32, "delta");
            read_ptr!(3, f32, "decay");
            inplace_ptr!(4, T, "pending_conv_input");
            inplace_ptr!(5, f32, "pending_key");
            inplace_ptr!(6, f32, "pending_delta");
            inplace_ptr!(7, f32, "pending_decay");
            inplace_ptr!(8, u32, "pending_keep_rows");
            inplace_ptr!(9, u32, "pending_epochs");
        }
        let pointers = pointer_segments
            .into_iter()
            .flatten()
            .map(|pointer| pointer as i64)
            .collect::<Vec<_>>();
        let pointer_table = Tensor::from_vec(
            pointers,
            (GDN_TRANSITION_STAGE_POINTER_SEGMENTS * layers.len(),),
            device,
        )?;
        let (pointer_table_storage, pointer_table_layout) = pointer_table.storage_and_layout();
        let (pointer_table_ptr, pointer_table_guard) = cuda_read_ptr_with_guard::<i64>(
            &pointer_table_storage,
            pointer_table_layout,
            &cuda_stream,
            "pointer_table",
        )?;
        let (keep_rows_storage, keep_rows_layout) = keep_rows.storage_and_layout();
        let (keep_rows_ptr, keep_rows_guard) = cuda_read_ptr_with_guard::<u32>(
            &keep_rows_storage,
            keep_rows_layout,
            &cuda_stream,
            "keep_rows",
        )?;
        let (destination_slots_storage, destination_slots_layout) =
            destination_slots.storage_and_layout();
        let (destination_slots_ptr, destination_slots_guard) = cuda_read_ptr_with_guard::<u32>(
            &destination_slots_storage,
            destination_slots_layout,
            &cuda_stream,
            "destination_slots",
        )?;
        let phase_timer = crate::pipeline::cuda_graph::CudaPhaseTimer::start(&cuda_stream)?;
        unsafe {
            crate::cuda::ffi::gdn_speculative_transition_stage_batched(
                pointer_table_ptr as *const u64,
                keep_rows_ptr as *const u32,
                destination_slots_ptr as *const u32,
                layers.len() as i32,
                batch_size as i32,
                seq_len as i32,
                max_rows as i32,
                destination_capacity as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                conv_dim as i32,
                activation_dtype,
                cuda_stream.cu_stream() as i64,
            );
        }
        drop(destination_slots_guard);
        drop(keep_rows_guard);
        drop(pointer_table_guard);
        drop(pointer_guards);
        if let Some(timer) = phase_timer {
            timer.finish("gdn_transition_stage", batch_size, seq_len)?;
        }
        Ok(())
    }

    match stage.layers.first().map(|layer| layer.conv_input.dtype()) {
        Some(DType::F16) => cuda_stage::<half::f16>(stage, 0),
        Some(DType::BF16) => cuda_stage::<half::bf16>(stage, 1),
        Some(other) => candle_core::bail!(
            "GDN transition stage only supports f16/bf16 activations, got {other:?}"
        ),
        None => Ok(()),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub fn speculative_transition_stage_batched_cuda(
    _stage: GdnSpeculativeTransitionStage<'_>,
) -> Result<()> {
    candle_core::bail!("speculative_transition_stage_batched_cuda requires the cuda feature")
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct GdnPendingTransitionApplyLayer<'a> {
    pub pending_conv_input: &'a Tensor,
    pub pending_key: &'a Tensor,
    pub pending_delta: &'a Tensor,
    pub pending_decay: &'a Tensor,
    pub pending_keep_rows: &'a Tensor,
    pub pending_epochs: &'a Tensor,
    pub conv_applied_epochs: &'a Tensor,
    pub recurrent_applied_epochs: &'a Tensor,
    pub conv_state: &'a Tensor,
    pub recurrent_state: &'a Tensor,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct GdnPendingTransitionApply<'a> {
    pub layers: &'a [GdnPendingTransitionApplyLayer<'a>],
    pub active_slots: &'a Tensor,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_dim: usize,
    pub conv_width: usize,
    pub tiled_v_heads: bool,
    pub state_layout: RecurrentStateLayout,
}

#[cfg(feature = "cuda")]
pub fn pending_transition_apply_batched_cuda(apply: GdnPendingTransitionApply<'_>) -> Result<()> {
    use candle_core as candle;

    fn cuda_apply<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        apply: GdnPendingTransitionApply<'_>,
        activation_dtype: i32,
    ) -> Result<()> {
        let GdnPendingTransitionApply {
            layers,
            active_slots,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            conv_width,
            tiled_v_heads,
            state_layout,
        } = apply;
        if layers.is_empty() {
            return Ok(());
        }
        let batch_size = active_slots.dims1()?;
        if batch_size == 0 || active_slots.dtype() != DType::U32 || !active_slots.is_contiguous() {
            candle::bail!("GDN pending transition slots must be non-empty contiguous u32 [batch]");
        }
        if num_k_heads == 0
            || num_v_heads == 0
            || !num_v_heads.is_multiple_of(num_k_heads)
            || head_k_dim != GDN_DECODE_K_DIM
            || head_v_dim != GDN_DECODE_V_DIM
            || conv_dim == 0
            || conv_width == 0
            || conv_width > GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH
            || state_layout != RecurrentStateLayout::GdnValueMajor
        {
            candle::bail!("GDN pending transition apply has incompatible dimensions");
        }
        let (capacity, max_rows, pending_conv_dim) = layers[0].pending_conv_input.dims3()?;
        let conv_blocks = conv_dim.div_ceil(GDN_CHANNEL_BLOCK_SIZE);
        if capacity == 0
            || max_rows == 0
            || max_rows > GDN_SPEC_FUSED_MAX_TOKENS
            || pending_conv_dim != conv_dim
        {
            candle::bail!("GDN pending transition apply storage is incompatible");
        }
        let device = layers[0].pending_conv_input.device();
        if !device.is_cuda() || !active_slots.device().same_device(device) {
            candle::bail!("GDN pending transition apply tensors must share one CUDA device");
        }
        let activation_tensor_dtype = layers[0].pending_conv_input.dtype();
        let recurrent_dtype = layers[0].recurrent_state.dtype();
        if !recurrent_state_dtype_supported(recurrent_dtype) {
            candle::bail!("GDN pending transition apply state has unsupported dtype");
        }

        for layer in layers {
            if layer.pending_conv_input.dims() != [capacity, max_rows, conv_dim]
                || layer.pending_key.dims() != [capacity, max_rows, num_k_heads, head_k_dim]
                || layer.pending_delta.dims() != [capacity, max_rows, num_v_heads, head_v_dim]
                || layer.pending_decay.dims() != [capacity, max_rows, num_v_heads]
                || layer.pending_keep_rows.dims() != [capacity]
                || layer.pending_epochs.dims() != [capacity]
                || layer.conv_applied_epochs.dims() != [capacity, conv_blocks]
                || layer.recurrent_applied_epochs.dims() != [capacity, num_v_heads]
                || layer.conv_state.dims() != [capacity, conv_dim, conv_width]
                || layer.recurrent_state.dims() != [capacity, num_v_heads, head_v_dim, head_k_dim]
            {
                candle::bail!("GDN pending transition apply layer shapes are incompatible");
            }
            if layer.pending_conv_input.dtype() != activation_tensor_dtype
                || layer.conv_state.dtype() != activation_tensor_dtype
                || layer.pending_key.dtype() != DType::F32
                || layer.pending_delta.dtype() != DType::F32
                || layer.pending_decay.dtype() != DType::F32
                || layer.pending_keep_rows.dtype() != DType::U32
                || layer.pending_epochs.dtype() != DType::U32
                || layer.conv_applied_epochs.dtype() != DType::U32
                || layer.recurrent_applied_epochs.dtype() != DType::U32
                || layer.recurrent_state.dtype() != recurrent_dtype
            {
                candle::bail!("GDN pending transition apply layer dtypes are incompatible");
            }
            for tensor in [
                layer.pending_conv_input,
                layer.pending_key,
                layer.pending_delta,
                layer.pending_decay,
                layer.pending_keep_rows,
                layer.pending_epochs,
                layer.conv_applied_epochs,
                layer.recurrent_applied_epochs,
                layer.conv_state,
                layer.recurrent_state,
            ] {
                if !tensor.device().same_device(device) || !tensor.is_contiguous() {
                    candle::bail!(
                        "GDN pending transition apply layer tensors must be contiguous on one device"
                    );
                }
            }
        }
        let cuda_stream = device.as_cuda_device()?.cuda_stream();
        let layer_storage_layouts = layers
            .iter()
            .map(|layer| {
                [
                    layer.pending_conv_input.storage_and_layout(),
                    layer.pending_key.storage_and_layout(),
                    layer.pending_delta.storage_and_layout(),
                    layer.pending_decay.storage_and_layout(),
                    layer.pending_keep_rows.storage_and_layout(),
                    layer.pending_epochs.storage_and_layout(),
                    layer.conv_applied_epochs.storage_and_layout(),
                    layer.recurrent_applied_epochs.storage_and_layout(),
                    layer.conv_state.storage_and_layout(),
                    layer.recurrent_state.storage_and_layout(),
                ]
            })
            .collect::<Vec<_>>();
        let mut pointer_segments = (0..GDN_TRANSITION_APPLY_POINTER_SEGMENTS)
            .map(|_| Vec::with_capacity(layers.len()))
            .collect::<Vec<_>>();
        let mut pointer_guards =
            Vec::with_capacity(GDN_TRANSITION_APPLY_POINTER_SEGMENTS.saturating_mul(layers.len()));
        let mut recurrent_state_dtype = None;
        for storage_layouts in &layer_storage_layouts {
            macro_rules! read_ptr {
                ($segment:expr, $ty:ty, $name:literal) => {{
                    let (storage, layout) = &storage_layouts[$segment];
                    let (pointer, guard) =
                        cuda_read_ptr_with_guard::<$ty>(storage, layout, &cuda_stream, $name)?;
                    pointer_segments[$segment].push(pointer);
                    pointer_guards.push(guard);
                }};
            }
            macro_rules! inplace_ptr {
                ($segment:expr, $ty:ty, $name:literal) => {{
                    let (storage, layout) = &storage_layouts[$segment];
                    let (pointer, guard) =
                        cuda_inplace_ptr_with_guard::<$ty>(storage, layout, &cuda_stream, $name)?;
                    pointer_segments[$segment].push(pointer);
                    pointer_guards.push(guard);
                }};
            }
            read_ptr!(0, T, "pending_conv_input");
            read_ptr!(1, f32, "pending_key");
            read_ptr!(2, f32, "pending_delta");
            read_ptr!(3, f32, "pending_decay");
            read_ptr!(4, u32, "pending_keep_rows");
            read_ptr!(5, u32, "pending_epochs");
            inplace_ptr!(6, u32, "conv_applied_epochs");
            inplace_ptr!(7, u32, "recurrent_applied_epochs");
            inplace_ptr!(8, T, "conv_state");
            let (storage, layout) = &storage_layouts[9];
            let (state_ptr, state_dtype, state_guard) =
                cuda_recurrent_state_inplace_ptr_with_guard(
                    recurrent_dtype,
                    storage,
                    layout,
                    &cuda_stream,
                    "recurrent_state",
                )?;
            if recurrent_state_dtype
                .replace(state_dtype)
                .is_some_and(|dtype| dtype != state_dtype)
            {
                candle::bail!("GDN pending transition apply state dtypes diverged");
            }
            pointer_segments[9].push(state_ptr);
            pointer_guards.push(state_guard);
        }
        let pointers = pointer_segments
            .into_iter()
            .flatten()
            .map(|pointer| pointer as i64)
            .collect::<Vec<_>>();
        let pointer_table = Tensor::from_vec(
            pointers,
            (GDN_TRANSITION_APPLY_POINTER_SEGMENTS * layers.len(),),
            device,
        )?;
        let (pointer_table_storage, pointer_table_layout) = pointer_table.storage_and_layout();
        let (pointer_table_ptr, pointer_table_guard) = cuda_read_ptr_with_guard::<i64>(
            &pointer_table_storage,
            pointer_table_layout,
            &cuda_stream,
            "pointer_table",
        )?;
        let (active_slots_storage, active_slots_layout) = active_slots.storage_and_layout();
        let (active_slots_ptr, active_slots_guard) = cuda_read_ptr_with_guard::<u32>(
            &active_slots_storage,
            active_slots_layout,
            &cuda_stream,
            "active_slots",
        )?;
        let phase_timer = crate::pipeline::cuda_graph::CudaPhaseTimer::start(&cuda_stream)?;
        unsafe {
            crate::cuda::ffi::gdn_pending_transition_apply_batched(
                pointer_table_ptr as *const u64,
                active_slots_ptr as *const u32,
                layers.len() as i32,
                batch_size as i32,
                max_rows as i32,
                capacity as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                conv_dim as i32,
                conv_width as i32,
                i32::from(tiled_v_heads),
                activation_dtype,
                recurrent_state_dtype.expect("layers are non-empty"),
                cuda_stream.cu_stream() as i64,
            );
        }
        drop(active_slots_guard);
        drop(pointer_table_guard);
        drop(pointer_guards);
        if let Some(timer) = phase_timer {
            timer.finish("gdn_transition_apply", batch_size, max_rows)?;
        }
        Ok(())
    }

    match apply
        .layers
        .first()
        .map(|layer| layer.pending_conv_input.dtype())
    {
        Some(DType::F16) => cuda_apply::<half::f16>(apply, 0),
        Some(DType::BF16) => cuda_apply::<half::bf16>(apply, 1),
        Some(other) => candle_core::bail!(
            "GDN pending transition apply only supports f16/bf16 activations, got {other:?}"
        ),
        None => Ok(()),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub fn pending_transition_apply_batched_cuda(_apply: GdnPendingTransitionApply<'_>) -> Result<()> {
    candle_core::bail!("pending_transition_apply_batched_cuda requires the cuda feature")
}

/// CUDA RMSNorm with a SiLU gate; packed final dimensions are split by the norm weight width.
#[cfg(feature = "cuda")]
pub fn rmsnorm_gated_cuda(x: &Tensor, gate: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn normalize_layout(
        dims: &[usize],
        strides: &[usize],
        hidden_dim: usize,
    ) -> Result<([usize; 4], [usize; 4])> {
        match (dims, strides) {
            ([d0, d1], [s0, s1]) if d1 % hidden_dim == 0 => Ok((
                [1, *d0, d1 / hidden_dim, hidden_dim],
                [0, *s0, hidden_dim * *s1, *s1],
            )),
            ([d0, d1, d2], [s0, s1, s2]) if d2 % hidden_dim == 0 => Ok((
                [*d0, *d1, d2 / hidden_dim, hidden_dim],
                [*s0, *s1, hidden_dim * *s2, *s2],
            )),
            ([d0, d1, d2, d3], [s0, s1, s2, s3]) if *d3 == hidden_dim => {
                Ok(([*d0, *d1, *d2, *d3], [*s0, *s1, *s2, *s3]))
            }
            _ => candle::bail!(
                "gated RMSNorm expects rank 2-4 with a final dimension divisible by {hidden_dim}"
            ),
        }
    }

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        x: &Tensor,
        gate: &Tensor,
        weight: &Tensor,
        eps: f64,
        dtype_code: i32,
    ) -> Result<Tensor> {
        let weight = weight.contiguous()?;
        let hidden_dim = weight.dims1()?;
        let dev = x.device().as_cuda_device()?;

        let (x_s, x_l) = x.storage_and_layout();
        let x_s = match &*x_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("x must be a cuda tensor"),
        };
        let x_offset = x_l.start_offset();
        let (dims, x_stride) = normalize_layout(x.dims(), x_l.stride(), hidden_dim)?;

        let (gate_s, gate_l) = gate.storage_and_layout();
        let gate_s = match &*gate_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("gate must be a cuda tensor"),
        };
        let gate_offset = gate_l.start_offset();
        let (gate_dims, gate_stride) = normalize_layout(gate.dims(), gate_l.stride(), hidden_dim)?;
        if gate_dims != dims {
            candle::bail!("gated RMSNorm inputs have incompatible logical shapes");
        }

        let (weight_s, weight_l) = weight.storage_and_layout();
        let weight_s = match &*weight_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };
        let weight_offset = weight_l.start_offset();

        let rows = dims[0] * dims[1] * dims[2];
        let output_buf = unsafe { dev.alloc::<T>(rows * hidden_dim) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        unsafe {
            crate::cuda::ffi::gdn_rmsnorm_gated(
                x_s.slice(x_offset..).device_ptr(x_s.stream()).0 as *const c_void,
                gate_s.slice(gate_offset..).device_ptr(gate_s.stream()).0 as *const c_void,
                weight_s
                    .slice(weight_offset..)
                    .device_ptr(weight_s.stream())
                    .0 as *const c_void,
                output_buf.device_ptr(output_buf.stream()).0 as *mut c_void,
                rows as i32,
                hidden_dim as i32,
                dims[1] as i32,
                dims[2] as i32,
                x_stride[0] as i64,
                x_stride[1] as i64,
                x_stride[2] as i64,
                x_stride[3] as i64,
                gate_stride[0] as i64,
                gate_stride[1] as i64,
                gate_stride[2] as i64,
                gate_stride[3] as i64,
                eps as f32,
                dtype_code,
                stream,
            );
        }

        let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
        Ok(Tensor::from((
            candle::Storage::Cuda(output_storage),
            x.shape().clone(),
        )))
    }

    match x.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(x, gate, weight, eps, 0),
        DType::BF16 => cuda_fwd::<half::bf16>(x, gate, weight, eps, 1),
        other => candle_core::bail!("rmsnorm_gated_cuda only supports f16/bf16, got {:?}", other),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn rmsnorm_gated_cuda(
    _x: &Tensor,
    _gate: &Tensor,
    _weight: &Tensor,
    _eps: f64,
) -> Result<Tensor> {
    candle_core::bail!("rmsnorm_gated_cuda requires the cuda feature")
}

/// b, a: [total_elements] in f16/bf16
/// a_log, dt_bias: [num_heads] in f32
///
/// Returns: (beta, g) in original dtype
#[cfg(feature = "cuda")]
pub fn fused_gdn_gating_cuda(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        b: &Tensor,
        a: &Tensor,
        a_log: &Tensor,
        dt_bias: &Tensor,
        dtype_code: i32,
    ) -> Result<(Tensor, Tensor)> {
        let total_elements = b.elem_count();
        let num_heads = a_log.elem_count();
        let shape = b.shape().clone();
        let dev = b.device().as_cuda_device()?;

        let (b_s, b_l) = b.storage_and_layout();
        let b_s = match &*b_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("b must be a cuda tensor"),
        };
        let b_offset = b_l.start_offset();

        let (a_s, a_l) = a.storage_and_layout();
        let a_s = match &*a_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("a must be a cuda tensor"),
        };
        let a_offset = a_l.start_offset();

        let (alog_s, alog_l) = a_log.storage_and_layout();
        let alog_s = match &*alog_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("a_log must be a cuda tensor"),
        };
        let alog_offset = alog_l.start_offset();

        let (dtb_s, dtb_l) = dt_bias.storage_and_layout();
        let dtb_s = match &*dtb_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("dt_bias must be a cuda tensor"),
        };
        let dtb_offset = dtb_l.start_offset();

        let beta_buf = unsafe { dev.alloc::<T>(total_elements) }?;
        let g_buf = unsafe { dev.alloc::<T>(total_elements) }?;

        let stream = dev.cuda_stream().cu_stream() as i64;

        unsafe {
            crate::cuda::ffi::fused_gdn_gating(
                b_s.slice(b_offset..).device_ptr(b_s.stream()).0 as *const c_void,
                a_s.slice(a_offset..).device_ptr(a_s.stream()).0 as *const c_void,
                alog_s.slice(alog_offset..).device_ptr(alog_s.stream()).0 as *const f32,
                dtb_s.slice(dtb_offset..).device_ptr(dtb_s.stream()).0 as *const f32,
                beta_buf.device_ptr(beta_buf.stream()).0 as *mut c_void,
                g_buf.device_ptr(g_buf.stream()).0 as *mut c_void,
                total_elements as i32,
                num_heads as i32,
                dtype_code,
                stream,
            );
        }

        let beta_storage = candle::CudaStorage::wrap_cuda_slice(beta_buf, dev.clone());
        let beta = Tensor::from((candle::Storage::Cuda(beta_storage), shape.clone()));

        let g_storage = candle::CudaStorage::wrap_cuda_slice(g_buf, dev.clone());
        let g = Tensor::from((candle::Storage::Cuda(g_storage), shape));

        Ok((beta, g))
    }

    match b.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(b, a, a_log, dt_bias, 0),
        DType::BF16 => cuda_fwd::<half::bf16>(b, a, a_log, dt_bias, 1),
        other => candle_core::bail!(
            "fused_gdn_gating_cuda only supports f16/bf16, got {:?}",
            other
        ),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn fused_gdn_gating_cuda(
    _b: &Tensor,
    _a: &Tensor,
    _a_log: &Tensor,
    _dt_bias: &Tensor,
) -> Result<(Tensor, Tensor)> {
    candle_core::bail!("fused_gdn_gating_cuda requires the cuda feature")
}

#[cfg(test)]
mod dispatch_tests {
    use super::{
        automatic_decode_kernel, automatic_prefill_kernel, parse_decode_kernel,
        parse_prefill_kernel, prefill_kernel_supported, select_decode_kernel,
        select_prefill_kernel, GdnDecodeKernel, GdnDecodePolicy, GdnPrefillKernel,
        GdnPrefillPolicy,
    };
    use crate::kv_cache::RecurrentStateLayout;

    const TEST_SM_COUNT: usize = 132;

    fn sm90_policy(state_blocks: usize) -> GdnDecodePolicy {
        GdnDecodePolicy {
            compute_major: 9,
            multiprocessor_count: TEST_SM_COUNT,
            state_blocks,
            head_k_dim: 128,
            head_v_dim: 128,
            vector_aligned: true,
            bf16: true,
            state_layout: RecurrentStateLayout::GdnKeyMajor,
        }
    }

    fn sm90_value_major_policy(batch_size: usize) -> GdnDecodePolicy {
        GdnDecodePolicy {
            state_blocks: batch_size * 48,
            state_layout: RecurrentStateLayout::GdnValueMajor,
            ..sm90_policy(0)
        }
    }

    fn sm90_prefill_policy(state_blocks: usize, seq_len: usize) -> GdnPrefillPolicy {
        GdnPrefillPolicy {
            compute_major: 9,
            multiprocessor_count: TEST_SM_COUNT,
            state_blocks,
            seq_len,
            head_k_dim: 128,
            head_v_dim: 128,
            bf16: true,
            state_layout: RecurrentStateLayout::GdnValueMajor,
        }
    }

    #[test]
    fn decode_dispatch_uses_sm90_state_waves() {
        assert_eq!(
            automatic_decode_kernel(sm90_policy(TEST_SM_COUNT)),
            GdnDecodeKernel::Cooperative
        );
        assert_eq!(
            automatic_decode_kernel(sm90_policy(2 * TEST_SM_COUNT - 1)),
            GdnDecodeKernel::Cooperative
        );
        assert_eq!(
            automatic_decode_kernel(sm90_policy(2 * TEST_SM_COUNT)),
            GdnDecodeKernel::Pipelined
        );
        assert_eq!(
            automatic_decode_kernel(sm90_policy(3 * TEST_SM_COUNT)),
            GdnDecodeKernel::Pipelined
        );
        assert_eq!(
            automatic_decode_kernel(sm90_policy(4 * TEST_SM_COUNT)),
            GdnDecodeKernel::Baseline
        );
        assert_eq!(
            automatic_decode_kernel(sm90_policy(6 * TEST_SM_COUNT)),
            GdnDecodeKernel::Baseline
        );
        assert_eq!(
            automatic_decode_kernel(sm90_policy(8 * TEST_SM_COUNT - 1)),
            GdnDecodeKernel::Baseline
        );
        assert_eq!(
            automatic_decode_kernel(sm90_policy(8 * TEST_SM_COUNT)),
            GdnDecodeKernel::Pipelined
        );
    }

    #[test]
    fn automatic_decode_dispatch_requires_a_tuned_architecture_and_shape() {
        let mut unsupported = sm90_policy(12 * TEST_SM_COUNT);
        unsupported.compute_major = 8;
        assert_eq!(
            automatic_decode_kernel(unsupported),
            GdnDecodeKernel::Baseline
        );
        unsupported = sm90_policy(12 * TEST_SM_COUNT);
        unsupported.compute_major = 10;
        assert_eq!(
            automatic_decode_kernel(unsupported),
            GdnDecodeKernel::Baseline
        );
        unsupported = sm90_policy(12 * TEST_SM_COUNT);
        unsupported.head_k_dim = 64;
        assert_eq!(
            automatic_decode_kernel(unsupported),
            GdnDecodeKernel::Baseline
        );
        unsupported = sm90_policy(12 * TEST_SM_COUNT);
        unsupported.head_v_dim = 127;
        assert_eq!(
            automatic_decode_kernel(unsupported),
            GdnDecodeKernel::Baseline
        );
        unsupported = sm90_policy(12 * TEST_SM_COUNT);
        unsupported.head_v_dim = 32;
        assert_eq!(
            automatic_decode_kernel(unsupported),
            GdnDecodeKernel::Baseline
        );
        unsupported = sm90_policy(12 * TEST_SM_COUNT);
        unsupported.vector_aligned = false;
        assert_eq!(
            automatic_decode_kernel(unsupported),
            GdnDecodeKernel::Baseline
        );
    }

    #[test]
    fn value_major_dispatch_uses_the_measured_sm90_crossover() {
        for batch_size in 1..=5 {
            assert_eq!(
                automatic_decode_kernel(sm90_value_major_policy(batch_size)),
                GdnDecodeKernel::ValueMajor4
            );
        }
        for batch_size in [6, 8, 16] {
            assert_eq!(
                automatic_decode_kernel(sm90_value_major_policy(batch_size)),
                GdnDecodeKernel::ValueMajor32
            );
        }

        let mut unsupported = sm90_value_major_policy(8);
        unsupported.bf16 = false;
        assert!(!super::decode_kernel_supported(
            GdnDecodeKernel::ValueMajor32,
            unsupported
        ));
        unsupported = sm90_value_major_policy(8);
        unsupported.compute_major = 10;
        assert!(!super::decode_kernel_supported(
            GdnDecodeKernel::ValueMajor32,
            unsupported
        ));
    }

    #[test]
    fn decode_kernel_override_is_forceable_and_validated() {
        assert_eq!(parse_decode_kernel("auto").unwrap(), None);
        assert_eq!(
            parse_decode_kernel("BASELINE").unwrap(),
            Some(GdnDecodeKernel::Baseline)
        );
        assert_eq!(
            parse_decode_kernel("cooperative").unwrap(),
            Some(GdnDecodeKernel::Cooperative)
        );
        assert_eq!(
            parse_decode_kernel("pipelined").unwrap(),
            Some(GdnDecodeKernel::Pipelined)
        );
        assert_eq!(
            parse_decode_kernel("vmajor4").unwrap(),
            Some(GdnDecodeKernel::ValueMajor4)
        );
        assert_eq!(
            parse_decode_kernel("vmajor32").unwrap(),
            Some(GdnDecodeKernel::ValueMajor32)
        );
        assert!(parse_decode_kernel("unknown").is_err());

        let policy = sm90_policy(3 * TEST_SM_COUNT);
        assert_eq!(
            select_decode_kernel(policy, Some(GdnDecodeKernel::Baseline)).unwrap(),
            GdnDecodeKernel::Baseline
        );
        let mut unsupported = policy;
        unsupported.compute_major = 8;
        assert_eq!(
            select_decode_kernel(unsupported, Some(GdnDecodeKernel::Pipelined)),
            Err(GdnDecodeKernel::Pipelined)
        );
        let mut untuned = policy;
        untuned.compute_major = 10;
        assert_eq!(
            select_decode_kernel(untuned, Some(GdnDecodeKernel::Pipelined)).unwrap(),
            GdnDecodeKernel::Pipelined
        );
    }

    #[test]
    fn prefill_auto_dispatch_tracks_available_parallelism() {
        for (state_blocks, seq_len, expected) in [
            (1, 2, GdnPrefillKernel::ValueMajor2),
            (48, 64, GdnPrefillKernel::ValueMajor2),
            (48, 8_192, GdnPrefillKernel::ValueMajor2),
            (8 * 48, 2_048, GdnPrefillKernel::ValueMajor4),
            (16 * 48, 2_048, GdnPrefillKernel::ValueMajor8),
        ] {
            assert_eq!(
                automatic_prefill_kernel(sm90_prefill_policy(state_blocks, seq_len)),
                expected
            );
        }
    }

    #[test]
    fn prefill_kernel_override_is_forceable_and_validated() {
        assert_eq!(parse_prefill_kernel("auto").unwrap(), None);
        assert_eq!(
            parse_prefill_kernel("flashinfer-sm90").unwrap(),
            Some(GdnPrefillKernel::FlashInferSm90)
        );
        assert_eq!(
            parse_prefill_kernel("VMAJOR1").unwrap(),
            Some(GdnPrefillKernel::ValueMajor1)
        );
        assert_eq!(
            parse_prefill_kernel("vmajor2").unwrap(),
            Some(GdnPrefillKernel::ValueMajor2)
        );
        assert_eq!(
            parse_prefill_kernel("vmajor4").unwrap(),
            Some(GdnPrefillKernel::ValueMajor4)
        );
        assert_eq!(
            parse_prefill_kernel("vmajor8").unwrap(),
            Some(GdnPrefillKernel::ValueMajor8)
        );
        assert_eq!(
            parse_prefill_kernel("legacy-chunked").unwrap(),
            Some(GdnPrefillKernel::LegacyChunked)
        );
        assert!(parse_prefill_kernel("unknown").is_err());

        let policy = sm90_prefill_policy(48, 129);
        assert!(!prefill_kernel_supported(
            GdnPrefillKernel::FlashInferSm90,
            policy
        ));
        for kernel in [
            GdnPrefillKernel::ValueMajor1,
            GdnPrefillKernel::ValueMajor2,
            GdnPrefillKernel::ValueMajor4,
            GdnPrefillKernel::ValueMajor8,
            GdnPrefillKernel::LegacyChunked,
        ] {
            assert!(prefill_kernel_supported(kernel, policy));
            assert_eq!(select_prefill_kernel(policy, Some(kernel)), Ok(kernel));
        }

        let mut unsupported = policy;
        unsupported.compute_major = 8;
        assert_eq!(
            select_prefill_kernel(unsupported, Some(GdnPrefillKernel::ValueMajor4)),
            Err(GdnPrefillKernel::ValueMajor4)
        );
        unsupported = policy;
        unsupported.bf16 = false;
        assert!(!prefill_kernel_supported(
            GdnPrefillKernel::ValueMajor4,
            unsupported
        ));
        unsupported = policy;
        unsupported.head_v_dim = 64;
        assert!(!prefill_kernel_supported(
            GdnPrefillKernel::ValueMajor4,
            unsupported
        ));
        unsupported = policy;
        unsupported.seq_len = 0;
        assert!(!prefill_kernel_supported(
            GdnPrefillKernel::ValueMajor4,
            unsupported
        ));
    }

    fn state_index(
        layout: RecurrentStateLayout,
        key: usize,
        value: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> usize {
        match layout {
            RecurrentStateLayout::GdnKeyMajor => key * value_dim + value,
            RecurrentStateLayout::GdnValueMajor => value * key_dim + key,
            RecurrentStateLayout::Opaque => unreachable!(),
        }
    }

    struct ReferenceRecurrence<'a> {
        q: &'a [f32],
        k: &'a [f32],
        v: &'a [f32],
        g: &'a [f32],
        beta: &'a [f32],
        key_dim: usize,
        value_dim: usize,
    }

    fn reference_recurrence(
        layout: RecurrentStateLayout,
        state: &mut [f32],
        inputs: &ReferenceRecurrence<'_>,
    ) -> Vec<f32> {
        let seq_len = inputs.g.len();
        let mut output = vec![0.0f32; seq_len * inputs.value_dim];
        for token in 0..seq_len {
            let decay = inputs.g[token].exp();
            for value in 0..inputs.value_dim {
                let mut state_dot_k = 0.0f32;
                for key in 0..inputs.key_dim {
                    let index = state_index(layout, key, value, inputs.key_dim, inputs.value_dim);
                    state[index] *= decay;
                    state_dot_k += state[index] * inputs.k[token * inputs.key_dim + key];
                }
                let delta =
                    (inputs.v[token * inputs.value_dim + value] - state_dot_k) * inputs.beta[token];
                let mut state_dot_q = 0.0f32;
                for key in 0..inputs.key_dim {
                    let index = state_index(layout, key, value, inputs.key_dim, inputs.value_dim);
                    state[index] += inputs.k[token * inputs.key_dim + key] * delta;
                    state_dot_q += state[index] * inputs.q[token * inputs.key_dim + key];
                }
                output[token * inputs.value_dim + value] = state_dot_q;
            }
        }
        output
    }

    #[test]
    fn recurrence_reference_is_layout_invariant() {
        const KEY_DIM: usize = 3;
        const VALUE_DIM: usize = 2;
        const SEQ_LEN: usize = 5;

        let key_major = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
        let mut value_major = vec![0.0f32; KEY_DIM * VALUE_DIM];
        for key in 0..KEY_DIM {
            for value in 0..VALUE_DIM {
                value_major[state_index(
                    RecurrentStateLayout::GdnValueMajor,
                    key,
                    value,
                    KEY_DIM,
                    VALUE_DIM,
                )] = key_major[state_index(
                    RecurrentStateLayout::GdnKeyMajor,
                    key,
                    value,
                    KEY_DIM,
                    VALUE_DIM,
                )];
            }
        }

        let q = (0..SEQ_LEN * KEY_DIM)
            .map(|index| (f32::from(u16::try_from(index).unwrap()) - 5.0) * 0.03)
            .collect::<Vec<_>>();
        let k = (0..SEQ_LEN * KEY_DIM)
            .map(|index| (7.0 - f32::from(u16::try_from(index).unwrap())) * 0.02)
            .collect::<Vec<_>>();
        let v = (0..SEQ_LEN * VALUE_DIM)
            .map(|index| (f32::from(u16::try_from(index).unwrap()) - 3.0) * 0.04)
            .collect::<Vec<_>>();
        let g = vec![-0.01, -0.03, -0.02, -0.04, -0.05];
        let beta = vec![0.2, 0.7, 0.4, 0.9, 0.5];
        let inputs = ReferenceRecurrence {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
            key_dim: KEY_DIM,
            value_dim: VALUE_DIM,
        };

        let mut key_major_result = key_major;
        let key_major_output = reference_recurrence(
            RecurrentStateLayout::GdnKeyMajor,
            &mut key_major_result,
            &inputs,
        );
        let value_major_output = reference_recurrence(
            RecurrentStateLayout::GdnValueMajor,
            &mut value_major,
            &inputs,
        );

        for (left, right) in key_major_output.iter().zip(&value_major_output) {
            assert!((left - right).abs() <= 1.0e-6);
        }
        for key in 0..KEY_DIM {
            for value in 0..VALUE_DIM {
                let key_major_index = state_index(
                    RecurrentStateLayout::GdnKeyMajor,
                    key,
                    value,
                    KEY_DIM,
                    VALUE_DIM,
                );
                let value_major_index = state_index(
                    RecurrentStateLayout::GdnValueMajor,
                    key,
                    value,
                    KEY_DIM,
                    VALUE_DIM,
                );
                assert!(
                    (key_major_result[key_major_index] - value_major[value_major_index]).abs()
                        <= 1.0e-6
                );
            }
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::{Device, D};

    #[derive(Clone, Copy)]
    struct RecurrenceCase {
        bh: usize,
        seq_len: usize,
        k_dim: usize,
        v_dim: usize,
    }

    fn patterned(len: usize, salt: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let x = ((i.wrapping_mul(37) + salt.wrapping_mul(17)) % 257) as u16 as f32;
                ((x / 128.0) - 1.0) * scale + offset
            })
            .collect()
    }

    fn tensor2(data: Vec<f32>, shape: (usize, usize), dev: &Device) -> Result<Tensor> {
        Tensor::from_vec(data, shape, dev)
    }

    fn tensor3(data: Vec<f32>, shape: (usize, usize, usize), dev: &Device) -> Result<Tensor> {
        Tensor::from_vec(data, shape, dev)
    }

    fn flat(tensor: &Tensor) -> Result<Vec<f32>> {
        tensor
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()
    }

    fn max_abs_diff(lhs: &[f32], rhs: &[f32]) -> (f32, usize, f32, f32) {
        let mut max_diff = 0.0f32;
        let mut max_idx = 0usize;
        let mut lhs_at_max = 0.0f32;
        let mut rhs_at_max = 0.0f32;
        for (idx, (&left, &right)) in lhs.iter().zip(rhs).enumerate() {
            let diff = (left - right).abs();
            if diff > max_diff || diff.is_nan() {
                max_diff = diff;
                max_idx = idx;
                lhs_at_max = left;
                rhs_at_max = right;
            }
        }
        (max_diff, max_idx, lhs_at_max, rhs_at_max)
    }

    fn assert_close(label: &str, lhs: &[f32], rhs: &[f32], tol: f32) {
        let lhs_nan = lhs.iter().filter(|x| x.is_nan()).count();
        let rhs_nan = rhs.iter().filter(|x| x.is_nan()).count();
        let (max_diff, max_idx, lhs_at_max, rhs_at_max) = max_abs_diff(lhs, rhs);
        assert!(
            lhs_nan == 0 && rhs_nan == 0 && max_diff <= tol,
            "{label}: max_diff={max_diff} at {max_idx}, lhs={lhs_at_max}, rhs={rhs_at_max}, lhs_nan={lhs_nan}, rhs_nan={rhs_nan}"
        );
    }

    fn run_case(case: RecurrenceCase, dev: &Device) -> Result<()> {
        let q = tensor3(
            patterned(case.bh * case.seq_len * case.k_dim, 1, 0.02, 0.0),
            (case.bh, case.seq_len, case.k_dim),
            dev,
        )?;
        let k = tensor3(
            patterned(case.bh * case.seq_len * case.k_dim, 2, 0.02, 0.0),
            (case.bh, case.seq_len, case.k_dim),
            dev,
        )?;
        let v = tensor3(
            patterned(case.bh * case.seq_len * case.v_dim, 3, 0.05, 0.0),
            (case.bh, case.seq_len, case.v_dim),
            dev,
        )?;
        let g = tensor2(
            patterned(case.bh * case.seq_len, 4, 0.03, -0.08),
            (case.bh, case.seq_len),
            dev,
        )?;
        let beta = tensor2(
            patterned(case.bh * case.seq_len, 5, 0.15, 0.5),
            (case.bh, case.seq_len),
            dev,
        )?;
        let state = patterned(case.bh * case.k_dim * case.v_dim, 6, 0.01, 0.0);

        let mut state_scalar = tensor3(state.clone(), (case.bh, case.k_dim, case.v_dim), dev)?;
        let scalar = gated_delta_rule_recurrence_cuda(
            RecurrenceInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            },
            &mut state_scalar,
            GdnStateSlots::Gathered,
        )?;
        let mut state_chunked = tensor3(state.clone(), (case.bh, case.k_dim, case.v_dim), dev)?;
        let chunked = chunked_gated_delta_rule_recurrence_cuda(
            RecurrenceInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            },
            &mut state_chunked,
            GdnStateSlots::Gathered,
        )?;
        let mut state_warp = tensor3(state, (case.bh, case.k_dim, case.v_dim), dev)?;
        let warp = warp_gated_delta_rule_recurrence_cuda(
            RecurrenceInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            },
            &mut state_warp,
            GdnStateSlots::Gathered,
        )?;

        let scalar_flat = flat(&scalar)?;
        let scalar_state_flat = flat(&state_scalar)?;
        let chunked_flat = flat(&chunked)?;
        let chunked_state_flat = flat(&state_chunked)?;
        let warp_flat = flat(&warp)?;
        let warp_state_flat = flat(&state_warp)?;

        let name = format!(
            "bh={},seq={},k={},v={}",
            case.bh, case.seq_len, case.k_dim, case.v_dim
        );
        assert_close(
            &format!("{name} chunked output"),
            &scalar_flat,
            &chunked_flat,
            3.0e-4,
        );
        assert_close(
            &format!("{name} chunked state"),
            &scalar_state_flat,
            &chunked_state_flat,
            3.0e-4,
        );
        assert_close(
            &format!("{name} warp output"),
            &scalar_flat,
            &warp_flat,
            3.0e-4,
        );
        assert_close(
            &format!("{name} warp state"),
            &scalar_state_flat,
            &warp_state_flat,
            3.0e-4,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn warp_recurrence_matches_scalar_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for case in [
            RecurrenceCase {
                bh: 1,
                seq_len: 1,
                k_dim: 64,
                v_dim: 64,
            },
            RecurrenceCase {
                bh: 1,
                seq_len: 65,
                k_dim: 64,
                v_dim: 64,
            },
            RecurrenceCase {
                bh: 2,
                seq_len: 128,
                k_dim: 128,
                v_dim: 64,
            },
            RecurrenceCase {
                bh: 8,
                seq_len: 256,
                k_dim: 128,
                v_dim: 64,
            },
            RecurrenceCase {
                bh: 32,
                seq_len: 512,
                k_dim: 128,
                v_dim: 128,
            },
        ] {
            run_case(case, &dev)?;
        }
        Ok(())
    }

    fn run_low_dtype_sequential_recurrence_case(
        dev: &Device,
        state_dtype: DType,
        kernel: RecurrenceKernel,
        head_dim: usize,
    ) -> Result<()> {
        const BATCH_SIZE: usize = 2;
        const NUM_HEADS: usize = 2;
        const CAPACITY: usize = 4;
        const SEQ_LEN: usize = 5;
        const STEPS: usize = 3;

        let bh = BATCH_SIZE * NUM_HEADS;
        let q = tensor3(
            patterned(bh * SEQ_LEN * head_dim, 8, 0.02, 0.0),
            (bh, SEQ_LEN, head_dim),
            dev,
        )?;
        let k = tensor3(
            patterned(bh * SEQ_LEN * head_dim, 9, 0.02, 0.0),
            (bh, SEQ_LEN, head_dim),
            dev,
        )?;
        let v = tensor3(
            patterned(bh * SEQ_LEN * head_dim, 10, 0.05, 0.0),
            (bh, SEQ_LEN, head_dim),
            dev,
        )?;
        let g = tensor2(patterned(bh * SEQ_LEN, 11, 0.03, -0.08), (bh, SEQ_LEN), dev)?;
        let beta = tensor2(patterned(bh * SEQ_LEN, 12, 0.15, 0.5), (bh, SEQ_LEN), dev)?;
        let state_shape = (CAPACITY, NUM_HEADS, head_dim, head_dim);
        let initial = Tensor::from_vec(
            patterned(CAPACITY * NUM_HEADS * head_dim * head_dim, 13, 0.01, 0.0),
            state_shape,
            dev,
        )?
        .to_dtype(state_dtype)?;
        let mut low_state = initial.copy()?;
        let mut reference_state = initial.to_dtype(DType::F32)?;
        let slot_indices =
            Tensor::from_vec(vec![CAPACITY as u32 - 1, GDN_PAD_SLOT], (BATCH_SIZE,), dev)?;
        let slots = GdnStateSlots::Pooled(&slot_indices);
        let inputs = RecurrenceInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
        };

        for step in 0..STEPS {
            let low_output = launch_recurrence(kernel, inputs, &mut low_state, slots)?;
            let reference_output = launch_recurrence(kernel, inputs, &mut reference_state, slots)?;
            assert_close(
                &format!("{kernel:?} {state_dtype:?} output step {step}"),
                &flat(&low_output)?,
                &flat(&reference_output)?,
                3.0e-5,
            );
            reference_state = reference_state
                .to_dtype(state_dtype)?
                .to_dtype(DType::F32)?;
            assert_close(
                &format!("{kernel:?} {state_dtype:?} state step {step}"),
                &flat(&low_state.to_dtype(DType::F32)?)?,
                &flat(&reference_state)?,
                0.0,
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn low_dtype_recurrence_matches_sequential_rounding_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for state_dtype in [DType::BF16, DType::F16] {
            for kernel in [
                RecurrenceKernel::Scalar,
                RecurrenceKernel::Warp,
                RecurrenceKernel::Chunked,
            ] {
                run_low_dtype_sequential_recurrence_case(&dev, state_dtype, kernel, 64)?;
            }
            run_low_dtype_sequential_recurrence_case(
                &dev,
                state_dtype,
                RecurrenceKernel::ValueMajorWarp,
                128,
            )?;
            for kernel in [
                RecurrenceKernel::ValueMajorWarp2,
                RecurrenceKernel::ValueMajorWarp4,
                RecurrenceKernel::ValueMajorWarp8,
            ] {
                run_low_dtype_sequential_recurrence_case(&dev, state_dtype, kernel, 128)?;
            }
            run_low_dtype_sequential_recurrence_case(
                &dev,
                state_dtype,
                RecurrenceKernel::ValueMajorChunked,
                128,
            )?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn value_major_prefill_kernels_match_scalar_with_shuffled_slots() -> Result<()> {
        const BATCH_SIZE: usize = 2;
        const NUM_HEADS: usize = 4;
        const SEQ_LEN: usize = 129;
        const HEAD_DIM: usize = 128;
        const CAPACITY: usize = 5;

        let dev = Device::new_cuda(0)?;
        let bh = BATCH_SIZE * NUM_HEADS;
        let q = tensor3(
            patterned(bh * SEQ_LEN * HEAD_DIM, 20, 0.02, 0.0),
            (bh, SEQ_LEN, HEAD_DIM),
            &dev,
        )?;
        let k = tensor3(
            patterned(bh * SEQ_LEN * HEAD_DIM, 21, 0.02, 0.0),
            (bh, SEQ_LEN, HEAD_DIM),
            &dev,
        )?;
        let v = tensor3(
            patterned(bh * SEQ_LEN * HEAD_DIM, 22, 0.05, 0.0),
            (bh, SEQ_LEN, HEAD_DIM),
            &dev,
        )?;
        let g = tensor2(
            patterned(bh * SEQ_LEN, 23, 0.03, -0.08),
            (bh, SEQ_LEN),
            &dev,
        )?;
        let beta = tensor2(patterned(bh * SEQ_LEN, 24, 0.15, 0.5), (bh, SEQ_LEN), &dev)?;
        let initial_state = Tensor::from_vec(
            patterned(CAPACITY * NUM_HEADS * HEAD_DIM * HEAD_DIM, 25, 0.01, 0.0),
            (CAPACITY, NUM_HEADS, HEAD_DIM, HEAD_DIM),
            &dev,
        )?;
        let mut key_major_state = initial_state.clone();
        let value_major_state = initial_state.transpose(2, 3)?.contiguous()?;
        let mut value_major_warp_state = value_major_state.copy()?;
        let mut value_major_chunked_state = value_major_state.copy()?;
        let slot_indices = Tensor::from_vec(vec![4u32, 1], (BATCH_SIZE,), &dev)?;
        let slots = GdnStateSlots::Pooled(&slot_indices);
        let inputs = RecurrenceInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
        };

        for step in 1..=2 {
            let reference = gated_delta_rule_recurrence_cuda(inputs, &mut key_major_state, slots)?;
            let value_major_warp = vmajor_warp_gated_delta_rule_recurrence_cuda(
                inputs,
                &mut value_major_warp_state,
                slots,
            )?;
            let value_major_chunked = vmajor_chunked_gated_delta_rule_recurrence_cuda(
                inputs,
                &mut value_major_chunked_state,
                slots,
            )?;
            assert_close(
                &format!("value-major warp prefill output step {step}"),
                &flat(&value_major_warp)?,
                &flat(&reference)?,
                3.0e-4,
            );
            assert_close(
                &format!("value-major warp prefill state step {step}"),
                &flat(&value_major_warp_state.transpose(2, 3)?.contiguous()?)?,
                &flat(&key_major_state)?,
                3.0e-4,
            );
            assert_close(
                &format!("value-major chunked prefill output step {step}"),
                &flat(&value_major_chunked)?,
                &flat(&reference)?,
                3.0e-4,
            );
            assert_close(
                &format!("value-major chunked prefill state step {step}"),
                &flat(&value_major_chunked_state.transpose(2, 3)?.contiguous()?)?,
                &flat(&key_major_state)?,
                3.0e-4,
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an SM90 CUDA device"]
    fn value_major_grouped_prefill_matches_warp_at_sequence_boundaries() -> Result<()> {
        const BH: usize = 48;
        const HEAD_DIM: usize = 128;

        let dev = Device::new_cuda(0)?;
        for seq_len in [2usize, 63, 64, 65, 129] {
            let q = tensor3(
                patterned(BH * seq_len * HEAD_DIM, 120, 0.02, 0.0),
                (BH, seq_len, HEAD_DIM),
                &dev,
            )?;
            let k = tensor3(
                patterned(BH * seq_len * HEAD_DIM, 121, 0.02, 0.0),
                (BH, seq_len, HEAD_DIM),
                &dev,
            )?;
            let v = tensor3(
                patterned(BH * seq_len * HEAD_DIM, 122, 0.05, 0.0),
                (BH, seq_len, HEAD_DIM),
                &dev,
            )?;
            let g = tensor2(
                patterned(BH * seq_len, 123, 0.03, -0.08),
                (BH, seq_len),
                &dev,
            )?;
            let beta = tensor2(patterned(BH * seq_len, 124, 0.15, 0.5), (BH, seq_len), &dev)?;
            let initial = Tensor::from_vec(
                patterned(BH * HEAD_DIM * HEAD_DIM, 125, 0.01, 0.0),
                (BH, HEAD_DIM, HEAD_DIM),
                &dev,
            )?;
            let inputs = RecurrenceInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            };
            let mut reference_state = initial.copy()?;
            let reference = launch_recurrence(
                RecurrenceKernel::ValueMajorWarp,
                inputs,
                &mut reference_state,
                GdnStateSlots::Gathered,
            )?;
            let reference_output = flat(&reference)?;
            let reference_state = flat(&reference_state)?;

            for kernel in [
                RecurrenceKernel::ValueMajorWarp2,
                RecurrenceKernel::ValueMajorWarp4,
                RecurrenceKernel::ValueMajorWarp8,
            ] {
                let mut state = initial.copy()?;
                let output =
                    launch_recurrence(kernel, inputs, &mut state, GdnStateSlots::Gathered)?;
                assert_close(
                    &format!("{kernel:?} S={seq_len} output"),
                    &flat(&output)?,
                    &reference_output,
                    0.0,
                );
                assert_close(
                    &format!("{kernel:?} S={seq_len} state"),
                    &flat(&state)?,
                    &reference_state,
                    0.0,
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an SM90 CUDA device"]
    fn value_major_grouped_prefill_preserves_pooled_padding() -> Result<()> {
        const BATCH_SIZE: usize = 3;
        const NUM_HEADS: usize = 48;
        const CAPACITY: usize = 5;
        const SEQ_LEN: usize = 65;
        const HEAD_DIM: usize = 128;

        let dev = Device::new_cuda(0)?;
        let bh = BATCH_SIZE * NUM_HEADS;
        let q = tensor3(
            patterned(bh * SEQ_LEN * HEAD_DIM, 130, 0.02, 0.0),
            (bh, SEQ_LEN, HEAD_DIM),
            &dev,
        )?;
        let k = tensor3(
            patterned(bh * SEQ_LEN * HEAD_DIM, 131, 0.02, 0.0),
            (bh, SEQ_LEN, HEAD_DIM),
            &dev,
        )?;
        let v = tensor3(
            patterned(bh * SEQ_LEN * HEAD_DIM, 132, 0.05, 0.0),
            (bh, SEQ_LEN, HEAD_DIM),
            &dev,
        )?;
        let g = tensor2(
            patterned(bh * SEQ_LEN, 133, 0.03, -0.08),
            (bh, SEQ_LEN),
            &dev,
        )?;
        let beta = tensor2(patterned(bh * SEQ_LEN, 134, 0.15, 0.5), (bh, SEQ_LEN), &dev)?;
        let initial_host = patterned(CAPACITY * NUM_HEADS * HEAD_DIM * HEAD_DIM, 135, 0.01, 0.0);
        let initial = Tensor::from_vec(
            initial_host.clone(),
            (CAPACITY, NUM_HEADS, HEAD_DIM, HEAD_DIM),
            &dev,
        )?;
        let slot_indices = Tensor::from_vec(vec![4u32, GDN_PAD_SLOT, 1], (BATCH_SIZE,), &dev)?;
        let slots = GdnStateSlots::Pooled(&slot_indices);
        let inputs = RecurrenceInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
        };
        let mut reference_state = initial.copy()?;
        let reference = launch_recurrence(
            RecurrenceKernel::ValueMajorWarp,
            inputs,
            &mut reference_state,
            slots,
        )?;
        let reference_output = flat(&reference)?;
        let reference_state = flat(&reference_state)?;

        for kernel in [
            RecurrenceKernel::ValueMajorWarp2,
            RecurrenceKernel::ValueMajorWarp4,
            RecurrenceKernel::ValueMajorWarp8,
        ] {
            let mut state = initial.copy()?;
            let output = launch_recurrence(kernel, inputs, &mut state, slots)?;
            assert_close(
                &format!("{kernel:?} pooled output"),
                &flat(&output)?,
                &reference_output,
                0.0,
            );
            let state_host = flat(&state)?;
            assert_close(
                &format!("{kernel:?} pooled state"),
                &state_host,
                &reference_state,
                0.0,
            );
            assert_zero(
                &format!("{kernel:?} padding output"),
                &output.narrow(0, NUM_HEADS, NUM_HEADS)?,
            )?;
            for row in [0usize, 2, 3] {
                let span = NUM_HEADS * HEAD_DIM * HEAD_DIM;
                assert_close(
                    &format!("{kernel:?} untouched row {row}"),
                    &state_host[row * span..(row + 1) * span],
                    &initial_host[row * span..(row + 1) * span],
                    0.0,
                );
            }
        }
        Ok(())
    }

    #[cfg(has_flashinfer_gdn_sm90_kernel)]
    #[derive(Clone, Copy)]
    enum FusedPrefillStateSource {
        Gathered,
        Pooled,
    }

    #[cfg(has_flashinfer_gdn_sm90_kernel)]
    #[derive(Clone, Copy)]
    struct FusedPrefillCase {
        batch_size: usize,
        seq_len: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        tiled_v_heads: bool,
        state_source: FusedPrefillStateSource,
    }

    #[cfg(has_flashinfer_gdn_sm90_kernel)]
    fn run_fused_prefill_case(dev: &Device, case: FusedPrefillCase) -> Result<()> {
        const HEAD_DIM: usize = 128;
        const POOLED_PADDING_BATCH: usize = 1;

        let key_dim = case.num_k_heads * HEAD_DIM;
        let value_dim = case.num_v_heads * HEAD_DIM;
        let conv_dim = 2 * key_dim + value_dim;
        let mixed_qkv = tensor3(
            patterned(case.batch_size * case.seq_len * conv_dim, 140, 0.08, 0.01),
            (case.batch_size, case.seq_len, conv_dim),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let b = tensor3(
            patterned(
                case.batch_size * case.seq_len * case.num_v_heads,
                141,
                0.2,
                0.1,
            ),
            (case.batch_size, case.seq_len, case.num_v_heads),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let a = tensor3(
            patterned(
                case.batch_size * case.seq_len * case.num_v_heads,
                142,
                0.18,
                -0.04,
            ),
            (case.batch_size, case.seq_len, case.num_v_heads),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let a_log = Tensor::from_vec(
            patterned(case.num_v_heads, 143, 0.05, -0.2),
            (case.num_v_heads,),
            dev,
        )?;
        let dt_bias = Tensor::from_vec(
            patterned(case.num_v_heads, 144, 0.1, 0.3),
            (case.num_v_heads,),
            dev,
        )?;
        let (state_rows, slots_tensor, active_rows) = match case.state_source {
            FusedPrefillStateSource::Gathered => {
                (case.batch_size, None, (0..case.batch_size).collect())
            }
            FusedPrefillStateSource::Pooled => {
                assert!(case.batch_size >= 3);
                let capacity = case.batch_size + 2;
                let mut slots = (0..case.batch_size)
                    .map(|batch| (capacity - 1 - batch) as u32)
                    .collect::<Vec<_>>();
                slots[POOLED_PADDING_BATCH] = GDN_PAD_SLOT;
                let active_rows = slots
                    .iter()
                    .filter(|&&slot| slot != GDN_PAD_SLOT)
                    .map(|&slot| slot as usize)
                    .collect::<Vec<_>>();
                (
                    capacity,
                    Some(Tensor::from_vec(slots, (case.batch_size,), dev)?),
                    active_rows,
                )
            }
        };
        let initial = Tensor::from_vec(
            patterned(
                state_rows * case.num_v_heads * HEAD_DIM * HEAD_DIM,
                145,
                0.01,
                0.0,
            ),
            (state_rows, case.num_v_heads, HEAD_DIM, HEAD_DIM),
            dev,
        )?
        .to_dtype(DType::F32)?;
        let initial = match case.state_source {
            FusedPrefillStateSource::Gathered => {
                initial.reshape((case.batch_size * case.num_v_heads, HEAD_DIM, HEAD_DIM))?
            }
            FusedPrefillStateSource::Pooled => initial,
        };
        let initial_host = flat(&initial.to_dtype(DType::F32)?)?;
        let slots = GdnStateSlots::from_option(slots_tensor.as_ref());
        let (q, k, v, g, beta) = prepare_recurrence_inputs_cuda(
            &mixed_qkv,
            &b,
            &a,
            &a_log,
            &dt_bias,
            case.batch_size,
            case.seq_len,
            case.num_k_heads,
            case.num_v_heads,
            HEAD_DIM,
            HEAD_DIM,
            case.tiled_v_heads,
        )?;
        let mut reference_state = initial.copy()?;
        let reference = launch_recurrence(
            RecurrenceKernel::ValueMajorWarp,
            RecurrenceInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            },
            &mut reference_state,
            slots,
        )?;
        let mut fused_state = initial.copy()?;
        let fused_output = flashinfer_sm90_prefill_dispatch(FusedPrefillRecurrence {
            mixed_qkv: &mixed_qkv,
            b: &b,
            a: &a,
            a_log: &a_log,
            dt_bias: &dt_bias,
            state: &mut fused_state,
            batch_size: case.batch_size,
            num_k_heads: case.num_k_heads,
            num_v_heads: case.num_v_heads,
            head_k_dim: HEAD_DIM,
            head_v_dim: HEAD_DIM,
            tiled_v_heads: case.tiled_v_heads,
            state_layout: RecurrentStateLayout::GdnValueMajor,
            slots,
        })?;
        let fused_output = match fused_output {
            FusedPrefillOutput::TokenMajor(output) => output
                .transpose(1, 2)?
                .contiguous()?
                .reshape((case.batch_size * case.num_v_heads, case.seq_len, HEAD_DIM))?,
        };

        let source = match case.state_source {
            FusedPrefillStateSource::Gathered => "gathered",
            FusedPrefillStateSource::Pooled => "pooled",
        };
        let label = format!(
            "FlashInferSm90 B={} S={} HK={} HV={} tiled={} {:?} {source}",
            case.batch_size,
            case.seq_len,
            case.num_k_heads,
            case.num_v_heads,
            case.tiled_v_heads,
            DType::F32,
        );
        assert_close(
            &format!("{label} output"),
            &flat(&fused_output.to_dtype(DType::F32)?)?,
            &flat(&reference)?,
            2.0e-2,
        );
        let fused_state_host = flat(&fused_state.to_dtype(DType::F32)?)?;
        let reference_state_host = flat(&reference_state.to_dtype(DType::F32)?)?;
        assert_close(
            &format!("{label} state"),
            &fused_state_host,
            &reference_state_host,
            1.0e-2,
        );

        let row_span = case.num_v_heads * HEAD_DIM * HEAD_DIM;
        for &row in &active_rows {
            for key_tile in 0..HEAD_DIM / 16 {
                let mut tile_changed = false;
                for head in 0..case.num_v_heads {
                    for value in 0..HEAD_DIM {
                        let start = row * row_span
                            + head * HEAD_DIM * HEAD_DIM
                            + value * HEAD_DIM
                            + key_tile * 16;
                        if reference_state_host[start..start + 16]
                            .iter()
                            .zip(&initial_host[start..start + 16])
                            .any(|(&updated, &initial)| (updated - initial).abs() > 1.0e-6)
                        {
                            tile_changed = true;
                            break;
                        }
                    }
                    if tile_changed {
                        break;
                    }
                }
                assert!(
                    tile_changed,
                    "{label} state K tile {key_tile} was not exercised"
                );
            }
        }

        if matches!(case.state_source, FusedPrefillStateSource::Pooled) {
            assert_zero(
                &format!("{label} padding output"),
                &fused_output.narrow(
                    0,
                    POOLED_PADDING_BATCH * case.num_v_heads,
                    case.num_v_heads,
                )?,
            )?;
            for row in 0..state_rows {
                if active_rows.contains(&row) {
                    continue;
                }
                assert_close(
                    &format!("{label} untouched row {row}"),
                    &fused_state_host[row * row_span..(row + 1) * row_span],
                    &initial_host[row * row_span..(row + 1) * row_span],
                    0.0,
                );
            }
        }
        Ok(())
    }

    #[cfg(has_flashinfer_gdn_sm90_kernel)]
    #[test]
    #[ignore = "requires an SM90 CUDA device"]
    fn flashinfer_sm90_prefill_matches_sequential_recurrence() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for case in [
            FusedPrefillCase {
                batch_size: 3,
                seq_len: 65,
                num_k_heads: 16,
                num_v_heads: 48,
                tiled_v_heads: false,
                state_source: FusedPrefillStateSource::Pooled,
            },
            FusedPrefillCase {
                batch_size: 1,
                seq_len: 129,
                num_k_heads: 16,
                num_v_heads: 48,
                tiled_v_heads: false,
                state_source: FusedPrefillStateSource::Gathered,
            },
        ] {
            run_fused_prefill_case(&dev, case)?;
        }
        Ok(())
    }

    struct ValueMajorDecodeCase {
        batch_size: usize,
        kernel: GdnDecodeKernel,
    }

    fn run_value_major_decode_case(dev: &Device, case: ValueMajorDecodeCase) -> Result<()> {
        const NUM_K_HEADS: usize = 16;
        const NUM_V_HEADS: usize = 48;
        const HEAD_DIM: usize = 128;
        const STEPS: usize = 8;

        let key_dim = NUM_K_HEADS * HEAD_DIM;
        let value_dim = NUM_V_HEADS * HEAD_DIM;
        let conv_dim = 2 * key_dim + value_dim;
        let mixed_qkv = tensor3(
            patterned(case.batch_size * conv_dim, 60, 0.08, 0.01),
            (case.batch_size, 1, conv_dim),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let b = tensor3(
            patterned(case.batch_size * NUM_V_HEADS, 61, 0.2, 0.1),
            (case.batch_size, 1, NUM_V_HEADS),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let a = tensor3(
            patterned(case.batch_size * NUM_V_HEADS, 62, 0.18, -0.04),
            (case.batch_size, 1, NUM_V_HEADS),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let a_log = Tensor::from_vec(patterned(NUM_V_HEADS, 63, 0.05, -0.2), (NUM_V_HEADS,), dev)?;
        let dt_bias = Tensor::from_vec(patterned(NUM_V_HEADS, 64, 0.1, 0.3), (NUM_V_HEADS,), dev)?;
        let capacity = case.batch_size + 3;
        let initial_state = Tensor::from_vec(
            patterned(capacity * NUM_V_HEADS * HEAD_DIM * HEAD_DIM, 65, 0.02, 0.0),
            (capacity, NUM_V_HEADS, HEAD_DIM, HEAD_DIM),
            dev,
        )?;
        let mut key_major_state = initial_state.clone();
        let mut value_major_state = initial_state.transpose(2, 3)?.contiguous()?;
        let slot_indices = Tensor::from_vec(
            (0..case.batch_size)
                .map(|idx| (capacity - 1 - idx) as u32)
                .collect::<Vec<_>>(),
            (case.batch_size,),
            dev,
        )?;
        let slots = GdnStateSlots::Pooled(&slot_indices);
        let (q, k, v, g, beta) = prepare_recurrence_inputs_cuda(
            &mixed_qkv,
            &b,
            &a,
            &a_log,
            &dt_bias,
            case.batch_size,
            1,
            NUM_K_HEADS,
            NUM_V_HEADS,
            HEAD_DIM,
            HEAD_DIM,
            false,
        )?;
        let reference_inputs = RecurrenceInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
        };

        for step in 1..=STEPS {
            let value_major = fused_decode_recurrence_cuda_impl(GdnDecodeLaunch {
                mixed_qkv: &mixed_qkv,
                b: &b,
                a: &a,
                a_log: &a_log,
                dt_bias: &dt_bias,
                state: &mut value_major_state,
                batch_size: case.batch_size,
                num_k_heads: NUM_K_HEADS,
                num_v_heads: NUM_V_HEADS,
                head_k_dim: HEAD_DIM,
                head_v_dim: HEAD_DIM,
                tiled_v_heads: false,
                state_layout: RecurrentStateLayout::GdnValueMajor,
                slots,
                requested_kernel: Some(case.kernel),
            })?;
            let reference =
                gated_delta_rule_recurrence_cuda(reference_inputs, &mut key_major_state, slots)?;
            assert_close(
                &format!(
                    "production value-major output B{} step {step}",
                    case.batch_size
                ),
                &flat(&value_major.to_dtype(DType::F32)?)?,
                &flat(&reference)?,
                2.0e-4,
            );
            assert_close(
                &format!(
                    "production value-major state B{} step {step}",
                    case.batch_size
                ),
                &flat(&value_major_state.transpose(2, 3)?.contiguous()?)?,
                &flat(&key_major_state)?,
                2.0e-5,
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an SM90 CUDA device"]
    fn sm90_value_major_decode_repeats_with_shuffled_slots() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for case in [
            ValueMajorDecodeCase {
                batch_size: 1,
                kernel: GdnDecodeKernel::ValueMajor4,
            },
            ValueMajorDecodeCase {
                batch_size: 8,
                kernel: GdnDecodeKernel::ValueMajor32,
            },
            ValueMajorDecodeCase {
                batch_size: 16,
                kernel: GdnDecodeKernel::ValueMajor32,
            },
        ] {
            run_value_major_decode_case(&dev, case)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_fused_decode_state_case(
        dev: &Device,
        batch_size: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        tiled_v_heads: bool,
        dtype: DType,
        state_dtype: DType,
        pooled: bool,
        strided_gates: bool,
        requested_kernel: Option<GdnDecodeKernel>,
    ) -> Result<()> {
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let mixed_qkv = tensor3(
            patterned(batch_size * conv_dim, 40, 0.08, 0.01),
            (batch_size, 1, conv_dim),
            dev,
        )?
        .to_dtype(dtype)?;
        let b_reference = tensor3(
            patterned(batch_size * num_v_heads, 41, 0.2, 0.1),
            (batch_size, 1, num_v_heads),
            dev,
        )?
        .to_dtype(dtype)?;
        let a_reference = tensor3(
            patterned(batch_size * num_v_heads, 42, 0.18, -0.04),
            (batch_size, 1, num_v_heads),
            dev,
        )?
        .to_dtype(dtype)?;
        let packed_gates = strided_gates
            .then(|| Tensor::cat(&[&b_reference, &a_reference], D::Minus1))
            .transpose()?;
        let (b, a) = if let Some(packed_gates) = packed_gates {
            assert!(!packed_gates
                .narrow(D::Minus1, 0, num_v_heads)?
                .is_contiguous());
            (
                packed_gates.narrow(D::Minus1, 0, num_v_heads)?,
                packed_gates.narrow(D::Minus1, num_v_heads, num_v_heads)?,
            )
        } else {
            (b_reference.clone(), a_reference.clone())
        };
        let a_log = Tensor::from_vec(patterned(num_v_heads, 43, 0.05, -0.2), (num_v_heads,), dev)?;
        let dt_bias = Tensor::from_vec(patterned(num_v_heads, 44, 0.1, 0.3), (num_v_heads,), dev)?;

        let capacity = if pooled { batch_size + 2 } else { batch_size };
        let state = Tensor::from_vec(
            patterned(
                capacity * num_v_heads * head_k_dim * head_v_dim,
                45,
                0.02,
                0.0,
            ),
            (capacity, num_v_heads, head_k_dim, head_v_dim),
            dev,
        )?
        .to_dtype(state_dtype)?;
        let slots = if pooled {
            Some(Tensor::from_vec(
                (0..batch_size)
                    .map(|idx| (capacity - 1 - idx) as u32)
                    .collect::<Vec<_>>(),
                (batch_size,),
                dev,
            )?)
        } else {
            None
        };
        let state_slots = GdnStateSlots::from_option(slots.as_ref());
        let mut fused_state = state.copy()?;
        let mut reference_state = state.copy()?;

        let fused = fused_decode_recurrence_cuda_impl(GdnDecodeLaunch {
            mixed_qkv: &mixed_qkv,
            b: &b,
            a: &a,
            a_log: &a_log,
            dt_bias: &dt_bias,
            state: &mut fused_state,
            batch_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads,
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: state_slots,
            requested_kernel,
        })?;
        let (q, k, v, g, beta) = prepare_recurrence_inputs_cuda(
            &mixed_qkv,
            &b_reference,
            &a_reference,
            &a_log,
            &dt_bias,
            batch_size,
            1,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads,
        )?;
        let reference = gated_delta_rule_recurrence_cuda(
            RecurrenceInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            },
            &mut reference_state,
            state_slots,
        )?;

        let output_tolerance = if dtype == DType::BF16 { 8.0e-3 } else { 1.0e-3 };
        assert_close(
            "fused decode output",
            &flat(&fused.to_dtype(DType::F32)?)?,
            &flat(&reference)?,
            output_tolerance,
        );
        assert_close(
            "fused decode state",
            &flat(&fused_state.to_dtype(DType::F32)?)?,
            &flat(&reference_state.to_dtype(DType::F32)?)?,
            2.0e-3,
        );

        let fused_second = fused_decode_recurrence_cuda_impl(GdnDecodeLaunch {
            mixed_qkv: &mixed_qkv,
            b: &b,
            a: &a,
            a_log: &a_log,
            dt_bias: &dt_bias,
            state: &mut fused_state,
            batch_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads,
            state_layout: RecurrentStateLayout::GdnKeyMajor,
            slots: state_slots,
            requested_kernel,
        })?;
        let reference_second = gated_delta_rule_recurrence_cuda(
            RecurrenceInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            },
            &mut reference_state,
            state_slots,
        )?;
        assert_close(
            "fused decode second output",
            &flat(&fused_second.to_dtype(DType::F32)?)?,
            &flat(&reference_second)?,
            output_tolerance,
        );
        assert_close(
            "fused decode second state",
            &flat(&fused_state.to_dtype(DType::F32)?)?,
            &flat(&reference_state.to_dtype(DType::F32)?)?,
            2.0e-3,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_fused_decode_case(
        dev: &Device,
        batch_size: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        tiled_v_heads: bool,
        dtype: DType,
        pooled: bool,
        strided_gates: bool,
        requested_kernel: Option<GdnDecodeKernel>,
    ) -> Result<()> {
        run_fused_decode_state_case(
            dev,
            batch_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads,
            dtype,
            DType::F32,
            pooled,
            strided_gates,
            requested_kernel,
        )
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn fused_decode_recurrence_matches_decomposed_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for state_dtype in [DType::BF16, DType::F16] {
            run_fused_decode_state_case(
                &dev,
                3,
                2,
                4,
                128,
                128,
                true,
                DType::BF16,
                state_dtype,
                true,
                true,
                Some(GdnDecodeKernel::Baseline),
            )?;
        }
        run_fused_decode_case(
            &dev,
            1,
            2,
            4,
            128,
            128,
            false,
            DType::F16,
            false,
            false,
            None,
        )?;
        run_fused_decode_case(&dev, 2, 2, 4, 64, 64, false, DType::F16, false, false, None)?;
        run_fused_decode_case(&dev, 2, 2, 6, 128, 128, true, DType::BF16, true, true, None)?;
        run_fused_decode_case(
            &dev,
            8,
            4,
            8,
            128,
            128,
            false,
            DType::BF16,
            false,
            true,
            None,
        )?;
        run_fused_decode_case(&dev, 3, 2, 4, 128, 128, true, DType::BF16, true, true, None)?;
        run_fused_decode_case(
            &dev,
            4,
            2,
            4,
            128,
            128,
            false,
            DType::BF16,
            false,
            false,
            None,
        )
    }

    fn run_speculative_state_commit_case(
        dev: &Device,
        state_layout: RecurrentStateLayout,
        state_dtype: DType,
    ) -> Result<()> {
        let batch_size = 3;
        let seq_len = 4;
        let num_k_heads = 1;
        let num_v_heads = 2;
        let head_k_dim = 128;
        let head_v_dim = 128;
        let kernel_size = 4;
        let capacity = 5;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let conv_dim = 2 * key_dim + value_dim;
        let row_state_elements = num_v_heads * head_k_dim * head_v_dim;
        let row_conv_elements = conv_dim * kernel_size;
        let keep_rows_host = vec![1u32, 3, 0];
        let slots_host = vec![4u32, 1, 3];

        let mixed_qkv = tensor3(
            patterned(batch_size * seq_len * conv_dim, 110, 0.08, 0.01),
            (batch_size, seq_len, conv_dim),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let conv_weight = tensor2(
            patterned(conv_dim * kernel_size, 111, 0.05, -0.01),
            (conv_dim, kernel_size),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let initial_conv_state = tensor3(
            patterned(batch_size * row_conv_elements, 112, 0.03, 0.0),
            (batch_size, conv_dim, kernel_size),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let (convolved_qkv, _) = causal_conv1d_cuda(
            &mixed_qkv,
            &conv_weight,
            &initial_conv_state,
            kernel_size,
            false,
            GdnStateSlots::Gathered,
        )?;
        let b = tensor3(
            patterned(batch_size * seq_len * num_v_heads, 113, 0.2, 0.1),
            (batch_size, seq_len, num_v_heads),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let a = tensor3(
            patterned(batch_size * seq_len * num_v_heads, 114, 0.18, -0.04),
            (batch_size, seq_len, num_v_heads),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let a_log = Tensor::from_vec(patterned(num_v_heads, 115, 0.05, -0.2), (num_v_heads,), dev)?;
        let dt_bias = Tensor::from_vec(patterned(num_v_heads, 116, 0.1, 0.3), (num_v_heads,), dev)?;
        let state_shape = match state_layout {
            RecurrentStateLayout::GdnKeyMajor => {
                vec![batch_size, num_v_heads, head_k_dim, head_v_dim]
            }
            RecurrentStateLayout::GdnValueMajor => {
                vec![batch_size, num_v_heads, head_v_dim, head_k_dim]
            }
            RecurrentStateLayout::Opaque => unreachable!(),
        };
        let pool_shape = match state_layout {
            RecurrentStateLayout::GdnKeyMajor => {
                vec![capacity, num_v_heads, head_k_dim, head_v_dim]
            }
            RecurrentStateLayout::GdnValueMajor => {
                vec![capacity, num_v_heads, head_v_dim, head_k_dim]
            }
            RecurrentStateLayout::Opaque => unreachable!(),
        };
        let initial_recurrent_state = Tensor::from_vec(
            patterned(batch_size * row_state_elements, 117, 0.02, 0.0),
            state_shape,
            dev,
        )?
        .to_dtype(state_dtype)?;
        let conv_state_pool = Tensor::from_vec(
            patterned(capacity * row_conv_elements, 118, 0.04, 0.0),
            (capacity, conv_dim, kernel_size),
            dev,
        )?
        .to_dtype(DType::BF16)?;
        let recurrent_state_pool = Tensor::from_vec(
            patterned(capacity * row_state_elements, 119, 0.02, 0.0),
            pool_shape,
            dev,
        )?
        .to_dtype(state_dtype)?;
        let mut expected_conv = flat(&conv_state_pool.to_dtype(DType::F32)?)?;
        let mut expected_recurrent = flat(&recurrent_state_pool.to_dtype(DType::F32)?)?;

        for batch_idx in 0..batch_size {
            let rows = keep_rows_host[batch_idx] as usize;
            if rows == 0 {
                continue;
            }
            let mixed_row = mixed_qkv.narrow(0, batch_idx, 1)?.narrow(1, 0, rows)?;
            let initial_conv_row = initial_conv_state.narrow(0, batch_idx, 1)?;
            let (_, conv_state) = causal_conv1d_cuda(
                &mixed_row,
                &conv_weight,
                &initial_conv_row,
                kernel_size,
                false,
                GdnStateSlots::Gathered,
            )?;
            let conv_state = flat(&conv_state.to_dtype(DType::F32)?)?;
            let conv_destination = slots_host[batch_idx] as usize * row_conv_elements;
            expected_conv[conv_destination..conv_destination + row_conv_elements]
                .copy_from_slice(&conv_state);

            let convolved_row = convolved_qkv.narrow(0, batch_idx, 1)?.narrow(1, 0, rows)?;
            let b_row = b.narrow(0, batch_idx, 1)?.narrow(1, 0, rows)?;
            let a_row = a.narrow(0, batch_idx, 1)?.narrow(1, 0, rows)?;
            let (q, k, v, g, beta) = prepare_recurrence_inputs_cuda(
                &convolved_row,
                &b_row,
                &a_row,
                &a_log,
                &dt_bias,
                1,
                rows,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                false,
            )?;
            let mut state = initial_recurrent_state.narrow(0, batch_idx, 1)?.copy()?;
            let inputs = RecurrenceInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            };
            if state_layout == RecurrentStateLayout::GdnValueMajor {
                vmajor_warp_gated_delta_rule_recurrence_cuda(
                    inputs,
                    &mut state,
                    GdnStateSlots::Gathered,
                )?;
            } else {
                gated_delta_rule_recurrence_cuda(inputs, &mut state, GdnStateSlots::Gathered)?;
            }
            let state = flat(&state.to_dtype(DType::F32)?)?;
            let state_destination = slots_host[batch_idx] as usize * row_state_elements;
            expected_recurrent[state_destination..state_destination + row_state_elements]
                .copy_from_slice(&state);
        }

        speculative_state_commit_cuda(GdnSpeculativeStateCommit {
            mixed_qkv: &mixed_qkv,
            convolved_qkv: &convolved_qkv,
            b: &b,
            a: &a,
            initial_conv_state: &initial_conv_state,
            initial_recurrent_state: &initial_recurrent_state,
            a_log: &a_log,
            dt_bias: &dt_bias,
            conv_state_pool: &conv_state_pool,
            recurrent_state_pool: &recurrent_state_pool,
            keep_rows: &Tensor::from_vec(keep_rows_host, (batch_size,), dev)?,
            slot_indices: &Tensor::from_vec(slots_host.clone(), (batch_size,), dev)?,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads: false,
            state_layout,
        })?;
        assert_close(
            "speculative conv state",
            &flat(&conv_state_pool.to_dtype(DType::F32)?)?,
            &expected_conv,
            0.0,
        );
        assert_close(
            "speculative recurrent state",
            &flat(&recurrent_state_pool.to_dtype(DType::F32)?)?,
            &expected_recurrent,
            2.0e-4,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn speculative_state_commit_matches_prefix_replay_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for state_dtype in [DType::F32, DType::BF16, DType::F16] {
            run_speculative_state_commit_case(
                &dev,
                RecurrentStateLayout::GdnKeyMajor,
                state_dtype,
            )?;
            run_speculative_state_commit_case(
                &dev,
                RecurrentStateLayout::GdnValueMajor,
                state_dtype,
            )?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an SM90 CUDA device"]
    fn speculative_checkpoint_kernels_match_serial_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let batch_size = 3;
        let seq_len = 8;
        let checkpoint_lanes = 8;
        let capacity = batch_size * checkpoint_lanes;
        let active_slots_host = vec![7u32, 14, GDN_PAD_SLOT];
        let active_slots = Tensor::from_vec(active_slots_host.clone(), (batch_size,), &dev)?;

        let conv_dim = 37;
        let kernel_size = 4;
        let x = tensor3(
            patterned(batch_size * seq_len * conv_dim, 130, 0.08, 0.01),
            (batch_size, seq_len, conv_dim),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let weight = tensor2(
            patterned(conv_dim * kernel_size, 131, 0.05, -0.01),
            (conv_dim, kernel_size),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let conv_pool = tensor3(
            patterned(capacity * conv_dim * kernel_size, 132, 0.03, 0.0),
            (capacity, conv_dim, kernel_size),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let mut expected_conv_pool = flat(&conv_pool.to_dtype(DType::F32)?)?;
        let mut expected_conv_outputs = Vec::with_capacity(batch_size);
        for (batch_idx, &active_slot) in active_slots_host.iter().enumerate() {
            if active_slot == GDN_PAD_SLOT {
                expected_conv_outputs.push(Tensor::zeros(
                    (1, seq_len, conv_dim),
                    DType::BF16,
                    &dev,
                )?);
                continue;
            }
            let input = x.narrow(0, batch_idx, 1)?;
            let initial = conv_pool.narrow(0, active_slot as usize, 1)?.copy()?;
            let (output, _) = causal_conv1d_cuda(
                &input,
                &weight,
                &initial,
                kernel_size,
                false,
                GdnStateSlots::Gathered,
            )?;
            expected_conv_outputs.push(output);
            let base_slot = active_slot as usize / checkpoint_lanes * checkpoint_lanes;
            for position in 0..seq_len {
                let (_, state) = causal_conv1d_cuda(
                    &input.narrow(1, 0, position + 1)?,
                    &weight,
                    &initial,
                    kernel_size,
                    false,
                    GdnStateSlots::Gathered,
                )?;
                let state = flat(&state.to_dtype(DType::F32)?)?;
                let destination = (base_slot + position) * conv_dim * kernel_size;
                expected_conv_pool[destination..destination + state.len()].copy_from_slice(&state);
            }
        }
        let expected_conv_output = Tensor::cat(&expected_conv_outputs, 0)?;
        let actual_conv_output =
            speculative_conv_checkpoints_cuda(GdnSpeculativeConvCheckpoints {
                x: &x,
                weight: &weight,
                state_pool: &conv_pool,
                active_slots: &active_slots,
                checkpoint_lanes,
                write_checkpoints: true,
                pending: None,
            })?;
        assert_close(
            "speculative checkpoint convolution output",
            &flat(&actual_conv_output.to_dtype(DType::F32)?)?,
            &flat(&expected_conv_output.to_dtype(DType::F32)?)?,
            2.0e-3,
        );
        assert_close(
            "speculative checkpoint convolution state",
            &flat(&conv_pool.to_dtype(DType::F32)?)?,
            &expected_conv_pool,
            0.0,
        );

        let num_k_heads = 2;
        let num_v_heads = 4;
        let head_k_dim = 128;
        let head_v_dim = 128;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let recurrent_conv_dim = 2 * key_dim + value_dim;
        let mixed_qkv = tensor3(
            patterned(batch_size * seq_len * recurrent_conv_dim, 133, 0.08, 0.01),
            (batch_size, seq_len, recurrent_conv_dim),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let b_storage_width = num_v_heads + 3;
        let b = tensor3(
            patterned(batch_size * seq_len * b_storage_width, 134, 0.2, 0.1),
            (batch_size, seq_len, b_storage_width),
            &dev,
        )?
        .to_dtype(DType::BF16)?
        .narrow(2, 2, num_v_heads)?;
        let a_storage_width = num_v_heads + 5;
        let a = tensor3(
            patterned(batch_size * seq_len * a_storage_width, 135, 0.18, -0.04),
            (batch_size, seq_len, a_storage_width),
            &dev,
        )?
        .to_dtype(DType::BF16)?
        .narrow(2, 3, num_v_heads)?;
        let a_log = Tensor::from_vec(
            patterned(num_v_heads, 136, 0.05, -0.2),
            (num_v_heads,),
            &dev,
        )?;
        let dt_bias =
            Tensor::from_vec(patterned(num_v_heads, 137, 0.1, 0.3), (num_v_heads,), &dev)?;
        let norm_eps = 1.0e-6;
        let norm_weight =
            Tensor::from_vec(patterned(head_v_dim, 139, 0.1, 1.0), (head_v_dim,), &dev)?
                .to_dtype(DType::BF16)?;
        let gate = Tensor::from_vec(
            patterned(
                batch_size * seq_len * num_v_heads * head_v_dim,
                140,
                0.2,
                0.0,
            ),
            (batch_size, seq_len, num_v_heads, head_v_dim),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let state_elements = num_v_heads * head_v_dim * head_k_dim;
        for state_layout in [
            RecurrentStateLayout::GdnKeyMajor,
            RecurrentStateLayout::GdnValueMajor,
        ] {
            for state_dtype in [DType::F32, DType::BF16, DType::F16] {
                let recurrent_pool = Tensor::from_vec(
                    patterned(capacity * state_elements, 138, 0.02, 0.0),
                    (capacity, num_v_heads, head_v_dim, head_k_dim),
                    &dev,
                )?
                .to_dtype(state_dtype)?;
                let mut expected_recurrent_pool = flat(&recurrent_pool.to_dtype(DType::F32)?)?;
                let mut expected_recurrent_outputs = Vec::with_capacity(batch_size);
                for (batch_idx, &active_slot) in active_slots_host.iter().enumerate() {
                    if active_slot == GDN_PAD_SLOT {
                        expected_recurrent_outputs.push(Tensor::zeros(
                            (num_v_heads, seq_len, head_v_dim),
                            DType::F32,
                            &dev,
                        )?);
                        continue;
                    }
                    let mixed_row = mixed_qkv.narrow(0, batch_idx, 1)?;
                    let b_row = b.narrow(0, batch_idx, 1)?;
                    let a_row = a.narrow(0, batch_idx, 1)?;
                    let (q, k, v, g, beta) = prepare_recurrence_inputs_cuda(
                        &mixed_row,
                        &b_row,
                        &a_row,
                        &a_log,
                        &dt_bias,
                        1,
                        seq_len,
                        num_k_heads,
                        num_v_heads,
                        head_k_dim,
                        head_v_dim,
                        true,
                    )?;
                    let initial = recurrent_pool.narrow(0, active_slot as usize, 1)?.copy()?;
                    let mut final_state = initial.copy()?;
                    let inputs = RecurrenceInputs {
                        q: &q,
                        k: &k,
                        v: &v,
                        g: &g,
                        beta: &beta,
                    };
                    let output = if state_layout == RecurrentStateLayout::GdnValueMajor {
                        vmajor_warp_gated_delta_rule_recurrence_cuda(
                            inputs,
                            &mut final_state,
                            GdnStateSlots::Gathered,
                        )?
                    } else {
                        gated_delta_rule_recurrence_cuda(
                            inputs,
                            &mut final_state,
                            GdnStateSlots::Gathered,
                        )?
                    };
                    expected_recurrent_outputs.push(output);

                    let base_slot = active_slot as usize / checkpoint_lanes * checkpoint_lanes;
                    for position in 0..seq_len {
                        let prefix_len = position + 1;
                        let q_prefix = q.narrow(1, 0, prefix_len)?.contiguous()?;
                        let k_prefix = k.narrow(1, 0, prefix_len)?.contiguous()?;
                        let v_prefix = v.narrow(1, 0, prefix_len)?.contiguous()?;
                        let g_prefix = g.narrow(1, 0, prefix_len)?.contiguous()?;
                        let beta_prefix = beta.narrow(1, 0, prefix_len)?.contiguous()?;
                        let mut checkpoint_state = initial.copy()?;
                        let prefix_inputs = RecurrenceInputs {
                            q: &q_prefix,
                            k: &k_prefix,
                            v: &v_prefix,
                            g: &g_prefix,
                            beta: &beta_prefix,
                        };
                        if state_layout == RecurrentStateLayout::GdnValueMajor {
                            vmajor_warp_gated_delta_rule_recurrence_cuda(
                                prefix_inputs,
                                &mut checkpoint_state,
                                GdnStateSlots::Gathered,
                            )?;
                        } else {
                            gated_delta_rule_recurrence_cuda(
                                prefix_inputs,
                                &mut checkpoint_state,
                                GdnStateSlots::Gathered,
                            )?;
                        }
                        let checkpoint_state = flat(&checkpoint_state.to_dtype(DType::F32)?)?;
                        let destination = (base_slot + position) * state_elements;
                        expected_recurrent_pool[destination..destination + checkpoint_state.len()]
                            .copy_from_slice(&checkpoint_state);
                    }
                }
                let expected_recurrent_output = Tensor::cat(&expected_recurrent_outputs, 0)?;
                let fused_pool = recurrent_pool.copy()?;
                let actual_recurrent_output =
                    speculative_recurrence_checkpoints_cuda(GdnSpeculativeRecurrenceCheckpoints {
                        mixed_qkv: &mixed_qkv,
                        b: &b,
                        a: &a,
                        a_log: &a_log,
                        dt_bias: &dt_bias,
                        state_pool: &recurrent_pool,
                        active_slots: &active_slots,
                        checkpoint_lanes,
                        num_k_heads,
                        num_v_heads,
                        head_k_dim,
                        head_v_dim,
                        tiled_v_heads: true,
                        state_layout,
                        post_op: None,
                        record_transitions: false,
                        pending: None,
                    })?;
                let actual_recurrent_output = actual_recurrent_output.output;
                let label = format!("{state_layout:?} {state_dtype:?}");
                assert_close(
                    &format!("{label} speculative checkpoint recurrence output"),
                    &flat(&actual_recurrent_output.to_dtype(DType::F32)?)?,
                    &flat(&expected_recurrent_output)?,
                    3.0e-3,
                );
                assert_close(
                    &format!("{label} speculative checkpoint recurrence state"),
                    &flat(&recurrent_pool.to_dtype(DType::F32)?)?,
                    &expected_recurrent_pool,
                    2.0e-3,
                );
                if state_layout == RecurrentStateLayout::GdnValueMajor {
                    let expected_normalized = expected_recurrent_output
                        .reshape((batch_size, num_v_heads, seq_len, head_v_dim))?
                        .transpose(1, 2)?
                        .to_dtype(DType::BF16)?;
                    let expected_normalized =
                        rmsnorm_gated_cuda(&expected_normalized, &gate, &norm_weight, norm_eps)?;
                    let actual_normalized = speculative_recurrence_checkpoints_cuda(
                        GdnSpeculativeRecurrenceCheckpoints {
                            mixed_qkv: &mixed_qkv,
                            b: &b,
                            a: &a,
                            a_log: &a_log,
                            dt_bias: &dt_bias,
                            state_pool: &fused_pool,
                            active_slots: &active_slots,
                            checkpoint_lanes,
                            num_k_heads,
                            num_v_heads,
                            head_k_dim,
                            head_v_dim,
                            tiled_v_heads: true,
                            state_layout,
                            post_op: Some(GdnSpeculativeRmsNormGate {
                                gate: &gate,
                                weight: &norm_weight,
                                eps: norm_eps,
                            }),
                            record_transitions: false,
                            pending: None,
                        },
                    )?;
                    let actual_normalized = actual_normalized.output;
                    assert_close(
                        &format!("{label} fused speculative normalization state"),
                        &flat(&fused_pool.to_dtype(DType::F32)?)?,
                        &expected_recurrent_pool,
                        2.0e-3,
                    );
                    assert_close(
                        &format!("{label} fused speculative normalization output"),
                        &flat(&actual_normalized.to_dtype(DType::F32)?)?,
                        &flat(&expected_normalized.to_dtype(DType::F32)?)?,
                        3.0e-2,
                    );
                }
            }
        }

        Ok(())
    }

    fn run_speculative_transition_commit_case(
        dev: &Device,
        seq_len: usize,
        activation_dtype: DType,
        state_dtype: DType,
        tiled_v_heads: bool,
    ) -> Result<()> {
        struct LayerCase {
            conv_input: Tensor,
            weight: Tensor,
            b: Tensor,
            a: Tensor,
            a_log: Tensor,
            dt_bias: Tensor,
            gate: Tensor,
            norm_weight: Tensor,
            transitions: GdnSpeculativeTransitions,
            conv_state: Tensor,
            recurrent_state: Tensor,
            pending_conv_input: Tensor,
            pending_key: Tensor,
            pending_delta: Tensor,
            pending_decay: Tensor,
            pending_keep_rows: Tensor,
            pending_epochs: Tensor,
            conv_applied_epochs: Tensor,
            recurrent_applied_epochs: Tensor,
            staged_conv_state: Tensor,
            staged_recurrent_state: Tensor,
            lazy_conv_applied_epochs: Tensor,
            lazy_recurrent_applied_epochs: Tensor,
            lazy_conv_state: Tensor,
            lazy_recurrent_state: Tensor,
            expected_conv_state: Tensor,
            expected_recurrent_state: Tensor,
        }

        let batch_size = seq_len + 2;
        let capacity = 2 * batch_size + 3;
        let num_k_heads = 2;
        let num_v_heads = 4;
        let head_k_dim = 128;
        let head_v_dim = 128;
        let conv_width = 4;
        let conv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim;
        let mut active_slots_host = (0..=seq_len)
            .map(|row| ((row * 2 + 1) % capacity) as u32)
            .collect::<Vec<_>>();
        active_slots_host.push(GDN_PAD_SLOT);
        let mut keep_rows_host = (1..=seq_len).map(|rows| rows as u32).collect::<Vec<_>>();
        keep_rows_host.push(0);
        keep_rows_host.push(0);
        let active_slots = Tensor::from_vec(active_slots_host.clone(), (batch_size,), dev)?;
        let keep_rows = Tensor::from_vec(keep_rows_host.clone(), (batch_size,), dev)?;
        let mut cases = Vec::new();

        for layer_idx in 0..2 {
            let seed = 200 + layer_idx * 20 + seq_len;
            let conv_input = tensor3(
                patterned(batch_size * seq_len * conv_dim, seed, 0.08, 0.01),
                (batch_size, seq_len, conv_dim),
                dev,
            )?
            .to_dtype(activation_dtype)?;
            let weight = tensor2(
                patterned(conv_dim * conv_width, seed + 1, 0.05, -0.01),
                (conv_dim, conv_width),
                dev,
            )?
            .to_dtype(activation_dtype)?;
            let conv_state = tensor3(
                patterned(capacity * conv_dim * conv_width, seed + 2, 0.03, 0.0),
                (capacity, conv_dim, conv_width),
                dev,
            )?
            .to_dtype(activation_dtype)?;
            let recurrent_state = Tensor::from_vec(
                patterned(
                    capacity * num_v_heads * head_v_dim * head_k_dim,
                    seed + 3,
                    0.02,
                    0.0,
                ),
                (capacity, num_v_heads, head_v_dim, head_k_dim),
                dev,
            )?
            .to_dtype(state_dtype)?;
            let initial_conv = flat(&conv_state.to_dtype(DType::F32)?)?;
            let initial_recurrent = flat(&recurrent_state.to_dtype(DType::F32)?)?;
            let convolved = speculative_conv_checkpoints_cuda(GdnSpeculativeConvCheckpoints {
                x: &conv_input,
                weight: &weight,
                state_pool: &conv_state,
                active_slots: &active_slots,
                checkpoint_lanes: 1,
                write_checkpoints: false,
                pending: None,
            })?;
            assert_close(
                "transition convolution leaves state untouched",
                &flat(&conv_state.to_dtype(DType::F32)?)?,
                &initial_conv,
                0.0,
            );

            let b = tensor3(
                patterned(batch_size * seq_len * num_v_heads, seed + 4, 0.2, 0.1),
                (batch_size, seq_len, num_v_heads),
                dev,
            )?
            .to_dtype(activation_dtype)?;
            let a = tensor3(
                patterned(batch_size * seq_len * num_v_heads, seed + 5, 0.18, -0.04),
                (batch_size, seq_len, num_v_heads),
                dev,
            )?
            .to_dtype(activation_dtype)?;
            let a_log = Tensor::from_vec(
                patterned(num_v_heads, seed + 6, 0.05, -0.2),
                (num_v_heads,),
                dev,
            )?;
            let dt_bias = Tensor::from_vec(
                patterned(num_v_heads, seed + 7, 0.1, 0.3),
                (num_v_heads,),
                dev,
            )?;
            let gate = Tensor::from_vec(
                patterned(
                    batch_size * seq_len * num_v_heads * head_v_dim,
                    seed + 8,
                    0.2,
                    0.0,
                ),
                (batch_size, seq_len, num_v_heads, head_v_dim),
                dev,
            )?
            .to_dtype(activation_dtype)?;
            let norm_weight = Tensor::from_vec(
                patterned(head_v_dim, seed + 9, 0.1, 1.0),
                (head_v_dim,),
                dev,
            )?
            .to_dtype(activation_dtype)?;
            let recurrence =
                speculative_recurrence_checkpoints_cuda(GdnSpeculativeRecurrenceCheckpoints {
                    mixed_qkv: &convolved,
                    b: &b,
                    a: &a,
                    a_log: &a_log,
                    dt_bias: &dt_bias,
                    state_pool: &recurrent_state,
                    active_slots: &active_slots,
                    checkpoint_lanes: 1,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    tiled_v_heads,
                    state_layout: RecurrentStateLayout::GdnValueMajor,
                    post_op: Some(GdnSpeculativeRmsNormGate {
                        gate: &gate,
                        weight: &norm_weight,
                        eps: 1.0e-6,
                    }),
                    record_transitions: true,
                    pending: None,
                })?;
            assert_close(
                "transition recurrence leaves state untouched",
                &flat(&recurrent_state.to_dtype(DType::F32)?)?,
                &initial_recurrent,
                0.0,
            );

            let expected_conv_state = conv_state.copy()?;
            let expected_recurrent_state = recurrent_state.copy()?;
            let initial_conv_rows = active_slots_host
                .iter()
                .map(|&slot| {
                    if slot == GDN_PAD_SLOT {
                        Tensor::zeros((1, conv_dim, conv_width), activation_dtype, dev)
                    } else {
                        conv_state.narrow(0, slot as usize, 1)?.copy()
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let initial_recurrent_rows = active_slots_host
                .iter()
                .map(|&slot| {
                    if slot == GDN_PAD_SLOT {
                        Tensor::zeros((1, num_v_heads, head_v_dim, head_k_dim), state_dtype, dev)
                    } else {
                        recurrent_state.narrow(0, slot as usize, 1)?.copy()
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            speculative_state_commit_cuda(GdnSpeculativeStateCommit {
                mixed_qkv: &conv_input,
                convolved_qkv: &convolved,
                b: &b,
                a: &a,
                initial_conv_state: &Tensor::cat(&initial_conv_rows, 0)?,
                initial_recurrent_state: &Tensor::cat(&initial_recurrent_rows, 0)?,
                a_log: &a_log,
                dt_bias: &dt_bias,
                conv_state_pool: &expected_conv_state,
                recurrent_state_pool: &expected_recurrent_state,
                keep_rows: &keep_rows,
                slot_indices: &active_slots,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                tiled_v_heads,
                state_layout: RecurrentStateLayout::GdnValueMajor,
            })?;
            let pending_conv_input =
                Tensor::zeros((capacity, seq_len, conv_dim), activation_dtype, dev)?;
            let pending_key = Tensor::zeros(
                (capacity, seq_len, num_k_heads, head_k_dim),
                DType::F32,
                dev,
            )?;
            let pending_delta = Tensor::zeros(
                (capacity, seq_len, num_v_heads, head_v_dim),
                DType::F32,
                dev,
            )?;
            let pending_decay = Tensor::zeros((capacity, seq_len, num_v_heads), DType::F32, dev)?;
            let pending_keep_rows = Tensor::zeros(capacity, DType::U32, dev)?;
            let pending_epochs = Tensor::zeros(capacity, DType::U32, dev)?;
            let conv_applied_epochs = Tensor::zeros(
                (capacity, conv_dim.div_ceil(GDN_CHANNEL_BLOCK_SIZE)),
                DType::U32,
                dev,
            )?;
            let recurrent_applied_epochs = Tensor::zeros((capacity, num_v_heads), DType::U32, dev)?;
            cases.push(LayerCase {
                conv_input,
                weight,
                b,
                a,
                a_log,
                dt_bias,
                gate,
                norm_weight,
                transitions: recurrence
                    .transitions
                    .expect("record_transitions produced no transition tensors"),
                staged_conv_state: conv_state.copy()?,
                staged_recurrent_state: recurrent_state.copy()?,
                lazy_conv_applied_epochs: conv_applied_epochs.copy()?,
                lazy_recurrent_applied_epochs: recurrent_applied_epochs.copy()?,
                lazy_conv_state: conv_state.copy()?,
                lazy_recurrent_state: recurrent_state.copy()?,
                conv_state,
                recurrent_state,
                pending_conv_input,
                pending_key,
                pending_delta,
                pending_decay,
                pending_keep_rows,
                pending_epochs,
                conv_applied_epochs,
                recurrent_applied_epochs,
                expected_conv_state,
                expected_recurrent_state,
            });
        }

        let layers = cases
            .iter()
            .map(|case| GdnSpeculativeTransitionLayer {
                conv_input: &case.conv_input,
                key: &case.transitions.key,
                delta: &case.transitions.delta,
                decay: &case.transitions.decay,
                conv_state: &case.conv_state,
                recurrent_state: &case.recurrent_state,
            })
            .collect::<Vec<_>>();
        speculative_transition_commit_batched_cuda(GdnSpeculativeTransitionCommit {
            layers: &layers,
            keep_rows: &keep_rows,
            active_slots: &active_slots,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            conv_width,
            tiled_v_heads,
            state_layout: RecurrentStateLayout::GdnValueMajor,
        })?;
        let stage_layers = cases
            .iter()
            .map(|case| GdnSpeculativeTransitionStageLayer {
                conv_input: &case.conv_input,
                key: &case.transitions.key,
                delta: &case.transitions.delta,
                decay: &case.transitions.decay,
                pending_conv_input: &case.pending_conv_input,
                pending_key: &case.pending_key,
                pending_delta: &case.pending_delta,
                pending_decay: &case.pending_decay,
                pending_keep_rows: &case.pending_keep_rows,
                pending_epochs: &case.pending_epochs,
            })
            .collect::<Vec<_>>();
        speculative_transition_stage_batched_cuda(GdnSpeculativeTransitionStage {
            layers: &stage_layers,
            keep_rows: &keep_rows,
            destination_slots: &active_slots,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
        })?;
        let apply_layers = cases
            .iter()
            .map(|case| GdnPendingTransitionApplyLayer {
                pending_conv_input: &case.pending_conv_input,
                pending_key: &case.pending_key,
                pending_delta: &case.pending_delta,
                pending_decay: &case.pending_decay,
                pending_keep_rows: &case.pending_keep_rows,
                pending_epochs: &case.pending_epochs,
                conv_applied_epochs: &case.conv_applied_epochs,
                recurrent_applied_epochs: &case.recurrent_applied_epochs,
                conv_state: &case.staged_conv_state,
                recurrent_state: &case.staged_recurrent_state,
            })
            .collect::<Vec<_>>();
        let apply = || {
            pending_transition_apply_batched_cuda(GdnPendingTransitionApply {
                layers: &apply_layers,
                active_slots: &active_slots,
                num_k_heads,
                num_v_heads,
                head_k_dim,
                head_v_dim,
                conv_dim,
                conv_width,
                tiled_v_heads,
                state_layout: RecurrentStateLayout::GdnValueMajor,
            })
        };
        apply()?;
        apply()?;
        for case in &cases {
            assert_close(
                "batched transition convolution state",
                &flat(&case.conv_state.to_dtype(DType::F32)?)?,
                &flat(&case.expected_conv_state.to_dtype(DType::F32)?)?,
                0.0,
            );
            assert_close(
                "batched transition recurrent state",
                &flat(&case.recurrent_state.to_dtype(DType::F32)?)?,
                &flat(&case.expected_recurrent_state.to_dtype(DType::F32)?)?,
                3.0e-3,
            );
            assert_close(
                "staged transition convolution state",
                &flat(&case.staged_conv_state.to_dtype(DType::F32)?)?,
                &flat(&case.expected_conv_state.to_dtype(DType::F32)?)?,
                0.0,
            );
            assert_close(
                "staged transition recurrent state",
                &flat(&case.staged_recurrent_state.to_dtype(DType::F32)?)?,
                &flat(&case.expected_recurrent_state.to_dtype(DType::F32)?)?,
                3.0e-3,
            );

            let expected_convolved =
                speculative_conv_checkpoints_cuda(GdnSpeculativeConvCheckpoints {
                    x: &case.conv_input,
                    weight: &case.weight,
                    state_pool: &case.expected_conv_state,
                    active_slots: &active_slots,
                    checkpoint_lanes: 1,
                    write_checkpoints: false,
                    pending: None,
                })?;
            let expected_output =
                speculative_recurrence_checkpoints_cuda(GdnSpeculativeRecurrenceCheckpoints {
                    mixed_qkv: &expected_convolved,
                    b: &case.b,
                    a: &case.a,
                    a_log: &case.a_log,
                    dt_bias: &case.dt_bias,
                    state_pool: &case.expected_recurrent_state,
                    active_slots: &active_slots,
                    checkpoint_lanes: 1,
                    num_k_heads,
                    num_v_heads,
                    head_k_dim,
                    head_v_dim,
                    tiled_v_heads,
                    state_layout: RecurrentStateLayout::GdnValueMajor,
                    post_op: Some(GdnSpeculativeRmsNormGate {
                        gate: &case.gate,
                        weight: &case.norm_weight,
                        eps: 1.0e-6,
                    }),
                    record_transitions: true,
                    pending: None,
                })?
                .output;
            let run_lazy = || {
                let convolved = speculative_conv_checkpoints_cuda(GdnSpeculativeConvCheckpoints {
                    x: &case.conv_input,
                    weight: &case.weight,
                    state_pool: &case.lazy_conv_state,
                    active_slots: &active_slots,
                    checkpoint_lanes: 1,
                    write_checkpoints: false,
                    pending: Some(GdnPendingSpeculativeConv {
                        conv_input: &case.pending_conv_input,
                        keep_rows: &case.pending_keep_rows,
                        epochs: &case.pending_epochs,
                        applied_epochs: &case.lazy_conv_applied_epochs,
                    }),
                })?;
                let output =
                    speculative_recurrence_checkpoints_cuda(GdnSpeculativeRecurrenceCheckpoints {
                        mixed_qkv: &convolved,
                        b: &case.b,
                        a: &case.a,
                        a_log: &case.a_log,
                        dt_bias: &case.dt_bias,
                        state_pool: &case.lazy_recurrent_state,
                        active_slots: &active_slots,
                        checkpoint_lanes: 1,
                        num_k_heads,
                        num_v_heads,
                        head_k_dim,
                        head_v_dim,
                        tiled_v_heads,
                        state_layout: RecurrentStateLayout::GdnValueMajor,
                        post_op: Some(GdnSpeculativeRmsNormGate {
                            gate: &case.gate,
                            weight: &case.norm_weight,
                            eps: 1.0e-6,
                        }),
                        record_transitions: true,
                        pending: Some(GdnPendingSpeculativeRecurrence {
                            key: &case.pending_key,
                            delta: &case.pending_delta,
                            decay: &case.pending_decay,
                            keep_rows: &case.pending_keep_rows,
                            epochs: &case.pending_epochs,
                            applied_epochs: &case.lazy_recurrent_applied_epochs,
                        }),
                    })?
                    .output;
                Ok::<_, candle_core::Error>((convolved, output))
            };
            let (lazy_convolved, lazy_output) = run_lazy()?;
            let (retried_convolved, retried_output) = run_lazy()?;
            assert_close(
                "lazy transition convolution state",
                &flat(&case.lazy_conv_state.to_dtype(DType::F32)?)?,
                &flat(&case.expected_conv_state.to_dtype(DType::F32)?)?,
                0.0,
            );
            assert_close(
                "lazy transition recurrent state",
                &flat(&case.lazy_recurrent_state.to_dtype(DType::F32)?)?,
                &flat(&case.expected_recurrent_state.to_dtype(DType::F32)?)?,
                3.0e-3,
            );
            for (label, actual, expected, tolerance) in [
                (
                    "lazy transition convolution output",
                    &lazy_convolved,
                    &expected_convolved,
                    0.0,
                ),
                (
                    "retried transition convolution output",
                    &retried_convolved,
                    &expected_convolved,
                    0.0,
                ),
                (
                    "lazy transition recurrence output",
                    &lazy_output,
                    &expected_output,
                    3.0e-2,
                ),
                (
                    "retried transition recurrence output",
                    &retried_output,
                    &expected_output,
                    3.0e-2,
                ),
            ] {
                assert_close(
                    label,
                    &flat(&actual.to_dtype(DType::F32)?)?,
                    &flat(&expected.to_dtype(DType::F32)?)?,
                    tolerance,
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an SM90 CUDA device"]
    fn speculative_transition_commit_matches_prefix_replay_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for seq_len in [4, 8] {
            for activation_dtype in [DType::F16, DType::BF16] {
                for state_dtype in [DType::F32, DType::BF16, DType::F16] {
                    for tiled_v_heads in [false, true] {
                        run_speculative_transition_commit_case(
                            &dev,
                            seq_len,
                            activation_dtype,
                            state_dtype,
                            tiled_v_heads,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    fn run_speculative_transition_stage_case(
        dev: &Device,
        seq_len: usize,
        activation_dtype: DType,
    ) -> Result<()> {
        struct LayerCase {
            conv_input: Tensor,
            key: Tensor,
            delta: Tensor,
            decay: Tensor,
            pending_conv_input: Tensor,
            pending_key: Tensor,
            pending_delta: Tensor,
            pending_decay: Tensor,
            pending_keep_rows: Tensor,
            pending_epochs: Tensor,
            expected_conv_input: Vec<f32>,
            expected_key: Vec<f32>,
            expected_delta: Vec<f32>,
            expected_decay: Vec<f32>,
            expected_keep_rows: Vec<u32>,
            expected_epochs: Vec<u32>,
        }

        fn stage_rows(
            destination: &mut [f32],
            source: &[f32],
            destination_slot: usize,
            batch_idx: usize,
            rows: usize,
            max_rows: usize,
            seq_len: usize,
            row_elements: usize,
        ) {
            let source_start = batch_idx * seq_len * row_elements;
            let destination_start = destination_slot * max_rows * row_elements;
            let elements = rows * row_elements;
            destination[destination_start..destination_start + elements]
                .copy_from_slice(&source[source_start..source_start + elements]);
        }

        let batch_size = 4;
        let max_rows = 8;
        let capacity = 7;
        let num_k_heads = 2;
        let num_v_heads = 4;
        let head_k_dim = 8;
        let head_v_dim = 6;
        let conv_dim = 37;
        let keep_rows_host = vec![1u32, seq_len as u32, 0, (seq_len / 2) as u32];
        let destination_slots_host = vec![3u32, 1, 5, GDN_PAD_SLOT];
        let keep_rows = Tensor::from_vec(keep_rows_host.clone(), (batch_size,), dev)?;
        let destination_slots =
            Tensor::from_vec(destination_slots_host.clone(), (batch_size,), dev)?;
        let mut cases = Vec::new();

        for layer_idx in 0..2 {
            let salt = 400 + layer_idx * 20 + seq_len;
            let conv_input = Tensor::from_vec(
                patterned(batch_size * seq_len * conv_dim, salt, 0.2, 0.01),
                (batch_size, seq_len, conv_dim),
                dev,
            )?
            .to_dtype(activation_dtype)?;
            let key = Tensor::from_vec(
                patterned(
                    batch_size * seq_len * num_k_heads * head_k_dim,
                    salt + 1,
                    0.3,
                    -0.02,
                ),
                (batch_size, seq_len, num_k_heads, head_k_dim),
                dev,
            )?;
            let delta = Tensor::from_vec(
                patterned(
                    batch_size * seq_len * num_v_heads * head_v_dim,
                    salt + 2,
                    0.25,
                    0.03,
                ),
                (batch_size, seq_len, num_v_heads, head_v_dim),
                dev,
            )?;
            let decay = Tensor::from_vec(
                patterned(batch_size * seq_len * num_v_heads, salt + 3, 0.1, 0.8),
                (batch_size, seq_len, num_v_heads),
                dev,
            )?;
            let pending_conv_input = Tensor::from_vec(
                patterned(capacity * max_rows * conv_dim, salt + 4, 0.05, -0.4),
                (capacity, max_rows, conv_dim),
                dev,
            )?
            .to_dtype(activation_dtype)?;
            let pending_key = Tensor::from_vec(
                patterned(
                    capacity * max_rows * num_k_heads * head_k_dim,
                    salt + 5,
                    0.05,
                    -0.5,
                ),
                (capacity, max_rows, num_k_heads, head_k_dim),
                dev,
            )?;
            let pending_delta = Tensor::from_vec(
                patterned(
                    capacity * max_rows * num_v_heads * head_v_dim,
                    salt + 6,
                    0.05,
                    -0.6,
                ),
                (capacity, max_rows, num_v_heads, head_v_dim),
                dev,
            )?;
            let pending_decay = Tensor::from_vec(
                patterned(capacity * max_rows * num_v_heads, salt + 7, 0.05, -0.7),
                (capacity, max_rows, num_v_heads),
                dev,
            )?;
            let pending_keep_rows = Tensor::from_vec(
                (0..capacity).map(|idx| 40 + idx as u32).collect::<Vec<_>>(),
                (capacity,),
                dev,
            )?;
            let pending_epochs = Tensor::from_vec(
                (0..capacity).map(|idx| 70 + idx as u32).collect::<Vec<_>>(),
                (capacity,),
                dev,
            )?;

            let source_conv = flat(&conv_input.to_dtype(DType::F32)?)?;
            let source_key = flat(&key)?;
            let source_delta = flat(&delta)?;
            let source_decay = flat(&decay)?;
            let mut expected_conv_input = flat(&pending_conv_input.to_dtype(DType::F32)?)?;
            let mut expected_key = flat(&pending_key)?;
            let mut expected_delta = flat(&pending_delta)?;
            let mut expected_decay = flat(&pending_decay)?;
            let mut expected_keep_rows: Vec<u32> =
                pending_keep_rows.to_device(&Device::Cpu)?.to_vec1()?;
            let mut expected_epochs: Vec<u32> =
                pending_epochs.to_device(&Device::Cpu)?.to_vec1()?;
            for batch_idx in 0..batch_size {
                let slot = destination_slots_host[batch_idx];
                if slot == GDN_PAD_SLOT {
                    continue;
                }
                let slot = slot as usize;
                let rows = keep_rows_host[batch_idx] as usize;
                if rows == 0 {
                    expected_keep_rows[slot] = 0;
                    expected_epochs[slot] = expected_epochs[slot].wrapping_add(1).max(1);
                    continue;
                }
                stage_rows(
                    &mut expected_conv_input,
                    &source_conv,
                    slot,
                    batch_idx,
                    rows,
                    max_rows,
                    seq_len,
                    conv_dim,
                );
                stage_rows(
                    &mut expected_key,
                    &source_key,
                    slot,
                    batch_idx,
                    rows,
                    max_rows,
                    seq_len,
                    num_k_heads * head_k_dim,
                );
                stage_rows(
                    &mut expected_delta,
                    &source_delta,
                    slot,
                    batch_idx,
                    rows,
                    max_rows,
                    seq_len,
                    num_v_heads * head_v_dim,
                );
                stage_rows(
                    &mut expected_decay,
                    &source_decay,
                    slot,
                    batch_idx,
                    rows,
                    max_rows,
                    seq_len,
                    num_v_heads,
                );
                expected_keep_rows[slot] = rows as u32;
                expected_epochs[slot] = expected_epochs[slot].wrapping_add(1).max(1);
            }
            cases.push(LayerCase {
                conv_input,
                key,
                delta,
                decay,
                pending_conv_input,
                pending_key,
                pending_delta,
                pending_decay,
                pending_keep_rows,
                pending_epochs,
                expected_conv_input,
                expected_key,
                expected_delta,
                expected_decay,
                expected_keep_rows,
                expected_epochs,
            });
        }

        let layers = cases
            .iter()
            .map(|case| GdnSpeculativeTransitionStageLayer {
                conv_input: &case.conv_input,
                key: &case.key,
                delta: &case.delta,
                decay: &case.decay,
                pending_conv_input: &case.pending_conv_input,
                pending_key: &case.pending_key,
                pending_delta: &case.pending_delta,
                pending_decay: &case.pending_decay,
                pending_keep_rows: &case.pending_keep_rows,
                pending_epochs: &case.pending_epochs,
            })
            .collect::<Vec<_>>();
        speculative_transition_stage_batched_cuda(GdnSpeculativeTransitionStage {
            layers: &layers,
            keep_rows: &keep_rows,
            destination_slots: &destination_slots,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
        })?;

        for case in cases {
            assert_close(
                "staged convolution input",
                &flat(&case.pending_conv_input.to_dtype(DType::F32)?)?,
                &case.expected_conv_input,
                0.0,
            );
            assert_close(
                "staged recurrence key",
                &flat(&case.pending_key)?,
                &case.expected_key,
                0.0,
            );
            assert_close(
                "staged recurrence delta",
                &flat(&case.pending_delta)?,
                &case.expected_delta,
                0.0,
            );
            assert_close(
                "staged recurrence decay",
                &flat(&case.pending_decay)?,
                &case.expected_decay,
                0.0,
            );
            assert_eq!(
                case.pending_keep_rows
                    .to_device(&Device::Cpu)?
                    .to_vec1::<u32>()?,
                case.expected_keep_rows
            );
            assert_eq!(
                case.pending_epochs
                    .to_device(&Device::Cpu)?
                    .to_vec1::<u32>()?,
                case.expected_epochs
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn speculative_transition_stage_is_slot_indexed_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for seq_len in [4, 8] {
            for activation_dtype in [DType::F16, DType::BF16] {
                run_speculative_transition_stage_case(&dev, seq_len, activation_dtype)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an SM90 CUDA device"]
    fn sm90_fused_decode_kernel_variants_match_decomposed_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        run_fused_decode_case(
            &dev,
            1,
            2,
            4,
            128,
            128,
            false,
            DType::F16,
            false,
            false,
            Some(GdnDecodeKernel::Cooperative),
        )?;
        run_fused_decode_case(
            &dev,
            2,
            2,
            6,
            128,
            128,
            true,
            DType::F16,
            true,
            true,
            Some(GdnDecodeKernel::Pipelined),
        )?;
        run_fused_decode_case(
            &dev,
            8,
            4,
            8,
            128,
            128,
            false,
            DType::BF16,
            false,
            true,
            Some(GdnDecodeKernel::Pipelined),
        )
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn causal_conv1d_width4_update_matches_full_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let batch_size = 3;
        let conv_dim = 257;
        let kernel_size = 4;
        let x = tensor3(
            patterned(batch_size * conv_dim, 50, 0.08, 0.01),
            (batch_size, 1, conv_dim),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let weight = tensor2(
            patterned(conv_dim * kernel_size, 51, 0.05, -0.01),
            (conv_dim, kernel_size),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let state = tensor3(
            patterned(batch_size * conv_dim * kernel_size, 52, 0.03, 0.0),
            (batch_size, conv_dim, kernel_size),
            &dev,
        )?
        .to_dtype(DType::BF16)?;

        let update_state_input = state.copy()?;
        let full_state_input = state.copy()?;
        let (update, update_state) = causal_conv1d_cuda(
            &x,
            &weight,
            &update_state_input,
            kernel_size,
            true,
            GdnStateSlots::Gathered,
        )?;
        let (full, full_state) = causal_conv1d_cuda(
            &x,
            &weight,
            &full_state_input,
            kernel_size,
            false,
            GdnStateSlots::Gathered,
        )?;
        assert_close(
            "width-4 conv output",
            &flat(&update.to_dtype(DType::F32)?)?,
            &flat(&full.to_dtype(DType::F32)?)?,
            0.0,
        );
        assert_close(
            "width-4 conv state",
            &flat(&update_state.to_dtype(DType::F32)?)?,
            &flat(&full_state.to_dtype(DType::F32)?)?,
            0.0,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn causal_conv1d_full_continuation_matches_one_shot_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let batch_size = 2;
        let conv_dim = 19;
        let seq_len = 7;
        let split = 3;
        let kernel_size = 4;
        let x = tensor3(
            patterned(batch_size * conv_dim * seq_len, 20, 0.08, 0.01),
            (batch_size, seq_len, conv_dim),
            &dev,
        )?
        .to_dtype(DType::F16)?;
        let weight = tensor2(
            patterned(conv_dim * kernel_size, 21, 0.05, -0.01),
            (conv_dim, kernel_size),
            &dev,
        )?
        .to_dtype(DType::F16)?;
        let initial_state = tensor3(
            patterned(batch_size * conv_dim * kernel_size, 22, 0.03, 0.0),
            (batch_size, conv_dim, kernel_size),
            &dev,
        )?
        .to_dtype(DType::F16)?;

        let (one_shot, one_shot_state) = causal_conv1d_cuda(
            &x,
            &weight,
            &initial_state,
            kernel_size,
            false,
            GdnStateSlots::Gathered,
        )?;
        let (first, first_state) = causal_conv1d_cuda(
            &x.narrow(1, 0, split)?,
            &weight,
            &initial_state,
            kernel_size,
            false,
            GdnStateSlots::Gathered,
        )?;
        let (second, chunked_state) = causal_conv1d_cuda(
            &x.narrow(1, split, seq_len - split)?,
            &weight,
            &first_state,
            kernel_size,
            false,
            GdnStateSlots::Gathered,
        )?;
        let chunked = Tensor::cat(&[first, second], 1)?;

        assert_close(
            "causal conv output",
            &flat(&one_shot.to_dtype(DType::F32)?)?,
            &flat(&chunked.to_dtype(DType::F32)?)?,
            2.0e-3,
        );
        assert_close(
            "causal conv state",
            &flat(&one_shot_state.to_dtype(DType::F32)?)?,
            &flat(&chunked_state.to_dtype(DType::F32)?)?,
            2.0e-3,
        );
        Ok(())
    }

    #[derive(Clone, Copy)]
    struct ConvShape {
        batch_size: usize,
        seq_len: usize,
        conv_dim: usize,
        kernel_size: usize,
    }

    fn causal_conv_reference(
        x: &[f32],
        weight: &[f32],
        initial_state: &[f32],
        shape: ConvShape,
    ) -> (Vec<f32>, Vec<f32>) {
        let ConvShape {
            batch_size,
            seq_len,
            conv_dim,
            kernel_size,
        } = shape;
        let mut state = initial_state.to_vec();
        let mut output = vec![0.0f32; batch_size * seq_len * conv_dim];
        for b in 0..batch_size {
            for pos in 0..seq_len {
                for ch in 0..conv_dim {
                    let state_base = (b * conv_dim + ch) * kernel_size;
                    state.copy_within(state_base + 1..state_base + kernel_size, state_base);
                    state[state_base + kernel_size - 1] = x[(b * seq_len + pos) * conv_dim + ch];
                    let weight_base = ch * kernel_size;
                    let mut sum = 0.0f32;
                    for k in 0..kernel_size {
                        sum += state[state_base + k] * weight[weight_base + k];
                    }
                    output[(b * seq_len + pos) * conv_dim + ch] = sum / (1.0 + (-sum).exp());
                }
            }
        }
        (output, state)
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn causal_conv1d_strided_nonzero_offset_matches_reference_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let batch_size = 3;
        let conv_dim = 19;
        let kernel_size = 4;
        let prefix = 3;
        let physical_dim = conv_dim + 7;

        for seq_len in [1usize, 5] {
            let logical = patterned(batch_size * seq_len * conv_dim, 60 + seq_len, 0.08, 0.01);
            let mut packed = vec![-7.0f32; batch_size * seq_len * physical_dim];
            for b in 0..batch_size {
                for pos in 0..seq_len {
                    let logical_base = (b * seq_len + pos) * conv_dim;
                    let packed_base = (b * seq_len + pos) * physical_dim + prefix;
                    packed[packed_base..packed_base + conv_dim]
                        .copy_from_slice(&logical[logical_base..logical_base + conv_dim]);
                }
            }
            let x = Tensor::from_vec(packed, (batch_size, seq_len, physical_dim), &dev)?
                .to_dtype(DType::F16)?
                .narrow(2, prefix, conv_dim)?;
            assert!(!x.is_contiguous());
            assert!(x.layout().start_offset() > 0);

            let weight = tensor2(
                patterned(conv_dim * kernel_size, 70 + seq_len, 0.05, -0.01),
                (conv_dim, kernel_size),
                &dev,
            )?
            .to_dtype(DType::F16)?;
            let state = tensor3(
                patterned(batch_size * conv_dim * kernel_size, 80 + seq_len, 0.03, 0.0),
                (batch_size, conv_dim, kernel_size),
                &dev,
            )?
            .to_dtype(DType::F16)?;
            let x_host = flat(&x.to_dtype(DType::F32)?.contiguous()?)?;
            let weight_host = flat(&weight.to_dtype(DType::F32)?)?;
            let state_host = flat(&state.to_dtype(DType::F32)?)?;
            let (expected, expected_state) = causal_conv_reference(
                &x_host,
                &weight_host,
                &state_host,
                ConvShape {
                    batch_size,
                    seq_len,
                    conv_dim,
                    kernel_size,
                },
            );

            let (actual, actual_state) = causal_conv1d_cuda(
                &x,
                &weight,
                &state,
                kernel_size,
                seq_len == 1,
                GdnStateSlots::Gathered,
            )?;
            assert_eq!(actual.dims3()?, (batch_size, seq_len, conv_dim));
            assert!(actual.is_contiguous());
            assert_close(
                "strided causal conv output",
                &flat(&actual.to_dtype(DType::F32)?)?,
                &expected,
                2.0e-3,
            );
            assert_close(
                "strided causal conv state",
                &flat(&actual_state.to_dtype(DType::F32)?)?,
                &expected_state,
                5.0e-4,
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn rmsnorm_gated_strided_nonzero_offset_matches_reference_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (batch_size, seq_len, heads, hidden_dim) = (3, 2, 5, 17);
        let x_physical_dim = hidden_dim + 7;
        let value_dim = heads * hidden_dim;
        let gate_physical_dim = value_dim + 7;
        let x = Tensor::from_vec(
            patterned(batch_size * seq_len * heads * x_physical_dim, 91, 0.2, 0.01),
            (batch_size, heads, seq_len, x_physical_dim),
            &dev,
        )?
        .to_dtype(DType::BF16)?
        .narrow(3, 2, hidden_dim)?
        .transpose(1, 2)?;
        let gate = Tensor::from_vec(
            patterned(batch_size * seq_len * gate_physical_dim, 92, 0.3, -0.02),
            (batch_size, seq_len, gate_physical_dim),
            &dev,
        )?
        .to_dtype(DType::BF16)?
        .narrow(2, 3, value_dim)?;
        let weight = Tensor::from_vec(patterned(hidden_dim, 93, 0.1, 1.0), (hidden_dim,), &dev)?
            .to_dtype(DType::BF16)?;
        assert!(!x.is_contiguous());
        assert!(!gate.is_contiguous());
        assert!(x.layout().start_offset() > 0 && gate.layout().start_offset() > 0);

        let rows = batch_size * seq_len * heads;
        let x_host = flat(&x.to_dtype(DType::F32)?.contiguous()?)?;
        let gate_host = flat(&gate.to_dtype(DType::F32)?.contiguous()?)?;
        let weight_host = flat(&weight.to_dtype(DType::F32)?)?;
        let eps = 1.0e-6;
        let mut expected = vec![0.0f32; rows * hidden_dim];
        for row in 0..rows {
            let base = row * hidden_dim;
            let mean_square = x_host[base..base + hidden_dim]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / hidden_dim as f32;
            let inv_rms = (mean_square + eps as f32).sqrt().recip();
            for col in 0..hidden_dim {
                let gate_value = gate_host[base + col];
                let silu_gate = gate_value / (1.0 + (-gate_value).exp());
                expected[base + col] = x_host[base + col] * inv_rms * weight_host[col] * silu_gate;
            }
        }

        let actual = rmsnorm_gated_cuda(&x, &gate, &weight, eps)?;
        assert_eq!(actual.shape(), x.shape());
        assert!(actual.is_contiguous());
        assert_close(
            "strided gated RMSNorm",
            &flat(&actual.to_dtype(DType::F32)?)?,
            &expected,
            2.0e-3,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn rmsnorm_gated_hidden128_matches_reference_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (batch_size, rows, hidden_dim) = (1, 1027, 128);
        for dtype in [DType::BF16, DType::F16] {
            let x = Tensor::from_vec(
                patterned(rows * hidden_dim, 94, 0.2, 0.01),
                (batch_size, rows, hidden_dim),
                &dev,
            )?
            .to_dtype(dtype)?;
            let gate = Tensor::from_vec(
                patterned(rows * hidden_dim, 95, 0.3, -0.02),
                (batch_size, rows, hidden_dim),
                &dev,
            )?
            .to_dtype(dtype)?;
            let weight =
                Tensor::from_vec(patterned(hidden_dim, 96, 0.1, 1.0), (hidden_dim,), &dev)?
                    .to_dtype(dtype)?;
            let x_host = flat(&x.to_dtype(DType::F32)?)?;
            let gate_host = flat(&gate.to_dtype(DType::F32)?)?;
            let weight_host = flat(&weight.to_dtype(DType::F32)?)?;
            let eps = 1.0e-6;
            let mut expected = vec![0.0f32; rows * hidden_dim];
            for row in 0..rows {
                let base = row * hidden_dim;
                let mean_square = x_host[base..base + hidden_dim]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    / hidden_dim as f32;
                let inv_rms = (mean_square + eps as f32).sqrt().recip();
                for col in 0..hidden_dim {
                    let gate_value = gate_host[base + col];
                    let silu_gate = gate_value / (1.0 + (-gate_value).exp());
                    expected[base + col] =
                        x_host[base + col] * inv_rms * weight_host[col] * silu_gate;
                }
            }

            let actual = rmsnorm_gated_cuda(&x, &gate, &weight, eps)?;
            assert_close(
                &format!("hidden-128 gated RMSNorm {dtype:?}"),
                &flat(&actual.to_dtype(DType::F32)?)?,
                &expected,
                2.0e-3,
            );
        }
        Ok(())
    }

    // Pooled kernels addressed through a permuted slot table must match the gathered kernels on
    // the same rows and leave every other pool row untouched.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn pooled_state_kernels_match_gathered_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let capacity = 6usize;
        let batch = 3usize;
        let slots_host: Vec<u32> = vec![4, 1, 5];
        let slots = Tensor::from_vec(slots_host.clone(), (batch,), &dev)?;
        let num_heads = 4usize;
        let k_dim = 128usize;
        let v_dim = 64usize;
        let conv_dim = 3 * num_heads * k_dim;
        let kernel_size = 4usize;

        let pool_rec_host = patterned(capacity * num_heads * k_dim * v_dim, 30, 0.01, 0.0);
        let pool_rec = Tensor::from_vec(
            pool_rec_host.clone(),
            (capacity, num_heads, k_dim, v_dim),
            &dev,
        )?;
        let pool_conv_host = patterned(capacity * conv_dim * kernel_size, 31, 0.03, 0.0);
        let pool_conv = Tensor::from_vec(pool_conv_host, (capacity, conv_dim, kernel_size), &dev)?
            .to_dtype(DType::BF16)?;
        let gathered_rec = pool_rec.index_select(&slots, 0)?.contiguous()?;
        let gathered_conv = pool_conv.index_select(&slots, 0)?.contiguous()?;

        for seq_len in [1usize, 3, 70] {
            let bh = batch * num_heads;
            let q = tensor3(
                patterned(bh * seq_len * k_dim, 1, 0.02, 0.0),
                (bh, seq_len, k_dim),
                &dev,
            )?;
            let k = tensor3(
                patterned(bh * seq_len * k_dim, 2, 0.02, 0.0),
                (bh, seq_len, k_dim),
                &dev,
            )?;
            let v = tensor3(
                patterned(bh * seq_len * v_dim, 3, 0.05, 0.0),
                (bh, seq_len, v_dim),
                &dev,
            )?;
            let g = tensor2(patterned(bh * seq_len, 4, 0.03, -0.08), (bh, seq_len), &dev)?;
            let beta = tensor2(patterned(bh * seq_len, 5, 0.15, 0.5), (bh, seq_len), &dev)?;

            let inputs = RecurrenceInputs {
                q: &q,
                k: &k,
                v: &v,
                g: &g,
                beta: &beta,
            };
            let mut state_gathered = gathered_rec.reshape((bh, k_dim, v_dim))?.copy()?;
            let mut state_pooled = pool_rec.copy()?;
            for kernel in [
                RecurrenceKernel::Scalar,
                RecurrenceKernel::Warp,
                RecurrenceKernel::Chunked,
            ] {
                let mut sg = state_gathered.copy()?;
                let mut sp = state_pooled.copy()?;
                let out_g = launch_recurrence(kernel, inputs, &mut sg, GdnStateSlots::Gathered)?;
                let out_p =
                    launch_recurrence(kernel, inputs, &mut sp, GdnStateSlots::Pooled(&slots))?;
                assert_close(
                    "pooled recurrence output",
                    &flat(&out_g)?,
                    &flat(&out_p)?,
                    1.0e-6,
                );
                let sp_rows = sp.index_select(&slots, 0)?.reshape((bh, k_dim, v_dim))?;
                assert_close(
                    "pooled recurrence state",
                    &flat(&sg)?,
                    &flat(&sp_rows)?,
                    1.0e-6,
                );
                let untouched = flat(&sp)?;
                for row in (0..capacity).filter(|r| !slots_host.contains(&(*r as u32))) {
                    let span = num_heads * k_dim * v_dim;
                    assert_close(
                        "pooled recurrence untouched row",
                        &untouched[row * span..(row + 1) * span],
                        &pool_rec_host[row * span..(row + 1) * span],
                        0.0,
                    );
                }
                state_gathered = sg;
                state_pooled = sp;
            }

            let x = tensor3(
                patterned(batch * conv_dim * seq_len, 20, 0.08, 0.01),
                (batch, seq_len, conv_dim),
                &dev,
            )?
            .to_dtype(DType::BF16)?;
            let weight = tensor2(
                patterned(conv_dim * kernel_size, 21, 0.05, -0.01),
                (conv_dim, kernel_size),
                &dev,
            )?
            .to_dtype(DType::BF16)?;
            let is_update = seq_len == 1;
            let (out_g, cs_g) = causal_conv1d_cuda(
                &x,
                &weight,
                &gathered_conv.copy()?,
                kernel_size,
                is_update,
                GdnStateSlots::Gathered,
            )?;
            let pool_copy = pool_conv.copy()?;
            let (out_p, cs_p) = causal_conv1d_cuda(
                &x,
                &weight,
                &pool_copy,
                kernel_size,
                is_update,
                GdnStateSlots::Pooled(&slots),
            )?;
            assert_close(
                "pooled conv output",
                &flat(&out_g.to_dtype(DType::F32)?)?,
                &flat(&out_p.to_dtype(DType::F32)?)?,
                0.0,
            );
            let cs_p_rows = cs_p.index_select(&slots, 0)?;
            assert_close(
                "pooled conv state",
                &flat(&cs_g.to_dtype(DType::F32)?)?,
                &flat(&cs_p_rows.to_dtype(DType::F32)?)?,
                0.0,
            );
        }
        Ok(())
    }

    fn assert_zero(label: &str, tensor: &Tensor) -> Result<()> {
        let values = flat(&tensor.to_dtype(DType::F32)?)?;
        let nonzero = values.iter().position(|value| *value != 0.0);
        assert!(nonzero.is_none(), "{label}: nonzero value at {nonzero:?}");
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn pooled_causal_conv_padding_rows_are_zero_and_stateless_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let capacity = 5usize;
        let batch = 3usize;
        let conv_dim = 7usize;
        let slots_host = vec![3u32, GDN_PAD_SLOT, 1];
        let slots = Tensor::from_vec(slots_host, (batch,), &dev)?;
        let real_batch = Tensor::from_vec(vec![0u32, 2], (2,), &dev)?;
        let real_slots = Tensor::from_vec(vec![3u32, 1], (2,), &dev)?;

        for dtype in [DType::F16, DType::BF16] {
            for kernel_size in [3usize, 4] {
                for (seq_len, is_update) in [(1usize, true), (3, false)] {
                    let x = tensor3(
                        patterned(batch * seq_len * conv_dim, 80, 0.08, 0.01),
                        (batch, seq_len, conv_dim),
                        &dev,
                    )?
                    .to_dtype(dtype)?;
                    let weight = tensor2(
                        patterned(conv_dim * kernel_size, 81, 0.05, -0.01),
                        (conv_dim, kernel_size),
                        &dev,
                    )?
                    .to_dtype(dtype)?;
                    let initial = tensor3(
                        patterned(capacity * conv_dim * kernel_size, 82, 0.03, 0.0),
                        (capacity, conv_dim, kernel_size),
                        &dev,
                    )?
                    .to_dtype(dtype)?;
                    let x_real = x.index_select(&real_batch, 0)?.contiguous()?;
                    let gathered_initial = initial.index_select(&real_slots, 0)?.contiguous()?;
                    let (expected_output, expected_state) = causal_conv1d_cuda(
                        &x_real,
                        &weight,
                        &gathered_initial,
                        kernel_size,
                        is_update,
                        GdnStateSlots::Gathered,
                    )?;
                    let pool = initial.copy()?;
                    let (actual_output, actual_state) = causal_conv1d_cuda(
                        &x,
                        &weight,
                        &pool,
                        kernel_size,
                        is_update,
                        GdnStateSlots::Pooled(&slots),
                    )?;

                    assert_zero(
                        "causal convolution padded output",
                        &actual_output.narrow(0, 1, 1)?,
                    )?;
                    assert_close(
                        "causal convolution real output",
                        &flat(
                            &actual_output
                                .index_select(&real_batch, 0)?
                                .to_dtype(DType::F32)?,
                        )?,
                        &flat(&expected_output.to_dtype(DType::F32)?)?,
                        0.0,
                    );
                    assert_close(
                        "causal convolution real state",
                        &flat(
                            &actual_state
                                .index_select(&real_slots, 0)?
                                .to_dtype(DType::F32)?,
                        )?,
                        &flat(&expected_state.to_dtype(DType::F32)?)?,
                        0.0,
                    );
                    for row in [0usize, 2, 4] {
                        assert_close(
                            "causal convolution untouched state",
                            &flat(&actual_state.narrow(0, row, 1)?.to_dtype(DType::F32)?)?,
                            &flat(&initial.narrow(0, row, 1)?.to_dtype(DType::F32)?)?,
                            0.0,
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn pooled_decomposed_recurrence_padding_rows_are_zero_and_stateless_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let capacity = 5usize;
        let batch = 3usize;
        let num_heads = 2usize;
        let seq_len = 3usize;
        let k_dim = 128usize;
        let v_dim = 128usize;
        let bh = batch * num_heads;
        let real_bh = 2 * num_heads;
        let slots = Tensor::from_vec(vec![4u32, GDN_PAD_SLOT, 1], (batch,), &dev)?;
        let real_slots = Tensor::from_vec(vec![4u32, 1], (2,), &dev)?;
        let real_heads = Tensor::from_vec(vec![0u32, 1, 4, 5], (real_bh,), &dev)?;
        let q = tensor3(
            patterned(bh * seq_len * k_dim, 90, 0.02, 0.0),
            (bh, seq_len, k_dim),
            &dev,
        )?;
        let k = tensor3(
            patterned(bh * seq_len * k_dim, 91, 0.02, 0.0),
            (bh, seq_len, k_dim),
            &dev,
        )?;
        let v = tensor3(
            patterned(bh * seq_len * v_dim, 92, 0.05, 0.0),
            (bh, seq_len, v_dim),
            &dev,
        )?;
        let g = tensor2(
            patterned(bh * seq_len, 93, 0.03, -0.08),
            (bh, seq_len),
            &dev,
        )?;
        let beta = tensor2(patterned(bh * seq_len, 94, 0.15, 0.5), (bh, seq_len), &dev)?;
        let real_q = q.index_select(&real_heads, 0)?.contiguous()?;
        let real_k = k.index_select(&real_heads, 0)?.contiguous()?;
        let real_v = v.index_select(&real_heads, 0)?.contiguous()?;
        let real_g = g.index_select(&real_heads, 0)?.contiguous()?;
        let real_beta = beta.index_select(&real_heads, 0)?.contiguous()?;
        let pooled_inputs = RecurrenceInputs {
            q: &q,
            k: &k,
            v: &v,
            g: &g,
            beta: &beta,
        };
        let gathered_inputs = RecurrenceInputs {
            q: &real_q,
            k: &real_k,
            v: &real_v,
            g: &real_g,
            beta: &real_beta,
        };
        let initial = Tensor::from_vec(
            patterned(capacity * num_heads * k_dim * v_dim, 95, 0.01, 0.0),
            (capacity, num_heads, k_dim, v_dim),
            &dev,
        )?;

        for (kernel, label) in [
            (RecurrenceKernel::Scalar, "scalar"),
            (RecurrenceKernel::Warp, "warp"),
            (RecurrenceKernel::Chunked, "chunked"),
        ] {
            let mut pooled_state = initial.copy()?;
            let mut gathered_state = initial
                .index_select(&real_slots, 0)?
                .reshape((real_bh, k_dim, v_dim))?
                .contiguous()?;
            let actual = launch_recurrence(
                kernel,
                pooled_inputs,
                &mut pooled_state,
                GdnStateSlots::Pooled(&slots),
            )?;
            let expected = launch_recurrence(
                kernel,
                gathered_inputs,
                &mut gathered_state,
                GdnStateSlots::Gathered,
            )?;
            assert_zero(
                &format!("{label} padded output"),
                &actual.narrow(0, num_heads, num_heads)?,
            )?;
            assert_close(
                &format!("{label} real output"),
                &flat(&actual.index_select(&real_heads, 0)?)?,
                &flat(&expected)?,
                1.0e-6,
            );
            assert_close(
                &format!("{label} real state"),
                &flat(
                    &pooled_state
                        .index_select(&real_slots, 0)?
                        .reshape((real_bh, k_dim, v_dim))?,
                )?,
                &flat(&gathered_state)?,
                1.0e-6,
            );
            for row in [0usize, 2, 3] {
                assert_close(
                    &format!("{label} untouched state"),
                    &flat(&pooled_state.narrow(0, row, 1)?)?,
                    &flat(&initial.narrow(0, row, 1)?)?,
                    0.0,
                );
            }
        }

        let initial = initial.transpose(2, 3)?.contiguous()?;
        let mut pooled_state = initial.copy()?;
        let mut gathered_state = initial
            .index_select(&real_slots, 0)?
            .reshape((real_bh, v_dim, k_dim))?
            .contiguous()?;
        let actual = launch_recurrence(
            RecurrenceKernel::ValueMajorWarp,
            pooled_inputs,
            &mut pooled_state,
            GdnStateSlots::Pooled(&slots),
        )?;
        let expected = launch_recurrence(
            RecurrenceKernel::ValueMajorWarp,
            gathered_inputs,
            &mut gathered_state,
            GdnStateSlots::Gathered,
        )?;
        assert_zero(
            "value-major warp padded output",
            &actual.narrow(0, num_heads, num_heads)?,
        )?;
        assert_close(
            "value-major warp real output",
            &flat(&actual.index_select(&real_heads, 0)?)?,
            &flat(&expected)?,
            1.0e-6,
        );
        assert_close(
            "value-major warp real state",
            &flat(
                &pooled_state
                    .index_select(&real_slots, 0)?
                    .reshape((real_bh, v_dim, k_dim))?,
            )?,
            &flat(&gathered_state)?,
            1.0e-6,
        );
        for row in [0usize, 2, 3] {
            assert_close(
                "value-major warp untouched state",
                &flat(&pooled_state.narrow(0, row, 1)?)?,
                &flat(&initial.narrow(0, row, 1)?)?,
                0.0,
            );
        }
        Ok(())
    }

    fn run_fused_decode_padding_case(
        dev: &Device,
        kernel: GdnDecodeKernel,
        head_k_dim: usize,
        head_v_dim: usize,
        dtype: DType,
        state_layout: RecurrentStateLayout,
    ) -> Result<()> {
        let capacity = 4usize;
        let batch = 3usize;
        let num_k_heads = 1usize;
        let num_v_heads = 1usize;
        let conv_dim = 2 * head_k_dim + head_v_dim;
        let slots = Tensor::from_vec(vec![2u32, GDN_PAD_SLOT, 0], (batch,), dev)?;
        let real_batch = Tensor::from_vec(vec![0u32, 2], (2,), dev)?;
        let real_slots = Tensor::from_vec(vec![2u32, 0], (2,), dev)?;
        let mixed_qkv = tensor3(
            patterned(batch * conv_dim, 100, 0.08, 0.01),
            (batch, 1, conv_dim),
            dev,
        )?
        .to_dtype(dtype)?;
        let b = tensor3(
            patterned(batch * num_v_heads, 101, 0.2, 0.1),
            (batch, 1, num_v_heads),
            dev,
        )?
        .to_dtype(dtype)?;
        let a = tensor3(
            patterned(batch * num_v_heads, 102, 0.18, -0.04),
            (batch, 1, num_v_heads),
            dev,
        )?
        .to_dtype(dtype)?;
        let a_log = Tensor::from_vec(patterned(num_v_heads, 103, 0.05, -0.2), (num_v_heads,), dev)?;
        let dt_bias = Tensor::from_vec(patterned(num_v_heads, 104, 0.1, 0.3), (num_v_heads,), dev)?;
        let initial = Tensor::from_vec(
            patterned(
                capacity * num_v_heads * head_k_dim * head_v_dim,
                105,
                0.02,
                0.0,
            ),
            (capacity, num_v_heads, head_k_dim, head_v_dim),
            dev,
        )?;
        let initial = if state_layout == RecurrentStateLayout::GdnValueMajor {
            initial.transpose(2, 3)?.contiguous()?
        } else {
            initial
        };
        let mut pooled_state = initial.copy()?;
        let mut gathered_state = initial.index_select(&real_slots, 0)?.contiguous()?;
        let actual = fused_decode_recurrence_cuda_impl(GdnDecodeLaunch {
            mixed_qkv: &mixed_qkv,
            b: &b,
            a: &a,
            a_log: &a_log,
            dt_bias: &dt_bias,
            state: &mut pooled_state,
            batch_size: batch,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads: false,
            state_layout,
            slots: GdnStateSlots::Pooled(&slots),
            requested_kernel: Some(kernel),
        })?;
        let mixed_qkv_real = mixed_qkv.index_select(&real_batch, 0)?.contiguous()?;
        let b_real = b.index_select(&real_batch, 0)?.contiguous()?;
        let a_real = a.index_select(&real_batch, 0)?.contiguous()?;
        let expected = fused_decode_recurrence_cuda_impl(GdnDecodeLaunch {
            mixed_qkv: &mixed_qkv_real,
            b: &b_real,
            a: &a_real,
            a_log: &a_log,
            dt_bias: &dt_bias,
            state: &mut gathered_state,
            batch_size: 2,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads: false,
            state_layout,
            slots: GdnStateSlots::Gathered,
            requested_kernel: Some(kernel),
        })?;

        assert_zero("fused decode padded output", &actual.narrow(0, 1, 1)?)?;
        assert_close(
            "fused decode real output",
            &flat(&actual.index_select(&real_batch, 0)?.to_dtype(DType::F32)?)?,
            &flat(&expected.to_dtype(DType::F32)?)?,
            0.0,
        );
        assert_close(
            "fused decode real state",
            &flat(&pooled_state.index_select(&real_slots, 0)?)?,
            &flat(&gathered_state)?,
            0.0,
        );
        for row in [1usize, 3] {
            assert_close(
                "fused decode untouched state",
                &flat(&pooled_state.narrow(0, row, 1)?)?,
                &flat(&initial.narrow(0, row, 1)?)?,
                0.0,
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn fused_decode_dispatches_zero_padding_without_touching_state_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        run_fused_decode_padding_case(
            &dev,
            GdnDecodeKernel::Baseline,
            64,
            64,
            DType::F16,
            RecurrentStateLayout::GdnKeyMajor,
        )?;
        run_fused_decode_padding_case(
            &dev,
            GdnDecodeKernel::Baseline,
            96,
            64,
            DType::BF16,
            RecurrentStateLayout::GdnKeyMajor,
        )?;
        run_fused_decode_padding_case(
            &dev,
            GdnDecodeKernel::Baseline,
            128,
            128,
            DType::BF16,
            RecurrentStateLayout::GdnKeyMajor,
        )?;

        let properties = gdn_cuda_device_properties(dev.as_cuda_device()?)?;
        if properties.compute_major >= GDN_DECODE_MIN_COMPUTE_MAJOR {
            for kernel in [GdnDecodeKernel::Cooperative, GdnDecodeKernel::Pipelined] {
                run_fused_decode_padding_case(
                    &dev,
                    kernel,
                    128,
                    128,
                    DType::BF16,
                    RecurrentStateLayout::GdnKeyMajor,
                )?;
            }
        }
        if properties.compute_major == GDN_DECODE_TUNED_COMPUTE_MAJOR {
            for kernel in [GdnDecodeKernel::ValueMajor4, GdnDecodeKernel::ValueMajor32] {
                run_fused_decode_padding_case(
                    &dev,
                    kernel,
                    128,
                    128,
                    DType::BF16,
                    RecurrentStateLayout::GdnValueMajor,
                )?;
            }
        }
        Ok(())
    }
}
