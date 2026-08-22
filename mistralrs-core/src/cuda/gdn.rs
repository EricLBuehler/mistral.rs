#![allow(clippy::cast_possible_truncation)]

use candle_core::{DType, Device, Result, Tensor};

use crate::kv_cache::RecurrentStateLayout;

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) const GDN_PAD_SLOT: u32 = u32::MAX;

#[cfg(any(feature = "cuda", test))]
const GDN_DECODE_MIN_COMPUTE_MAJOR: i32 = 9;
#[cfg_attr(not(any(feature = "cuda", test)), allow(dead_code))]
const GDN_DECODE_TUNED_COMPUTE_MAJOR: i32 = 9;
const GDN_DECODE_K_DIM: usize = 128;
const GDN_DECODE_V_DIM: usize = 128;
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
const GDN_DECODE_VECTOR_ALIGNMENT: usize = 16;
#[cfg(feature = "cuda")]
const GDN_DECODE_INPUT_ALIGNMENT: usize = 8;
pub(crate) const GDN_SPEC_COMMIT_MAX_K: usize = 256;
pub(crate) const GDN_SPEC_CHECKPOINT_MAX_CONV_WIDTH: usize = 16;
pub(crate) const GDN_SPEC_CHECKPOINT_MAX_K: usize = 256;
#[cfg(feature = "cuda")]
const GDN_DECODE_KERNEL_ENV: &str = "MISTRALRS_GDN_DECODE_KERNEL";

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
    if input_dtype != DType::BF16
        || key_dim != GDN_DECODE_K_DIM
        || value_dim != GDN_DECODE_V_DIM
        || !(key_dim * std::mem::size_of::<f32>()).is_multiple_of(GDN_DECODE_VECTOR_ALIGNMENT)
    {
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
#[derive(Clone, Copy)]
enum RecurrenceKernel {
    Scalar,
    Warp,
    ValueMajorWarp,
    Chunked,
}

/// `state` is `[BH, K, V]` (gathered) or the `[cap, H, K, V]` pool (pooled), mutated in place.
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
    let state_ptr = f32_ptr!(state, "state");

    let output_buf = unsafe { dev.alloc::<f32>(bh * seq_len * v_dim) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;

    with_slot_indices(slots, |slot_ptr, batch| {
        let num_heads = bh.checked_div(batch).unwrap_or(1);
        let launcher = match kernel {
            RecurrenceKernel::Scalar => crate::cuda::ffi::gated_delta_rule_recurrence,
            RecurrenceKernel::Warp => crate::cuda::ffi::warp_gated_delta_rule_recurrence,
            RecurrenceKernel::ValueMajorWarp => {
                crate::cuda::ffi::vmajor_warp_gated_delta_rule_recurrence
            }
            RecurrenceKernel::Chunked => crate::cuda::ffi::chunked_gated_delta_rule_recurrence,
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
pub fn vmajor_warp_gated_delta_rule_recurrence_cuda(
    inputs: RecurrenceInputs<'_>,
    state: &mut Tensor,
    slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    launch_recurrence(RecurrenceKernel::ValueMajorWarp, inputs, state, slots)
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

        let (state_s, state_l) = state.storage_and_layout();
        let state_s = match &*state_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("state must be a cuda tensor"),
        };
        let state_offset = state_l.start_offset();
        let state_ptr = state_s.slice(state_offset..).device_ptr(state_s.stream()).0 as *mut f32;
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
        if initial_recurrent_state.dtype() != DType::F32
            || recurrent_state_pool.dtype() != DType::F32
        {
            candle::bail!("GDN speculative commit requires f32 recurrent state");
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
        let initial_recurrent_ptr =
            typed_ptr!(initial_recurrent_state, f32, "initial_recurrent_state") as *const f32;
        let a_log_ptr = typed_ptr!(a_log, f32, "a_log") as *const f32;
        let dt_bias_ptr = typed_ptr!(dt_bias, f32, "dt_bias") as *const f32;
        let conv_pool_ptr = typed_ptr!(conv_state_pool, T, "conv_state_pool") as *mut c_void;
        let recurrent_pool_ptr =
            typed_ptr!(recurrent_state_pool, f32, "recurrent_state_pool") as *mut f32;
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
                initial_recurrent_ptr,
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
pub struct GdnSpeculativeConvCheckpoints<'a> {
    pub x: &'a Tensor,
    pub weight: &'a Tensor,
    pub state_pool: &'a Tensor,
    pub active_slots: &'a Tensor,
    pub checkpoint_lanes: usize,
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
        if checkpoint_lanes == 0 || seq_len > checkpoint_lanes {
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
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub fn speculative_recurrence_checkpoints_cuda(
    context: GdnSpeculativeRecurrenceCheckpoints<'_>,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use core::ffi::c_void;

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        context: GdnSpeculativeRecurrenceCheckpoints<'_>,
        dtype_code: i32,
    ) -> Result<Tensor> {
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
        if checkpoint_lanes == 0 || seq_len > checkpoint_lanes {
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
        if state_pool.dtype() != DType::F32 {
            candle::bail!("GDN speculative recurrence state must be f32");
        }
        for tensor in [b, a, a_log, dt_bias, state_pool, active_slots] {
            if !tensor.device().same_device(mixed_qkv.device()) {
                candle::bail!("GDN speculative recurrence tensors must share one device");
            }
        }
        if !state_pool.is_contiguous() || !active_slots.is_contiguous() {
            candle::bail!("GDN speculative recurrence state and slots must be contiguous");
        }

        let mixed_qkv = mixed_qkv.contiguous()?;
        let b = b.contiguous()?;
        let a = a.contiguous()?;
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
        let state_ptr = cuda_ptr!(state_pool, f32, "state_pool") as *mut f32;
        let slots_ptr = cuda_ptr!(active_slots, u32, "active_slots") as *const u32;
        let output = unsafe { dev.alloc::<T>(batch_size * num_v_heads * seq_len * head_v_dim) }?;
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
                batch_size as i32,
                seq_len as i32,
                num_k_heads as i32,
                num_v_heads as i32,
                head_k_dim as i32,
                head_v_dim as i32,
                checkpoint_lanes as i32,
                i32::from(tiled_v_heads),
                i32::from(value_major),
                dtype_code,
                stream,
            );
        }

        Ok(Tensor::from((
            candle::Storage::Cuda(candle::CudaStorage::wrap_cuda_slice(output, dev.clone())),
            (batch_size * num_v_heads, seq_len, head_v_dim),
        )))
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
) -> Result<Tensor> {
    candle_core::bail!("speculative_recurrence_checkpoints_cuda requires the cuda feature")
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
        automatic_decode_kernel, parse_decode_kernel, select_decode_kernel, GdnDecodeKernel,
        GdnDecodePolicy,
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

    #[test]
    #[ignore = "requires a CUDA device"]
    fn value_major_warp_prefill_matches_scalar_with_shuffled_slots() -> Result<()> {
        const BATCH_SIZE: usize = 2;
        const NUM_HEADS: usize = 4;
        const SEQ_LEN: usize = 17;
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
        let mut value_major_state = initial_state.transpose(2, 3)?.contiguous()?;
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
            let value_major = vmajor_warp_gated_delta_rule_recurrence_cuda(
                inputs,
                &mut value_major_state,
                slots,
            )?;
            assert_close(
                &format!("value-major prefill output step {step}"),
                &flat(&value_major)?,
                &flat(&reference)?,
                3.0e-4,
            );
            assert_close(
                &format!("value-major prefill state step {step}"),
                &flat(&value_major_state.transpose(2, 3)?.contiguous()?)?,
                &flat(&key_major_state)?,
                3.0e-4,
            );
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
        )?;
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
            &flat(&fused_state)?,
            &flat(&reference_state)?,
            2.0e-5,
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
            &flat(&fused_state)?,
            &flat(&reference_state)?,
            4.0e-5,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn fused_decode_recurrence_matches_decomposed_cuda() -> Result<()> {
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
        )?;
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
        )?;
        let mut expected_conv = flat(&conv_state_pool.to_dtype(DType::F32)?)?;
        let mut expected_recurrent = flat(&recurrent_state_pool)?;

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
            let state = flat(&state)?;
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
            &flat(&recurrent_state_pool)?,
            &expected_recurrent,
            2.0e-5,
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn speculative_state_commit_matches_prefix_replay_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        run_speculative_state_commit_case(&dev, RecurrentStateLayout::GdnKeyMajor)?;
        run_speculative_state_commit_case(&dev, RecurrentStateLayout::GdnValueMajor)
    }

    #[test]
    #[ignore = "requires an SM90 CUDA device"]
    fn speculative_checkpoint_kernels_match_serial_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let batch_size = 2;
        let seq_len = 4;
        let checkpoint_lanes = 4;
        let capacity = batch_size * checkpoint_lanes;
        let active_slots_host = vec![3u32, 6];
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
        let b = tensor3(
            patterned(batch_size * seq_len * num_v_heads, 134, 0.2, 0.1),
            (batch_size, seq_len, num_v_heads),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let a = tensor3(
            patterned(batch_size * seq_len * num_v_heads, 135, 0.18, -0.04),
            (batch_size, seq_len, num_v_heads),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let a_log = Tensor::from_vec(
            patterned(num_v_heads, 136, 0.05, -0.2),
            (num_v_heads,),
            &dev,
        )?;
        let dt_bias =
            Tensor::from_vec(patterned(num_v_heads, 137, 0.1, 0.3), (num_v_heads,), &dev)?;
        let state_elements = num_v_heads * head_v_dim * head_k_dim;
        let recurrent_pool = Tensor::from_vec(
            patterned(capacity * state_elements, 138, 0.02, 0.0),
            (capacity, num_v_heads, head_v_dim, head_k_dim),
            &dev,
        )?;
        let mut expected_recurrent_pool = flat(&recurrent_pool)?;
        let mut expected_recurrent_outputs = Vec::with_capacity(batch_size);
        for (batch_idx, &active_slot) in active_slots_host.iter().enumerate() {
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
            let mut state = recurrent_pool.narrow(0, active_slot as usize, 1)?.copy()?;
            let base_slot = active_slot as usize / checkpoint_lanes * checkpoint_lanes;
            let mut row_outputs = Vec::with_capacity(seq_len);
            for position in 0..seq_len {
                let q = q.narrow(1, position, 1)?.contiguous()?;
                let k = k.narrow(1, position, 1)?.contiguous()?;
                let v = v.narrow(1, position, 1)?.contiguous()?;
                let g = g.narrow(1, position, 1)?.contiguous()?;
                let beta = beta.narrow(1, position, 1)?.contiguous()?;
                let output = vmajor_warp_gated_delta_rule_recurrence_cuda(
                    RecurrenceInputs {
                        q: &q,
                        k: &k,
                        v: &v,
                        g: &g,
                        beta: &beta,
                    },
                    &mut state,
                    GdnStateSlots::Gathered,
                )?;
                row_outputs.push(output);
                let state = flat(&state)?;
                let destination = (base_slot + position) * state_elements;
                expected_recurrent_pool[destination..destination + state.len()]
                    .copy_from_slice(&state);
            }
            expected_recurrent_outputs.push(Tensor::cat(&row_outputs, 1)?);
        }
        let expected_recurrent_output = Tensor::cat(&expected_recurrent_outputs, 0)?;
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
                state_layout: RecurrentStateLayout::GdnValueMajor,
            })?;
        assert_close(
            "speculative checkpoint recurrence output",
            &flat(&actual_recurrent_output.to_dtype(DType::F32)?)?,
            &flat(&expected_recurrent_output)?,
            3.0e-3,
        );
        assert_close(
            "speculative checkpoint recurrence state",
            &flat(&recurrent_pool)?,
            &expected_recurrent_pool,
            2.0e-4,
        );
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
        let conv_dim = 19;
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
