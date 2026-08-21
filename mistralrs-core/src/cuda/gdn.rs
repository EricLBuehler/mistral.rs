#![allow(clippy::cast_possible_truncation)]

use candle_core::{Result, Tensor};

#[cfg(feature = "cuda")]
use candle_core::DType;

/// Which rows of the recurrent state a GDN kernel reads and writes. `Gathered` means the state is a
/// `[B*H, ...]` copy addressed by batch row; `Pooled` addresses the whole pool `[cap, H, ...]`
/// through a `[B]` u32 slot table, so the kernels update it in place without gather/scatter copies.
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
#[allow(clippy::too_many_arguments)]
pub fn fused_decode_recurrence_cuda(
    mixed_qkv: &Tensor,
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    state: &mut Tensor,
    batch_size: usize,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    tiled_v_heads: bool,
    slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
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
        state: &mut Tensor,
        batch_size: usize,
        num_k_heads: usize,
        num_v_heads: usize,
        head_k_dim: usize,
        head_v_dim: usize,
        tiled_v_heads: bool,
        slots: GdnStateSlots<'_>,
        dtype_code: i32,
    ) -> Result<Tensor> {
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

        let (state_s, state_l) = state.storage_and_layout();
        let state_s = match &*state_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
            _ => candle::bail!("state must be a cuda tensor"),
        };
        let state_offset = state_l.start_offset();

        let bh = batch_size * num_v_heads;
        let output_buf = unsafe { dev.alloc::<T>(bh * head_v_dim) }?;
        let stream = dev.cuda_stream().cu_stream() as i64;

        with_slot_indices(slots, |slot_ptr, _| {
            unsafe {
                crate::cuda::ffi::gdn_decode_recurrence(
                    mixed_s.slice(mixed_offset..).device_ptr(mixed_s.stream()).0 as *const c_void,
                    b_s.slice(b_offset..).device_ptr(b_s.stream()).0 as *const c_void,
                    a_s.slice(a_offset..).device_ptr(a_s.stream()).0 as *const c_void,
                    alog_s.slice(alog_offset..).device_ptr(alog_s.stream()).0 as *const f32,
                    dtb_s.slice(dtb_offset..).device_ptr(dtb_s.stream()).0 as *const f32,
                    state_s.slice(state_offset..).device_ptr(state_s.stream()).0 as *mut f32,
                    output_buf.device_ptr(output_buf.stream()).0 as *mut c_void,
                    batch_size as i32,
                    num_k_heads as i32,
                    num_v_heads as i32,
                    head_k_dim as i32,
                    head_v_dim as i32,
                    i32::from(tiled_v_heads),
                    slot_ptr,
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

    match mixed_qkv.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(
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
            slots,
            0,
        ),
        DType::BF16 => cuda_fwd::<half::bf16>(
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
            slots,
            1,
        ),
        other => candle_core::bail!(
            "fused_decode_recurrence_cuda only supports f16/bf16, got {:?}",
            other
        ),
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused, clippy::too_many_arguments)]
pub fn fused_decode_recurrence_cuda(
    _mixed_qkv: &Tensor,
    _b: &Tensor,
    _a: &Tensor,
    _a_log: &Tensor,
    _dt_bias: &Tensor,
    _state: &mut Tensor,
    _batch_size: usize,
    _num_k_heads: usize,
    _num_v_heads: usize,
    _head_k_dim: usize,
    _head_v_dim: usize,
    _tiled_v_heads: bool,
    _slots: GdnStateSlots<'_>,
) -> Result<Tensor> {
    candle_core::bail!("fused_decode_recurrence_cuda requires the cuda feature")
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

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::Device;

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
        let b = tensor3(
            patterned(batch_size * num_v_heads, 41, 0.2, 0.1),
            (batch_size, 1, num_v_heads),
            dev,
        )?
        .to_dtype(dtype)?;
        let a = tensor3(
            patterned(batch_size * num_v_heads, 42, 0.18, -0.04),
            (batch_size, 1, num_v_heads),
            dev,
        )?
        .to_dtype(dtype)?;
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

        let fused = fused_decode_recurrence_cuda(
            &mixed_qkv,
            &b,
            &a,
            &a_log,
            &dt_bias,
            &mut fused_state,
            batch_size,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            tiled_v_heads,
            state_slots,
        )?;
        let (q, k, v, g, beta) = prepare_recurrence_inputs_cuda(
            &mixed_qkv,
            &b,
            &a,
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
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn fused_decode_recurrence_matches_decomposed_cuda() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        run_fused_decode_case(&dev, 2, 2, 4, 64, 64, false, DType::F16, false)?;
        run_fused_decode_case(&dev, 2, 2, 6, 128, 128, true, DType::BF16, true)
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
}
