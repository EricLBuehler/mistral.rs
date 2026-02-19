#![allow(clippy::cast_possible_truncation)]

use candle_core::{Result, Tensor};

#[cfg(feature = "cuda")]
use candle_core::DType;

/// CUDA-accelerated gated delta rule recurrence.
///
/// Inputs (all contiguous, f32):
///   q, k: [BH, S, K]  v: [BH, S, V]  g, beta: [BH, S]
///   state: [BH, K, V] (mutated in place)
///
/// Returns: output [BH, S, V]
#[cfg(feature = "cuda")]
pub fn gated_delta_rule_recurrence_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;

    let (bh, seq_len, k_dim) = q.dims3()?;
    let v_dim = v.dim(2)?;

    let dev = q.device().as_cuda_device()?;

    let (q_s, q_l) = q.storage_and_layout();
    let q_s = match &*q_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("q must be a cuda tensor"),
    };
    let q_offset = q_l.start_offset();

    let (k_s, k_l) = k.storage_and_layout();
    let k_s = match &*k_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("k must be a cuda tensor"),
    };
    let k_offset = k_l.start_offset();

    let (v_s, v_l) = v.storage_and_layout();
    let v_s = match &*v_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("v must be a cuda tensor"),
    };
    let v_offset = v_l.start_offset();

    let (g_s, g_l) = g.storage_and_layout();
    let g_s = match &*g_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("g must be a cuda tensor"),
    };
    let g_offset = g_l.start_offset();

    let (beta_s, beta_l) = beta.storage_and_layout();
    let beta_s = match &*beta_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("beta must be a cuda tensor"),
    };
    let beta_offset = beta_l.start_offset();

    let (state_s, state_l) = state.storage_and_layout();
    let state_s = match &*state_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("state must be a cuda tensor"),
    };
    let state_offset = state_l.start_offset();

    let output_buf = unsafe { dev.alloc::<f32>(bh * seq_len * v_dim) }?;

    let stream = dev.cuda_stream().cu_stream() as i64;

    unsafe {
        crate::cuda::ffi::gated_delta_rule_recurrence(
            q_s.slice(q_offset..).device_ptr(q_s.stream()).0 as *const f32,
            k_s.slice(k_offset..).device_ptr(k_s.stream()).0 as *const f32,
            v_s.slice(v_offset..).device_ptr(v_s.stream()).0 as *const f32,
            g_s.slice(g_offset..).device_ptr(g_s.stream()).0 as *const f32,
            beta_s.slice(beta_offset..).device_ptr(beta_s.stream()).0 as *const f32,
            state_s.slice(state_offset..).device_ptr(state_s.stream()).0 as *mut f32,
            output_buf.device_ptr(output_buf.stream()).0 as *mut f32,
            bh as i32,
            seq_len as i32,
            k_dim as i32,
            v_dim as i32,
            stream,
        );
    }

    // The kernel wrote state in-place via the raw pointer; rewrap
    // (state tensor's underlying CudaSlice was modified directly)

    let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
    Ok(Tensor::from((
        candle::Storage::Cuda(output_storage),
        (bh, seq_len, v_dim),
    )))
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn gated_delta_rule_recurrence_cuda(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _g: &Tensor,
    _beta: &Tensor,
    _state: &mut Tensor,
) -> Result<Tensor> {
    candle_core::bail!("gated_delta_rule_recurrence_cuda requires the cuda feature")
}

/// CUDA-accelerated causal conv1d (both update and full paths).
///
/// For update (is_update=true):
///   x: [B, conv_dim, 1]  weight: [conv_dim, kernel_size]
///   conv_state: [B, conv_dim, kernel_size] (mutated in place for update)
///   Returns: (output [B, conv_dim, 1], updated conv_state)
///
/// For full (is_update=false):
///   x: [B, conv_dim, S]  weight: [conv_dim, kernel_size]
///   Returns: (output [B, conv_dim, S], new conv_state [B, conv_dim, kernel_size])
#[cfg(feature = "cuda")]
pub fn causal_conv1d_cuda(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &Tensor,
    kernel_size: usize,
    is_update: bool,
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
        dtype_code: i32,
    ) -> Result<(Tensor, Tensor)> {
        let dev = x.device().as_cuda_device()?;
        let (batch_size, conv_dim, seq_len) = x.dims3()?;

        let (x_s, x_l) = x.storage_and_layout();
        let x_s = match &*x_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("x must be a cuda tensor"),
        };
        let x_offset = x_l.start_offset();

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

                unsafe {
                    crate::cuda::ffi::causal_conv1d_update(
                        x_s.slice(x_offset..).device_ptr(x_s.stream()).0 as *const c_void,
                        w_s.slice(w_offset..).device_ptr(w_s.stream()).0 as *const c_void,
                        cs_s.slice(cs_offset..).device_ptr(cs_s.stream()).0 as *mut c_void,
                        output_buf.device_ptr(output_buf.stream()).0 as *mut c_void,
                        batch_size as i32,
                        conv_dim as i32,
                        kernel_size as i32,
                        dtype_code,
                        stream,
                    );
                }
            }

            let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
            let output = Tensor::from((
                candle::Storage::Cuda(output_storage),
                (batch_size, conv_dim, 1usize),
            ));

            Ok((output, conv_state_new))
        } else {
            // Full path: allocate new conv_state and output
            let output_buf = unsafe { dev.alloc::<T>(batch_size * conv_dim * seq_len) }?;
            let cs_buf = unsafe { dev.alloc::<T>(batch_size * conv_dim * kernel_size) }?;

            unsafe {
                crate::cuda::ffi::causal_conv1d_full(
                    x_s.slice(x_offset..).device_ptr(x_s.stream()).0 as *const c_void,
                    w_s.slice(w_offset..).device_ptr(w_s.stream()).0 as *const c_void,
                    cs_buf.device_ptr(cs_buf.stream()).0 as *mut c_void,
                    output_buf.device_ptr(output_buf.stream()).0 as *mut c_void,
                    batch_size as i32,
                    conv_dim as i32,
                    seq_len as i32,
                    kernel_size as i32,
                    dtype_code,
                    stream,
                );
            }

            let output_storage = candle::CudaStorage::wrap_cuda_slice(output_buf, dev.clone());
            let output = Tensor::from((
                candle::Storage::Cuda(output_storage),
                (batch_size, conv_dim, seq_len),
            ));

            let cs_storage = candle::CudaStorage::wrap_cuda_slice(cs_buf, dev.clone());
            let new_conv_state = Tensor::from((
                candle::Storage::Cuda(cs_storage),
                (batch_size, conv_dim, kernel_size),
            ));

            Ok((output, new_conv_state))
        }
    }

    match x.dtype() {
        DType::F16 => cuda_fwd::<half::f16>(x, weight, conv_state, kernel_size, is_update, 0),
        DType::BF16 => cuda_fwd::<half::bf16>(x, weight, conv_state, kernel_size, is_update, 1),
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
) -> Result<(Tensor, Tensor)> {
    candle_core::bail!("causal_conv1d_cuda requires the cuda feature")
}

/// CUDA-accelerated fused GDN gating computation.
///
/// Computes: beta = sigmoid(b), g = -exp(a_log) * softplus(a + dt_bias)
///
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
