/// CUDA selective scan kernel for Mamba SSM.
///
/// Replaces the per-timestep Rust loop in `MambaLayer::forward_full` with a
/// single GPU kernel launch.
///
/// Inputs:
/// - `x`:       (batch, seq_len, n_heads, head_dim) - input after conv1d+SiLU, f32
/// - `dt`:      (batch, seq_len, n_heads) - timestep (pre-bias, pre-softplus), f32
/// - `a`:       (n_heads,) - negative exp of A_log, f32
/// - `b`:       (batch, seq_len, n_heads, d_state) - input projection (already expanded from groups), f32
/// - `c`:       (batch, seq_len, n_heads, d_state) - output projection (already expanded from groups), f32
/// - `d`:       (n_heads,) - skip connection weight, f32
/// - `dt_bias`: (n_heads,) - dt bias, f32
/// - `state`:   (batch, n_heads, head_dim, d_state) - SSM state (mutated in-place), f32
/// - `dt_min`:  minimum dt clamp value
/// - `dt_max`:  maximum dt clamp value
///
/// Returns:
/// - `y`:       (batch, seq_len, n_heads, head_dim) - output, f32
#[cfg(feature = "cuda")]
pub fn selective_scan_cuda(
    x: &candle_core::Tensor,
    dt: &candle_core::Tensor,
    a: &candle_core::Tensor,
    b: &candle_core::Tensor,
    c: &candle_core::Tensor,
    d: &candle_core::Tensor,
    dt_bias: &candle_core::Tensor,
    state: &mut candle_core::Tensor,
    dt_min: f32,
    dt_max: f32,
) -> candle_core::Result<candle_core::Tensor> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;

    let x = x.contiguous()?;
    let dt = dt.contiguous()?;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let c = c.contiguous()?;
    let d = d.contiguous()?;
    let dt_bias = dt_bias.contiguous()?;

    // x: (batch, seq_len, n_heads, head_dim)
    let (batch_size, seq_len, n_heads, head_dim) = x.dims4()?;
    let (_, d_state) = {
        // b: (batch, seq_len, n_heads, d_state)
        let dims = b.dims4()?;
        (dims.2, dims.3)
    };

    // Reshape to flat layout expected by kernel
    let x_flat = x.reshape((batch_size, seq_len, n_heads * head_dim))?;
    let b_flat = b.reshape((batch_size, seq_len, n_heads * d_state))?;
    let c_flat = c.reshape((batch_size, seq_len, n_heads * d_state))?;

    let dev = x_flat.device().as_cuda_device()?;

    // Extract device pointers
    macro_rules! get_ptr {
        ($tensor:expr) => {{
            let (s, l) = $tensor.storage_and_layout();
            let s = match &*s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("selective_scan_cuda: tensor must be on CUDA"),
            };
            let ptr = s.slice(l.start_offset()..).device_ptr(s.stream()).0 as *const f32;
            ptr
        }};
    }

    let x_ptr = get_ptr!(x_flat);
    let dt_ptr = get_ptr!(dt);
    let a_ptr = get_ptr!(a);
    let b_ptr = get_ptr!(b_flat);
    let c_ptr = get_ptr!(c_flat);
    let d_ptr = get_ptr!(d);
    let dt_bias_ptr = get_ptr!(dt_bias);

    // State is mutable
    let (state_s, state_l) = state.storage_and_layout();
    let state_s = match &*state_s {
        candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle::bail!("selective_scan_cuda: state must be on CUDA"),
    };
    let state_ptr = {
        let ptr = state_s.slice(state_l.start_offset()..).device_ptr(state_s.stream()).0 as *mut f32;
        ptr
    };
    let _ = state_s;
    let _ = state_l;

    // Allocate output
    let y_elems = batch_size * seq_len * n_heads * head_dim;
    let y_buf = unsafe { dev.alloc::<f32>(y_elems) }?;

    let stream = dev.cuda_stream().cu_stream() as i64;

    unsafe {
        crate::cuda::ffi::selective_scan_cuda(
            x_ptr,
            dt_ptr,
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr,
            dt_bias_ptr,
            state_ptr,
            {let p = y_buf.device_ptr(y_buf.stream()).0 as *mut f32; p},
            batch_size as i32,
            n_heads as i32,
            head_dim as i32,
            d_state as i32,
            seq_len as i32,
            dt_min,
            dt_max,
            stream,
        );
    }

    let y_storage = candle::CudaStorage::wrap_cuda_slice(y_buf, dev.clone());
    let y = candle_core::Tensor::from((
        candle::Storage::Cuda(y_storage),
        (batch_size, seq_len, n_heads, head_dim),
    ));

    Ok(y)
}

#[cfg(not(feature = "cuda"))]
pub fn selective_scan_cuda(
    _x: &candle_core::Tensor,
    _dt: &candle_core::Tensor,
    _a: &candle_core::Tensor,
    _b: &candle_core::Tensor,
    _c: &candle_core::Tensor,
    _d: &candle_core::Tensor,
    _dt_bias: &candle_core::Tensor,
    _state: &mut candle_core::Tensor,
    _dt_min: f32,
    _dt_max: f32,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("selective_scan_cuda requires the cuda feature")
}
