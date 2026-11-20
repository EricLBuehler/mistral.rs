use super::*;

#[cfg(feature = "cuda")]
pub(crate) fn try_cuda_rms_norm_strided_4d(
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Option<Tensor>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !input.device().is_cuda() || input.rank() != 4 {
        return Ok(None);
    }

    let dtype = input.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32) || weight.dtype() != dtype {
        return Ok(None);
    }
    if !weight.device().same_device(input.device()) {
        return Ok(None);
    }

    let (batch, heads, seq_len, head_dim) = input.dims4()?;
    if weight.dims1()? != head_dim {
        candle_core::bail!(
            "cuda_rms_norm_strided_4d weight size {} does not match head dim {head_dim}",
            weight.dims1()?
        );
    }
    if input.elem_count() == 0 {
        return Ok(None);
    }
    for (name, value) in [
        ("batch", batch),
        ("heads", heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("cuda_rms_norm_strided_4d {name} is too large: {value}");
        }
    }

    let (input_storage, input_layout) = input.storage_and_layout();
    if input_layout.is_contiguous() {
        return Ok(None);
    }
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let weight = weight.contiguous()?;
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let weight_storage = match &*weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();
    let elem_count = input.elem_count();
    let stride = input_layout.stride();
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let heads_i32 = i32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_rms_norm_strided_4d input dtype mismatch");
            };
            let CudaStorageSlice::$variant(weight_src) = &weight_storage.slice else {
                candle_core::bail!("cuda_rms_norm_strided_4d weight dtype mismatch");
            };
            let mut out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (weight_ptr, weight_guard) = weight_src.device_ptr(&stream);
            let (out_ptr, out_guard) = out.device_ptr_mut(&stream);
            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let weight_ptr =
                unsafe { (weight_ptr as *const $ty).add(weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    weight_ptr as *const c_void,
                    out_ptr as *mut c_void,
                    stride[0] as i64,
                    stride[1] as i64,
                    stride[2] as i64,
                    stride[3] as i64,
                    batch_i32,
                    heads_i32,
                    seq_len_i32,
                    head_dim_i32,
                    eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(weight_guard);
            drop(out_guard);

            let out_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(out),
                device: dev.clone(),
            };
            Ok(Some(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                shape,
            ))))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, rms_norm_strided_4d_bf16),
        DType::F16 => launch!(F16, half::f16, rms_norm_strided_4d_f16),
        DType::F32 => launch!(F32, f32, rms_norm_strided_4d_f32),
        _ => Ok(None),
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_rms_norm_residual(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    scale: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.shape() != residual.shape() {
        candle_core::bail!(
            "cuda_rms_norm_residual input/residual shape mismatch: {:?} vs {:?}",
            input.shape(),
            residual.shape()
        );
    }
    if input.dtype() != residual.dtype() || input.dtype() != weight.dtype() {
        candle_core::bail!(
            "cuda_rms_norm_residual dtype mismatch: input {:?}, residual {:?}, weight {:?}",
            input.dtype(),
            residual.dtype(),
            weight.dtype()
        );
    }
    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "cuda_rms_norm_residual only supports BF16/F16/F32, got {:?}",
            input.dtype()
        );
    }
    if !residual.device().same_device(input.device())
        || !weight.device().same_device(input.device())
    {
        candle_core::bail!("cuda_rms_norm_residual tensors must be on the same CUDA device");
    }
    if let Some(scale) = scale {
        if scale.elem_count() != 1 {
            candle_core::bail!(
                "cuda_rms_norm_residual scale must have one element, got {}",
                scale.elem_count()
            );
        }
        if scale.dtype() != input.dtype() {
            candle_core::bail!(
                "cuda_rms_norm_residual scale dtype mismatch: input {:?}, scale {:?}",
                input.dtype(),
                scale.dtype()
            );
        }
        if !scale.device().same_device(input.device()) {
            candle_core::bail!("cuda_rms_norm_residual scale must be on the same CUDA device");
        }
    }

    let ncols = input.dim(D::Minus1)?;
    if weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_rms_norm_residual weight size {} does not match last dim {ncols}",
            weight.dims1()?
        );
    }
    let elem_count = input.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_rms_norm_residual got empty input");
    }
    let nrows = elem_count / ncols;
    if nrows > i32::MAX as usize || ncols > i32::MAX as usize {
        candle_core::bail!(
            "cuda_rms_norm_residual input is too large: nrows={nrows}, ncols={ncols}"
        );
    }
    let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;
    let scale = scale.map(Tensor::contiguous).transpose()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA input"),
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let residual_storage = match &*residual_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA residual"),
    };
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let weight_storage = match &*weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA weight"),
    };
    let scale_storage_and_layout = scale.as_ref().map(|scale| scale.storage_and_layout());

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual input dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_src) = &residual_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual residual dtype mismatch");
            };
            let CudaStorageSlice::$variant(weight_src) = &weight_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual weight dtype mismatch");
            };
            let (scale_ptr, scale_guard) =
                if let Some((scale_storage, scale_layout)) = &scale_storage_and_layout {
                    let scale_storage = match &**scale_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => candle_core::bail!("cuda_rms_norm_residual requires CUDA scale"),
                    };
                    let CudaStorageSlice::$variant(scale_src) = &scale_storage.slice else {
                        candle_core::bail!("cuda_rms_norm_residual scale dtype mismatch");
                    };
                    let (scale_ptr, scale_guard) = scale_src.device_ptr(&stream);
                    (
                        unsafe { (scale_ptr as *const $ty).add(scale_layout.start_offset()) }
                            as *const c_void,
                        Some(scale_guard),
                    )
                } else {
                    (std::ptr::null(), None)
                };

            let mut out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (residual_ptr, residual_guard) = residual_src.device_ptr(&stream);
            let (weight_ptr, weight_guard) = weight_src.device_ptr(&stream);
            let (out_ptr, out_guard) = out.device_ptr_mut(&stream);

            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let residual_ptr =
                unsafe { (residual_ptr as *const $ty).add(residual_layout.start_offset()) };
            let weight_ptr =
                unsafe { (weight_ptr as *const $ty).add(weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    residual_ptr as *const c_void,
                    weight_ptr as *const c_void,
                    scale_ptr,
                    out_ptr as *mut c_void,
                    nrows_i32,
                    ncols_i32,
                    eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(residual_guard);
            drop(weight_guard);
            drop(scale_guard);
            drop(out_guard);

            let out_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(out),
                device: dev.clone(),
            };
            Ok(Tensor::from((
                candle_core::Storage::Cuda(out_storage),
                shape,
            )))
        }};
    }

    match input.dtype() {
        DType::BF16 => launch!(BF16, half::bf16, rms_norm_residual_bf16),
        DType::F16 => launch!(F16, half::f16, rms_norm_residual_f16),
        DType::F32 => launch!(F32, f32, rms_norm_residual_f32),
        dtype => candle_core::bail!("cuda_rms_norm_residual unsupported dtype {dtype:?}"),
    }
}

#[cfg(feature = "cuda")]
pub fn cuda_add_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.shape() != residual.shape() {
        candle_core::bail!(
            "cuda_add_rms_norm input/residual shape mismatch: {:?} vs {:?}",
            input.shape(),
            residual.shape()
        );
    }
    if input.dtype() != residual.dtype() || input.dtype() != weight.dtype() {
        candle_core::bail!(
            "cuda_add_rms_norm dtype mismatch: input {:?}, residual {:?}, weight {:?}",
            input.dtype(),
            residual.dtype(),
            weight.dtype()
        );
    }
    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "cuda_add_rms_norm only supports BF16/F16/F32, got {:?}",
            input.dtype()
        );
    }
    if !residual.device().same_device(input.device())
        || !weight.device().same_device(input.device())
    {
        candle_core::bail!("cuda_add_rms_norm tensors must be on the same CUDA device");
    }

    let ncols = input.dim(D::Minus1)?;
    if weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_add_rms_norm weight size {} does not match last dim {ncols}",
            weight.dims1()?
        );
    }
    let elem_count = input.elem_count();
    if ncols == 0 || elem_count == 0 {
        candle_core::bail!("cuda_add_rms_norm got empty input");
    }
    let nrows = elem_count / ncols;
    if nrows > i32::MAX as usize || ncols > i32::MAX as usize {
        candle_core::bail!("cuda_add_rms_norm input is too large: nrows={nrows}, ncols={ncols}");
    }
    let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("cuda_add_rms_norm requires CUDA input"),
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let residual_storage = match &*residual_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("cuda_add_rms_norm requires CUDA residual"),
    };
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let weight_storage = match &*weight_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => candle_core::bail!("cuda_add_rms_norm requires CUDA weight"),
    };

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_add_rms_norm input dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_src) = &residual_storage.slice else {
                candle_core::bail!("cuda_add_rms_norm residual dtype mismatch");
            };
            let CudaStorageSlice::$variant(weight_src) = &weight_storage.slice else {
                candle_core::bail!("cuda_add_rms_norm weight dtype mismatch");
            };

            let mut residual_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let mut norm_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (residual_ptr, residual_guard) = residual_src.device_ptr(&stream);
            let (weight_ptr, weight_guard) = weight_src.device_ptr(&stream);
            let (residual_out_ptr, residual_out_guard) = residual_out.device_ptr_mut(&stream);
            let (norm_out_ptr, norm_out_guard) = norm_out.device_ptr_mut(&stream);

            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let residual_ptr =
                unsafe { (residual_ptr as *const $ty).add(residual_layout.start_offset()) };
            let weight_ptr =
                unsafe { (weight_ptr as *const $ty).add(weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    residual_ptr as *const c_void,
                    weight_ptr as *const c_void,
                    residual_out_ptr as *mut c_void,
                    norm_out_ptr as *mut c_void,
                    nrows_i32,
                    ncols_i32,
                    eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(residual_guard);
            drop(weight_guard);
            drop(residual_out_guard);
            drop(norm_out_guard);

            let residual_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(residual_out),
                device: dev.clone(),
            };
            let norm_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(norm_out),
                device: dev.clone(),
            };
            Ok((
                Tensor::from((candle_core::Storage::Cuda(residual_storage), shape.clone())),
                Tensor::from((candle_core::Storage::Cuda(norm_storage), shape)),
            ))
        }};
    }

    match input.dtype() {
        DType::BF16 => launch!(BF16, half::bf16, add_rms_norm_bf16),
        DType::F16 => launch!(F16, half::f16, add_rms_norm_f16),
        DType::F32 => launch!(F32, f32, add_rms_norm_f32),
        dtype => candle_core::bail!("cuda_add_rms_norm unsupported dtype {dtype:?}"),
    }
}

#[cfg(feature = "metal")]
pub fn metal_rms_norm_residual(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    scale: Option<&Tensor>,
    eps: f32,
) -> Result<Option<Tensor>> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    if input.shape() != residual.shape() {
        return Ok(None);
    }
    let n_cols = input.dim(D::Minus1)?;
    if weight.dims1()? != n_cols {
        return Ok(None);
    }
    let n_rows = input.elem_count() / n_cols;
    if n_rows == 0 {
        return Ok(None);
    }
    if let Some(scale) = scale {
        if scale.elem_count() != 1 {
            return Ok(None);
        }
    }

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;
    let scale_t = scale.map(Tensor::contiguous).transpose()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let Storage::Metal(input_storage) = &*input_storage else {
        return Ok(None);
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let Storage::Metal(residual_storage) = &*residual_storage else {
        return Ok(None);
    };
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let Storage::Metal(weight_storage) = &*weight_storage else {
        return Ok(None);
    };
    let scale_storage_and_layout = scale_t.as_ref().map(|s| s.storage_and_layout());
    let scale_metal = match scale_storage_and_layout.as_ref() {
        Some((s, l)) => {
            let Storage::Metal(s) = &**s else {
                return Ok(None);
            };
            Some((s, l))
        }
        None => None,
    };

    let device = input_storage.device().clone();
    let dtype = input.dtype();
    let out_buf = device.new_buffer(input.elem_count(), dtype, "rmsnorm-residual-out")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("rmsnorm-residual");

    let x_offset = input_layout.start_offset() * dtype.size_in_bytes();
    let res_offset = residual_layout.start_offset() * dtype.size_in_bytes();
    let w_offset = weight_layout.start_offset() * dtype.size_in_bytes();
    let scale_arg = scale_metal
        .as_ref()
        .map(|(s, l)| (s.buffer(), l.start_offset() * dtype.size_in_bytes()));

    mistralrs_quant::metal_kernels::call_rmsnorm_residual(
        device.device(),
        &encoder,
        &mistralrs_quant::metal_kernels::Kernels::new(),
        dtype,
        (input_storage.buffer(), x_offset),
        (residual_storage.buffer(), res_offset),
        (weight_storage.buffer(), w_offset),
        scale_arg,
        &out_buf,
        n_cols,
        n_rows,
        eps,
    )
    .map_err(candle_core::Error::wrap)?;

    let out = Tensor::from((
        Storage::Metal(MetalStorage::new(
            out_buf,
            device.clone(),
            input.elem_count(),
            dtype,
        )),
        Shape::from(input.dims()),
    ));
    Ok(Some(out))
}

/// Fused pre-norm residual on Metal: returns `(x + residual, rms_norm(x + residual) * weight)`.
/// Callers check shapes and dtypes; see `layers::fused_add_rms_norm_applies`.
#[cfg(feature = "metal")]
pub fn metal_add_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    let n_cols = input.dim(D::Minus1)?;
    let n_rows = input.elem_count() / n_cols;
    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let weight = weight.contiguous()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let (weight_storage, weight_layout) = weight.storage_and_layout();
    let (
        Storage::Metal(input_storage),
        Storage::Metal(residual_storage),
        Storage::Metal(weight_storage),
    ) = (&*input_storage, &*residual_storage, &*weight_storage)
    else {
        candle_core::bail!("metal_add_rms_norm expects Metal tensors");
    };

    let device = input_storage.device().clone();
    let dtype = input.dtype();
    let elem_size = dtype.size_in_bytes();
    let sum_buf = device.new_buffer(input.elem_count(), dtype, "add-rmsnorm-sum")?;
    let norm_buf = device.new_buffer(input.elem_count(), dtype, "add-rmsnorm-norm")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("add-rmsnorm");
    mistralrs_quant::metal_kernels::call_add_rmsnorm(
        device.device(),
        &encoder,
        &mistralrs_quant::metal_kernels::Kernels::new(),
        dtype,
        (
            input_storage.buffer(),
            input_layout.start_offset() * elem_size,
        ),
        (
            residual_storage.buffer(),
            residual_layout.start_offset() * elem_size,
        ),
        (
            weight_storage.buffer(),
            weight_layout.start_offset() * elem_size,
        ),
        &sum_buf,
        &norm_buf,
        n_cols,
        n_rows,
        eps,
    )
    .map_err(candle_core::Error::wrap)?;
    drop(encoder);

    let to_tensor = |buf| {
        Tensor::from((
            Storage::Metal(MetalStorage::new(
                buf,
                device.clone(),
                input.elem_count(),
                dtype,
            )),
            Shape::from(input.dims()),
        ))
    };
    Ok((to_tensor(sum_buf), to_tensor(norm_buf)))
}

/// Shape of a fused per-head RMSNorm + RoPE call, from [`qk_norm_rope_shape`].
#[derive(Clone, Copy, Debug)]
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
pub(crate) struct QkNormRopeShape {
    pub batch: usize,
    pub q_heads: usize,
    pub k_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
    pub rot_dim: usize,
}

/// The one eligibility check for the fused QK(V)-norm + RoPE kernels on every GPU backend: `Ok(None)` means
/// fall back to separate ops. `positions` holds one cos/sin row index per token.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qk_norm_rope_shape(
    q: &Tensor,
    k: Option<&Tensor>,
    q_weight: &Tensor,
    k_weight: Option<&Tensor>,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
) -> Result<Option<QkNormRopeShape>> {
    let dtype = q.dtype();
    let device = q.device();
    let on_device = |t: &Tensor| t.device().same_device(device);
    if !(device.is_cuda() || device.is_metal())
        || !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || [q_weight, cos, sin]
            .iter()
            .any(|t| t.dtype() != dtype || !on_device(t))
        || k.into_iter()
            .chain(k_weight)
            .any(|t| t.dtype() != dtype || !on_device(t))
        || positions.dtype() != DType::U32
        || !on_device(positions)
    {
        return Ok(None);
    }
    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    if positions.dims1()? != batch * seq_len {
        candle_core::bail!(
            "positions length {} does not match token count {}",
            positions.dims1()?,
            batch * seq_len
        );
    }
    let k_heads = match (k, k_weight) {
        (Some(k), Some(k_weight)) => {
            let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
            if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim)
                || k_weight.dims1()? != head_dim
            {
                return Ok(None);
            }
            k_heads
        }
        (None, None) => 0,
        _ => return Ok(None),
    };
    let (cos_rows, rot_dim) = cos.dims2()?;
    if q_weight.dims1()? != head_dim
        || sin.dims2()? != (cos_rows, rot_dim)
        || rot_dim == 0
        || rot_dim * 2 > head_dim
        || [batch, q_heads, k_heads, seq_len, head_dim]
            .iter()
            .any(|&d| d > i32::MAX as usize)
    {
        return Ok(None);
    }
    Ok(Some(QkNormRopeShape {
        batch,
        q_heads,
        k_heads,
        seq_len,
        head_dim,
        rot_dim,
    }))
}

/// Fused RMSNorm + RoPE of q and (optionally) k, on whichever GPU backend holds them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fused_qk_rms_norm_rope_positions(
    q: &Tensor,
    k: Option<&Tensor>,
    q_weight: &Tensor,
    k_weight: Option<&Tensor>,
    q_eps: f32,
    k_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    let Some(shape) = qk_norm_rope_shape(q, k, q_weight, k_weight, cos, sin, positions)? else {
        return Ok(None);
    };
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        let _ = shape;
        return try_cuda_qk_rms_norm_rope_positions(
            q, k, q_weight, k_weight, q_eps, k_eps, cos, sin, positions, is_neox,
        );
    }
    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        let (q, k, _) = metal_qk_norm_rope(MetalQkNormRope {
            shape,
            q: (q, q_weight, q_eps),
            k: k.zip(k_weight).map(|(k, w)| (k, w, k_eps)),
            v: None,
            cos,
            sin,
            positions,
            is_neox,
            token_major_output: true,
        })?;
        return Ok(Some((q, k)));
    }
    let _ = (shape, q_eps, k_eps, is_neox);
    Ok(None)
}

/// Fused RMSNorm + RoPE of q and k plus RMSNorm of v, on whichever GPU backend holds them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_fused_qkv_rms_norm_rope_positions(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    q_eps: f32,
    k_eps: f32,
    v_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    let Some(shape) =
        qk_norm_rope_shape(q, Some(k), q_weight, Some(k_weight), cos, sin, positions)?
    else {
        return Ok(None);
    };
    if v.dims4()? != k.dims4()?
        || v.dtype() != q.dtype()
        || v_weight.dtype() != q.dtype()
        || v_weight.dims1()? != shape.head_dim
        || !v.device().same_device(q.device())
    {
        return Ok(None);
    }
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        return try_cuda_qkv_rms_norm_rope_positions(
            q, k, v, q_weight, k_weight, v_weight, q_eps, k_eps, v_eps, cos, sin, positions,
            is_neox,
        );
    }
    #[cfg(feature = "metal")]
    if q.device().is_metal() {
        let (q, k, v) = metal_qk_norm_rope(MetalQkNormRope {
            shape,
            q: (q, q_weight, q_eps),
            k: Some((k, k_weight, k_eps)),
            v: Some((v, v_weight, v_eps)),
            cos,
            sin,
            positions,
            is_neox,
            token_major_output: false,
        })?;
        return Ok(Some((
            q,
            k.expect("k was requested"),
            v.expect("v was requested"),
        )));
    }
    let _ = (q_eps, k_eps, v_eps, is_neox);
    Ok(None)
}

#[cfg(feature = "metal")]
pub(super) struct MetalQkNormRope<'a> {
    shape: QkNormRopeShape,
    q: (&'a Tensor, &'a Tensor, f32),
    k: Option<(&'a Tensor, &'a Tensor, f32)>,
    v: Option<(&'a Tensor, &'a Tensor, f32)>,
    cos: &'a Tensor,
    sin: &'a Tensor,
    positions: &'a Tensor,
    is_neox: bool,
    /// Write token-major (and return transposed views) when q already arrives token-major.
    token_major_output: bool,
}

#[cfg(feature = "metal")]
pub(super) type MetalQkNormRopeOutput = (Tensor, Option<Tensor>, Option<Tensor>);

#[cfg(feature = "metal")]
pub(super) fn metal_qk_norm_rope(args: MetalQkNormRope<'_>) -> Result<MetalQkNormRopeOutput> {
    use candle_core::{backend::BackendStorage, MetalDevice, MetalStorage, Shape, Storage};
    use mistralrs_quant::metal_kernels::{QkNormRopeBuffers, QkNormRopeParams};

    let MetalQkNormRope {
        shape,
        q: (q, q_weight, q_eps),
        k,
        v,
        cos,
        sin,
        positions,
        is_neox,
        token_major_output,
    } = args;
    let head_dim = shape.head_dim;
    let token_major = token_major_output && q.stride()[1] == head_dim && q.stride()[3] == 1;
    let (cos, sin, positions) = (
        cos.contiguous()?,
        sin.contiguous()?,
        positions.contiguous()?,
    );
    let q_weight = q_weight.contiguous()?;
    let k_weight = k.map(|(_, w, _)| w.contiguous()).transpose()?;
    let v_weight = v.map(|(_, w, _)| w.contiguous()).transpose()?;
    let v_heads = v.map(|(v, _, _)| v.dim(1)).transpose()?.unwrap_or(0);
    let stride4 = |t: &Tensor| -> [i64; 4] {
        let s = t.stride();
        [s[0] as i64, s[1] as i64, s[2] as i64, s[3] as i64]
    };

    let buffer_of =
        |t: &Tensor| -> Result<(candle_metal_kernels::metal::Buffer, usize, MetalDevice)> {
            let (storage, layout) = t.storage_and_layout();
            match &*storage {
                Storage::Metal(m) => Ok((
                    m.buffer().clone(),
                    layout.start_offset() * m.dtype().size_in_bytes(),
                    m.device().clone(),
                )),
                _ => candle_core::bail!("fused qk norm rope expects Metal tensors"),
            }
        };
    let (q_buf, q_off, device) = buffer_of(q)?;
    let k_buf = k.map(|(k, _, _)| buffer_of(k)).transpose()?;
    let v_buf = v.map(|(v, _, _)| buffer_of(v)).transpose()?;
    let (qw_buf, qw_off, _) = buffer_of(&q_weight)?;
    let kw_buf = k_weight.as_ref().map(buffer_of).transpose()?;
    let vw_buf = v_weight.as_ref().map(buffer_of).transpose()?;
    let (cos_buf, cos_off, _) = buffer_of(&cos)?;
    let (sin_buf, sin_off, _) = buffer_of(&sin)?;
    let (pos_buf, pos_off, _) = buffer_of(&positions)?;

    let dtype = q.dtype();
    let q_count = q.elem_count();
    let k_count = k.map(|(k, _, _)| k.elem_count()).unwrap_or(0);
    let v_count = v.map(|(v, _, _)| v.elem_count()).unwrap_or(0);
    let q_out = device.new_buffer(q_count, dtype, "qk-norm-rope-q")?;
    let k_out = (k_count > 0)
        .then(|| device.new_buffer(k_count, dtype, "qk-norm-rope-k"))
        .transpose()?;
    let v_out = (v_count > 0)
        .then(|| device.new_buffer(v_count, dtype, "qk-norm-rope-v"))
        .transpose()?;

    let to_i32 = |value: usize| i32::try_from(value).map_err(candle_core::Error::wrap);
    let params = QkNormRopeParams {
        q_stride: stride4(q),
        k_stride: k.map(|(k, _, _)| stride4(k)).unwrap_or_default(),
        v_stride: v.map(|(v, _, _)| stride4(v)).unwrap_or_default(),
        batch: to_i32(shape.batch)?,
        q_heads: to_i32(shape.q_heads)?,
        k_heads: to_i32(shape.k_heads)?,
        v_heads: to_i32(v_heads)?,
        seq_len: to_i32(shape.seq_len)?,
        head_dim: to_i32(head_dim)?,
        rot_dim: to_i32(shape.rot_dim)?,
        q_eps,
        k_eps: k.map(|(_, _, e)| e).unwrap_or(q_eps),
        v_eps: v.map(|(_, _, e)| e).unwrap_or(q_eps),
        token_major: i32::from(token_major),
        use_positions: 1,
        cos_batch_stride: 0,
    };
    {
        let encoder = device.command_encoder()?;
        encoder.set_label("qk-norm-rope");
        mistralrs_quant::metal_kernels::call_qk_norm_rope(
            device.device(),
            &encoder,
            &mistralrs_quant::metal_kernels::Kernels::new(),
            dtype,
            is_neox,
            QkNormRopeBuffers {
                q: (&q_buf, q_off),
                k: k_buf.as_ref().map(|(b, off, _)| (b, *off)),
                v: v_buf.as_ref().map(|(b, off, _)| (b, *off)),
                q_weight: (&qw_buf, qw_off),
                k_weight: kw_buf.as_ref().map(|(b, off, _)| (b, *off)),
                v_weight: vw_buf.as_ref().map(|(b, off, _)| (b, *off)),
                cos: (&cos_buf, cos_off),
                sin: (&sin_buf, sin_off),
                positions: Some((&pos_buf, pos_off)),
                q_out: &q_out,
                k_out: k_out.as_deref(),
                v_out: v_out.as_deref(),
            },
            &params,
        )
        .map_err(candle_core::Error::wrap)?;
    }

    let (batch, seq_len) = (shape.batch, shape.seq_len);
    let wrap = |buf, heads: usize, count: usize, token_major: bool| -> Result<Tensor> {
        let dims = if token_major {
            [batch, seq_len, heads, head_dim]
        } else {
            [batch, heads, seq_len, head_dim]
        };
        let t = Tensor::from((
            Storage::Metal(MetalStorage::new(buf, device.clone(), count, dtype)),
            Shape::from_dims(&dims),
        ));
        if token_major {
            t.transpose(1, 2)
        } else {
            Ok(t)
        }
    };
    let q_t = wrap(q_out, shape.q_heads, q_count, token_major)?;
    let k_t = k_out
        .map(|b| wrap(b, shape.k_heads, k_count, token_major))
        .transpose()?;
    let v_t = v_out
        .map(|b| wrap(b, v_heads, v_count, false))
        .transpose()?;
    Ok((q_t, k_t, v_t))
}

#[cfg(feature = "cuda")]
pub fn cuda_rms_norm_residual_then_rms_norm(
    input: &Tensor,
    residual: &Tensor,
    residual_weight: &Tensor,
    scale: Option<&Tensor>,
    norm_weight: &Tensor,
    residual_eps: f32,
    norm_eps: f32,
) -> Result<(Tensor, Tensor)> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if input.shape() != residual.shape() {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm input/residual shape mismatch: {:?} vs {:?}",
            input.shape(),
            residual.shape()
        );
    }
    if input.dtype() != residual.dtype()
        || input.dtype() != residual_weight.dtype()
        || input.dtype() != norm_weight.dtype()
    {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm dtype mismatch: input {:?}, residual {:?}, residual_weight {:?}, norm_weight {:?}",
            input.dtype(),
            residual.dtype(),
            residual_weight.dtype(),
            norm_weight.dtype()
        );
    }
    if !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm only supports BF16/F16/F32, got {:?}",
            input.dtype()
        );
    }
    if !residual.device().same_device(input.device())
        || !residual_weight.device().same_device(input.device())
        || !norm_weight.device().same_device(input.device())
    {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm tensors must be on the same CUDA device"
        );
    }
    if let Some(scale) = scale {
        if scale.elem_count() != 1 {
            candle_core::bail!(
                "cuda_rms_norm_residual_then_rms_norm scale must have one element, got {}",
                scale.elem_count()
            );
        }
        if scale.dtype() != input.dtype() {
            candle_core::bail!(
                "cuda_rms_norm_residual_then_rms_norm scale dtype mismatch: input {:?}, scale {:?}",
                input.dtype(),
                scale.dtype()
            );
        }
        if !scale.device().same_device(input.device()) {
            candle_core::bail!(
                "cuda_rms_norm_residual_then_rms_norm scale must be on the same CUDA device"
            );
        }
    }

    let ncols = input.dim(D::Minus1)?;
    if residual_weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm residual weight size {} does not match last dim {ncols}",
            residual_weight.dims1()?
        );
    }
    if norm_weight.dims1()? != ncols {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm norm weight size {} does not match last dim {ncols}",
            norm_weight.dims1()?
        );
    }
    let elem_count = input.elem_count();
    if elem_count == 0 {
        candle_core::bail!("cuda_rms_norm_residual_then_rms_norm got empty input");
    }
    let nrows = elem_count / ncols;
    if nrows > i32::MAX as usize || ncols > i32::MAX as usize {
        candle_core::bail!(
            "cuda_rms_norm_residual_then_rms_norm input is too large: nrows={nrows}, ncols={ncols}"
        );
    }
    let nrows_i32 = i32::try_from(nrows).map_err(candle_core::Error::wrap)?;
    let ncols_i32 = i32::try_from(ncols).map_err(candle_core::Error::wrap)?;

    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let residual_weight = residual_weight.contiguous()?;
    let norm_weight = norm_weight.contiguous()?;
    let scale = scale.map(Tensor::contiguous).transpose()?;

    let (input_storage, input_layout) = input.storage_and_layout();
    let input_storage = match &*input_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA input"),
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let residual_storage = match &*residual_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA residual"),
    };
    let (residual_weight_storage, residual_weight_layout) = residual_weight.storage_and_layout();
    let residual_weight_storage = match &*residual_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => {
            candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA residual weight")
        }
    };
    let (norm_weight_storage, norm_weight_layout) = norm_weight.storage_and_layout();
    let norm_weight_storage = match &*norm_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => candle_core::bail!("cuda_rms_norm_residual_then_rms_norm requires CUDA norm weight"),
    };
    let scale_storage_and_layout = scale.as_ref().map(|scale| scale.storage_and_layout());

    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let shape = input.shape().clone();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi_fn:ident) => {{
            let CudaStorageSlice::$variant(src) = &input_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual_then_rms_norm input dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_src) = &residual_storage.slice else {
                candle_core::bail!("cuda_rms_norm_residual_then_rms_norm residual dtype mismatch");
            };
            let CudaStorageSlice::$variant(residual_weight_src) = &residual_weight_storage.slice
            else {
                candle_core::bail!(
                    "cuda_rms_norm_residual_then_rms_norm residual weight dtype mismatch"
                );
            };
            let CudaStorageSlice::$variant(norm_weight_src) = &norm_weight_storage.slice else {
                candle_core::bail!(
                    "cuda_rms_norm_residual_then_rms_norm norm weight dtype mismatch"
                );
            };
            let (scale_ptr, scale_guard) = if let Some((scale_storage, scale_layout)) =
                &scale_storage_and_layout
            {
                let scale_storage = match &**scale_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => candle_core::bail!(
                        "cuda_rms_norm_residual_then_rms_norm requires CUDA scale"
                    ),
                };
                let CudaStorageSlice::$variant(scale_src) = &scale_storage.slice else {
                    candle_core::bail!("cuda_rms_norm_residual_then_rms_norm scale dtype mismatch");
                };
                let (scale_ptr, scale_guard) = scale_src.device_ptr(&stream);
                (
                    unsafe { (scale_ptr as *const $ty).add(scale_layout.start_offset()) }
                        as *const c_void,
                    Some(scale_guard),
                )
            } else {
                (std::ptr::null(), None)
            };

            let mut residual_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let mut norm_out = unsafe { dev.alloc::<$ty>(elem_count) }?;
            let (src_ptr, src_guard) = src.device_ptr(&stream);
            let (residual_ptr, residual_guard) = residual_src.device_ptr(&stream);
            let (residual_weight_ptr, residual_weight_guard) =
                residual_weight_src.device_ptr(&stream);
            let (norm_weight_ptr, norm_weight_guard) = norm_weight_src.device_ptr(&stream);
            let (residual_out_ptr, residual_out_guard) = residual_out.device_ptr_mut(&stream);
            let (norm_out_ptr, norm_out_guard) = norm_out.device_ptr_mut(&stream);

            let src_ptr = unsafe { (src_ptr as *const $ty).add(input_layout.start_offset()) };
            let residual_ptr =
                unsafe { (residual_ptr as *const $ty).add(residual_layout.start_offset()) };
            let residual_weight_ptr = unsafe {
                (residual_weight_ptr as *const $ty).add(residual_weight_layout.start_offset())
            };
            let norm_weight_ptr =
                unsafe { (norm_weight_ptr as *const $ty).add(norm_weight_layout.start_offset()) };

            unsafe {
                ffi::$ffi_fn(
                    src_ptr as *const c_void,
                    residual_ptr as *const c_void,
                    residual_weight_ptr as *const c_void,
                    scale_ptr,
                    norm_weight_ptr as *const c_void,
                    residual_out_ptr as *mut c_void,
                    norm_out_ptr as *mut c_void,
                    nrows_i32,
                    ncols_i32,
                    residual_eps,
                    norm_eps,
                    stream_ptr,
                );
            }

            drop(src_guard);
            drop(residual_guard);
            drop(residual_weight_guard);
            drop(norm_weight_guard);
            drop(scale_guard);
            drop(residual_out_guard);
            drop(norm_out_guard);

            let residual_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(residual_out),
                device: dev.clone(),
            };
            let norm_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(norm_out),
                device: dev.clone(),
            };
            Ok((
                Tensor::from((candle_core::Storage::Cuda(residual_storage), shape.clone())),
                Tensor::from((candle_core::Storage::Cuda(norm_storage), shape)),
            ))
        }};
    }

    match input.dtype() {
        DType::BF16 => launch!(BF16, half::bf16, rms_norm_residual_then_rms_norm_bf16),
        DType::F16 => launch!(F16, half::f16, rms_norm_residual_then_rms_norm_f16),
        DType::F32 => launch!(F32, f32, rms_norm_residual_then_rms_norm_f32),
        dtype => {
            candle_core::bail!("cuda_rms_norm_residual_then_rms_norm unsupported dtype {dtype:?}")
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QkRopeOutputLayout {
    HeadsFirst,
    TokensFirst,
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_cuda_qk_rms_norm_rope(
    q: &Tensor,
    k: Option<&Tensor>,
    q_weight: &Tensor,
    k_weight: Option<&Tensor>,
    q_eps: f32,
    k_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    is_neox: bool,
    output_layout: QkRopeOutputLayout,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !q.device().is_cuda() {
        return Ok(None);
    }

    let dtype = q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || q_weight.dtype() != dtype
        || k_weight.is_some_and(|weight| weight.dtype() != dtype)
        || cos.dtype() != dtype
        || sin.dtype() != dtype
    {
        return Ok(None);
    }

    if !q_weight.device().same_device(q.device())
        || !cos.device().same_device(q.device())
        || !sin.device().same_device(q.device())
        || k.is_some_and(|k| !k.device().same_device(q.device()) || k.dtype() != dtype)
        || k_weight.is_some_and(|weight| !weight.device().same_device(q.device()))
    {
        return Ok(None);
    }

    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    if seq_len == 1 && q.is_contiguous() && k.is_none_or(Tensor::is_contiguous) {
        return Ok(None);
    }

    let (k_heads, k_elem_count) = if let Some(k) = k {
        let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
        if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
            candle_core::bail!(
                "q/k shape mismatch for fused qk norm rope: {:?} vs {:?}",
                q.shape(),
                k.shape()
            );
        }
        let Some(k_weight) = k_weight else {
            candle_core::bail!("missing k norm weight for fused qk norm rope");
        };
        if k_weight.dims1()? != head_dim {
            candle_core::bail!(
                "k norm weight size {} does not match head dim {head_dim}",
                k_weight.dims1()?
            );
        }
        (k_heads, k.elem_count())
    } else {
        (0, 0)
    };

    if q_weight.dims1()? != head_dim {
        candle_core::bail!(
            "q norm weight size {} does not match head dim {head_dim}",
            q_weight.dims1()?
        );
    }

    let (cos_rows, rot_dim) = cos.dims2()?;
    if sin.dims2()? != (cos_rows, rot_dim) {
        candle_core::bail!(
            "cos/sin shape mismatch for fused qk norm rope: {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        return Ok(None);
    }

    let cos_batch_stride = if cos_rows == seq_len {
        0
    } else if cos_rows == batch * seq_len {
        seq_len
    } else {
        candle_core::bail!(
            "cos/sin rows {cos_rows} do not match seq_len {seq_len} or batch*seq_len {}",
            batch * seq_len
        );
    };

    for (name, value) in [
        ("batch", batch),
        ("q_heads", q_heads),
        ("k_heads", k_heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
        ("rot_dim", rot_dim),
        ("cos_batch_stride", cos_batch_stride),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("fused qk norm rope {name} is too large: {value}");
        }
    }
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let q_heads_i32 = i32::try_from(q_heads).map_err(candle_core::Error::wrap)?;
    let k_heads_i32 = i32::try_from(k_heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
    let rot_dim_i32 = i32::try_from(rot_dim).map_err(candle_core::Error::wrap)?;
    let cos_batch_stride_i32 = i32::try_from(cos_batch_stride).map_err(candle_core::Error::wrap)?;

    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let q_weight = q_weight.contiguous()?;
    let k_weight = k_weight.map(Tensor::contiguous).transpose()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let q_storage = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_storage_and_layout = k.map(Tensor::storage_and_layout);
    let (q_weight_storage, q_weight_layout) = q_weight.storage_and_layout();
    let q_weight_storage = match &*q_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_weight_storage_and_layout = k_weight.as_ref().map(Tensor::storage_and_layout);
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let cos_storage = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let sin_storage = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };

    let dev = q_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let q_shape = match output_layout {
        QkRopeOutputLayout::HeadsFirst => Shape::from_dims(&[batch, q_heads, seq_len, head_dim]),
        QkRopeOutputLayout::TokensFirst => Shape::from_dims(&[batch, seq_len, q_heads, head_dim]),
    };
    let k_shape = match output_layout {
        QkRopeOutputLayout::HeadsFirst => Shape::from_dims(&[batch, k_heads, seq_len, head_dim]),
        QkRopeOutputLayout::TokensFirst => Shape::from_dims(&[batch, seq_len, k_heads, head_dim]),
    };
    let q_elem_count = q.elem_count();

    let q_stride = q_layout.stride();
    let k_stride = k_storage_and_layout
        .as_ref()
        .map(|(_, layout)| layout.stride())
        .unwrap_or(&[0, 0, 0, 0]);

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let CudaStorageSlice::$variant(q_src) = &q_storage.slice else {
                candle_core::bail!("fused qk norm rope q dtype mismatch");
            };
            let CudaStorageSlice::$variant(q_weight_src) = &q_weight_storage.slice else {
                candle_core::bail!("fused qk norm rope q weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(cos_src) = &cos_storage.slice else {
                candle_core::bail!("fused qk norm rope cos dtype mismatch");
            };
            let CudaStorageSlice::$variant(sin_src) = &sin_storage.slice else {
                candle_core::bail!("fused qk norm rope sin dtype mismatch");
            };

            let mut q_out_buf = unsafe { dev.alloc::<$ty>(q_elem_count) }?;
            let mut k_out_buf = if k_elem_count == 0 {
                None
            } else {
                Some(unsafe { dev.alloc::<$ty>(k_elem_count) }?)
            };

            let (q_ptr, q_guard) = q_src.device_ptr(&stream);
            let q_ptr = unsafe { (q_ptr as *const $ty).add(q_layout.start_offset()) };
            let (q_weight_ptr, q_weight_guard) = q_weight_src.device_ptr(&stream);
            let q_weight_ptr =
                unsafe { (q_weight_ptr as *const $ty).add(q_weight_layout.start_offset()) };
            let (cos_ptr, cos_guard) = cos_src.device_ptr(&stream);
            let cos_ptr = unsafe { (cos_ptr as *const $ty).add(cos_layout.start_offset()) };
            let (sin_ptr, sin_guard) = sin_src.device_ptr(&stream);
            let sin_ptr = unsafe { (sin_ptr as *const $ty).add(sin_layout.start_offset()) };

            let mut k_guard = None;
            let k_ptr = if let Some((k_storage, k_layout)) = &k_storage_and_layout {
                let k_storage = match &**k_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => return Ok(None),
                };
                let CudaStorageSlice::$variant(k_src) = &k_storage.slice else {
                    candle_core::bail!("fused qk norm rope k dtype mismatch");
                };
                let (ptr, guard) = k_src.device_ptr(&stream);
                k_guard = Some(guard);
                unsafe { (ptr as *const $ty).add(k_layout.start_offset()) }
            } else {
                std::ptr::null()
            };

            let mut k_weight_guard = None;
            let k_weight_ptr =
                if let Some((k_weight_storage, k_weight_layout)) = &k_weight_storage_and_layout {
                    let k_weight_storage = match &**k_weight_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => return Ok(None),
                    };
                    let CudaStorageSlice::$variant(k_weight_src) = &k_weight_storage.slice else {
                        candle_core::bail!("fused qk norm rope k weight dtype mismatch");
                    };
                    let (ptr, guard) = k_weight_src.device_ptr(&stream);
                    k_weight_guard = Some(guard);
                    unsafe { (ptr as *const $ty).add(k_weight_layout.start_offset()) }
                } else {
                    q_weight_ptr
                };

            let (q_out_ptr, q_out_guard) = q_out_buf.device_ptr_mut(&stream);
            let mut k_out_guard = None;
            let k_out_ptr = if let Some(k_out_buf) = &mut k_out_buf {
                let (ptr, guard) = k_out_buf.device_ptr_mut(&stream);
                k_out_guard = Some(guard);
                ptr as *mut $ty
            } else {
                std::ptr::null_mut()
            };

            crate::cuda::check_launch("qk_rms_norm_rope", unsafe {
                ffi::qk_rms_norm_rope(&crate::cuda::abi::QkRmsNormRopeParams {
                    q: q_ptr as *const c_void,
                    k: k_ptr as *const c_void,
                    q_weight: q_weight_ptr as *const c_void,
                    k_weight: k_weight_ptr as *const c_void,
                    cos: cos_ptr as *const c_void,
                    sin: sin_ptr as *const c_void,
                    q_out: q_out_ptr as *mut c_void,
                    k_out: k_out_ptr as *mut c_void,
                    q_stride_b: q_stride[0] as i64,
                    q_stride_h: q_stride[1] as i64,
                    q_stride_s: q_stride[2] as i64,
                    q_stride_d: q_stride[3] as i64,
                    k_stride_b: k_stride[0] as i64,
                    k_stride_h: k_stride[1] as i64,
                    k_stride_s: k_stride[2] as i64,
                    k_stride_d: k_stride[3] as i64,
                    batch: batch_i32,
                    q_heads: q_heads_i32,
                    k_heads: k_heads_i32,
                    seq_len: seq_len_i32,
                    head_dim: head_dim_i32,
                    rot_dim: rot_dim_i32,
                    cos_batch_stride: cos_batch_stride_i32,
                    q_eps,
                    k_eps,
                    is_neox: i32::from(is_neox),
                    dtype: $dtype_id,
                    output_token_major: i32::from(output_layout == QkRopeOutputLayout::TokensFirst),
                    stream: stream_ptr,
                })
            })?;

            drop(q_guard);
            drop(q_weight_guard);
            drop(cos_guard);
            drop(sin_guard);
            drop(k_guard);
            drop(k_weight_guard);
            drop(q_out_guard);
            drop(k_out_guard);

            let q_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(q_out_buf),
                device: dev.clone(),
            };
            let q_tensor = Tensor::from((candle_core::Storage::Cuda(q_storage), q_shape));

            let k_tensor = if let Some(k_out_buf) = k_out_buf {
                let k_storage = CudaStorage {
                    slice: CudaStorageSlice::$variant(k_out_buf),
                    device: dev.clone(),
                };
                Some(Tensor::from((
                    candle_core::Storage::Cuda(k_storage),
                    k_shape,
                )))
            } else {
                None
            };

            Ok(Some((q_tensor, k_tensor)))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => Ok(None),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn try_cuda_rope_sincos_positions(
    positions: &Tensor,
    inv_freq: &Tensor,
    dtype: DType,
) -> Result<Option<(Tensor, Tensor)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !positions.device().is_cuda()
        || positions.dtype() != DType::U32
        || inv_freq.dtype() != DType::F32
        || !inv_freq.device().same_device(positions.device())
        || !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
    {
        return Ok(None);
    }

    let rows = positions.dims1()?;
    let width = inv_freq.dims1()?;
    if rows == 0 || width == 0 {
        return Ok(None);
    }
    let rows_i32 = i32::try_from(rows).map_err(candle_core::Error::wrap)?;
    let width_i32 = i32::try_from(width).map_err(candle_core::Error::wrap)?;
    let elements = rows
        .checked_mul(width)
        .ok_or_else(|| candle_core::Error::msg("RoPE sincos output size overflow"))?;

    let positions = positions.contiguous()?;
    let inv_freq = inv_freq.contiguous()?;
    let (positions_storage, positions_layout) = positions.storage_and_layout();
    let positions_storage = match &*positions_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => return Ok(None),
    };
    let (inv_freq_storage, inv_freq_layout) = inv_freq.storage_and_layout();
    let inv_freq_storage = match &*inv_freq_storage {
        candle_core::Storage::Cuda(storage) => storage,
        _ => return Ok(None),
    };
    let CudaStorageSlice::U32(positions_src) = &positions_storage.slice else {
        candle_core::bail!("RoPE sincos positions dtype mismatch");
    };
    let CudaStorageSlice::F32(inv_freq_src) = &inv_freq_storage.slice else {
        candle_core::bail!("RoPE sincos inverse frequency dtype mismatch");
    };

    let dev = positions_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let (positions_ptr, positions_guard) = positions_src.device_ptr(&stream);
    let positions_ptr =
        unsafe { (positions_ptr as *const u32).add(positions_layout.start_offset()) };
    let (inv_freq_ptr, inv_freq_guard) = inv_freq_src.device_ptr(&stream);
    let inv_freq_ptr = unsafe { (inv_freq_ptr as *const f32).add(inv_freq_layout.start_offset()) };
    let output_shape = Shape::from_dims(&[rows, width]);

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let mut cos_buf = unsafe { dev.alloc::<$ty>(elements) }?;
            let mut sin_buf = unsafe { dev.alloc::<$ty>(elements) }?;
            let (cos_ptr, cos_guard) = cos_buf.device_ptr_mut(&stream);
            let (sin_ptr, sin_guard) = sin_buf.device_ptr_mut(&stream);
            unsafe {
                ffi::rope_sincos_positions(
                    positions_ptr as *const c_void,
                    inv_freq_ptr as *const c_void,
                    cos_ptr as *mut c_void,
                    sin_ptr as *mut c_void,
                    rows_i32,
                    width_i32,
                    $dtype_id,
                    stream_ptr,
                );
            }
            drop(cos_guard);
            drop(sin_guard);

            let cos_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(cos_buf),
                device: dev.clone(),
            };
            let sin_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(sin_buf),
                device: dev.clone(),
            };
            let cos = Tensor::from((
                candle_core::Storage::Cuda(cos_storage),
                output_shape.clone(),
            ));
            let sin = Tensor::from((
                candle_core::Storage::Cuda(sin_storage),
                output_shape.clone(),
            ));
            Ok(Some((cos, sin)))
        }};
    }

    let result = match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => unreachable!(),
    };
    drop(positions_guard);
    drop(inv_freq_guard);
    result
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_cuda_qk_rms_norm_rope_positions(
    q: &Tensor,
    k: Option<&Tensor>,
    q_weight: &Tensor,
    k_weight: Option<&Tensor>,
    q_eps: f32,
    k_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<Option<(Tensor, Option<Tensor>)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !q.device().is_cuda() {
        return Ok(None);
    }

    let dtype = q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || q_weight.dtype() != dtype
        || k_weight.is_some_and(|weight| weight.dtype() != dtype)
        || cos.dtype() != dtype
        || sin.dtype() != dtype
        || positions.dtype() != DType::U32
    {
        return Ok(None);
    }

    if !q_weight.device().same_device(q.device())
        || !cos.device().same_device(q.device())
        || !sin.device().same_device(q.device())
        || !positions.device().same_device(q.device())
        || k.is_some_and(|k| !k.device().same_device(q.device()) || k.dtype() != dtype)
        || k_weight.is_some_and(|weight| !weight.device().same_device(q.device()))
    {
        return Ok(None);
    }

    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    let expected_positions = batch * seq_len;
    if positions.dims1()? != expected_positions {
        candle_core::bail!(
            "positions length {} does not match token count {expected_positions}",
            positions.dims1()?
        );
    }

    let (k_heads, k_elem_count) = if let Some(k) = k {
        let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
        if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim) {
            candle_core::bail!(
                "q/k shape mismatch for fused qk norm rope positions: {:?} vs {:?}",
                q.shape(),
                k.shape()
            );
        }
        let Some(k_weight) = k_weight else {
            candle_core::bail!("missing k norm weight for fused qk norm rope positions");
        };
        if k_weight.dims1()? != head_dim {
            candle_core::bail!(
                "k norm weight size {} does not match head dim {head_dim}",
                k_weight.dims1()?
            );
        }
        (k_heads, k.elem_count())
    } else {
        (0, 0)
    };

    if q_weight.dims1()? != head_dim {
        candle_core::bail!(
            "q norm weight size {} does not match head dim {head_dim}",
            q_weight.dims1()?
        );
    }

    let (cos_rows, rot_dim) = cos.dims2()?;
    if sin.dims2()? != (cos_rows, rot_dim) {
        candle_core::bail!(
            "cos/sin shape mismatch for fused qk norm rope positions: {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        return Ok(None);
    }

    for (name, value) in [
        ("batch", batch),
        ("q_heads", q_heads),
        ("k_heads", k_heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
        ("rot_dim", rot_dim),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("fused qk norm rope positions {name} is too large: {value}");
        }
    }
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let q_heads_i32 = i32::try_from(q_heads).map_err(candle_core::Error::wrap)?;
    let k_heads_i32 = i32::try_from(k_heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
    let rot_dim_i32 = i32::try_from(rot_dim).map_err(candle_core::Error::wrap)?;

    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.contiguous()?;
    let q_weight = q_weight.contiguous()?;
    let k_weight = k_weight.map(Tensor::contiguous).transpose()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let q_storage = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_storage_and_layout = k.map(Tensor::storage_and_layout);
    let (q_weight_storage, q_weight_layout) = q_weight.storage_and_layout();
    let q_weight_storage = match &*q_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let k_weight_storage_and_layout = k_weight.as_ref().map(Tensor::storage_and_layout);
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let cos_storage = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let sin_storage = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (positions_storage, positions_layout) = positions.storage_and_layout();
    let positions_storage = match &*positions_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };

    let dev = q_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let output_token_major = q.stride()[1] == head_dim && q.stride()[3] == 1;
    let output_shape = |heads| {
        Shape::from_dims(&if output_token_major {
            [batch, seq_len, heads, head_dim]
        } else {
            [batch, heads, seq_len, head_dim]
        })
    };
    let q_shape = output_shape(q_heads);
    let k_shape = output_shape(k_heads);
    let q_elem_count = q.elem_count();

    let q_stride = q_layout.stride();
    let k_stride = k_storage_and_layout
        .as_ref()
        .map(|(_, layout)| layout.stride())
        .unwrap_or(&[0, 0, 0, 0]);

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let CudaStorageSlice::$variant(q_src) = &q_storage.slice else {
                candle_core::bail!("fused qk norm rope positions q dtype mismatch");
            };
            let CudaStorageSlice::$variant(q_weight_src) = &q_weight_storage.slice else {
                candle_core::bail!("fused qk norm rope positions q weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(cos_src) = &cos_storage.slice else {
                candle_core::bail!("fused qk norm rope positions cos dtype mismatch");
            };
            let CudaStorageSlice::$variant(sin_src) = &sin_storage.slice else {
                candle_core::bail!("fused qk norm rope positions sin dtype mismatch");
            };
            let CudaStorageSlice::U32(positions_src) = &positions_storage.slice else {
                candle_core::bail!("fused qk norm rope positions dtype mismatch");
            };

            let mut q_out_buf = unsafe { dev.alloc::<$ty>(q_elem_count) }?;
            let mut k_out_buf = if k_elem_count == 0 {
                None
            } else {
                Some(unsafe { dev.alloc::<$ty>(k_elem_count) }?)
            };

            let (q_ptr, q_guard) = q_src.device_ptr(&stream);
            let q_ptr = unsafe { (q_ptr as *const $ty).add(q_layout.start_offset()) };
            let (q_weight_ptr, q_weight_guard) = q_weight_src.device_ptr(&stream);
            let q_weight_ptr =
                unsafe { (q_weight_ptr as *const $ty).add(q_weight_layout.start_offset()) };
            let (cos_ptr, cos_guard) = cos_src.device_ptr(&stream);
            let cos_ptr = unsafe { (cos_ptr as *const $ty).add(cos_layout.start_offset()) };
            let (sin_ptr, sin_guard) = sin_src.device_ptr(&stream);
            let sin_ptr = unsafe { (sin_ptr as *const $ty).add(sin_layout.start_offset()) };
            let (positions_ptr, positions_guard) = positions_src.device_ptr(&stream);
            let positions_ptr =
                unsafe { (positions_ptr as *const u32).add(positions_layout.start_offset()) };

            let mut k_guard = None;
            let k_ptr = if let Some((k_storage, k_layout)) = &k_storage_and_layout {
                let k_storage = match &**k_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => return Ok(None),
                };
                let CudaStorageSlice::$variant(k_src) = &k_storage.slice else {
                    candle_core::bail!("fused qk norm rope positions k dtype mismatch");
                };
                let (ptr, guard) = k_src.device_ptr(&stream);
                k_guard = Some(guard);
                unsafe { (ptr as *const $ty).add(k_layout.start_offset()) }
            } else {
                std::ptr::null()
            };

            let mut k_weight_guard = None;
            let k_weight_ptr =
                if let Some((k_weight_storage, k_weight_layout)) = &k_weight_storage_and_layout {
                    let k_weight_storage = match &**k_weight_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => return Ok(None),
                    };
                    let CudaStorageSlice::$variant(k_weight_src) = &k_weight_storage.slice else {
                        candle_core::bail!("fused qk norm rope positions k weight dtype mismatch");
                    };
                    let (ptr, guard) = k_weight_src.device_ptr(&stream);
                    k_weight_guard = Some(guard);
                    unsafe { (ptr as *const $ty).add(k_weight_layout.start_offset()) }
                } else {
                    q_weight_ptr
                };

            let (q_out_ptr, q_out_guard) = q_out_buf.device_ptr_mut(&stream);
            let mut k_out_guard = None;
            let k_out_ptr = if let Some(k_out_buf) = &mut k_out_buf {
                let (ptr, guard) = k_out_buf.device_ptr_mut(&stream);
                k_out_guard = Some(guard);
                ptr as *mut $ty
            } else {
                std::ptr::null_mut()
            };

            crate::cuda::check_launch("qk_rms_norm_rope_positions", unsafe {
                ffi::qk_rms_norm_rope_positions(&crate::cuda::abi::QkRmsNormRopePositionsParams {
                    q: q_ptr as *const c_void,
                    k: k_ptr as *const c_void,
                    q_weight: q_weight_ptr as *const c_void,
                    k_weight: k_weight_ptr as *const c_void,
                    cos: cos_ptr as *const c_void,
                    sin: sin_ptr as *const c_void,
                    positions: positions_ptr as *const c_void,
                    q_out: q_out_ptr as *mut c_void,
                    k_out: k_out_ptr as *mut c_void,
                    q_stride_b: q_stride[0] as i64,
                    q_stride_h: q_stride[1] as i64,
                    q_stride_s: q_stride[2] as i64,
                    q_stride_d: q_stride[3] as i64,
                    k_stride_b: k_stride[0] as i64,
                    k_stride_h: k_stride[1] as i64,
                    k_stride_s: k_stride[2] as i64,
                    k_stride_d: k_stride[3] as i64,
                    batch: batch_i32,
                    q_heads: q_heads_i32,
                    k_heads: k_heads_i32,
                    seq_len: seq_len_i32,
                    head_dim: head_dim_i32,
                    rot_dim: rot_dim_i32,
                    q_eps,
                    k_eps,
                    is_neox: i32::from(is_neox),
                    dtype: $dtype_id,
                    output_token_major: i32::from(output_token_major),
                    stream: stream_ptr,
                })
            })?;

            drop(q_guard);
            drop(q_weight_guard);
            drop(cos_guard);
            drop(sin_guard);
            drop(positions_guard);
            drop(k_guard);
            drop(k_weight_guard);
            drop(q_out_guard);
            drop(k_out_guard);

            let q_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(q_out_buf),
                device: dev.clone(),
            };
            let q_tensor = Tensor::from((candle_core::Storage::Cuda(q_storage), q_shape));

            let k_tensor = if let Some(k_out_buf) = k_out_buf {
                let k_storage = CudaStorage {
                    slice: CudaStorageSlice::$variant(k_out_buf),
                    device: dev.clone(),
                };
                Some(Tensor::from((
                    candle_core::Storage::Cuda(k_storage),
                    k_shape,
                )))
            } else {
                None
            };

            let (q_tensor, k_tensor) = if output_token_major {
                (
                    q_tensor.transpose(1, 2)?,
                    k_tensor.map(|tensor| tensor.transpose(1, 2)).transpose()?,
                )
            } else {
                (q_tensor, k_tensor)
            };
            Ok(Some((q_tensor, k_tensor)))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => Ok(None),
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_cuda_qkv_rms_norm_rope_positions(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    q_weight: &Tensor,
    k_weight: &Tensor,
    v_weight: &Tensor,
    q_eps: f32,
    k_eps: f32,
    v_eps: f32,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    is_neox: bool,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    use candle_core::backend::BackendStorage;
    use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
    use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
    use std::ffi::c_void;

    if !q.device().is_cuda() {
        return Ok(None);
    }

    let dtype = q.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || k.dtype() != dtype
        || v.dtype() != dtype
        || q_weight.dtype() != dtype
        || k_weight.dtype() != dtype
        || v_weight.dtype() != dtype
        || cos.dtype() != dtype
        || sin.dtype() != dtype
        || positions.dtype() != DType::U32
    {
        return Ok(None);
    }

    if !q_weight.device().same_device(q.device())
        || !k_weight.device().same_device(q.device())
        || !v_weight.device().same_device(q.device())
        || !cos.device().same_device(q.device())
        || !sin.device().same_device(q.device())
        || !positions.device().same_device(q.device())
        || !k.device().same_device(q.device())
        || !v.device().same_device(q.device())
    {
        return Ok(None);
    }

    let (batch, q_heads, seq_len, head_dim) = q.dims4()?;
    let (k_batch, k_heads, k_seq_len, k_head_dim) = k.dims4()?;
    let (v_batch, v_heads, v_seq_len, v_head_dim) = v.dims4()?;
    if (k_batch, k_seq_len, k_head_dim) != (batch, seq_len, head_dim)
        || (v_batch, v_heads, v_seq_len, v_head_dim) != (batch, k_heads, seq_len, head_dim)
    {
        candle_core::bail!(
            "q/k/v shape mismatch for fused qkv norm rope positions: {:?}, {:?}, {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );
    }
    let expected_positions = batch * seq_len;
    if positions.dims1()? != expected_positions {
        candle_core::bail!(
            "positions length {} does not match token count {expected_positions}",
            positions.dims1()?
        );
    }
    if q_weight.dims1()? != head_dim
        || k_weight.dims1()? != head_dim
        || v_weight.dims1()? != head_dim
    {
        candle_core::bail!("qkv norm weight size does not match head dim {head_dim}");
    }

    let (cos_rows, rot_dim) = cos.dims2()?;
    if sin.dims2()? != (cos_rows, rot_dim) {
        candle_core::bail!(
            "cos/sin shape mismatch for fused qkv norm rope positions: {:?} vs {:?}",
            cos.shape(),
            sin.shape()
        );
    }
    if rot_dim == 0 || rot_dim * 2 > head_dim {
        return Ok(None);
    }

    for (name, value) in [
        ("batch", batch),
        ("q_heads", q_heads),
        ("k_heads", k_heads),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
        ("rot_dim", rot_dim),
    ] {
        if value > i32::MAX as usize {
            candle_core::bail!("fused qkv norm rope positions {name} is too large: {value}");
        }
    }
    let batch_i32 = i32::try_from(batch).map_err(candle_core::Error::wrap)?;
    let q_heads_i32 = i32::try_from(q_heads).map_err(candle_core::Error::wrap)?;
    let k_heads_i32 = i32::try_from(k_heads).map_err(candle_core::Error::wrap)?;
    let seq_len_i32 = i32::try_from(seq_len).map_err(candle_core::Error::wrap)?;
    let head_dim_i32 = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
    let rot_dim_i32 = i32::try_from(rot_dim).map_err(candle_core::Error::wrap)?;

    let cos = cos.contiguous()?;
    let sin = sin.contiguous()?;
    let positions = positions.contiguous()?;
    let q_weight = q_weight.contiguous()?;
    let k_weight = k_weight.contiguous()?;
    let v_weight = v_weight.contiguous()?;

    let (q_storage, q_layout) = q.storage_and_layout();
    let q_storage = match &*q_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (k_storage, k_layout) = k.storage_and_layout();
    let k_storage = match &*k_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (v_storage, v_layout) = v.storage_and_layout();
    let v_storage = match &*v_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (q_weight_storage, q_weight_layout) = q_weight.storage_and_layout();
    let q_weight_storage = match &*q_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (k_weight_storage, k_weight_layout) = k_weight.storage_and_layout();
    let k_weight_storage = match &*k_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (v_weight_storage, v_weight_layout) = v_weight.storage_and_layout();
    let v_weight_storage = match &*v_weight_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (cos_storage, cos_layout) = cos.storage_and_layout();
    let cos_storage = match &*cos_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (sin_storage, sin_layout) = sin.storage_and_layout();
    let sin_storage = match &*sin_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };
    let (positions_storage, positions_layout) = positions.storage_and_layout();
    let positions_storage = match &*positions_storage {
        candle_core::Storage::Cuda(s) => s,
        _ => return Ok(None),
    };

    let dev = q_storage.device();
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as i64;
    let q_shape = Shape::from_dims(&[batch, q_heads, seq_len, head_dim]);
    let kv_shape = Shape::from_dims(&[batch, k_heads, seq_len, head_dim]);
    let q_elem_count = q.elem_count();
    let kv_elem_count = k.elem_count();

    let q_stride = q_layout.stride();
    let k_stride = k_layout.stride();
    let v_stride = v_layout.stride();

    macro_rules! launch {
        ($variant:ident, $ty:ty, $dtype_id:expr) => {{
            let CudaStorageSlice::$variant(q_src) = &q_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions q dtype mismatch");
            };
            let CudaStorageSlice::$variant(k_src) = &k_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions k dtype mismatch");
            };
            let CudaStorageSlice::$variant(v_src) = &v_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions v dtype mismatch");
            };
            let CudaStorageSlice::$variant(q_weight_src) = &q_weight_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions q weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(k_weight_src) = &k_weight_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions k weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(v_weight_src) = &v_weight_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions v weight dtype mismatch");
            };
            let CudaStorageSlice::$variant(cos_src) = &cos_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions cos dtype mismatch");
            };
            let CudaStorageSlice::$variant(sin_src) = &sin_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions sin dtype mismatch");
            };
            let CudaStorageSlice::U32(positions_src) = &positions_storage.slice else {
                candle_core::bail!("fused qkv norm rope positions dtype mismatch");
            };

            let mut q_out_buf = unsafe { dev.alloc::<$ty>(q_elem_count) }?;
            let mut k_out_buf = unsafe { dev.alloc::<$ty>(kv_elem_count) }?;
            let mut v_out_buf = unsafe { dev.alloc::<$ty>(kv_elem_count) }?;

            let (q_ptr, q_guard) = q_src.device_ptr(&stream);
            let q_ptr = unsafe { (q_ptr as *const $ty).add(q_layout.start_offset()) };
            let (k_ptr, k_guard) = k_src.device_ptr(&stream);
            let k_ptr = unsafe { (k_ptr as *const $ty).add(k_layout.start_offset()) };
            let (v_ptr, v_guard) = v_src.device_ptr(&stream);
            let v_ptr = unsafe { (v_ptr as *const $ty).add(v_layout.start_offset()) };
            let (q_weight_ptr, q_weight_guard) = q_weight_src.device_ptr(&stream);
            let q_weight_ptr =
                unsafe { (q_weight_ptr as *const $ty).add(q_weight_layout.start_offset()) };
            let (k_weight_ptr, k_weight_guard) = k_weight_src.device_ptr(&stream);
            let k_weight_ptr =
                unsafe { (k_weight_ptr as *const $ty).add(k_weight_layout.start_offset()) };
            let (v_weight_ptr, v_weight_guard) = v_weight_src.device_ptr(&stream);
            let v_weight_ptr =
                unsafe { (v_weight_ptr as *const $ty).add(v_weight_layout.start_offset()) };
            let (cos_ptr, cos_guard) = cos_src.device_ptr(&stream);
            let cos_ptr = unsafe { (cos_ptr as *const $ty).add(cos_layout.start_offset()) };
            let (sin_ptr, sin_guard) = sin_src.device_ptr(&stream);
            let sin_ptr = unsafe { (sin_ptr as *const $ty).add(sin_layout.start_offset()) };
            let (positions_ptr, positions_guard) = positions_src.device_ptr(&stream);
            let positions_ptr =
                unsafe { (positions_ptr as *const u32).add(positions_layout.start_offset()) };

            let (q_out_ptr, q_out_guard) = q_out_buf.device_ptr_mut(&stream);
            let (k_out_ptr, k_out_guard) = k_out_buf.device_ptr_mut(&stream);
            let (v_out_ptr, v_out_guard) = v_out_buf.device_ptr_mut(&stream);

            crate::cuda::check_launch("qkv_rms_norm_rope_positions", unsafe {
                ffi::qkv_rms_norm_rope_positions(&crate::cuda::abi::QkvRmsNormRopePositionsParams {
                    q: q_ptr as *const c_void,
                    k: k_ptr as *const c_void,
                    v: v_ptr as *const c_void,
                    q_weight: q_weight_ptr as *const c_void,
                    k_weight: k_weight_ptr as *const c_void,
                    v_weight: v_weight_ptr as *const c_void,
                    cos: cos_ptr as *const c_void,
                    sin: sin_ptr as *const c_void,
                    positions: positions_ptr as *const c_void,
                    q_out: q_out_ptr as *mut c_void,
                    k_out: k_out_ptr as *mut c_void,
                    v_out: v_out_ptr as *mut c_void,
                    q_stride_b: q_stride[0] as i64,
                    q_stride_h: q_stride[1] as i64,
                    q_stride_s: q_stride[2] as i64,
                    q_stride_d: q_stride[3] as i64,
                    k_stride_b: k_stride[0] as i64,
                    k_stride_h: k_stride[1] as i64,
                    k_stride_s: k_stride[2] as i64,
                    k_stride_d: k_stride[3] as i64,
                    v_stride_b: v_stride[0] as i64,
                    v_stride_h: v_stride[1] as i64,
                    v_stride_s: v_stride[2] as i64,
                    v_stride_d: v_stride[3] as i64,
                    batch: batch_i32,
                    q_heads: q_heads_i32,
                    k_heads: k_heads_i32,
                    seq_len: seq_len_i32,
                    head_dim: head_dim_i32,
                    rot_dim: rot_dim_i32,
                    q_eps,
                    k_eps,
                    v_eps,
                    is_neox: i32::from(is_neox),
                    dtype: $dtype_id,
                    stream: stream_ptr,
                })
            })?;

            drop(q_guard);
            drop(k_guard);
            drop(v_guard);
            drop(q_weight_guard);
            drop(k_weight_guard);
            drop(v_weight_guard);
            drop(cos_guard);
            drop(sin_guard);
            drop(positions_guard);
            drop(q_out_guard);
            drop(k_out_guard);
            drop(v_out_guard);

            let q_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(q_out_buf),
                device: dev.clone(),
            };
            let k_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(k_out_buf),
                device: dev.clone(),
            };
            let v_storage = CudaStorage {
                slice: CudaStorageSlice::$variant(v_out_buf),
                device: dev.clone(),
            };
            Ok(Some((
                Tensor::from((candle_core::Storage::Cuda(q_storage), q_shape)),
                Tensor::from((candle_core::Storage::Cuda(k_storage), kv_shape.clone())),
                Tensor::from((candle_core::Storage::Cuda(v_storage), kv_shape)),
            )))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, 1),
        DType::F16 => launch!(F16, half::f16, 0),
        DType::F32 => launch!(F32, f32, 2),
        _ => Ok(None),
    }
}
