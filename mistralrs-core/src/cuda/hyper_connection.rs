#![allow(clippy::cast_possible_truncation)]

use candle_core::{
    cuda_backend::cudarc::driver::DevicePtr, DType, Device, Result, Storage, Tensor,
};

const FUSED_MAX_TOKENS: usize = 1;

mod ffi {
    use core::ffi::c_void;
    extern "C" {
        pub(super) fn mistral_hc_up_mix(
            lo: *const c_void,
            up: *const c_void,
            xn: *const c_void,
            out: *mut c_void,
            tokens: i32,
            hidden: i32,
            rank: i32,
            hc: i32,
            format: i32,
            dtype: i32,
            multiprocessors: i32,
            stream: i64,
        );
        pub(super) fn mistral_hc_silu(
            input: *const c_void,
            output: *mut c_void,
            elements: usize,
            columns: usize,
            row_stride: usize,
            hc: i32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn mistral_hc_norm(
            x: *const c_void,
            weight: *const f32,
            out: *mut c_void,
            rows: i32,
            hc: i32,
            hidden: i32,
            eps: f32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn mistral_hc_mix(
            xn: *const c_void,
            gate: *const c_void,
            out: *mut c_void,
            tokens: i32,
            hc: i32,
            hidden: i32,
            dtype: i32,
            stream: i64,
        );
        pub(super) fn mistral_hc_combine(
            res: *const c_void,
            block_out: *const c_void,
            inject: *const c_void,
            res_out: *mut c_void,
            next_weight: *const f32,
            xn_out: *mut c_void,
            inject_stride: usize,
            rows: i32,
            hc: i32,
            hidden: i32,
            eps: f32,
            dtype: i32,
            stream: i64,
        );
    }
}

fn launch(
    inputs: &[Option<&Tensor>],
    shapes: &[candle_core::Shape],
    dtype: DType,
    device: &Device,
    callback: impl FnOnce(&[u64], &[u64], i64),
) -> Result<Vec<Tensor>> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtrMut;
    use candle_core::CudaStorage;

    let dev = device.as_cuda_device()?;
    let stream = dev.cuda_stream();
    let inputs_storage = inputs
        .iter()
        .map(|tensor| tensor.map(Tensor::storage_and_layout))
        .collect::<Vec<_>>();
    let mut input_pointers = Vec::with_capacity(inputs.len());
    let mut input_guards = Vec::with_capacity(inputs.len());
    for (tensor, storage) in inputs.iter().zip(&inputs_storage) {
        let Some((storage, layout)) = storage else {
            input_pointers.push(0);
            continue;
        };
        let Storage::Cuda(storage) = &**storage else {
            candle_core::bail!("hyper-connection inputs must be on CUDA");
        };
        macro_rules! pointer {
            ($ty:ty) => {{
                let (pointer, guard) = storage.as_cuda_slice::<$ty>()?.device_ptr(&stream);
                input_guards.push(guard);
                pointer + (layout.start_offset() * std::mem::size_of::<$ty>()) as u64
            }};
        }
        input_pointers.push(match tensor.unwrap().dtype() {
            DType::F16 => pointer!(half::f16),
            DType::BF16 => pointer!(half::bf16),
            DType::F32 => pointer!(f32),
            other => candle_core::bail!("unsupported hyper-connection dtype {other:?}"),
        });
    }
    let mut outputs = shapes
        .iter()
        .map(|shape| {
            macro_rules! allocate {
                ($ty:ty) => {
                    CudaStorage::wrap_cuda_slice(
                        unsafe { dev.alloc::<$ty>(shape.elem_count())? },
                        dev.clone(),
                    )
                };
            }
            Ok(match dtype {
                DType::F16 => allocate!(half::f16),
                DType::BF16 => allocate!(half::bf16),
                DType::F32 => allocate!(f32),
                other => candle_core::bail!("unsupported hyper-connection dtype {other:?}"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut output_pointers = Vec::with_capacity(outputs.len());
    let mut output_guards = Vec::with_capacity(outputs.len());
    for output in &mut outputs {
        macro_rules! pointer {
            ($ty:ty) => {{
                let (pointer, guard) = output.as_cuda_slice_mut::<$ty>()?.device_ptr_mut(&stream);
                output_guards.push(guard);
                pointer
            }};
        }
        output_pointers.push(match dtype {
            DType::F16 => pointer!(half::f16),
            DType::BF16 => pointer!(half::bf16),
            DType::F32 => pointer!(f32),
            _ => unreachable!(),
        });
    }
    callback(&input_pointers, &output_pointers, stream.cu_stream() as i64);
    drop(output_guards);
    drop(input_guards);
    Ok(outputs
        .into_iter()
        .zip(shapes)
        .map(|(storage, shape)| Tensor::from((Storage::Cuda(storage), shape.clone())))
        .collect())
}

fn dtype_code(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        other => candle_core::bail!("Hyper-connection kernels do not support {other:?}"),
    }
}

fn half_dtype_code(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        other => candle_core::bail!("Hyper-connection kernels require f16/bf16, got {other:?}"),
    }
}

pub(crate) fn try_hc_up_mix(
    xn: &Tensor,
    lo: &Tensor,
    projection: &dyn mistralrs_quant::QuantMethod,
    hc: usize,
) -> Result<Option<Tensor>> {
    if !xn.device().is_cuda()
        || !matches!(xn.dtype(), DType::BF16 | DType::F16)
        || projection.has_bias()
        || projection.is_dynamic_lora_active()
        || projection.stats_snapshot().is_some()
    {
        return Ok(None);
    }
    if xn.elem_count() / xn.dim(candle_core::D::Minus1)? > FUSED_MAX_TOKENS {
        #[cfg(feature = "cutile")]
        if let Some(weight) = projection.unquant_forward_weight() {
            return mistralrs_quant::cutile::try_hc_prefill_up_mix(xn, lo, &weight, hc);
        }
        return Ok(None);
    }
    let Some((weight, _)) = projection.unquant_weight_bias() else {
        return Ok(None);
    };
    if !weight.device().same_device(xn.device())
        || !weight.is_contiguous()
        || !matches!(weight.dtype(), DType::F16 | DType::BF16)
    {
        return Ok(None);
    }
    let xn = xn.contiguous()?;
    let lo = lo.contiguous()?;
    let width = xn.dim(candle_core::D::Minus1)?;
    let tokens = xn.elem_count() / width;
    let hidden = width / hc;
    let mut shape = xn.dims().to_vec();
    *shape.last_mut().unwrap() = hidden;
    let dtype = half_dtype_code(xn.dtype())?;
    let rank = lo.dim(candle_core::D::Minus1)?;
    let multiprocessors = xn
        .device()
        .as_cuda_device()?
        .cuda_stream()
        .context()
        .attribute(candle_core::cuda::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(candle_core::Error::wrap)?;
    let outputs = launch(
        &[Some(&lo), Some(&weight), Some(&xn)],
        &[shape.into()],
        xn.dtype(),
        xn.device(),
        |inputs, outputs, stream| unsafe {
            ffi::mistral_hc_up_mix(
                inputs[0] as *const _,
                inputs[1] as *const _,
                inputs[2] as *const _,
                outputs[0] as *mut _,
                tokens as i32,
                hidden as i32,
                rank as i32,
                hc as i32,
                i32::from(weight.dtype() == DType::BF16),
                dtype,
                multiprocessors,
                stream,
            );
        },
    )?;
    Ok(outputs.into_iter().next())
}

fn contiguous_rows(tensor: &Tensor) -> Result<Tensor> {
    let last = tensor.rank() - 1;
    let strides = tensor.stride();
    if strides[last] != 1 {
        return tensor.contiguous();
    }
    let mut expected = row_stride(tensor);
    for index in (0..last).rev() {
        if tensor.dims()[index] > 1 && strides[index] != expected {
            return tensor.contiguous();
        }
        expected *= tensor.dims()[index];
    }
    Ok(tensor.clone())
}

fn row_stride(tensor: &Tensor) -> usize {
    let columns = tensor.dims()[tensor.rank() - 1];
    if tensor.rank() == 1 || tensor.elem_count() == columns {
        columns
    } else {
        tensor.stride()[tensor.rank() - 2]
    }
}

pub(crate) fn hc_silu(lo: &Tensor, hc: usize) -> Result<Tensor> {
    let lo = contiguous_rows(lo)?;
    let columns = lo.dim(candle_core::D::Minus1)?;
    let dtype = dtype_code(lo.dtype())?;
    let outputs = launch(
        &[Some(&lo)],
        &[lo.shape().clone()],
        lo.dtype(),
        lo.device(),
        |inputs, outputs, stream| unsafe {
            ffi::mistral_hc_silu(
                inputs[0] as *const _,
                outputs[0] as *mut _,
                lo.elem_count(),
                columns,
                row_stride(&lo),
                hc as i32,
                dtype,
                stream,
            );
        },
    )?;
    Ok(outputs.into_iter().next().unwrap())
}

pub(crate) fn hc_norm(x: &Tensor, weight: &Tensor, hc: usize, eps: f64) -> Result<Tensor> {
    let x = x.contiguous()?;
    let width = x.dim(candle_core::D::Minus1)?;
    let rows = x.elem_count() / (width / hc);
    let dtype = dtype_code(x.dtype())?;
    let outputs = launch(
        &[Some(&x), Some(weight)],
        &[x.shape().clone()],
        x.dtype(),
        x.device(),
        |inputs, outputs, stream| unsafe {
            ffi::mistral_hc_norm(
                inputs[0] as *const _,
                inputs[1] as *const f32,
                outputs[0] as *mut _,
                rows as i32,
                hc as i32,
                (width / hc) as i32,
                eps as f32,
                dtype,
                stream,
            );
        },
    )?;
    Ok(outputs.into_iter().next().unwrap())
}

pub(crate) fn hc_mix(xn: &Tensor, gate: &Tensor, hc: usize) -> Result<Tensor> {
    let xn = xn.contiguous()?;
    let gate = gate.contiguous()?;
    let width = xn.dim(candle_core::D::Minus1)?;
    let hidden = width / hc;
    let tokens = xn.elem_count() / width;
    let mut shape = xn.dims().to_vec();
    *shape.last_mut().expect("rank >= 1") = hidden;
    let dtype = dtype_code(xn.dtype())?;
    let outputs = launch(
        &[Some(&xn), Some(&gate)],
        &[shape.into()],
        xn.dtype(),
        xn.device(),
        |inputs, outputs, stream| unsafe {
            ffi::mistral_hc_mix(
                inputs[0] as *const _,
                inputs[1] as *const _,
                outputs[0] as *mut _,
                tokens as i32,
                hc as i32,
                hidden as i32,
                dtype,
                stream,
            );
        },
    )?;
    Ok(outputs.into_iter().next().unwrap())
}

pub(crate) fn hc_combine(
    res: &Tensor,
    block_out: &Tensor,
    inject: &Tensor,
    next_norm: Option<&Tensor>,
    hc: usize,
    eps: f64,
) -> Result<(Tensor, Option<Tensor>)> {
    let res = res.contiguous()?;
    let block_out = block_out.contiguous()?;
    let inject = contiguous_rows(inject)?;
    let width = res.dim(candle_core::D::Minus1)?;
    let hidden = width / hc;
    let rows = res.elem_count() / hidden;
    let shapes = vec![res.shape().clone(); 1 + usize::from(next_norm.is_some())];
    let dtype = dtype_code(res.dtype())?;
    let outputs = launch(
        &[Some(&res), Some(&block_out), Some(&inject), next_norm],
        &shapes,
        res.dtype(),
        res.device(),
        |inputs, outputs, stream| unsafe {
            ffi::mistral_hc_combine(
                inputs[0] as *const _,
                inputs[1] as *const _,
                inputs[2] as *const _,
                outputs[0] as *mut _,
                inputs[3] as *const f32,
                outputs.get(1).copied().unwrap_or(0) as *mut _,
                row_stride(&inject),
                rows as i32,
                hc as i32,
                hidden as i32,
                eps as f32,
                dtype,
                stream,
            );
        },
    )?;
    let mut outputs = outputs.into_iter();
    Ok((outputs.next().unwrap(), outputs.next()))
}
