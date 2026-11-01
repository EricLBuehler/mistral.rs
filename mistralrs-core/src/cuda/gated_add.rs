use candle_core::{
    cuda::cudarc::driver::{DevicePtr, DevicePtrMut},
    CudaStorage, DType, Result, Storage, Tensor,
};

mod ffi {
    use core::ffi::c_void;
    extern "C" {
        pub(super) fn launch_shared_gated_add(
            residual: *const c_void,
            value: *const c_void,
            gate: *const c_void,
            output: *mut c_void,
            elements: usize,
            columns: usize,
            dtype: i32,
            stream: *mut c_void,
        );
    }
}

pub(crate) fn try_sigmoid_gated_add(
    residual: &Tensor,
    value: &Tensor,
    gate: &Tensor,
) -> Result<Option<Tensor>> {
    if !residual.device().is_cuda()
        || residual.rank() == 0
        || residual.shape() != value.shape()
        || residual.dtype() != value.dtype()
        || residual.dtype() != gate.dtype()
        || !residual.device().same_device(value.device())
        || !residual.device().same_device(gate.device())
        || !residual.is_contiguous()
        || !value.is_contiguous()
        || !gate.is_contiguous()
        || !matches!(residual.dtype(), DType::BF16 | DType::F16 | DType::F32)
    {
        return Ok(None);
    }
    let columns = residual.dim(candle_core::D::Minus1)?;
    let elements = residual.elem_count();
    if columns == 0
        || elements == 0
        || gate.rank() != residual.rank()
        || gate.dims().last() != Some(&1)
        || gate.dims()[..gate.rank() - 1] != residual.dims()[..residual.rank() - 1]
    {
        return Ok(None);
    }
    let device = residual.device().as_cuda_device()?;
    let stream = device.cuda_stream();
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let (value_storage, value_layout) = value.storage_and_layout();
    let (gate_storage, gate_layout) = gate.storage_and_layout();
    let (
        Storage::Cuda(residual_storage),
        Storage::Cuda(value_storage),
        Storage::Cuda(gate_storage),
    ) = (&*residual_storage, &*value_storage, &*gate_storage)
    else {
        unreachable!()
    };
    macro_rules! launch {
        ($type:ty, $dtype:expr) => {{
            let (r, _rg) = residual_storage
                .as_cuda_slice::<$type>()?
                .device_ptr(&stream);
            let (v, _vg) = value_storage.as_cuda_slice::<$type>()?.device_ptr(&stream);
            let (g, _gg) = gate_storage.as_cuda_slice::<$type>()?.device_ptr(&stream);
            let mut output = unsafe { device.alloc::<$type>(elements)? };
            {
                let (o, _og) = output.device_ptr_mut(&stream);
                let bytes = std::mem::size_of::<$type>() as u64;
                unsafe {
                    ffi::launch_shared_gated_add(
                        (r + bytes * residual_layout.start_offset() as u64) as *const _,
                        (v + bytes * value_layout.start_offset() as u64) as *const _,
                        (g + bytes * gate_layout.start_offset() as u64) as *const _,
                        o as *mut _,
                        elements,
                        columns,
                        $dtype,
                        stream.cu_stream() as *mut _,
                    );
                }
            }
            Ok(Some(Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, device.clone())),
                residual.shape().clone(),
            ))))
        }};
    }
    match residual.dtype() {
        DType::F32 => launch!(f32, 0),
        DType::F16 => launch!(half::f16, 1),
        DType::BF16 => launch!(half::bf16, 2),
        _ => unreachable!(),
    }
}
