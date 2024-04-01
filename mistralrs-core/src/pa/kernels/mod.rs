pub mod attention;
pub mod cache;
pub mod rope;

use crate::pa::kernels::sys::CUstream;
use candle_core::{
    cuda_backend::{
        cudarc::driver::{sys, DevicePtr, DeviceRepr},
        WrapErr,
    },
    DType, Device, Storage, Tensor,
};
use std::ops::Deref;

#[derive(Copy)]
#[repr(C)]
pub struct DeviceDataPtr {
    ptr: u64,
}

unsafe impl DeviceRepr for DeviceDataPtr {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.ptr) as *const sys::CUdeviceptr as *mut std::ffi::c_void
    }
}

impl Clone for DeviceDataPtr {
    fn clone(&self) -> Self {
        *self
    }
}

impl DeviceDataPtr {
    pub fn as_ffi_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr as *mut std::ffi::c_void
    }
    pub fn as_ptr_int(&self) -> u64 {
        self.ptr
    }
    pub fn null() -> Self {
        Self { ptr: 0 }
    }
    pub fn advance(&mut self, n: usize) {
        self.ptr += n as u64;
    }
}

pub fn get_tensor_device_ptr(tensor: &Tensor) -> candle_core::Result<DeviceDataPtr> {
    let (storage, _) = tensor.storage_and_layout();
    let start_offset = tensor.layout().start_offset();
    let data = match storage.deref() {
        Storage::Cuda(cuda_storage) => match tensor.dtype() {
            candle_core::DType::U8 => *cuda_storage.as_cuda_slice::<u8>()?.device_ptr(),
            candle_core::DType::F16 => *cuda_storage.as_cuda_slice::<half::f16>()?.device_ptr(),
            candle_core::DType::BF16 => *cuda_storage.as_cuda_slice::<half::bf16>()?.device_ptr(),
            candle_core::DType::F32 => *cuda_storage.as_cuda_slice::<f32>()?.device_ptr(),
            candle_core::DType::U32 => *cuda_storage.as_cuda_slice::<u32>()?.device_ptr(),
            candle_core::DType::F64 => *cuda_storage.as_cuda_slice::<f64>()?.device_ptr(),
            candle_core::DType::I64 => *cuda_storage.as_cuda_slice::<i64>()?.device_ptr(),
        },
        Storage::Cpu(cpu_storage) => match tensor.dtype() {
            candle_core::DType::U8 => cpu_storage.as_slice::<u8>()?.as_ptr() as u64,
            candle_core::DType::F16 => cpu_storage.as_slice::<half::f16>()?.as_ptr() as u64,
            candle_core::DType::BF16 => cpu_storage.as_slice::<half::bf16>()?.as_ptr() as u64,
            candle_core::DType::F32 => cpu_storage.as_slice::<f32>()?.as_ptr() as u64,
            candle_core::DType::U32 => cpu_storage.as_slice::<u32>()?.as_ptr() as u64,
            candle_core::DType::F64 => cpu_storage.as_slice::<f64>()?.as_ptr() as u64,
            candle_core::DType::I64 => cpu_storage.as_slice::<i64>()?.as_ptr() as u64,
        },
        _ => unreachable!("unexpected storage type"),
    };
    // println!("###start_offset:{}", start_offset);
    let ptr_int = if start_offset > 0 {
        data + (start_offset * tensor.dtype().size_in_bytes()) as u64
    } else {
        data
    };
    Ok(DeviceDataPtr { ptr: ptr_int })
}

#[derive(Debug)]
#[repr(C)]
pub enum ScalarType {
    DataU8 = 0,
    DataF16,
    DataBF16,
    DataF32,
    DataF64,
    DataU32,
    DataI64,

    DataUnsupported = 100,
}

pub fn get_scalar_type(dtype: DType) -> ScalarType {
    match dtype {
        DType::BF16 => ScalarType::DataBF16,
        DType::U8 => ScalarType::DataU8,
        DType::U32 => ScalarType::DataU32,
        DType::I64 => ScalarType::DataI64,
        DType::F16 => ScalarType::DataF16,
        DType::F32 => ScalarType::DataF32,
        DType::F64 => ScalarType::DataF64,
    }
}

pub fn cuda_get_default_stream(device: &Device) -> candle_core::Result<sys::CUstream> {
    match device {
        Device::Cuda(cuda) => Ok(*cuda.cu_stream()),
        _ => {
            candle_core::bail!("unexpected device")
        }
    }
}

pub fn cuda_stream_synchronize(stream: CUstream) -> candle_core::Result<()> {
    unsafe { candle_core::cuda_backend::cudarc::driver::result::stream::synchronize(stream).w() }
}

pub fn cuda_get_free_mem_size() -> anyhow::Result<usize> {
    let (free, _) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?;
    Ok(free)
}
pub fn cuda_get_mem_usage() -> anyhow::Result<(usize, usize)> {
    let (free, total) = candle_core::cuda_backend::cudarc::driver::result::mem_get_info()?;
    Ok((free, total))
}
