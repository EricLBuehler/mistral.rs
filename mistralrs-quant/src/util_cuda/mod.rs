use candle_core::{
    cuda::{cudarc::driver::DevicePtr, CudaDType},
    CudaDevice, Device, Storage, Tensor, WithDType,
};

mod ffi;
mod ops;

pub use ops::{BitWiseOp, LeftshiftOp};

pub(crate) fn get_cuda_slice<T: WithDType + CudaDType>(x: &Tensor) -> *const T {
    match &*x.storage_and_layout().0 {
        Storage::Cuda(a_storage) => *a_storage
            .as_cuda_slice::<T>()
            .expect("DType is not T")
            .device_ptr() as *const T,
        _ => panic!("Expected CUDA storage."),
    }
}

pub(crate) fn get_cuda_device(x: &Tensor) -> &CudaDevice {
    match x.device() {
        Device::Cuda(dev) => dev,
        _ => panic!("Expected CUDA device"),
    }
}
