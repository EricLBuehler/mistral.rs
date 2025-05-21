#[cfg(feature = "cuda")]
mod ffi;
pub(crate) mod isq;
pub mod log;
mod ops;
mod uqff;

pub use ops::{BitWiseOp, CumSumOp, LeftshiftOp, NonZeroOp, SortOp};
pub use uqff::UQFF_QUANT_TYPE_OFFSET;
pub(crate) use uqff::{
    deserialize_tensor, fake_deserialize_tensor, read_dtype, serialize_tensor,
    version_is_compatible, write_dtype, UQFF_VERSION,
};

#[cfg(feature = "cuda")]
use candle_core::{
    cuda::{cudarc::driver::DevicePtr, CudaDType},
    CudaDevice, Device, Storage, Tensor, WithDType,
};

#[cfg(feature = "cuda")]
pub(crate) fn get_cuda_slice<T: WithDType + CudaDType>(
    x: &Tensor,
) -> candle_core::Result<*const T> {
    let offset = x.layout().start_offset();
    match &*x.storage_and_layout().0 {
        Storage::Cuda(a_storage) => {
            Ok(*a_storage.as_cuda_slice::<T>()?.slice(offset..).device_ptr() as *const T)
        }
        _ => candle_core::bail!("Expected CUDA storage."),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn get_cuda_device(x: &Tensor) -> candle_core::Result<&CudaDevice> {
    match x.device() {
        Device::Cuda(dev) => Ok(dev),
        _ => candle_core::bail!("Expected CUDA device"),
    }
}
