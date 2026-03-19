#[cfg(feature = "cuda")]
mod ffi;
pub(crate) mod isq;
pub mod log;
mod ops;
mod uqff;

pub use ops::flash_attn_sinks_metal;
pub use ops::flash_attn_sinks_varlen_metal;
#[cfg(feature = "cuda")]
pub use ops::gptoss_swiglu_fused;
#[cfg(feature = "cuda")]
pub use ops::gptoss_swiglu_interleaved;
pub use ops::softmax_with_sinks;
pub use ops::{fused_glu, GluActivationType};
pub use ops::{BitWiseOp, CumSumOp, LeftshiftOp, NonZeroOp, SortOp};
pub use uqff::UQFF_QUANT_TYPE_OFFSET;
pub(crate) use uqff::{
    deserialize_tensor, fake_deserialize_tensor, read_dtype, serialize_tensor,
    version_is_compatible, write_dtype, UQFF_VERSION,
};

#[cfg(feature = "cuda")]
use candle_core::{
    cuda::cudarc::{
        self,
        driver::{CudaSlice, DevicePtr, DeviceRepr},
    },
    CudaDevice, Device, Tensor,
};

#[cfg(feature = "cuda")]
pub(crate) fn get_cuda_device(x: &Tensor) -> candle_core::Result<&CudaDevice> {
    match x.device() {
        Device::Cuda(dev) => Ok(dev),
        _ => candle_core::bail!("Expected CUDA device"),
    }
}

#[cfg(feature = "cuda")]
pub fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    let (_, guard) = v.device_ptr(v.stream());
    let (ptr, _) = v.slice(lo..).device_ptr(v.stream());
    (ptr, guard)
}
