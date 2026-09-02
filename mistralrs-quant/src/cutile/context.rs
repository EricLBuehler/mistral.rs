//! Bridge candle's CUDA stream into a non-owning cuTile stream.

use candle_core::CudaDevice;
use core::ffi::{c_int, c_void};
use cutile::cuda_core::{Device as CutileDevice, Stream as CutileStream};
use std::sync::Arc;

/// Borrow candle's current stream for cuTile launches while retaining its owners.
pub fn stream(dev: &CudaDevice) -> Arc<CutileStream> {
    let stream = dev.cuda_stream();
    let ctx = stream.context();
    let cu_ctx = ctx.cu_ctx();
    let cu_device = ctx.cu_device();
    let ordinal = ctx.ordinal();
    let cu_stream = stream.cu_stream();

    // SAFETY: the retained cudarc owners keep both borrowed handles alive.
    let cdev = unsafe {
        CutileDevice::borrow_with_owner(
            cu_ctx as *mut c_void,
            cu_device as c_int,
            ordinal,
            ctx.clone(),
        )
    };
    unsafe { CutileStream::borrow_with_owner(cu_stream as *mut c_void, &cdev, stream) }
}
