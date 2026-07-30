//! Bridge candle's CUDA stream into a cuTile `ExecutionContext`; borrowed handles are non-owning and never destroy candle's context/stream.

use candle_core::CudaDevice;
use core::ffi::{c_int, c_void};
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{Device as CutileDevice, Stream as CutileStream};

/// Build a cuTile `ExecutionContext` that enqueues onto candle's current stream.
pub fn execution_context(dev: &CudaDevice) -> ExecutionContext {
    let stream = dev.cuda_stream();
    let ctx = stream.context();
    let cu_ctx = ctx.cu_ctx();
    let cu_device = ctx.cu_device();
    let ordinal = ctx.ordinal();
    let cu_stream = stream.cu_stream();

    // SAFETY: handles come from candle's live context/stream and outlive the returned context.
    let cdev =
        unsafe { CutileDevice::borrow_raw(cu_ctx as *mut c_void, cu_device as c_int, ordinal) };
    let cstream = unsafe { CutileStream::borrow_raw(cu_stream as *mut c_void, &cdev) };
    ExecutionContext::new(cstream)
}
