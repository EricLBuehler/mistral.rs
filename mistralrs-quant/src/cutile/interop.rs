//! Bridge candle's CUDA device/stream into a cuTile `ExecutionContext` so cuTile
//! kernels run on candle's existing stream with candle-owned buffers (passed as
//! raw pointers). Borrowed handles are non-owning: dropping them never destroys
//! candle's context/stream.

use candle_core::CudaDevice;
use cuda_async::device_operation::ExecutionContext;
use cuda_core::{Device as CutileDevice, Stream as CutileStream};
use core::ffi::{c_int, c_void};

/// Build a cuTile `ExecutionContext` that enqueues onto candle's current stream.
pub fn execution_context(dev: &CudaDevice) -> ExecutionContext {
    let stream = dev.cuda_stream();
    let ctx = stream.context();
    let cu_ctx = ctx.cu_ctx();
    let cu_device = ctx.cu_device();
    let ordinal = ctx.ordinal();
    let cu_stream = stream.cu_stream();

    // SAFETY: handles come from candle's live context/stream and outlive the
    // returned context (held only for the duration of the launch below).
    let cdev = unsafe { CutileDevice::borrow_raw(cu_ctx as *mut c_void, cu_device as c_int, ordinal) };
    let cstream = unsafe { CutileStream::borrow_raw(cu_stream as *mut c_void, &cdev) };
    ExecutionContext::new(cstream)
}
