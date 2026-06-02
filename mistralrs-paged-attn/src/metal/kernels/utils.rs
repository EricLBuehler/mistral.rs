use candle_metal_kernels::metal::{CommandBuffer, CommandsGuard, ComputeCommandEncoder};
use std::ffi::c_void;

pub trait RawBytesEncoder {
    fn set_bytes_raw(&self, index: usize, length: usize, bytes: *const c_void);
}

impl RawBytesEncoder for ComputeCommandEncoder {
    fn set_bytes_raw(&self, index: usize, length: usize, bytes: *const c_void) {
        self.set_bytes_directly(index, length, bytes);
    }
}

pub trait EncoderProvider {
    type Encoder<'a>: AsRef<ComputeCommandEncoder>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_>;
}

pub struct WrappedEncoder<'a> {
    inner: &'a ComputeCommandEncoder,
    end_encoding_on_drop: bool,
}

impl Drop for WrappedEncoder<'_> {
    fn drop(&mut self) {
        if self.end_encoding_on_drop {
            self.inner.end_encoding()
        }
    }
}

impl AsRef<ComputeCommandEncoder> for WrappedEncoder<'_> {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self.inner
    }
}

impl EncoderProvider for &CommandBuffer {
    type Encoder<'a>
        = ComputeCommandEncoder
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        self.compute_command_encoder_no_fence()
    }
}

impl EncoderProvider for &ComputeCommandEncoder {
    type Encoder<'a>
        = WrappedEncoder<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        WrappedEncoder {
            inner: self,
            end_encoding_on_drop: false,
        }
    }
}

impl EncoderProvider for &CommandsGuard<'_> {
    type Encoder<'a>
        = &'a CommandsGuard<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        self
    }
}
