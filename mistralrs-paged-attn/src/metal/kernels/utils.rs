use candle_metal_kernels::metal::{Buffer, CommandBuffer, ComputeCommandEncoder};
use std::ffi::c_void;

pub trait RawBytesEncoder {
    fn set_bytes_raw(&self, index: usize, length: usize, bytes: *const c_void);
}

impl RawBytesEncoder for ComputeCommandEncoder {
    fn set_bytes_raw(&self, index: usize, length: usize, bytes: *const c_void) {
        self.set_bytes_directly(index, length, bytes);
    }
}

pub fn set_param<P: EncoderParam>(encoder: &ComputeCommandEncoder, position: usize, data: P) {
    <P as EncoderParam>::set_param(encoder, position, data)
}

/// Helper functions to create the various objects on the compute command encoder
/// on a single line.
/// Prevents getting wrong some arguments number and mixing length and size in bytes.
pub trait EncoderParam {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self);
}
macro_rules! primitive {
    ($type:ty) => {
        impl EncoderParam for $type {
            fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
                encoder.set_bytes(position, &data);
            }
        }
    };
}
primitive!(bool);
primitive!(usize);
primitive!(i32);
primitive!(i64);
primitive!(u32);
primitive!(u64);
primitive!(f32);

pub struct BufferOffset<'a> {
    pub buffer: &'a Buffer,
    pub offset_in_bytes: usize,
}

impl<T> EncoderParam for &[T] {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes_directly(position, core::mem::size_of_val(data), data.as_ptr().cast());
    }
}

impl EncoderParam for &Buffer {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1);
    }
}

impl EncoderParam for &BufferOffset<'_> {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data.buffer), data.offset_in_bytes);
    }
}

impl EncoderParam for &mut Buffer {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&mut Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1);
    }
}

#[macro_export]
macro_rules! set_params {
    ($encoder:ident, ($($param:expr),+)) => (
        let mut _index = 0;
        $(
            $crate::metal::kernels::utils::set_param($encoder, _index, $param);
            _index += 1;
        )*
    );
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
        self.compute_command_encoder()
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
