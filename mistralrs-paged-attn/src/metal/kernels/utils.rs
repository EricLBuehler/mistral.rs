use metal::{Buffer, ComputeCommandEncoderRef};
use std::ffi::c_void;

pub fn set_param<P: EncoderParam>(encoder: &ComputeCommandEncoderRef, position: u64, data: P) {
    <P as EncoderParam>::set_param(encoder, position, data)
}

/// Helper functions to create the various objects on the compute command encoder
/// on a single line.
/// Prevents getting wrong some arguments number and mixing length and size in bytes.
pub trait EncoderParam {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self);
}
macro_rules! primitive {
    ($type:ty) => {
        impl EncoderParam for $type {
            fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
                encoder.set_bytes(
                    position,
                    core::mem::size_of::<$type>() as u64,
                    &data as *const $type as *const c_void,
                );
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
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_bytes(
            position,
            core::mem::size_of_val(data) as u64,
            data.as_ptr() as *const c_void,
        );
    }
}

impl EncoderParam for &Buffer {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1 as u64);
    }
}

impl EncoderParam for &BufferOffset<'_> {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data.buffer), data.offset_in_bytes as u64);
    }
}

impl EncoderParam for &mut Buffer {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data), 0);
    }
}

impl EncoderParam for (&mut Buffer, usize) {
    fn set_param(encoder: &ComputeCommandEncoderRef, position: u64, data: Self) {
        encoder.set_buffer(position, Some(data.0), data.1 as u64);
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
    type Encoder<'a>: AsRef<metal::ComputeCommandEncoderRef>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_>;
}

pub struct WrappedEncoder<'a> {
    inner: &'a ComputeCommandEncoderRef,
    end_encoding_on_drop: bool,
}

impl Drop for WrappedEncoder<'_> {
    fn drop(&mut self) {
        if self.end_encoding_on_drop {
            self.inner.end_encoding()
        }
    }
}

impl AsRef<metal::ComputeCommandEncoderRef> for WrappedEncoder<'_> {
    fn as_ref(&self) -> &metal::ComputeCommandEncoderRef {
        self.inner
    }
}

impl EncoderProvider for &metal::CommandBuffer {
    type Encoder<'a>
        = WrappedEncoder<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        WrappedEncoder {
            inner: self.new_compute_command_encoder(),
            end_encoding_on_drop: true,
        }
    }
}

impl EncoderProvider for &metal::CommandBufferRef {
    type Encoder<'a>
        = WrappedEncoder<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        WrappedEncoder {
            inner: self.new_compute_command_encoder(),
            end_encoding_on_drop: true,
        }
    }
}

impl EncoderProvider for &ComputeCommandEncoderRef {
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
