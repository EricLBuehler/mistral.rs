use metal::{Buffer, ComputeCommandEncoderRef, ComputePipelineState, MTLSize};
use std::ffi::c_void;

pub(crate) fn get_2d_grid_dims(shape: &[usize], strides: &[usize]) -> MTLSize {
    let mut grid_x: usize = 1;
    let mut grid_y: usize = 1;

    for i in 0..shape.len() {
        if strides[i] == 0 {
            continue;
        }
        if grid_x.saturating_mul(shape[i]) < u32::MAX as usize {
            grid_x *= shape[i];
        } else {
            grid_y *= shape[i];
        }
    }

    if grid_y > u32::MAX as usize || grid_x > u32::MAX as usize {
        panic!("Unable to safely factor shape.");
    }

    if grid_y > grid_x {
        std::mem::swap(&mut grid_x, &mut grid_y);
    }

    MTLSize {
        width: grid_x as u64,
        height: grid_y as u64,
        depth: 1,
    }
}

pub(crate) fn get_2d_grid_dims_divisor(
    shape: &[usize],
    strides: &[usize],
    mut divisor: usize,
) -> MTLSize {
    let mut grid_x: usize = 1;
    let mut grid_y: usize = 1;

    for i in 0..shape.len() {
        if strides[i] == 0 {
            continue;
        }

        // No need to add this shape, we can just remove it from the divisor
        if divisor % shape[i] == 0 {
            divisor /= shape[i];
            continue;
        }

        if grid_x.saturating_mul(shape[i]) < u32::MAX as usize {
            grid_x *= shape[i];
        } else {
            grid_y *= shape[i];
        }

        if divisor > 1 {
            if grid_x % divisor == 0 {
                grid_x /= divisor;
                divisor = 1;
            } else if grid_y % divisor == 0 {
                grid_y /= divisor;
                divisor = 1;
            }
        }
    }

    if grid_y > u32::MAX as usize || grid_x > u32::MAX as usize {
        panic!("Unable to safely factor shape.");
    }

    if grid_y > grid_x {
        std::mem::swap(&mut grid_x, &mut grid_y);
    }

    MTLSize {
        width: grid_x as u64,
        height: grid_y as u64,
        depth: 1,
    }
}

/// Choose a 3‑D thread‑group size whose total thread count is a power‑of‑two
/// (default 2¹⁰ = 1024) while staying within the extents of each dimension.
///
/// This is a direct port of MLX’s `get_block_dims` helper used by the copy
/// kernels.
///
/// * `dim0`, `dim1`, `dim2` – logical extents of the tensor “tile” in each
///   dimension (with `dim0` varying fastest).
/// * `pow2` – desired power‑of‑two for the total number of threads
///   (`10 → 1024`, `9 → 512`, …).
pub(crate) fn get_block_dims(dim0: usize, dim1: usize, dim2: usize, pow2: usize) -> MTLSize {
    let mut pows = [0usize; 3];
    let mut sum = 0usize;

    loop {
        let presum = sum;

        // Try to increment along dim‑0
        if dim0 >= (1usize << (pows[0] + 1)) {
            pows[0] += 1;
            sum += 1;
        }
        if sum == pow2 {
            break;
        }

        // Then along dim‑1
        if dim1 >= (1usize << (pows[1] + 1)) {
            pows[1] += 1;
            sum += 1;
        }
        if sum == pow2 {
            break;
        }

        // Finally along dim‑2
        if dim2 >= (1usize << (pows[2] + 1)) {
            pows[2] += 1;
            sum += 1;
        }

        // If we made no progress, or hit the target thread‑count, stop.
        if sum == presum || sum == pow2 {
            break;
        }
    }

    MTLSize {
        width: 1u64 << pows[0],
        height: 1u64 << pows[1],
        depth: 1u64 << pows[2],
    }
}

/// Most kernels apply similarly across the tensors
/// This creates a strategy that uses the maximum amount of threads per threadgroup (capped at the
/// actual total buffer length).
/// Then kernels can just do their op on their single point in the buffer.
pub(crate) fn linear_split(pipeline: &ComputePipelineState, length: usize) -> (MTLSize, MTLSize) {
    let size = length as u64;
    let width = std::cmp::min(pipeline.max_total_threads_per_threadgroup(), size);
    let count = size.div_ceil(width);
    let thread_group_count = MTLSize {
        width: count,
        height: 1,
        depth: 1,
    };

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    (thread_group_count, thread_group_size)
}

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
            $crate::metal_kernels::utils::set_param($encoder, _index, $param);
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
