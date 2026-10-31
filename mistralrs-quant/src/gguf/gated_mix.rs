use candle_core::{CudaStorage, DType, Result, Shape, Storage, Tensor};
use core::ffi::c_void;

use crate::{
    utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream},
    QuantMethod,
};

use super::{ffi, ggml_dtype_to_uqff_code, q8_1};

const MAX_TOKENS: usize = 1;

#[derive(Clone, Copy, Debug)]
pub enum QuantizedInputTransform {
    Identity,
    ScaledSilu { divisor: f32 },
}

extern "C" {
    fn launch_gguf_quantize_silu_q8_1(
        input: *const c_void,
        output: *mut c_void,
        columns: i32,
        padded_columns: i32,
        divisor: f32,
        dtype: i32,
        stream: *mut c_void,
    );
    fn launch_gguf_gated_mix(
        weights: *const c_void,
        input: *const c_void,
        residual: *const c_void,
        output: *mut c_void,
        rank: i32,
        hidden: i32,
        streams: i32,
        format: i32,
        dtype: i32,
        stream: *mut c_void,
    );
}

/// Projects a gate and reduces sigmoid-weighted residual streams without writing the expanded gate.
pub fn try_fused_quantized_gated_mix(
    input: &Tensor,
    projection: &dyn QuantMethod,
    residual: &Tensor,
    streams: usize,
    transform: QuantizedInputTransform,
) -> Result<Option<Tensor>> {
    if !input.device().is_cuda()
        || projection.has_bias()
        || !crate::group_allowed(input, &[projection])
    {
        return Ok(None);
    }
    let rank = input.dim(candle_core::D::Minus1)?;
    if streams == 0 || rank == 0 || input.elem_count() / rank != MAX_TOKENS {
        return Ok(None);
    }
    let Some(weight) = projection.get_qtensor() else {
        return Ok(None);
    };
    if !rank.is_multiple_of(weight.dtype().block_size()) {
        return Ok(None);
    }
    if !super::fast_mmvq::supports(weight.dtype())
        && weight.dtype() != candle_core::quantized::GgmlDType::Q8_1
    {
        return Ok(None);
    }
    let width = residual.dim(candle_core::D::Minus1)?;
    if width == 0
        || residual.elem_count() != width
        || !width.is_multiple_of(streams)
        || weight.shape().dims() != [width, rank]
        || residual.dtype() != input.dtype()
        || !residual.device().same_device(input.device())
        || !weight.device().same_device(input.device())
    {
        return Ok(None);
    }
    let input = input.contiguous()?;
    let residual = residual.contiguous()?;
    let hidden = width / streams;
    let dev = input.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let stream_ptr = stream.cu_stream() as *mut c_void;
    let (weight_ptr, _weight_guard) = weight.device_ptr_with_guard(&stream)?;
    let padded_rank = rank.next_multiple_of(q8_1::ROW_PADDING);
    let mut scratch =
        unsafe { dev.alloc::<u8>(padded_rank / q8_1::BLOCK_ELEMENTS * q8_1::BLOCK_BYTES)? };
    let (scratch_ptr, _scratch_guard) = slice_ptr_mut_on_stream(&mut scratch, 0, &stream);
    let (input_storage, input_layout) = input.storage_and_layout();
    let Storage::Cuda(input_storage) = &*input_storage else {
        unreachable!()
    };
    let (residual_storage, residual_layout) = residual.storage_and_layout();
    let Storage::Cuda(residual_storage) = &*residual_storage else {
        unreachable!()
    };
    let mut shape = residual.dims().to_vec();
    *shape.last_mut().unwrap() = hidden;
    macro_rules! launch {
        ($ty:ty, $dtype:expr, $quantize:ident) => {{
            let input_slice = input_storage.as_cuda_slice::<$ty>()?;
            let residual_slice = residual_storage.as_cuda_slice::<$ty>()?;
            let mut output = unsafe { dev.alloc::<$ty>(hidden)? };
            {
                let (input_ptr, _input_guard) =
                    slice_ptr_on_stream(input_slice, input_layout.start_offset(), &stream);
                let (residual_ptr, _residual_guard) =
                    slice_ptr_on_stream(residual_slice, residual_layout.start_offset(), &stream);
                let (output_ptr, _output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
                unsafe {
                    match transform {
                        QuantizedInputTransform::Identity => ffi::$quantize(
                            input_ptr as *const c_void,
                            scratch_ptr as *mut c_void,
                            rank as i32,
                            padded_rank as i32,
                            MAX_TOKENS as i32,
                            stream_ptr,
                        ),
                        QuantizedInputTransform::ScaledSilu { divisor } => {
                            launch_gguf_quantize_silu_q8_1(
                                input_ptr as *const c_void,
                                scratch_ptr as *mut c_void,
                                rank as i32,
                                padded_rank as i32,
                                divisor,
                                $dtype,
                                stream_ptr,
                            );
                        }
                    }
                    launch_gguf_gated_mix(
                        weight_ptr as *const c_void,
                        scratch_ptr as *const c_void,
                        residual_ptr as *const c_void,
                        output_ptr as *mut c_void,
                        rank as i32,
                        hidden as i32,
                        streams as i32,
                        ggml_dtype_to_uqff_code(weight.dtype()) as i32,
                        $dtype,
                        stream_ptr,
                    );
                }
            }
            Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(output, dev.clone())),
                Shape::from(shape),
            ))
        }};
    }
    Ok(Some(match input.dtype() {
        DType::F16 => launch!(half::f16, 0, launch_mmvq_gguf_quantize_q8_1_f16),
        DType::BF16 => launch!(half::bf16, 1, launch_mmvq_gguf_quantize_q8_1_bf16),
        DType::F32 => launch!(f32, 2, launch_mmvq_gguf_quantize_q8_1_f32),
        _ => unreachable!(),
    }))
}
