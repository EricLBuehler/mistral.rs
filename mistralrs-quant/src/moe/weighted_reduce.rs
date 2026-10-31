use candle_core::{CudaStorage, DType, Result, Storage, Tensor};
use half::{bf16, f16};

use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

#[repr(C)]
struct WeightedReduce {
    input: *const std::ffi::c_void,
    weights: *const f32,
    output: *mut std::ffi::c_void,
    tokens: usize,
    hidden: usize,
    topk: usize,
    input_type: i32,
    output_type: i32,
}

unsafe extern "C" {
    fn mistralrs_moe_weighted_reduce(
        args: *const WeightedReduce,
        stream: candle_core::cuda::cudarc::driver::sys::CUstream,
    ) -> i32;
}

fn dtype_code(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        _ => candle_core::bail!("MoE weighted reduction requires F16, BF16 or F32 values"),
    }
}

pub fn moe_weighted_reduce(input: &Tensor, weights: &Tensor, dtype: DType) -> Result<Tensor> {
    let (tokens, topk, hidden) = input.dims3()?;
    if weights.dims() != [tokens, topk]
        || weights.dtype() != DType::F32
        || !input.device().same_device(weights.device())
    {
        candle_core::bail!(
            "MoE weighted reduction requires F32 [tokens, topk] weights on the input device"
        );
    }
    let input_type = dtype_code(input.dtype())?;
    let output_type = dtype_code(dtype)?;
    if tokens == 0 || hidden == 0 {
        return Tensor::zeros((tokens, hidden), dtype, input.device());
    }
    let input = input.contiguous()?;
    let weights = weights.contiguous()?;
    let dev = input.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let (input_storage, input_layout) = input.storage_and_layout();
    let (weight_storage, weight_layout) = weights.storage_and_layout();
    let (Storage::Cuda(input_storage), Storage::Cuda(weight_storage)) =
        (&*input_storage, &*weight_storage)
    else {
        candle_core::bail!("MoE weighted reduction requires CUDA tensors")
    };
    macro_rules! input_pointer {
        ($ty:ty) => {
            slice_ptr_on_stream(
                input_storage.as_cuda_slice::<$ty>()?,
                input_layout.start_offset(),
                &stream,
            )
        };
    }
    let (input_ptr, input_guard) = match input.dtype() {
        DType::F16 => input_pointer!(f16),
        DType::BF16 => input_pointer!(bf16),
        DType::F32 => input_pointer!(f32),
        _ => unreachable!(),
    };
    let (weight_ptr, weight_guard) = slice_ptr_on_stream(
        weight_storage.as_cuda_slice::<f32>()?,
        weight_layout.start_offset(),
        &stream,
    );
    macro_rules! launch {
        ($ty:ty) => {{
            let mut output = unsafe { dev.alloc::<$ty>(tokens * hidden)? };
            let (output_ptr, output_guard) = slice_ptr_mut_on_stream(&mut output, 0, &stream);
            let status = unsafe {
                mistralrs_moe_weighted_reduce(
                    &WeightedReduce {
                        input: input_ptr as *const _,
                        weights: weight_ptr as *const f32,
                        output: output_ptr as *mut _,
                        tokens,
                        hidden,
                        topk,
                        input_type,
                        output_type,
                    },
                    stream.cu_stream(),
                )
            };
            if status != 0 {
                candle_core::bail!("MoE weighted reduction CUDA launch failed with error {status}");
            }
            drop(output_guard);
            CudaStorage::wrap_cuda_slice(output, dev.clone())
        }};
    }
    let output = match dtype {
        DType::F16 => launch!(f16),
        DType::BF16 => launch!(bf16),
        DType::F32 => launch!(f32),
        _ => unreachable!(),
    };
    drop((input_guard, weight_guard));
    Ok(Tensor::from((Storage::Cuda(output), (tokens, hidden))))
}
