use candle_core::{
    cuda::cudarc::driver::{DevicePtr, DevicePtrMut},
    CudaStorage, DType, Result, Storage, Tensor,
};
use std::ffi::c_void;

#[repr(C)]
struct NormRopeArgs {
    input: *const c_void,
    weight: *const c_void,
    cos: *const c_void,
    sin: *const c_void,
    output: *mut c_void,
    batch: i32,
    sequence: i32,
    heads: i32,
    columns: i32,
    half_rot: i32,
    strides: [i64; 4],
    epsilon: f32,
}
unsafe extern "C" {
    fn launch_rounded_norm_rope(args: NormRopeArgs, dtype: i32, stream: *mut c_void);
}

pub(crate) fn try_rms_norm_partial_rotary_neox(
    input: &Tensor,
    weight: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    epsilon: f32,
) -> Result<Option<Tensor>> {
    if !input.device().is_cuda()
        || !(2..=4).contains(&input.rank())
        || cos.rank() + 1 != input.rank()
        || !matches!(input.dtype(), DType::BF16 | DType::F16 | DType::F32)
        || [weight, cos, sin].iter().any(|tensor| {
            !tensor.is_contiguous()
                || tensor.dtype() != input.dtype()
                || !tensor.device().same_device(input.device())
        })
    {
        return Ok(None);
    }
    let rank = input.rank();
    let mut dimensions = [1; 4];
    dimensions[4 - rank..].copy_from_slice(input.dims());
    let [batch, sequence, heads, columns] = dimensions;
    let half_rot = cos.dim(cos.rank() - 1)?;
    if weight.dims() != [columns]
        || cos.dims()[..cos.rank() - 1] != input.dims()[..rank - 2]
        || sin.dims() != cos.dims()
        || half_rot == 0
        || half_rot > columns / 2
        || input.elem_count() == 0
        || input.elem_count() > i32::MAX as usize
        || input
            .stride()
            .iter()
            .any(|&stride| i64::try_from(stride).is_err())
    {
        return Ok(None);
    }
    let dev = input.device().as_cuda_device()?;
    let stream = dev.cuda_stream();
    let (input_s, input_l) = input.storage_and_layout();
    let (weight_s, weight_l) = weight.storage_and_layout();
    let (cos_s, cos_l) = cos.storage_and_layout();
    let (sin_s, sin_l) = sin.storage_and_layout();
    let (
        Storage::Cuda(input_s),
        Storage::Cuda(weight_s),
        Storage::Cuda(cos_s),
        Storage::Cuda(sin_s),
    ) = (&*input_s, &*weight_s, &*cos_s, &*sin_s)
    else {
        unreachable!()
    };
    macro_rules! launch {
        ($ty:ty, $dtype:expr) => {{
            let (input_p, _ig) = input_s.as_cuda_slice::<$ty>()?.device_ptr(&stream);
            let (weight_p, _wg) = weight_s.as_cuda_slice::<$ty>()?.device_ptr(&stream);
            let (cos_p, _cg) = cos_s.as_cuda_slice::<$ty>()?.device_ptr(&stream);
            let (sin_p, _sg) = sin_s.as_cuda_slice::<$ty>()?.device_ptr(&stream);
            let mut out = unsafe { dev.alloc::<$ty>(input.elem_count())? };
            {
                let (out_p, _og) = out.device_ptr_mut(&stream);
                let bytes = std::mem::size_of::<$ty>() as u64;
                let args = NormRopeArgs {
                    input: (input_p + input_l.start_offset() as u64 * bytes) as *const _,
                    weight: (weight_p + weight_l.start_offset() as u64 * bytes) as *const _,
                    cos: (cos_p + cos_l.start_offset() as u64 * bytes) as *const _,
                    sin: (sin_p + sin_l.start_offset() as u64 * bytes) as *const _,
                    output: out_p as *mut _,
                    batch: i32::try_from(batch).unwrap(),
                    sequence: i32::try_from(sequence).unwrap(),
                    heads: i32::try_from(heads).unwrap(),
                    columns: i32::try_from(columns).unwrap(),
                    half_rot: i32::try_from(half_rot).unwrap(),
                    strides: std::array::from_fn(|index| {
                        index
                            .checked_sub(4 - rank)
                            .map_or(0, |index| input_l.stride()[index] as i64)
                    }),
                    epsilon,
                };
                unsafe { launch_rounded_norm_rope(args, $dtype, stream.cu_stream() as *mut _) };
            }
            Ok(Some(Tensor::from((
                Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
                input.shape().clone(),
            ))))
        }};
    }
    match input.dtype() {
        DType::F32 => launch!(f32, 0),
        DType::F16 => launch!(half::f16, 1),
        DType::BF16 => launch!(half::bf16, 2),
        _ => unreachable!(),
    }
}
