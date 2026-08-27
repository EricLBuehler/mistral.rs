use std::ffi::c_void;

use candle_core::backend::BackendStorage;
use candle_core::cuda_backend::cudarc::driver::{DevicePtr, DevicePtrMut};
use candle_core::cuda_backend::{CudaStorage, CudaStorageSlice};
use candle_core::{DType, Result, Shape, Tensor};

const MAX_DFLASH_CONTEXT_TAPS: usize = 64;

unsafe extern "C" {
    fn dflash_pack_taps_f16(
        inputs: *const *const c_void,
        widths: *const i32,
        taps: i32,
        row_indices: *const c_void,
        output: *mut c_void,
        output_rows: i32,
        output_width: i32,
        row_start: i32,
        stream: i64,
    ) -> i32;

    fn dflash_pack_taps_bf16(
        inputs: *const *const c_void,
        widths: *const i32,
        taps: i32,
        row_indices: *const c_void,
        output: *mut c_void,
        output_rows: i32,
        output_width: i32,
        row_start: i32,
        stream: i64,
    ) -> i32;

    fn dflash_pack_taps_f32(
        inputs: *const *const c_void,
        widths: *const i32,
        taps: i32,
        row_indices: *const c_void,
        output: *mut c_void,
        output_rows: i32,
        output_width: i32,
        row_start: i32,
        stream: i64,
    ) -> i32;

    fn dflash_context_keys_f16(
        input: *const c_void,
        norm_weights: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const c_void,
        output: *mut c_void,
        layers: i32,
        heads: i32,
        rows: i32,
        head_dim: i32,
        rot_dim: i32,
        eps: f32,
        stream: i64,
    ) -> i32;

    fn dflash_context_keys_bf16(
        input: *const c_void,
        norm_weights: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const c_void,
        output: *mut c_void,
        layers: i32,
        heads: i32,
        rows: i32,
        head_dim: i32,
        rot_dim: i32,
        eps: f32,
        stream: i64,
    ) -> i32;

    fn dflash_context_keys_f32(
        input: *const c_void,
        norm_weights: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        positions: *const c_void,
        output: *mut c_void,
        layers: i32,
        heads: i32,
        rows: i32,
        head_dim: i32,
        rot_dim: i32,
        eps: f32,
        stream: i64,
    ) -> i32;
}

fn contiguous_row_range(indices: &[u32]) -> Option<(usize, usize)> {
    let start = usize::try_from(*indices.first()?).ok()?;
    indices
        .iter()
        .enumerate()
        .all(|(offset, index)| usize::try_from(*index).ok() == start.checked_add(offset))
        .then_some((start, indices.len()))
}

pub(crate) fn pack_taps(taps: &[Tensor], row_indices: &[u32]) -> Result<Option<Tensor>> {
    if taps.len() <= 1
        || taps.len() > MAX_DFLASH_CONTEXT_TAPS
        || row_indices.is_empty()
        || !taps[0].device().is_cuda()
    {
        return Ok(None);
    }
    let dtype = taps[0].dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || taps.iter().any(|tap| {
            tap.dtype() != dtype
                || !tap.device().same_device(taps[0].device())
                || !tap.is_contiguous()
        })
    {
        return Ok(None);
    }

    let (source_batch, source_rows, _) = taps[0].dims3()?;
    let source_len = source_batch
        .checked_mul(source_rows)
        .ok_or_else(|| candle_core::Error::msg("DFlash context tap row count overflow"))?;
    if row_indices
        .iter()
        .any(|index| usize::try_from(*index).map_or(true, |index| index >= source_len))
    {
        return Ok(None);
    }
    let mut widths = Vec::with_capacity(taps.len());
    let mut output_width = 0usize;
    for tap in taps {
        let (batch, rows, width) = tap.dims3()?;
        if (batch, rows) != (source_batch, source_rows) || width == 0 {
            return Ok(None);
        }
        output_width = output_width
            .checked_add(width)
            .ok_or_else(|| candle_core::Error::msg("DFlash context tap width overflow"))?;
        widths.push(i32::try_from(width).map_err(candle_core::Error::wrap)?);
    }

    let contiguous = contiguous_row_range(row_indices);
    let indices = if contiguous.is_none() {
        Some(Tensor::from_vec(
            row_indices.to_vec(),
            (row_indices.len(),),
            taps[0].device(),
        )?)
    } else {
        None
    };
    let row_start = contiguous.map_or(0, |(start, _)| start);
    if row_start > source_len || row_indices.len() > source_len.saturating_sub(row_start) {
        return Ok(None);
    }

    let storage_layouts = taps
        .iter()
        .map(Tensor::storage_and_layout)
        .collect::<Vec<_>>();
    let first_storage = match &*storage_layouts[0].0 {
        candle_core::Storage::Cuda(storage) => storage,
        _ => return Ok(None),
    };
    let dev = first_storage.device();
    let stream = dev.cuda_stream();
    let elements = row_indices
        .len()
        .checked_mul(output_width)
        .ok_or_else(|| candle_core::Error::msg("DFlash packed tap size overflow"))?;
    let output_shape = Shape::from_dims(&[row_indices.len(), output_width]);
    let output_rows = i32::try_from(row_indices.len()).map_err(candle_core::Error::wrap)?;
    let output_width_i32 = i32::try_from(output_width).map_err(candle_core::Error::wrap)?;
    let row_start = i32::try_from(row_start).map_err(candle_core::Error::wrap)?;
    let taps_len = i32::try_from(taps.len()).map_err(candle_core::Error::wrap)?;

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi:ident) => {{
            let mut pointers = Vec::with_capacity(taps.len());
            let mut input_guards = Vec::with_capacity(taps.len());
            for (storage, layout) in &storage_layouts {
                let candle_core::Storage::Cuda(storage) = &**storage else {
                    return Ok(None);
                };
                let CudaStorageSlice::$variant(slice) = &storage.slice else {
                    return Ok(None);
                };
                let (pointer, guard) = slice.device_ptr(&stream);
                let pointer =
                    unsafe { (pointer as *const $ty).add(layout.start_offset()) as *const c_void };
                pointers.push(pointer);
                input_guards.push(guard);
            }

            let indices_storage_layout = indices.as_ref().map(Tensor::storage_and_layout);
            let mut indices_guard = None;
            let indices_pointer = if let Some((storage, layout)) = &indices_storage_layout {
                let candle_core::Storage::Cuda(storage) = &**storage else {
                    return Ok(None);
                };
                let CudaStorageSlice::U32(slice) = &storage.slice else {
                    return Ok(None);
                };
                let (pointer, guard) = slice.device_ptr(&stream);
                indices_guard = Some(guard);
                unsafe { (pointer as *const u32).add(layout.start_offset()) as *const c_void }
            } else {
                std::ptr::null()
            };

            let mut output = unsafe { dev.alloc::<$ty>(elements) }?;
            let (output_pointer, output_guard) = output.device_ptr_mut(&stream);
            let status = unsafe {
                $ffi(
                    pointers.as_ptr(),
                    widths.as_ptr(),
                    taps_len,
                    indices_pointer,
                    output_pointer as *mut c_void,
                    output_rows,
                    output_width_i32,
                    row_start,
                    stream.cu_stream() as i64,
                )
            };
            drop(output_guard);
            drop(indices_guard);
            drop(input_guards);
            if status != 0 {
                candle_core::bail!("dflash_pack_taps failed with status {status}");
            }
            let storage = CudaStorage {
                slice: CudaStorageSlice::$variant(output),
                device: dev.clone(),
            };
            Ok(Some(Tensor::from((
                candle_core::Storage::Cuda(storage),
                output_shape,
            ))))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, dflash_pack_taps_bf16),
        DType::F16 => launch!(F16, half::f16, dflash_pack_taps_f16),
        DType::F32 => launch!(F32, f32, dflash_pack_taps_f32),
        _ => Ok(None),
    }
}

pub(crate) fn context_keys(
    input: &Tensor,
    norm_weights: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
    eps: f32,
) -> Result<Option<Tensor>> {
    if !input.device().is_cuda() || !input.is_contiguous() {
        return Ok(None);
    }
    let dtype = input.dtype();
    if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32)
        || [norm_weights, cos, sin].iter().any(|tensor| {
            tensor.dtype() != dtype
                || !tensor.device().same_device(input.device())
                || !tensor.is_contiguous()
        })
        || positions.dtype() != DType::U32
        || !positions.device().same_device(input.device())
        || !positions.is_contiguous()
    {
        return Ok(None);
    }

    let (layers, heads, rows, head_dim) = input.dims4()?;
    if norm_weights.dims2()? != (layers, head_dim) || positions.dims1()? != rows {
        return Ok(None);
    }
    let (rope_rows, rot_dim) = cos.dims2()?;
    if sin.dims2()? != (rope_rows, rot_dim) || rot_dim == 0 || rot_dim * 2 > head_dim {
        return Ok(None);
    }

    let tensors = [input, norm_weights, cos, sin, positions];
    let storage_layouts = tensors
        .iter()
        .map(|tensor| tensor.storage_and_layout())
        .collect::<Vec<_>>();
    let input_storage = match &*storage_layouts[0].0 {
        candle_core::Storage::Cuda(storage) => storage,
        _ => return Ok(None),
    };
    let dev = input_storage.device();
    let stream = dev.cuda_stream();
    let output_shape = input.shape().clone();
    let layers = i32::try_from(layers).map_err(candle_core::Error::wrap)?;
    let heads = i32::try_from(heads).map_err(candle_core::Error::wrap)?;
    let rows = i32::try_from(rows).map_err(candle_core::Error::wrap)?;
    let head_dim = i32::try_from(head_dim).map_err(candle_core::Error::wrap)?;
    let rot_dim = i32::try_from(rot_dim).map_err(candle_core::Error::wrap)?;

    macro_rules! launch {
        ($variant:ident, $ty:ty, $ffi:ident) => {{
            let mut pointers = Vec::with_capacity(4);
            let mut guards = Vec::with_capacity(5);
            for (storage, layout) in &storage_layouts[..4] {
                let candle_core::Storage::Cuda(storage) = &**storage else {
                    return Ok(None);
                };
                let CudaStorageSlice::$variant(slice) = &storage.slice else {
                    return Ok(None);
                };
                let (pointer, guard) = slice.device_ptr(&stream);
                pointers.push(unsafe {
                    (pointer as *const $ty).add(layout.start_offset()) as *const c_void
                });
                guards.push(guard);
            }
            let candle_core::Storage::Cuda(position_storage) = &*storage_layouts[4].0 else {
                return Ok(None);
            };
            let CudaStorageSlice::U32(position_slice) = &position_storage.slice else {
                return Ok(None);
            };
            let (position_pointer, position_guard) = position_slice.device_ptr(&stream);
            let position_pointer = unsafe {
                (position_pointer as *const u32).add(storage_layouts[4].1.start_offset())
                    as *const c_void
            };

            let mut output = unsafe { dev.alloc::<$ty>(input.elem_count()) }?;
            let (output_pointer, output_guard) = output.device_ptr_mut(&stream);
            let status = unsafe {
                $ffi(
                    pointers[0],
                    pointers[1],
                    pointers[2],
                    pointers[3],
                    position_pointer,
                    output_pointer as *mut c_void,
                    layers,
                    heads,
                    rows,
                    head_dim,
                    rot_dim,
                    eps,
                    stream.cu_stream() as i64,
                )
            };
            drop(output_guard);
            drop(position_guard);
            drop(guards);
            if status != 0 {
                candle_core::bail!("dflash_context_keys failed with status {status}");
            }
            let storage = CudaStorage {
                slice: CudaStorageSlice::$variant(output),
                device: dev.clone(),
            };
            Ok(Some(Tensor::from((
                candle_core::Storage::Cuda(storage),
                output_shape,
            ))))
        }};
    }

    match dtype {
        DType::BF16 => launch!(BF16, half::bf16, dflash_context_keys_bf16),
        DType::F16 => launch!(F16, half::f16, dflash_context_keys_f16),
        DType::F32 => launch!(F32, f32, dflash_context_keys_f32),
        _ => Ok(None),
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use candle_core::{DType, Device, IndexOp, Result, Tensor};

    use super::{context_keys, pack_taps};

    #[test]
    #[ignore = "requires CUDA"]
    fn packed_taps_match_cuda_reference() -> Result<()> {
        let device = Device::new_cuda(0)?;
        for indices in [vec![2u32, 3, 4], vec![4u32, 1, 5]] {
            let first = Tensor::arange(0f32, 48f32, &device)?
                .reshape((2, 3, 8))?
                .to_dtype(DType::BF16)?;
            let second = Tensor::arange(48f32, 84f32, &device)?
                .reshape((2, 3, 6))?
                .to_dtype(DType::BF16)?;
            let actual =
                pack_taps(&[first.clone(), second.clone()], &indices)?.expect("CUDA pack path");
            let index = Tensor::from_vec(indices, (3,), &device)?;
            let first = first.reshape((6, 8))?.index_select(&index, 0)?;
            let second = second.reshape((6, 6))?.index_select(&index, 0)?;
            let expected = Tensor::cat(&[first, second], 1)?;
            assert_eq!(
                actual.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                expected.to_dtype(DType::F32)?.to_vec2::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn context_keys_match_layerwise_reference() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let layers = 3;
        let heads = 2;
        let rows = 4;
        let head_dim = 8;
        let rot_dim = head_dim / 2;
        let eps = 1e-6f32;
        let input = Tensor::arange(1f32, (layers * heads * rows * head_dim + 1) as f32, &device)?
            .affine(0.01, -0.5)?
            .reshape((layers, heads, rows, head_dim))?
            .to_dtype(DType::BF16)?;
        let weights = Tensor::arange(1f32, (layers * head_dim + 1) as f32, &device)?
            .affine(0.02, 0.5)?
            .reshape((layers, head_dim))?
            .to_dtype(DType::BF16)?;
        let positions = Tensor::from_vec(vec![1u32, 4, 2, 7], (rows,), &device)?;
        let angles = Tensor::arange(0f32, 8f32, &device)?
            .reshape((8, 1))?
            .broadcast_mul(&Tensor::from_vec(
                vec![1f32, 0.5, 0.25, 0.125],
                (1, rot_dim),
                &device,
            )?)?;
        let cos = angles.cos()?.to_dtype(DType::BF16)?;
        let sin = angles.sin()?.to_dtype(DType::BF16)?;
        let actual = context_keys(&input, &weights, &cos, &sin, &positions, eps)?
            .expect("CUDA context key path");

        let position_cos = cos.index_select(&positions, 0)?;
        let position_sin = sin.index_select(&positions, 0)?;
        let mut expected = Vec::with_capacity(layers);
        for layer in 0..layers {
            let normalized =
                candle_nn::ops::rms_norm(&input.i(layer)?.contiguous()?, &weights.i(layer)?, eps)?;
            expected.push(
                candle_nn::rotary_emb::rope(
                    &normalized.unsqueeze(0)?,
                    &position_cos,
                    &position_sin,
                )?
                .squeeze(0)?,
            );
        }
        let expected = Tensor::stack(&expected, 0)?;
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 0.02, "{actual} != {expected}");
        }
        Ok(())
    }
}
