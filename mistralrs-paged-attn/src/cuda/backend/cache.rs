use std::{collections::HashMap, iter::zip, mem::size_of, sync::Arc};

use crate::cuda::backend::{slice_ptr, slice_ptr_on_stream};
use crate::cuda::ffi::{copy_blocks_bf16, copy_blocks_f16, copy_blocks_f32, copy_blocks_u8};
use candle_core::backend::BackendDevice;
use candle_core::cuda_backend::cudarc::driver::sys::CUstreamCaptureStatus;
use candle_core::cuda_backend::cudarc::driver::CudaStream;
use candle_core::cuda_backend::CudaStorageSlice;
use candle_core::Result;
use candle_core::{cuda_backend::cudarc::driver::CudaSlice, Device, Storage, Tensor};

fn ensure_allocation_stream<T>(
    slice: &CudaSlice<T>,
    stream: &Arc<CudaStream>,
    layer: usize,
    cache: &str,
) -> Result<()> {
    if !Arc::ptr_eq(slice.stream(), stream) {
        candle_core::bail!(
            "copy_blocks {cache} cache layer {layer} was allocated on a different CUDA stream"
        );
    }
    Ok(())
}

pub fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: &HashMap<usize, Vec<usize>>,
) -> Result<()> {
    if key_caches.is_empty() {
        if value_caches.is_empty() {
            return Ok(());
        }
        candle_core::bail!("copy_blocks requires the same number of key and value caches");
    }
    if key_caches.len() != value_caches.len() {
        candle_core::bail!("copy_blocks requires the same number of key and value caches");
    }
    if block_mapping.values().all(Vec::is_empty) {
        return Ok(());
    }

    let cache_dev = key_caches[0].device();
    let Device::Cuda(dev) = cache_dev else {
        candle_core::bail!("copy_blocks requires CUDA caches")
    };
    let stream = dev.cuda_stream();
    let capture_status = stream
        .capture_status()
        .map_err(|error| candle_core::Error::Cuda(Box::new(error)))?;
    if capture_status != CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE {
        candle_core::bail!("copy_blocks cannot run during CUDA graph capture");
    }

    let dtype = key_caches[0].dtype();
    if key_caches[0].rank() == 0 || value_caches[0].rank() == 0 {
        candle_core::bail!("copy_blocks caches must have a block dimension");
    }
    let numel_per_block_key = key_caches[0].dims()[1..].iter().product::<usize>();
    let numel_per_block_value = value_caches[0].dims()[1..].iter().product::<usize>();
    let num_layers = i32::try_from(key_caches.len())?;

    let mut key_cache_ptrs = Vec::new();
    key_cache_ptrs.reserve_exact(num_layers as usize);
    let mut value_cache_ptrs = Vec::new();
    value_cache_ptrs.reserve_exact(num_layers as usize);

    for (layer, (key_cache, value_cache)) in zip(&key_caches, &value_caches).enumerate() {
        let (Device::Cuda(key_dev), Device::Cuda(value_dev)) =
            (key_cache.device(), value_cache.device())
        else {
            candle_core::bail!("copy_blocks cache layer {layer} is not on CUDA");
        };
        if !cache_dev.same_device(key_cache.device())
            || !cache_dev.same_device(value_cache.device())
        {
            candle_core::bail!("copy_blocks cache layer {layer} is on a different CUDA device");
        }
        if !Arc::ptr_eq(&stream, &key_dev.cuda_stream())
            || !Arc::ptr_eq(&stream, &value_dev.cuda_stream())
        {
            candle_core::bail!("copy_blocks cache layer {layer} uses a different CUDA stream");
        }
        if key_cache.dtype() != dtype || value_cache.dtype() != dtype {
            candle_core::bail!("copy_blocks cache layer {layer} has a mismatched dtype");
        }
        if key_cache.rank() == 0 || value_cache.rank() == 0 {
            candle_core::bail!("copy_blocks cache layer {layer} must have a block dimension");
        }
        if !key_cache.is_contiguous() || !value_cache.is_contiguous() {
            candle_core::bail!("copy_blocks cache layer {layer} must be contiguous");
        }
        if key_cache.dims()[1..].iter().product::<usize>() != numel_per_block_key
            || value_cache.dims()[1..].iter().product::<usize>() != numel_per_block_value
        {
            candle_core::bail!("copy_blocks cache layer {layer} has a mismatched block shape");
        }
        for (src_block, dst_blocks) in block_mapping {
            if *src_block >= key_cache.dims()[0] || *src_block >= value_cache.dims()[0] {
                candle_core::bail!(
                    "copy_blocks source block {src_block} is out of bounds for cache layer {layer}"
                );
            }
            if let Some(dst_block) = dst_blocks
                .iter()
                .find(|dst| **dst >= key_cache.dims()[0] || **dst >= value_cache.dims()[0])
            {
                candle_core::bail!(
                    "copy_blocks destination block {dst_block} is out of bounds for cache layer {layer}"
                );
            }
        }
    }

    let key_storage_layouts = key_caches
        .iter()
        .map(|cache| cache.storage_and_layout())
        .collect::<Vec<_>>();
    let value_storage_layouts = value_caches
        .iter()
        .map(|cache| cache.storage_and_layout())
        .collect::<Vec<_>>();
    let mut cache_guards = Vec::with_capacity(num_layers as usize * 2);
    for (layer, ((key_storage, key_layout), (value_storage, value_layout))) in
        zip(&key_storage_layouts, &value_storage_layouts).enumerate()
    {
        let Storage::Cuda(key_storage) = &**key_storage else {
            unreachable!()
        };
        let Storage::Cuda(value_storage) = &**value_storage else {
            unreachable!()
        };

        let (key_ptr, value_ptr, key_guard, value_guard) =
            match (&key_storage.slice, &value_storage.slice) {
                (CudaStorageSlice::BF16(slice_key), CudaStorageSlice::BF16(slice_value)) => {
                    ensure_allocation_stream(slice_key, &stream, layer, "key")?;
                    ensure_allocation_stream(slice_value, &stream, layer, "value")?;
                    let (ptr_key, key_guard) = slice_ptr_on_stream(slice_key, 0, &stream);
                    let (ptr_value, value_guard) = slice_ptr_on_stream(slice_value, 0, &stream);
                    (ptr_key, ptr_value, key_guard, value_guard)
                }
                (CudaStorageSlice::F16(slice_key), CudaStorageSlice::F16(slice_value)) => {
                    ensure_allocation_stream(slice_key, &stream, layer, "key")?;
                    ensure_allocation_stream(slice_value, &stream, layer, "value")?;
                    let (ptr_key, key_guard) = slice_ptr_on_stream(slice_key, 0, &stream);
                    let (ptr_value, value_guard) = slice_ptr_on_stream(slice_value, 0, &stream);
                    (ptr_key, ptr_value, key_guard, value_guard)
                }
                (CudaStorageSlice::F32(slice_key), CudaStorageSlice::F32(slice_value)) => {
                    ensure_allocation_stream(slice_key, &stream, layer, "key")?;
                    ensure_allocation_stream(slice_value, &stream, layer, "value")?;
                    let (ptr_key, key_guard) = slice_ptr_on_stream(slice_key, 0, &stream);
                    let (ptr_value, value_guard) = slice_ptr_on_stream(slice_value, 0, &stream);
                    (ptr_key, ptr_value, key_guard, value_guard)
                }
                (CudaStorageSlice::F8E4M3(slice_key), CudaStorageSlice::F8E4M3(slice_value)) => {
                    ensure_allocation_stream(slice_key, &stream, layer, "key")?;
                    ensure_allocation_stream(slice_value, &stream, layer, "value")?;
                    let (ptr_key, key_guard) = slice_ptr_on_stream(slice_key, 0, &stream);
                    let (ptr_value, value_guard) = slice_ptr_on_stream(slice_value, 0, &stream);
                    (ptr_key, ptr_value, key_guard, value_guard)
                }
                _ => {
                    candle_core::bail!(
                        "only f32, f16, bf16 and f8e4m3 input data types are supported"
                    );
                }
            };
        cache_guards.push(key_guard);
        cache_guards.push(value_guard);
        let element_size = dtype.size_in_bytes() as u64;
        key_cache_ptrs
            .push((key_ptr + u64::try_from(key_layout.start_offset())? * element_size) as i64);
        value_cache_ptrs
            .push((value_ptr + u64::try_from(value_layout.start_offset())? * element_size) as i64);
    }

    let mut block_mapping_vec: Vec<i64> = Vec::new();
    for (src_block_number, dst_blocks) in block_mapping {
        for dst_block_number in dst_blocks {
            block_mapping_vec.push(i64::try_from(*src_block_number)?);
            block_mapping_vec.push(i64::try_from(*dst_block_number)?);
        }
    }
    let num_pairs = i32::try_from(block_mapping_vec.len() / 2)?;

    let value_ptr_offset = key_cache_ptrs.len();
    let mapping_offset = value_ptr_offset + value_cache_ptrs.len();
    key_cache_ptrs.extend(value_cache_ptrs);
    key_cache_ptrs.extend(block_mapping_vec);
    let device_metadata = stream
        .clone_htod(&key_cache_ptrs)
        .map_err(|error| candle_core::Error::Cuda(Box::new(error)))?;
    let (metadata_ptr, _metadata_guard) = slice_ptr_on_stream(&device_metadata, 0, &stream);
    let key_cache_ptr = metadata_ptr as *mut core::ffi::c_void;
    let value_cache_ptr =
        (metadata_ptr + (value_ptr_offset * size_of::<i64>()) as u64) as *mut core::ffi::c_void;
    let block_mapping_ptr =
        (metadata_ptr + (mapping_offset * size_of::<i64>()) as u64) as *const core::ffi::c_void;
    let numel_per_block_key = i32::try_from(numel_per_block_key)?;
    let numel_per_block_value = i32::try_from(numel_per_block_value)?;

    match dtype {
        candle_core::DType::BF16 => unsafe {
            copy_blocks_bf16(
                key_cache_ptr,
                value_cache_ptr,
                block_mapping_ptr,
                num_layers,
                num_pairs,
                numel_per_block_key,
                numel_per_block_value,
                stream.cu_stream() as i64,
            );
        },
        candle_core::DType::F16 => unsafe {
            copy_blocks_f16(
                key_cache_ptr,
                value_cache_ptr,
                block_mapping_ptr,
                num_layers,
                num_pairs,
                numel_per_block_key,
                numel_per_block_value,
                stream.cu_stream() as i64,
            );
        },
        candle_core::DType::F32 => unsafe {
            copy_blocks_f32(
                key_cache_ptr,
                value_cache_ptr,
                block_mapping_ptr,
                num_layers,
                num_pairs,
                numel_per_block_key,
                numel_per_block_value,
                stream.cu_stream() as i64,
            );
        },
        candle_core::DType::F8E4M3 => unsafe {
            copy_blocks_u8(
                key_cache_ptr,
                value_cache_ptr,
                block_mapping_ptr,
                num_layers,
                num_pairs,
                numel_per_block_key,
                numel_per_block_value,
                stream.cu_stream() as i64,
            );
        },
        _ => unreachable!(),
    }
    drop(cache_guards);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_blocks_uses_device_metadata_and_view_offsets() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let key_base = Tensor::from_vec(
            (0..24).map(|value| value as f32).collect::<Vec<_>>(),
            (6, 4),
            &device,
        )?
        .to_dtype(candle_core::DType::BF16)?;
        let value_base = Tensor::from_vec(
            (100..124).map(|value| value as f32).collect::<Vec<_>>(),
            (6, 4),
            &device,
        )?
        .to_dtype(candle_core::DType::BF16)?;
        let mut key = key_base.narrow(0, 1, 4)?;
        let mut value = value_base.narrow(0, 1, 4)?;
        let mapping = HashMap::from([(0, vec![2]), (1, vec![3])]);

        copy_blocks(vec![&mut key], vec![&mut value], &mapping)?;
        device.synchronize()?;

        assert_eq!(
            key_base
                .to_dtype(candle_core::DType::F32)?
                .to_vec2::<f32>()?,
            vec![
                vec![0.0, 1.0, 2.0, 3.0],
                vec![4.0, 5.0, 6.0, 7.0],
                vec![8.0, 9.0, 10.0, 11.0],
                vec![4.0, 5.0, 6.0, 7.0],
                vec![8.0, 9.0, 10.0, 11.0],
                vec![20.0, 21.0, 22.0, 23.0],
            ]
        );
        assert_eq!(
            value_base
                .to_dtype(candle_core::DType::F32)?
                .to_vec2::<f32>()?,
            vec![
                vec![100.0, 101.0, 102.0, 103.0],
                vec![104.0, 105.0, 106.0, 107.0],
                vec![108.0, 109.0, 110.0, 111.0],
                vec![104.0, 105.0, 106.0, 107.0],
                vec![108.0, 109.0, 110.0, 111.0],
                vec![120.0, 121.0, 122.0, 123.0],
            ]
        );
        Ok(())
    }

    #[test]
    fn copy_blocks_rejects_alternate_allocation_streams() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let Device::Cuda(dev) = &device else {
            unreachable!()
        };
        let stream = dev.cuda_stream();
        let alternate_stream = stream
            .context()
            .new_stream()
            .map_err(|error| candle_core::Error::Cuda(Box::new(error)))?;
        assert!(!Arc::ptr_eq(&stream, &alternate_stream));
        let key_slice = alternate_stream
            .clone_htod(&(0..16).map(|value| value as f32).collect::<Vec<_>>())
            .map_err(|error| candle_core::Error::Cuda(Box::new(error)))?;
        let key_storage = candle_core::CudaStorage::wrap_cuda_slice(key_slice, dev.clone());
        let mut key = Tensor::from((Storage::Cuda(key_storage), (4, 4)));
        let mut value = Tensor::zeros((4, 4), candle_core::DType::F32, &device)?;
        let mapping = HashMap::from([(0, vec![1])]);

        let error = copy_blocks(vec![&mut key], vec![&mut value], &mapping)
            .unwrap_err()
            .to_string();
        assert!(error.contains("allocated on a different CUDA stream"));
        Ok(())
    }
}

// `dst` REALLY should be &mut. That's the only reason this is unsafe.
/// # Safety
/// `dst` is the only shared reference and upholds the `&mut` aliasing guarantee.
pub unsafe fn swap_blocks(
    src: Tensor,
    dst: &Tensor,
    block_mapping: HashMap<usize, usize>,
) -> Result<()> {
    let block_size_in_bytes = src.dtype().size_in_bytes() * src.dims()[0];
    match (src.device(), dst.device()) {
        (Device::Cuda(src_dev), Device::Cuda(dst_dev)) => {
            if src_dev.location() != dst_dev.location() {
                candle_core::bail!("Tensors must be on the same device to copy, got locations {:?} (src) and {:?} (dst).", src_dev.location(), dst_dev.location());
            }
            let (src_storage, src_layout) = src.storage_and_layout();
            let (dst_storage, dst_layout) = dst.storage_and_layout();
            assert!(matches!(&*src_storage, Storage::Cuda(_)));
            assert!(matches!(&*dst_storage, Storage::Cuda(_)));
            let Storage::Cuda(src_storage) = &*src_storage else {
                unreachable!()
            };
            let Storage::Cuda(dst_storage) = &*dst_storage else {
                unreachable!()
            };
            let (src_ptr, dst_ptr) = match (&src_storage.slice, &dst_storage.slice) {
                (CudaStorageSlice::BF16(slice_src), CudaStorageSlice::BF16(slice_dst)) => {
                    let (ptr_src, _src_guard) = slice_ptr(slice_src, src_layout.start_offset());
                    let (ptr_dst, _dst_guard) = slice_ptr(slice_dst, dst_layout.start_offset());
                    (ptr_src, ptr_dst)
                }
                (CudaStorageSlice::F16(slice_src), CudaStorageSlice::F16(slice_dst)) => {
                    let (ptr_src, _src_guard) = slice_ptr(slice_src, src_layout.start_offset());
                    let (ptr_dst, _dst_guard) = slice_ptr(slice_dst, dst_layout.start_offset());
                    (ptr_src, ptr_dst)
                }
                (CudaStorageSlice::F32(slice_src), CudaStorageSlice::F32(slice_dst)) => {
                    let (ptr_src, _src_guard) = slice_ptr(slice_src, src_layout.start_offset());
                    let (ptr_dst, _dst_guard) = slice_ptr(slice_dst, dst_layout.start_offset());
                    (ptr_src, ptr_dst)
                }
                (CudaStorageSlice::F8E4M3(slice_src), CudaStorageSlice::F8E4M3(slice_dst)) => {
                    let (ptr_src, _src_guard) = slice_ptr(slice_src, src_layout.start_offset());
                    let (ptr_dst, _dst_guard) = slice_ptr(slice_dst, dst_layout.start_offset());
                    (ptr_src, ptr_dst)
                }
                _ => {
                    candle_core::bail!(
                        "only f32, f16, bf16 and f8e4m3 input data types are supported"
                    )
                }
            };

            for (src_block_number, dst_block_number) in block_mapping {
                let src_offset: u64 = (src_block_number * block_size_in_bytes).try_into().unwrap();
                let dst_offset: u64 = (dst_block_number * block_size_in_bytes).try_into().unwrap();
                // u8s because we copy by bytes
                let src_slice: CudaSlice<u8> = unsafe {
                    src_dev
                        .cuda_stream()
                        .upgrade_device_ptr(src_ptr + src_offset, block_size_in_bytes)
                };
                let mut dst_slice = unsafe {
                    dst_dev
                        .cuda_stream()
                        .upgrade_device_ptr(dst_ptr + dst_offset, block_size_in_bytes)
                };

                src_dev.memcpy_dtod(&src_slice, &mut dst_slice)?;
            }
        }
        (Device::Cpu, Device::Cuda(dst_dev)) => {
            let (src_storage, _src_layout) = src.storage_and_layout();
            let (dst_storage, dst_layout) = dst.storage_and_layout();
            assert!(matches!(&*src_storage, Storage::Cpu(_)));
            assert!(matches!(&*dst_storage, Storage::Cuda(_)));
            let Storage::Cpu(src_storage) = &*src_storage else {
                unreachable!()
            };
            let Storage::Cuda(dst_storage) = &*dst_storage else {
                unreachable!()
            };
            let (dst_ptr, _guard_dst) = slice_ptr(
                dst_storage.as_cuda_slice::<u8>()?,
                dst_layout.start_offset(),
            );
            let src_slice = src_storage.as_slice::<u8>()?;

            for (src_block_number, dst_block_number) in block_mapping {
                let src_offset = src_block_number * block_size_in_bytes;
                let dst_offset: u64 = (dst_block_number * block_size_in_bytes).try_into().unwrap();
                // u8s because we copy by bytes
                let mut dst_slice: CudaSlice<u8> = unsafe {
                    dst_dev
                        .cuda_stream()
                        .upgrade_device_ptr(dst_ptr + dst_offset, block_size_in_bytes)
                };

                dst_dev.memcpy_htod(
                    &src_slice[src_offset..src_offset + block_size_in_bytes],
                    &mut dst_slice,
                )?;
            }
        }
        (src, dst) => {
            candle_core::bail!("Tensors must be on either the GPU or CPU to swap, got {src:?} (src) and {dst:?} (dst).");
        }
    }

    Ok(())
}
