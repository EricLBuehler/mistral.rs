use std::{collections::HashMap, iter::zip};

use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi::{copy_blocks_bf16, copy_blocks_f16, copy_blocks_f32};
use candle_core::backend::BackendDevice;
use candle_core::cuda_backend::CudaStorageSlice;
use candle_core::Result;
use candle_core::{
    cuda_backend::cudarc::driver::CudaSlice, DType, Device, IndexOp, Storage, Tensor,
};

pub fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: &HashMap<usize, Vec<usize>>,
) -> Result<()> {
    let cache_dev = key_caches.first().unwrap().device();
    let Device::Cuda(dev) = cache_dev else {
        panic!("Expected the key caches to be on a CUDA device.")
    };
    if !cache_dev.same_device(value_caches.first().unwrap().device()) {
        candle_core::bail!(
            "`key` and `value` caches have different devices, got {:?} and {:?} respectively.",
            cache_dev,
            value_caches.first().unwrap().device()
        );
    }
    if key_caches.first().unwrap().dtype() != value_caches.first().unwrap().dtype() {
        candle_core::bail!(
            "Key and value caches have different types, got {:?} and {:?}.",
            key_caches.first().unwrap().dtype(),
            value_caches.first().unwrap().dtype()
        );
    }
    let num_layers: u32 = key_caches.len().try_into().unwrap();
    if num_layers == 0 {
        return Ok(());
    }

    let mut key_cache_ptrs = Vec::new();
    key_cache_ptrs.reserve_exact(num_layers as usize);
    let mut value_cache_ptrs = Vec::new();
    value_cache_ptrs.reserve_exact(num_layers as usize);
    let mut dtype = DType::F32;

    for (key_cache, value_cache) in zip(&key_caches, &value_caches) {
        key_cache.to_device(cache_dev)?;
        value_cache.to_device(cache_dev)?;

        let key_offset: u64 = key_cache
            .storage_and_layout()
            .1
            .start_offset()
            .try_into()
            .unwrap();
        let Storage::Cuda(key_storage) = &*key_cache.storage_and_layout().0 else {
            unreachable!()
        };

        let value_offset: u64 = value_cache
            .storage_and_layout()
            .1
            .start_offset()
            .try_into()
            .unwrap();
        let Storage::Cuda(value_storage) = &*value_cache.storage_and_layout().0 else {
            unreachable!()
        };

        let (key_ptr, value_ptr) = match (&key_storage.slice, &value_storage.slice) {
            (CudaStorageSlice::BF16(slice_key), CudaStorageSlice::BF16(slice_value)) => {
                let (ptr_key, _key_guard) = slice_ptr(slice_key, 0);
                let (ptr_value, _value_guard) = slice_ptr(slice_value, 0);
                dtype = DType::BF16;
                (ptr_key, ptr_value)
            }
            (CudaStorageSlice::F16(slice_key), CudaStorageSlice::F16(slice_value)) => {
                let (ptr_key, _key_guard) = slice_ptr(slice_key, 0);
                let (ptr_value, _value_guard) = slice_ptr(slice_value, 0);
                dtype = DType::F16;
                (ptr_key, ptr_value)
            }
            (CudaStorageSlice::F32(slice_key), CudaStorageSlice::F32(slice_value)) => {
                let (ptr_key, _key_guard) = slice_ptr(slice_key, 0);
                let (ptr_value, _value_guard) = slice_ptr(slice_value, 0);
                (ptr_key, ptr_value)
            }
            _ => {
                candle_core::bail!("only f32, f16 and bf16 input data type supported!",);
            }
        };
        key_cache_ptrs.push(key_ptr + key_offset);
        value_cache_ptrs.push(value_ptr + value_offset);
    }

    let mut block_mapping_vec: Vec<i64> = Vec::new();
    for (src_block_number, dst_blocks) in block_mapping {
        for dst_block_number in dst_blocks {
            block_mapping_vec.push((*src_block_number).try_into().unwrap());
            block_mapping_vec.push((*dst_block_number).try_into().unwrap());
        }
    }
    let num_pairs: u32 = (block_mapping_vec.len() / 2).try_into().unwrap();

    let key_cache_ptr = key_cache_ptrs.as_mut_ptr() as *mut core::ffi::c_void;
    let value_cache_ptr = value_cache_ptrs.as_mut_ptr() as *mut core::ffi::c_void;
    let block_mapping_ptr = block_mapping_vec.as_mut_ptr() as *const core::ffi::c_void;

    let numel_per_block_key: u32 = key_caches
        .first()
        .unwrap()
        .i(0)?
        .shape()
        .dims()
        .iter()
        .product::<usize>()
        .try_into()
        .unwrap();
    let numel_per_block_value: u32 = value_caches
        .first()
        .unwrap()
        .i(0)?
        .shape()
        .dims()
        .iter()
        .product::<usize>()
        .try_into()
        .unwrap();

    match dtype {
        candle_core::DType::BF16 => unsafe {
            copy_blocks_bf16(
                key_cache_ptr,
                value_cache_ptr,
                block_mapping_ptr,
                num_layers as i32,
                num_pairs as i32,
                numel_per_block_key as i32,
                numel_per_block_value as i32,
                dev.cuda_stream().cu_stream() as i64,
            );
        },
        candle_core::DType::F16 => unsafe {
            copy_blocks_f16(
                key_cache_ptr,
                value_cache_ptr,
                block_mapping_ptr,
                num_layers as i32,
                num_pairs as i32,
                numel_per_block_key as i32,
                numel_per_block_value as i32,
                dev.cuda_stream().cu_stream() as i64,
            );
        },
        candle_core::DType::F32 => unsafe {
            copy_blocks_f32(
                key_cache_ptr,
                value_cache_ptr,
                block_mapping_ptr,
                num_layers as i32,
                num_pairs as i32,
                numel_per_block_key as i32,
                numel_per_block_value as i32,
                dev.cuda_stream().cu_stream() as i64,
            );
        },
        _ => {}
    }

    Ok(())
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
                _ => {
                    candle_core::bail!("only f32, f16 and bf16 input data type supported!")
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
