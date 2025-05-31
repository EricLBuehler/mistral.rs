use std::{collections::HashMap, iter::zip, ptr::NonNull};

use crate::cuda::backend::get_or_load_func;

use candle_core::cuda::cudarc::driver::LaunchAsync;
use candle_core::cuda::WrapErr;
use candle_core::cuda_backend::CudaStorageSlice;
use candle_core::Result;
use candle_core::{
    cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig},
    Device, IndexOp, Storage, Tensor,
};

use super::{Conjoined, COPY_BLOCKS_KERNEL_NAME};
use crate::COPY_BLOCKS_KERNEL;

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
                let ptr_key = *slice_key.slice(0..).device_ptr();
                let ptr_value = *slice_value.slice(0..).device_ptr();
                (ptr_key, ptr_value)
            }
            (CudaStorageSlice::F16(slice_key), CudaStorageSlice::F16(slice_value)) => {
                let ptr_key = *slice_key.slice(0..).device_ptr();
                let ptr_value = *slice_value.slice(0..).device_ptr();
                (ptr_key, ptr_value)
            }
            (CudaStorageSlice::F32(slice_key), CudaStorageSlice::F32(slice_value)) => {
                let ptr_key = *slice_key.slice(0..).device_ptr();
                let ptr_value = *slice_value.slice(0..).device_ptr();
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
    let block_mapping_ptr = Conjoined::new(
        NonNull::new(block_mapping_vec.as_mut_ptr()).unwrap(),
        &mut block_mapping_vec,
    );

    let key_cache_ptr = Conjoined::new(
        NonNull::new(key_cache_ptrs.as_mut_ptr()).unwrap(),
        &mut key_cache_ptrs,
    );
    let value_cache_ptr = Conjoined::new(
        NonNull::new(value_cache_ptrs.as_mut_ptr()).unwrap(),
        &mut value_cache_ptrs,
    );

    let numel_per_block: u32 = key_caches
        .first()
        .unwrap()
        .i(0)?
        .shape()
        .dims()
        .iter()
        .product::<usize>()
        .try_into()
        .unwrap();
    let launch_conf = LaunchConfig {
        grid_dim: (num_layers, num_pairs, 1u32),
        block_dim: (numel_per_block.min(1024), 1u32, 1u32),
        shared_mem_bytes: 0,
    };
    let stream = dev.fork_default_stream().w()?;

    let kernel = get_or_load_func(
        COPY_BLOCKS_KERNEL,
        COPY_BLOCKS_KERNEL_NAME,
        key_caches.first().unwrap().dtype(),
        None,
        dev,
    )?;

    unsafe {
        kernel
            .launch_on_stream(
                &stream,
                launch_conf,
                (
                    key_cache_ptr,
                    value_cache_ptr,
                    block_mapping_ptr,
                    numel_per_block as i32,
                ),
            )
            .w()?;
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
            if src_dev.ordinal() != dst_dev.ordinal() {
                candle_core::bail!("Tensors must be on the same device to copy, got ordinals {} (src) and {} (dst).", src_dev.ordinal(), dst_dev.ordinal());
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
                    let ptr_src = *slice_src.slice(src_layout.start_offset()..).device_ptr();
                    let ptr_dst = *slice_dst.slice(dst_layout.start_offset()..).device_ptr();
                    (ptr_src, ptr_dst)
                }
                (CudaStorageSlice::F16(slice_src), CudaStorageSlice::F16(slice_dst)) => {
                    let ptr_src = *slice_src.slice(src_layout.start_offset()..).device_ptr();
                    let ptr_dst = *slice_dst.slice(dst_layout.start_offset()..).device_ptr();
                    (ptr_src, ptr_dst)
                }
                (CudaStorageSlice::F32(slice_src), CudaStorageSlice::F32(slice_dst)) => {
                    let ptr_src = *slice_src.slice(src_layout.start_offset()..).device_ptr();
                    let ptr_dst = *slice_dst.slice(dst_layout.start_offset()..).device_ptr();
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
                    src_dev.upgrade_device_ptr(src_ptr + src_offset, block_size_in_bytes)
                };
                let mut dst_slice = unsafe {
                    dst_dev.upgrade_device_ptr(dst_ptr + dst_offset, block_size_in_bytes)
                };

                src_dev.dtod_copy(&src_slice, &mut dst_slice).w()?;
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
            let dst_ptr = dst_storage.as_cuda_slice::<u8>()?.device_ptr()
                + TryInto::<u64>::try_into(dst_layout.start_offset()).unwrap();
            let src_slice = src_storage.as_slice()?;

            for (src_block_number, dst_block_number) in block_mapping {
                let src_offset = src_block_number * block_size_in_bytes;
                let dst_offset: u64 = (dst_block_number * block_size_in_bytes).try_into().unwrap();
                // u8s because we copy by bytes
                let mut dst_slice: CudaSlice<u8> = unsafe {
                    dst_dev.upgrade_device_ptr(dst_ptr + dst_offset, block_size_in_bytes)
                };

                dst_dev
                    .htod_sync_copy_into(
                        &src_slice[src_offset..src_offset + block_size_in_bytes],
                        &mut dst_slice,
                    )
                    .w()?;
            }
        }
        (src, dst) => {
            candle_core::bail!("Tensors must be on either the GPU or CPU to swap, got {src:?} (src) and {dst:?} (dst).");
        }
    }

    Ok(())
}
