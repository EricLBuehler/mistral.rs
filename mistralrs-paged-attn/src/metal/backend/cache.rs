use std::{collections::HashMap, iter::zip};

use candle_core::{
    backend::BackendStorage, CpuStorage, Device, IndexOp, Layout, MetalDevice, MetalStorage,
    Result, Storage, Tensor, WithDType,
};

use crate::metal::kernels;

pub fn copy_blocks(
    key_caches: Vec<&mut Tensor>,
    value_caches: Vec<&mut Tensor>,
    block_mapping: &HashMap<usize, Vec<usize>>,
) -> Result<()> {
    let cache_dev = key_caches.first().unwrap().device();
    let Device::Metal(dev) = cache_dev else {
        panic!("Expected the key caches to be on a Metal device.")
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

    let mut block_mapping_vec: Vec<i64> = Vec::new();
    for (src_block_number, dst_blocks) in block_mapping {
        for dst_block_number in dst_blocks {
            block_mapping_vec.push((*src_block_number).try_into().unwrap());
            block_mapping_vec.push((*dst_block_number).try_into().unwrap());
        }
    }
    let block_mapping = dev.new_buffer_with_data(&block_mapping_vec)?;

    let num_pairs: u64 = (block_mapping_vec.len() / 2).try_into().unwrap();

    let numel_per_block_key: u64 = key_caches
        .first()
        .unwrap()
        .i(0)?
        .shape()
        .dims()
        .iter()
        .product::<usize>()
        .try_into()
        .unwrap();
    let numel_per_block_value: u64 = value_caches
        .first()
        .unwrap()
        .i(0)?
        .shape()
        .dims()
        .iter()
        .product::<usize>()
        .try_into()
        .unwrap();
    assert_eq!(
        numel_per_block_key, numel_per_block_value,
        "key and value blocks must be the same size"
    );
    if numel_per_block_key != numel_per_block_key {
        candle_core::bail!(
            "numel_per_block_key ({numel_per_block_key}) and numel_per_block_value ({numel_per_block_value}) must be the same",
        );
    }

    for (key_cache, value_cache) in zip(&key_caches, &value_caches) {
        key_cache.to_device(cache_dev)?;
        value_cache.to_device(cache_dev)?;

        let key_offset = key_cache.storage_and_layout().1.start_offset();
        let Storage::Metal(key_storage) = &*key_cache.storage_and_layout().0 else {
            unreachable!()
        };

        let value_offset = value_cache.storage_and_layout().1.start_offset();
        let Storage::Metal(value_storage) = &*value_cache.storage_and_layout().0 else {
            unreachable!()
        };

        let encoder = dev.command_encoder()?;
        encoder.set_label("copy-blocks");

        kernels::call_copy_blocks(
            dev.device(),
            &encoder,
            &kernels::Kernels::new(),
            key_cache.dtype(),
            key_storage.buffer(),
            key_offset * key_storage.dtype().size_in_bytes(),
            value_storage.buffer(),
            value_offset * value_storage.dtype().size_in_bytes(),
            &block_mapping,
            0,
            num_pairs,
            numel_per_block_key,
            numel_per_block_value,
        )
        .map_err(candle_core::Error::wrap)?;
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
    if src.device().location() != dst.device().location() {
        candle_core::bail!(
            "Tensors must be on the same device to copy, got locations {:?} (src) and {:?} (dst).",
            src.device().location(),
            dst.device().location()
        );
    }
    match (src.device(), dst.device()) {
        (Device::Metal(src_dev), Device::Metal(_)) => {
            let (src_storage, src_layout) = src.storage_and_layout();
            let (dst_storage, dst_layout) = dst.storage_and_layout();
            assert!(matches!(&*src_storage, Storage::Metal(_)));
            assert!(matches!(&*dst_storage, Storage::Metal(_)));
            let Storage::Metal(src_storage) = &*src_storage else {
                unreachable!()
            };
            let Storage::Metal(dst_storage) = &*dst_storage else {
                unreachable!()
            };

            for (src_block_number, dst_block_number) in block_mapping {
                // We copy by bytes
                let src_offset = src_block_number * block_size_in_bytes
                    + src_layout.start_offset() * src_storage.dtype().size_in_bytes();
                let dst_offset = dst_block_number * block_size_in_bytes
                    + dst_layout.start_offset() * dst_storage.dtype().size_in_bytes();

                let blit = src_dev.blit_command_encoder()?;
                blit.set_label("swap-blocks-gpu-gpu");
                let length = src_layout.shape().elem_count() * src_storage.dtype().size_in_bytes();
                blit.copy_from_buffer(
                    src_storage.buffer(),
                    src_offset,
                    dst_storage.buffer(),
                    dst_offset,
                    length,
                );
                blit.end_encoding();
            }
        }
        (Device::Cpu, Device::Metal(dev)) => {
            let (src_storage, src_layout) = src.storage_and_layout();
            let (dst_storage, dst_layout) = dst.storage_and_layout();
            assert!(matches!(&*src_storage, Storage::Cpu(_)));
            assert!(matches!(&*dst_storage, Storage::Metal(_)));
            let Storage::Cpu(src_storage) = &*src_storage else {
                unreachable!()
            };
            let Storage::Metal(dst_storage) = &*dst_storage else {
                unreachable!()
            };

            fn swap_thunk<SRCT: WithDType>(
                src_slice: &[SRCT],
                src_layout: &Layout,
                dst_storage: &MetalStorage,
                dst_layout: &Layout,
                dev: &MetalDevice,
                block_size_in_bytes: usize,
                block_mapping: HashMap<usize, usize>,
            ) -> Result<()> {
                for (src_block_number, dst_block_number) in block_mapping {
                    let src_offset = src_block_number * block_size_in_bytes
                        + src_layout.start_offset() * SRCT::DTYPE.size_in_bytes();
                    let dst_offset = dst_block_number * block_size_in_bytes
                        + dst_layout.start_offset() * dst_storage.dtype().size_in_bytes();
                    // We copy by bytes
                    let src_buffer = dev.new_buffer_with_data(
                        &src_slice[src_offset..src_offset + block_size_in_bytes],
                    )?;

                    let blit = dev.blit_command_encoder()?;
                    blit.set_label("swap-blocks-cpu-gpu");
                    let length = src_layout.shape().elem_count() * SRCT::DTYPE.size_in_bytes();
                    blit.copy_from_buffer(
                        &src_buffer,
                        src_offset,
                        dst_storage.buffer(),
                        dst_offset,
                        length,
                    );
                    blit.end_encoding();
                }
                Ok(())
            }

            match src_storage {
                CpuStorage::BF16(s) => swap_thunk(
                    s,
                    src_layout,
                    dst_storage,
                    dst_layout,
                    dev,
                    block_size_in_bytes,
                    block_mapping,
                )?,
                CpuStorage::F16(s) => swap_thunk(
                    s,
                    src_layout,
                    dst_storage,
                    dst_layout,
                    dev,
                    block_size_in_bytes,
                    block_mapping,
                )?,
                CpuStorage::F32(s) => swap_thunk(
                    s,
                    src_layout,
                    dst_storage,
                    dst_layout,
                    dev,
                    block_size_in_bytes,
                    block_mapping,
                )?,
                _ => candle_core::bail!("expected bf16, f16, or f32 for cpu<>gpu swap-blocks"),
            }
        }
        (src, dst) => {
            candle_core::bail!("Tensors must be on either the GPU or CPU to swap, got {src:?} (src) and {dst:?} (dst).");
        }
    }

    Ok(())
}
