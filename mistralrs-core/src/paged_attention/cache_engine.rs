use std::{
    str::FromStr,
    sync::{Arc, Mutex, MutexGuard},
};

use candle_core::{DType, Device, Result, Tensor};
use serde::{Deserialize, Serialize};

use super::config::{KvCacheLayout, ModelConfigLike};

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn cuda_supports_fp8(device: &Device) -> bool {
    use candle_core::cuda::cudarc::driver::{result, sys};

    if !mistralrs_paged_attn::USE_FP8 {
        return false;
    }
    let Device::Cuda(cuda) = device else {
        return false;
    };
    let ordinal = cuda.cuda_stream().context().ordinal();
    #[allow(clippy::cast_possible_truncation)]
    let Ok(device) = result::device::get(ordinal as i32) else {
        return false;
    };
    unsafe {
        result::device::get_attribute(
            device,
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .is_ok_and(|major| major >= 8)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Default)]
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
pub enum PagedCacheType {
    #[default]
    Auto,
    F8E4M3,
}

impl PagedCacheType {
    pub fn to_dtype(&self, act_dtype: DType) -> DType {
        match self {
            PagedCacheType::F8E4M3 => DType::F8E4M3,
            PagedCacheType::Auto => act_dtype,
        }
    }

    pub fn validate(
        &self,
        act_dtype: DType,
        model_config: &dyn ModelConfigLike,
        device: &Device,
        layer_devices: &[Option<Device>],
    ) -> std::result::Result<(), String> {
        if *self == Self::Auto {
            return Ok(());
        }
        if !matches!(act_dtype, DType::F16 | DType::BF16 | DType::F32) {
            return Err(format!(
                "FP8 KV cache requires f16, bf16, or f32 activations, got {act_dtype:?}"
            ));
        }

        for layer_idx in 0..model_config.num_layers() {
            if !model_config.layer_has_paged_kv_cache(layer_idx) {
                continue;
            }
            if matches!(
                model_config.kv_cache_layout_for_layer(layer_idx),
                KvCacheLayout::Mla { .. }
            ) {
                return Err(format!(
                    "FP8 KV cache is not supported for MLA layer {layer_idx}"
                ));
            }
            let layer_device = layer_devices
                .get(layer_idx)
                .and_then(Option::as_ref)
                .unwrap_or(device);
            if layer_device.is_cuda() {
                #[cfg(all(feature = "cuda", target_family = "unix"))]
                if !cuda_supports_fp8(layer_device) {
                    return Err(
                        "FP8 KV cache requires CUDA compute capability 8.0 or newer and a matching CUDA build"
                            .to_string(),
                    );
                }
                #[cfg(not(all(feature = "cuda", target_family = "unix")))]
                return Err("FP8 KV cache requires the CUDA paged-attention backend".to_string());
            } else if layer_device.is_metal() {
                #[cfg(not(feature = "metal"))]
                return Err("FP8 KV cache requires the Metal paged-attention backend".to_string());
            } else {
                return Err(format!(
                    "FP8 KV cache is only supported on CUDA or Metal, got {layer_device:?} for layer {layer_idx}"
                ));
            }
        }
        Ok(())
    }
}

impl FromStr for PagedCacheType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "f8e4m3" => Ok(Self::F8E4M3),
            other => Err(format!(
                "Unexpected `PagedCacheType`, got `{other}` but expected `auto` and `f8e4m3`."
            )),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub cache_type: PagedCacheType,
    pub kv_cache_group_ids: Vec<u32>,
}

pub type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Self> {
        cache_config
            .cache_type
            .validate(dtype, model_config, device, &layer_devices)
            .map_err(candle_core::Error::msg)?;
        let dtype = cache_config.cache_type.to_dtype(dtype);
        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_gpu_cache(
                model_config,
                cache_config,
                dtype,
                device,
                layer_devices,
            )?)),
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        // Use blocking lock instead of busy-wait spin loop to avoid CPU waste
        // and potential thread starvation issues
        self.gpu_cache.lock().expect("KV cache mutex was poisoned")
    }

    fn allocate_gpu_cache(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Vec<KVCache>> {
        let mut gpu_cache = Vec::new();

        for (layer_idx, device) in layer_devices
            .iter()
            .take(model_config.num_layers())
            .map(|x| x.as_ref().unwrap_or(device))
            .enumerate()
        {
            // Hybrid models keep no paged cache on linear/recurrent layers, but the vec stays indexed
            // by absolute layer index, so those get an empty tensor of the right rank instead.
            let num_gpu_blocks = if model_config.layer_has_paged_kv_cache(layer_idx) {
                cache_config.num_gpu_blocks
            } else {
                0
            };
            let requested_kv_cache_layout = model_config.kv_cache_layout_for_layer(layer_idx);
            let kv_cache_layout =
                if matches!(requested_kv_cache_layout, KvCacheLayout::FlashInferHnd)
                    && !device.is_cuda()
                {
                    KvCacheLayout::Standard
                } else {
                    requested_kv_cache_layout
                };
            let (key_blocks, value_blocks) = match kv_cache_layout {
                KvCacheLayout::Standard | KvCacheLayout::StandardNoFlashInfer => {
                    let key_block_shape = Self::calculate_key_block_shape(
                        model_config,
                        dtype,
                        cache_config.block_size,
                        layer_idx,
                    );
                    let value_block_shape = Self::calculate_value_block_shape(
                        model_config,
                        cache_config.block_size,
                        layer_idx,
                    );
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2
                                * key_block_shape.3;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                    key_block_shape.3,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = num_gpu_blocks
                                * value_block_shape.0
                                * value_block_shape.1
                                * value_block_shape.2;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    num_gpu_blocks,
                                    value_block_shape.0,
                                    value_block_shape.1,
                                    value_block_shape.2,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
                KvCacheLayout::FlashInferHnd => {
                    let key_block_shape = Self::calculate_flashinfer_block_shape(
                        model_config,
                        cache_config.block_size,
                        layer_idx,
                    );
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count = num_gpu_blocks
                                * key_block_shape.0
                                * key_block_shape.1
                                * key_block_shape.2;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (
                                    num_gpu_blocks,
                                    key_block_shape.0,
                                    key_block_shape.1,
                                    key_block_shape.2,
                                ),
                                dtype,
                                device,
                            )?
                        }
                    };
                    let value_blocks = unsafe {
                        Tensor::empty(
                            (
                                num_gpu_blocks,
                                key_block_shape.0,
                                key_block_shape.1,
                                key_block_shape.2,
                            ),
                            dtype,
                            device,
                        )?
                    };
                    (key_blocks, value_blocks)
                }
                KvCacheLayout::Mla {
                    kv_lora_rank,
                    kpe_head_dim,
                } => {
                    #[allow(unused)]
                    let key_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count =
                                num_gpu_blocks * cache_config.block_size * kv_lora_rank;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "k_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    cache_config.block_size,
                                    kv_lora_rank,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (num_gpu_blocks, cache_config.block_size, kv_lora_rank),
                                dtype,
                                device,
                            )?
                        }
                    };
                    #[allow(unused)]
                    let value_blocks = if let Device::Metal(dev) = &device {
                        #[cfg(feature = "metal")]
                        {
                            use candle_core::{MetalStorage, Shape, Storage};

                            let elem_count =
                                num_gpu_blocks * cache_config.block_size * kpe_head_dim;
                            let buffer = dev.new_private_buffer(elem_count, dtype, "v_cache")?;
                            let storage = Storage::Metal(MetalStorage::new(
                                buffer,
                                dev.clone(),
                                elem_count,
                                dtype,
                            ));
                            Tensor::from((
                                storage,
                                Shape::from_dims(&[
                                    num_gpu_blocks,
                                    cache_config.block_size,
                                    kpe_head_dim,
                                ]),
                            ))
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            unreachable!()
                        }
                    } else {
                        unsafe {
                            Tensor::empty(
                                (num_gpu_blocks, cache_config.block_size, kpe_head_dim),
                                dtype,
                                device,
                            )?
                        }
                    };
                    (key_blocks, value_blocks)
                }
            };
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    fn calculate_key_block_shape(
        model_config: &dyn ModelConfigLike,
        dtype: DType,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            model_config.k_head_dim_for_layer(layer_idx) / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            model_config.v_head_dim_for_layer(layer_idx),
            block_size,
        )
    }

    fn calculate_flashinfer_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
        layer_idx: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_kv_heads_for_layer(layer_idx),
            block_size,
            model_config.k_head_dim_for_layer(layer_idx),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::config::ModelConfigMetadata;

    fn model_config(layout: KvCacheLayout) -> ModelConfigMetadata {
        ModelConfigMetadata {
            max_seq_len: 4096,
            num_layers: 2,
            hidden_size: 1024,
            num_kv_heads: 4,
            num_attn_heads: 16,
            sliding_window: None,
            k_head_dim: 128,
            v_head_dim: 128,
            kv_cache_layout: layout,
        }
    }

    #[test]
    fn fp8_cache_rejects_cpu_before_allocation() {
        let err = PagedCacheType::F8E4M3
            .validate(
                DType::BF16,
                &model_config(KvCacheLayout::Standard),
                &Device::Cpu,
                &[],
            )
            .unwrap_err();
        assert!(err.contains("only supported on CUDA or Metal"));
    }

    #[test]
    fn fp8_cache_rejects_mla_before_allocation() {
        let err = PagedCacheType::F8E4M3
            .validate(
                DType::BF16,
                &model_config(KvCacheLayout::Mla {
                    kv_lora_rank: 512,
                    kpe_head_dim: 64,
                }),
                &Device::Cpu,
                &[],
            )
            .unwrap_err();
        assert!(err.contains("not supported for MLA layer 0"));
    }
}
