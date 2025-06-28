use std::{
    collections::HashMap,
    str::FromStr,
    sync::{Arc, Mutex, MutexGuard},
};

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_paged_attn::{copy_blocks, swap_blocks};
use serde::{Deserialize, Serialize};

use super::config::ModelConfigLike;

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
    pub num_cpu_blocks: usize,
    pub cache_type: PagedCacheType,
}

pub type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
    cpu_cache: Vec<KVCache>,
    num_layers: usize,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Self> {
        let dtype = cache_config.cache_type.to_dtype(dtype);
        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_gpu_cache(
                model_config,
                cache_config,
                dtype,
                device,
                layer_devices,
            )?)),
            cpu_cache: Self::allocate_cpu_cache(model_config, cache_config, dtype, device)?,
            num_layers: model_config.num_layers(),
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        loop {
            if let Ok(v) = self.gpu_cache.try_lock() {
                return v;
            }
        }
    }

    fn allocate_gpu_cache(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
        layer_devices: Vec<Option<Device>>,
    ) -> Result<Vec<KVCache>> {
        let key_block_shape =
            Self::calculate_key_block_shape(model_config, dtype, cache_config.block_size);
        let value_block_shape =
            Self::calculate_value_block_shape(model_config, cache_config.block_size);
        let mut gpu_cache = Vec::new();

        for device in layer_devices
            .iter()
            .take(model_config.num_layers())
            .map(|x| x.as_ref().unwrap_or(device))
        {
            #[allow(unused)]
            let key_blocks = if let Device::Metal(dev) = &device {
                #[cfg(feature = "metal")]
                {
                    use candle_core::{from_storage_no_op, MetalStorage, Shape, Storage};

                    let elem_count = cache_config.num_gpu_blocks
                        * key_block_shape.0
                        * key_block_shape.1
                        * key_block_shape.2
                        * key_block_shape.3;
                    let buffer = dev.new_buffer_private(elem_count, dtype, "k_cache")?;
                    let storage =
                        Storage::Metal(MetalStorage::new(buffer, dev.clone(), elem_count, dtype));
                    from_storage_no_op(
                        storage,
                        Shape::from_dims(&[
                            cache_config.num_gpu_blocks,
                            key_block_shape.0,
                            key_block_shape.1,
                            key_block_shape.2,
                            key_block_shape.3,
                        ]),
                        false,
                    )
                }

                #[cfg(not(feature = "metal"))]
                {
                    unreachable!()
                }
            } else {
                unsafe {
                    Tensor::empty(
                        (
                            cache_config.num_gpu_blocks,
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
                    use candle_core::{from_storage_no_op, MetalStorage, Shape, Storage};

                    let elem_count = cache_config.num_gpu_blocks
                        * value_block_shape.0
                        * value_block_shape.1
                        * value_block_shape.2;
                    let buffer = dev.new_buffer_private(elem_count, dtype, "v_cache")?;
                    let storage =
                        Storage::Metal(MetalStorage::new(buffer, dev.clone(), elem_count, dtype));
                    from_storage_no_op(
                        storage,
                        Shape::from_dims(&[
                            cache_config.num_gpu_blocks,
                            value_block_shape.0,
                            value_block_shape.1,
                            value_block_shape.2,
                        ]),
                        false,
                    )
                }

                #[cfg(not(feature = "metal"))]
                {
                    unreachable!()
                }
            } else {
                unsafe {
                    Tensor::empty(
                        (
                            cache_config.num_gpu_blocks,
                            value_block_shape.0,
                            value_block_shape.1,
                            value_block_shape.2,
                        ),
                        dtype,
                        device,
                    )?
                }
            };
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    fn allocate_cpu_cache(
        model_config: &dyn ModelConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Vec<KVCache>> {
        let key_block_shape =
            Self::calculate_key_block_shape(model_config, dtype, cache_config.block_size);
        let value_block_shape =
            Self::calculate_value_block_shape(model_config, cache_config.block_size);
        let mut cpu_cache = Vec::new();
        for _ in 0..model_config.num_layers() {
            let key_blocks = unsafe {
                Tensor::empty(
                    (
                        cache_config.num_cpu_blocks,
                        key_block_shape.0,
                        key_block_shape.1,
                        key_block_shape.2,
                        key_block_shape.3,
                    ),
                    dtype,
                    device,
                )?
            };
            let value_blocks = unsafe {
                Tensor::empty(
                    (
                        cache_config.num_cpu_blocks,
                        value_block_shape.0,
                        value_block_shape.1,
                        value_block_shape.2,
                    ),
                    dtype,
                    device,
                )?
            };
            cpu_cache.push((key_blocks, value_blocks));
        }
        Ok(cpu_cache)
    }
}

impl CacheEngine {
    fn calculate_key_block_shape(
        model_config: &dyn ModelConfigLike,
        dtype: DType,
        block_size: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            model_config.num_kv_heads(),
            model_config.k_head_dim() / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        model_config: &dyn ModelConfigLike,
        block_size: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.num_kv_heads(),
            model_config.v_head_dim(),
            block_size,
        )
    }
}

impl CacheEngine {
    pub fn execute_scheduler_ops(
        &self,
        blocks_to_swap_in: &HashMap<usize, usize>,
        blocks_to_swap_out: &HashMap<usize, usize>,
        blocks_to_copy: &HashMap<usize, Vec<usize>>,
    ) -> Result<()> {
        if !blocks_to_swap_in.is_empty() {
            self.swap_in(blocks_to_swap_in)?;
        }
        if !blocks_to_swap_out.is_empty() {
            self.swap_out(blocks_to_swap_out)?;
        }
        if !blocks_to_copy.is_empty() {
            self.copy(blocks_to_copy)?;
        }
        Ok(())
    }

    pub fn swap_in(&self, src_to_dst: &HashMap<usize, usize>) -> Result<()> {
        for i in 0..self.num_layers {
            let (src_key_cache, src_value_cache) = self.cpu_cache.get(i).unwrap();
            let gpu_cache = self.get_kv_cache();
            let (dst_key_cache, dst_value_cache) = gpu_cache.get(i).unwrap();
            // Swap (copy) key blocks
            unsafe { swap_blocks(src_key_cache.clone(), dst_key_cache, src_to_dst.clone())? };
            // Swap (copy) key blocks
            unsafe { swap_blocks(src_value_cache.clone(), dst_value_cache, src_to_dst.clone())? };
        }
        Ok(())
    }

    pub fn swap_out(&self, src_to_dst: &HashMap<usize, usize>) -> Result<()> {
        for i in 0..self.num_layers {
            let gpu_cache = self.get_kv_cache();
            let (src_key_cache, src_value_cache) = gpu_cache.get(i).unwrap().clone();
            drop(gpu_cache);

            let (dst_key_cache, dst_value_cache) = self.cpu_cache.get(i).unwrap();
            // Swap (copy) key blocks
            unsafe { swap_blocks(src_key_cache.clone(), dst_key_cache, src_to_dst.clone())? };
            // Swap (copy) key blocks
            unsafe { swap_blocks(src_value_cache.clone(), dst_value_cache, src_to_dst.clone())? };
        }
        Ok(())
    }

    pub fn copy(&self, src_to_dst: &HashMap<usize, Vec<usize>>) -> Result<()> {
        let mut gpu_cache = self.get_kv_cache();
        #[allow(clippy::map_identity)]
        let caches: (Vec<&mut Tensor>, Vec<&mut Tensor>) =
            gpu_cache.iter_mut().map(|(a, b)| (a, b)).unzip();
        let (key_caches, value_caches) = caches;

        // NOTE(EricLBuehler): This may synchronize the CPU and GPU
        copy_blocks(key_caches, value_caches, src_to_dst)?;

        Ok(())
    }
}
