use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::backend::{copy_blocks, swap_blocks};
use super::ConfigLike;

#[derive(Clone)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: Option<usize>, // Set after profiling init
    pub num_cpu_blocks: Option<usize>, // Set after profiling init
    pub fully_init: bool,
}

impl CacheConfig {
    pub fn set_num_gpu_blocks(&mut self, num_gpu_blocks: usize) {
        if self.num_cpu_blocks.is_some() {
            self.fully_init = true;
        }
        self.num_gpu_blocks = Some(num_gpu_blocks);
    }
    pub fn set_num_cpu_blocks(&mut self, num_cpu_blocks: usize) {
        if self.num_gpu_blocks.is_some() {
            self.fully_init = true;
        }
        self.num_cpu_blocks = Some(num_cpu_blocks);
    }
}

pub type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    gpu_cache: Arc<Mutex<Vec<KVCache>>>,
    cpu_cache: Vec<KVCache>,
    num_layers: usize,
}

impl CacheEngine {
    pub fn new(
        model_config: Box<dyn ConfigLike>,
        cache_config: CacheConfig,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_gpu_cache(
                &*model_config,
                &cache_config,
                dtype,
            )?)),
            cpu_cache: Self::allocate_cpu_cache(&*model_config, &cache_config, dtype)?,
            num_layers: model_config.get_num_hidden_layers(),
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
        model_config: &dyn ConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
    ) -> Result<Vec<KVCache>> {
        assert!(cache_config.fully_init);

        let key_block_shape =
            Self::calculate_key_block_shape(model_config, dtype, cache_config.block_size);
        let value_block_shape =
            Self::calculate_value_block_shape(model_config, cache_config.block_size);
        let mut gpu_cache = Vec::new();
        for _ in 0..model_config.get_num_hidden_layers() {
            let cuda_device = Device::new_cuda(0)?;
            let key_blocks = Tensor::zeros(
                (
                    cache_config.num_gpu_blocks.unwrap(),
                    key_block_shape.0,
                    key_block_shape.1,
                    key_block_shape.2,
                    key_block_shape.3,
                ),
                dtype,
                &cuda_device,
            )?;
            let value_blocks = Tensor::zeros(
                (
                    cache_config.num_gpu_blocks.unwrap(),
                    value_block_shape.0,
                    value_block_shape.1,
                    value_block_shape.2,
                ),
                dtype,
                &cuda_device,
            )?;
            gpu_cache.push((key_blocks, value_blocks));
        }
        Ok(gpu_cache)
    }

    fn allocate_cpu_cache(
        model_config: &dyn ConfigLike,
        cache_config: &CacheConfig,
        dtype: DType,
    ) -> Result<Vec<KVCache>> {
        assert!(cache_config.fully_init);

        let key_block_shape =
            Self::calculate_key_block_shape(model_config, dtype, cache_config.block_size);
        let value_block_shape =
            Self::calculate_value_block_shape(model_config, cache_config.block_size);
        let mut cpu_cache = Vec::new();
        for _ in 0..model_config.get_num_hidden_layers() {
            let cuda_device = Device::new_cuda(0)?;
            let key_blocks = Tensor::zeros(
                (
                    cache_config.num_cpu_blocks.unwrap(),
                    key_block_shape.0,
                    key_block_shape.1,
                    key_block_shape.2,
                    key_block_shape.3,
                ),
                dtype,
                &cuda_device,
            )?;
            let value_blocks = Tensor::zeros(
                (
                    cache_config.num_cpu_blocks.unwrap(),
                    value_block_shape.0,
                    value_block_shape.1,
                    value_block_shape.2,
                ),
                dtype,
                &cuda_device,
            )?;
            cpu_cache.push((key_blocks, value_blocks));
        }
        Ok(cpu_cache)
    }
}

impl CacheEngine {
    fn calculate_key_block_shape(
        model_config: &dyn ConfigLike,
        dtype: DType,
        block_size: usize,
    ) -> (usize, usize, usize, usize) {
        let element_size = dtype.size_in_bytes();
        let x = 16 / element_size;
        (
            model_config.get_num_kv_heads(),
            model_config.get_head_size() / x,
            block_size,
            x,
        )
    }

    fn calculate_value_block_shape(
        model_config: &dyn ConfigLike,
        block_size: usize,
    ) -> (usize, usize, usize) {
        (
            model_config.get_num_kv_heads(),
            model_config.get_head_size(),
            block_size,
        )
    }
}

impl CacheEngine {
    pub fn swap_in(&self, src_to_dst: HashMap<usize, usize>) {
        for i in 0..self.num_layers {
            let (src_key_cache, src_value_cache) = self.cpu_cache.get(i).unwrap();
            let mut gpu_cache = self.get_kv_cache();
            let (dst_key_cache, dst_value_cache) = gpu_cache.get_mut(i).unwrap();
            // Swap (copy) key blocks
            swap_blocks(src_key_cache.clone(), dst_key_cache, src_to_dst.clone());
            // Swap (copy) key blocks
            swap_blocks(src_value_cache.clone(), dst_value_cache, src_to_dst.clone());
        }
    }

    pub fn swap_out(&mut self, src_to_dst: HashMap<usize, usize>) {
        for i in 0..self.num_layers {
            let gpu_cache = self.get_kv_cache();
            let (src_key_cache, src_value_cache) = gpu_cache.get(i).unwrap().clone();
            drop(gpu_cache);

            let (dst_key_cache, dst_value_cache) = self.cpu_cache.get_mut(i).unwrap();
            // Swap (copy) key blocks
            swap_blocks(src_key_cache.clone(), dst_key_cache, src_to_dst.clone());
            // Swap (copy) key blocks
            swap_blocks(src_value_cache.clone(), dst_value_cache, src_to_dst.clone());
        }
    }

    pub fn copy(&mut self, src_to_dst: HashMap<usize, Vec<usize>>) {
        let mut gpu_cache = self.get_kv_cache();
        #[allow(clippy::map_identity)]
        let caches: (Vec<&mut Tensor>, Vec<&mut Tensor>) =
            gpu_cache.iter_mut().map(|(a, b)| (a, b)).unzip();
        let (key_caches, value_caches) = caches;

        // NOTE(EricLBuehler): This may synchronize the CPU and GPU
        unsafe { copy_blocks(key_caches, value_caches, src_to_dst) };
    }
}
