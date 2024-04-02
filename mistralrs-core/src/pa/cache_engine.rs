use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use candle_core::{cuda_backend::cudarc::driver::CudaStream, DType, Device, Result, Tensor};

use crate::pipeline::ConfigLike;

use super::kernels::{
    cache::{apply_copy_blocks, swap_blocks},
    cuda_stream_synchronize,
};

#[derive(Clone, Copy)]
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
    cache_stream: CudaStream,
}

impl CacheEngine {
    pub fn new(
        model_config: &dyn ConfigLike,
        cache_config: CacheConfig,
        dtype: DType,
        cuda_device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            gpu_cache: Arc::new(Mutex::new(Self::allocate_gpu_cache(
                model_config,
                &cache_config,
                dtype,
                &cuda_device,
            )?)),
            cpu_cache: Self::allocate_cpu_cache(model_config, &cache_config, dtype, &cuda_device)?,
            num_layers: model_config.get_num_hidden_layers(),
            cache_stream: match cuda_device {
                Device::Cuda(d) => d.fork_default_stream().unwrap(),
                _ => candle_core::bail!("Unexpected device"),
            },
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
        cuda_device: &Device,
    ) -> Result<Vec<KVCache>> {
        assert!(cache_config.fully_init);

        let key_block_shape =
            Self::calculate_key_block_shape(model_config, dtype, cache_config.block_size);
        let value_block_shape =
            Self::calculate_value_block_shape(model_config, cache_config.block_size);
        let mut gpu_cache = Vec::new();
        for _ in 0..model_config.get_num_hidden_layers() {
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
        cuda_device: &Device,
    ) -> Result<Vec<KVCache>> {
        assert!(cache_config.fully_init);

        let key_block_shape =
            Self::calculate_key_block_shape(model_config, dtype, cache_config.block_size);
        let value_block_shape =
            Self::calculate_value_block_shape(model_config, cache_config.block_size);
        let mut cpu_cache = Vec::new();
        for _ in 0..model_config.get_num_hidden_layers() {
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
    fn swap(
        &self,
        src: &Vec<KVCache>,
        dst: &Vec<KVCache>,
        src_to_dst: &HashMap<usize, usize>,
    ) -> Result<()> {
        for i in 0..self.num_layers {
            let (src_key_cache, src_value_cache) = &src[i];
            let (dst_key_cache, dst_value_cache) = &dst[i];
            swap_blocks(
                src_key_cache,
                dst_key_cache,
                src_to_dst,
                self.cache_stream.stream,
            )?;
            swap_blocks(
                src_value_cache,
                dst_value_cache,
                src_to_dst,
                self.cache_stream.stream,
            )?;
        }
        cuda_stream_synchronize(self.cache_stream.stream)?;
        Ok(())
    }

    pub fn swap_in(&self, src_to_dst: HashMap<usize, usize>) -> Result<()> {
        println!("swap in");
        dbg!(&src_to_dst);
        self.swap(&self.cpu_cache, &self.get_kv_cache(), &src_to_dst)
    }

    pub fn swap_out(&mut self, src_to_dst: HashMap<usize, usize>) -> Result<()> {
        println!("swap out");
        dbg!(&src_to_dst);
        self.swap(&self.get_kv_cache(), &self.cpu_cache, &src_to_dst)
    }

    pub fn copy(&mut self, block_mapping: HashMap<usize, Vec<usize>>) -> Result<()> {
        println!("copy");
        dbg!(&block_mapping);
        let gpu_cache = self.get_kv_cache();

        let key_caches = gpu_cache
            .iter()
            .map(|(key_cache, _)| key_cache)
            .collect::<Vec<&Tensor>>();
        let value_caches = gpu_cache
            .iter()
            .map(|(_, value_cache)| value_cache)
            .collect::<Vec<&Tensor>>();

        apply_copy_blocks(key_caches, value_caches, &block_mapping)?;
        Ok(())
    }
}
