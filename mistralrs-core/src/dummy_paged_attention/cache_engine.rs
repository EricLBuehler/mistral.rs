use std::{
    collections::HashMap,
    sync::{Arc, Mutex, MutexGuard},
};

use candle_core::{DType, Device, Result, Tensor};

use super::config::ModelConfigLike;

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub block_size: usize,
    pub num_gpu_blocks: usize,
    pub num_cpu_blocks: usize,
}

pub type KVCache = (Tensor, Tensor);

pub struct CacheEngine {
    dummy_cache: Arc<Mutex<Vec<KVCache>>>,
}

impl CacheEngine {
    pub fn new(
        _model_config: &dyn ModelConfigLike,
        _cache_config: &CacheConfig,
        _dtype: DType,
        _device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            dummy_cache: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn get_kv_cache(&self) -> MutexGuard<'_, Vec<KVCache>> {
        loop {
            if let Ok(v) = self.dummy_cache.try_lock() {
                return v;
            }
        }
    }
}

impl CacheEngine {
    pub fn execute_scheduler_ops(
        &self,
        blocks_to_swap_in: HashMap<usize, usize>,
        blocks_to_swap_out: HashMap<usize, usize>,
        blocks_to_copy: HashMap<usize, Vec<usize>>,
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

    pub fn swap_in(&self, _src_to_dst: HashMap<usize, usize>) -> Result<()> {
        Ok(())
    }

    pub fn swap_out(&self, _src_to_dst: HashMap<usize, usize>) -> Result<()> {
        Ok(())
    }

    pub fn copy(&self, _src_to_dst: HashMap<usize, Vec<usize>>) -> Result<()> {
        Ok(())
    }
}
