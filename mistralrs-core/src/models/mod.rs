use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::Tensor;

use crate::get_mut_arcmutex;

pub(crate) mod gemma;
pub(crate) mod mistral;
pub(crate) mod quantized_llama;

pub type LayerCaches = Vec<Option<(Tensor, Tensor)>>;

#[derive(Debug, Clone)]
pub struct Cache {
    cache: Arc<Mutex<LayerCaches>>,
    xlora_cache: Option<Arc<Mutex<LayerCaches>>>,
}

impl Cache {
    pub(crate) fn new(len: usize, is_xlora: bool) -> Self {
        Self {
            cache: Arc::new(Mutex::new(vec![None; len])),
            xlora_cache: if is_xlora {
                Some(Arc::new(Mutex::new(vec![None; len])))
            } else {
                None
            },
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.cache)
    }

    /// # Panics
    /// If there is no xlora cache
    pub(crate) fn xlora_lock(&self) -> MutexGuard<'_, LayerCaches> {
        get_mut_arcmutex!(self.xlora_cache.as_ref().unwrap())
    }

    pub(crate) fn is_xlora(&self) -> bool {
        self.xlora_cache.is_some()
    }
}
