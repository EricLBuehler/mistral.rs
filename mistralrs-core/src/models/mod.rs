use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::Tensor;

use crate::get_mut_arcmutex;

pub(crate) mod mistral;

#[derive(Debug, Clone)]
pub struct Cache {
    cache: Arc<Mutex<Vec<Option<(Tensor, Tensor)>>>>,
}

impl Cache {
    pub(crate) fn new(len: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(vec![None; len])),
        }
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, Vec<Option<(Tensor, Tensor)>>> {
        get_mut_arcmutex!(self.cache)
    }
}
