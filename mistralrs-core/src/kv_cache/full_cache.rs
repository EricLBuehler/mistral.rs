use std::sync::{Arc, Mutex, MutexGuard};

use candle_core::Tensor;

use super::{Cache, NormalCache};

pub type LayerCaches = Vec<Option<(Tensor, Tensor)>>;

#[derive(Debug, Clone)]
pub enum EitherCache {
    Normal(Arc<Mutex<NormalCache>>),
    Full(Cache),
}

impl EitherCache {
    /// Panics otherwise!
    pub fn full(&self) -> &Cache {
        match self {
            Self::Full(full) => full,
            Self::Normal(_) => panic!("Got normal cache, expected full cache."),
        }
    }

    /// Panics otherwise!
    pub fn normal(&self) -> MutexGuard<'_, NormalCache> {
        match self {
            Self::Normal(normal) => normal.lock().unwrap(),
            Self::Full(_) => panic!("Got full cache, expected normal cache."),
        }
    }
}
